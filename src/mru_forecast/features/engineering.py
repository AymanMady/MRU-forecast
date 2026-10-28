"""Feature engineering (Phase 5).

The single rule this module exists to guarantee:

    every feature on row ``t`` is computable from information available at
    close of business on day ``t``, and nothing later.

Concretely: rolling windows are trailing (never ``center=True``), auxiliary
series are lagged by at least one day, and the target is built by shifting the
**label** backwards (``y_t = f(value_{t+h})``) rather than by shifting features
forwards.  ``tests/test_features.py`` re-derives each feature from a truncated
history and asserts the value is unchanged.

Target transform
----------------
The label is the *h-step log return*, not the level::

    y_t = log(value_{t+h}) - log(value_t)

Gradient-boosted trees predict a constant within each leaf, so they cannot
produce a value outside the range they were trained on.  A rate that has drifted
from 35.75 to 40.03 over the sample would therefore be structurally capped at
40.03 by a level model.  Predicting the return and anchoring it to the last
observed level removes that ceiling:

    prediction_{t+h} = value_t * exp(y_hat_t)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Calendar features that encode absolute time. A tree can split on them but the
# split cannot generalise past the training period, so they stay out of the
# model matrix by default.
NON_EXTRAPOLABLE = {"year"}


@dataclass
class SupervisedDataset:
    """A leakage-checked (X, y) pair plus what is needed to invert the target."""

    X: pd.DataFrame
    y: pd.Series
    anchor: pd.Series  # value at t, the base of the log return
    target_dates: pd.DatetimeIndex  # the date each label refers to (t + h)
    target: str
    horizon: int
    transform: str
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(self.X.columns)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def index(self) -> pd.DatetimeIndex:
        """Origin dates: the last day whose information the features may use."""
        return pd.DatetimeIndex(self.X.index)

    def slice_by_position(self, start: int, stop: int) -> SupervisedDataset:
        return SupervisedDataset(
            X=self.X.iloc[start:stop],
            y=self.y.iloc[start:stop],
            anchor=self.anchor.iloc[start:stop],
            target_dates=self.target_dates[start:stop],
            target=self.target,
            horizon=self.horizon,
            transform=self.transform,
            feature_names=self.feature_names,
        )


# ---------------------------------------------------------------------------
# Individual feature families
# ---------------------------------------------------------------------------
def add_calendar_features(index: pd.DatetimeIndex, parts: list[str] | None = None) -> pd.DataFrame:
    """Calendar position. Cyclical parts are encoded as sin/cos pairs.

    December and January are one month apart, but 12 and 1 are eleven integers
    apart; the sin/cos encoding is what makes that wrap-around visible to the
    model.
    """
    parts = parts or ["day_of_week", "day_of_month", "week", "month", "quarter", "year"]
    frame = pd.DataFrame(index=index)
    if "day_of_week" in parts:
        frame["day_of_week"] = index.dayofweek
        frame["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 5)
        frame["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 5)
    if "day_of_month" in parts:
        frame["day_of_month"] = index.day
    if "week" in parts:
        frame["week"] = index.isocalendar().week.astype("int32").to_numpy()
    if "month" in parts:
        frame["month"] = index.month
        frame["month_sin"] = np.sin(2 * np.pi * index.month / 12)
        frame["month_cos"] = np.cos(2 * np.pi * index.month / 12)
    if "quarter" in parts:
        frame["quarter"] = index.quarter
    if "year" in parts:
        frame["year"] = index.year
    # Month-end and quarter-end often carry settlement flows.
    frame["is_month_end"] = index.is_month_end.astype("int8")
    frame["is_quarter_end"] = index.is_quarter_end.astype("int8")
    return frame


def add_lag_features(series: pd.Series, lags: list[int], prefix: str) -> pd.DataFrame:
    """``lag_k`` = the value k business days ago. All lags must be >= 1."""
    if any(lag < 1 for lag in lags):
        raise ValueError(f"lags must be >= 1 (a lag of 0 is the value itself): {lags}")
    return pd.DataFrame({f"{prefix}_lag_{lag}": series.shift(lag) for lag in lags})


def add_rolling_features(
    series: pd.Series, means: list[int], stds: list[int], prefix: str
) -> pd.DataFrame:
    """Trailing rolling statistics, inclusive of ``t``.

    Windows are never centred: a centred window at ``t`` reads values after
    ``t``, which is exactly the leak this project is built to avoid.
    """
    frame = pd.DataFrame(index=series.index)
    for window in means:
        rolling = series.rolling(window, min_periods=max(window // 2, 2))
        frame[f"{prefix}_roll_mean_{window}"] = rolling.mean()
        # Distance from the local mean, scaled — the shape that survives a
        # change in the level of the series.
        frame[f"{prefix}_dist_mean_{window}"] = series / rolling.mean() - 1.0
    for window in stds:
        frame[f"{prefix}_roll_std_{window}"] = series.rolling(
            window, min_periods=max(window // 2, 2)
        ).std()
    for window in sorted(set(means) | set(stds)):
        rolling = series.rolling(window, min_periods=max(window // 2, 2))
        frame[f"{prefix}_roll_min_{window}"] = rolling.min()
        frame[f"{prefix}_roll_max_{window}"] = rolling.max()
    return frame


def add_return_features(series: pd.Series, prefix: str, lags: list[int]) -> pd.DataFrame:
    """Past returns over several look-backs, plus realised volatility."""
    log_value = np.log(series)
    frame = pd.DataFrame(index=series.index)
    frame[f"{prefix}_log_return_1"] = log_value.diff()
    frame[f"{prefix}_pct_change_1"] = series.pct_change()
    for lag in lags:
        if lag <= 1:
            continue
        frame[f"{prefix}_log_return_{lag}"] = log_value.diff(lag)
    daily = log_value.diff()
    for window in (7, 21):
        frame[f"{prefix}_realised_vol_{window}"] = daily.rolling(
            window, min_periods=max(window // 2, 2)
        ).std()
    # Days since the rate last moved: on a managed currency this is the single
    # most informative state variable, and it is unavailable to a plain lag.
    unchanged = (series.diff().abs() < 1e-9).astype("int64")
    frame[f"{prefix}_days_since_move"] = unchanged.groupby((unchanged == 0).cumsum()).cumsum()
    return frame


def add_momentum_features(series: pd.Series, windows: list[int], prefix: str) -> pd.DataFrame:
    """Trend strength over each window, normalised by the level."""
    frame = pd.DataFrame(index=series.index)
    for window in windows:
        frame[f"{prefix}_momentum_{window}"] = series / series.shift(window) - 1.0
        rolling = series.rolling(window, min_periods=max(window // 2, 2))
        span = rolling.max() - rolling.min()
        # Position inside the recent range, 0 at the low and 1 at the high.
        frame[f"{prefix}_range_position_{window}"] = np.where(
            span > 0, (series - rolling.min()) / span.replace(0, np.nan), 0.5
        )
    return frame


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_features(
    frame: pd.DataFrame,
    target: str,
    config: dict[str, Any],
    *,
    include_non_extrapolable: bool = False,
) -> pd.DataFrame:
    """Full feature matrix for ``target``, aligned to ``frame.index``."""
    settings = config["features"]
    if target not in frame.columns:
        raise KeyError(f"target {target!r} not in frame columns {list(frame.columns)}")

    series = frame[target]
    blocks = [
        add_calendar_features(pd.DatetimeIndex(frame.index), settings.get("calendar")),
        add_lag_features(series, settings["lags"], target),
        add_rolling_features(series, settings["rolling_means"], settings["rolling_stds"], target),
        add_return_features(series, target, settings["lags"]),
        add_momentum_features(series, settings["momentum_windows"], target),
    ]

    # Auxiliary series: lagged by at least one day.  Even a same-day quote from
    # another market is only safe if it closes before the target's cut-off, and
    # that cannot be verified from the data, so it is never assumed.
    auxiliary_lags = settings.get("auxiliary_lags", [1])
    if min(auxiliary_lags, default=1) < 1:
        raise ValueError("auxiliary_lags must all be >= 1")
    for name in settings.get("include_auxiliary", []):
        if name not in frame.columns or name == target:
            continue
        aux = frame[name]
        blocks.append(add_lag_features(aux, auxiliary_lags, name))
        aux_returns = np.log(aux).diff()
        blocks.append(
            pd.DataFrame(
                {f"{name}_log_return_lag_{lag}": aux_returns.shift(lag) for lag in auxiliary_lags}
            )
        )

    features = pd.concat(blocks, axis=1)
    features = features.loc[:, ~features.columns.duplicated()]

    if not include_non_extrapolable:
        features = features.drop(columns=list(NON_EXTRAPOLABLE & set(features.columns)))

    features = features.replace([np.inf, -np.inf], np.nan)
    features.index.name = frame.index.name or "date"
    return features


def make_supervised(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    config: dict[str, Any],
    *,
    transform: str | None = None,
    dropna: bool = True,
) -> SupervisedDataset:
    """Build ``(X, y)`` for a single ``(target, horizon)`` pair.

    Row ``t`` pairs features known at ``t`` with the outcome at ``t + horizon``
    business days.  The last ``horizon`` rows therefore have no label and are
    dropped from training — but they are exactly the rows used to forecast the
    future, which :func:`latest_feature_row` returns.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    use_log = (
        config["models"]["xgboost"].get("log_target", True)
        if transform is None
        else transform == "log_return"
    )
    features = build_features(frame, target, config)
    series = frame[target]
    future = series.shift(-horizon)

    if use_log:
        label = np.log(future) - np.log(series)
        transform_name = "log_return"
    else:
        label = future
        transform_name = "level"

    # The calendar date each label refers to: `horizon` rows further down the
    # modelling calendar, not `horizon` calendar days later.
    target_dates = pd.DatetimeIndex(series.index.to_series().shift(-horizon).to_numpy())

    dataset = pd.concat([features, label.rename("__y__"), series.rename("__anchor__")], axis=1)
    dataset["__target_date__"] = target_dates

    if dropna:
        dataset = dataset.dropna(subset=["__y__", "__anchor__"])
        # Rows whose features are entirely missing carry no information; rows
        # with a few gaps are kept because XGBoost handles NaN natively.
        dataset = dataset.dropna(subset=list(features.columns), how="all")

    return SupervisedDataset(
        X=dataset[features.columns],
        y=dataset["__y__"],
        anchor=dataset["__anchor__"],
        target_dates=pd.DatetimeIndex(dataset["__target_date__"]),
        target=target,
        horizon=horizon,
        transform=transform_name,
    )


def latest_feature_row(
    frame: pd.DataFrame, target: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    """Features for the most recent date, for forecasting beyond the data.

    Returns ``(X_last, anchor_value, anchor_date)``.
    """
    features = build_features(frame, target, config)
    anchor_date = pd.Timestamp(frame.index[-1])
    return features.iloc[[-1]], float(frame[target].iloc[-1]), anchor_date


def inverse_transform(
    prediction: np.ndarray | pd.Series, anchor: np.ndarray | pd.Series, transform: str
) -> np.ndarray:
    """Map a model output back to a rate level."""
    values = np.asarray(prediction, dtype="float64")
    if transform == "level":
        return values
    if transform == "log_return":
        return np.asarray(anchor, dtype="float64") * np.exp(values)
    raise ValueError(f"unknown transform: {transform!r}")


def forward_transform(
    level: np.ndarray | pd.Series, anchor: np.ndarray | pd.Series, transform: str
) -> np.ndarray:
    """Inverse of :func:`inverse_transform`: a level back into model space."""
    values = np.asarray(level, dtype="float64")
    if transform == "level":
        return values
    if transform == "log_return":
        return np.log(values) - np.log(np.asarray(anchor, dtype="float64"))
    raise ValueError(f"unknown transform: {transform!r}")


def future_business_days(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """The next ``horizon`` business days strictly after ``last_date``."""
    return pd.date_range(last_date + pd.tseries.offsets.BDay(1), periods=horizon, freq="B")
