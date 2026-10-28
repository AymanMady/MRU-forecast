"""Conformal prediction intervals (Phase 10).

Split (inductive) conformal prediction turns any point forecaster into an
interval forecaster with a finite-sample coverage guarantee, without assuming
anything about the error distribution.

    Training set  -> fit the base model
    Calibration set -> non-conformity scores s_i = |y_i - yhat_i|
                    -> conformal quantile q = the ceil((n+1)(1-alpha))-th
                       smallest score
    New point     -> interval [yhat - q, yhat + q]

The ``(n+1)`` in the rank is not a detail.  With ``n`` calibration points the
plain empirical quantile under-covers slightly; the corrected rank is what makes
the guarantee

    P(y in interval) >= 1 - alpha

hold exactly in finite samples, under exchangeability.

Exchangeability, honestly
-------------------------
Time series are **not** exchangeable: volatility clusters and regimes shift, so
the guarantee above is not strictly earned here.  Two things are done about it,
and the residual gap is measured rather than assumed away:

* ``normalize_by_volatility`` divides each score by a local volatility estimate,
  so the interval widens in turbulent periods instead of being an average of
  calm and stormy ones.  This is the locally-adaptive (normalised) variant.
* :class:`AdaptiveConformalPredictor` implements Adaptive Conformal Inference
  (Gibbs & Candes, 2021), which updates the working miscoverage rate online from
  realised hits and misses, and provably tracks the target coverage over time
  even under distribution shift.

Empirical coverage is then reported for every level in Phase 11, which is the
only thing that actually demonstrates the intervals are usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd

from mru_forecast.models.base import Forecaster, NotFittedError
from mru_forecast.utils.logging import get_logger

logger = get_logger(__name__)

# A scale of exactly zero would make a normalised score infinite; a managed rate
# sits perfectly still for weeks, so this case is routine rather than exotic.
MIN_SCALE = 1e-6


def level_tag(confidence: float) -> str:
    """The suffix used for a confidence level in every column name and API field.

    ``0.90 -> "90"``, giving ``lower_90`` / ``upper_90``.  One definition, so a
    column written by the conformal predictor and a column read by the metrics
    can never drift apart.  The explicit ``int`` matters: ``confidence`` often
    arrives as a ``numpy.float64`` from a DataFrame, and ``round`` on one of
    those returns a float, not an int.
    """
    return str(int(round(float(confidence) * 100)))  # noqa: RUF046 - numpy floats


def conformal_quantile(scores: np.ndarray, confidence: float) -> float:
    """The split-conformal quantile of ``scores`` at ``confidence``.

    Uses the finite-sample rank ``ceil((n + 1) * confidence)``.  When that rank
    exceeds ``n`` the calibration set is too small to certify the level and the
    quantile is ``+inf`` — reported as an infinite interval rather than silently
    returning the maximum score, which would claim a coverage that has not been
    established.
    """
    clean = np.asarray(scores, dtype="float64")
    clean = clean[np.isfinite(clean)]
    n = clean.size
    if n == 0:
        return float("nan")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rank = int(np.ceil((n + 1) * confidence))
    if rank > n:
        logger.warning(
            "calibration set of %d is too small to certify %.0f%% coverage (needs at least %d)",
            n,
            confidence * 100,
            int(np.ceil(confidence / (1 - confidence))),
        )
        return float("inf")
    return float(np.sort(clean)[rank - 1])


def volatility_scale(
    frame: pd.DataFrame,
    target: str,
    origins: pd.DatetimeIndex,
    *,
    window: int = 21,
    horizon: int = 1,
) -> np.ndarray:
    """A per-origin scale for normalised conformal scores.

    Trailing realised volatility of log-returns, converted to level units and
    scaled by ``sqrt(horizon)`` (the random-walk scaling of an h-step error).
    Trailing, so the scale at origin ``o`` is knowable at ``o``.
    """
    series = frame[target].astype("float64")
    log_returns = np.log(series).diff()
    sigma = log_returns.rolling(window, min_periods=max(window // 2, 2)).std()
    scale = (series * sigma * np.sqrt(horizon)).reindex(origins)
    # A floor keeps the scale usable through the long flat stretches; the median
    # is a sensible, data-driven floor rather than an arbitrary constant.
    floor = float(np.nanmedian(scale.to_numpy())) * 0.1
    floor = max(floor, MIN_SCALE)
    return np.maximum(scale.fillna(floor).to_numpy(dtype="float64"), floor)


@dataclass
class CalibrationSummary:
    """What calibration produced — logged and surfaced in the dashboard."""

    n_scores: int
    method: str
    symmetric: bool
    quantiles: dict[float, float] = field(default_factory=dict)
    lower_quantiles: dict[float, float] = field(default_factory=dict)
    upper_quantiles: dict[float, float] = field(default_factory=dict)
    calibration_start: str | None = None
    calibration_end: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_scores": self.n_scores,
            "method": self.method,
            "symmetric": self.symmetric,
            "quantiles": {str(k): v for k, v in self.quantiles.items()},
            "lower_quantiles": {str(k): v for k, v in self.lower_quantiles.items()},
            "upper_quantiles": {str(k): v for k, v in self.upper_quantiles.items()},
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
        }


class ConformalPredictor:
    """Wrap any :class:`~mru_forecast.models.base.Forecaster` in prediction intervals.

    Parameters
    ----------
    base_model
        The point forecaster.  It is fitted on the training block only.
    confidence_levels
        Nominal coverages to calibrate, e.g. ``[0.80, 0.90, 0.95]``.
    normalize_by_volatility
        Divide scores by a local volatility estimate, giving intervals that
        breathe with the market instead of one width for all conditions.
    symmetric
        ``True`` uses ``|residual|`` and produces a symmetric interval.
        ``False`` calibrates the two tails separately, which matters when the
        errors are skewed — a managed currency that is devalued in steps but
        revalued gradually is exactly that case.
    """

    def __init__(
        self,
        base_model: Forecaster,
        *,
        confidence_levels: list[float] | tuple[float, ...] = (0.80, 0.90, 0.95),
        normalize_by_volatility: bool = True,
        volatility_window: int = 21,
        symmetric: bool = True,
        min_calibration_size: int = 50,
    ) -> None:
        self.base_model = base_model
        self.confidence_levels = sorted(float(c) for c in confidence_levels)
        self.normalize_by_volatility = bool(normalize_by_volatility)
        self.volatility_window = int(volatility_window)
        self.symmetric = bool(symmetric)
        self.min_calibration_size = int(min_calibration_size)

        self.quantiles: dict[float, float] = {}
        self.lower_quantiles: dict[float, float] = {}
        self.upper_quantiles: dict[float, float] = {}
        self.scores_: np.ndarray | None = None
        self.summary: CalibrationSummary | None = None
        self._calibrated = False

    # -- properties --------------------------------------------------------
    @property
    def name(self) -> str:
        return f"{self.base_model.name}_conformal"

    @property
    def target(self) -> str | None:
        return self.base_model.target

    @property
    def horizon(self) -> int | None:
        return self.base_model.horizon

    @property
    def method(self) -> str:
        return "normalized" if self.normalize_by_volatility else "split"

    # -- step 1: fit -------------------------------------------------------
    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        horizon: int,
        config: dict[str, Any],
        *,
        train_end: pd.Timestamp | None = None,
    ) -> Self:
        """Train the base model on the training block. No calibration yet."""
        self.base_model.fit(frame, target, horizon, config, train_end=train_end)
        self._calibrated = False
        return self

    # -- step 2: calibrate -------------------------------------------------
    def calibrate(
        self,
        frame: pd.DataFrame,
        calibration_origins: pd.DatetimeIndex,
        *,
        actuals: pd.Series | None = None,
    ) -> Self:
        """Compute non-conformity scores and the conformal quantiles.

        ``calibration_origins`` must be disjoint from, and strictly after, the
        data the base model was trained on — the guarantee comes from the model
        never having seen these outcomes.
        """
        origins = pd.DatetimeIndex(calibration_origins)
        if self.base_model.train_end is not None:
            overlapping = origins[origins <= self.base_model.train_end]
            if len(overlapping):
                raise ValueError(
                    f"{len(overlapping)} calibration origin(s) fall inside the training "
                    f"window (train_end={self.base_model.train_end.date()}); split "
                    "conformal requires a disjoint calibration set"
                )

        predictions = self.base_model.predict_origins(frame, origins)
        observed = actuals if actuals is not None else self.base_model.actuals(frame, origins)
        residuals = (observed - predictions).astype("float64")

        scale = (
            volatility_scale(
                frame,
                self.target,
                origins,
                window=self.volatility_window,
                horizon=int(self.horizon),
            )
            if self.normalize_by_volatility
            else np.ones(len(origins))
        )

        usable = np.isfinite(residuals.to_numpy()) & np.isfinite(scale)
        signed = residuals.to_numpy()[usable] / scale[usable]
        if signed.size < self.min_calibration_size:
            raise ValueError(
                f"{signed.size} usable calibration scores, need at least "
                f"{self.min_calibration_size}"
            )

        self.scores_ = np.abs(signed)
        self.quantiles, self.lower_quantiles, self.upper_quantiles = {}, {}, {}
        for level in self.confidence_levels:
            self.quantiles[level] = conformal_quantile(self.scores_, level)
            if not self.symmetric:
                # Two one-sided quantiles at level (1 + c)/2 each, so the pair
                # together covers `c`.
                tail = (1.0 + level) / 2.0
                self.upper_quantiles[level] = conformal_quantile(signed, tail)
                self.lower_quantiles[level] = conformal_quantile(-signed, tail)

        used = origins[usable]
        self.summary = CalibrationSummary(
            n_scores=int(signed.size),
            method=self.method,
            symmetric=self.symmetric,
            quantiles=dict(self.quantiles),
            lower_quantiles=dict(self.lower_quantiles),
            upper_quantiles=dict(self.upper_quantiles),
            calibration_start=str(used.min().date()) if len(used) else None,
            calibration_end=str(used.max().date()) if len(used) else None,
        )
        self._calibrated = True
        logger.debug(
            "calibrated %s on %d scores; q(0.90)=%.5f",
            self.name,
            signed.size,
            self.quantiles.get(0.90, float("nan")),
        )
        return self

    # -- step 3: predict ---------------------------------------------------
    def predict(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.Series:
        """Point forecast — unchanged by conformalisation."""
        return self.base_model.predict_origins(frame, origins)

    def predict_interval(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        *,
        confidence: float | list[float] | None = None,
    ) -> pd.DataFrame:
        """Point forecast plus ``lower_<level>`` / ``upper_<level>`` columns."""
        if not self._calibrated:
            raise NotFittedError(f"{self.name}: call calibrate() before predict_interval()")

        origin_index = pd.DatetimeIndex(origins)
        predictions = self.predict(frame, origin_index)
        levels = self._resolve_levels(confidence)

        scale = (
            volatility_scale(
                frame,
                self.target,
                origin_index,
                window=self.volatility_window,
                horizon=int(self.horizon),
            )
            if self.normalize_by_volatility
            else np.ones(len(origin_index))
        )

        out = pd.DataFrame({"prediction": predictions.to_numpy()}, index=origin_index)
        out["scale"] = scale
        for level in levels:
            if self.symmetric:
                margin = self.quantiles[level] * scale
                lower, upper = predictions.to_numpy() - margin, predictions.to_numpy() + margin
            else:
                lower = predictions.to_numpy() - self.lower_quantiles[level] * scale
                upper = predictions.to_numpy() + self.upper_quantiles[level] * scale
            tag = level_tag(level)
            # An exchange rate cannot be negative; clipping at zero never
            # narrows a bound that mattered and keeps the output coherent.
            out[f"lower_{tag}"] = np.maximum(lower, 0.0)
            out[f"upper_{tag}"] = upper
        return out

    def forecast_future(
        self, frame: pd.DataFrame, *, confidence: float | list[float] | None = None
    ) -> pd.DataFrame:
        """Interval forecast for ``horizon`` business days past the data."""
        last = pd.Timestamp(frame.index[-1])
        out = self.predict_interval(frame, pd.DatetimeIndex([last]), confidence=confidence)
        future_date = pd.date_range(
            last + pd.tseries.offsets.BDay(1), periods=int(self.horizon), freq="B"
        )[-1]
        out.index = pd.DatetimeIndex([future_date], name="date")
        out.insert(0, "origin", last)
        out.insert(1, "horizon", int(self.horizon))
        return out

    # -- internals ---------------------------------------------------------
    def _resolve_levels(self, confidence: float | list[float] | None) -> list[float]:
        if confidence is None:
            return list(self.confidence_levels)
        requested = [confidence] if isinstance(confidence, (int, float)) else list(confidence)
        unknown = [c for c in requested if float(c) not in self.quantiles]
        if unknown:
            raise KeyError(f"not calibrated for {unknown}; available: {sorted(self.quantiles)}")
        return [float(c) for c in requested]

    def __repr__(self) -> str:
        state = f"calibrated on {self.summary.n_scores}" if self._calibrated else "uncalibrated"
        return f"ConformalPredictor({self.base_model.name}, {self.method}, {state})"


class AdaptiveConformalPredictor(ConformalPredictor):
    """Adaptive Conformal Inference (Gibbs & Candes, 2021).

    After each observation the working miscoverage ``alpha_t`` is nudged by the
    realised hit or miss::

        alpha_{t+1} = alpha_t + gamma * (alpha_target - 1{y_t not in interval_t})

    A run of misses widens the interval; a long run of easy hits narrows it.
    Split conformal's guarantee is marginal and static; ACI instead guarantees
    that the *long-run* empirical miscoverage converges to the target, which is
    the property that survives the volatility regimes this series actually goes
    through.

    ``update`` must be called with realised outcomes as they arrive; the state
    is per confidence level.
    """

    def __init__(self, *args: Any, gamma: float = 0.01, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)
        self.alpha_t: dict[float, float] = {}
        self.history: list[dict[str, Any]] = []

    def calibrate(self, *args: Any, **kwargs: Any) -> Self:
        super().calibrate(*args, **kwargs)
        self.alpha_t = {level: 1.0 - level for level in self.confidence_levels}
        self.history.clear()
        return self

    def _effective_quantile(self, level: float) -> float:
        """Quantile at the current working alpha, clipped to what the scores support."""
        if self.scores_ is None:
            raise NotFittedError("calibrate() first")
        working = float(np.clip(1.0 - self.alpha_t.get(level, 1.0 - level), 0.01, 0.999))
        return conformal_quantile(self.scores_, working)

    def predict_interval(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        *,
        confidence: float | list[float] | None = None,
    ) -> pd.DataFrame:
        if not self._calibrated:
            raise NotFittedError(f"{self.name}: call calibrate() before predict_interval()")
        origin_index = pd.DatetimeIndex(origins)
        predictions = self.predict(frame, origin_index).to_numpy()
        levels = self._resolve_levels(confidence)
        scale = (
            volatility_scale(
                frame,
                self.target,
                origin_index,
                window=self.volatility_window,
                horizon=int(self.horizon),
            )
            if self.normalize_by_volatility
            else np.ones(len(origin_index))
        )
        out = pd.DataFrame({"prediction": predictions}, index=origin_index)
        out["scale"] = scale
        for level in levels:
            margin = self._effective_quantile(level) * scale
            tag = level_tag(level)
            out[f"lower_{tag}"] = np.maximum(predictions - margin, 0.0)
            out[f"upper_{tag}"] = predictions + margin
            out[f"alpha_{tag}"] = self.alpha_t.get(level, 1.0 - level)
        return out

    def update(self, actual: float, interval: dict[float, tuple[float, float]]) -> None:
        """Feed one realised outcome back into the working miscoverage rates."""
        for level, (lower, upper) in interval.items():
            covered = bool(lower <= actual <= upper)
            target_alpha = 1.0 - level
            self.alpha_t[level] = float(
                np.clip(
                    self.alpha_t.get(level, target_alpha)
                    + self.gamma * (target_alpha - (0.0 if covered else 1.0)),
                    1e-3,
                    0.5,
                )
            )
            self.history.append(
                {"level": level, "covered": covered, "alpha_t": self.alpha_t[level]}
            )

    def run_online(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        *,
        actuals: pd.Series | None = None,
        confidence: float | list[float] | None = None,
    ) -> pd.DataFrame:
        """Walk the origins one at a time, updating alpha after each outcome.

        This is what ACI actually is: a sequential procedure.  Calling
        ``predict_interval`` over a whole block without updating would report
        the *initial* widths for every date and understate the method.
        """
        origin_index = pd.DatetimeIndex(origins)
        observed = actuals if actuals is not None else self.base_model.actuals(frame, origin_index)
        levels = self._resolve_levels(confidence)
        rows = []
        for origin in origin_index:
            row = self.predict_interval(frame, pd.DatetimeIndex([origin]), confidence=levels)
            rows.append(row)
            outcome = float(observed.loc[origin]) if origin in observed.index else np.nan
            if np.isfinite(outcome):
                self.update(
                    outcome,
                    {
                        level: (
                            float(row[f"lower_{level_tag(level)}"].iloc[0]),
                            float(row[f"upper_{level_tag(level)}"].iloc[0]),
                        )
                        for level in levels
                    },
                )
        return pd.concat(rows)


def build_conformal(base_model: Forecaster, config: dict[str, Any], *, adaptive: bool = False):
    """Construct a conformal wrapper from the ``conformal`` config block."""
    settings = config["conformal"]
    cls = AdaptiveConformalPredictor if adaptive else ConformalPredictor
    return cls(
        base_model,
        confidence_levels=settings["confidence_levels"],
        normalize_by_volatility=settings.get("normalize_by_volatility", True),
        volatility_window=int(settings.get("volatility_window", 21)),
        symmetric=settings.get("method", "split") != "asymmetric",
        min_calibration_size=int(settings.get("min_calibration_size", 50)),
    )
