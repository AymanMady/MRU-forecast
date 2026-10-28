"""Prediction-interval evaluation (Phase 11).

A point forecast is judged by how close it is.  An interval is judged on two
axes at once, and reporting either alone is misleading:

* **coverage** — does the realised value fall inside it as often as promised?
* **width** — how wide did it have to be to get there?

``[0, +inf)`` has perfect coverage and no value at all.  Every table produced
here therefore reports the pair, and the scoring rules (**Winkler**, **pinball**,
**CRPS**) combine the two into a single number that cannot be gamed by widening.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _clean(*arrays: Any) -> tuple[np.ndarray, ...]:
    """Drop positions where any input is missing.

    Only ``NaN`` counts as missing.  An infinite bound is a *meaningful* value
    here: the conformal quantile is deliberately ``+inf`` when the calibration
    set is too small to certify a level, and such an interval must score as
    "covers everything, infinitely wide" rather than quietly vanish from the
    sample and make an uncertifiable interval look like a missing measurement.
    """
    stacked = [np.asarray(a, dtype="float64").ravel() for a in arrays]
    shapes = {a.shape for a in stacked}
    if len(shapes) > 1:
        raise ValueError(f"shape mismatch across inputs: {shapes}")
    usable = np.logical_and.reduce([~np.isnan(a) for a in stacked])
    return tuple(a[usable] for a in stacked)


# ---------------------------------------------------------------------------
# Coverage & width
# ---------------------------------------------------------------------------
def empirical_coverage(y_true: Any, lower: Any, upper: Any) -> float:
    """Share of outcomes inside the interval, as a fraction.

    coverage = #{ lower_i <= y_i <= upper_i } / n
    """
    true, low, high = _clean(y_true, lower, upper)
    if true.size == 0:
        return float("nan")
    return float(np.mean((true >= low) & (true <= high)))


def coverage_standard_error(coverage: float, n: int) -> float:
    """Binomial standard error of an empirical coverage estimate.

    Coverage measured on 252 dates is an estimate, not a constant: at a nominal
    90% the standard error is ~1.9pp, so 88% and 92% are the same number.  This
    is reported next to every coverage figure so the difference is not
    over-read.
    """
    if n <= 0 or not np.isfinite(coverage):
        return float("nan")
    return float(np.sqrt(max(coverage * (1.0 - coverage), 0.0) / n))


def mean_interval_width(lower: Any, upper: Any) -> float:
    low, high = _clean(lower, upper)
    return float(np.mean(high - low)) if low.size else float("nan")


def median_interval_width(lower: Any, upper: Any) -> float:
    low, high = _clean(lower, upper)
    return float(np.median(high - low)) if low.size else float("nan")


def relative_interval_width(y_true: Any, lower: Any, upper: Any) -> float:
    """Mean width as a percentage of the realised level — comparable across pairs."""
    true, low, high = _clean(y_true, lower, upper)
    usable = np.abs(true) > 1e-12
    if not usable.any():
        return float("nan")
    return float(np.mean((high - low)[usable] / np.abs(true[usable])) * 100.0)


# ---------------------------------------------------------------------------
# Proper scoring rules
# ---------------------------------------------------------------------------
def pinball_loss(y_true: Any, quantile_prediction: Any, tau: float) -> float:
    """Quantile (pinball) loss at level ``tau``. Lower is better.

    L = max( tau * (y - q), (tau - 1) * (y - q) )
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    true, prediction = _clean(y_true, quantile_prediction)
    if true.size == 0:
        return float("nan")
    difference = true - prediction
    return float(np.mean(np.maximum(tau * difference, (tau - 1.0) * difference)))


def interval_pinball_loss(y_true: Any, lower: Any, upper: Any, confidence: float) -> float:
    """Average pinball loss of the two bounds that define the interval."""
    alpha = 1.0 - confidence
    return float(
        np.mean(
            [
                pinball_loss(y_true, lower, alpha / 2.0),
                pinball_loss(y_true, upper, 1.0 - alpha / 2.0),
            ]
        )
    )


def winkler_score(y_true: Any, lower: Any, upper: Any, confidence: float) -> float:
    """Winkler interval score. Lower is better.

        W = (u - l)                     if l <= y <= u
          = (u - l) + 2/alpha * (l - y) if y < l
          = (u - l) + 2/alpha * (y - u) if y > u

    The proper scoring rule for an interval: width is paid for always, a miss is
    paid for in proportion to how far outside it landed.  Widening to buy
    coverage stops helping once the width penalty exceeds the misses avoided,
    which is why this is the headline interval metric here.
    """
    alpha = 1.0 - confidence
    true, low, high = _clean(y_true, lower, upper)
    if true.size == 0:
        return float("nan")
    width = high - low
    penalty = np.where(true < low, 2.0 / alpha * (low - true), 0.0) + np.where(
        true > high, 2.0 / alpha * (true - high), 0.0
    )
    return float(np.mean(width + penalty))


def crps_from_quantiles(y_true: Any, quantile_predictions: dict[float, Any]) -> float:
    """CRPS approximated from a discrete set of predictive quantiles.

        CRPS = 2 * integral_0^1 pinball_tau(y, q_tau) d tau

    Approximated by the trapezoid rule over the available ``tau``.  With the
    three conformal levels this gives six quantiles; the estimate is therefore
    coarse and is labelled as approximate everywhere it is reported.
    """
    taus = sorted(quantile_predictions)
    if len(taus) < 2:
        return float("nan")
    losses = [pinball_loss(y_true, quantile_predictions[tau], tau) for tau in taus]
    return float(2.0 * np.trapezoid(losses, taus) / (taus[-1] - taus[0]))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def evaluate_interval(y_true: Any, lower: Any, upper: Any, confidence: float) -> dict[str, float]:
    """Every interval metric for one nominal level."""
    true, low, high = _clean(y_true, lower, upper)
    coverage = empirical_coverage(true, low, high)
    return {
        "nominal_coverage": float(confidence),
        "empirical_coverage": coverage,
        "coverage_error_pp": (coverage - confidence) * 100.0,
        "coverage_se_pp": coverage_standard_error(coverage, int(true.size)) * 100.0,
        "mean_interval_width": mean_interval_width(low, high),
        "median_interval_width": median_interval_width(low, high),
        "relative_width_pct": relative_interval_width(true, low, high),
        "pinball_loss": interval_pinball_loss(true, low, high, confidence),
        "winkler_score": winkler_score(true, low, high, confidence),
        "n": int(true.size),
    }


def interval_metrics_table(
    frame: pd.DataFrame,
    *,
    group_by: list[str],
    levels: list[int] | None = None,
    actual_column: str = "actual",
) -> pd.DataFrame:
    """Interval metrics per group and per nominal level, from a tidy frame.

    Bound columns are expected as ``lower_80`` / ``upper_80`` etc.
    """
    available = levels or sorted(
        int(c.removeprefix("lower_")) for c in frame.columns if c.startswith("lower_")
    )
    rows = []
    for keys, group in frame.groupby(group_by, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        for level in available:
            low_col, high_col = f"lower_{level}", f"upper_{level}"
            if low_col not in group or high_col not in group:
                continue
            row: dict[str, Any] = dict(zip(group_by, key_tuple, strict=False))
            row.update(
                evaluate_interval(
                    group[actual_column], group[low_col], group[high_col], level / 100.0
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(
    frame: pd.DataFrame, *, group_by: list[str] | None = None, actual_column: str = "actual"
) -> pd.DataFrame:
    """Nominal vs empirical coverage, ready for :func:`viz.calibration_chart`."""
    groups = group_by or ["model"]
    table = interval_metrics_table(frame, group_by=groups, actual_column=actual_column)
    columns = [
        *groups,
        "nominal_coverage",
        "empirical_coverage",
        "coverage_error_pp",
        "coverage_se_pp",
        "mean_interval_width",
        "n",
    ]
    return table[columns].sort_values([*groups, "nominal_coverage"]).reset_index(drop=True)


def crps_table(
    frame: pd.DataFrame, *, group_by: list[str], actual_column: str = "actual"
) -> pd.DataFrame:
    """Approximate CRPS per group, built from the calibrated interval bounds."""
    levels = sorted(int(c.removeprefix("lower_")) for c in frame.columns if c.startswith("lower_"))
    rows = []
    for keys, group in frame.groupby(group_by, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        quantiles: dict[float, Any] = {}
        for level in levels:
            alpha = 1.0 - level / 100.0
            quantiles[alpha / 2.0] = group[f"lower_{level}"]
            quantiles[1.0 - alpha / 2.0] = group[f"upper_{level}"]
        if "prediction" in group:
            quantiles[0.5] = group["prediction"]
        row: dict[str, Any] = dict(zip(group_by, key_tuple, strict=False))
        row["crps_approx"] = crps_from_quantiles(group[actual_column], quantiles)
        row["n"] = int(group[actual_column].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Market-condition analysis (Phase 16 of the brief)
# ---------------------------------------------------------------------------
def assign_volatility_regime(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    origin_column: str = "origin",
    target_column: str = "target",
) -> pd.DataFrame:
    """Tag each prediction with the volatility regime in force at its origin.

    The regime is computed from **trailing** volatility and the cut-points come
    from the whole sample, so a date's regime is knowable on that date.
    """
    from mru_forecast.evaluation.eda import volatility_regimes

    settings = config["evaluation"]["volatility_regimes"]
    out = predictions.copy()
    out["regime"] = pd.NA
    for target in out[target_column].unique():
        if target not in frame.columns:
            continue
        regimes = volatility_regimes(
            frame[target],
            int(settings["window"]),
            tuple(settings["quantiles"]),
            tuple(settings["labels"]),
        )
        mask = out[target_column] == target
        out.loc[mask, "regime"] = out.loc[mask, origin_column].map(regimes).to_numpy()
    return out


def regime_performance(
    predictions: pd.DataFrame,
    *,
    group_by: list[str] | None = None,
    level: int = 90,
    actual_column: str = "actual",
) -> pd.DataFrame:
    """Point *and* interval performance per volatility regime.

    Answers the question the brief asks directly: does the model stay reliable
    when the market gets turbulent?  A well-calibrated adaptive interval keeps
    coverage roughly flat across regimes and pays for it in width; a
    non-adaptive one keeps width flat and loses coverage in the high regime.
    """
    from mru_forecast.evaluation.metrics import mae, rmse

    groups = [*(group_by or ["model"]), "regime"]
    low_col, high_col = f"lower_{level}", f"upper_{level}"
    rows = []
    for keys, group in predictions.dropna(subset=["regime"]).groupby(groups, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(groups, key_tuple, strict=False))
        row["n"] = int(group[actual_column].notna().sum())
        row["mae"] = mae(group[actual_column], group["prediction"])
        row["rmse"] = rmse(group[actual_column], group["prediction"])
        if low_col in group and high_col in group:
            row["coverage"] = empirical_coverage(
                group[actual_column], group[low_col], group[high_col]
            )
            row["mean_interval_width"] = mean_interval_width(group[low_col], group[high_col])
            row["winkler_score"] = winkler_score(
                group[actual_column], group[low_col], group[high_col], level / 100.0
            )
        rows.append(row)
    order = {"low": 0, "normal": 1, "high": 2}
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            [*(group_by or ["model"]), "regime"],
            key=lambda s: s.map(order) if s.name == "regime" else s,
        ).reset_index(drop=True)
    return table


def coverage_over_time(
    predictions: pd.DataFrame,
    *,
    level: int = 90,
    window: int = 60,
    actual_column: str = "actual",
    origin_column: str = "origin",
) -> pd.DataFrame:
    """Rolling empirical coverage — shows drift that a single average hides."""
    low_col, high_col = f"lower_{level}", f"upper_{level}"
    ordered = predictions.sort_values(origin_column)
    covered = (
        (ordered[actual_column] >= ordered[low_col]) & (ordered[actual_column] <= ordered[high_col])
    ).astype("float64")
    return pd.DataFrame(
        {
            origin_column: ordered[origin_column].to_numpy(),
            "rolling_coverage": covered.rolling(window, min_periods=window // 2).mean().to_numpy(),
            "nominal": level / 100.0,
        }
    )
