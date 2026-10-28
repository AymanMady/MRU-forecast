"""Point-forecast metrics (Phase 11).

Every function takes ``(y_true, y_pred)`` as array-likes, drops pairs where
either side is missing, and returns a float.  Missing values are dropped rather
than imputed: a metric computed over an invented value is not a measurement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _align(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype="float64").ravel()
    pred = np.asarray(y_pred, dtype="float64").ravel()
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    usable = np.isfinite(true) & np.isfinite(pred)
    return true[usable], pred[usable]


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error, in the units of the rate (MRU)."""
    true, pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(true - pred))) if true.size else float("nan")


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error — penalises large misses more than :func:`mae`."""
    true, pred = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2))) if true.size else float("nan")


def mape(y_true: Any, y_pred: Any) -> float:
    """Mean absolute percentage error (%).

    Safe here because an exchange rate is bounded away from zero; it would not
    be for a series that can cross zero.
    """
    true, pred = _align(y_true, y_pred)
    nonzero = np.abs(true) > 1e-12
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100.0)


def smape(y_true: Any, y_pred: Any) -> float:
    """Symmetric MAPE (%), bounded at 200 and undamaged by small denominators."""
    true, pred = _align(y_true, y_pred)
    denominator = (np.abs(true) + np.abs(pred)) / 2.0
    usable = denominator > 1e-12
    if not usable.any():
        return float("nan")
    return float(np.mean(np.abs(true[usable] - pred[usable]) / denominator[usable]) * 100.0)


def bias(y_true: Any, y_pred: Any) -> float:
    """Mean signed error. Positive means the forecast runs high."""
    true, pred = _align(y_true, y_pred)
    return float(np.mean(pred - true)) if true.size else float("nan")


def directional_accuracy(y_true: Any, y_pred: Any, anchor: Any) -> float:
    """Share of forecasts that get the direction of the move right (%).

    Needs the value at the origin: "up or down" is only defined relative to
    where the rate started.

    Two exclusions, both of which change the number a lot on this data:

    * days where the rate did not actually move (~30% of business days on a
      managed currency) — scoring them would hand out free correct answers;
    * forecasts that predict **no** move at all.  The naive forecast predicts
      exactly the current value by construction, so it never expresses a
      direction; the honest score for it is ``nan``, not 0%.
    """
    true = np.asarray(y_true, dtype="float64").ravel()
    pred = np.asarray(y_pred, dtype="float64").ravel()
    base = np.asarray(anchor, dtype="float64").ravel()
    usable = np.isfinite(true) & np.isfinite(pred) & np.isfinite(base)
    actual_move = true[usable] - base[usable]
    predicted_move = pred[usable] - base[usable]
    scorable = (np.abs(actual_move) > 1e-9) & (np.abs(predicted_move) > 1e-9)
    if not scorable.any():
        return float("nan")
    return float(
        np.mean(np.sign(actual_move[scorable]) == np.sign(predicted_move[scorable])) * 100.0
    )


def theil_u2(y_true: Any, y_pred: Any, y_naive: Any) -> float:
    """RMSE of the model divided by the RMSE of the naive forecast.

    Below 1 means the model beats a random walk; at or above 1 it does not.
    This is the number that decides whether the modelling effort was worth
    anything, so it is reported alongside the raw errors everywhere.
    """
    model_rmse = rmse(y_true, y_pred)
    naive_rmse = rmse(y_true, y_naive)
    if not np.isfinite(naive_rmse) or naive_rmse <= 0:
        return float("nan")
    return float(model_rmse / naive_rmse)


def skill_score(y_true: Any, y_pred: Any, y_reference: Any, metric=mae) -> float:
    """Relative improvement over a reference forecast (%). Positive is better."""
    model = metric(y_true, y_pred)
    reference = metric(y_true, y_reference)
    if not np.isfinite(reference) or reference <= 0:
        return float("nan")
    return float((1.0 - model / reference) * 100.0)


POINT_METRICS = {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "bias": bias}


def evaluate_point_forecast(
    y_true: Any,
    y_pred: Any,
    *,
    anchor: Any | None = None,
    y_naive: Any | None = None,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """All point metrics at once, plus skill against the naive forecast."""
    names = metrics or list(POINT_METRICS)
    result = {name: POINT_METRICS[name](y_true, y_pred) for name in names if name in POINT_METRICS}
    result["n"] = len(_align(y_true, y_pred)[0])
    if anchor is not None:
        result["directional_accuracy"] = directional_accuracy(y_true, y_pred, anchor)
    if y_naive is not None:
        result["theil_u2"] = theil_u2(y_true, y_pred, y_naive)
        result["mae_skill_vs_naive_pct"] = skill_score(y_true, y_pred, y_naive, mae)
    return result


def metrics_table(
    frame: pd.DataFrame,
    *,
    group_by: list[str],
    actual_column: str = "actual",
    prediction_column: str = "prediction",
    anchor_column: str | None = "anchor",
    naive_column: str | None = None,
) -> pd.DataFrame:
    """Point metrics computed per group of a tidy predictions frame."""
    rows = []
    for keys, group in frame.groupby(group_by, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(group_by, key_tuple, strict=False))
        row.update(
            evaluate_point_forecast(
                group[actual_column],
                group[prediction_column],
                anchor=group[anchor_column] if anchor_column in group else None,
                y_naive=group[naive_column] if naive_column and naive_column in group else None,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)
