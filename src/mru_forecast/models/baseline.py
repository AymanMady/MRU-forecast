"""Baseline forecasters (Phase 6).

These are not filler.  For an exchange rate the random walk is the benchmark the
academic literature has repeatedly found hard to beat, and on a *managed* rate
that does not move at all on ~30% of business days it is stronger still.  A
model that does not beat these has not earned its complexity, and the project
reports that outcome rather than hiding it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mru_forecast.models.base import Forecaster, positions_of


class NaiveForecaster(Forecaster):
    """Random walk: ``prediction(t + h) = value(t)``.

    Optimal when log-returns are a martingale difference sequence, which is the
    standard working hypothesis for a floating exchange rate.
    """

    name = "naive"

    def _fit(self, frame: pd.DataFrame) -> None:  # nothing to learn
        return None

    def _predict_origins(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        return frame[self.target].reindex(origins).to_numpy(dtype="float64")


class SeasonalNaiveForecaster(Forecaster):
    """``prediction(t + h) = value(t + h - m*s)`` for the smallest valid ``m``.

    With ``season_length=5`` (one business week) the forecast for next Tuesday is
    last Tuesday's rate.  Included to test whether any weekly structure exists —
    the EDA says it does not, and this baseline is how that claim is checked
    rather than asserted.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = 5) -> None:
        super().__init__()
        self.season_length = int(season_length)
        if self.season_length < 1:
            raise ValueError("season_length must be >= 1")

    def _fit(self, frame: pd.DataFrame) -> None:
        return None

    def _predict_origins(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        values = frame[self.target].to_numpy(dtype="float64")
        positions = positions_of(frame, origins)
        horizon = int(self.horizon)
        # Step back whole seasons from the target date until the reference lands
        # at or before the origin — never after it.
        seasons_back = int(np.ceil(horizon / self.season_length))
        offsets = horizon - seasons_back * self.season_length
        reference = positions + offsets
        reference = np.clip(reference, 0, len(values) - 1)
        return values[reference]


class DriftForecaster(Forecaster):
    """Random walk with drift estimated over a trailing window.

    ``prediction(t + h) = value(t) * (1 + mean_daily_log_return) ** h``, in log
    space.  The drift uses only the window ending at the origin, so it adapts as
    the crawl of a managed rate changes pace.
    """

    name = "drift"

    def __init__(self, window: int = 252) -> None:
        super().__init__()
        self.window = int(window)

    def _fit(self, frame: pd.DataFrame) -> None:
        return None

    def _predict_origins(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        series = frame[self.target].astype("float64")
        log_returns = np.log(series).diff()
        drift = log_returns.rolling(self.window, min_periods=20).mean().reindex(origins)
        level = series.reindex(origins)
        return (level * np.exp(drift.fillna(0.0) * self.horizon)).to_numpy(dtype="float64")


class MeanReversionForecaster(Forecaster):
    """Reversion of the level towards a trailing mean.

    ``prediction(t + h) = value(t) + kappa_h * (mean_w(t) - value(t))``

    ``kappa_h`` is estimated on the training window by least squares, so the
    model can learn "no reversion" (kappa ~ 0, i.e. the naive forecast) if the
    data says so.  Included because the EDA found ACF(1) ~ -0.30 on this series:
    that finding deserves a baseline that exploits it directly, before any claim
    that a gradient-boosted model was needed to capture it.
    """

    name = "mean_reversion"

    def __init__(self, window: int = 21) -> None:
        super().__init__()
        self.window = int(window)
        self.kappa: float = 0.0

    def _reversion_gap(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window, min_periods=max(self.window // 2, 2)).mean() - series

    def _fit(self, frame: pd.DataFrame) -> None:
        series = frame[self.target].astype("float64")
        gap = self._reversion_gap(series)
        change = series.shift(-int(self.horizon)) - series
        usable = pd.concat([gap.rename("gap"), change.rename("change")], axis=1).dropna()
        if len(usable) < 30 or float(usable["gap"].var()) <= 0:
            self.kappa = 0.0
            return
        # Least squares through the origin: kappa = <gap, change> / <gap, gap>
        self.kappa = float((usable["gap"] * usable["change"]).sum() / (usable["gap"] ** 2).sum())

    def _predict_origins(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        series = frame[self.target].astype("float64")
        gap = self._reversion_gap(series).reindex(origins).fillna(0.0)
        level = series.reindex(origins)
        return (level + self.kappa * gap).to_numpy(dtype="float64")


BASELINE_REGISTRY = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "drift": DriftForecaster,
    "mean_reversion": MeanReversionForecaster,
}


def build_baselines(config: dict) -> list[Forecaster]:
    """Instantiate the baselines enabled in the configuration."""
    settings = config["models"]["baseline"]
    models: list[Forecaster] = []
    if settings.get("naive", True):
        models.append(NaiveForecaster())
    if settings.get("seasonal_naive"):
        models.append(
            SeasonalNaiveForecaster(int(settings["seasonal_naive"].get("season_length", 5)))
        )
    if settings.get("drift", True):
        models.append(DriftForecaster())
    if settings.get("mean_reversion", True):
        models.append(MeanReversionForecaster())
    return models
