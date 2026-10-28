"""The forecaster interface every model in this project implements.

One interface means the backtester, the metrics, the conformal wrapper, the API
and the dashboard treat a naive baseline and a gradient-boosted model
identically — and that a comparison table is a like-for-like comparison.

The contract
------------
``fit(frame, target, horizon, config, train_end)``
    Train using **only** rows of ``frame`` with index <= ``train_end``.

``predict_origins(frame, origins)``
    For each origin date ``o``, return the forecast **level** for ``o + horizon``
    business days, using only rows with index <= ``o``.  The returned series is
    indexed by *origin*, not by target date; :meth:`target_dates` maps between
    the two.

The separation between "fit sees data up to train_end" and "predict may read up
to each origin" is deliberate: it is how forecasting actually works in
production, where the model is retrained periodically but consumes fresh
observations every day.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd


class NotFittedError(RuntimeError):
    """Raised when predict is called before fit."""


class Forecaster(ABC):
    """Base class for every point forecaster."""

    name: str = "forecaster"
    #: Set by subclasses that cannot honour an arbitrary origin without refitting.
    refit_per_origin: bool = False

    def __init__(self) -> None:
        self.target: str | None = None
        self.horizon: int | None = None
        self.config: dict[str, Any] = {}
        self.train_end: pd.Timestamp | None = None
        self._fitted = False

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def _fit(self, frame: pd.DataFrame) -> None:
        """Subclass hook; ``frame`` is already truncated to the training window."""

    @abstractmethod
    def _predict_origins(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        """Subclass hook; must return forecast **levels**, one per origin."""

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        horizon: int,
        config: dict[str, Any],
        *,
        train_end: pd.Timestamp | None = None,
    ) -> Self:
        if target not in frame.columns:
            raise KeyError(f"{target!r} not in frame columns {list(frame.columns)}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        self.target = target
        self.horizon = int(horizon)
        self.config = config
        self.train_end = (
            pd.Timestamp(train_end) if train_end is not None else pd.Timestamp(frame.index[-1])
        )

        training_frame = frame.loc[: self.train_end]
        if training_frame.empty:
            raise ValueError(f"no training rows at or before {self.train_end}")
        self._fit(training_frame)
        self._fitted = True
        return self

    def predict_origins(
        self, frame: pd.DataFrame, origins: pd.DatetimeIndex | pd.Series | list
    ) -> pd.Series:
        """Forecast levels for ``origin + horizon``, indexed by origin date."""
        if not self._fitted:
            raise NotFittedError(f"{self.name} must be fitted before predicting")
        origin_index = pd.DatetimeIndex(pd.Index(origins))
        missing = origin_index.difference(frame.index)
        if len(missing):
            raise KeyError(f"{self.name}: origins absent from the frame: {list(missing[:5])}")
        values = np.asarray(self._predict_origins(frame, origin_index), dtype="float64")
        if values.shape != (len(origin_index),):
            raise ValueError(
                f"{self.name} returned {values.shape}, expected ({len(origin_index)},)"
            )
        return pd.Series(values, index=origin_index, name=self.name)

    # -- helpers -----------------------------------------------------------
    def target_dates(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Map origin dates to the dates their forecasts refer to."""
        positions = frame.index.get_indexer(origins) + (self.horizon or 0)
        clipped = np.clip(positions, 0, len(frame.index) - 1)
        dates = frame.index.to_numpy()[clipped]
        # Positions past the end of the frame refer to dates that do not exist
        # yet; mark them rather than silently returning the last known date.
        return pd.DatetimeIndex(np.where(positions < len(frame.index), dates, np.datetime64("NaT")))

    def actuals(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.Series:
        """Realised value at each origin's target date; NaN where unknown."""
        dates = self.target_dates(frame, origins)
        values = [float(frame[self.target].loc[d]) if pd.notna(d) else np.nan for d in dates]
        return pd.Series(values, index=origins, name="actual")

    def forecast_future(self, frame: pd.DataFrame) -> tuple[pd.Timestamp, float]:
        """Forecast for ``horizon`` business days past the end of ``frame``."""
        last = pd.Timestamp(frame.index[-1])
        value = float(self.predict_origins(frame, pd.DatetimeIndex([last])).iloc[0])
        future_date = pd.date_range(
            last + pd.tseries.offsets.BDay(1), periods=self.horizon, freq="B"
        )[-1]
        return future_date, value

    def __repr__(self) -> str:
        state = f"target={self.target}, h={self.horizon}" if self._fitted else "unfitted"
        return f"{type(self).__name__}({state})"


def positions_of(frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
    """Integer positions of ``origins`` within ``frame.index``."""
    positions = frame.index.get_indexer(origins)
    if (positions < 0).any():
        raise KeyError("some origins are not present in the frame index")
    return positions
