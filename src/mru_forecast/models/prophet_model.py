"""Prophet forecaster (Phase 8).

Prophet decomposes a series into trend + seasonality + holidays and is fitted on
a history, not on a feature matrix.  It therefore does not naturally honour this
project's origin protocol, where a forecast made at origin ``o`` must use every
observation up to ``o``: Prophet only knows what it was fitted on.

The resolution is an explicit **refit cadence**.  ``refit_every=k`` refits the
model on the data up to an origin, then reuses it for the next ``k-1`` origins.
``k=1`` is exact but costs one Stan fit per origin; larger ``k`` trades a little
staleness for speed.  The cadence is recorded in the results so a comparison
never silently mixes protocols.

Why Prophet is here at all
--------------------------
It is the standard "does this series have trend and seasonality worth modelling
explicitly?" check.  On a managed rate the expected answer is *not much* — which
is a result worth measuring rather than assuming.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from mru_forecast.models.base import Forecaster
from mru_forecast.utils.logging import get_logger

logger = get_logger(__name__)

# cmdstanpy narrates every fit at INFO; silence it once, here.
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


class ProphetForecaster(Forecaster):
    """Additive trend + seasonality model, refitted on a configurable cadence."""

    name = "prophet"
    refit_per_origin = True

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        refit_every: int = 21,
        regressors: list[str] | None = None,
        max_history: int | None = 1500,
    ) -> None:
        super().__init__()
        self.params = dict(params or {})
        self.refit_every = max(int(refit_every), 1)
        self.regressors = list(regressors or [])
        # Prophet's runtime grows with history; a long trailing window keeps the
        # backtest tractable without discarding the shape of the series.
        self.max_history = max_history
        self.model = None
        self._cache: dict[pd.Timestamp, Any] = {}

    # -- internals ---------------------------------------------------------
    def _make_model(self):
        from prophet import Prophet

        settings = {**self.config["models"]["prophet"], **self.params}
        model = Prophet(
            growth=settings.get("growth", "linear"),
            yearly_seasonality=settings.get("yearly_seasonality", True),
            weekly_seasonality=settings.get("weekly_seasonality", True),
            daily_seasonality=settings.get("daily_seasonality", False),
            changepoint_prior_scale=settings.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=settings.get("seasonality_prior_scale", 10.0),
            interval_width=settings.get("interval_width", 0.90),
        )
        for regressor in self.regressors:
            model.add_regressor(regressor)
        return model

    def _training_frame(self, frame: pd.DataFrame, upto: pd.Timestamp) -> pd.DataFrame:
        history = frame.loc[:upto]
        if self.max_history:
            history = history.iloc[-self.max_history :]
        out = pd.DataFrame({"ds": history.index, "y": history[self.target].to_numpy()})
        for regressor in self.regressors:
            # Regressors are lagged by one day: their same-day value would not be
            # known when the forecast is made.
            out[regressor] = history[regressor].shift(1).bfill().to_numpy()
        return out

    def _fit_at(self, frame: pd.DataFrame, upto: pd.Timestamp):
        """Fit (or reuse a cached fit) on data up to ``upto``."""
        if upto in self._cache:
            return self._cache[upto]
        model = self._make_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self._training_frame(frame, upto))
        self._cache = {upto: model}  # only the most recent fit is ever reused
        return model

    def _fit(self, frame: pd.DataFrame) -> None:
        available = [r for r in self.regressors if r in frame.columns]
        if len(available) != len(self.regressors):
            missing = set(self.regressors) - set(available)
            logger.warning("prophet: dropping unavailable regressors %s", sorted(missing))
            self.regressors = available
        self._cache.clear()
        self.model = self._fit_at(frame, pd.Timestamp(frame.index[-1]))

    def _predict_origins(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        horizon = int(self.horizon)
        predictions = np.full(len(origins), np.nan)
        target_dates = list(self.target_dates(frame, origins))

        # Dates past the end of the frame do not exist yet; project them onto
        # the business-day calendar instead.
        for i, (origin, target_date) in enumerate(zip(origins, target_dates, strict=True)):
            if pd.isna(target_date):
                target_dates[i] = pd.date_range(
                    origin + pd.tseries.offsets.BDay(1), periods=horizon, freq="B"
                )[-1]

        # Origins are grouped into blocks that share one fit, and each block is
        # predicted in a single call: Prophet's per-call overhead dominates its
        # per-row cost, so batching a block is roughly free.
        positions = frame.index.get_indexer(origins)
        blocks: list[list[int]] = []
        anchor_position: int | None = None
        for i, position in enumerate(positions):
            if anchor_position is None or position - anchor_position >= self.refit_every:
                anchor_position = position
                blocks.append([])
            blocks[-1].append(i)

        for block in blocks:
            fit_origin = origins[block[0]]
            model = self._fit_at(frame, fit_origin)
            future = pd.DataFrame({"ds": [target_dates[i] for i in block]})
            for regressor in self.regressors:
                # Carry the last known regressor value forward: its future path
                # is itself unknown, and inventing one would be a leak.
                future[regressor] = float(frame[regressor].loc[:fit_origin].iloc[-1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yhat = model.predict(future)["yhat"].to_numpy(dtype="float64")
            predictions[block] = yhat

        return predictions

    def components(self, frame: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
        """Trend / seasonality decomposition, for the dashboard and notebooks."""
        if self.model is None:
            raise RuntimeError("model is not fitted")
        future = self.model.make_future_dataframe(periods=periods, freq="B")
        for regressor in self.regressors:
            future[regressor] = float(frame[regressor].iloc[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)
        keep = [
            c
            for c in ("ds", "trend", "yearly", "weekly", "yhat", "yhat_lower", "yhat_upper")
            if c in forecast.columns
        ]
        return forecast[keep].set_index("ds")


def build_prophet(config: dict[str, Any], **overrides: Any) -> ProphetForecaster:
    regressors = config["models"]["prophet"].get("regressors") or []
    settings = config["models"]["prophet"]
    return ProphetForecaster(
        params=overrides or None,
        refit_every=int(settings.get("refit_every", 21)),
        regressors=regressors,
        max_history=settings.get("max_history", 1500),
    )
