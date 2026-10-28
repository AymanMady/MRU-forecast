"""Gradient-boosted forecaster (Phase 7).

Multi-step strategy: **direct**
------------------------------
One model is trained per horizon, each mapping features known at ``t`` straight
to the outcome at ``t + h``.  The alternative, *recursive*, trains a single
one-step model and feeds its own output back in; each step then inherits the
previous step's error, and by h = 30 the compounded error dominates.  Direct
costs one model per horizon and buys an error that does not compound.  Since
this project only needs three horizons, the trade is clearly worth it.

Target: the h-step log return, anchored on the value at ``t`` (see
``features.engineering``).  Trees cannot extrapolate beyond the range of the
label they were trained on, so a level target would put a hard ceiling on any
forecast at the highest rate ever seen in training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from mru_forecast.features.engineering import (
    build_features,
    inverse_transform,
    make_supervised,
)
from mru_forecast.models.base import Forecaster
from mru_forecast.utils.logging import get_logger

logger = get_logger(__name__)


class XGBoostForecaster(Forecaster):
    """Direct-strategy XGBoost regressor on engineered lag/rolling features."""

    name = "xgboost"

    def __init__(
        self, params: dict[str, Any] | None = None, *, sample_weight_halflife: int | None = 504
    ) -> None:
        super().__init__()
        self.params = dict(params or {})
        # Exponentially decaying sample weights: a managed currency's regime
        # changes, and a 2019 observation should not count as much as a 2026 one.
        # `None` disables the weighting.
        self.sample_weight_halflife = sample_weight_halflife
        self.model: XGBRegressor | None = None
        self.transform: str = "log_return"
        self.feature_names: list[str] = []
        self.n_train: int = 0

    # -- internals ---------------------------------------------------------
    def _sample_weights(self, index: pd.DatetimeIndex) -> np.ndarray | None:
        if not self.sample_weight_halflife:
            return None
        age_days = (index[-1] - index).days.to_numpy(dtype="float64")
        return np.power(0.5, age_days / float(self.sample_weight_halflife))

    def _fit(self, frame: pd.DataFrame) -> None:
        params = {**self.config["models"]["xgboost"]["params"], **self.params}
        dataset = make_supervised(frame, self.target, self.horizon, self.config)
        if len(dataset) < 50:
            raise ValueError(
                f"only {len(dataset)} training rows for {self.target} h={self.horizon}; "
                "need at least 50"
            )

        self.transform = dataset.transform
        self.feature_names = list(dataset.X.columns)
        self.n_train = len(dataset)

        self.model = XGBRegressor(**params)
        self.model.fit(
            dataset.X,
            dataset.y,
            sample_weight=self._sample_weights(dataset.index),
            verbose=False,
        )
        logger.debug(
            "xgboost fitted: %s h=%d on %d rows, %d features",
            self.target,
            self.horizon,
            self.n_train,
            len(self.feature_names),
        )

    def _predict_origins(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        # Features are trailing-only, so computing them on the whole frame and
        # then selecting the origin rows reads no future information.
        features = build_features(frame, self.target, self.config)
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"features missing at predict time: {missing[:5]}")

        matrix = features.loc[origins, self.feature_names]
        raw = self.model.predict(matrix)
        anchor = frame[self.target].reindex(origins).to_numpy(dtype="float64")
        return inverse_transform(raw, anchor, self.transform)

    # -- introspection -----------------------------------------------------
    def feature_importance(self, top_n: int | None = None) -> pd.DataFrame:
        """Gain-based importance, normalised to sum to 1."""
        if self.model is None:
            raise RuntimeError("model is not fitted")
        booster = self.model.get_booster()
        gains = booster.get_score(importance_type="gain")
        total = sum(gains.values()) or 1.0
        table = (
            pd.DataFrame({"feature": list(gains), "gain": [v / total for v in gains.values()]})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
        return table.head(top_n) if top_n else table


def build_xgboost(config: dict[str, Any], **overrides: Any) -> XGBoostForecaster:
    return XGBoostForecaster(params=overrides or None)
