"""Service layer: everything the API needs, loaded once and cached.

Keeping this out of ``main.py`` means the same logic is reachable from the
dashboard and from tests without starting a web server.
"""

from __future__ import annotations

import functools
from typing import Any

import pandas as pd

from mru_forecast import __version__
from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.cleaning import load_processed
from mru_forecast.models.conformal import level_tag
from mru_forecast.pipeline.forecasting_pipeline import ForecastingPipeline
from mru_forecast.utils.logging import get_logger

logger = get_logger(__name__)

PAIR_ALIASES = {
    "usd-mru": "USD_MRU",
    "usd_mru": "USD_MRU",
    "usdmru": "USD_MRU",
    "eur-mru": "EUR_MRU",
    "eur_mru": "EUR_MRU",
    "eurmru": "EUR_MRU",
}


class ServiceUnavailableError(RuntimeError):
    """The artefacts the API needs are missing."""


def resolve_pair(pair: str) -> str:
    key = pair.strip().lower().replace("/", "-")
    if key in PAIR_ALIASES:
        return PAIR_ALIASES[key]
    raise KeyError(
        f"unknown currency pair {pair!r}; try one of {sorted(set(PAIR_ALIASES.values()))}"
    )


def _forecast_point(index, row, tag: str) -> dict[str, Any]:
    """One row of the served forecast path.

    ``interval_width`` is derived from the **rounded** bounds, so a client that
    computes ``upper - lower`` gets exactly the number the API reports.
    """
    lower = round(float(row[f"lower_{tag}"]), 6)
    upper = round(float(row[f"upper_{tag}"]), 6)
    return {
        "date": index.date(),
        "days_ahead": int(row["days_ahead"]),
        "prediction": round(float(row["prediction"]), 6),
        "lower": lower,
        "upper": upper,
        "interval_width": round(upper - lower, 6),
    }


class ForecastService:
    """Holds the trained pipeline and the processed dataset."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()
        self._frame: pd.DataFrame | None = None
        self._pipeline: ForecastingPipeline | None = None
        self.load_error: str | None = None
        self._reload()

    # -- lifecycle ---------------------------------------------------------
    def _reload(self) -> None:
        try:
            self._frame = load_processed(self.config)
            self._pipeline = ForecastingPipeline.load(config=self.config)
            self.load_error = None
        except FileNotFoundError as exc:
            # Serving a health endpoint that says "not ready" is more useful
            # than refusing to start.
            self.load_error = str(exc)
            logger.warning("service starting in degraded mode: %s", exc)

    def refresh(self) -> None:
        self._reload()

    @property
    def ready(self) -> bool:
        return self._frame is not None and self._pipeline is not None

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            raise ServiceUnavailableError(self.load_error or "processed dataset not loaded")
        return self._frame

    @property
    def pipeline(self) -> ForecastingPipeline:
        if self._pipeline is None:
            raise ServiceUnavailableError(self.load_error or "trained pipeline not loaded")
        return self._pipeline

    # -- target metadata ---------------------------------------------------
    def target_info(self, target: str) -> dict[str, Any]:
        for entry in self.config["targets"]:
            if entry["name"] == target:
                return entry
        raise KeyError(f"unknown target {target!r}")

    def health(self) -> dict[str, Any]:
        metadata = self._pipeline.metadata if self._pipeline else {}
        return {
            "status": "ok" if self.ready else "degraded",
            "version": __version__,
            "models_loaded": self.ready,
            "data_last_date": (self._frame.index.max().date() if self._frame is not None else None),
            "n_predictors": len(self._pipeline.steps) if self._pipeline else 0,
            "trained_at": metadata.get("trained_at"),
            "detail": self.load_error,
        }

    # -- endpoints ---------------------------------------------------------
    def max_horizon(self, target: str) -> int:
        horizons = [h for (t, h) in self.pipeline.steps if t == target]
        if not horizons:
            raise KeyError(f"no trained predictor for {target!r}")
        return max(horizons)

    def available_confidences(self) -> list[float]:
        return list(self.config["conformal"]["confidence_levels"])

    def forecast(self, target: str, horizon: int, confidence: float) -> dict[str, Any]:
        if confidence not in self.available_confidences():
            raise ValueError(
                f"confidence {confidence} is not calibrated; "
                f"available: {self.available_confidences()}"
            )
        limit = self.max_horizon(target)
        if not 1 <= horizon <= limit:
            raise ValueError(f"horizon must be between 1 and {limit}")

        path = self.pipeline.forecast(
            self.frame, target=target, horizon=horizon, confidence=confidence
        )
        tag = level_tag(confidence)
        info = self.target_info(target)

        return {
            "currency_pair": target.replace("_", "/"),
            "rate_definition": info.get("definition", ""),
            "rate_type": info.get("rate_type", ""),
            "is_derived_cross": not bool(info.get("observed", True)),
            "horizon": horizon,
            "confidence": confidence,
            "anchor_date": path["anchor_date"].iloc[0].date(),
            "anchor_value": round(float(path["anchor_value"].iloc[0]), 6),
            "model": self.pipeline.metadata.get("model", "unknown"),
            "forecast": [_forecast_point(index, row, tag) for index, row in path.iterrows()],
        }

    def history(self, target: str, days: int = 365) -> dict[str, Any]:
        series = self.frame[target].dropna().iloc[-max(days, 1) :]
        info = self.target_info(target)
        return {
            "currency_pair": target.replace("_", "/"),
            "rate_type": info.get("rate_type", ""),
            "n_observations": len(series),
            "first_date": series.index.min().date(),
            "last_date": series.index.max().date(),
            "observations": [
                {"date": index.date(), "value": round(float(value), 6)}
                for index, value in series.items()
            ],
        }

    def latest_rates(self) -> dict[str, Any]:
        targets = [t["name"] for t in self.config["targets"]]
        frame = self.frame[targets]
        last = frame.iloc[-1]
        change = frame.pct_change().iloc[-1] * 100.0
        return {
            "as_of": frame.index.max().date(),
            "rates": {name.replace("_", "/"): round(float(last[name]), 6) for name in targets},
            "daily_change_pct": {
                name.replace("_", "/"): round(float(change[name]), 4) for name in targets
            },
            "rate_type": self.target_info(targets[0]).get("rate_type", ""),
        }

    # -- backtest reports --------------------------------------------------
    @functools.cached_property
    def _reports_dir(self):
        return resolve_path("reports", self.config)

    def _read_report(self, filename: str) -> pd.DataFrame:
        path = self._reports_dir / filename
        if not path.exists():
            raise ServiceUnavailableError(
                f"{filename} not found — run `python scripts/backtest.py` first."
            )
        return pd.read_csv(path)

    def performance(self, target: str | None = None) -> dict[str, Any]:
        table = self._read_report("metrics_point.csv")
        if target:
            table = table[table["target"] == target]
        meta_path = self._reports_dir / "backtest_meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        columns = [
            "target",
            "horizon",
            "model",
            "mae",
            "rmse",
            "mape",
            "theil_u2",
            "mae_skill_vs_naive_pct",
            "n",
        ]
        available = [c for c in columns if c in table.columns]
        return {
            "backtest_period": meta.get("test_period"),
            "n_splits": meta.get("n_splits"),
            "rows": table[available].round(6).to_dict(orient="records"),
        }

    def calibration(self) -> dict[str, Any]:
        table = self._read_report("metrics_calibration.csv")
        columns = [
            "model",
            "nominal_coverage",
            "empirical_coverage",
            "coverage_error_pp",
            "coverage_se_pp",
            "mean_interval_width",
            "n",
        ]
        available = [c for c in columns if c in table.columns]
        return {"rows": table[available].round(6).to_dict(orient="records")}


@functools.lru_cache(maxsize=1)
def get_service() -> ForecastService:
    """Process-wide singleton; models are loaded once, not per request."""
    return ForecastService()
