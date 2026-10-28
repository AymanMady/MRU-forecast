"""End-to-end forecasting pipeline (Phases 12-14).

Turns the processed dataset into servable artefacts:

    processed data
        -> for each target and each step h in 1..max_horizon
             fit the point model on the training block
             calibrate the conformal quantiles on the held-out calibration block
        -> persist the fitted predictors + metadata
        -> produce a daily forecast path with prediction intervals

Why one model per step
----------------------
The direct strategy (see ``models.xgboost_model``) needs a model per horizon.
Serving a *path* — tomorrow, +2 days, ... +30 days — therefore means 30 models
per target.  They are small and train in about a second each, so the honest
error profile of the direct strategy is worth far more than the training time
it costs.

Calibration window
------------------
The calibration block is the most recent ``calibration_size`` origins whose
outcome is already known (``origin + h <= last date``).  Training stops before
it.  Each horizon therefore has a slightly different training cut-off, which is
recorded per model rather than glossed over.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.cleaning import load_processed
from mru_forecast.data.validation import validate_forecast_intervals
from mru_forecast.models.base import Forecaster
from mru_forecast.models.baseline import (
    DriftForecaster,
    MeanReversionForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from mru_forecast.models.conformal import ConformalPredictor, build_conformal, level_tag
from mru_forecast.models.xgboost_model import XGBoostForecaster
from mru_forecast.utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACT_FILENAME = "predictors.pkl"
METADATA_FILENAME = "pipeline_metadata.json"
FORECAST_FILENAME = "latest_forecast.parquet"

MODEL_BUILDERS = {
    "xgboost": lambda config: XGBoostForecaster(),
    "naive": lambda config: NaiveForecaster(),
    "drift": lambda config: DriftForecaster(),
    "mean_reversion": lambda config: MeanReversionForecaster(),
    "seasonal_naive": lambda config: SeasonalNaiveForecaster(
        int(config["models"]["baseline"]["seasonal_naive"]["season_length"])
    ),
    "prophet": lambda config: __import__(
        "mru_forecast.models.prophet_model", fromlist=["build_prophet"]
    ).build_prophet(config),
}


def build_model(name: str, config: dict[str, Any]) -> Forecaster:
    if name not in MODEL_BUILDERS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](config)


@dataclass
class FittedStep:
    """One (target, horizon) predictor and what it was fitted on."""

    target: str
    horizon: int
    model_name: str
    predictor: ConformalPredictor
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    n_train: int
    n_calibration: int

    def metadata(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "horizon": self.horizon,
            "model": self.model_name,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "calibration_start": str(self.calibration_start.date()),
            "calibration_end": str(self.calibration_end.date()),
            "n_train": self.n_train,
            "n_calibration": self.n_calibration,
            "conformal": self.predictor.summary.as_dict() if self.predictor.summary else None,
        }


@dataclass
class ForecastingPipeline:
    """Fit, persist and serve conformal forecasters for every target and step."""

    config: dict[str, Any] = field(default_factory=load_config)
    steps: dict[tuple[str, int], FittedStep] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- training ----------------------------------------------------------
    def train(
        self,
        frame: pd.DataFrame | None = None,
        *,
        targets: list[str] | None = None,
        horizons: list[int] | None = None,
        model_name: str | None = None,
    ) -> ForecastingPipeline:
        settings = self.config["pipeline"]
        data = frame if frame is not None else load_processed(self.config)
        target_names = targets or [t["name"] for t in self.config["targets"]]
        max_horizon = int(settings.get("max_horizon", 30))
        steps = horizons or list(range(1, max_horizon + 1))
        name = model_name or settings.get("model", "xgboost")
        calibration_size = int(settings.get("calibration_size", 250))
        adaptive = bool(settings.get("adaptive_conformal", False))

        self.steps.clear()
        failures: dict[str, str] = {}

        for target in target_names:
            for horizon in steps:
                try:
                    self.steps[(target, horizon)] = self._fit_step(
                        data, target, horizon, name, calibration_size, adaptive
                    )
                except Exception as exc:
                    logger.warning("%s h=%d failed: %s", target, horizon, exc)
                    failures[f"{target}_h{horizon}"] = str(exc)

        if not self.steps:
            raise RuntimeError("no step trained successfully")

        self.metadata = {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": name,
            "targets": target_names,
            "horizons": steps,
            "confidence_levels": self.config["conformal"]["confidence_levels"],
            "primary_level": self.config["conformal"]["primary_level"],
            "data_first_date": str(data.index.min().date()),
            "data_last_date": str(data.index.max().date()),
            "n_observations": len(data),
            "n_steps_trained": len(self.steps),
            "failures": failures,
            "steps": [step.metadata() for step in self.steps.values()],
        }
        logger.info(
            "pipeline trained: %d predictors (%s) on data to %s",
            len(self.steps),
            name,
            self.metadata["data_last_date"],
        )
        return self

    def _fit_step(
        self,
        frame: pd.DataFrame,
        target: str,
        horizon: int,
        model_name: str,
        calibration_size: int,
        adaptive: bool,
    ) -> FittedStep:
        index = pd.DatetimeIndex(frame.index)
        # Only origins whose outcome is already observed can be calibrated on.
        usable_end = len(index) - horizon
        calibration_start = usable_end - calibration_size
        if calibration_start <= 100:
            raise ValueError(
                f"not enough history for h={horizon}: {len(index)} rows, "
                f"calibration_size={calibration_size}"
            )

        calibration_origins = index[calibration_start:usable_end]
        train_end = index[calibration_start - 1]

        predictor = build_conformal(
            build_model(model_name, self.config), self.config, adaptive=adaptive
        )
        predictor.fit(frame, target, horizon, self.config, train_end=train_end)
        predictor.calibrate(frame, calibration_origins)

        return FittedStep(
            target=target,
            horizon=horizon,
            model_name=model_name,
            predictor=predictor,
            train_start=index[0],
            train_end=train_end,
            calibration_start=calibration_origins[0],
            calibration_end=calibration_origins[-1],
            n_train=int(calibration_start),
            n_calibration=len(calibration_origins),
        )

    # -- forecasting -------------------------------------------------------
    def forecast(
        self,
        frame: pd.DataFrame | None = None,
        *,
        target: str,
        horizon: int | None = None,
        confidence: float | list[float] | None = None,
    ) -> pd.DataFrame:
        """Daily forecast path for ``target`` out to ``horizon`` business days.

        One row per future business day, with the point forecast and every
        calibrated interval.  ``lower <= prediction <= upper`` is asserted
        before the frame is returned.
        """
        if not self.steps:
            raise RuntimeError("pipeline is not trained; call train() or load()")
        data = frame if frame is not None else load_processed(self.config)

        available = sorted(h for (t, h) in self.steps if t == target)
        if not available:
            raise KeyError(f"no trained step for target {target!r}")
        limit = int(horizon) if horizon else max(available)
        steps = [h for h in available if h <= limit]
        if not steps:
            raise KeyError(f"no trained step for {target!r} at horizon <= {limit}")

        rows = []
        for h in steps:
            predictor = self.steps[(target, h)].predictor
            row = predictor.forecast_future(data, confidence=confidence)
            rows.append(row)

        path = pd.concat(rows).sort_index()
        path.index.name = "date"
        path.insert(0, "target", target)
        path["days_ahead"] = steps
        path["anchor_value"] = float(data[target].iloc[-1])
        path["anchor_date"] = pd.Timestamp(data.index[-1])

        for level in self._levels(path):
            report = validate_forecast_intervals(
                path["prediction"],
                path[f"lower_{level}"],
                path[f"upper_{level}"],
                name=f"{target}_h{limit}_{level}",
            )
            report.raise_if_failed()
        return path

    def forecast_all(
        self, frame: pd.DataFrame | None = None, *, horizon: int | None = None
    ) -> pd.DataFrame:
        data = frame if frame is not None else load_processed(self.config)
        targets = sorted({t for (t, _) in self.steps})
        return pd.concat(
            [self.forecast(data, target=t, horizon=horizon) for t in targets]
        ).reset_index()

    @staticmethod
    def _levels(frame: pd.DataFrame) -> list[int]:
        return sorted(
            int(c.removeprefix("lower_")) for c in frame.columns if c.startswith("lower_")
        )

    # -- persistence -------------------------------------------------------
    def save(self, directory: str | Path | None = None) -> Path:
        target_dir = Path(directory) if directory else resolve_path("models", self.config)
        target_dir.mkdir(parents=True, exist_ok=True)
        with (target_dir / ARTIFACT_FILENAME).open("wb") as fh:
            pickle.dump(self.steps, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (target_dir / METADATA_FILENAME).write_text(
            json.dumps(self.metadata, indent=2, default=str), encoding="utf-8"
        )
        logger.info("pipeline artefacts -> %s", target_dir)
        return target_dir

    @classmethod
    def load(
        cls, directory: str | Path | None = None, config: dict[str, Any] | None = None
    ) -> ForecastingPipeline:
        cfg = config if config is not None else load_config()
        source = Path(directory) if directory else resolve_path("models", cfg)
        artefact = source / ARTIFACT_FILENAME
        if not artefact.exists():
            raise FileNotFoundError(f"{artefact} not found — run `python scripts/train.py` first.")
        with artefact.open("rb") as fh:
            steps = pickle.load(fh)
        metadata_path = source / METADATA_FILENAME
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        )
        return cls(config=cfg, steps=steps, metadata=metadata)

    # -- reporting ---------------------------------------------------------
    def calibration_summary(self) -> pd.DataFrame:
        """Conformal quantile per (target, horizon) — how wide, and from how many scores."""
        rows = []
        for (target, horizon), step in sorted(self.steps.items()):
            summary = step.predictor.summary
            row: dict[str, Any] = {
                "target": target,
                "horizon": horizon,
                "model": step.model_name,
                "n_calibration": step.n_calibration,
                "train_end": str(step.train_end.date()),
                "method": summary.method if summary else None,
            }
            for level, value in (summary.quantiles if summary else {}).items():
                row[f"q_{int(level * 100)}"] = value
            rows.append(row)
        return pd.DataFrame(rows)

    def feature_importance(self, target: str, horizon: int, top_n: int = 20) -> pd.DataFrame:
        step = self.steps.get((target, horizon))
        if step is None:
            raise KeyError(f"no step for ({target}, {horizon})")
        base = step.predictor.base_model
        if not hasattr(base, "feature_importance"):
            raise TypeError(f"{base.name} does not expose feature importance")
        return base.feature_importance(top_n)


# ---------------------------------------------------------------------------
# MLflow (Phase 14)
# ---------------------------------------------------------------------------
def log_to_mlflow(
    pipeline: ForecastingPipeline,
    *,
    point_metrics: pd.DataFrame | None = None,
    interval_metrics: pd.DataFrame | None = None,
    run_name: str | None = None,
    artifacts: list[Path] | None = None,
) -> str | None:
    """Record the run in MLflow: parameters, metrics, artefacts and dataset version.

    Returns the run id, or ``None`` if MLflow is unavailable — tracking must
    never be the reason a training run fails.
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        logger.warning("mlflow is not installed; skipping tracking")
        return None

    settings = pipeline.config.get("mlflow", {})
    try:
        mlflow.set_tracking_uri(settings.get("tracking_uri", "sqlite:///mlflow.db"))
        mlflow.set_experiment(settings.get("experiment_name", "mru-forecast"))
    except Exception as exc:
        logger.warning("mlflow unavailable (%s); skipping tracking", exc)
        return None

    metadata = pipeline.metadata
    with mlflow.start_run(run_name=run_name or f"{metadata.get('model')}-conformal") as run:
        mlflow.log_params(
            {
                "model": metadata.get("model"),
                "n_steps": metadata.get("n_steps_trained"),
                "max_horizon": max(metadata.get("horizons", [0])),
                "targets": ",".join(metadata.get("targets", [])),
                "conformal_method": pipeline.config["conformal"].get("method"),
                "normalize_by_volatility": pipeline.config["conformal"].get(
                    "normalize_by_volatility"
                ),
                "confidence_levels": str(metadata.get("confidence_levels")),
                "calibration_size": pipeline.config["pipeline"].get("calibration_size"),
                **{
                    f"xgb_{k}": v for k, v in pipeline.config["models"]["xgboost"]["params"].items()
                },
            }
        )
        # Dataset version: the span and size the models were actually fitted on.
        mlflow.set_tags(
            {
                "dataset_first_date": metadata.get("data_first_date"),
                "dataset_last_date": metadata.get("data_last_date"),
                "dataset_n_observations": metadata.get("n_observations"),
                "trained_at": metadata.get("trained_at"),
                "rate_type": pipeline.config["targets"][0].get("rate_type"),
            }
        )

        if point_metrics is not None and not point_metrics.empty:
            for _, row in point_metrics.iterrows():
                suffix = f"{row['target']}_h{int(row['horizon'])}_{row['model']}"
                for metric in ("mae", "rmse", "mape", "theil_u2", "mae_skill_vs_naive_pct"):
                    value = row.get(metric)
                    if value is not None and np.isfinite(value):
                        mlflow.log_metric(f"{metric}__{suffix}", float(value))

        if interval_metrics is not None and not interval_metrics.empty:
            for _, row in interval_metrics.iterrows():
                level = level_tag(row["nominal_coverage"])
                suffix = f"{row['target']}_h{int(row['horizon'])}_{row['model']}_{level}"
                for metric in ("empirical_coverage", "mean_interval_width", "winkler_score"):
                    value = row.get(metric)
                    if value is not None and np.isfinite(value):
                        mlflow.log_metric(f"{metric}__{suffix}", float(value))

        for artifact in artifacts or []:
            path = Path(artifact)
            if path.exists():
                mlflow.log_artifact(str(path))

        logger.info("mlflow run %s logged", run.info.run_id)
        return run.info.run_id
