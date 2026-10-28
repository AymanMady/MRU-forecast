#!/usr/bin/env python
"""Phases 10 & 14 entry point: train the production predictors and log the run.

python scripts/train.py
python scripts/train.py --model xgboost --max-horizon 30
python scripts/train.py --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import pandas as pd

from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.cleaning import load_processed
from mru_forecast.pipeline.forecasting_pipeline import ForecastingPipeline, log_to_mlflow
from mru_forecast.utils.logging import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
logger = get_logger("train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="production point forecaster")
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    if args.max_horizon:
        config["pipeline"]["max_horizon"] = args.max_horizon
    if args.model:
        config["pipeline"]["model"] = args.model

    frame = load_processed(config)
    print(
        f"Training on {len(frame)} business days: "
        f"{frame.index.min().date()} -> {frame.index.max().date()}"
    )

    started = time.perf_counter()
    pipeline = ForecastingPipeline(config=config).train(frame, targets=args.targets)
    elapsed = time.perf_counter() - started
    print(f"Trained {len(pipeline.steps)} predictors in {elapsed:.0f}s")

    if pipeline.metadata.get("failures"):
        print("\nFailed steps:")
        for key, reason in pipeline.metadata["failures"].items():
            print(f"  {key}: {reason}")

    artifacts_dir = pipeline.save()

    print("\n=== Conformal quantiles (normalised units; multiply by the local scale) ===")
    summary = pipeline.calibration_summary()
    headline = summary[summary["horizon"].isin(config["horizons"])]
    print(headline.round(4).to_string(index=False))

    print("\n=== Forecast path preview ===")
    for target in sorted({t for (t, _) in pipeline.steps}):
        path = pipeline.forecast(frame, target=target, horizon=7)
        display = path[["days_ahead", "prediction", "lower_90", "upper_90"]].copy()
        display["width"] = display["upper_90"] - display["lower_90"]
        print(
            f"\n{target} (anchor {path['anchor_value'].iloc[0]:.4f} "
            f"on {path['anchor_date'].iloc[0].date()}):"
        )
        print(display.round(4).to_string())

    # Persist the full path so the API and dashboard can serve without retraining.
    reports_dir = resolve_path("reports", config)
    full_path = pipeline.forecast_all(frame)
    full_path.to_parquet(reports_dir / "latest_forecast.parquet", index=False)
    print(f"\nWrote forecast -> {reports_dir / 'latest_forecast.parquet'}")

    if not args.no_mlflow:
        point = interval = None
        point_path = reports_dir / "metrics_point.csv"
        interval_path = reports_dir / "metrics_interval.csv"
        if point_path.exists():
            point = pd.read_csv(point_path)
        if interval_path.exists():
            interval = pd.read_csv(interval_path)
        run_id = log_to_mlflow(
            pipeline,
            point_metrics=point,
            interval_metrics=interval,
            artifacts=[
                artifacts_dir / "pipeline_metadata.json",
                reports_dir / "latest_forecast.parquet",
            ],
        )
        print(f"MLflow run: {run_id}" if run_id else "MLflow: skipped")

    print(f"\nArtefacts -> {artifacts_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
