#!/usr/bin/env python
"""Phases 9-11 entry point: expanding-window backtest with conformal intervals.

python scripts/backtest.py                 # every model, every horizon
python scripts/backtest.py --no-prophet    # fast: skips the slow model
python scripts/backtest.py --horizons 1 7  --targets USD_MRU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import pandas as pd

from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.cleaning import load_processed
from mru_forecast.evaluation import uncertainty
from mru_forecast.evaluation.backtesting import (
    default_model_factories,
    make_splits,
    run_backtest,
    splits_summary,
)
from mru_forecast.evaluation.metrics import metrics_table
from mru_forecast.utils.logging import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
logger = get_logger("backtest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--no-prophet", action="store_true", help="skip Prophet (much faster)")
    parser.add_argument("--splits", type=int, default=None, help="override n_splits")
    parser.add_argument("--level", type=int, default=90, help="headline interval level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    if args.splits:
        config["backtesting"]["n_splits"] = args.splits

    frame = load_processed(config)
    targets = args.targets or [t["name"] for t in config["targets"]]
    horizons = args.horizons or list(config["horizons"])
    factories = default_model_factories(config, include_prophet=not args.no_prophet)

    splits = make_splits(pd.DatetimeIndex(frame.index), config, max_horizon=max(horizons))
    print("\n=== Backtest design (expanding window, chronological) ===")
    print(splits_summary(splits).to_string(index=False))
    print(
        f"\nmodels    : {', '.join(factories)}"
        f"\ntargets   : {', '.join(targets)}"
        f"\nhorizons  : {horizons}"
        f"\ncells     : {len(splits) * len(targets) * len(horizons) * len(factories)}"
    )

    started = time.perf_counter()
    results = run_backtest(
        frame,
        config,
        factories,
        targets=targets,
        horizons=horizons,
        conformal=True,
        splits=splits,
    )
    elapsed = time.perf_counter() - started
    print(f"\nCompleted in {elapsed:.0f}s — {len(results)} predictions")

    reports_dir = resolve_path("reports", config)
    results.to_parquet(reports_dir)

    predictions = results.predictions
    level = args.level

    # --- point forecast accuracy -----------------------------------------
    point = metrics_table(
        predictions, group_by=["target", "horizon", "model"], naive_column="naive"
    )
    print(
        f"\n=== Point-forecast accuracy (out-of-sample, {predictions['origin'].nunique()} origins) ==="
    )
    print(
        point[
            [
                "target",
                "horizon",
                "model",
                "mae",
                "rmse",
                "mape",
                "directional_accuracy",
                "theil_u2",
                "mae_skill_vs_naive_pct",
                "n",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    # --- interval quality -------------------------------------------------
    intervals = uncertainty.interval_metrics_table(
        predictions, group_by=["target", "horizon", "model"], levels=[level]
    )
    print(f"\n=== {level}% prediction intervals ===")
    print(
        intervals[
            [
                "target",
                "horizon",
                "model",
                "empirical_coverage",
                "coverage_error_pp",
                "coverage_se_pp",
                "mean_interval_width",
                "relative_width_pct",
                "winkler_score",
                "n",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    # --- calibration across levels ---------------------------------------
    calibration = uncertainty.calibration_table(predictions, group_by=["model"])
    print("\n=== Calibration: nominal vs empirical coverage (all targets & horizons pooled) ===")
    print(calibration.round(4).to_string(index=False))

    # --- CRPS -------------------------------------------------------------
    crps = uncertainty.crps_table(predictions, group_by=["target", "horizon", "model"])
    print("\n=== Approximate CRPS (lower is better) ===")
    print(crps.round(5).to_string(index=False))

    # --- market conditions -----------------------------------------------
    tagged = uncertainty.assign_volatility_regime(predictions, frame, config)
    regimes = uncertainty.regime_performance(
        tagged, group_by=["target", "horizon", "model"], level=level
    )
    print(f"\n=== Performance by trailing volatility regime ({level}% intervals) ===")
    print(regimes.round(4).to_string(index=False))

    # --- persist ----------------------------------------------------------
    point.to_csv(reports_dir / "metrics_point.csv", index=False)
    uncertainty.interval_metrics_table(predictions, group_by=["target", "horizon", "model"]).to_csv(
        reports_dir / "metrics_interval.csv", index=False
    )
    calibration.to_csv(reports_dir / "metrics_calibration.csv", index=False)
    crps.to_csv(reports_dir / "metrics_crps.csv", index=False)
    regimes.to_csv(reports_dir / "metrics_regime.csv", index=False)
    tagged.to_parquet(reports_dir / "backtest_predictions_tagged.parquet", index=False)

    (reports_dir / "backtest_meta.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 1),
                "n_predictions": len(results),
                "models": list(factories),
                "targets": targets,
                "horizons": horizons,
                "n_splits": len(splits),
                "test_period": [
                    str(predictions["origin"].min().date()),
                    str(predictions["target_date"].max().date()),
                ],
                "confidence_levels": config["conformal"]["confidence_levels"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote results -> {reports_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
