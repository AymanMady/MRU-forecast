#!/usr/bin/env python
"""Produce a forecast from the trained artefacts.

python scripts/predict.py
python scripts/predict.py --target USD_MRU --horizon 7 --confidence 0.90
python scripts/predict.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

from mru_forecast.config import load_config
from mru_forecast.data.cleaning import load_processed
from mru_forecast.models.conformal import level_tag
from mru_forecast.pipeline.forecasting_pipeline import ForecastingPipeline

warnings.filterwarnings("ignore", category=FutureWarning)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=None, help="USD_MRU or EUR_MRU (default: both)")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    config = load_config()
    frame = load_processed(config)
    pipeline = ForecastingPipeline.load(config=config)

    targets = [args.target] if args.target else sorted({t for (t, _) in pipeline.steps})
    level = level_tag(args.confidence or config["conformal"]["primary_level"])

    payload = []
    for target in targets:
        path = pipeline.forecast(
            frame,
            target=target,
            horizon=args.horizon,
            confidence=args.confidence,
        )
        payload.append(
            {
                "currency_pair": target.replace("_", "/"),
                "rate_definition": next(
                    (t["definition"] for t in config["targets"] if t["name"] == target), ""
                ),
                "rate_type": next(
                    (t["rate_type"] for t in config["targets"] if t["name"] == target), ""
                ),
                "anchor_date": str(path["anchor_date"].iloc[0].date()),
                "anchor_value": round(float(path["anchor_value"].iloc[0]), 4),
                "horizon": args.horizon,
                "confidence": int(level) / 100.0,
                "forecast": [
                    {
                        "date": str(date.date()),
                        "days_ahead": int(row["days_ahead"]),
                        "prediction": round(float(row["prediction"]), 4),
                        "lower": round(float(row[f"lower_{level}"]), 4),
                        "upper": round(float(row[f"upper_{level}"]), 4),
                    }
                    for date, row in path.iterrows()
                ],
            }
        )

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for block in payload:
        print(f"\n{block['currency_pair']}  ({block['rate_definition']}, {block['rate_type']})")
        print(
            f"anchor: {block['anchor_value']} on {block['anchor_date']}   "
            f"interval: {block['confidence']:.0%}"
        )
        print(
            f"\n{'Date':<12} {'Ahead':>6} {'Prediction':>12} "
            f"{'Lower':>10} {'Upper':>10} {'Width':>8}"
        )
        print("-" * 62)
        for row in block["forecast"]:
            print(
                f"{row['date']:<12} {row['days_ahead']:>6} {row['prediction']:>12.4f} "
                f"{row['lower']:>10.4f} {row['upper']:>10.4f} "
                f"{row['upper'] - row['lower']:>8.4f}"
            )
    print(
        "\nForecasts are experimental and derived from indicative market rates, "
        "not the official BCM rate. Do not use them as a basis for financial decisions."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
