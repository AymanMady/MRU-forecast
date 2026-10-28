#!/usr/bin/env python
"""Phase 2 entry point: fetch every configured series into ``data/interim/``.

python scripts/download_data.py                 # full history from config
python scripts/download_data.py --no-cross-check # skip the second provider
python scripts/download_data.py --start 2018-01-01
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.ingestion import (
    CROSSCHECK_FILENAME,
    compare_with_cross_check,
    ingest,
    to_wide,
)
from mru_forecast.data.sources import IngestionError
from mru_forecast.utils.logging import get_logger

logger = get_logger("download_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Override ingestion.start_date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override ingestion.end_date (YYYY-MM-DD)")
    parser.add_argument(
        "--no-cross-check",
        action="store_true",
        help="Skip the independent second-provider verification",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    if args.start:
        config["ingestion"]["start_date"] = args.start
    if args.end:
        config["ingestion"]["end_date"] = args.end

    try:
        tidy, manifest = ingest(config, cross_check=not args.no_cross_check)
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1

    wide = to_wide(tidy)
    print("\n=== Ingested series ===")
    summary = pd.DataFrame(
        [
            {
                "series": name,
                "rows": info["rows"],
                "first": info["first_date"],
                "last": info["last_date"],
                "rate_type": info["rate_type"],
                "role": info["role"],
                "derived": bool(info["derived_from"]),
            }
            for name, info in manifest["series"].items()
        ]
    )
    print(summary.to_string(index=False))

    if manifest["failures"]:
        print("\n=== Failures (auxiliary series are optional) ===")
        for name, reason in manifest["failures"].items():
            print(f"  {name}: {reason}")

    # Independent verification of the level, printed so it cannot be ignored.
    cc_path = resolve_path("data_interim", config) / CROSSCHECK_FILENAME
    if cc_path.exists() and not args.no_cross_check:
        cross = pd.read_csv(cc_path, index_col="date", parse_dates=["date"])
        gap = compare_with_cross_check(
            wide, cross, tolerance_pct=config["ingestion"]["cross_check"]["tolerance_pct"]
        )
        if not gap.empty:
            print("\n=== Cross-check vs currency-api (relative gap, %) ===")
            print(gap.round(3).to_string())
            ok = int(gap["within_tolerance"].sum())
            print(f"\nWithin tolerance: {ok}/{len(gap)} sampled dates")

    print(f"\nWrote: {resolve_path('data_interim', config)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
