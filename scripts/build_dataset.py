#!/usr/bin/env python
"""Phase 3 entry point: clean and validate the ingested data.

python scripts/build_dataset.py
python scripts/build_dataset.py --strict     # non-zero exit on ERROR issues
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.cleaning import build_processed_dataset
from mru_forecast.utils.logging import get_logger

logger = get_logger("build_dataset")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail on validation errors")
    parser.add_argument("--no-persist", action="store_true", help="do not write any file")
    args = parser.parse_args(argv)

    config = load_config()
    frame, report, metadata = build_processed_dataset(config, persist=not args.no_persist)

    print("\n=== Processed dataset ===")
    print(f"rows       : {metadata['n_rows']}")
    print(f"period     : {metadata['first_date']} -> {metadata['last_date']}")
    print(f"calendar   : {metadata['calendar']}")
    print(f"columns    : {', '.join(metadata['columns'])}")
    if metadata["dropped_low_coverage"]:
        print(f"dropped    : {', '.join(metadata['dropped_low_coverage'])} (coverage too low)")

    print("\n=== Cleaning actions per series ===")
    print(
        pd.DataFrame(metadata["per_series"]).drop(columns=["flagged_dates"]).to_string(index=False)
    )

    print("\n=== Validation ===")
    issues = report.to_frame()
    if issues.empty:
        print("no issues")
    else:
        print(issues[["severity", "series", "code", "count", "message"]].to_string(index=False))
    print(f"\n{report.summary()}")

    print("\n=== Head / tail ===")
    print(pd.concat([frame.head(3), frame.tail(3)]).round(4).to_string())

    if not args.no_persist:
        print(f"\nWrote: {resolve_path('data_processed', config)}")

    if args.strict and not report.ok:
        logger.error("strict mode: dataset has validation errors")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
