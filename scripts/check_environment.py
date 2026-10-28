#!/usr/bin/env python
"""Smoke check: verify the environment and report which pipeline stages are built.

Run it after creating the virtual environment:

    python scripts/check_environment.py

It imports every critical dependency, loads the configuration, and prints a
report.  Exits with a non-zero status if anything is missing, so it can be used
as a CI gate.
"""

from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("prophet", "prophet"),
    ("statsmodels", "statsmodels"),
    ("mapie", "mapie"),
    ("plotly", "plotly"),
    ("streamlit", "streamlit"),
    ("fastapi", "fastapi"),
    ("mlflow", "mlflow"),
    ("yaml", "pyyaml"),
    ("dotenv", "python-dotenv"),
    ("pydantic", "pydantic"),
]


def _version(module) -> str:
    return getattr(module, "__version__", "unknown")


def main() -> int:
    print("=" * 62)
    print("MRU-Forecast — environment check")
    print("=" * 62)
    print(f"Python      : {platform.python_version()} ({sys.executable})")

    failures: list[str] = []
    print("\nDependencies")
    print("-" * 62)
    for import_name, dist_name in REQUIRED:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:
            print(f"  [FAIL] {dist_name:<16} {type(exc).__name__}: {exc}")
            failures.append(dist_name)
        else:
            print(f"  [ ok ] {dist_name:<16} {_version(module)}")

    print("\nProject package")
    print("-" * 62)
    try:
        from mru_forecast import PROJECT_ROOT, __version__, get_settings, load_config
        from mru_forecast.config import resolve_path

        settings = get_settings()
        config = load_config()
        targets = [t["name"] for t in config["targets"]]

        print(f"  [ ok ] mru_forecast     {__version__}")
        print(f"  [ ok ] project root     {PROJECT_ROOT}")
        print(f"  [ ok ] config loaded    {len(config)} sections")
        print(f"  [ ok ] targets          {', '.join(targets)}")
        print(f"  [ ok ] horizons         {config['horizons']}")
        print(f"  [ ok ] conformal levels {config['conformal']['confidence_levels']}")
        print(f"  [ ok ] data/processed   {resolve_path('data_processed')}")
        print(f"  [ ok ] environment      {settings.mru_env}")
        providers = settings.available_providers()
        print(
            f"  [ .. ] API keys set     "
            f"{providers if providers else 'none (not required: every source used is keyless)'}"
        )
    except Exception as exc:
        print(f"  [FAIL] project package  {type(exc).__name__}: {exc}")
        failures.append("mru_forecast")
        config = None

    # --- which pipeline stages have produced artefacts ---------------------
    if config is not None:
        from mru_forecast.config import resolve_path

        print("\nPipeline artefacts")
        print("-" * 62)
        stages = [
            (
                "ingested data",
                resolve_path("data_interim") / "rates_interim.parquet",
                "python scripts/download_data.py",
            ),
            (
                "processed dataset",
                resolve_path("data_processed") / "rates_processed.parquet",
                "python scripts/build_dataset.py",
            ),
            (
                "trained predictors",
                resolve_path("models") / "predictors.pkl",
                "python scripts/train.py",
            ),
            (
                "backtest results",
                resolve_path("reports") / "metrics_point.csv",
                "python scripts/backtest.py",
            ),
        ]
        for label, path, command in stages:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                print(f"  [ ok ] {label:<18} {size_kb:>9,.0f} KB")
            else:
                print(f"  [ -- ] {label:<18} not built yet — run: {command}")

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED — missing or broken: {', '.join(failures)}")
        print("Fix with: pip install -e '.[dev]'")
        return 1
    print("ALL CHECKS PASSED — environment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
