#!/usr/bin/env python
"""Phase 4 entry point: exploratory analysis figures and a written report.

Everything the notebooks show is produced here too, so the findings can be
regenerated in CI without running a notebook.

    python scripts/run_eda.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mru_forecast import viz
from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.cleaning import load_processed
from mru_forecast.evaluation import eda
from mru_forecast.utils.logging import get_logger

logger = get_logger("run_eda")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    frame = load_processed(config)
    targets = [t["name"] for t in config["targets"]]
    figures_dir: Path = resolve_path("figures", config)
    reports_dir: Path = resolve_path("reports", config)

    returns = eda.compute_returns(frame, "log")
    findings: dict[str, object] = {}

    # --- 1. levels & trend ------------------------------------------------
    levels = eda.describe_levels(frame[targets])
    print("\n=== 1. Levels ===")
    print(
        levels[
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
                "flat_day_share",
                "total_change_pct",
                "annualised_drift_pct",
            ]
        ]
        .round(4)
        .to_string()
    )
    findings["levels"] = json.loads(
        levels.drop(columns=["first_date", "last_date"]).round(6).to_json(orient="index")
    )

    # --- 2. stationarity --------------------------------------------------
    print("\n=== 2. Stationarity ===")
    stationarity = []
    for name in targets:
        for label, series in (("level", frame[name]), ("log-return", returns[name])):
            table = eda.stationarity_tests(series)
            table.insert(0, "series", name)
            table.insert(1, "transform", label)
            stationarity.append(table)
    stationarity_frame = pd.concat(stationarity, ignore_index=True)
    print(
        stationarity_frame[
            ["series", "transform", "test", "statistic", "p_value", "interpretation"]
        ]
        .round(4)
        .to_string(index=False)
    )
    findings["stationarity"] = stationarity_frame.drop(columns=["null_hypothesis"]).to_dict(
        orient="records"
    )

    # --- 3. volatility ----------------------------------------------------
    window = int(config["evaluation"]["volatility_regimes"]["window"])
    volatility = eda.rolling_volatility(returns[targets], window)
    print(f"\n=== 3. Annualised volatility ({window}-day rolling, %) ===")
    print(volatility.describe().round(3).to_string())
    findings["volatility"] = json.loads(volatility.describe().round(4).to_json())

    # --- 4. seasonality ---------------------------------------------------
    print("\n=== 4. Seasonality ===")
    seasonality: dict[str, pd.DataFrame] = {}
    for by in ("day_of_week", "month"):
        table = eda.seasonality_table(returns[targets[0]], by)
        seasonality[by] = table
        print(f"\n{targets[0]} by {by}:")
        print(table.round(4).to_string())
    strongest = seasonality["day_of_week"]["t_stat"].abs().max()
    print(
        f"\nLargest |t| across the 5 weekdays: {strongest:.2f}. "
        "With 5 simultaneous comparisons, |t| around 2 is what random data produces; "
        "nothing here supports a tradeable day-of-week effect."
    )
    findings["seasonality_max_abs_t"] = float(strongest)

    # --- 5. correlation ---------------------------------------------------
    print("\n=== 5. Correlation of daily log-returns ===")
    correlation = eda.correlation_matrix(returns)
    print(correlation.round(3).to_string())
    findings["correlation"] = json.loads(correlation.round(4).to_json())

    print("\n=== 5b. Lagged correlation with USD_MRU returns ===")
    others = [c for c in returns.columns if c != "USD_MRU"]
    lagged = eda.lagged_correlation(returns, "USD_MRU", others, max_lag=5)
    print(lagged.round(3).to_string())

    # --- 6. autocorrelation ----------------------------------------------
    print("\n=== 6. Autocorrelation of USD_MRU log-returns ===")
    acf_table = eda.autocorrelation(returns["USD_MRU"], nlags=30)
    significant = acf_table.iloc[1:][
        acf_table.iloc[1:]["acf"].abs() > acf_table["conf_upper"].iloc[0]
    ]
    print(acf_table.head(11).round(4).to_string())
    print(f"\nLags with |ACF| beyond the 95% band (of 30): {list(significant.index)}")
    findings["significant_acf_lags"] = [int(i) for i in significant.index]

    # --- 7. step events ---------------------------------------------------
    print("\n=== 7. Discrete step moves (>= 1%) ===")
    steps = eda.detect_step_events(frame[targets[0]], 1.0)
    years = (frame.index[-1] - frame.index[0]).days / 365.25
    print(f"{len(steps)} events over {years:.1f} years ({len(steps) / years:.1f} per year)")
    print(steps.tail(8).round(3).to_string(index=False))
    findings["step_events_per_year"] = round(len(steps) / years, 2)

    # --- 8. regimes -------------------------------------------------------
    print("\n=== 8. Trailing volatility regimes ===")
    regimes = eda.volatility_regimes(
        frame[targets[0]],
        window,
        tuple(config["evaluation"]["volatility_regimes"]["quantiles"]),
        tuple(config["evaluation"]["volatility_regimes"]["labels"]),
    )
    regime_summary = (
        pd.DataFrame({"regime": regimes, "abs_return_pct": returns[targets[0]].abs() * 100})
        .dropna()
        .groupby("regime")["abs_return_pct"]
        .agg(["count", "mean", "std", "max"])
    )
    print(regime_summary.round(4).to_string())
    findings["regimes"] = json.loads(regime_summary.round(4).to_json())

    # --- figures ----------------------------------------------------------
    if not args.no_figures:
        charts = {
            "01_levels": viz.line_chart(
                frame[targets],
                title="USD/MRU and EUR/MRU, daily",
                y_title="MRU per unit of foreign currency",
            ),
            "02_levels_indexed": viz.line_chart(
                frame[targets].div(frame[targets].iloc[0]).mul(100),
                title="Indexed to 100 at the first observation",
                y_title="index (first day = 100)",
                hover_format=".1f",
            ),
            "03_volatility": viz.volatility_chart(volatility, window=window),
            "04_returns_hist": viz.returns_histogram(returns[targets]),
            "05_seasonality_dow": viz.seasonality_chart(
                seasonality["day_of_week"],
                title=f"{targets[0]}: mean daily log-return by weekday (+/- 1 s.e.)",
            ),
            "06_seasonality_month": viz.seasonality_chart(
                seasonality["month"],
                title=f"{targets[0]}: mean daily log-return by month (+/- 1 s.e.)",
            ),
            "07_correlation": viz.correlation_heatmap(correlation),
            "08_acf": viz.acf_chart(acf_table, column="acf"),
            "09_pacf": viz.acf_chart(acf_table.dropna(subset=["pacf"]), column="pacf"),
            "10_auxiliary": viz.line_chart(
                frame[[c for c in ("BRENT_OIL", "GOLD") if c in frame]]
                .div(frame[[c for c in ("BRENT_OIL", "GOLD") if c in frame]].iloc[0])
                .mul(100),
                title="Auxiliary series, indexed to 100",
                y_title="index",
                hover_format=".1f",
            ),
        }
        for name, figure in charts.items():
            viz.save_figure(figure, figures_dir / f"{name}.html")
        print(f"\nWrote {len(charts)} figures -> {figures_dir}")

    (reports_dir / "eda_findings.json").write_text(
        json.dumps(findings, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote report -> {reports_dir / 'eda_findings.json'}")

    # --- headline conclusions --------------------------------------------
    flat = float(levels.loc[targets[0], "flat_day_share"])
    drift = float(levels.loc[targets[0], "annualised_drift_pct"])
    print("\n=== What this means for modelling ===")
    print(f"- {targets[0]} does not move at all on {flat:.0%} of business days: this is a")
    print("  managed rate that steps, not a freely floating one. A random walk is therefore")
    print("  a genuinely strong baseline and must be beaten, not assumed away.")
    print(f"- Levels are non-stationary, log-returns are stationary; drift is ~{drift:.2f}%/year.")
    print("- Almost no linear autocorrelation survives in the returns, so the value of any")
    print("  model here is mostly in sizing uncertainty rather than in predicting direction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
