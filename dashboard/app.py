"""MRU-Forecast dashboard (Phase 12).

    streamlit run dashboard/app.py

Five pages: an overview, one per currency pair, a model comparison and a page
dedicated to how well the uncertainty is actually calibrated.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# `streamlit run dashboard/app.py` sets sys.path[0] to *this file's directory*,
# not the repository root, so `dashboard.components` is not importable without
# help. Adding the root keeps the package-qualified import working both when
# Streamlit runs this file and when a test imports it as `dashboard.app`.
# This block must stay above the first `dashboard.*` import.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import (  # noqa: E402
    PAIR_LABEL,
    forecast_table,
    get_config,
    get_forecast,
    get_frame,
    get_pipeline,
    get_report,
    is_dark,
    metric_row,
    page_header,
    render_disclaimer,
    require_artifacts,
    require_backtest,
)
from mru_forecast import viz  # noqa: E402
from mru_forecast.evaluation import eda, uncertainty  # noqa: E402
from mru_forecast.evaluation.metrics import metrics_table  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="MRU-Forecast",
    page_icon="MRU",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def page_overview() -> None:
    page_header(
        "MRU-Forecast",
        "Exchange-rate forecasting for the Mauritanian Ouguiya, with calibrated "
        "prediction intervals.",
    )
    config = get_config()
    frame = get_frame()
    targets = [t["name"] for t in config["targets"]]

    latest = frame[targets].iloc[-1]
    previous = frame[targets].iloc[-2]
    metric_row(
        [
            (
                PAIR_LABEL.get(name, name),
                f"{latest[name]:.3f}",
                f"change vs previous business day: {(latest[name] / previous[name] - 1) * 100:+.3f}%",
            )
            for name in targets
        ]
        + [
            ("As of", str(frame.index.max().date()), "Last business day with an observation"),
            ("History", f"{len(frame)} days", f"from {frame.index.min().date()}"),
        ]
    )

    st.divider()

    if not require_artifacts():
        render_disclaimer()
        return

    level = int(config["conformal"]["primary_level"] * 100)
    horizon = st.select_slider(
        "Forecast horizon (business days)", options=[1, 5, 7, 10, 14, 21, 30], value=7
    )

    columns = st.columns(len(targets))
    for column, target in zip(columns, targets, strict=True):
        path = get_forecast(target, horizon)
        if path is None:
            continue
        final = path.iloc[-1]
        with column:
            st.subheader(PAIR_LABEL.get(target, target))
            st.metric(
                f"Forecast in {horizon} business days",
                f"{final['prediction']:.3f}",
                f"{(final['prediction'] / latest[target] - 1) * 100:+.2f}% vs today",
            )
            st.caption(
                f"{level}% interval: **{final[f'lower_{level}']:.3f} — "
                f"{final[f'upper_{level}']:.3f}** "
                f"(width {final[f'upper_{level}'] - final[f'lower_{level}']:.3f} MRU)"
            )

    st.divider()
    st.subheader("Backtested performance")
    point = get_report("metrics_point.csv")
    intervals = get_report("metrics_interval.csv")
    if point is None or intervals is None:
        require_backtest()
    else:
        production = point[point["model"] == config["pipeline"]["model"]]
        production = production[production["horizon"] == horizon]
        if production.empty:
            production = point[
                (point["model"] == config["pipeline"]["model"])
                & (
                    point["horizon"]
                    == min(point["horizon"].unique(), key=lambda h: abs(h - horizon))
                )
            ]
        interval_rows = intervals[
            (intervals["model"] == config["pipeline"]["model"])
            & (np.isclose(intervals["nominal_coverage"], level / 100.0))
        ]
        if not production.empty:
            metric_row(
                [
                    (
                        "MAE",
                        f"{production['mae'].mean():.4f} MRU",
                        "Mean absolute error, out of sample",
                    ),
                    ("RMSE", f"{production['rmse'].mean():.4f} MRU", "Root mean squared error"),
                    (
                        f"{level}% coverage",
                        f"{interval_rows['empirical_coverage'].mean():.1%}"
                        if not interval_rows.empty
                        else "n/a",
                        f"Target {level}%. Binomial s.e. is around "
                        f"{interval_rows['coverage_se_pp'].mean():.1f}pp."
                        if not interval_rows.empty
                        else None,
                    ),
                    (
                        "Mean interval width",
                        f"{interval_rows['mean_interval_width'].mean():.3f} MRU"
                        if not interval_rows.empty
                        else "n/a",
                        "Average distance between the bounds",
                    ),
                ]
            )
            st.caption(
                "Averaged over both pairs at the nearest backtested horizon. "
                "See **Model comparison** for the full table."
            )

    st.divider()
    st.plotly_chart(
        viz.line_chart(
            frame[targets],
            title="Observed history",
            y_title="MRU per unit of foreign currency",
            dark=is_dark(),
            height=400,
        ),
        width="stretch",
    )
    render_disclaimer()


# ---------------------------------------------------------------------------
# Per-pair page
# ---------------------------------------------------------------------------
def render_pair(target: str) -> None:
    config = get_config()
    label = PAIR_LABEL.get(target, target)
    info = next(t for t in config["targets"] if t["name"] == target)
    page_header(
        label,
        f"{info['definition']} — {info['rate_type']}"
        + ("  ·  derived cross, not an observed quote" if not info.get("observed", True) else ""),
    )

    frame = get_frame()
    if not require_artifacts():
        return

    left, right = st.columns([2, 1])
    horizon = left.select_slider(
        "Horizon (business days)", options=[1, 5, 7, 10, 14, 21, 30], value=7, key=f"h_{target}"
    )
    level = int(
        right.selectbox(
            "Interval level",
            options=[int(c * 100) for c in config["conformal"]["confidence_levels"]],
            index=[int(c * 100) for c in config["conformal"]["confidence_levels"]].index(
                int(config["conformal"]["primary_level"] * 100)
            ),
            key=f"l_{target}",
        )
    )
    window = st.slider("History shown (business days)", 60, len(frame), 250, key=f"w_{target}")

    path = get_forecast(target, horizon)
    if path is None:
        return

    levels = [int(c * 100) for c in config["conformal"]["confidence_levels"]]
    bands = [(f"lower_{lv}", f"upper_{lv}", f"{lv}%") for lv in levels if lv <= level]

    st.plotly_chart(
        viz.forecast_chart(
            frame[target],
            path,
            title=f"{label}: observed history and {horizon}-business-day forecast",
            y_title="MRU",
            dark=is_dark(),
            history_window=window,
            interval_columns=bands,
            height=480,
        ),
        width="stretch",
    )

    table_column, stats_column = st.columns([3, 2])
    with table_column:
        st.subheader("Forecast path")
        st.dataframe(forecast_table(path, level), width="stretch")
    with stats_column:
        st.subheader("Recent behaviour")
        returns = eda.compute_returns(frame[[target]], "log")[target]
        recent = returns.iloc[-250:]
        st.metric(
            "Annualised volatility (30d)", f"{eda.rolling_volatility(returns, 30).iloc[-1]:.2f}%"
        )
        st.metric(
            "Days with no change (last 250)",
            f"{(recent.abs() < 1e-9).mean():.0%}",
            help="A managed rate spends much of its time perfectly flat.",
        )
        steps = eda.detect_step_events(frame[target], 1.0)
        st.metric(
            "Moves of 1% or more (last 12 months)",
            int((steps["date"] >= frame.index.max() - pd.Timedelta(days=365)).sum())
            if not steps.empty
            else 0,
        )

    with st.expander("Volatility over time"):
        st.plotly_chart(
            viz.volatility_chart(
                eda.rolling_volatility(eda.compute_returns(frame[[target]], "log"), 30),
                dark=is_dark(),
            ),
            width="stretch",
        )

    pipeline = get_pipeline()
    if pipeline and pipeline.metadata.get("model") == "xgboost":
        with st.expander("What the model is using"):
            try:
                importance = pipeline.feature_importance(target, horizon, 15)
                st.plotly_chart(
                    viz.metric_bar_chart(
                        importance,
                        "gain",
                        group_column="feature",
                        title=f"Top features, {label} at h={horizon}",
                        dark=is_dark(),
                        value_format=".3f",
                        lower_is_better=False,
                        height=460,
                    ),
                    width="stretch",
                )
            except (KeyError, TypeError) as exc:
                st.info(f"Feature importance unavailable: {exc}")

    render_disclaimer()


def page_usd() -> None:
    render_pair("USD_MRU")


def page_eur() -> None:
    render_pair("EUR_MRU")


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def page_models() -> None:
    page_header(
        "Model comparison",
        "Expanding-window backtest on held-out data. Every model is trained, "
        "calibrated and scored on exactly the same splits.",
    )
    predictions = require_backtest()
    if predictions is None:
        return
    config = get_config()

    left, middle, right = st.columns(3)
    target = left.selectbox(
        "Pair", sorted(predictions["target"].unique()), format_func=lambda t: PAIR_LABEL.get(t, t)
    )
    horizon = middle.selectbox("Horizon (business days)", sorted(predictions["horizon"].unique()))
    level = int(
        right.selectbox(
            "Interval level",
            [int(c * 100) for c in config["conformal"]["confidence_levels"]],
            index=1,
        )
    )

    subset = predictions[(predictions["target"] == target) & (predictions["horizon"] == horizon)]
    point = metrics_table(subset, group_by=["model"], naive_column="naive")
    intervals = uncertainty.interval_metrics_table(subset, group_by=["model"], levels=[level])
    merged = point.merge(intervals, on="model", suffixes=("", "_iv"))

    display = (
        merged[
            [
                "model",
                "mae",
                "rmse",
                "mape",
                "theil_u2",
                "mae_skill_vs_naive_pct",
                "empirical_coverage",
                "mean_interval_width",
                "winkler_score",
            ]
        ]
        .rename(
            columns={
                "model": "Model",
                "mae": "MAE",
                "rmse": "RMSE",
                "mape": "MAPE %",
                "theil_u2": "Theil U2",
                "mae_skill_vs_naive_pct": "MAE skill vs naive %",
                "empirical_coverage": f"Coverage {level}%",
                "mean_interval_width": "Interval width",
                "winkler_score": "Winkler",
            }
        )
        .sort_values("MAE")
    )

    st.dataframe(
        display.style.format(
            {
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "MAPE %": "{:.3f}",
                "Theil U2": "{:.3f}",
                "MAE skill vs naive %": "{:+.1f}",
                f"Coverage {level}%": "{:.1%}",
                "Interval width": "{:.4f}",
                "Winkler": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "**Theil U2** is the model's RMSE divided by the random walk's. Below 1 the "
        "model beats a random walk; at or above 1 it does not, and on a managed "
        "currency that is a real possibility rather than a formality."
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            viz.metric_bar_chart(merged, "mae", title="MAE (MRU, lower is better)", dark=is_dark()),
            width="stretch",
        )
    with chart_right:
        st.plotly_chart(
            viz.metric_bar_chart(
                merged,
                "winkler_score",
                title=f"Winkler score at {level}% (lower is better)",
                dark=is_dark(),
            ),
            width="stretch",
        )

    st.divider()
    st.subheader("Performance by market condition")
    st.caption(
        "Origins are grouped by the trailing 30-day realised volatility in force "
        "on that date. This is the answer to *does the model hold up when the "
        "market gets turbulent?*"
    )
    if "regime" in subset.columns:
        regimes = uncertainty.regime_performance(subset, group_by=["model"], level=level)
    else:
        tagged = uncertainty.assign_volatility_regime(subset, get_frame(), config)
        regimes = uncertainty.regime_performance(tagged, group_by=["model"], level=level)

    if not regimes.empty:
        metric = st.radio(
            "Metric",
            ["mae", "coverage", "mean_interval_width", "winkler_score"],
            horizontal=True,
            format_func=lambda m: m.replace("_", " ").title(),
        )
        st.plotly_chart(viz.regime_chart(regimes, metric, dark=is_dark()), width="stretch")
        st.dataframe(regimes.round(4), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Out-of-sample paths")
    models = st.multiselect(
        "Models to plot",
        sorted(subset["model"].unique()),
        default=[m for m in ("naive", "xgboost") if m in set(subset["model"])],
    )
    if models:
        pivot = subset[subset["model"].isin(models)].pivot_table(
            index="target_date", columns="model", values="prediction"
        )
        actual = subset.drop_duplicates("target_date").set_index("target_date")["actual"]
        best = (
            subset[subset["model"] == models[0]]
            .drop_duplicates("target_date")
            .set_index("target_date")
        )
        st.plotly_chart(
            viz.backtest_chart(
                actual.sort_index(),
                pivot.sort_index(),
                lower=best[f"lower_{level}"].sort_index() if f"lower_{level}" in best else None,
                upper=best[f"upper_{level}"].sort_index() if f"upper_{level}" in best else None,
                title=f"{PAIR_LABEL.get(target, target)} at h={horizon}: realised vs forecast "
                f"({models[0]} interval shown)",
                y_title="MRU",
                dark=is_dark(),
            ),
            width="stretch",
        )
    render_disclaimer()


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def page_uncertainty() -> None:
    page_header(
        "Uncertainty and calibration",
        "An interval is only useful if it covers the truth as often as it claims. "
        "This page measures that, rather than asserting it.",
    )
    predictions = require_backtest()
    if predictions is None:
        return
    config = get_config()

    st.markdown(
        """
**How the intervals are built.** Split conformal prediction:

1. the point model is fitted on a **training** block;
2. on a disjoint **calibration** block, non-conformity scores
   `s = |y - ŷ|` are computed and divided by a local volatility estimate;
3. the conformal quantile is the `ceil((n+1)·c)`-th smallest score — the `n+1`
   correction is what makes the coverage guarantee hold in finite samples;
4. a new forecast gets the interval `ŷ ± q · scale`.

Dividing by the local volatility is what makes the interval **breathe**: it
widens in turbulent periods instead of averaging calm and storm into one width.

**The caveat.** The guarantee assumes exchangeability, which a time series
violates. That is why the coverage below is *measured*, not claimed.
"""
    )
    st.divider()

    left, right = st.columns(2)
    target = left.selectbox("Pair", ["(all)", *sorted(predictions["target"].unique())])
    horizon = right.selectbox(
        "Horizon", ["(all)", *sorted(predictions["horizon"].unique().tolist())]
    )
    subset = predictions
    if target != "(all)":
        subset = subset[subset["target"] == target]
    if horizon != "(all)":
        subset = subset[subset["horizon"] == horizon]

    calibration = uncertainty.calibration_table(subset, group_by=["model"])
    st.subheader("Nominal vs empirical coverage")
    st.plotly_chart(
        viz.calibration_chart(calibration, dark=is_dark(), height=460),
        width="stretch",
    )
    st.dataframe(
        calibration.rename(
            columns={
                "model": "Model",
                "nominal_coverage": "Nominal",
                "empirical_coverage": "Empirical",
                "coverage_error_pp": "Error (pp)",
                "coverage_se_pp": "s.e. (pp)",
                "mean_interval_width": "Mean width",
                "n": "n",
            }
        ).style.format(
            {
                "Nominal": "{:.0%}",
                "Empirical": "{:.1%}",
                "Error (pp)": "{:+.2f}",
                "s.e. (pp)": "{:.2f}",
                "Mean width": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "A gap smaller than about twice the standard error is indistinguishable "
        "from sampling noise — read the **s.e.** column before reading the **Error** one."
    )

    st.divider()
    st.subheader("Interval width across confidence levels")
    widths = uncertainty.interval_metrics_table(subset, group_by=["model"])
    st.plotly_chart(
        viz.regime_chart(
            widths.assign(regime=widths["nominal_coverage"].map(lambda c: f"{c:.0%}")),
            "mean_interval_width",
            regime_column="regime",
            title="Mean interval width by nominal level",
            dark=is_dark(),
        ),
        width="stretch",
    )
    st.caption(
        "Width has to grow with the level. A model whose 95% interval is barely "
        "wider than its 80% one is not expressing more uncertainty — it is "
        "mis-calibrated."
    )

    st.divider()
    st.subheader("Coverage over time")
    model = st.selectbox(
        "Model",
        sorted(subset["model"].unique()),
        index=sorted(subset["model"].unique()).index("xgboost")
        if "xgboost" in set(subset["model"])
        else 0,
    )
    level = int(config["conformal"]["primary_level"] * 100)
    rolling = uncertainty.coverage_over_time(
        subset[subset["model"] == model], level=level, window=60
    ).dropna()
    if not rolling.empty:
        chart_frame = rolling.set_index("origin")[["rolling_coverage"]].mul(100)
        chart_frame.columns = ["rolling coverage (60 origins)"]
        figure = viz.line_chart(
            chart_frame,
            title=f"{model}: rolling empirical coverage at {level}%",
            y_title="coverage (%)",
            dark=is_dark(),
            hover_format=".1f",
        )
        figure.add_hline(
            y=level,
            line={"width": 1.5, "dash": "dash", "color": "#73726d"},
            annotation={"text": f"nominal {level}%"},
            annotation_position="top left",
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "A single average can hide long stretches of under-coverage. This is "
            "where a static conformal quantile shows its limits under regime change."
        )
    render_disclaimer()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = [
    st.Page(page_overview, title="Overview", url_path="overview", default=True),
    st.Page(page_usd, title="USD/MRU", url_path="usd-mru"),
    st.Page(page_eur, title="EUR/MRU", url_path="eur-mru"),
    st.Page(page_models, title="Model comparison", url_path="models"),
    st.Page(page_uncertainty, title="Uncertainty", url_path="uncertainty"),
]

# Guarded so that `import dashboard.app` in a test does not run a page as a
# side effect. Streamlit executes the entry script as "__main__".
if __name__ == "__main__":
    st.navigation(PAGES).run()
