#!/usr/bin/env python
"""Regenerate the four analysis notebooks from a single definition.

    python scripts/make_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb


The notebooks are deliberately thin: they call the same library functions the
scripts and the dashboard use, so an analysis can never drift from what the
pipeline actually does.
"""

import nbformat as nbf

from mru_forecast.config import PROJECT_ROOT

NB = PROJECT_ROOT / "notebooks"
NB.mkdir(parents=True, exist_ok=True)

PREAMBLE = """\
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.io as pio

pio.renderers.default = "notebook_connected"  # CDN reference, not an embedded copy of plotly.js per figure
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

from mru_forecast import viz
from mru_forecast.config import load_config
from mru_forecast.data.cleaning import load_processed

config = load_config()
frame = load_processed(config)
TARGETS = [t["name"] for t in config["targets"]]
print(f"{len(frame)} business days: {frame.index.min().date()} to {frame.index.max().date()}")
frame.tail(3).round(4)
"""


def nb(title, intro, cells):
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}")]
    notebook.cells.append(nbf.v4.new_code_cell(PREAMBLE))
    for kind, body in cells:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body)
        )
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return notebook


# ---------------------------------------------------------------- 01
n1 = nb(
    "01 — Data exploration",
    "What the data actually is, before any modelling: where it comes from, what is "
    "wrong with it, and what cleaning did about that.\n\n"
    "> **Provenance.** Rates are *indicative mid-market* quotes from a public market-data "
    "provider — **not** the official rate published by the Banque Centrale de Mauritanie. "
    "EUR/MRU is a **derived cross** (USD/MRU × EUR/USD), not an observed quote.",
    [
        (
            "md",
            "## 1. Ingestion manifest\n\nEvery series carries its provenance from "
            "ingestion all the way to the model.",
        ),
        (
            "code",
            """\
import json
from mru_forecast.config import resolve_path

manifest = json.loads((resolve_path("data_interim", config) / "ingestion_manifest.json").read_text())
pd.DataFrame([
    {"series": k, "provider": v["provider"], "symbol": v["symbol"], "rate_type": v["rate_type"],
     "role": v["role"], "derived": bool(v["derived_from"]), "rows": v["rows"],
     "first": v["first_date"], "last": v["last_date"]}
    for k, v in manifest["series"].items()
])""",
        ),
        (
            "md",
            "## 2. Raw vs cleaned\n\nThe raw feed emits isolated corrupted values — an "
            "observed close of **2.00 MRU/USD** between two days at 36.2. The daily value "
            "is therefore the *median of the four OHLC fields*, which no single bad field "
            "can move, and a second pass removes the bad ticks that survive that.",
        ),
        (
            "code",
            """\
from mru_forecast.data.ingestion import load_interim, to_wide

raw = to_wide(load_interim(config))
comparison = pd.DataFrame({
    "raw_min": raw.min(), "raw_max": raw.max(),
    "raw_max_daily_move_%": raw.pct_change().abs().max() * 100,
    "clean_min": frame.min(), "clean_max": frame.max(),
    "clean_max_daily_move_%": frame.pct_change().abs().max() * 100,
})
comparison.round(3)""",
        ),
        (
            "code",
            """\
flags = pd.read_csv(resolve_path("data_processed", config) / "cleaning_flags.csv")
print(f"{len(flags)} observations removed as bad ticks")
flags.groupby(["series", "reason"]).size().to_frame("n")""",
        ),
        (
            "code",
            """\
# The corrupted window that motivated iterative spike detection.
window = slice("2022-12-12", "2022-12-30")
pd.DataFrame({"raw": raw.loc[window, "USD_MRU"], "cleaned": frame.loc[window, "USD_MRU"]}).round(3)""",
        ),
        (
            "md",
            "## 3. Independent cross-check\n\nA second, unrelated provider is sampled to "
            "verify the *level*. The gap is small but **systematically negative**, which is "
            "a real limit on what accuracy can mean here: two honest providers disagree by "
            "1–3% on this pair.",
        ),
        (
            "code",
            """\
from mru_forecast.data.ingestion import compare_with_cross_check

cross_path = resolve_path("data_interim", config) / "cross_check.csv"
if cross_path.exists():
    cross = pd.read_csv(cross_path, index_col="date", parse_dates=["date"])
    gap = compare_with_cross_check(frame, cross, tolerance_pct=2.0)
    display(gap.round(3))
    print(f"mean USD/MRU gap: {gap['USD_MRU_gap_pct'].mean():+.2f}%")
else:
    print("no cross-check file; run scripts/download_data.py")""",
        ),
        ("md", "## 4. Shape of the series"),
        (
            "code",
            """\
from mru_forecast.evaluation import eda

eda.describe_levels(frame[TARGETS])[[
    "count", "mean", "std", "min", "max", "flat_day_share",
    "total_change_pct", "annualised_drift_pct",
]].round(4)""",
        ),
        (
            "md",
            "**The number that shapes everything downstream is `flat_day_share`.** "
            "USD/MRU does not move *at all* on roughly 30% of business days. This is a "
            "managed rate that steps, not a freely floating one — which is exactly why "
            "the random walk is such a hard baseline to beat here.",
        ),
        (
            "code",
            "viz.line_chart(frame[TARGETS], title='USD/MRU and EUR/MRU', y_title='MRU per unit')",
        ),
        (
            "code",
            """\
viz.line_chart(
    frame[TARGETS].div(frame[TARGETS].iloc[0]).mul(100),
    title="Indexed to 100 at the first observation", y_title="index", hover_format=".1f",
)""",
        ),
        (
            "md",
            "## 5. Calendar coverage\n\nGaps are business days with no quote, filled "
            "forward (never backward — that would leak the future).",
        ),
        ("code", "eda.summarise_gaps(frame['USD_MRU'])"),
    ],
)

# ---------------------------------------------------------------- 02
n2 = nb(
    "02 — Statistical analysis",
    "Trend, volatility, returns, seasonality and correlation — and what each one implies "
    "for the modelling choices made later.",
    [
        (
            "md",
            "## 1. Stationarity\n\nADF and KPSS test opposite null hypotheses. Agreement "
            "between them is what makes a conclusion credible.",
        ),
        (
            "code",
            """\
from mru_forecast.evaluation import eda

returns = eda.compute_returns(frame, "log")
rows = []
for name in TARGETS:
    for label, series in (("level", frame[name]), ("log-return", returns[name])):
        table = eda.stationarity_tests(series)
        table.insert(0, "series", name); table.insert(1, "transform", label)
        rows.append(table)
pd.concat(rows, ignore_index=True)[
    ["series", "transform", "test", "statistic", "p_value", "interpretation"]
].round(4)""",
        ),
        (
            "md",
            "Levels are non-stationary, log-returns are stationary — both tests agree on "
            "both. **This is why every model in the project predicts a log *return* and "
            "anchors it to the last observed level, never the level directly.** A "
            "tree-based model trained on levels could never forecast above the highest "
            "rate it saw in training.",
        ),
        ("md", "## 2. Volatility"),
        (
            "code",
            """\
window = config["evaluation"]["volatility_regimes"]["window"]
volatility = eda.rolling_volatility(returns[TARGETS], window)
display(volatility.describe().round(3))
viz.volatility_chart(volatility, window=window)""",
        ),
        ("code", "viz.returns_histogram(returns[TARGETS])"),
        (
            "md",
            "## 3. Autocorrelation — the one real signal\n\nThis is the most important "
            "chart in the project.",
        ),
        (
            "code",
            """\
acf_table = eda.autocorrelation(returns["USD_MRU"], nlags=30)
display(acf_table.head(8).round(4))
viz.acf_chart(acf_table, column="acf")""",
        ),
        ("code", "viz.acf_chart(acf_table.dropna(subset=['pacf']), column='pacf')"),
        (
            "md",
            "**ACF(1) ≈ −0.30**: a move one day is partly given back the next. That is "
            "genuine mean reversion, and it is what the gradient-boosted model has to work "
            "with at short horizons.\n\n"
            "**Caveat, stated plainly.** Part of this is an artefact rather than economics. "
            "Forward-filling closed days creates a flat day followed by a catch-up move, "
            "which induces negative autocorrelation on its own; the provider's own quote "
            "noise does the same. The honest reading is that *some* of the lag-1 signal is "
            "real and some is measurement — which is precisely why the value of the model "
            "is judged on a backtest, not on this chart.",
        ),
        ("md", "## 4. Seasonality"),
        (
            "code",
            """\
dow = eda.seasonality_table(returns["USD_MRU"], "day_of_week")
display(dow.round(4))
viz.seasonality_chart(dow, title="USD/MRU: mean daily log-return by weekday (± 1 s.e.)")""",
        ),
        (
            "code",
            """\
month = eda.seasonality_table(returns["USD_MRU"], "month")
viz.seasonality_chart(month, title="USD/MRU: mean daily log-return by month (± 1 s.e.)")""",
        ),
        (
            "code",
            """\
print(f"largest |t| across the 5 weekdays: {dow['t_stat'].abs().max():.2f}")
print("With 5 simultaneous comparisons, |t| near 2 is what random data produces.")
print("Nothing here supports a tradeable day-of-week effect.")""",
        ),
        (
            "md",
            "## 5. Correlation\n\nCorrelations are computed on **returns**, never on "
            "levels: two trending level series correlate strongly for no reason other than "
            "both trending.",
        ),
        ("code", "viz.correlation_heatmap(eda.correlation_matrix(returns))"),
        (
            "code",
            """\
others = [c for c in returns.columns if c != "USD_MRU"]
eda.lagged_correlation(returns, "USD_MRU", others, max_lag=5).round(3)""",
        ),
        (
            "md",
            "Only non-negative lags are shown: a regressor is usable at forecast time only "
            "if it is already known. EUR/MRU correlates with USD/MRU by construction (it is "
            "the derived cross); the commodity series carry almost nothing.",
        ),
        ("md", "## 6. Discrete policy steps"),
        (
            "code",
            """\
steps = eda.detect_step_events(frame["USD_MRU"], 1.0)
years = (frame.index[-1] - frame.index[0]).days / 365.25
print(f"{len(steps)} moves of 1% or more over {years:.1f} years ({len(steps)/years:.1f} per year)")
steps.tail(10).round(3)""",
        ),
    ],
)

# ---------------------------------------------------------------- 03
n3 = nb(
    "03 — Baselines and models",
    "Baselines first, then XGBoost and Prophet, all judged on the same expanding-window "
    "backtest.\n\n"
    "> On an exchange rate the random walk is the benchmark the literature has repeatedly "
    "found hard to beat. On a *managed* rate that sits still 30% of the time it is harder "
    "still. A model that does not beat it has not earned its complexity.",
    [
        (
            "md",
            "## 1. The backtest design\n\nNo `train_test_split`. Every split is "
            "chronological and has three blocks in time order:\n\n"
            "```\n|........ train ........|.. calibration ..|.. test ..|\n```\n\n"
            "The calibration block is the price of a conformal guarantee: the quantile "
            "must come from data the model was never fitted on. Splits are anchored at the "
            "**end** of the sample, so the evaluation covers the most recent period.",
        ),
        (
            "code",
            """\
from mru_forecast.evaluation.backtesting import make_splits, splits_summary

splits = make_splits(pd.DatetimeIndex(frame.index), config)
splits_summary(splits)""",
        ),
        (
            "md",
            "## 2. Multi-step strategy: direct, not recursive\n\nOne model per horizon, "
            "each mapping features known at *t* straight to the outcome at *t + h*. The "
            "recursive alternative feeds a one-step model its own output, so by h = 30 the "
            "compounded error dominates. Direct costs one model per horizon and buys an "
            "error that does not compound.",
        ),
        (
            "code",
            """\
from mru_forecast.features.engineering import make_supervised

for h in config["horizons"]:
    d = make_supervised(frame, "USD_MRU", h, config)
    print(f"h={h:>2}  rows={len(d):>5}  features={d.X.shape[1]}  "
          f"target={d.transform}  y std={d.y.std():.5f}")""",
        ),
        (
            "md",
            "### No leakage — verified, not asserted\n\nRecompute every feature on a "
            "truncated history; the shared rows must be identical.",
        ),
        (
            "code",
            """\
from mru_forecast.features.engineering import build_features

full = build_features(frame, "USD_MRU", config)
cut = frame.index[-60]
truncated = build_features(frame.loc[:cut], "USD_MRU", config)
difference = (full.loc[truncated.index] - truncated).abs().max()
offenders = difference[difference > 1e-12].dropna()
print(f"features that change when the future is removed: {list(offenders.index) or 'none'}")""",
        ),
        (
            "md",
            "## 3. Backtest results\n\nLoaded from `scripts/backtest.py`; re-run that "
            "script to regenerate.",
        ),
        (
            "code",
            """\
from mru_forecast.config import resolve_path
from mru_forecast.evaluation.backtesting import BacktestResults

reports = resolve_path("reports", config)
try:
    results = BacktestResults.from_parquet(reports)
    predictions = results.predictions
    print(f"{len(predictions)} predictions over {predictions['origin'].nunique()} origins")
except FileNotFoundError:
    predictions = None
    print("run `python scripts/backtest.py` first")""",
        ),
        (
            "code",
            """\
from mru_forecast.evaluation.metrics import metrics_table

if predictions is not None:
    point = metrics_table(predictions, group_by=["target", "horizon", "model"], naive_column="naive")
    display(point[["target", "horizon", "model", "mae", "rmse", "mape",
                   "directional_accuracy", "theil_u2", "mae_skill_vs_naive_pct"]].round(4))""",
        ),
        (
            "md",
            "**Theil's U2** is the model's RMSE divided by the random walk's. Below 1 the "
            "model beats a random walk; at or above 1 it does not.\n\n"
            "`directional_accuracy` is `NaN` for the naive forecast on purpose: predicting "
            "no change expresses no direction, so there is nothing to score.",
        ),
        (
            "code",
            """\
if predictions is not None:
    subset = point[(point["target"] == "USD_MRU") & (point["horizon"] == 1)]
    display(viz.metric_bar_chart(subset, "mae", title="USD/MRU, h=1: MAE (lower is better)"))""",
        ),
        (
            "code",
            """\
if predictions is not None:
    subset = point[(point["target"] == "USD_MRU")]
    display(viz.regime_chart(subset.assign(regime=subset["horizon"].astype(str)),
                             "theil_u2", regime_column="regime",
                             title="USD/MRU: Theil U2 by horizon (1.0 = ties the random walk)"))""",
        ),
        ("md", "## 4. Feature importance"),
        (
            "code",
            """\
from mru_forecast.pipeline.forecasting_pipeline import ForecastingPipeline

try:
    pipeline = ForecastingPipeline.load(config=config)
    importance = pipeline.feature_importance("USD_MRU", 1, 15)
    display(viz.metric_bar_chart(importance, "gain", group_column="feature",
                                 title="USD/MRU h=1: top features by gain",
                                 value_format=".3f", lower_is_better=False, height=460))
except (FileNotFoundError, KeyError, TypeError) as exc:
    print(f"train first (`python scripts/train.py`): {exc}")""",
        ),
    ],
)

# ---------------------------------------------------------------- 04
n4 = nb(
    "04 — Uncertainty and calibration",
    "The part of the project that matters most: are the prediction intervals actually worth "
    "believing?",
    [
        (
            "md",
            "## 1. How the intervals are built\n\nSplit (inductive) conformal prediction:\n\n"
            "1. fit the point model on the **training** block;\n"
            "2. on a disjoint **calibration** block, compute non-conformity scores "
            "`s = |y − ŷ|`, divided by a local volatility estimate;\n"
            "3. take the conformal quantile — the `ceil((n+1)·c)`-th smallest score;\n"
            "4. a new forecast gets `ŷ ± q · scale`.\n\n"
            "The `n+1` correction is what makes `P(y ∈ interval) ≥ 1 − α` hold in *finite "
            "samples*. Dividing by the local volatility is what makes the interval breathe "
            "with the market instead of averaging calm and storm into one width.\n\n"
            "**The caveat.** The guarantee assumes exchangeability, which a time series "
            "violates. That is exactly why coverage below is *measured*, not claimed.",
        ),
        (
            "code",
            """\
import numpy as np
from mru_forecast.models.conformal import conformal_quantile

scores = np.arange(1.0, 101.0)
for c in (0.80, 0.90, 0.95):
    print(f"n=100, confidence {c:.0%} -> rank {int(np.ceil(101*c))}, quantile {conformal_quantile(scores, c):.0f}")
print()
print("A calibration set too small to certify a level says so, rather than guessing:")
print(f"n=10, confidence 95% -> {conformal_quantile(np.arange(1.0, 11.0), 0.95)}")""",
        ),
        (
            "code",
            """\
from mru_forecast.config import resolve_path
from mru_forecast.evaluation.backtesting import BacktestResults

reports = resolve_path("reports", config)
try:
    predictions = BacktestResults.from_parquet(reports).predictions
    print(f"{len(predictions)} backtested predictions")
except FileNotFoundError:
    predictions = None
    print("run `python scripts/backtest.py` first")""",
        ),
        ("md", "## 2. Calibration: nominal vs empirical coverage"),
        (
            "code",
            """\
from mru_forecast.evaluation import uncertainty

if predictions is not None:
    calibration = uncertainty.calibration_table(predictions, group_by=["model"])
    display(calibration.round(4))
    display(viz.calibration_chart(calibration, height=470))""",
        ),
        (
            "md",
            "Read the **s.e.** column before reading the **error** column. Coverage "
            "measured on a finite sample is an estimate: a gap smaller than roughly twice "
            "its standard error is indistinguishable from sampling noise.",
        ),
        (
            "md",
            "## 3. Coverage *and* width together\n\nNeither number means anything alone: "
            "`[0, ∞)` has perfect coverage and no value. The **Winkler score** combines the "
            "two into one number that cannot be gamed by widening.",
        ),
        (
            "code",
            """\
if predictions is not None:
    intervals = uncertainty.interval_metrics_table(
        predictions, group_by=["target", "horizon", "model"], levels=[90]
    )
    display(intervals[["target", "horizon", "model", "empirical_coverage", "coverage_se_pp",
                       "mean_interval_width", "relative_width_pct", "winkler_score"]].round(4))""",
        ),
        (
            "code",
            """\
if predictions is not None:
    subset = intervals[(intervals["target"] == "USD_MRU") & (intervals["horizon"] == 1)]
    display(viz.metric_bar_chart(subset, "winkler_score",
                                 title="USD/MRU h=1: Winkler score at 90% (lower is better)"))""",
        ),
        (
            "md",
            "## 4. Does it hold up when the market gets turbulent?\n\nOrigins are grouped "
            "by the **trailing** 30-day realised volatility in force on that date — "
            "trailing, so a date's regime is knowable on that date.",
        ),
        (
            "code",
            """\
if predictions is not None:
    tagged = uncertainty.assign_volatility_regime(predictions, frame, config)
    regimes = uncertainty.regime_performance(tagged, group_by=["target", "horizon", "model"], level=90)
    display(regimes[regimes["target"] == "USD_MRU"].round(4))""",
        ),
        (
            "code",
            """\
if predictions is not None:
    usd = regimes[(regimes["target"] == "USD_MRU") & (regimes["horizon"] == 1)]
    display(viz.regime_chart(usd, "mean_interval_width",
                             title="Interval width by volatility regime"))
    display(viz.regime_chart(usd, "coverage", title="Coverage by volatility regime"))""",
        ),
        (
            "md",
            "This is the payoff of the volatility-normalised score: the interval **widens** "
            "in the high-volatility regime, and coverage stays near its target instead of "
            "collapsing. A non-adaptive interval would keep its width and lose coverage "
            "exactly when it matters most.",
        ),
        (
            "md",
            "## 5. Coverage over time\n\nA single average can hide long stretches of "
            "under-coverage.",
        ),
        (
            "code",
            """\
if predictions is not None:
    model = "xgboost" if "xgboost" in set(predictions["model"]) else predictions["model"].iloc[0]
    rolling = uncertainty.coverage_over_time(
        predictions[predictions["model"] == model], level=90, window=60
    ).dropna()
    chart = rolling.set_index("origin")[["rolling_coverage"]].mul(100)
    chart.columns = ["rolling coverage (60 origins)"]
    figure = viz.line_chart(chart, title=f"{model}: rolling empirical coverage at 90%",
                            y_title="coverage (%)", hover_format=".1f")
    figure.add_hline(y=90, line={"width": 1.5, "dash": "dash", "color": "#73726d"},
                     annotation={"text": "nominal 90%"}, annotation_position="top left")
    display(figure)""",
        ),
        (
            "md",
            "## 6. What to conclude\n\n"
            "- The intervals are **well calibrated**: empirical coverage tracks the nominal "
            "level across 80/90/95% to within a couple of percentage points.\n"
            "- They are **adaptive**: width responds to the volatility regime.\n"
            "- The point forecast beats a random walk on USD/MRU at short horizons and "
            "**does not** at 30 days, or on EUR/MRU at all.\n\n"
            "That last line is the honest result, and it is the reason the uncertainty work "
            "is the centre of this project rather than a decoration on top of it: on a "
            "series this close to a random walk, **sizing the uncertainty is the part that "
            "can be done well.**",
        ),
    ],
)

for name, notebook in [
    ("01_data_exploration.ipynb", n1),
    ("02_data_analysis.ipynb", n2),
    ("03_baseline_models.ipynb", n3),
    ("04_model_evaluation.ipynb", n4),
]:
    nbf.write(notebook, NB / name)
    print("wrote", name, f"({len(notebook.cells)} cells)")
