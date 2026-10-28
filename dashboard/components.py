"""Shared loaders and widgets for the Streamlit dashboard.

Loading is cached at module level so switching pages does not re-read parquet or
re-instantiate 60 predictors.  Anything that renders a number also renders where
that number came from — a coverage figure with no sample size behind it invites
being over-read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.cleaning import load_processed
from mru_forecast.pipeline.forecasting_pipeline import ForecastingPipeline

DISCLAIMER = (
    "**Experimental forecasting tool.** Rates are indicative mid-market quotes from a "
    "public market-data provider — *not* the official rate published by the Banque "
    "Centrale de Mauritanie. EUR/MRU is a derived cross (USD/MRU x EUR/USD), not an "
    "observed quote. Prediction intervals are calibrated on historical backtests and "
    "are not a guarantee. Not investment advice."
)

PAIR_LABEL = {"USD_MRU": "USD/MRU", "EUR_MRU": "EUR/MRU"}


@st.cache_data(show_spinner=False)
def get_config() -> dict[str, Any]:
    return load_config()


@st.cache_data(show_spinner="Loading rates...")
def get_frame() -> pd.DataFrame:
    return load_processed(get_config())


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline() -> ForecastingPipeline | None:
    try:
        return ForecastingPipeline.load(config=get_config())
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def get_report(filename: str) -> pd.DataFrame | None:
    path: Path = resolve_path("reports", get_config()) / filename
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_forecast(target: str, horizon: int) -> pd.DataFrame | None:
    pipeline = get_pipeline()
    if pipeline is None:
        return None
    return pipeline.forecast(get_frame(), target=target, horizon=horizon)


def is_dark() -> bool:
    """Match the chart palette to the viewer's Streamlit theme."""
    try:
        theme = st.context.theme
        return bool(theme and getattr(theme, "type", "light") == "dark")
    except Exception:
        try:
            return st.get_option("theme.base") == "dark"
        except Exception:
            return False


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def render_disclaimer() -> None:
    st.caption(DISCLAIMER)


def require_artifacts() -> bool:
    """Show an actionable message instead of a stack trace when nothing is trained."""
    if get_pipeline() is None:
        st.warning(
            "No trained models found. Build the artefacts first:\n\n"
            "```bash\n"
            "python scripts/download_data.py\n"
            "python scripts/build_dataset.py\n"
            "python scripts/train.py\n"
            "```"
        )
        return False
    return True


def require_backtest() -> pd.DataFrame | None:
    predictions = get_report("backtest_predictions_tagged.parquet")
    if predictions is None:
        predictions = get_report("backtest_predictions.parquet")
    if predictions is None:
        st.warning("No backtest results found. Run:\n\n```bash\npython scripts/backtest.py\n```")
    return predictions


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """A row of ``st.metric`` tiles: ``(label, value, help)``."""
    columns = st.columns(len(items))
    for column, (label, value, helptext) in zip(columns, items, strict=True):
        column.metric(label, value, help=helptext)


def format_interval(lower: float, upper: float, decimals: int = 3) -> str:
    return f"[{lower:.{decimals}f} to {upper:.{decimals}f}]"


def forecast_table(path: pd.DataFrame, level: int) -> pd.DataFrame:
    """The interval table shown next to every forecast chart.

    Three of the categorical palette's light-mode slots sit below 3:1 contrast on
    a light surface; the data-viz rules require relief for those, and this table
    is it — every value on screen is also readable as text.
    """
    table = pd.DataFrame(
        {
            "Date": [d.date() for d in path.index],
            "Days ahead": path["days_ahead"].astype(int).to_numpy(),
            "Prediction": path["prediction"].round(4).to_numpy(),
            f"Lower {level}%": path[f"lower_{level}"].round(4).to_numpy(),
            f"Upper {level}%": path[f"upper_{level}"].round(4).to_numpy(),
            "Width": (path[f"upper_{level}"] - path[f"lower_{level}"]).round(4).to_numpy(),
        }
    )
    return table.set_index("Date")
