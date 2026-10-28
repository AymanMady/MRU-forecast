"""Plotly figures, shared by the EDA notebooks, the figure script and the dashboard.

One module so every chart in the project reads as one system: the same palette,
the same grid weight, the same hover behaviour.

Palette
-------
A validated categorical set (adjacent-pair CVD dE >= 8, normal-vision dE >= 15 in
both light and dark).  Three light-mode slots sit below 3:1 contrast on the light
surface, so every chart that uses them ships either direct labels or the
accompanying table — the callers in ``dashboard/`` and ``scripts/run_eda.py`` do.

Rules enforced here rather than left to the caller: one y-axis per figure (never
a second scale), a legend whenever two or more series are drawn, 2px lines,
markers at least 8px, and a recessive grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- palette -----------------------------------------------------------------
CATEGORICAL_LIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_BLUE_RED_LIGHT = [
    "#0d366b",
    "#256abf",
    "#86b6ef",
    "#f0efec",
    "#f2a3a2",
    "#e34948",
    "#8f1f1e",
]
DIVERGING_BLUE_RED_DARK = [
    "#0d366b",
    "#256abf",
    "#86b6ef",
    "#383835",
    "#f2a3a2",
    "#e66767",
    "#8f1f1e",
]


@dataclass(frozen=True)
class Theme:
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    axis: str
    categorical: list[str]
    diverging: list[str]

    def color(self, index: int) -> str:
        return self.categorical[index % len(self.categorical)]


LIGHT = Theme(
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#73726d",
    grid="#eceae4",
    axis="#d3d1c9",
    categorical=CATEGORICAL_LIGHT,
    diverging=DIVERGING_BLUE_RED_LIGHT,
)
DARK = Theme(
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#9a998f",
    grid="#2e2e2c",
    axis="#45443f",
    categorical=CATEGORICAL_DARK,
    diverging=DIVERGING_BLUE_RED_DARK,
)

# Stable colour per entity: a filter that removes a series must never repaint
# the ones that remain.
SERIES_SLOT = {
    "USD_MRU": 0,
    "EUR_MRU": 1,
    "EUR_USD": 2,
    "BRENT_OIL": 3,
    "GOLD": 4,
    "naive": 0,
    "seasonal_naive": 1,
    "drift": 2,
    "xgboost": 3,
    "prophet": 4,
}


def get_theme(dark: bool = False) -> Theme:
    return DARK if dark else LIGHT


def series_color(name: str, theme: Theme, fallback: int = 0) -> str:
    return theme.color(SERIES_SLOT.get(name, fallback))


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(
    theme: Theme, title: str = "", *, height: int = 420, y_title: str = "", x_title: str = ""
) -> dict[str, Any]:
    """Shared layout: recessive chrome, unified hover, legend above the plot."""
    return {
        "template": "plotly_white" if theme is LIGHT else "plotly_dark",
        "title": {
            "text": title,
            "font": {"size": 16, "color": theme.text_primary},
            "x": 0,
            "xanchor": "left",
            "pad": {"b": 12},
        },
        "height": height,
        "paper_bgcolor": theme.surface,
        "plot_bgcolor": theme.surface,
        "font": {
            "family": "system-ui, -apple-system, Segoe UI, sans-serif",
            "size": 12,
            "color": theme.text_secondary,
        },
        "margin": {"l": 56, "r": 24, "t": 64 if title else 32, "b": 48},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": theme.surface,
            "bordercolor": theme.axis,
            "font": {"color": theme.text_primary, "size": 12},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 11}},
            "gridcolor": theme.grid,
            "linecolor": theme.axis,
            "zeroline": False,
            "showspikes": False,
        },
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 11}},
            "gridcolor": theme.grid,
            "linecolor": theme.axis,
            "zeroline": False,
        },
    }


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def line_chart(
    frame: pd.DataFrame,
    *,
    title: str = "",
    y_title: str = "",
    dark: bool = False,
    columns: Sequence[str] | None = None,
    height: int = 420,
    hover_format: str = ".4f",
) -> go.Figure:
    """One line per column, on a single shared y-axis.

    Columns on different scales belong in separate figures, not on a second
    axis — a dual-axis chart lets the author choose where the lines cross.
    """
    theme = get_theme(dark)
    cols = list(columns) if columns else list(frame.columns)
    fig = go.Figure()
    for name in cols:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[name],
                name=name.replace("_", "/"),
                mode="lines",
                line={"width": 2, "color": series_color(name, theme)},
                hovertemplate=f"%{{y:{hover_format}}}<extra>{name.replace('_', '/')}</extra>",
            )
        )
    layout = base_layout(theme, title, height=height, y_title=y_title)
    # A single series is named by the title; a legend box would be noise.
    layout["showlegend"] = len(cols) > 1
    fig.update_layout(**layout)
    return fig


def forecast_chart(
    history: pd.Series,
    forecast: pd.DataFrame,
    *,
    title: str = "",
    y_title: str = "",
    dark: bool = False,
    history_window: int | None = 180,
    interval_columns: Sequence[tuple[str, str, str]] = (("lower", "upper", "90%"),),
    height: int = 460,
) -> go.Figure:
    """History, point forecast and one or more prediction-interval bands.

    ``forecast`` is indexed by date with a ``prediction`` column plus the bound
    columns named in ``interval_columns`` as ``(lower, upper, label)``.  Bands
    are drawn widest-first so a narrower one is never hidden underneath.
    """
    theme = get_theme(dark)
    name = history.name or "series"
    base = series_color(str(name), theme)
    shown = history.iloc[-history_window:] if history_window else history

    fig = go.Figure()

    ordered = sorted(
        interval_columns,
        key=lambda c: float((forecast[c[1]] - forecast[c[0]]).mean()) if c[0] in forecast else 0.0,
        reverse=True,
    )
    for depth, (low_col, high_col, label) in enumerate(ordered):
        if low_col not in forecast or high_col not in forecast:
            continue
        alpha = 0.10 + 0.07 * depth
        fig.add_trace(
            go.Scatter(
                x=list(forecast.index) + list(forecast.index[::-1]),
                y=list(forecast[high_col]) + list(forecast[low_col][::-1]),
                fill="toself",
                fillcolor=_hex_to_rgba(base, alpha),
                line={"width": 0},
                hoverinfo="skip",
                name=f"{label} interval",
                legendgroup=label,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=shown.index,
            y=shown.to_numpy(),
            name="Observed",
            mode="lines",
            line={"width": 2, "color": base},
            hovertemplate="%{y:.4f}<extra>Observed</extra>",
        )
    )

    if len(shown) and len(forecast):
        # Join the last observation to the first forecast so the two lines read
        # as one path rather than two disconnected segments.
        bridge_x = [shown.index[-1], forecast.index[0]]
        bridge_y = [shown.iloc[-1], forecast["prediction"].iloc[0]]
        fig.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_y,
                mode="lines",
                showlegend=False,
                line={"width": 2, "color": base, "dash": "dot"},
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["prediction"],
            name="Forecast",
            mode="lines+markers",
            line={"width": 2, "color": base, "dash": "dash"},
            marker={"size": 8, "line": {"width": 2, "color": theme.surface}},
            hovertemplate="%{y:.4f}<extra>Forecast</extra>",
        )
    )

    if len(shown):
        fig.add_vline(
            x=shown.index[-1],
            line={"width": 1, "color": theme.axis, "dash": "dot"},
            annotation={"text": "forecast start", "font": {"size": 10, "color": theme.text_muted}},
            annotation_position="top left",
        )

    fig.update_layout(**base_layout(theme, title, height=height, y_title=y_title))
    return fig


def volatility_chart(
    volatility: pd.DataFrame | pd.Series,
    *,
    title: str = "",
    dark: bool = False,
    window: int = 30,
    height: int = 360,
) -> go.Figure:
    frame = volatility.to_frame() if isinstance(volatility, pd.Series) else volatility
    return line_chart(
        frame,
        title=title or f"Annualised volatility ({window}-day rolling)",
        y_title="annualised sigma (%)",
        dark=dark,
        height=height,
        hover_format=".2f",
    )


def returns_histogram(
    returns: pd.DataFrame,
    *,
    title: str = "",
    dark: bool = False,
    bins: int = 80,
    height: int = 380,
    clip_quantile: float = 0.999,
) -> go.Figure:
    """Overlaid return distributions, clipped so one outlier cannot set the range."""
    theme = get_theme(dark)
    fig = go.Figure()
    limit = float(np.nanquantile(returns.abs().to_numpy(), clip_quantile))
    for name in returns.columns:
        values = returns[name].dropna() * 100.0
        fig.add_trace(
            go.Histogram(
                x=values,
                name=name.replace("_", "/"),
                nbinsx=bins,
                opacity=0.65,
                marker={
                    "color": series_color(name, theme),
                    "line": {"width": 2, "color": theme.surface},
                },
                hovertemplate="%{x:.2f}%%: %{y}<extra>" + name.replace("_", "/") + "</extra>",
            )
        )
    layout = base_layout(
        theme,
        title or "Daily log-return distribution",
        height=height,
        y_title="days",
        x_title="log return (%)",
    )
    layout["barmode"] = "overlay"
    layout["hovermode"] = "closest"
    layout["xaxis"]["range"] = [-limit * 100, limit * 100]
    fig.update_layout(**layout)
    return fig


def seasonality_chart(
    table: pd.DataFrame,
    *,
    title: str = "",
    dark: bool = False,
    height: int = 360,
) -> go.Figure:
    """Mean return per calendar group, with +/-1 standard error.

    The error bars are the point: on this data most group means are well inside
    their own uncertainty, which is the honest conclusion.
    """
    theme = get_theme(dark)
    standard_error = table["std_pct"] / np.sqrt(table["n"].clip(lower=1))
    fig = go.Figure(
        go.Bar(
            x=[str(i) for i in table.index],
            y=table["mean_pct"],
            marker={"color": theme.color(0), "line": {"width": 2, "color": theme.surface}},
            error_y={
                "type": "data",
                "array": standard_error,
                "color": theme.text_muted,
                "thickness": 1.5,
                "width": 6,
            },
            text=[f"{v:+.3f}%" for v in table["mean_pct"]],
            textposition="outside",
            textfont={"size": 10, "color": theme.text_secondary},
            hovertemplate="%{x}: %{y:.4f}%<extra></extra>",
            showlegend=False,
        )
    )
    layout = base_layout(
        theme,
        title or "Mean daily return by calendar group",
        height=height,
        y_title="mean log return (%)",
    )
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    fig.add_hline(y=0, line={"width": 1, "color": theme.axis})
    return fig


def correlation_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str = "",
    dark: bool = False,
    height: int = 440,
) -> go.Figure:
    """Diverging blue<->red with a neutral midpoint: correlation has a true zero."""
    theme = get_theme(dark)
    labels = [c.replace("_", "/") for c in matrix.columns]
    colorscale = [[i / (len(theme.diverging) - 1), c] for i, c in enumerate(theme.diverging)]
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=colorscale,
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in matrix.to_numpy()],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
            colorbar={"title": {"text": "rho", "font": {"size": 11}}, "thickness": 12},
        )
    )
    layout = base_layout(theme, title or "Correlation of daily log-returns", height=height)
    layout["hovermode"] = "closest"
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)
    return fig


def acf_chart(
    table: pd.DataFrame,
    *,
    column: str = "acf",
    title: str = "",
    dark: bool = False,
    height: int = 340,
) -> go.Figure:
    theme = get_theme(dark)
    fig = go.Figure(
        go.Bar(
            x=table.index,
            y=table[column],
            marker={"color": theme.color(0), "line": {"width": 2, "color": theme.surface}},
            hovertemplate="lag %{x}: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    for bound, dash in ((table["conf_upper"].iloc[0], "dot"), (table["conf_lower"].iloc[0], "dot")):
        fig.add_hline(y=bound, line={"width": 1, "color": theme.text_muted, "dash": dash})
    layout = base_layout(
        theme,
        title or f"{column.upper()} of daily log-returns",
        height=height,
        y_title=column.upper(),
        x_title="lag (business days)",
    )
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def calibration_chart(
    calibration: pd.DataFrame,
    *,
    title: str = "",
    dark: bool = False,
    height: int = 420,
) -> go.Figure:
    """Nominal vs empirical coverage against the 45-degree line.

    ``calibration`` needs ``nominal_coverage`` and ``empirical_coverage`` columns
    (as fractions); an optional ``model`` column draws one line per model.
    """
    theme = get_theme(dark)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.5, 1.0],
            y=[0.5, 1.0],
            mode="lines",
            name="perfect calibration",
            line={"width": 1.5, "color": theme.text_muted, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    groups = (
        calibration.groupby("model")
        if "model" in calibration.columns
        else [("observed", calibration)]
    )
    for index, (name, group) in enumerate(groups):
        group = group.sort_values("nominal_coverage")
        fig.add_trace(
            go.Scatter(
                x=group["nominal_coverage"],
                y=group["empirical_coverage"],
                name=str(name),
                mode="lines+markers+text",
                line={"width": 2, "color": theme.color(index)},
                marker={"size": 10, "line": {"width": 2, "color": theme.surface}},
                text=[f"{v:.1%}" for v in group["empirical_coverage"]],
                textposition="bottom right",
                textfont={"size": 10, "color": theme.text_secondary},
                hovertemplate="nominal %{x:.0%} -> empirical %{y:.1%}<extra>"
                + str(name)
                + "</extra>",
            )
        )
    layout = base_layout(
        theme,
        title or "Interval calibration",
        height=height,
        y_title="empirical coverage",
        x_title="nominal coverage",
    )
    layout["hovermode"] = "closest"
    layout["xaxis"]["tickformat"] = ".0%"
    layout["yaxis"]["tickformat"] = ".0%"
    fig.update_layout(**layout)
    return fig


def metric_bar_chart(
    table: pd.DataFrame,
    metric: str,
    *,
    group_column: str = "model",
    title: str = "",
    dark: bool = False,
    height: int = 380,
    value_format: str = ".4f",
    lower_is_better: bool = True,
) -> go.Figure:
    """Ranked bars with the value printed on every bar (the relief rule)."""
    theme = get_theme(dark)
    ordered = table.sort_values(metric, ascending=lower_is_better)
    fig = go.Figure(
        go.Bar(
            x=ordered[metric],
            y=[str(v).replace("_", " ") for v in ordered[group_column]],
            orientation="h",
            marker={
                "color": [
                    series_color(str(v), theme, i) for i, v in enumerate(ordered[group_column])
                ],
                "line": {"width": 2, "color": theme.surface},
            },
            text=[format(v, value_format) for v in ordered[metric]],
            textposition="outside",
            textfont={"size": 11, "color": theme.text_secondary},
            hovertemplate="%{y}: %{x:" + value_format + "}<extra></extra>",
            showlegend=False,
        )
    )
    layout = base_layout(theme, title or metric.upper(), height=height, x_title=metric)
    layout["hovermode"] = "closest"
    layout["yaxis"]["autorange"] = "reversed"
    layout["margin"]["l"] = 150
    fig.update_layout(**layout)
    return fig


def regime_chart(
    table: pd.DataFrame,
    metric: str,
    *,
    regime_column: str = "regime",
    model_column: str = "model",
    title: str = "",
    dark: bool = False,
    height: int = 400,
) -> go.Figure:
    """Grouped bars: one group per volatility regime, one bar per model."""
    theme = get_theme(dark)
    order = [r for r in ("low", "normal", "high") if r in set(table[regime_column])]
    fig = go.Figure()
    for index, model in enumerate(sorted(table[model_column].unique())):
        subset = table[table[model_column] == model].set_index(regime_column).reindex(order)
        fig.add_trace(
            go.Bar(
                x=order,
                y=subset[metric],
                name=str(model).replace("_", " "),
                marker={
                    "color": series_color(str(model), theme, index),
                    "line": {"width": 2, "color": theme.surface},
                },
                hovertemplate="%{x}: %{y:.4f}<extra>" + str(model) + "</extra>",
            )
        )
    layout = base_layout(
        theme,
        title or f"{metric.upper()} by volatility regime",
        height=height,
        y_title=metric,
        x_title="trailing volatility regime",
    )
    layout["barmode"] = "group"
    layout["bargap"] = 0.28
    layout["bargroupgap"] = 0.08
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig


def backtest_chart(
    actual: pd.Series,
    predictions: pd.DataFrame,
    *,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    title: str = "",
    dark: bool = False,
    height: int = 440,
    y_title: str = "",
) -> go.Figure:
    """Out-of-sample backtest: realised values against one or more model paths."""
    theme = get_theme(dark)
    fig = go.Figure()
    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=list(upper.index) + list(lower.index[::-1]),
                y=list(upper.to_numpy()) + list(lower.to_numpy()[::-1]),
                fill="toself",
                fillcolor=_hex_to_rgba(theme.color(0), 0.12),
                line={"width": 0},
                name="prediction interval",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.to_numpy(),
            name="Actual",
            mode="lines",
            line={"width": 2, "color": theme.text_primary},
            hovertemplate="%{y:.4f}<extra>Actual</extra>",
        )
    )
    for index, column in enumerate(predictions.columns):
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions[column],
                name=str(column).replace("_", " "),
                mode="lines",
                line={"width": 2, "dash": "dash", "color": series_color(str(column), theme, index)},
                hovertemplate="%{y:.4f}<extra>" + str(column) + "</extra>",
            )
        )
    fig.update_layout(**base_layout(theme, title, height=height, y_title=y_title))
    return fig


def save_figure(fig: go.Figure, path, *, include_plotlyjs: str = "cdn") -> None:
    """Write a self-contained interactive HTML figure."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(target), include_plotlyjs=include_plotlyjs, full_html=True)
