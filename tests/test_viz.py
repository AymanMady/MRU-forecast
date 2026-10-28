"""Chart construction: the rules that must hold for every figure in the project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mru_forecast import viz


@pytest.fixture
def history() -> pd.Series:
    index = pd.date_range("2025-01-01", periods=200, freq="B")
    return pd.Series(np.linspace(39, 40, 200), index=index, name="USD_MRU")


@pytest.fixture
def forecast_path() -> pd.DataFrame:
    index = pd.date_range("2026-09-15", periods=7, freq="B")
    prediction = np.linspace(40.0, 40.3, 7)
    return pd.DataFrame(
        {
            "prediction": prediction,
            "lower_80": prediction - 0.2,
            "upper_80": prediction + 0.2,
            "lower_90": prediction - 0.3,
            "upper_90": prediction + 0.3,
            "lower_95": prediction - 0.4,
            "upper_95": prediction + 0.4,
        },
        index=index,
    )


class TestPalette:
    def test_light_and_dark_have_the_same_slot_count(self):
        assert len(viz.CATEGORICAL_LIGHT) == len(viz.CATEGORICAL_DARK) == 8

    def test_every_colour_is_a_hex_triplet(self):
        for colour in viz.CATEGORICAL_LIGHT + viz.CATEGORICAL_DARK:
            assert colour.startswith("#") and len(colour) == 7

    def test_colour_follows_the_entity_not_its_position(self):
        """Removing a series must not repaint the ones that remain."""
        theme = viz.get_theme(False)
        assert viz.series_color("USD_MRU", theme) == viz.series_color("USD_MRU", theme)
        assert viz.series_color("USD_MRU", theme) != viz.series_color("EUR_MRU", theme)

    def test_diverging_scale_has_a_neutral_midpoint(self):
        middle = viz.DIVERGING_BLUE_RED_LIGHT[len(viz.DIVERGING_BLUE_RED_LIGHT) // 2]
        red, green, blue = (int(middle[i : i + 2], 16) for i in (1, 3, 5))
        assert max(red, green, blue) - min(red, green, blue) < 20, "midpoint must read as grey"


class TestSingleAxisRule:
    @pytest.mark.parametrize(
        "builder",
        [
            lambda f: viz.line_chart(f),
            lambda f: viz.returns_histogram(f.pct_change().dropna()),
            lambda f: viz.correlation_heatmap(f.corr()),
        ],
    )
    def test_no_figure_uses_a_second_y_axis(self, builder):
        frame = pd.DataFrame(
            {"USD_MRU": np.linspace(39, 40, 100), "BRENT_OIL": np.linspace(70, 110, 100)},
            index=pd.date_range("2025-01-01", periods=100, freq="B"),
        )
        layout = builder(frame).layout
        assert getattr(layout, "yaxis2", None) is None, "dual-axis charts are not allowed"


class TestLineChart:
    def test_one_trace_per_column(self, history):
        frame = history.to_frame().assign(EUR_MRU=history * 1.16)
        assert len(viz.line_chart(frame).data) == 2

    def test_a_single_series_gets_no_legend_box(self, history):
        assert viz.line_chart(history.to_frame()).layout.showlegend is False

    def test_two_series_get_a_legend(self, history):
        frame = history.to_frame().assign(EUR_MRU=history * 1.16)
        assert viz.line_chart(frame).layout.showlegend is not False

    def test_lines_are_two_pixels(self, history):
        figure = viz.line_chart(history.to_frame())
        assert all(trace.line.width == 2 for trace in figure.data)


class TestForecastChart:
    def test_draws_history_forecast_and_one_band_per_level(self, history, forecast_path):
        figure = viz.forecast_chart(
            history,
            forecast_path,
            interval_columns=[("lower_80", "upper_80", "80%"), ("lower_90", "upper_90", "90%")],
        )
        names = [t.name for t in figure.data]
        assert "Observed" in names and "Forecast" in names
        assert sum("interval" in (n or "") for n in names) == 2

    def test_widest_band_is_drawn_first(self, history, forecast_path):
        figure = viz.forecast_chart(
            history,
            forecast_path,
            interval_columns=[("lower_80", "upper_80", "80%"), ("lower_95", "upper_95", "95%")],
        )
        band_names = [t.name for t in figure.data if "interval" in (t.name or "")]
        assert band_names[0] == "95% interval", "a narrow band must not hide under a wide one"

    def test_missing_bound_columns_are_skipped_not_fatal(self, history, forecast_path):
        figure = viz.forecast_chart(
            history, forecast_path, interval_columns=[("lower_99", "upper_99", "99%")]
        )
        assert not any("interval" in (t.name or "") for t in figure.data)

    def test_history_window_limits_what_is_drawn(self, history, forecast_path):
        figure = viz.forecast_chart(history, forecast_path, history_window=30)
        observed = next(t for t in figure.data if t.name == "Observed")
        assert len(observed.x) == 30

    def test_forecast_markers_are_large_enough_to_hit(self, history, forecast_path):
        figure = viz.forecast_chart(history, forecast_path)
        marker = next(t for t in figure.data if t.name == "Forecast").marker
        assert marker.size >= 8


class TestEvaluationCharts:
    def test_calibration_chart_includes_the_diagonal(self):
        table = pd.DataFrame(
            {
                "model": ["xgboost"] * 3,
                "nominal_coverage": [0.8, 0.9, 0.95],
                "empirical_coverage": [0.81, 0.91, 0.94],
            }
        )
        names = [t.name for t in viz.calibration_chart(table).data]
        assert "perfect calibration" in names

    def test_metric_bars_are_labelled(self):
        table = pd.DataFrame({"model": ["naive", "xgboost"], "mae": [0.16, 0.14]})
        figure = viz.metric_bar_chart(table, "mae")
        assert figure.data[0].text is not None, "the relief rule requires visible values"

    def test_metric_bars_are_ranked(self):
        table = pd.DataFrame({"model": ["a", "b", "c"], "mae": [0.3, 0.1, 0.2]})
        figure = viz.metric_bar_chart(table, "mae", lower_is_better=True)
        assert list(figure.data[0].x) == [0.1, 0.2, 0.3]

    def test_regime_chart_orders_low_normal_high(self):
        table = pd.DataFrame(
            {"model": ["naive"] * 3, "regime": ["high", "low", "normal"], "mae": [0.3, 0.1, 0.2]}
        )
        figure = viz.regime_chart(table, "mae")
        assert list(figure.data[0].x) == ["low", "normal", "high"]

    def test_seasonality_chart_has_error_bars(self):
        table = pd.DataFrame(
            {"mean_pct": [0.01, -0.02], "std_pct": [0.5, 0.4], "n": [400, 400]},
            index=["Mon", "Tue"],
        )
        assert viz.seasonality_chart(table).data[0].error_y.array is not None


def test_save_figure_writes_standalone_html(tmp_path, history):
    target = tmp_path / "nested" / "figure.html"
    viz.save_figure(viz.line_chart(history.to_frame()), target)
    assert target.exists()
    assert "<html" in target.read_text(encoding="utf-8").lower()
