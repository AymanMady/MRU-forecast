"""Phase 11: point and interval metrics.

Each metric is checked against a value computed by hand or against a property it
must satisfy (a proper scoring rule is minimised by the truth, a coverage is a
fraction, widening must cost something).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mru_forecast.evaluation.metrics import (
    bias,
    directional_accuracy,
    evaluate_point_forecast,
    mae,
    mape,
    metrics_table,
    rmse,
    skill_score,
    smape,
    theil_u2,
)
from mru_forecast.evaluation.uncertainty import (
    calibration_table,
    coverage_over_time,
    coverage_standard_error,
    crps_from_quantiles,
    empirical_coverage,
    evaluate_interval,
    interval_metrics_table,
    mean_interval_width,
    pinball_loss,
    relative_interval_width,
    winkler_score,
)


class TestPointMetrics:
    def test_known_values(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([11.0, 18.0, 33.0])
        assert mae(actual, predicted) == pytest.approx(2.0)
        assert rmse(actual, predicted) == pytest.approx(np.sqrt((1 + 4 + 9) / 3))
        assert bias(actual, predicted) == pytest.approx((1 - 2 + 3) / 3)

    def test_perfect_forecast_scores_zero(self):
        values = np.array([36.0, 37.0, 38.0])
        assert mae(values, values) == 0.0
        assert rmse(values, values) == 0.0
        assert mape(values, values) == 0.0

    def test_rmse_is_at_least_mae(self):
        rng = np.random.default_rng(0)
        actual, predicted = rng.normal(40, 1, 500), rng.normal(40, 1, 500)
        assert rmse(actual, predicted) >= mae(actual, predicted)

    def test_missing_values_are_dropped_not_imputed(self):
        actual = np.array([10.0, np.nan, 30.0])
        predicted = np.array([11.0, 20.0, 33.0])
        assert mae(actual, predicted) == pytest.approx(2.0)
        assert evaluate_point_forecast(actual, predicted)["n"] == 2

    def test_all_missing_gives_nan(self):
        assert np.isnan(mae([np.nan, np.nan], [1.0, 2.0]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            mae([1.0, 2.0], [1.0])

    def test_smape_is_bounded(self):
        assert smape([1.0], [1e9]) <= 200.0


class TestDirectionalAccuracy:
    def test_counts_only_days_the_rate_actually_moved(self):
        anchor = np.array([40.0, 40.0, 40.0])
        actual = np.array([41.0, 40.0, 39.0])  # up, flat, down
        predicted = np.array([41.5, 45.0, 39.5])  # up, up, down
        assert directional_accuracy(actual, predicted, anchor) == pytest.approx(100.0)

    def test_a_forecast_of_no_change_has_no_direction(self):
        """The naive forecast expresses no view; the honest score is nan, not 0%."""
        anchor = np.array([40.0, 40.0])
        actual = np.array([41.0, 39.0])
        assert np.isnan(directional_accuracy(actual, anchor, anchor))

    def test_always_wrong_scores_zero(self):
        anchor = np.array([40.0, 40.0])
        actual = np.array([41.0, 39.0])
        predicted = np.array([39.0, 41.0])
        assert directional_accuracy(actual, predicted, anchor) == pytest.approx(0.0)


class TestSkillMetrics:
    def test_theil_u2_is_one_for_the_naive_forecast_itself(self):
        rng = np.random.default_rng(1)
        actual, naive = rng.normal(40, 1, 300), rng.normal(40, 1, 300)
        assert theil_u2(actual, naive, naive) == pytest.approx(1.0)

    def test_theil_u2_below_one_means_better_than_naive(self):
        actual = np.array([40.0, 41.0, 42.0])
        better = np.array([40.1, 41.1, 42.1])
        naive = np.array([39.0, 40.0, 41.0])
        assert theil_u2(actual, better, naive) < 1.0

    def test_skill_score_is_positive_when_the_model_wins(self):
        actual = np.array([40.0, 41.0])
        better, worse = np.array([40.1, 41.1]), np.array([42.0, 43.0])
        assert skill_score(actual, better, worse) > 0

    def test_evaluate_point_forecast_reports_everything_requested(self):
        actual = np.array([40.0, 41.0, 42.0])
        predicted = np.array([40.2, 40.9, 42.3])
        naive = np.array([39.5, 40.0, 41.0])
        result = evaluate_point_forecast(actual, predicted, anchor=naive, y_naive=naive)
        assert {"mae", "rmse", "mape", "theil_u2", "directional_accuracy", "n"} <= set(result)


class TestCoverage:
    def test_coverage_is_the_fraction_inside(self):
        actual = np.array([1.0, 5.0, 10.0, 20.0])
        lower, upper = np.zeros(4), np.array([2.0, 6.0, 9.0, 19.0])
        assert empirical_coverage(actual, lower, upper) == pytest.approx(0.5)

    def test_an_infinite_interval_covers_everything(self):
        actual = np.array([1.0, 2.0])
        assert empirical_coverage(actual, [-np.inf, -np.inf], [np.inf, np.inf]) == 1.0

    def test_standard_error_shrinks_with_sample_size(self):
        assert coverage_standard_error(0.9, 1000) < coverage_standard_error(0.9, 100)

    def test_standard_error_at_the_documented_sample_size(self):
        assert coverage_standard_error(0.9, 252) == pytest.approx(0.0189, abs=1e-4)

    def test_width_helpers(self):
        lower, upper = np.array([1.0, 2.0]), np.array([3.0, 6.0])
        assert mean_interval_width(lower, upper) == pytest.approx(3.0)
        assert relative_interval_width([2.0, 4.0], lower, upper) == pytest.approx(100.0)


class TestProperScoringRules:
    def test_pinball_is_minimised_at_the_true_quantile(self):
        rng = np.random.default_rng(5)
        sample = rng.normal(0, 1, 60000)
        tau = 0.9
        truth = float(np.quantile(sample, tau))
        at_truth = pinball_loss(sample, np.full_like(sample, truth), tau)
        for offset in (-0.4, -0.1, 0.1, 0.4):
            assert pinball_loss(sample, np.full_like(sample, truth + offset), tau) > at_truth

    def test_pinball_rejects_an_invalid_tau(self):
        with pytest.raises(ValueError):
            pinball_loss([1.0], [1.0], 0.0)

    def test_winkler_is_minimised_by_the_correct_interval(self):
        rng = np.random.default_rng(6)
        sample = rng.normal(0, 1, 60000)
        correct = winkler_score(
            sample, np.full_like(sample, -1.6449), np.full_like(sample, 1.6449), 0.90
        )
        too_wide = winkler_score(
            sample, np.full_like(sample, -3.3), np.full_like(sample, 3.3), 0.90
        )
        too_narrow = winkler_score(
            sample, np.full_like(sample, -0.8), np.full_like(sample, 0.8), 0.90
        )
        assert correct < too_wide, "widening must be penalised"
        assert correct < too_narrow, "missing must be penalised"

    def test_winkler_equals_the_width_when_everything_is_covered(self):
        actual = np.array([5.0, 5.0])
        assert winkler_score(actual, [0.0, 0.0], [10.0, 10.0], 0.90) == pytest.approx(10.0)

    def test_crps_is_lower_for_a_sharper_correct_forecast(self):
        rng = np.random.default_rng(8)
        sample = rng.normal(0, 1, 20000)
        sharp = {0.05: np.full(20000, -1.64), 0.5: np.zeros(20000), 0.95: np.full(20000, 1.64)}
        blunt = {0.05: np.full(20000, -5.0), 0.5: np.zeros(20000), 0.95: np.full(20000, 5.0)}
        assert crps_from_quantiles(sample, sharp) < crps_from_quantiles(sample, blunt)

    def test_crps_needs_at_least_two_quantiles(self):
        assert np.isnan(crps_from_quantiles([1.0], {0.5: [1.0]}))


class TestTables:
    @pytest.fixture
    def tidy(self):
        rng = np.random.default_rng(12)
        n = 300
        origins = pd.date_range("2025-01-01", periods=n, freq="B")
        actual = 40 + rng.normal(0, 0.2, n)
        rows = []
        for model, spread in (("naive", 0.35), ("xgboost", 0.30)):
            prediction = actual + rng.normal(0, 0.1, n)
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "target": "USD_MRU",
                        "horizon": 1,
                        "origin": origins,
                        "actual": actual,
                        "prediction": prediction,
                        "anchor": actual - 0.05,
                        "naive": actual + rng.normal(0, 0.12, n),
                        "lower_80": prediction - spread * 0.8,
                        "upper_80": prediction + spread * 0.8,
                        "lower_90": prediction - spread,
                        "upper_90": prediction + spread,
                        "lower_95": prediction - spread * 1.3,
                        "upper_95": prediction + spread * 1.3,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def test_metrics_table_has_one_row_per_group(self, tidy):
        table = metrics_table(tidy, group_by=["model"], naive_column="naive")
        assert len(table) == 2
        assert {"mae", "rmse", "theil_u2"} <= set(table.columns)

    def test_interval_table_covers_every_level(self, tidy):
        table = interval_metrics_table(tidy, group_by=["model"])
        assert len(table) == 6  # 2 models x 3 levels
        assert set(table["nominal_coverage"].round(2)) == {0.80, 0.90, 0.95}

    def test_wider_levels_have_wider_intervals(self, tidy):
        table = interval_metrics_table(tidy, group_by=["model"]).sort_values(
            ["model", "nominal_coverage"]
        )
        for _, group in table.groupby("model"):
            assert group["mean_interval_width"].is_monotonic_increasing

    def test_calibration_table_is_sorted_and_complete(self, tidy):
        table = calibration_table(tidy, group_by=["model"])
        assert list(table.columns[:2]) == ["model", "nominal_coverage"]
        assert table["empirical_coverage"].between(0, 1).all()

    def test_evaluate_interval_returns_every_metric(self, tidy):
        subset = tidy[tidy["model"] == "naive"]
        result = evaluate_interval(subset["actual"], subset["lower_90"], subset["upper_90"], 0.90)
        assert {
            "empirical_coverage",
            "mean_interval_width",
            "pinball_loss",
            "winkler_score",
            "coverage_se_pp",
            "n",
        } <= set(result)

    def test_rolling_coverage_tracks_time(self, tidy):
        rolling = coverage_over_time(tidy[tidy["model"] == "naive"], level=90, window=50)
        assert len(rolling) == 300
        assert rolling["rolling_coverage"].dropna().between(0, 1).all()
