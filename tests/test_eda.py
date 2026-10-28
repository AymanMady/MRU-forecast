"""Phase 4: exploratory analysis primitives.

Each function is checked against a series built so that the right answer is
known in advance, rather than against whatever the real data happens to return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mru_forecast.evaluation.eda import (
    autocorrelation,
    compute_returns,
    correlation_matrix,
    describe_levels,
    detect_step_events,
    lagged_correlation,
    rolling_volatility,
    seasonality_table,
    stationarity_tests,
    summarise_gaps,
    volatility_regimes,
)


class TestDescribeLevels:
    def test_flat_day_share_is_measured_correctly(self):
        index = pd.date_range("2024-01-01", periods=11, freq="B")
        # 10 diffs, of which 5 are exactly zero.
        values = [36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0]
        summary = describe_levels(pd.DataFrame({"X": values}, index=index))
        assert summary.loc["X", "flat_day_share"] == pytest.approx(5 / 11)

    def test_total_change_matches_the_endpoints(self):
        index = pd.date_range("2024-01-01", periods=3, freq="B")
        summary = describe_levels(pd.DataFrame({"X": [100.0, 105.0, 110.0]}, index=index))
        assert summary.loc["X", "total_change_pct"] == pytest.approx(10.0)


class TestReturns:
    def test_log_returns_are_additive(self):
        index = pd.date_range("2024-01-01", periods=4, freq="B")
        series = pd.DataFrame({"X": [100.0, 110.0, 121.0, 133.1]}, index=index)
        returns = compute_returns(series, "log")["X"]
        assert returns.sum() == pytest.approx(np.log(133.1 / 100.0))

    def test_pct_change_matches_the_definition(self):
        index = pd.date_range("2024-01-01", periods=2, freq="B")
        returns = compute_returns(pd.DataFrame({"X": [100.0, 110.0]}, index=index), "pct")
        assert returns["X"].iloc[0] == pytest.approx(0.10)

    def test_unknown_kind_raises(self, synthetic_frame):
        with pytest.raises(ValueError, match="unknown return kind"):
            compute_returns(synthetic_frame[["USD_MRU"]], "nope")

    def test_volatility_is_annualised(self):
        rng = np.random.default_rng(0)
        index = pd.date_range("2020-01-01", periods=1000, freq="B")
        returns = pd.Series(rng.normal(0, 0.01, 1000), index=index)
        annual = rolling_volatility(returns, 252, annualise=True).iloc[-1]
        assert annual == pytest.approx(0.01 * np.sqrt(252) * 100, rel=0.15)


class TestStationarity:
    def test_a_random_walk_level_is_non_stationary(self):
        rng = np.random.default_rng(1)
        index = pd.date_range("2018-01-01", periods=1500, freq="B")
        series = pd.Series(36 * np.exp(np.cumsum(rng.normal(0, 0.004, 1500))), index=index)
        table = stationarity_tests(series).set_index("test")
        assert table.loc["ADF", "interpretation"] == "non-stationary"

    def test_its_returns_are_stationary(self):
        rng = np.random.default_rng(1)
        index = pd.date_range("2018-01-01", periods=1500, freq="B")
        series = pd.Series(36 * np.exp(np.cumsum(rng.normal(0, 0.004, 1500))), index=index)
        table = stationarity_tests(np.log(series).diff().dropna()).set_index("test")
        assert table.loc["ADF", "interpretation"] == "stationary"
        assert table.loc["KPSS", "interpretation"] == "stationary"

    def test_both_tests_are_reported(self, synthetic_frame):
        table = stationarity_tests(synthetic_frame["USD_MRU"])
        assert set(table["test"]) == {"ADF", "KPSS"}
        assert table["null_hypothesis"].nunique() == 2


class TestSeasonality:
    def test_grouping_labels_weekdays(self, synthetic_frame):
        returns = compute_returns(synthetic_frame[["USD_MRU"]], "log")["USD_MRU"]
        table = seasonality_table(returns, "day_of_week")
        assert list(table.index) == ["Mon", "Tue", "Wed", "Thu", "Fri"]
        assert table["n"].sum() == len(returns)

    def test_counts_are_reported_next_to_every_mean(self, synthetic_frame):
        returns = compute_returns(synthetic_frame[["USD_MRU"]], "log")["USD_MRU"]
        table = seasonality_table(returns, "month")
        assert {"mean_pct", "std_pct", "n", "t_stat"} <= set(table.columns)

    def test_unknown_grouping_raises(self, synthetic_frame):
        returns = compute_returns(synthetic_frame[["USD_MRU"]], "log")["USD_MRU"]
        with pytest.raises(ValueError, match="unknown grouping"):
            seasonality_table(returns, "decade")


class TestCorrelation:
    def test_perfect_correlation_is_one(self):
        index = pd.date_range("2024-01-01", periods=100, freq="B")
        rng = np.random.default_rng(2)
        base = rng.normal(size=100)
        frame = pd.DataFrame({"A": base, "B": base * 2.0}, index=index)
        assert correlation_matrix(frame).loc["A", "B"] == pytest.approx(1.0)

    def test_lagged_correlation_reports_only_usable_lags(self, synthetic_frame):
        returns = compute_returns(synthetic_frame, "log")
        table = lagged_correlation(returns, "USD_MRU", ["EUR_USD"], max_lag=5)
        assert list(table.index) == list(range(6))
        assert (table.index >= 0).all(), "a negative lag would not be known at forecast time"

    def test_autocorrelation_has_confidence_bounds(self, synthetic_frame):
        returns = compute_returns(synthetic_frame[["USD_MRU"]], "log")["USD_MRU"]
        table = autocorrelation(returns, nlags=20)
        assert table["acf"].iloc[0] == pytest.approx(1.0)
        assert (table["conf_upper"] > 0).all()
        assert (table["conf_lower"] < 0).all()


class TestRegimesAndEvents:
    def test_step_events_collapse_consecutive_days(self):
        index = pd.date_range("2024-01-01", periods=10, freq="B")
        values = [36.0, 36.0, 38.0, 38.4, 38.4, 38.4, 38.4, 38.4, 38.4, 38.4]
        events = detect_step_events(pd.Series(values, index=index), 1.0, min_gap_days=5)
        assert len(events) == 1, "a two-day adjustment must count once"

    def test_no_events_on_a_flat_series(self):
        index = pd.date_range("2024-01-01", periods=20, freq="B")
        assert detect_step_events(pd.Series(36.0, index=index), 1.0).empty

    def test_regimes_use_the_configured_labels(self, synthetic_frame):
        regimes = volatility_regimes(synthetic_frame["USD_MRU"], 30)
        assert set(regimes.dropna().unique()) <= {"low", "normal", "high"}

    def test_regimes_are_trailing_not_centred(self):
        """A trailing window cannot see a volatility spike before it happens.

        The series is calm for 400 days and then 60x more volatile.  With a
        *centred* window the last calm days would already be labelled "high";
        with a trailing one they cannot be.
        """
        rng = np.random.default_rng(4)
        index = pd.date_range("2020-01-01", periods=600, freq="B")
        returns = np.concatenate([rng.normal(0, 0.0005, 400), rng.normal(0, 0.03, 200)])
        series = pd.Series(36 * np.exp(np.cumsum(returns)), index=index)
        regimes = volatility_regimes(series, 30)

        assert "high" not in set(regimes.iloc[:400].dropna()), (
            "a date inside the calm period was labelled high-volatility"
        )
        assert (regimes.iloc[450:] == "high").all(), "the spike must be labelled high"

    def test_gap_summary_counts_weekends(self, synthetic_frame):
        gaps = summarise_gaps(synthetic_frame["USD_MRU"])
        assert set(gaps["gap_days"]) <= {1, 3}  # business days: Mon-Fri, then Fri->Mon
        assert gaps["n"].sum() == len(synthetic_frame) - 1
