"""Phase 3: cleaning and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mru_forecast.data.cleaning import (
    build_calendar,
    clean_series,
    deduplicate_and_sort,
    detect_isolated_spikes,
    detect_out_of_band,
    detect_rolling_median_outliers,
    project_on_calendar,
)
from mru_forecast.data.validation import (
    DataValidationError,
    Severity,
    check_gaps,
    check_index,
    check_values,
    validate_forecast_intervals,
    validate_series,
)


class TestDeduplication:
    def test_duplicates_are_dropped_and_order_restored(self):
        index = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"])
        series = pd.Series([3.0, 1.0, 2.0, 2.5], index=index)
        cleaned, dropped = deduplicate_and_sort(series, keep="last")
        assert dropped == 1
        assert cleaned.index.is_monotonic_increasing
        assert cleaned.loc["2024-01-02"] == 2.5

    def test_keep_first_keeps_the_first(self):
        index = pd.to_datetime(["2024-01-02", "2024-01-02"])
        cleaned, _ = deduplicate_and_sort(pd.Series([2.0, 2.5], index=index), keep="first")
        assert cleaned.iloc[0] == 2.0


class TestSpikeDetection:
    def test_injected_bad_ticks_are_flagged(self, series_with_bad_ticks):
        corrupted, dates = series_with_bad_ticks
        flags = detect_isolated_spikes(corrupted, 4.0)
        for date in dates:
            assert flags.loc[date], f"bad tick at {date.date()} was not detected"

    def test_a_genuine_step_survives(self):
        """A real devaluation deviates from the previous value, not the next one."""
        index = pd.date_range("2024-01-01", periods=12, freq="B")
        values = [36.0] * 5 + [38.0] * 7  # a permanent +5.6% step on day 6
        flags = detect_isolated_spikes(pd.Series(values, index=index), 4.0)
        assert not flags.any(), "a permanent step must not be treated as a bad tick"

    def test_clean_series_is_a_noop_on_clean_data(self, managed_series, config):
        flags = detect_isolated_spikes(managed_series, 4.0)
        assert flags.sum() == 0

    def test_out_of_band_values_are_flagged_regardless_of_neighbours(self):
        index = pd.date_range("2024-01-01", periods=5, freq="B")
        series = pd.Series([36.0, 36.1, 500.0, 36.2, 36.3], index=index)
        assert detect_out_of_band(series, (10.0, 120.0)).sum() == 1

    def test_no_band_flags_nothing(self, managed_series):
        assert not detect_out_of_band(managed_series, None).any()

    def test_rolling_median_guard_catches_a_corrupted_run(self, managed_series):
        corrupted = managed_series.copy()
        corrupted.iloc[500:505] = corrupted.iloc[500] * 0.7  # a 30% run, clearly wrong
        flags = detect_rolling_median_outliers(corrupted, 21, 12.0)
        assert flags.iloc[500:505].sum() >= 3


class TestCalendarAndFilling:
    def test_business_calendar_has_no_weekends(self, managed_series):
        calendar = build_calendar(managed_series.index, "business_day")
        assert (calendar.dayofweek < 5).all()

    def test_forward_fill_is_bounded(self):
        index = pd.date_range("2024-01-01", periods=20, freq="B")
        series = pd.Series(np.nan, index=index)
        series.iloc[0] = 36.0
        filled, n_filled = project_on_calendar(series, index, max_ffill_days=3)
        assert n_filled == 3
        assert filled.iloc[4:].isna().all(), "ffill must stop at the configured limit"

    def test_filling_never_looks_backwards(self):
        """The first rows stay missing: back-filling would leak the future."""
        index = pd.date_range("2024-01-01", periods=10, freq="B")
        series = pd.Series([np.nan] * 5 + [36.0] * 5, index=index)
        filled, _ = project_on_calendar(series, index, max_ffill_days=10)
        assert filled.iloc[:5].isna().all()


class TestCleanSeries:
    def test_bad_ticks_removed_and_reported(self, series_with_bad_ticks, config):
        corrupted, dates = series_with_bad_ticks
        cleaned, stats, flags = clean_series(corrupted, "USD_MRU", config)
        assert stats.n_spikes_masked >= 2
        assert set(str(d.date()) for d in dates) <= set(str(d.date()) for d in flags["date"])
        assert cleaned.min() > 10.0, "an impossible value survived cleaning"

    def test_output_is_on_a_continuous_calendar(self, managed_series, config):
        cleaned, stats, _ = clean_series(managed_series, "USD_MRU", config)
        assert cleaned.index.is_monotonic_increasing
        assert not cleaned.index.has_duplicates
        assert stats.coverage_pct > 95.0

    def test_flags_record_a_reason(self, series_with_bad_ticks, config):
        corrupted, _ = series_with_bad_ticks
        _, _, flags = clean_series(corrupted, "USD_MRU", config)
        assert set(flags["reason"]) <= {
            "out_of_plausible_band",
            "isolated_reverting_spike",
            "rolling_median_outlier",
        }


class TestValidation:
    def test_duplicate_dates_are_an_error(self):
        index = pd.to_datetime(["2024-01-01", "2024-01-01"])
        report = check_index(pd.Series([1.0, 2.0], index=index), "x", {})
        assert any(
            i.code == "DUPLICATE_DATES" and i.severity is Severity.ERROR for i in report.issues
        )

    def test_unsorted_dates_are_an_error(self):
        index = pd.to_datetime(["2024-01-02", "2024-01-01"])
        report = check_index(pd.Series([1.0, 2.0], index=index), "x", {})
        assert any(i.code == "UNSORTED_DATES" for i in report.issues)

    def test_non_positive_rate_is_an_error(self, managed_series):
        broken = managed_series.copy()
        broken.iloc[10] = -1.0
        report = check_values(broken, "USD_MRU", {"min_rate": 0.0})
        assert not report.ok
        assert any(i.code == "NON_POSITIVE_RATE" for i in report.issues)

    def test_value_outside_the_plausible_band_is_an_error(self, managed_series):
        broken = managed_series.copy()
        broken.iloc[10] = 2.0
        report = check_values(
            broken, "USD_MRU", {"min_rate": 0.0, "plausible_range": {"USD_MRU": [10.0, 120.0]}}
        )
        assert any(i.code == "OUTSIDE_PLAUSIBLE_RANGE" for i in report.issues)

    def test_a_long_gap_is_a_warning_not_an_error(self, managed_series):
        gapped = managed_series.drop(managed_series.index[100:130])
        report = check_gaps(gapped, "USD_MRU", {"max_consecutive_missing_days": 10})
        assert report.ok, "a gap must not invalidate the dataset"
        assert any(i.code == "LONG_GAP" and i.severity is Severity.WARNING for i in report.issues)

    def test_clean_series_passes_validation(self, managed_series, config):
        assert validate_series(managed_series, "USD_MRU", config).ok

    def test_raise_if_failed_raises_only_on_errors(self, managed_series):
        broken = managed_series.copy()
        broken.iloc[5] = 0.0
        report = check_values(broken, "USD_MRU", {"min_rate": 0.0})
        with pytest.raises(DataValidationError):
            report.raise_if_failed()


class TestIntervalValidation:
    def test_ordered_interval_passes(self):
        report = validate_forecast_intervals([40.0, 41.0], [39.0, 40.0], [41.0, 42.0])
        assert report.ok

    def test_inverted_interval_is_an_error(self):
        report = validate_forecast_intervals([40.0], [41.0], [39.0])
        assert any(i.code == "INVERTED_INTERVAL" for i in report.issues)
        assert not report.ok

    def test_prediction_outside_its_own_interval_is_an_error(self):
        report = validate_forecast_intervals([45.0], [39.0], [41.0])
        assert any(i.code == "PREDICTION_OUTSIDE_INTERVAL" for i in report.issues)

    def test_negative_lower_bound_is_flagged(self):
        report = validate_forecast_intervals([1.0], [-1.0], [3.0])
        assert any(i.code == "NON_POSITIVE_LOWER_BOUND" for i in report.issues)

    def test_shape_mismatch_is_an_error(self):
        report = validate_forecast_intervals([1.0, 2.0], [0.0], [3.0])
        assert not report.ok


@pytest.mark.parametrize("target", ["USD_MRU", "EUR_MRU"])
class TestRealProcessedData:
    def test_targets_are_complete_and_positive(self, processed_frame, target):
        series = processed_frame[target]
        assert series.notna().all(), "a target must have no gaps after cleaning"
        assert (series > 0).all()

    def test_no_implausible_value_survives(self, processed_frame, config, target):
        low, high = config["validation"]["plausible_range"][target]
        assert processed_frame[target].between(low, high).all()

    def test_no_extreme_daily_move_survives(self, processed_frame, target):
        """After cleaning, no day-on-day move should exceed the configured limit."""
        changes = processed_frame[target].pct_change().abs() * 100
        assert changes.max() < 10.0, f"largest surviving move: {changes.max():.2f}%"
