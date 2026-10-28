"""Phase 5: feature engineering, with data leakage as the central concern.

The decisive test is :meth:`TestNoLeakage.test_features_are_identical_when_the_future_is_removed`:
it recomputes every feature on a truncated history and requires the shared rows
to be bit-identical.  Any feature that peeks forward changes, and the test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mru_forecast.features.engineering import (
    NON_EXTRAPOLABLE,
    add_calendar_features,
    add_lag_features,
    build_features,
    forward_transform,
    future_business_days,
    inverse_transform,
    latest_feature_row,
    make_supervised,
)


class TestNoLeakage:
    @pytest.mark.parametrize("cut_back", [1, 5, 40, 200])
    def test_features_are_identical_when_the_future_is_removed(
        self, synthetic_frame, config, cut_back
    ):
        full = build_features(synthetic_frame, "USD_MRU", config)
        cut = synthetic_frame.index[-cut_back - 1]
        truncated = build_features(synthetic_frame.loc[:cut], "USD_MRU", config)

        shared = truncated.index
        difference = (full.loc[shared] - truncated).abs()
        offenders = difference.max()[difference.max() > 1e-12].dropna()
        assert offenders.empty, f"these features read the future: {list(offenders.index)}"

    def test_label_is_exactly_the_future_log_return(self, synthetic_frame, config):
        dataset = make_supervised(synthetic_frame, "USD_MRU", 7, config)
        series = synthetic_frame["USD_MRU"]
        for position in (0, 50, len(dataset) - 1):
            origin = dataset.index[position]
            target_date = dataset.target_dates[position]
            expected = np.log(series.loc[target_date]) - np.log(series.loc[origin])
            assert np.isclose(dataset.y.iloc[position], expected)

    def test_target_date_is_exactly_horizon_rows_ahead(self, synthetic_frame, config):
        horizon = 5
        dataset = make_supervised(synthetic_frame, "USD_MRU", horizon, config)
        index = synthetic_frame.index
        for position in (0, 100):
            origin_position = index.get_loc(dataset.index[position])
            assert index[origin_position + horizon] == dataset.target_dates[position]

    def test_rolling_windows_are_trailing_not_centred(self, synthetic_frame, config):
        """A centred window is the classic silent leak; assert the value directly."""
        features = build_features(synthetic_frame, "USD_MRU", config)
        series = synthetic_frame["USD_MRU"]
        position = 500
        date = series.index[position]
        expected = series.iloc[position - 6 : position + 1].mean()
        assert np.isclose(features["USD_MRU_roll_mean_7"].loc[date], expected)

    def test_lag_zero_is_rejected(self, synthetic_frame):
        with pytest.raises(ValueError, match="lags must be >= 1"):
            add_lag_features(synthetic_frame["USD_MRU"], [0, 1], "USD_MRU")

    def test_auxiliary_lag_zero_is_rejected(self, synthetic_frame, config):
        broken = {**config, "features": {**config["features"], "auxiliary_lags": [0]}}
        with pytest.raises(ValueError, match="auxiliary_lags"):
            build_features(synthetic_frame, "USD_MRU", broken)

    def test_auxiliary_series_are_lagged(self, synthetic_frame, config):
        """No auxiliary column may equal its own same-day value."""
        features = build_features(synthetic_frame, "USD_MRU", config)
        for name in ("EUR_USD", "BRENT_OIL", "GOLD"):
            same_day = [
                column
                for column in features.columns
                if column.startswith(name)
                and features[column].equals(synthetic_frame[name].rename(column))
            ]
            assert not same_day, f"{name} appears un-lagged: {same_day}"


class TestFeatureContent:
    def test_calendar_features_are_cyclical_where_it_matters(self, business_index):
        calendar = add_calendar_features(business_index, ["day_of_week", "month"])
        assert {"dow_sin", "dow_cos", "month_sin", "month_cos"} <= set(calendar.columns)
        assert np.allclose(calendar["month_sin"] ** 2 + calendar["month_cos"] ** 2, 1.0)

    def test_year_is_excluded_from_the_model_matrix(self, synthetic_frame, config):
        features = build_features(synthetic_frame, "USD_MRU", config)
        assert not (NON_EXTRAPOLABLE & set(features.columns)), (
            "absolute-time features cannot generalise past the training period"
        )

    def test_year_can_be_opted_into_explicitly(self, synthetic_frame, config):
        features = build_features(synthetic_frame, "USD_MRU", config, include_non_extrapolable=True)
        assert "year" in features.columns

    def test_days_since_move_counts_flat_days(self):
        index = pd.date_range("2024-01-01", periods=8, freq="B")
        series = pd.Series([36.0, 36.0, 36.0, 36.5, 36.5, 36.5, 36.5, 37.0], index=index)
        from mru_forecast.features.engineering import add_return_features

        feature = add_return_features(series, "X", [1])["X_days_since_move"]
        assert feature.iloc[2] == 2  # two flat days after the first move
        assert feature.iloc[3] == 0  # the rate moved
        assert feature.iloc[6] == 3

    def test_rolling_features_have_no_infinities(self, synthetic_frame, config):
        features = build_features(synthetic_frame, "USD_MRU", config)
        assert not np.isinf(features.select_dtypes("number").to_numpy()).any()

    def test_unknown_target_raises(self, synthetic_frame, config):
        with pytest.raises(KeyError):
            build_features(synthetic_frame, "NOT_A_TARGET", config)


class TestTargetTransform:
    def test_transforms_round_trip(self):
        anchor = np.array([36.0, 40.0, 45.0])
        level = np.array([36.5, 39.5, 46.0])
        model_space = forward_transform(level, anchor, "log_return")
        assert np.allclose(inverse_transform(model_space, anchor, "log_return"), level)

    def test_level_transform_is_identity(self):
        values = np.array([1.0, 2.0])
        assert np.allclose(inverse_transform(values, np.array([9.0, 9.0]), "level"), values)

    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="unknown transform"):
            inverse_transform(np.array([1.0]), np.array([1.0]), "nope")

    def test_log_return_target_lets_the_forecast_exceed_the_training_range(self):
        """The reason the target is a return and not a level."""
        anchor = np.array([100.0])
        assert inverse_transform(np.array([0.05]), anchor, "log_return")[0] > 100.0


class TestSupervisedDataset:
    def test_horizon_must_be_positive(self, synthetic_frame, config):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            make_supervised(synthetic_frame, "USD_MRU", 0, config)

    def test_last_rows_without_a_label_are_dropped(self, synthetic_frame, config):
        horizon = 10
        dataset = make_supervised(synthetic_frame, "USD_MRU", horizon, config)
        assert dataset.index.max() <= synthetic_frame.index[-horizon - 1]

    def test_slice_keeps_everything_aligned(self, synthetic_frame, config):
        dataset = make_supervised(synthetic_frame, "USD_MRU", 3, config)
        piece = dataset.slice_by_position(10, 20)
        assert len(piece) == 10
        assert piece.X.index.equals(piece.y.index)
        assert len(piece.target_dates) == 10

    def test_latest_row_is_the_last_date(self, synthetic_frame, config):
        row, anchor, anchor_date = latest_feature_row(synthetic_frame, "USD_MRU", config)
        assert len(row) == 1
        assert anchor_date == synthetic_frame.index[-1]
        assert anchor == pytest.approx(float(synthetic_frame["USD_MRU"].iloc[-1]))

    def test_future_business_days_skip_weekends(self):
        days = future_business_days(pd.Timestamp("2026-09-11"), 3)  # a Friday
        assert list(days.dayofweek) == [0, 1, 2]
        assert days[0] == pd.Timestamp("2026-09-14")


@pytest.mark.parametrize("target", ["USD_MRU", "EUR_MRU"])
def test_no_leakage_on_the_real_dataset(processed_frame, config, target):
    full = build_features(processed_frame, target, config)
    cut = processed_frame.index[-60]
    truncated = build_features(processed_frame.loc[:cut], target, config)
    difference = (full.loc[truncated.index] - truncated).abs().max()
    offenders = difference[difference > 1e-12].dropna()
    assert offenders.empty, f"leaking features: {list(offenders.index)}"
