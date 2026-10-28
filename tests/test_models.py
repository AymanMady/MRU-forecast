"""Phases 6-9: forecasters, the origin protocol and the backtest splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mru_forecast.evaluation.backtesting import (
    make_splits,
    run_backtest,
    splits_summary,
)
from mru_forecast.models.base import Forecaster, NotFittedError
from mru_forecast.models.baseline import (
    DriftForecaster,
    MeanReversionForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    build_baselines,
)
from mru_forecast.models.xgboost_model import XGBoostForecaster

ALL_BASELINES = [NaiveForecaster, SeasonalNaiveForecaster, DriftForecaster, MeanReversionForecaster]


@pytest.fixture(scope="module")
def split_dates(synthetic_frame):
    index = synthetic_frame.index
    return index[-400], index[-399:-100], index[-100:-40]


class TestForecasterContract:
    @pytest.mark.parametrize("factory", ALL_BASELINES)
    def test_predict_before_fit_raises(self, factory, synthetic_frame):
        with pytest.raises(NotFittedError):
            factory().predict_origins(synthetic_frame, synthetic_frame.index[-5:])

    @pytest.mark.parametrize("factory", ALL_BASELINES)
    def test_returns_one_finite_level_per_origin(
        self, factory, synthetic_frame, config, split_dates
    ):
        train_end, _, test = split_dates
        model = factory().fit(synthetic_frame, "USD_MRU", 7, config, train_end=train_end)
        predictions = model.predict_origins(synthetic_frame, test)
        assert len(predictions) == len(test)
        assert predictions.index.equals(test)
        assert np.isfinite(predictions.to_numpy()).all()
        assert (predictions > 0).all(), "an exchange rate forecast must be positive"

    @pytest.mark.parametrize("factory", ALL_BASELINES)
    def test_unknown_origin_raises(self, factory, synthetic_frame, config, split_dates):
        train_end, _, _ = split_dates
        model = factory().fit(synthetic_frame, "USD_MRU", 1, config, train_end=train_end)
        with pytest.raises(KeyError):
            model.predict_origins(synthetic_frame, pd.DatetimeIndex([pd.Timestamp("1999-01-04")]))

    def test_unknown_target_raises(self, synthetic_frame, config):
        with pytest.raises(KeyError):
            NaiveForecaster().fit(synthetic_frame, "NOPE", 1, config)

    def test_non_positive_horizon_raises(self, synthetic_frame, config):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            NaiveForecaster().fit(synthetic_frame, "USD_MRU", 0, config)

    @pytest.mark.parametrize("factory", ALL_BASELINES)
    def test_prediction_uses_no_information_after_the_origin(
        self, factory, synthetic_frame, config, split_dates
    ):
        """Overwrite everything after an origin; the forecast must not move."""
        train_end, _, test = split_dates
        origin = test[10]
        model = factory().fit(synthetic_frame, "USD_MRU", 5, config, train_end=train_end)
        before = model.predict_origins(synthetic_frame, pd.DatetimeIndex([origin])).iloc[0]

        tampered = synthetic_frame.copy()
        after_origin = tampered.index > origin
        tampered.loc[after_origin, "USD_MRU"] *= 3.0
        after = model.predict_origins(tampered, pd.DatetimeIndex([origin])).iloc[0]
        assert np.isclose(before, after), f"{factory.__name__} reads data after the origin"


class TestBaselineSemantics:
    def test_naive_returns_the_value_at_the_origin(self, synthetic_frame, config, split_dates):
        train_end, _, test = split_dates
        model = NaiveForecaster().fit(synthetic_frame, "USD_MRU", 7, config, train_end=train_end)
        predictions = model.predict_origins(synthetic_frame, test)
        assert np.allclose(predictions, synthetic_frame["USD_MRU"].reindex(test))

    def test_seasonal_naive_looks_back_whole_seasons(self, config):
        index = pd.date_range("2024-01-01", periods=40, freq="B")
        series = pd.Series(np.arange(40, dtype="float64") + 30.0, index=index)
        frame = pd.DataFrame({"USD_MRU": series})
        model = SeasonalNaiveForecaster(5).fit(frame, "USD_MRU", 7, config)
        origin = index[20]
        # h=7 with a season of 5 -> step back 2 seasons: 7 - 10 = -3 rows.
        assert model.predict_origins(frame, pd.DatetimeIndex([origin])).iloc[0] == series.iloc[17]

    def test_seasonal_naive_rejects_a_zero_season(self):
        with pytest.raises(ValueError):
            SeasonalNaiveForecaster(0)

    def test_drift_extends_a_trend(self, config):
        index = pd.date_range("2020-01-01", periods=400, freq="B")
        series = pd.Series(30.0 * np.exp(np.arange(400) * 0.0005), index=index)
        frame = pd.DataFrame({"USD_MRU": series})
        model = DriftForecaster(252).fit(frame, "USD_MRU", 10, config)
        origin = index[-1]
        prediction = model.predict_origins(frame, pd.DatetimeIndex([origin])).iloc[0]
        assert prediction > series.iloc[-1], "an upward drift must forecast upward"

    def test_mean_reversion_learns_no_reversion_on_a_random_walk(self, config):
        rng = np.random.default_rng(3)
        index = pd.date_range("2020-01-01", periods=900, freq="B")
        series = pd.Series(36.0 * np.exp(np.cumsum(rng.normal(0, 0.003, 900))), index=index)
        model = MeanReversionForecaster(21).fit(
            pd.DataFrame({"USD_MRU": series}), "USD_MRU", 1, config
        )
        assert abs(model.kappa) < 0.6, "no strong reversion should be found in a random walk"

    def test_build_baselines_respects_the_config(self, config):
        names = {m.name for m in build_baselines(config)}
        assert {"naive", "seasonal_naive", "drift"} <= names


class TestXGBoost:
    @pytest.mark.slow
    def test_fits_and_predicts_positive_levels(self, synthetic_frame, config, split_dates):
        train_end, _, test = split_dates
        model = XGBoostForecaster().fit(synthetic_frame, "USD_MRU", 7, config, train_end=train_end)
        predictions = model.predict_origins(synthetic_frame, test)
        assert (predictions > 0).all()
        assert np.isfinite(predictions).all()

    @pytest.mark.slow
    def test_feature_importance_is_normalised(self, synthetic_frame, config, split_dates):
        train_end, _, _ = split_dates
        model = XGBoostForecaster().fit(synthetic_frame, "USD_MRU", 1, config, train_end=train_end)
        importance = model.feature_importance()
        assert not importance.empty
        assert importance["gain"].sum() == pytest.approx(1.0, abs=1e-6)
        assert importance["gain"].is_monotonic_decreasing

    @pytest.mark.slow
    def test_training_ignores_data_after_train_end(self, synthetic_frame, config, split_dates):
        """Corrupting the post-train_end rows must not change what was learnt."""
        train_end, _, _test = split_dates
        clean = XGBoostForecaster().fit(synthetic_frame, "USD_MRU", 7, config, train_end=train_end)
        tampered_frame = synthetic_frame.copy()
        tampered_frame.loc[tampered_frame.index > train_end, "USD_MRU"] *= 5.0
        tampered = XGBoostForecaster().fit(
            tampered_frame, "USD_MRU", 7, config, train_end=train_end
        )
        subset = synthetic_frame.loc[:train_end].index[-20:]
        assert np.allclose(
            clean.predict_origins(synthetic_frame, subset),
            tampered.predict_origins(synthetic_frame, subset),
        )

    def test_too_little_data_raises_clearly(self, synthetic_frame, config):
        tiny = synthetic_frame.iloc[:60]
        with pytest.raises(ValueError, match="training rows"):
            XGBoostForecaster().fit(tiny, "USD_MRU", 30, config)


class TestSplits:
    def test_blocks_are_chronological_and_disjoint(self, synthetic_frame, config):
        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), config, max_horizon=30)
        for split in splits:
            assert split.train[-1] < split.calibration[0] < split.calibration[-1] < split.test[0]
            assert not set(split.train) & set(split.calibration)
            assert not set(split.calibration) & set(split.test)
            assert not set(split.train) & set(split.test)

    def test_every_test_origin_has_a_realised_outcome(self, synthetic_frame, config):
        horizon = max(config["horizons"])
        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), config, max_horizon=horizon)
        index = synthetic_frame.index
        for split in splits:
            last_position = index.get_loc(split.test[-1])
            assert last_position + horizon < len(index)

    def test_splits_move_forward_in_time(self, synthetic_frame, config):
        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), config, max_horizon=30)
        starts = [s.test[0] for s in splits]
        assert starts == sorted(starts)
        assert [s.index for s in splits] == list(range(len(splits)))

    def test_expanding_scheme_grows_the_training_set(self, synthetic_frame, config):
        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), config, max_horizon=30)
        sizes = [len(s.train) for s in splits]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_rolling_scheme_keeps_the_training_set_bounded(self, synthetic_frame, config):
        rolling = {**config, "backtesting": {**config["backtesting"], "scheme": "rolling"}}
        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), rolling, max_horizon=30)
        sizes = {len(s.train) for s in splits}
        assert len(sizes) == 1

    def test_too_little_data_raises(self, config):
        index = pd.date_range("2024-01-01", periods=100, freq="B")
        with pytest.raises(ValueError, match="no valid split"):
            make_splits(index, config, max_horizon=30)

    def test_summary_has_one_row_per_split(self, synthetic_frame, config):
        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), config, max_horizon=30)
        assert len(splits_summary(splits)) == len(splits)


class TestBacktestRunner:
    @pytest.mark.slow
    def test_produces_one_row_per_origin_and_model(self, synthetic_frame, config):
        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), config, max_horizon=7)[:2]
        factories = {"naive": NaiveForecaster, "drift": DriftForecaster}
        results = run_backtest(
            synthetic_frame,
            config,
            factories,
            targets=["USD_MRU"],
            horizons=[7],
            splits=splits,
            progress=False,
        )
        expected = sum(len(s.test) for s in splits) * len(factories)
        assert len(results) == expected
        assert set(results.predictions["model"]) == set(factories)
        assert results.interval_levels() == [80, 90, 95]

    @pytest.mark.slow
    def test_intervals_are_ordered_everywhere(self, synthetic_frame, config):
        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), config, max_horizon=7)[:2]
        results = run_backtest(
            synthetic_frame,
            config,
            {"naive": NaiveForecaster},
            targets=["USD_MRU"],
            horizons=[7],
            splits=splits,
            progress=False,
        )
        frame = results.predictions
        for level in results.interval_levels():
            assert (frame[f"lower_{level}"] <= frame["prediction"] + 1e-9).all()
            assert (frame["prediction"] <= frame[f"upper_{level}"] + 1e-9).all()

    @pytest.mark.slow
    def test_wider_levels_are_nested(self, synthetic_frame, config):
        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), config, max_horizon=7)[:1]
        frame = run_backtest(
            synthetic_frame,
            config,
            {"naive": NaiveForecaster},
            targets=["USD_MRU"],
            horizons=[7],
            splits=splits,
            progress=False,
        ).predictions
        assert (frame["lower_95"] <= frame["lower_90"] + 1e-9).all()
        assert (frame["upper_95"] >= frame["upper_90"] - 1e-9).all()
        assert (frame["lower_90"] <= frame["lower_80"] + 1e-9).all()

    @pytest.mark.slow
    def test_a_failing_model_does_not_sink_the_run(self, synthetic_frame, config):
        class Broken(Forecaster):
            name = "broken"

            def _fit(self, frame):
                raise RuntimeError("intentional")

            def _predict_origins(self, frame, origins):
                raise RuntimeError("intentional")

        splits = make_splits(pd.DatetimeIndex(synthetic_frame.index), config, max_horizon=7)[:1]
        results = run_backtest(
            synthetic_frame,
            config,
            {"naive": NaiveForecaster, "broken": Broken},
            targets=["USD_MRU"],
            horizons=[7],
            splits=splits,
            progress=False,
        )
        assert set(results.predictions["model"]) == {"naive"}
