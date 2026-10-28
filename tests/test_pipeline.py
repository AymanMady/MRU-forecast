"""Phases 12-14: the end-to-end pipeline, persistence and the served forecast."""

from __future__ import annotations

import pandas as pd
import pytest

from mru_forecast.pipeline.forecasting_pipeline import (
    ForecastingPipeline,
    build_model,
    log_to_mlflow,
)


class TestModelRegistry:
    @pytest.mark.parametrize(
        "name", ["naive", "drift", "seasonal_naive", "mean_reversion", "xgboost"]
    )
    def test_every_registered_model_can_be_built(self, name, config):
        assert build_model(name, config).name == name

    def test_unknown_model_raises(self, config):
        with pytest.raises(KeyError, match="unknown model"):
            build_model("nope", config)


@pytest.fixture(scope="module")
def small_pipeline(synthetic_frame, config):
    """A cheap pipeline: naive base model, three horizons, small calibration block."""
    fast = {
        **config,
        "pipeline": {**config["pipeline"], "model": "naive", "calibration_size": 200},
    }
    # A contiguous 1..7 grid, as production uses (1..max_horizon): the served
    # path is only dense if every step in between was trained.
    return ForecastingPipeline(config=fast).train(
        synthetic_frame, targets=["USD_MRU"], horizons=list(range(1, 8))
    )


class TestTraining:
    def test_one_predictor_per_step(self, small_pipeline):
        assert set(small_pipeline.steps) == {("USD_MRU", h) for h in range(1, 8)}

    def test_training_stops_before_the_calibration_block(self, small_pipeline):
        for step in small_pipeline.steps.values():
            assert step.train_end < step.calibration_start
            assert step.n_calibration == 200

    def test_longer_horizons_train_on_less_data(self, small_pipeline):
        """Calibration needs observed outcomes, so a longer horizon cuts back further."""
        sizes = [small_pipeline.steps[("USD_MRU", h)].n_train for h in range(1, 8)]
        assert sizes == sorted(sizes, reverse=True)

    def test_metadata_records_the_dataset_version(self, small_pipeline, synthetic_frame):
        metadata = small_pipeline.metadata
        assert metadata["data_last_date"] == str(synthetic_frame.index.max().date())
        assert metadata["n_observations"] == len(synthetic_frame)
        assert metadata["model"] == "naive"

    def test_calibration_summary_has_a_quantile_per_level(self, small_pipeline):
        summary = small_pipeline.calibration_summary()
        assert len(summary) == len(small_pipeline.steps)
        assert {"q_80", "q_90", "q_95"} <= set(summary.columns)
        assert (summary["q_80"] <= summary["q_90"]).all()
        assert (summary["q_90"] <= summary["q_95"]).all()


class TestForecastPath:
    def test_path_is_a_row_per_business_day(self, small_pipeline, synthetic_frame):
        path = small_pipeline.forecast(synthetic_frame, target="USD_MRU", horizon=3)
        assert len(path) == 3
        assert list(path["days_ahead"]) == [1, 2, 3]
        assert (path.index.dayofweek < 5).all()

    def test_every_forecast_date_is_in_the_future(self, small_pipeline, synthetic_frame):
        path = small_pipeline.forecast(synthetic_frame, target="USD_MRU", horizon=7)
        assert (path.index > synthetic_frame.index[-1]).all()
        assert path.index.is_monotonic_increasing

    def test_bounds_bracket_the_prediction(self, small_pipeline, synthetic_frame):
        path = small_pipeline.forecast(synthetic_frame, target="USD_MRU", horizon=7)
        for level in (80, 90, 95):
            assert (path[f"lower_{level}"] <= path["prediction"] + 1e-9).all()
            assert (path["prediction"] <= path[f"upper_{level}"] + 1e-9).all()
            assert (path[f"lower_{level}"] >= 0).all()

    def test_intervals_widen_with_the_horizon(self, small_pipeline, synthetic_frame):
        path = small_pipeline.forecast(synthetic_frame, target="USD_MRU", horizon=7)
        widths = (path["upper_90"] - path["lower_90"]).to_numpy()
        assert widths[-1] > widths[0], "uncertainty must grow further into the future"

    def test_anchor_is_the_last_observation(self, small_pipeline, synthetic_frame):
        path = small_pipeline.forecast(synthetic_frame, target="USD_MRU", horizon=3)
        assert path["anchor_value"].iloc[0] == pytest.approx(
            float(synthetic_frame["USD_MRU"].iloc[-1])
        )
        assert path["anchor_date"].iloc[0] == synthetic_frame.index[-1]

    def test_unknown_target_raises(self, small_pipeline, synthetic_frame):
        with pytest.raises(KeyError):
            small_pipeline.forecast(synthetic_frame, target="EUR_MRU")

    def test_a_sparse_horizon_grid_yields_a_sparse_path(self, synthetic_frame, config):
        """Only trained steps are served, and `days_ahead` makes the gaps visible."""
        fast = {
            **config,
            "pipeline": {**config["pipeline"], "model": "naive", "calibration_size": 200},
        }
        sparse = ForecastingPipeline(config=fast).train(
            synthetic_frame, targets=["USD_MRU"], horizons=[1, 5, 10]
        )
        path = sparse.forecast(synthetic_frame, target="USD_MRU", horizon=10)
        assert list(path["days_ahead"]) == [1, 5, 10]

    def test_horizon_beyond_what_was_trained_is_capped(self, small_pipeline, synthetic_frame):
        path = small_pipeline.forecast(synthetic_frame, target="USD_MRU", horizon=999)
        assert int(path["days_ahead"].max()) == 7

    def test_untrained_pipeline_raises(self, config):
        with pytest.raises(RuntimeError, match="not trained"):
            ForecastingPipeline(config=config).forecast(target="USD_MRU")


class TestPersistence:
    def test_round_trip_preserves_the_forecast(self, small_pipeline, synthetic_frame, tmp_path):
        before = small_pipeline.forecast(synthetic_frame, target="USD_MRU", horizon=3)
        small_pipeline.save(tmp_path)
        reloaded = ForecastingPipeline.load(tmp_path, config=small_pipeline.config)
        after = reloaded.forecast(synthetic_frame, target="USD_MRU", horizon=3)
        pd.testing.assert_frame_equal(before, after)

    def test_metadata_survives_the_round_trip(self, small_pipeline, tmp_path):
        small_pipeline.save(tmp_path)
        reloaded = ForecastingPipeline.load(tmp_path, config=small_pipeline.config)
        assert reloaded.metadata["model"] == small_pipeline.metadata["model"]
        assert len(reloaded.steps) == len(small_pipeline.steps)

    def test_loading_from_an_empty_directory_raises(self, tmp_path, config):
        with pytest.raises(FileNotFoundError, match=r"scripts/train\.py"):
            ForecastingPipeline.load(tmp_path / "empty", config=config)


class TestMlflow:
    def test_logging_never_breaks_training(self, small_pipeline, tmp_path):
        """Tracking is best-effort: an unreachable backend must not raise."""
        broken = {
            **small_pipeline.config,
            "mlflow": {
                "tracking_uri": f"file:{tmp_path / 'mlruns'}",
                "experiment_name": "pytest-mru",
            },
        }
        pipeline = ForecastingPipeline(
            config=broken, steps=small_pipeline.steps, metadata=small_pipeline.metadata
        )
        run_id = log_to_mlflow(pipeline, run_name="pytest")
        assert run_id is None or isinstance(run_id, str)


@pytest.mark.slow
class TestRealArtifacts:
    def test_every_configured_horizon_is_trained(self, trained_pipeline, config):
        for target in (t["name"] for t in config["targets"]):
            for horizon in config["horizons"]:
                assert (target, horizon) in trained_pipeline.steps

    @pytest.mark.parametrize("target", ["USD_MRU", "EUR_MRU"])
    def test_served_forecast_is_coherent(self, trained_pipeline, processed_frame, target):
        path = trained_pipeline.forecast(processed_frame, target=target, horizon=30)
        assert len(path) == 30
        assert (path["prediction"] > 0).all()
        assert (path["lower_90"] <= path["prediction"]).all()
        assert (path["prediction"] <= path["upper_90"]).all()

    def test_forecast_stays_within_a_plausible_band(
        self, trained_pipeline, processed_frame, config
    ):
        """A 30-day forecast that leaves the historical band would be a red flag."""
        low, high = config["validation"]["plausible_range"]["USD_MRU"]
        path = trained_pipeline.forecast(processed_frame, target="USD_MRU", horizon=30)
        assert path["prediction"].between(low, high).all()

    def test_feature_importance_is_available_for_xgboost(self, trained_pipeline, config):
        if trained_pipeline.metadata.get("model") != "xgboost":
            pytest.skip("production model is not xgboost")
        importance = trained_pipeline.feature_importance("USD_MRU", 1, 10)
        assert len(importance) == 10
        assert importance["gain"].is_monotonic_decreasing
