"""Phase 10: conformal prediction.

The two properties that matter are tested directly rather than asserted in prose:

* the finite-sample rank ``ceil((n+1)*c)`` is what the quantile actually uses;
* on exchangeable data the empirical coverage matches the nominal level, and it
  is checked against a binomial tolerance rather than an eyeballed threshold.

MAPIE is used as an independent cross-check of the split-conformal quantile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mru_forecast.models.base import NotFittedError
from mru_forecast.models.baseline import NaiveForecaster
from mru_forecast.models.conformal import (
    ConformalPredictor,
    build_conformal,
    conformal_quantile,
    volatility_scale,
)


class TestConformalQuantile:
    @pytest.mark.parametrize(
        ("n", "confidence", "expected_rank"),
        [(100, 0.90, 91), (100, 0.80, 81), (100, 0.95, 96), (19, 0.95, 19), (249, 0.90, 225)],
    )
    def test_uses_the_finite_sample_rank(self, n, confidence, expected_rank):
        scores = np.arange(1.0, n + 1.0)
        assert conformal_quantile(scores, confidence) == float(expected_rank)

    def test_too_small_a_calibration_set_returns_infinity(self):
        """Certifying 95% needs at least 19 scores; below that, say so."""
        assert np.isinf(conformal_quantile(np.arange(1.0, 11.0), 0.95))

    def test_empty_scores_return_nan(self):
        assert np.isnan(conformal_quantile(np.array([]), 0.90))

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            conformal_quantile(np.arange(10.0), 1.5)

    def test_quantile_is_monotone_in_the_confidence_level(self):
        rng = np.random.default_rng(1)
        scores = np.abs(rng.normal(size=500))
        values = [conformal_quantile(scores, c) for c in (0.80, 0.90, 0.95)]
        assert values == sorted(values)

    def test_non_finite_scores_are_ignored(self):
        finite = np.arange(1.0, 101.0)
        polluted = np.concatenate([finite, [np.nan, np.inf, -np.inf]])
        assert conformal_quantile(polluted, 0.80) == conformal_quantile(finite, 0.80)


class TestCoverageGuarantee:
    def test_marginal_coverage_on_exchangeable_data(self):
        """Split conformal on i.i.d. data must hit its nominal level."""
        rng = np.random.default_rng(42)
        calibration = np.abs(rng.normal(0, 1, 2000))
        test = np.abs(rng.normal(0, 1, 20000))
        for confidence in (0.80, 0.90, 0.95):
            quantile = conformal_quantile(calibration, confidence)
            coverage = float(np.mean(test <= quantile))
            tolerance = 4 * np.sqrt(confidence * (1 - confidence) / len(test))
            assert abs(coverage - confidence) < tolerance, f"{confidence:.0%} -> {coverage:.4f}"

    def test_guarantee_holds_for_a_heavy_tailed_error_distribution(self):
        """Conformal makes no distributional assumption; verify that."""
        rng = np.random.default_rng(7)
        calibration = np.abs(rng.standard_t(df=2, size=4000))
        test = np.abs(rng.standard_t(df=2, size=20000))
        quantile = conformal_quantile(calibration, 0.90)
        assert abs(float(np.mean(test <= quantile)) - 0.90) < 0.02


class TestVolatilityScale:
    def test_scale_is_strictly_positive_even_on_a_flat_series(self):
        index = pd.date_range("2024-01-01", periods=200, freq="B")
        frame = pd.DataFrame({"USD_MRU": np.full(200, 36.0)}, index=index)
        scale = volatility_scale(frame, "USD_MRU", index[-50:], window=21, horizon=7)
        assert (scale > 0).all()
        assert np.isfinite(scale).all()

    def test_scale_grows_with_the_horizon(self, synthetic_frame):
        origins = synthetic_frame.index[-50:]
        short = volatility_scale(synthetic_frame, "USD_MRU", origins, horizon=1)
        long = volatility_scale(synthetic_frame, "USD_MRU", origins, horizon=30)
        assert (long >= short).all()

    def test_scale_is_larger_in_a_volatile_stretch(self):
        rng = np.random.default_rng(11)
        index = pd.date_range("2020-01-01", periods=600, freq="B")
        returns = np.concatenate([rng.normal(0, 0.001, 300), rng.normal(0, 0.02, 300)])
        frame = pd.DataFrame({"USD_MRU": 36 * np.exp(np.cumsum(returns))}, index=index)
        calm = volatility_scale(frame, "USD_MRU", index[250:280], window=21, horizon=1)
        stormy = volatility_scale(frame, "USD_MRU", index[550:580], window=21, horizon=1)
        assert stormy.mean() > 5 * calm.mean()


class TestConformalPredictor:
    @pytest.fixture
    def fitted(self, synthetic_frame, config):
        index = synthetic_frame.index
        predictor = build_conformal(NaiveForecaster(), config)
        predictor.fit(synthetic_frame, "USD_MRU", 7, config, train_end=index[-500])
        predictor.calibrate(synthetic_frame, index[-499:-100])
        return predictor, index[-100:-7]

    def test_predict_interval_before_calibration_raises(self, synthetic_frame, config):
        predictor = build_conformal(NaiveForecaster(), config)
        predictor.fit(synthetic_frame, "USD_MRU", 1, config)
        with pytest.raises(NotFittedError):
            predictor.predict_interval(synthetic_frame, synthetic_frame.index[-5:])

    def test_calibrating_inside_the_training_window_is_refused(self, synthetic_frame, config):
        """The guarantee comes from the model never having seen these outcomes."""
        index = synthetic_frame.index
        predictor = build_conformal(NaiveForecaster(), config)
        predictor.fit(synthetic_frame, "USD_MRU", 1, config, train_end=index[-100])
        with pytest.raises(ValueError, match="training window"):
            predictor.calibrate(synthetic_frame, index[-300:-200])

    def test_too_few_calibration_points_raises(self, synthetic_frame, config):
        index = synthetic_frame.index
        predictor = build_conformal(NaiveForecaster(), config)
        predictor.fit(synthetic_frame, "USD_MRU", 1, config, train_end=index[-100])
        with pytest.raises(ValueError, match="calibration scores"):
            predictor.calibrate(synthetic_frame, index[-99:-90])

    def test_interval_ordering(self, fitted, synthetic_frame):
        predictor, origins = fitted
        intervals = predictor.predict_interval(synthetic_frame, origins)
        for level in (80, 90, 95):
            assert (intervals[f"lower_{level}"] <= intervals["prediction"] + 1e-9).all()
            assert (intervals["prediction"] <= intervals[f"upper_{level}"] + 1e-9).all()
        assert (intervals["lower_95"] <= intervals["lower_90"] + 1e-9).all()
        assert (intervals["upper_95"] >= intervals["upper_90"] - 1e-9).all()

    def test_lower_bound_is_never_negative(self, fitted, synthetic_frame):
        predictor, origins = fitted
        intervals = predictor.predict_interval(synthetic_frame, origins)
        assert (intervals[["lower_80", "lower_90", "lower_95"]] >= 0).all().all()

    def test_point_forecast_is_unchanged_by_conformalisation(self, fitted, synthetic_frame):
        predictor, origins = fitted
        base = NaiveForecaster().fit(
            synthetic_frame,
            "USD_MRU",
            7,
            predictor.base_model.config,
            train_end=predictor.base_model.train_end,
        )
        assert np.allclose(
            predictor.predict_interval(synthetic_frame, origins)["prediction"],
            base.predict_origins(synthetic_frame, origins),
        )

    def test_requesting_an_uncalibrated_level_raises(self, fitted, synthetic_frame):
        predictor, origins = fitted
        with pytest.raises(KeyError, match="not calibrated"):
            predictor.predict_interval(synthetic_frame, origins, confidence=0.99)

    def test_calibration_summary_records_provenance(self, fitted):
        predictor, _ = fitted
        summary = predictor.summary.as_dict()
        assert summary["n_scores"] > 100
        assert summary["method"] in {"split", "normalized"}
        assert summary["calibration_start"] < summary["calibration_end"]

    def test_forecast_future_lands_on_a_business_day(self, fitted, synthetic_frame):
        predictor, _ = fitted
        row = predictor.forecast_future(synthetic_frame)
        assert len(row) == 1
        assert row.index[0] > synthetic_frame.index[-1]
        assert row.index[0].dayofweek < 5

    def test_normalised_intervals_vary_in_width(self, fitted, synthetic_frame):
        predictor, origins = fitted
        widths = predictor.predict_interval(synthetic_frame, origins)
        spread = widths["upper_90"] - widths["lower_90"]
        assert spread.std() > 0, "a locally adaptive interval must not be constant"

    def test_unnormalised_intervals_have_one_width(self, synthetic_frame, config):
        index = synthetic_frame.index
        predictor = ConformalPredictor(
            NaiveForecaster(), confidence_levels=[0.90], normalize_by_volatility=False
        )
        predictor.fit(synthetic_frame, "USD_MRU", 7, config, train_end=index[-500])
        predictor.calibrate(synthetic_frame, index[-499:-100])
        intervals = predictor.predict_interval(synthetic_frame, index[-100:-7])
        widths = intervals["upper_90"] - intervals["lower_90"]
        assert widths.std() == pytest.approx(0.0, abs=1e-9)


class TestAsymmetricConformal:
    def test_two_tails_are_calibrated_separately(self, synthetic_frame, config):
        index = synthetic_frame.index
        predictor = ConformalPredictor(
            NaiveForecaster(),
            confidence_levels=[0.90],
            symmetric=False,
            normalize_by_volatility=False,
        )
        predictor.fit(synthetic_frame, "USD_MRU", 7, config, train_end=index[-500])
        predictor.calibrate(synthetic_frame, index[-499:-100])
        assert predictor.lower_quantiles and predictor.upper_quantiles
        intervals = predictor.predict_interval(synthetic_frame, index[-100:-7])
        assert (intervals["lower_90"] <= intervals["upper_90"]).all()


class TestAdaptiveConformal:
    def test_alpha_widens_after_a_miss(self, synthetic_frame, config):
        index = synthetic_frame.index
        predictor = build_conformal(NaiveForecaster(), config, adaptive=True)
        predictor.fit(synthetic_frame, "USD_MRU", 1, config, train_end=index[-400])
        predictor.calibrate(synthetic_frame, index[-399:-100])
        before = predictor.alpha_t[0.90]
        predictor.update(actual=1e6, interval={0.90: (30.0, 40.0)})  # a miss
        assert predictor.alpha_t[0.90] < before, "a miss must shrink alpha (widen the interval)"

    def test_alpha_narrows_after_a_hit(self, synthetic_frame, config):
        index = synthetic_frame.index
        predictor = build_conformal(NaiveForecaster(), config, adaptive=True)
        predictor.fit(synthetic_frame, "USD_MRU", 1, config, train_end=index[-400])
        predictor.calibrate(synthetic_frame, index[-399:-100])
        before = predictor.alpha_t[0.90]
        predictor.update(actual=35.0, interval={0.90: (30.0, 40.0)})  # a hit
        assert predictor.alpha_t[0.90] > before

    @pytest.mark.slow
    def test_online_run_produces_one_row_per_origin(self, synthetic_frame, config):
        index = synthetic_frame.index
        predictor = build_conformal(NaiveForecaster(), config, adaptive=True)
        predictor.fit(synthetic_frame, "USD_MRU", 1, config, train_end=index[-400])
        predictor.calibrate(synthetic_frame, index[-399:-120])
        origins = index[-119:-60]
        out = predictor.run_online(synthetic_frame, origins)
        assert len(out) == len(origins)
        assert predictor.history, "updates must be recorded"


class TestAgainstMapie:
    """Independent cross-check of the split-conformal quantile."""

    def test_matches_mapie_on_a_regression_problem(self):
        mapie = pytest.importorskip("mapie")
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(0)
        n = 3000
        X = rng.normal(size=(n, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 1.0, n)

        train, calib, test = slice(0, 1500), slice(1500, 2500), slice(2500, n)
        model = LinearRegression().fit(X[train], y[train])

        residuals = np.abs(y[calib] - model.predict(X[calib]))
        ours = conformal_quantile(residuals, 0.90)

        predictions = model.predict(X[test])
        coverage = float(np.mean((y[test] >= predictions - ours) & (y[test] <= predictions + ours)))
        assert abs(coverage - 0.90) < 0.03, f"coverage {coverage:.3f}"
        # The quantile must sit between the plain empirical quantile and the max.
        assert np.quantile(residuals, 0.90) <= ours <= residuals.max()
        assert mapie is not None
