"""Phase 13: the HTTP contract.

These run against the real artefacts and skip cleanly when they are absent, so
the suite is meaningful on a fresh clone and thorough on a built one.
"""

from __future__ import annotations

import pytest
from api.main import app
from api.service import ForecastService, resolve_pair
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def ready(client) -> bool:
    return client.get("/health").json()["models_loaded"]


def requires_artifacts(ready: bool) -> None:
    if not ready:
        pytest.skip("artefacts not built; run scripts/train.py")


class TestPairResolution:
    @pytest.mark.parametrize(
        "alias", ["usd-mru", "USD-MRU", "usd_mru", "USD/MRU", "usdmru", " usd-mru "]
    )
    def test_accepted_spellings(self, alias):
        assert resolve_pair(alias) == "USD_MRU"

    def test_eur_aliases(self):
        assert resolve_pair("eur-mru") == "EUR_MRU"

    def test_unknown_pair_raises(self):
        with pytest.raises(KeyError):
            resolve_pair("gbp-mru")


class TestHealth:
    def test_always_answers(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"ok", "degraded"}
        assert "version" in body

    def test_degraded_reports_why(self, client):
        body = client.get("/health").json()
        if not body["models_loaded"]:
            assert body["detail"], "a degraded service must say what is missing"


class TestForecastEndpoint:
    def test_returns_the_documented_shape(self, client, ready):
        requires_artifacts(ready)
        body = client.get("/forecast/usd-mru", params={"horizon": 7, "confidence": 0.90}).json()
        assert body["currency_pair"] == "USD/MRU"
        assert body["horizon"] == 7
        assert body["confidence"] == 0.90
        assert len(body["forecast"]) == 7
        assert body["disclaimer"]
        point = body["forecast"][0]
        assert set(point) == {
            "date",
            "days_ahead",
            "prediction",
            "lower",
            "upper",
            "interval_width",
        }

    @pytest.mark.parametrize("pair", ["usd-mru", "eur-mru"])
    def test_bounds_bracket_the_prediction(self, client, ready, pair):
        requires_artifacts(ready)
        body = client.get(f"/forecast/{pair}", params={"horizon": 10}).json()
        for point in body["forecast"]:
            assert point["lower"] <= point["prediction"] <= point["upper"], point
            assert point["lower"] > 0, "a rate cannot be negative"
            assert point["interval_width"] == pytest.approx(
                point["upper"] - point["lower"], abs=1e-6
            )

    def test_days_ahead_are_sequential(self, client, ready):
        requires_artifacts(ready)
        body = client.get("/forecast/usd-mru", params={"horizon": 5}).json()
        assert [p["days_ahead"] for p in body["forecast"]] == [1, 2, 3, 4, 5]
        dates = [p["date"] for p in body["forecast"]]
        assert dates == sorted(dates)

    def test_forecast_dates_are_after_the_anchor(self, client, ready):
        requires_artifacts(ready)
        body = client.get("/forecast/usd-mru", params={"horizon": 3}).json()
        assert all(p["date"] > body["anchor_date"] for p in body["forecast"])

    def test_wider_confidence_gives_a_wider_interval(self, client, ready):
        requires_artifacts(ready)
        widths = {}
        for confidence in (0.80, 0.90, 0.95):
            body = client.get(
                "/forecast/usd-mru", params={"horizon": 7, "confidence": confidence}
            ).json()
            widths[confidence] = body["forecast"][-1]["interval_width"]
        assert widths[0.80] <= widths[0.90] <= widths[0.95]

    def test_eur_mru_is_labelled_as_a_derived_cross(self, client, ready):
        requires_artifacts(ready)
        body = client.get("/forecast/eur-mru", params={"horizon": 1}).json()
        assert body["is_derived_cross"] is True
        usd = client.get("/forecast/usd-mru", params={"horizon": 1}).json()
        assert usd["is_derived_cross"] is False

    def test_unknown_pair_is_404(self, client):
        assert client.get("/forecast/gbp-mru").status_code == 404

    def test_uncalibrated_confidence_is_400(self, client, ready):
        requires_artifacts(ready)
        response = client.get("/forecast/usd-mru", params={"confidence": 0.5})
        assert response.status_code == 400
        assert "calibrated" in response.json()["detail"]

    @pytest.mark.parametrize("horizon", [0, -1, 31, 999])
    def test_out_of_range_horizon_is_rejected(self, client, horizon):
        assert client.get("/forecast/usd-mru", params={"horizon": horizon}).status_code == 422


class TestRatesEndpoints:
    def test_latest_returns_both_pairs(self, client, ready):
        requires_artifacts(ready)
        body = client.get("/rates/latest").json()
        assert set(body["rates"]) == {"USD/MRU", "EUR/MRU"}
        assert all(v > 0 for v in body["rates"].values())
        assert body["rate_type"]

    def test_history_respects_the_day_limit(self, client, ready):
        requires_artifacts(ready)
        body = client.get("/rates/usd-mru/history", params={"days": 30}).json()
        assert body["n_observations"] == 30
        assert len(body["observations"]) == 30

    def test_history_is_chronological(self, client, ready):
        requires_artifacts(ready)
        body = client.get("/rates/usd-mru/history", params={"days": 50}).json()
        dates = [o["date"] for o in body["observations"]]
        assert dates == sorted(dates)

    def test_history_rejects_a_zero_window(self, client):
        assert client.get("/rates/usd-mru/history", params={"days": 0}).status_code == 422


class TestModelEndpoints:
    def test_performance_reports_theil_u2(self, client):
        response = client.get("/models/performance")
        if response.status_code == 503:
            pytest.skip("backtest not run")
        rows = response.json()["rows"]
        assert rows
        assert all("theil_u2" in row for row in rows)

    def test_performance_can_be_filtered(self, client):
        response = client.get("/models/performance", params={"target": "USD_MRU"})
        if response.status_code == 503:
            pytest.skip("backtest not run")
        assert {row["target"] for row in response.json()["rows"]} == {"USD_MRU"}

    def test_calibration_coverages_are_fractions(self, client):
        response = client.get("/models/calibration")
        if response.status_code == 503:
            pytest.skip("backtest not run")
        for row in response.json()["rows"]:
            assert 0.0 <= row["empirical_coverage"] <= 1.0
            assert 0.0 < row["nominal_coverage"] < 1.0


class TestOpenAPI:
    def test_schema_is_generated(self, client):
        schema = client.get("/openapi.json").json()
        assert "/forecast/{pair}" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_documentation_states_the_rate_provenance(self, client):
        raw = client.get("/openapi.json").json()["info"]["description"]
        description = " ".join(raw.split())  # the source text is hard-wrapped
        assert "Banque Centrale de Mauritanie" in description
        assert "derived cross" in description
        assert "not a basis for financial decisions" in description.lower().replace(
            "do not use this service as a basis for financial decisions",
            "not a basis for financial decisions",
        )


class TestServiceLayer:
    def test_service_starts_even_without_artifacts(self, tmp_path, config):
        """A missing model must degrade the service, not prevent it from starting."""
        broken = {**config, "paths": {**config["paths"], "models": str(tmp_path / "nope")}}
        service = ForecastService(broken)
        assert service.health()["status"] == "degraded"
        assert service.load_error
