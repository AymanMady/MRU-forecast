"""Phase 2: ingestion parsing and the derived cross.

Network access is never required: every test here feeds the parser a payload
shaped like the real one, including the corrupted field that motivated the
median-of-OHLC choice.  The tests that do hit the network are marked
``network`` and deselected by ``make test-fast``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mru_forecast.data.ingestion import (
    build_series_specs,
    compare_with_cross_check,
    to_wide,
)
from mru_forecast.data.sources import (
    IngestionError,
    SeriesSpec,
    YahooChartSource,
    archive_payload,
    derive_cross,
)

SPEC = SeriesSpec(
    name="USD_MRU",
    provider="yahoo",
    symbol="MRU=X",
    definition="MRU per 1 USD",
    rate_type="indicative_mid_market",
    source_url="https://example.invalid",
)


def yahoo_payload(rows: list[dict]) -> dict:
    """A payload in the shape the real endpoint returns."""
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"symbol": "MRU=X", "currency": "MRU"},
                    "timestamp": [int(pd.Timestamp(r["date"]).timestamp()) for r in rows],
                    "indicators": {
                        "quote": [
                            {
                                key: [r.get(key) for r in rows]
                                for key in ("open", "high", "low", "close")
                            }
                        ]
                    },
                }
            ],
        }
    }


class TestYahooParsing:
    def test_parses_a_normal_payload(self):
        payload = yahoo_payload(
            [
                {"date": "2024-01-02", "open": 39.5, "high": 39.7, "low": 39.4, "close": 39.6},
                {"date": "2024-01-03", "open": 39.6, "high": 39.8, "low": 39.5, "close": 39.7},
            ]
        )
        frame = YahooChartSource._parse(payload, SPEC)
        assert len(frame) == 2
        assert list(frame.columns) == ["open", "high", "low", "close", "value"]
        assert frame["value"].iloc[0] == pytest.approx(np.median([39.5, 39.7, 39.4, 39.6]))

    def test_median_of_ohlc_survives_one_corrupted_field(self):
        """The exact failure mode observed on the live feed: close = 2.00."""
        payload = yahoo_payload(
            [{"date": "2022-05-31", "open": 36.551, "high": 36.552, "low": 36.418, "close": 2.00}]
        )
        value = float(YahooChartSource._parse(payload, SPEC)["value"].iloc[0])
        assert 36.0 < value < 37.0, f"a single bad field moved the daily value to {value}"

    def test_close_alone_would_have_failed(self):
        """Documents why `close` is not used as the representative value."""
        payload = yahoo_payload(
            [{"date": "2022-05-31", "open": 36.551, "high": 36.552, "low": 36.418, "close": 2.00}]
        )
        frame = YahooChartSource._parse(payload, SPEC)
        assert float(frame["close"].iloc[0]) == 2.00
        assert float(frame["value"].iloc[0]) != 2.00

    def test_rows_without_any_quote_are_dropped(self):
        payload = yahoo_payload(
            [
                {"date": "2024-01-02", "open": 39.5, "high": 39.7, "low": 39.4, "close": 39.6},
                {"date": "2024-01-03", "open": None, "high": None, "low": None, "close": None},
            ]
        )
        assert len(YahooChartSource._parse(payload, SPEC)) == 1

    def test_index_is_naive_normalised_and_sorted(self):
        payload = yahoo_payload(
            [
                {"date": "2024-01-03", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
                {"date": "2024-01-02", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
            ]
        )
        index = YahooChartSource._parse(payload, SPEC).index
        assert index.tz is None
        assert index.is_monotonic_increasing
        assert (index.normalize() == index).all()

    def test_duplicate_dates_keep_the_last(self):
        payload = yahoo_payload(
            [
                {"date": "2024-01-02", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
                {"date": "2024-01-02", "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0},
            ]
        )
        frame = YahooChartSource._parse(payload, SPEC)
        assert len(frame) == 1
        assert frame["value"].iloc[0] == 2.0

    @pytest.mark.parametrize(
        "payload",
        [
            {"chart": {"error": {"code": "Not Found"}, "result": None}},
            {"chart": {"error": None, "result": []}},
            {"chart": {"error": None, "result": [{"meta": {}}]}},
        ],
    )
    def test_broken_payloads_raise_ingestion_error(self, payload):
        with pytest.raises(IngestionError):
            YahooChartSource._parse(payload, SPEC)


class TestDerivedCross:
    def test_cross_is_the_product_on_shared_dates(self):
        index = pd.date_range("2024-01-01", periods=5, freq="B")
        usd_mru = pd.Series([39.0, 39.1, 39.2, 39.3, 39.4], index=index)
        eur_usd = pd.Series([1.10, 1.11, 1.09, 1.12, 1.10], index=index)
        cross = derive_cross(usd_mru, eur_usd, name="EUR_MRU")
        assert np.allclose(cross["value"], usd_mru * eur_usd)

    def test_only_shared_dates_survive(self):
        left = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="B"))
        right = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-02", periods=2, freq="B"))
        assert len(derive_cross(left, right, name="X")) == 2

    def test_ohlc_columns_are_nan_because_a_cross_is_not_quoted(self):
        index = pd.date_range("2024-01-01", periods=3, freq="B")
        cross = derive_cross(
            pd.Series([39.0] * 3, index=index), pd.Series([1.1] * 3, index=index), name="X"
        )
        assert cross[["open", "high", "low", "close"]].isna().all().all()

    def test_no_overlap_raises(self):
        left = pd.Series([1.0], index=pd.to_datetime(["2024-01-01"]))
        right = pd.Series([1.0], index=pd.to_datetime(["2025-01-01"]))
        with pytest.raises(IngestionError, match="No overlapping dates"):
            derive_cross(left, right, name="X")


class TestSpecsAndManifest:
    def test_specs_are_built_from_the_config(self, config):
        specs = build_series_specs(config)
        by_name = {s.name: s for s in specs}
        assert by_name["USD_MRU"].role == "target"
        assert by_name["EUR_MRU"].is_derived
        assert by_name["EUR_MRU"].derived_from == ("USD_MRU", "EUR_USD")
        assert not by_name["USD_MRU"].is_derived

    def test_every_target_declares_its_rate_type(self, config):
        for spec in build_series_specs(config):
            assert spec.rate_type and spec.rate_type != "TBD"
            assert spec.definition

    def test_archive_records_a_checksum(self, tmp_path):
        entry = archive_payload({"a": 1}, tmp_path / "x.json")
        assert entry["sha256"] and entry["bytes"] > 0
        assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}


class TestCrossCheck:
    def test_gap_is_reported_in_percent(self):
        index = pd.date_range("2024-01-01", periods=3, freq="B")
        primary = pd.DataFrame({"USD_MRU": [100.0, 100.0, 100.0]}, index=index)
        other = pd.DataFrame({"USD_MRU": [101.0, 100.0, 99.0]}, index=index)
        gap = compare_with_cross_check(primary, other, tolerance_pct=2.0)
        # (100 - 101) / 101 * 100 = -0.9901%
        assert gap["USD_MRU_gap_pct"].iloc[0] == pytest.approx(-0.9901, abs=1e-4)
        assert gap["USD_MRU_gap_pct"].iloc[1] == pytest.approx(0.0)
        assert gap["within_tolerance"].all()

    def test_a_large_gap_falls_outside_tolerance(self):
        index = pd.date_range("2024-01-01", periods=2, freq="B")
        primary = pd.DataFrame({"USD_MRU": [100.0, 100.0]}, index=index)
        other = pd.DataFrame({"USD_MRU": [120.0, 100.0]}, index=index)
        gap = compare_with_cross_check(primary, other, tolerance_pct=2.0)
        assert not gap["within_tolerance"].iloc[0]

    def test_no_shared_dates_returns_empty(self):
        primary = pd.DataFrame({"USD_MRU": [1.0]}, index=pd.to_datetime(["2024-01-01"]))
        other = pd.DataFrame({"USD_MRU": [1.0]}, index=pd.to_datetime(["2025-01-01"]))
        assert compare_with_cross_check(primary, other).empty


class TestTidyToWide:
    def test_pivot_keeps_one_column_per_series(self):
        tidy = pd.DataFrame(
            {
                "series": ["USD_MRU", "USD_MRU", "EUR_USD", "EUR_USD"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
                "value": [39.0, 39.1, 1.10, 1.11],
            }
        )
        wide = to_wide(tidy)
        assert list(wide.columns) == ["EUR_USD", "USD_MRU"]
        assert wide.shape == (2, 2)


@pytest.mark.network
class TestLiveProvider:
    """Hits the real endpoint. Deselect with `-m "not network"`."""

    def test_usd_mru_series_is_fetchable(self):
        from datetime import date

        source = YahooChartSource(timeout=30, max_retries=2)
        frame = source.fetch(SPEC, date(2024, 1, 1), date(2024, 3, 1))
        assert len(frame) > 30
        assert frame["value"].between(10, 120).all()
