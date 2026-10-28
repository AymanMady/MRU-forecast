"""Data providers.

Every provider implements :class:`DataSource`: given a :class:`SeriesSpec` and a
date range, return a tidy OHLC frame indexed by date.  Nothing here cleans,
resamples or interpolates — ingestion stays a faithful transcription of what the
provider returned, so that Phase 3 can be audited against the archived payload.

Provider notes
--------------
``YahooChartSource``
    Public chart endpoint of Yahoo Finance.  No API key.  Quotes are *indicative
    mid-market provider rates*, **not** the official rate published by the
    Banque Centrale de Mauritanie.  The feed is known to emit corrupted single
    fields (an observed ``close`` of 2.00 MRU/USD against an open of 36.55); the
    per-day representative value is therefore the **median of the four OHLC
    fields**, which is unaffected by any single bad field.  See
    ``docs/DATA_SOURCES.md``.

``CurrencyApiSource``
    fawazahmed0/currency-api mirrored on jsDelivr.  One HTTP request per date,
    so it is used as an *independent cross-check* on a sample of dates rather
    than as a bulk history provider.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mru_forecast.utils.logging import get_logger

logger = get_logger(__name__)

# Columns every provider returns, in this order.
OHLC_COLUMNS = ["open", "high", "low", "close"]
VALUE_COLUMN = "value"


class IngestionError(RuntimeError):
    """A provider could not deliver a usable series."""


@dataclass(frozen=True)
class SeriesSpec:
    """What to fetch, and what it means.

    ``rate_type`` and ``source`` travel with the data all the way to the
    processed dataset: mixing an official rate with a provider rate would
    invalidate the study, so the provenance is never allowed to get lost.
    """

    name: str
    provider: str
    symbol: str
    definition: str
    rate_type: str
    source_url: str
    role: str = "target"  # target | auxiliary
    derived_from: tuple[str, ...] = field(default=())

    @property
    def is_derived(self) -> bool:
        return bool(self.derived_from)


class DataSource(Protocol):
    """Minimal provider contract."""

    name: str

    def fetch(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        """Return a frame indexed by naive daily ``DatetimeIndex``.

        Columns: ``open``, ``high``, ``low``, ``close``, ``value``.
        """


def _build_session(timeout: int, max_retries: int) -> requests.Session:
    """A session that retries idempotent failures with exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "mru-forecast/0.1 (+research project)"})
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def archive_payload(payload: Any, destination: Path) -> dict[str, Any]:
    """Write a provider payload verbatim and return its manifest entry.

    Keeping the untouched response makes the whole pipeline re-runnable offline
    and lets any cleaning decision be re-checked against the original bytes.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    destination.write_bytes(blob)
    return {
        "path": str(destination),
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


class YahooChartSource:
    """Yahoo Finance public chart endpoint."""

    name = "yahoo"
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(
        self,
        *,
        timeout: int = 30,
        max_retries: int = 3,
        raw_dir: Path | None = None,
    ) -> None:
        self.timeout = timeout
        self.session = _build_session(timeout, max_retries)
        self.raw_dir = raw_dir
        self.manifest: dict[str, dict[str, Any]] = {}

    def fetch(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        payload = self._request(spec.symbol, start, end)

        if self.raw_dir is not None:
            entry = archive_payload(payload, self.raw_dir / self.name / f"{spec.name}.json")
            entry.update(symbol=spec.symbol, provider=self.name)
            self.manifest[spec.name] = entry

        frame = self._parse(payload, spec)
        logger.info(
            "yahoo: %s (%s) -> %d rows, %s to %s",
            spec.name,
            spec.symbol,
            len(frame),
            frame.index.min().date() if len(frame) else "-",
            frame.index.max().date() if len(frame) else "-",
        )
        return frame

    # -- internals ---------------------------------------------------------
    def _request(self, symbol: str, start: date, end: date) -> dict[str, Any]:
        params = {
            # The endpoint downsamples to weekly/monthly when given `range=max`,
            # so the window is always passed as explicit epoch bounds.
            "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
            "period2": int(datetime.combine(end, datetime.max.time()).timestamp()),
            "interval": "1d",
        }
        url = self.BASE_URL.format(symbol=symbol)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise IngestionError(f"Yahoo request failed for {symbol}: {exc}") from exc

    @staticmethod
    def _parse(payload: dict[str, Any], spec: SeriesSpec) -> pd.DataFrame:
        chart = payload.get("chart") or {}
        if chart.get("error"):
            raise IngestionError(f"Yahoo returned an error for {spec.symbol}: {chart['error']}")
        results = chart.get("result")
        if not results:
            raise IngestionError(f"Yahoo returned no result for {spec.symbol}")

        result = results[0]
        timestamps = result.get("timestamp")
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        if not timestamps or not quote:
            raise IngestionError(f"Yahoo payload for {spec.symbol} has no quote data")

        index = (
            pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert(None)
            .normalize()
            .rename("date")
        )
        frame = pd.DataFrame({col: quote.get(col) for col in OHLC_COLUMNS}, index=index)
        frame = frame.astype("float64")

        # Representative daily value: median of the OHLC fields.  A single
        # corrupted field cannot move the median of four, whereas `close` alone
        # is demonstrably unreliable on this feed.
        frame[VALUE_COLUMN] = frame[OHLC_COLUMNS].median(axis=1, skipna=True)
        frame = frame.dropna(subset=[VALUE_COLUMN])
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame


class CurrencyApiSource:
    """fawazahmed0/currency-api via jsDelivr — one snapshot per requested date.

    Used to cross-check the primary provider, not to build history: each date is
    a separate HTTP request.  Dated releases exist from 2024-03 onward.
    """

    name = "currency_api"
    BASE_URL = (
        "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{day}/v1/currencies/{base}.min.json"
    )

    def __init__(self, *, timeout: int = 30, max_retries: int = 3) -> None:
        self.timeout = timeout
        self.session = _build_session(timeout, max_retries)

    def fetch_rate(self, base: str, quote: str, day: date) -> float | None:
        """Return ``quote`` units per 1 ``base`` on ``day``, or ``None``."""
        url = self.BASE_URL.format(day=day.isoformat(), base=base.lower())
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            table = response.json().get(base.lower(), {})
        except (requests.RequestException, ValueError):
            return None
        value = table.get(quote.lower())
        return float(value) if isinstance(value, (int, float)) else None

    def fetch_cross_check(
        self, sample_dates: list[date], pairs: dict[str, tuple[str, str]]
    ) -> pd.DataFrame:
        """Fetch ``pairs`` (name -> (base, quote)) on each date in the sample."""
        rows: list[dict[str, Any]] = []
        for day in sample_dates:
            row: dict[str, Any] = {"date": pd.Timestamp(day)}
            usd_table_ok = False
            for name, (base, quote) in pairs.items():
                rate = self.fetch_rate(base, quote, day)
                row[name] = rate
                usd_table_ok |= rate is not None
            if usd_table_ok:
                rows.append(row)
        if not rows:
            logger.warning("currency-api cross-check returned nothing; is the CDN reachable?")
            return pd.DataFrame(columns=["date", *pairs]).set_index("date")
        return pd.DataFrame(rows).set_index("date").sort_index()


def derive_cross(numerator: pd.Series, multiplier: pd.Series, *, name: str) -> pd.DataFrame:
    """Build a cross rate as ``numerator * multiplier`` on the shared calendar.

    EUR/MRU is not quoted with usable history by the primary provider, so it is
    derived: ``EUR/MRU = (USD/MRU) x (EUR/USD)``.  The result is labelled as
    derived everywhere downstream — it inherits the noise of *both* legs and
    must never be presented as an observed quote.
    """
    aligned = pd.concat({"a": numerator, "b": multiplier}, axis=1, join="inner").dropna()
    if aligned.empty:
        raise IngestionError(f"No overlapping dates to derive {name}")
    value = aligned["a"] * aligned["b"]
    frame = pd.DataFrame(
        {col: np.nan for col in OHLC_COLUMNS}, index=aligned.index, dtype="float64"
    )
    frame[VALUE_COLUMN] = value
    frame.index.name = "date"
    return frame
