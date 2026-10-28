"""Ingestion orchestration (Phase 2).

Reads the series declared in ``configs/config.yaml``, fetches each one, derives
the EUR/MRU cross, archives the raw payloads and writes a single tidy interim
dataset plus a provenance manifest.

The output of this module is deliberately *not* clean: gaps, duplicates and bad
ticks are left in place so that Phase 3 has something real to act on and its
decisions stay auditable against ``data/raw/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.sources import (
    VALUE_COLUMN,
    CurrencyApiSource,
    IngestionError,
    SeriesSpec,
    YahooChartSource,
    derive_cross,
)
from mru_forecast.utils.logging import get_logger

logger = get_logger(__name__)

INTERIM_FILENAME = "rates_interim.parquet"
MANIFEST_FILENAME = "ingestion_manifest.json"
CROSSCHECK_FILENAME = "cross_check.csv"


def build_series_specs(config: dict[str, Any]) -> list[SeriesSpec]:
    """Translate the ``ingestion.series`` config block into specs."""
    specs: list[SeriesSpec] = []
    for entry in config["ingestion"]["series"]:
        specs.append(
            SeriesSpec(
                name=entry["name"],
                provider=entry["provider"],
                symbol=entry.get("symbol", ""),
                definition=entry["definition"],
                rate_type=entry["rate_type"],
                source_url=entry.get("source_url", ""),
                role=entry.get("role", "auxiliary"),
                derived_from=tuple(entry.get("derived_from", ())),
            )
        )
    return specs


def _resolve_window(config: dict[str, Any]) -> tuple[date, date]:
    ingestion = config["ingestion"]
    start = pd.Timestamp(ingestion["start_date"]).date()
    end_raw = ingestion.get("end_date")
    end = pd.Timestamp(end_raw).date() if end_raw else datetime.now(UTC).date()
    if start >= end:
        raise IngestionError(f"Empty ingestion window: {start} -> {end}")
    return start, end


def _sample_dates(index: pd.DatetimeIndex, n: int, earliest: str) -> list[date]:
    """Evenly spaced sample of dates for the independent cross-check."""
    usable = index[index >= pd.Timestamp(earliest)]
    if len(usable) == 0 or n <= 0:
        return []
    step = max(len(usable) // n, 1)
    return [ts.date() for ts in usable[::step][:n]]


def ingest(
    config: dict[str, Any] | None = None, *, cross_check: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch every configured series and return ``(tidy_frame, manifest)``.

    The returned frame is long/tidy: one row per ``(series, date)`` with the
    provider's OHLC fields and the representative ``value``.
    """
    cfg = config if config is not None else load_config()
    start, end = _resolve_window(cfg)
    raw_dir = resolve_path("data_raw", cfg)
    interim_dir = resolve_path("data_interim", cfg)

    specs = build_series_specs(cfg)
    yahoo = YahooChartSource(
        timeout=cfg["ingestion"]["request_timeout_s"],
        max_retries=cfg["ingestion"]["max_retries"],
        raw_dir=raw_dir if cfg["ingestion"].get("cache_raw", True) else None,
    )

    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    # --- 1. directly quoted series -------------------------------------
    for spec in (s for s in specs if not s.is_derived):
        if spec.provider != "yahoo":
            failures[spec.name] = f"unsupported provider '{spec.provider}'"
            continue
        try:
            frames[spec.name] = yahoo.fetch(spec, start, end)
        except IngestionError as exc:
            # An optional regressor failing must not sink the whole run; a
            # target failing must.
            if spec.role == "target":
                raise
            logger.warning("skipping auxiliary series %s: %s", spec.name, exc)
            failures[spec.name] = str(exc)

    # --- 2. derived series ---------------------------------------------
    for spec in (s for s in specs if s.is_derived):
        missing = [dep for dep in spec.derived_from if dep not in frames]
        if missing:
            message = f"cannot derive {spec.name}: missing {missing}"
            if spec.role == "target":
                raise IngestionError(message)
            logger.warning(message)
            failures[spec.name] = message
            continue
        left, right = (frames[dep][VALUE_COLUMN] for dep in spec.derived_from)
        frames[spec.name] = derive_cross(left, right, name=spec.name)
        logger.info(
            "derived %s = %s x %s (%d rows)", spec.name, *spec.derived_from, len(frames[spec.name])
        )

    # --- 3. tidy long frame --------------------------------------------
    tidy = (
        pd.concat(frames, names=["series", "date"])
        .reset_index()
        .sort_values(["series", "date"], kind="stable")
        .reset_index(drop=True)
    )

    # --- 4. independent cross-check ------------------------------------
    cross = pd.DataFrame()
    if cross_check and cfg["ingestion"].get("cross_check", {}).get("enabled", False):
        cc_cfg = cfg["ingestion"]["cross_check"]
        dates = _sample_dates(
            pd.DatetimeIndex(sorted(tidy["date"].unique())),
            int(cc_cfg.get("sample_size", 20)),
            cc_cfg.get("earliest_date", "2024-03-05"),
        )
        cross = CurrencyApiSource(
            timeout=cfg["ingestion"]["request_timeout_s"],
            max_retries=cfg["ingestion"]["max_retries"],
        ).fetch_cross_check(dates, {"USD_MRU": ("usd", "mru"), "EUR_USD": ("eur", "usd")})
        if not cross.empty:
            cross.to_csv(interim_dir / CROSSCHECK_FILENAME)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "series": {
            spec.name: {
                **asdict(spec),
                "derived_from": list(spec.derived_from),
                "rows": len(frames[spec.name]) if spec.name in frames else 0,
                "first_date": str(frames[spec.name].index.min().date())
                if spec.name in frames
                else None,
                "last_date": str(frames[spec.name].index.max().date())
                if spec.name in frames
                else None,
                "raw_payload": yahoo.manifest.get(spec.name),
            }
            for spec in specs
        },
        "failures": failures,
        "cross_check_rows": len(cross),
    }

    tidy.to_parquet(interim_dir / INTERIM_FILENAME, index=False)
    (interim_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "ingested %d series / %d rows -> %s",
        tidy["series"].nunique(),
        len(tidy),
        interim_dir / INTERIM_FILENAME,
    )
    return tidy, manifest


def load_interim(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read the tidy interim dataset written by :func:`ingest`."""
    path = resolve_path("data_interim", config) / INTERIM_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python scripts/download_data.py` first.")
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def to_wide(tidy: pd.DataFrame, column: str = VALUE_COLUMN) -> pd.DataFrame:
    """Pivot the tidy frame to ``date x series`` using ``column``.

    ``pivot`` rather than ``pivot_table`` on purpose: there must be exactly one
    value per (date, series), and ``pivot`` raises if there is not.
    ``pivot_table`` would silently average a duplicate away, turning a real data
    problem into a plausible-looking number.
    """
    wide = tidy.pivot(index="date", columns="series", values=column).sort_index()  # noqa: PD010
    wide.columns.name = None
    return wide


def compare_with_cross_check(
    wide: pd.DataFrame, cross: pd.DataFrame, *, tolerance_pct: float = 2.0
) -> pd.DataFrame:
    """Relative gap (%) between the primary series and the cross-check source."""
    shared_dates = wide.index.intersection(cross.index)
    shared_cols = [c for c in cross.columns if c in wide.columns]
    if len(shared_dates) == 0 or not shared_cols:
        return pd.DataFrame()
    primary = wide.loc[shared_dates, shared_cols]
    other = cross.loc[shared_dates, shared_cols]
    gap = (primary - other) / other * 100.0
    gap.columns = [f"{c}_gap_pct" for c in gap.columns]
    gap["within_tolerance"] = gap.abs().le(tolerance_pct).all(axis=1)
    return gap


def default_cross_check_dates(days_back: int = 540, n: int = 20) -> list[date]:
    """Convenience sample of past dates, newest first."""
    today = datetime.now(UTC).date()
    step = max(days_back // max(n, 1), 1)
    return [today - timedelta(days=i * step) for i in range(1, n + 1)]
