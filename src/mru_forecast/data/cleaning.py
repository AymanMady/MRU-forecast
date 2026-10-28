"""Cleaning (Phase 3).

Pipeline, in order:

1. drop duplicate dates
2. sort chronologically
3. flag bad ticks (isolated reverting spikes, then a rolling-median guard)
4. mask the flagged points to ``NaN`` — never overwrite with an invented value
5. project onto the modelling calendar (business days)
6. forward-fill closed days, up to ``max_ffill_days``
7. validate the result and persist it with a report

Bad-tick detection
------------------
The primary feed emits isolated corrupted values (an observed 2.00 MRU/USD
between two days at 36.2).  Such a point deviates from **both** neighbours in
the **same** direction.  A genuine step devaluation does not: it is above the
previous day and level with the next one.  ``detect_isolated_spikes`` encodes
exactly that asymmetry, which is why it removes the corrupted ticks without
erasing the real policy steps that matter most for this currency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from mru_forecast.config import load_config, resolve_path
from mru_forecast.data.ingestion import load_interim, to_wide
from mru_forecast.data.validation import ValidationReport, validate_dataset, validate_series
from mru_forecast.utils.logging import get_logger

logger = get_logger(__name__)

PROCESSED_FILENAME = "rates_processed.parquet"
FLAGS_FILENAME = "cleaning_flags.csv"
REPORT_FILENAME = "cleaning_report.json"


@dataclass
class CleaningStats:
    """What cleaning actually did to one series — reported, never hidden."""

    series: str
    n_input: int = 0
    n_duplicates_dropped: int = 0
    n_spikes_masked: int = 0
    n_median_outliers_masked: int = 0
    n_calendar_days: int = 0
    n_forward_filled: int = 0
    n_still_missing: int = 0
    coverage_pct: float = 0.0
    flagged_dates: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "series": self.series,
            "n_input": self.n_input,
            "n_duplicates_dropped": self.n_duplicates_dropped,
            "n_spikes_masked": self.n_spikes_masked,
            "n_median_outliers_masked": self.n_median_outliers_masked,
            "n_calendar_days": self.n_calendar_days,
            "n_forward_filled": self.n_forward_filled,
            "n_still_missing": self.n_still_missing,
            "coverage_pct": round(self.coverage_pct, 2),
            "flagged_dates": self.flagged_dates,
        }


# ---------------------------------------------------------------------------
# Step 1-2: dates
# ---------------------------------------------------------------------------
def deduplicate_and_sort(series: pd.Series, *, keep: str = "last") -> tuple[pd.Series, int]:
    """Drop duplicate dates and sort ascending. Returns ``(series, n_dropped)``."""
    duplicated = series.index.duplicated(keep=keep)
    n_dropped = int(duplicated.sum())
    cleaned = series[~duplicated].sort_index()
    return cleaned, n_dropped


# ---------------------------------------------------------------------------
# Step 3: bad-tick detection
# ---------------------------------------------------------------------------
def detect_isolated_spikes(series: pd.Series, tolerance_pct: float = 4.0) -> pd.Series:
    """Boolean mask of isolated reverting spikes.

    A point is flagged when it differs from the previous **and** the next
    observed value by more than ``tolerance_pct``, with both deviations in the
    same direction — i.e. the series jumps away and comes straight back.
    """
    values = series.astype("float64")
    observed = values.dropna()
    flags = pd.Series(False, index=values.index)
    if len(observed) < 3:
        return flags

    previous = observed.shift(1)
    following = observed.shift(-1)
    tolerance = tolerance_pct / 100.0

    dev_prev = observed / previous - 1.0
    dev_next = observed / following - 1.0

    spike = (
        (dev_prev.abs() > tolerance)
        & (dev_next.abs() > tolerance)
        & (np.sign(dev_prev) == np.sign(dev_next))
    )
    # Endpoints have only one neighbour, so they can never be confirmed as
    # isolated; leave them to the plausibility band instead of guessing.
    spike = spike.fillna(False)
    flags.loc[spike.index] = spike.to_numpy()
    return flags


def detect_rolling_median_outliers(
    series: pd.Series, window: int = 21, tolerance_pct: float = 12.0
) -> pd.Series:
    """Boolean mask of points far from a centred rolling median.

    Secondary guard for corrupted *runs* (several consecutive bad values),
    which the isolated-spike test cannot see.  The tolerance is intentionally
    generous: on a managed currency the rolling median lags real step moves,
    and a tight threshold would delete genuine devaluations.
    """
    values = series.astype("float64")
    observed = values.dropna()
    flags = pd.Series(False, index=values.index)
    if len(observed) < window:
        return flags

    reference = observed.rolling(window, center=True, min_periods=max(window // 2, 2)).median()
    relative_deviation = (observed - reference).abs() / reference.abs()
    outlier = (relative_deviation > tolerance_pct / 100.0).fillna(False)
    flags.loc[outlier.index] = outlier.to_numpy()
    return flags


def detect_out_of_band(series: pd.Series, band: tuple[float, float] | None) -> pd.Series:
    """Boolean mask of values outside a hard plausibility band."""
    flags = pd.Series(False, index=series.index)
    if not band:
        return flags
    low, high = band
    values = series.astype("float64")
    flags |= (values < low) | (values > high)
    return flags.fillna(False)


# ---------------------------------------------------------------------------
# Steps 5-6: calendar and filling
# ---------------------------------------------------------------------------
def build_calendar(index: pd.DatetimeIndex, calendar: str = "business_day") -> pd.DatetimeIndex:
    """Continuous modelling calendar spanning ``index``."""
    if len(index) == 0:
        return pd.DatetimeIndex([], name="date")
    freq = "B" if calendar == "business_day" else "D"
    return pd.date_range(index.min(), index.max(), freq=freq, name="date")


def project_on_calendar(
    series: pd.Series, calendar: pd.DatetimeIndex, *, max_ffill_days: int = 10
) -> tuple[pd.Series, int]:
    """Reindex onto ``calendar`` and forward-fill, bounded.

    Forward-fill only — an exchange rate on a closed day is, operationally, the
    last quoted rate.  Backward filling would leak a future observation into the
    past and is never used anywhere in this project.
    """
    projected = series.reindex(calendar)
    missing_before = projected.isna()
    filled = projected.ffill(limit=max_ffill_days)
    n_filled = int((missing_before & filled.notna()).sum())
    return filled, n_filled


# ---------------------------------------------------------------------------
# Series-level orchestration
# ---------------------------------------------------------------------------
def clean_series(
    series: pd.Series,
    name: str,
    config: dict[str, Any],
    *,
    calendar: pd.DatetimeIndex | None = None,
) -> tuple[pd.Series, CleaningStats, pd.DataFrame]:
    """Clean one series. Returns ``(cleaned, stats, flags_frame)``."""
    cleaning = config["cleaning"]
    validation = config["validation"]
    stats = CleaningStats(series=name, n_input=len(series))

    working, stats.n_duplicates_dropped = deduplicate_and_sort(
        series, keep="last" if cleaning.get("drop_duplicates") == "keep_last" else "first"
    )

    band = (validation.get("plausible_range") or {}).get(name)
    out_of_band = detect_out_of_band(working, tuple(band) if band else None)

    # Iterate: a corrupted *run* hides its members from the single-pass test,
    # because each bad point's neighbour is itself bad.  Once the outer members
    # are masked, the survivors become isolated and are caught on the next pass.
    spikes = out_of_band.copy()  # an impossible value is bad regardless of neighbours
    working = working.mask(spikes)
    for _ in range(int(cleaning.get("spike_max_passes", 3))):
        new_spikes = detect_isolated_spikes(working, cleaning.get("spike_reversion_pct", 4.0))
        if not new_spikes.any():
            break
        spikes |= new_spikes
        working = working.mask(new_spikes)
    stats.n_spikes_masked = int(spikes.sum())

    median_outliers = detect_rolling_median_outliers(
        working,
        int(cleaning.get("rolling_median_window", 21)),
        float(cleaning.get("rolling_median_tol_pct", 12.0)),
    )
    stats.n_median_outliers_masked = int(median_outliers.sum())
    working = working.mask(median_outliers)

    flagged = spikes | median_outliers
    flags_frame = pd.DataFrame(
        {
            "series": name,
            "date": series.index[flagged.reindex(series.index, fill_value=False)],
            "original_value": series[flagged.reindex(series.index, fill_value=False)].to_numpy(),
            "reason": np.where(
                out_of_band.reindex(series.index, fill_value=False)[
                    flagged.reindex(series.index, fill_value=False)
                ],
                "out_of_plausible_band",
                np.where(
                    spikes.reindex(series.index, fill_value=False)[
                        flagged.reindex(series.index, fill_value=False)
                    ],
                    "isolated_reverting_spike",
                    "rolling_median_outlier",
                ),
            ),
        }
    )
    stats.flagged_dates = [str(d.date()) for d in flags_frame["date"][:20]]

    target_calendar = (
        calendar
        if calendar is not None
        else build_calendar(working.dropna().index, cleaning.get("calendar", "business_day"))
    )
    cleaned, stats.n_forward_filled = project_on_calendar(
        working, target_calendar, max_ffill_days=int(cleaning.get("max_ffill_days", 10))
    )
    cleaned.name = name

    stats.n_calendar_days = len(target_calendar)
    stats.n_still_missing = int(cleaned.isna().sum())
    stats.coverage_pct = float(cleaned.notna().mean() * 100.0) if len(cleaned) else 0.0

    logger.info(
        "cleaned %-10s | dup=%d spikes=%d median=%d ffill=%d missing=%d coverage=%.1f%%",
        name,
        stats.n_duplicates_dropped,
        stats.n_spikes_masked,
        stats.n_median_outliers_masked,
        stats.n_forward_filled,
        stats.n_still_missing,
        stats.coverage_pct,
    )
    return cleaned, stats, flags_frame


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------
def build_processed_dataset(
    config: dict[str, Any] | None = None,
    *,
    tidy: pd.DataFrame | None = None,
    persist: bool = True,
) -> tuple[pd.DataFrame, ValidationReport, dict[str, Any]]:
    """Clean every ingested series and assemble the modelling dataset.

    The modelling calendar is driven by the **targets**: auxiliary series are
    projected onto it, never the other way round, so an auxiliary feed with a
    different trading calendar cannot extend or shrink the target history.
    """
    cfg = config if config is not None else load_config()
    source = tidy if tidy is not None else load_interim(cfg)
    wide = to_wide(source)

    target_names = [t["name"] for t in cfg["targets"]]
    missing_targets = [t for t in target_names if t not in wide.columns]
    if missing_targets:
        raise KeyError(f"Targets absent from the interim dataset: {missing_targets}")

    # Pre-clean the targets to get an honest span before fixing the calendar.
    target_span = pd.DatetimeIndex(
        sorted(set().union(*(set(wide[t].dropna().index) for t in target_names)))
    )
    calendar = build_calendar(target_span, cfg["cleaning"].get("calendar", "business_day"))

    columns: dict[str, pd.Series] = {}
    all_stats: list[dict[str, Any]] = []
    all_flags: list[pd.DataFrame] = []
    per_series_report = ValidationReport()

    for name in wide.columns:
        raw = wide[name].dropna()
        per_series_report.extend(validate_series(raw, f"{name} (raw)", cfg))
        cleaned, stats, flags = clean_series(raw, name, cfg, calendar=calendar)
        columns[name] = cleaned
        all_stats.append(stats.as_dict())
        if not flags.empty:
            all_flags.append(flags)

    frame = pd.DataFrame(columns, index=calendar)

    # Drop auxiliary series that do not cover enough of the modelling calendar;
    # a mostly-empty regressor is worse than no regressor.
    minimum = float(cfg["cleaning"].get("min_coverage_pct", 0.0))
    dropped: list[str] = []
    for name in list(frame.columns):
        if name in target_names:
            continue
        if frame[name].notna().mean() * 100.0 < minimum:
            dropped.append(name)
            frame = frame.drop(columns=name)

    # Targets must be complete: trim the leading rows before every target has a
    # value rather than imputing them.
    first_complete = frame[target_names].dropna().index.min()
    if pd.notna(first_complete):
        frame = frame.loc[first_complete:]
    frame = frame.dropna(subset=target_names)
    frame.index.name = "date"

    report = validate_dataset(frame, cfg)
    report.extend(per_series_report)

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_rows": len(frame),
        "n_columns": int(frame.shape[1]),
        "first_date": str(frame.index.min().date()) if len(frame) else None,
        "last_date": str(frame.index.max().date()) if len(frame) else None,
        "calendar": cfg["cleaning"].get("calendar", "business_day"),
        "columns": list(frame.columns),
        "targets": target_names,
        "dropped_low_coverage": dropped,
        "per_series": all_stats,
        "validation": {
            "ok": report.ok,
            "summary": report.summary(),
            "issues": [
                {
                    "severity": i.severity.value,
                    "series": i.series,
                    "code": i.code,
                    "count": i.count,
                    "message": i.message,
                }
                for i in report.issues
            ],
        },
    }

    if persist:
        processed_dir = resolve_path("data_processed", cfg)
        frame.to_parquet(processed_dir / PROCESSED_FILENAME)
        if all_flags:
            pd.concat(all_flags, ignore_index=True).to_csv(
                processed_dir / FLAGS_FILENAME, index=False
            )
        (processed_dir / REPORT_FILENAME).write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        logger.info("processed dataset -> %s", processed_dir / PROCESSED_FILENAME)

    return frame, report, metadata


def load_processed(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read the processed modelling dataset."""
    path = resolve_path("data_processed", config) / PROCESSED_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python scripts/build_dataset.py` first.")
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame
