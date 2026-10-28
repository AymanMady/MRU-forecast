"""Data validation (Phase 3).

Validation is separated from cleaning on purpose: cleaning *changes* the data,
validation only *reports* on it.  Every check returns a structured
:class:`ValidationIssue` rather than raising, so a run produces one complete
report instead of stopping at the first problem.  ``ValidationReport.raise_if_failed``
turns ERROR-level issues into an exception when a caller wants a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class Severity(StrEnum):
    ERROR = "ERROR"  # the dataset cannot be trusted for modelling
    WARNING = "WARNING"  # worth knowing; does not invalidate the dataset
    INFO = "INFO"


@dataclass
class ValidationIssue:
    code: str
    severity: Severity
    series: str
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = f"[{self.severity.value}] {self.series}/{self.code}: {self.message}"
        return f"{head} (examples: {', '.join(self.examples[:5])})" if self.examples else head


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def extend(self, other: ValidationReport) -> None:
        self.issues.extend(other.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["severity", "series", "code", "count", "message"])
        return pd.DataFrame(
            [
                {
                    "severity": i.severity.value,
                    "series": i.series,
                    "code": i.code,
                    "count": i.count,
                    "message": i.message,
                    "examples": ", ".join(i.examples[:5]),
                }
                for i in self.issues
            ]
        )

    def raise_if_failed(self) -> None:
        if self.errors:
            details = "\n".join(f"  - {issue}" for issue in self.errors)
            raise DataValidationError(f"{len(self.errors)} validation error(s):\n{details}")

    def summary(self) -> str:
        return (
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s), "
            f"{len(self.issues)} issue(s) total"
        )


class DataValidationError(ValueError):
    """Raised when a dataset fails an ERROR-level validation rule."""


def _examples(index: pd.Index, limit: int = 5) -> list[str]:
    """First few index entries as strings, dates rendered without a time part."""
    out: list[str] = []
    for value in index[:limit]:
        out.append(str(value.date()) if isinstance(value, pd.Timestamp) else str(value))
    return out


# ---------------------------------------------------------------------------
# Index-level checks
# ---------------------------------------------------------------------------
def check_index(series: pd.Series, name: str, rules: dict[str, Any]) -> ValidationReport:
    """Dates must be a unique, sorted, tz-naive daily index."""
    report = ValidationReport()
    index = series.index

    if not isinstance(index, pd.DatetimeIndex):
        report.add(
            ValidationIssue(
                "INDEX_NOT_DATETIME",
                Severity.ERROR,
                name,
                f"index is {type(index).__name__}, expected DatetimeIndex",
            )
        )
        return report

    if getattr(index, "tz", None) is not None:
        report.add(
            ValidationIssue(
                "INDEX_TZ_AWARE",
                Severity.WARNING,
                name,
                "index carries a timezone; daily series are handled tz-naive",
            )
        )

    if rules.get("require_unique_dates", True):
        duplicated = index[index.duplicated()]
        if len(duplicated):
            report.add(
                ValidationIssue(
                    "DUPLICATE_DATES",
                    Severity.ERROR,
                    name,
                    f"{len(duplicated)} duplicate date(s)",
                    len(duplicated),
                    _examples(duplicated),
                )
            )

    if rules.get("require_sorted_dates", True) and not index.is_monotonic_increasing:
        report.add(
            ValidationIssue(
                "UNSORTED_DATES", Severity.ERROR, name, "dates are not in increasing order"
            )
        )

    minimum = rules.get("min_observations", 0)
    if len(series) < minimum:
        report.add(
            ValidationIssue(
                "TOO_FEW_OBSERVATIONS",
                Severity.ERROR,
                name,
                f"{len(series)} observations, at least {minimum} required",
                len(series),
            )
        )
    return report


# ---------------------------------------------------------------------------
# Value-level checks
# ---------------------------------------------------------------------------
def check_values(series: pd.Series, name: str, rules: dict[str, Any]) -> ValidationReport:
    """Rates must be finite, strictly positive and inside a plausible band."""
    report = ValidationReport()
    values = series.astype("float64")

    missing = values.isna()
    if missing.any():
        report.add(
            ValidationIssue(
                "MISSING_VALUES",
                Severity.WARNING,
                name,
                f"{int(missing.sum())} missing value(s) ({missing.mean():.2%})",
                int(missing.sum()),
                _examples(values.index[missing]),
            )
        )

    observed = values.dropna()
    if observed.empty:
        report.add(ValidationIssue("EMPTY_SERIES", Severity.ERROR, name, "no observed values"))
        return report

    non_finite = observed[~np.isfinite(observed)]
    if len(non_finite):
        report.add(
            ValidationIssue(
                "NON_FINITE",
                Severity.ERROR,
                name,
                f"{len(non_finite)} non-finite value(s)",
                len(non_finite),
                _examples(non_finite.index),
            )
        )

    min_rate = rules.get("min_rate", 0.0)
    non_positive = observed[observed <= min_rate]
    if len(non_positive):
        report.add(
            ValidationIssue(
                "NON_POSITIVE_RATE",
                Severity.ERROR,
                name,
                f"{len(non_positive)} value(s) <= {min_rate}",
                len(non_positive),
                _examples(non_positive.index),
            )
        )

    band = (rules.get("plausible_range") or {}).get(name)
    if band:
        low, high = band
        outside = observed[(observed < low) | (observed > high)]
        if len(outside):
            report.add(
                ValidationIssue(
                    "OUTSIDE_PLAUSIBLE_RANGE",
                    Severity.ERROR,
                    name,
                    f"{len(outside)} value(s) outside [{low}, {high}]",
                    len(outside),
                    _examples(outside.index),
                )
            )
    return report


def check_returns(series: pd.Series, name: str, rules: dict[str, Any]) -> ValidationReport:
    """Report implausibly large day-on-day moves.

    A WARNING, never an ERROR: a managed currency genuinely does take large
    discrete steps, and silently dropping those would erase the very events the
    model most needs to represent.
    """
    report = ValidationReport()
    threshold = rules.get("max_daily_pct_change")
    if threshold is None:
        return report

    changes = series.dropna().pct_change().dropna().mul(100.0)
    extreme = changes[changes.abs() > threshold]
    if len(extreme):
        report.add(
            ValidationIssue(
                "EXTREME_DAILY_MOVE",
                Severity.WARNING,
                name,
                f"{len(extreme)} day(s) moved more than {threshold}% "
                f"(max {extreme.abs().max():.2f}%)",
                len(extreme),
                _examples(extreme.index),
            )
        )
    return report


def check_gaps(series: pd.Series, name: str, rules: dict[str, Any]) -> ValidationReport:
    """Report runs of consecutive business days with no observation."""
    report = ValidationReport()
    limit = rules.get("max_consecutive_missing_days")
    if limit is None or series.empty:
        return report

    calendar = pd.date_range(series.index.min(), series.index.max(), freq="B")
    present = series.dropna().reindex(calendar).notna()
    if present.all():
        return report

    gap_id = present.cumsum()
    runs = (~present).groupby(gap_id).sum()
    longest = int(runs.max()) if len(runs) else 0
    if longest > limit:
        offenders = runs[runs > limit]
        starts = [
            str(present.index[present.cumsum().eq(gid) & ~present][0].date())
            for gid in offenders.index[:5]
        ]
        report.add(
            ValidationIssue(
                "LONG_GAP",
                Severity.WARNING,
                name,
                f"{len(offenders)} gap(s) longer than {limit} business days (longest {longest})",
                len(offenders),
                starts,
            )
        )
    return report


def check_calendar_coverage(
    series: pd.Series, name: str, calendar: pd.DatetimeIndex, rules: dict[str, Any]
) -> ValidationReport:
    """Fraction of the modelling calendar for which the series has a value."""
    report = ValidationReport()
    if len(calendar) == 0:
        return report
    coverage = float(series.reindex(calendar).notna().mean() * 100.0)
    minimum = rules.get("min_coverage_pct", 0.0)
    severity = Severity.ERROR if coverage < minimum else Severity.INFO
    report.add(
        ValidationIssue(
            "CALENDAR_COVERAGE",
            severity,
            name,
            f"covers {coverage:.1f}% of the modelling calendar (minimum {minimum}%)",
            round(coverage),
        )
    )
    return report


def validate_series(
    series: pd.Series,
    name: str,
    config: dict[str, Any],
    *,
    calendar: pd.DatetimeIndex | None = None,
) -> ValidationReport:
    """Run every series-level check and return the combined report."""
    rules = {**config.get("validation", {}), **config.get("cleaning", {})}
    report = ValidationReport()
    report.extend(check_index(series, name, rules))
    report.extend(check_values(series, name, rules))
    report.extend(check_returns(series, name, rules))
    report.extend(check_gaps(series, name, rules))
    if calendar is not None:
        report.extend(check_calendar_coverage(series, name, calendar, rules))
    return report


def validate_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> ValidationReport:
    """Validate every column of a wide ``date x series`` frame."""
    report = ValidationReport()
    report.extend(
        check_index(
            frame.iloc[:, 0] if len(frame.columns) else pd.Series(dtype=float),
            "__index__",
            {**config.get("validation", {})},
        )
    )
    for column in frame.columns:
        report.extend(validate_series(frame[column], column, config, calendar=frame.index))

    targets = [t["name"] for t in config.get("targets", [])]
    for target in targets:
        if target not in frame.columns:
            report.add(
                ValidationIssue(
                    "MISSING_TARGET",
                    Severity.ERROR,
                    target,
                    "target column absent from the processed dataset",
                )
            )
        elif frame[target].isna().any():
            n = int(frame[target].isna().sum())
            report.add(
                ValidationIssue(
                    "TARGET_HAS_GAPS",
                    Severity.ERROR,
                    target,
                    f"{n} missing value(s) remain in a target after cleaning",
                    n,
                )
            )
    return report


def validate_forecast_intervals(
    predictions: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
    *,
    name: str = "forecast",
) -> ValidationReport:
    """An interval must satisfy ``lower <= prediction <= upper`` everywhere."""
    report = ValidationReport()
    pred = np.asarray(predictions, dtype="float64")
    low = np.asarray(lower, dtype="float64")
    high = np.asarray(upper, dtype="float64")

    if not (pred.shape == low.shape == high.shape):
        report.add(
            ValidationIssue(
                "INTERVAL_SHAPE_MISMATCH",
                Severity.ERROR,
                name,
                f"shapes differ: prediction {pred.shape}, lower {low.shape}, upper {high.shape}",
            )
        )
        return report

    inverted = int(np.sum(low > high))
    if inverted:
        report.add(
            ValidationIssue(
                "INVERTED_INTERVAL",
                Severity.ERROR,
                name,
                f"{inverted} interval(s) with lower > upper",
                inverted,
            )
        )
    outside = int(np.sum((pred < low) | (pred > high)))
    if outside:
        report.add(
            ValidationIssue(
                "PREDICTION_OUTSIDE_INTERVAL",
                Severity.ERROR,
                name,
                f"{outside} point forecast(s) outside their own interval",
                outside,
            )
        )
    if np.any(low <= 0):
        report.add(
            ValidationIssue(
                "NON_POSITIVE_LOWER_BOUND",
                Severity.WARNING,
                name,
                "some lower bounds are <= 0, which is impossible for an exchange rate",
                int(np.sum(low <= 0)),
            )
        )
    return report
