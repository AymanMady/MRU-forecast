"""Exploratory analysis primitives (Phase 4).

Pure functions returning frames, so the same code backs the notebooks, the
figure-generation script and the tests.  Nothing here plots — see
``mru_forecast.viz``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def describe_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Level statistics, plus the share of days with no change at all.

    ``flat_day_share`` is the diagnostic that matters most here: a floating
    currency moves almost every day, a managed one does not.  A high value is
    the signature of an administered rate and explains why a random walk is
    such a strong competitor.
    """
    out = frame.describe().T
    diffs = frame.diff()
    out["flat_day_share"] = (diffs.abs() < 1e-9).mean()
    out["first_date"] = frame.apply(lambda s: s.first_valid_index())
    out["last_date"] = frame.apply(lambda s: s.last_valid_index())
    out["total_change_pct"] = (frame.iloc[-1] / frame.iloc[0] - 1.0) * 100.0
    years = max((frame.index[-1] - frame.index[0]).days / 365.25, 1e-9)
    out["annualised_drift_pct"] = ((frame.iloc[-1] / frame.iloc[0]) ** (1 / years) - 1.0) * 100.0
    return out


def compute_returns(frame: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """Daily returns. ``kind`` is ``log``, ``pct`` or ``diff``.

    Log returns are the default: they are additive across horizons, symmetric
    for up and down moves, and are what the models operate on.
    """
    if kind == "log":
        return np.log(frame).diff().dropna(how="all")
    if kind == "pct":
        return frame.pct_change().dropna(how="all")
    if kind == "diff":
        return frame.diff().dropna(how="all")
    raise ValueError(f"unknown return kind: {kind!r}")


def rolling_volatility(
    returns: pd.Series | pd.DataFrame, window: int = 30, annualise: bool = True
) -> pd.Series | pd.DataFrame:
    """Rolling standard deviation of returns, optionally annualised."""
    vol = returns.rolling(window, min_periods=max(window // 2, 2)).std()
    if annualise:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol * 100.0


def seasonality_table(returns: pd.Series, by: str = "day_of_week") -> pd.DataFrame:
    """Mean / std / count of returns grouped by a calendar unit.

    ``n`` is reported alongside every mean so that a large-looking effect on a
    handful of observations is not mistaken for a pattern.
    """
    index = returns.index
    groups = {
        "day_of_week": index.dayofweek,
        "month": index.month,
        "quarter": index.quarter,
        "year": index.year,
        "day_of_month": index.day,
    }
    if by not in groups:
        raise ValueError(f"unknown grouping: {by!r}. Available: {sorted(groups)}")

    grouped = returns.groupby(groups[by])
    table = pd.DataFrame(
        {
            "mean_pct": grouped.mean() * 100.0,
            "std_pct": grouped.std() * 100.0,
            "n": grouped.size(),
        }
    )
    if by == "day_of_week":
        table.index = table.index.map(
            {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        )
    table.index.name = by
    # Crude two-sided t-like signal; a |t| below ~2 means "indistinguishable
    # from zero at this sample size".
    table["t_stat"] = table["mean_pct"] / (table["std_pct"] / np.sqrt(table["n"].clip(lower=1)))
    return table


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation of *returns*, never of levels.

    Two trending level series correlate strongly for no reason other than both
    trending; correlating returns removes that spurious component.
    """
    return returns.corr(method=method)


def lagged_correlation(
    returns: pd.DataFrame, target: str, others: list[str], max_lag: int = 10
) -> pd.DataFrame:
    """Correlation of ``target`` at t with each ``other`` at t-lag.

    Only non-negative lags are reported: a regressor is usable at prediction
    time only if it is already known.
    """
    rows = []
    for lag in range(0, max_lag + 1):
        row: dict[str, Any] = {"lag": lag}
        for other in others:
            if other not in returns.columns:
                continue
            row[other] = returns[target].corr(returns[other].shift(lag))
        rows.append(row)
    return pd.DataFrame(rows).set_index("lag")


def autocorrelation(series: pd.Series, nlags: int = 30) -> pd.DataFrame:
    """ACF and PACF with approximate 95% confidence bounds."""
    from statsmodels.tsa.stattools import acf, pacf

    clean = series.dropna()
    acf_values = acf(clean, nlags=nlags, fft=True)
    pacf_values = pacf(clean, nlags=min(nlags, len(clean) // 2 - 1))
    bound = 1.96 / np.sqrt(len(clean))
    return pd.DataFrame(
        {
            "acf": acf_values,
            "pacf": np.pad(
                pacf_values, (0, len(acf_values) - len(pacf_values)), constant_values=np.nan
            ),
            "conf_lower": -bound,
            "conf_upper": bound,
        },
        index=pd.RangeIndex(len(acf_values), name="lag"),
    )


def stationarity_tests(series: pd.Series) -> pd.DataFrame:
    """ADF and KPSS, which test opposite null hypotheses.

    ADF  H0: a unit root is present (non-stationary).
    KPSS H0: the series is stationary.
    Agreement between the two is what makes the conclusion credible.
    """
    import warnings

    from statsmodels.tsa.stattools import adfuller, kpss

    clean = series.dropna()
    with warnings.catch_warnings():
        # Both tests warn about presentation details (a future return type for
        # ADF, a p-value clipped at the lookup-table boundary for KPSS).
        # Neither affects the statistic or the decision.
        warnings.simplefilter("ignore")
        adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(clean, regression="c", nlags="auto")

    return pd.DataFrame(
        [
            {
                "test": "ADF",
                "null_hypothesis": "unit root (non-stationary)",
                "statistic": adf_stat,
                "p_value": adf_p,
                "reject_null_at_5pct": adf_p < 0.05,
                "interpretation": "stationary" if adf_p < 0.05 else "non-stationary",
            },
            {
                "test": "KPSS",
                "null_hypothesis": "stationary",
                "statistic": kpss_stat,
                "p_value": kpss_p,
                "reject_null_at_5pct": kpss_p < 0.05,
                "interpretation": "non-stationary" if kpss_p < 0.05 else "stationary",
            },
        ]
    )


def detect_step_events(
    series: pd.Series, threshold_pct: float = 1.0, min_gap_days: int = 5
) -> pd.DataFrame:
    """Large discrete moves — the policy steps of a managed currency.

    Consecutive days above the threshold are collapsed into one event so that a
    two-day adjustment is not counted twice.
    """
    changes = series.pct_change() * 100.0
    large = changes[changes.abs() >= threshold_pct].dropna()
    if large.empty:
        return pd.DataFrame(columns=["date", "change_pct", "level_before", "level_after"])

    events, last_date = [], None
    for date, change in large.items():
        if last_date is not None and (date - last_date).days < min_gap_days:
            continue
        position = series.index.get_loc(date)
        events.append(
            {
                "date": date,
                "change_pct": change,
                "level_before": series.iloc[max(position - 1, 0)],
                "level_after": series.loc[date],
            }
        )
        last_date = date
    return pd.DataFrame(events)


def volatility_regimes(
    series: pd.Series,
    window: int = 30,
    quantiles: tuple[float, float] = (0.33, 0.66),
    labels: tuple[str, str, str] = ("low", "normal", "high"),
) -> pd.Series:
    """Label each date by its *trailing* realised volatility regime.

    Trailing, not centred: the regime of a date must be knowable on that date,
    otherwise every regime-conditional metric computed later would be
    contaminated by the future.
    """
    returns = np.log(series).diff()
    vol = returns.rolling(window, min_periods=max(window // 2, 2)).std()
    low_cut, high_cut = vol.quantile(list(quantiles))
    regime = pd.Series(labels[1], index=series.index, dtype="object")
    regime[vol <= low_cut] = labels[0]
    regime[vol > high_cut] = labels[2]
    regime[vol.isna()] = pd.NA
    regime.name = "volatility_regime"
    return regime


def summarise_gaps(series: pd.Series) -> pd.DataFrame:
    """Distribution of calendar gaps between consecutive observations."""
    deltas = pd.Series(series.dropna().index).diff().dt.days.dropna()
    counts = deltas.value_counts().sort_index()
    return pd.DataFrame({"gap_days": counts.index.astype(int), "n": counts.to_numpy()})
