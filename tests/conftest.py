"""Shared fixtures.

The synthetic series here is deliberately built to look like the real one: a
slow upward crawl, long perfectly flat stretches, occasional discrete steps and
two injected bad ticks.  Tests that pass on a smooth random walk but fail on a
managed rate would not be testing this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mru_forecast.config import PROJECT_ROOT, load_config


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def business_index() -> pd.DatetimeIndex:
    return pd.date_range("2019-01-01", periods=1400, freq="B", name="date")


@pytest.fixture(scope="session")
def managed_series(business_index) -> pd.Series:
    """A synthetic managed exchange rate: crawl + flat stretches + steps."""
    rng = np.random.default_rng(20260914)
    n = len(business_index)
    values = np.empty(n)
    level = 36.0
    for i in range(n):
        if rng.random() < 0.30:
            pass  # a flat day: the rate simply does not move
        elif rng.random() < 0.02:
            level *= 1.0 + rng.normal(0.012, 0.006)  # a discrete policy step
        else:
            level *= 1.0 + rng.normal(0.00006, 0.0012)
        values[i] = level
    return pd.Series(values, index=business_index, name="USD_MRU")


@pytest.fixture
def series_with_bad_ticks(managed_series) -> tuple[pd.Series, list[pd.Timestamp]]:
    """``managed_series`` with two isolated corrupted values injected."""
    corrupted = managed_series.copy()
    positions = [400, 900]
    dates = [corrupted.index[p] for p in positions]
    corrupted.iloc[positions[0]] = 2.0  # the exact failure mode seen in the real feed
    corrupted.iloc[positions[1]] = 3.62
    return corrupted, dates


@pytest.fixture(scope="session")
def synthetic_frame(managed_series, business_index) -> pd.DataFrame:
    """A full wide frame with both targets and the auxiliary series."""
    rng = np.random.default_rng(7)
    eur_usd = pd.Series(
        1.10 * np.exp(np.cumsum(rng.normal(0, 0.004, len(business_index)))),
        index=business_index,
    )
    return pd.DataFrame(
        {
            "USD_MRU": managed_series,
            "EUR_USD": eur_usd,
            "EUR_MRU": managed_series * eur_usd,
            "BRENT_OIL": 80 * np.exp(np.cumsum(rng.normal(0, 0.015, len(business_index)))),
            "GOLD": 1800 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, len(business_index)))),
        },
        index=business_index,
    )


@pytest.fixture(scope="session")
def processed_frame(config):
    """The real processed dataset, skipped when it has not been built."""
    from mru_forecast.data.cleaning import load_processed

    try:
        return load_processed(config)
    except FileNotFoundError:
        pytest.skip("processed dataset not built; run scripts/build_dataset.py")


@pytest.fixture(scope="session")
def trained_pipeline(config):
    """The real trained artefacts, skipped when they have not been built."""
    from mru_forecast.pipeline.forecasting_pipeline import ForecastingPipeline

    try:
        return ForecastingPipeline.load(config=config)
    except FileNotFoundError:
        pytest.skip("pipeline not trained; run scripts/train.py")
