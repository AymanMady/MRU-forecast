"""Pydantic response models for the forecasting API.

Typed responses give the API a self-documenting OpenAPI schema and make the
contract explicit for the dashboard and any other client.  Every forecast
response carries its provenance — the rate type, whether the pair is a derived
cross, and the disclaimer — because a number without those is not interpretable.
"""

from __future__ import annotations

# Imported under an alias: several models have a field literally named `date`,
# and a field name that shadows its own annotation breaks Pydantic.
from datetime import date as DateType  # noqa: N812 - `date` is also a field name below
from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Experimental forecast built on indicative mid-market rates from a public "
    "market-data provider, not the official rate published by the Banque Centrale "
    "de Mauritanie. Intervals are empirically calibrated on historical backtests "
    "and are not a guarantee. Not investment advice."
)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    models_loaded: bool
    data_last_date: DateType | None = None
    n_predictors: int = 0
    trained_at: str | None = None
    detail: str | None = None


class ForecastPoint(BaseModel):
    date: DateType = Field(description="Business day the forecast refers to")
    days_ahead: int = Field(ge=1, description="Business days ahead of the anchor date")
    prediction: float = Field(gt=0, description="Point forecast, MRU per unit")
    lower: float = Field(ge=0, description="Lower bound of the prediction interval")
    upper: float = Field(gt=0, description="Upper bound of the prediction interval")
    interval_width: float = Field(ge=0)


class ForecastResponse(BaseModel):
    currency_pair: str = Field(examples=["USD/MRU"])
    rate_definition: str = Field(examples=["MRU per 1 USD"])
    rate_type: str = Field(examples=["indicative_mid_market"])
    is_derived_cross: bool = Field(
        description="True when the pair is computed from two other quotes rather than observed"
    )
    horizon: int
    confidence: float = Field(gt=0, lt=1)
    anchor_date: DateType = Field(description="Last observed business day")
    anchor_value: float = Field(gt=0, description="Last observed rate")
    model: str
    forecast: list[ForecastPoint]
    disclaimer: str = DISCLAIMER


class RateObservation(BaseModel):
    date: DateType
    value: float


class HistoryResponse(BaseModel):
    currency_pair: str
    rate_type: str
    n_observations: int
    first_date: DateType
    last_date: DateType
    observations: list[RateObservation]


class LatestRatesResponse(BaseModel):
    as_of: DateType
    rates: dict[str, float]
    daily_change_pct: dict[str, float]
    rate_type: str
    disclaimer: str = DISCLAIMER


class PerformanceRow(BaseModel):
    target: str
    horizon: int
    model: str
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    theil_u2: float | None = None
    mae_skill_vs_naive_pct: float | None = None
    n: int | None = None


class PerformanceResponse(BaseModel):
    backtest_period: list[str] | None = None
    n_splits: int | None = None
    rows: list[PerformanceRow]
    note: str = (
        "Metrics come from an expanding-window backtest on held-out data. "
        "theil_u2 below 1 means the model beats a random walk."
    )


class CalibrationRow(BaseModel):
    model: str
    nominal_coverage: float
    empirical_coverage: float
    coverage_error_pp: float
    coverage_se_pp: float | None = None
    mean_interval_width: float
    n: int


class CalibrationResponse(BaseModel):
    rows: list[CalibrationRow]
    note: str = (
        "Empirical coverage is an estimate; compare differences against "
        "coverage_se_pp before reading anything into them."
    )
