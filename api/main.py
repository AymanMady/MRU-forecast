"""FastAPI application (Phase 13).

    uvicorn api.main:app --reload --port 8000

Endpoints
---------
    GET /health
    GET /rates/latest
    GET /rates/{pair}/history?days=365
    GET /forecast/{pair}?horizon=7&confidence=0.90
    GET /models/performance
    GET /models/calibration

``/forecast/usd-mru`` and ``/forecast/eur-mru`` are the two that matter; the
path parameter accepts ``usd-mru``, ``usd_mru`` or ``USD/MRU``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    CalibrationResponse,
    ForecastResponse,
    HealthResponse,
    HistoryResponse,
    LatestRatesResponse,
    PerformanceResponse,
)
from api.service import ForecastService, ServiceUnavailableError, get_service, resolve_pair
from mru_forecast import __version__
from mru_forecast.utils.logging import get_logger, setup_logging

logger = get_logger("api")

DESCRIPTION = """
Experimental forecasting service for the Mauritanian Ouguiya (MRU).

Every forecast is returned with a **conformal prediction interval** whose
coverage has been measured on an expanding-window backtest, not assumed.

**Rate provenance.** Quotes are indicative mid-market rates from a public
market-data provider — *not* the official rate published by the Banque Centrale
de Mauritanie. EUR/MRU is a derived cross (USD/MRU x EUR/USD), not an observed
quote. Do not use this service as a basis for financial decisions.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    service = get_service()
    if service.ready:
        logger.info(
            "API ready: %d predictors, data to %s",
            len(service.pipeline.steps),
            service.frame.index.max().date(),
        )
    else:
        logger.warning("API started in degraded mode: %s", service.load_error)
    yield


app = FastAPI(
    title="MRU-Forecast API",
    description=DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    contact={"name": "MRU-Forecast"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # read-only public data; no credentials are ever accepted
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _pair(pair: str) -> str:
    try:
        return resolve_pair(pair)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: ForecastService = Depends(get_service)) -> HealthResponse:
    """Liveness and readiness. Reports `degraded` when artefacts are missing."""
    return HealthResponse(**service.health())


@app.get("/rates/latest", response_model=LatestRatesResponse, tags=["rates"])
def latest_rates(service: ForecastService = Depends(get_service)) -> LatestRatesResponse:
    """Most recent observed rate for every target pair."""
    try:
        return LatestRatesResponse(**service.latest_rates())
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/rates/{pair}/history", response_model=HistoryResponse, tags=["rates"])
def history(
    pair: str = Path(description="usd-mru or eur-mru"),
    days: int = Query(365, ge=1, le=5000, description="Number of business days to return"),
    service: ForecastService = Depends(get_service),
) -> HistoryResponse:
    """Observed history for one pair, most recent `days` business days."""
    try:
        return HistoryResponse(**service.history(_pair(pair), days))
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/forecast/{pair}", response_model=ForecastResponse, tags=["forecast"])
def forecast(
    pair: str = Path(description="usd-mru or eur-mru"),
    horizon: int = Query(7, ge=1, le=30, description="Business days ahead"),
    confidence: float = Query(
        0.90, description="Interval level; must be one of the calibrated levels"
    ),
    service: ForecastService = Depends(get_service),
) -> ForecastResponse:
    """Daily forecast path with conformal prediction intervals."""
    target = _pair(pair)
    try:
        return ForecastResponse(**service.forecast(target, horizon, confidence))
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/models/performance", response_model=PerformanceResponse, tags=["models"])
def performance(
    target: str | None = Query(None, description="Filter by USD_MRU or EUR_MRU"),
    service: ForecastService = Depends(get_service),
) -> PerformanceResponse:
    """Backtested point-forecast accuracy for every model and horizon."""
    try:
        return PerformanceResponse(**service.performance(target))
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/models/calibration", response_model=CalibrationResponse, tags=["models"])
def calibration(service: ForecastService = Depends(get_service)) -> CalibrationResponse:
    """Nominal vs empirical interval coverage, measured on the backtest."""
    try:
        return CalibrationResponse(**service.calibration())
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
