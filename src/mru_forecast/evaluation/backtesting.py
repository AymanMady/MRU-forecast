"""Time-series cross-validation (Phase 9).

``train_test_split`` with shuffling is meaningless here: it would train on 2026
and test on 2020.  Every split in this module is strictly chronological and has
three blocks, in time order::

    |............ train ............|.. calibration ..|.. test ..|
                                    ^                 ^
                              train_end        conformal quantile
                                                 computed here

The calibration block is the price of a conformal guarantee: the quantile has to
come from data the model has never been fitted on.  Splits are anchored at the
**end** of the sample and stepped backwards, so the evaluation always covers the
most recent period rather than an arbitrary early window.

Every test origin is guaranteed to have a realised outcome: the grid is shifted
back by ``max(horizons)`` so that a 30-day forecast made at the last test origin
can still be scored.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mru_forecast.models.base import Forecaster
from mru_forecast.models.baseline import NaiveForecaster
from mru_forecast.models.conformal import ConformalPredictor, build_conformal
from mru_forecast.utils.logging import get_logger

logger = get_logger(__name__)

ModelFactory = Callable[[], Forecaster]


@dataclass(frozen=True)
class Split:
    """One chronological train / calibration / test partition."""

    index: int
    train: pd.DatetimeIndex
    calibration: pd.DatetimeIndex
    test: pd.DatetimeIndex

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train[-1]

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.index,
            "train_start": str(self.train[0].date()),
            "train_end": str(self.train[-1].date()),
            "n_train": len(self.train),
            "calib_start": str(self.calibration[0].date()),
            "calib_end": str(self.calibration[-1].date()),
            "n_calib": len(self.calibration),
            "test_start": str(self.test[0].date()),
            "test_end": str(self.test[-1].date()),
            "n_test": len(self.test),
        }


def make_splits(
    index: pd.DatetimeIndex, config: dict[str, Any], *, max_horizon: int | None = None
) -> list[Split]:
    """Build the backtest splits, newest first in construction, oldest first on return."""
    settings = config["backtesting"]
    scheme = settings.get("scheme", "expanding")
    n_splits = int(settings["n_splits"])
    step = int(settings["step_size"])
    test_size = int(settings.get("test_size", step))
    calibration_size = int(settings["calibration_size"])
    initial_train = int(settings["initial_train_size"])
    gap = int(settings.get("gap", 0))

    horizon_margin = int(max_horizon if max_horizon is not None else max(config["horizons"]))
    n = len(index)
    usable_end = n - horizon_margin  # every test origin must have an outcome
    if usable_end <= 0:
        raise ValueError(f"only {n} rows: too few for a horizon of {horizon_margin}")

    splits: list[Split] = []
    for i in range(n_splits):
        test_stop = usable_end - i * step
        test_start = test_stop - test_size
        calibration_stop = test_start - gap
        calibration_start = calibration_stop - calibration_size
        train_stop = calibration_start - gap
        train_start = 0 if scheme == "expanding" else max(train_stop - initial_train, 0)

        if train_stop - train_start < initial_train or test_start < 0:
            logger.info(
                "stopping at %d split(s): not enough history for an earlier one", len(splits)
            )
            break

        splits.append(
            Split(
                index=n_splits - 1 - i,
                train=index[train_start:train_stop],
                calibration=index[calibration_start:calibration_stop],
                test=index[test_start:test_stop],
            )
        )

    if not splits:
        raise ValueError(
            f"no valid split: {n} rows cannot fit initial_train_size={initial_train} + "
            f"calibration_size={calibration_size} + test_size={test_size} + "
            f"horizon={horizon_margin}"
        )
    # Renumber chronologically so split 0 is the earliest.
    splits = sorted(splits, key=lambda s: s.test[0])
    return [
        Split(index=i, train=s.train, calibration=s.calibration, test=s.test)
        for i, s in enumerate(splits)
    ]


def splits_summary(splits: Iterable[Split]) -> pd.DataFrame:
    return pd.DataFrame([s.describe() for s in splits])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class BacktestResults:
    """Tidy, one row per (split, model, target, horizon, origin)."""

    predictions: pd.DataFrame
    splits: pd.DataFrame
    calibration: pd.DataFrame
    runtime: pd.DataFrame

    def __len__(self) -> int:
        return len(self.predictions)

    def interval_levels(self) -> list[int]:
        return sorted(
            int(c.removeprefix("lower_"))
            for c in self.predictions.columns
            if c.startswith("lower_")
        )

    def to_parquet(self, directory) -> None:
        from pathlib import Path

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.predictions.to_parquet(out / "backtest_predictions.parquet", index=False)
        self.splits.to_csv(out / "backtest_splits.csv", index=False)
        self.calibration.to_csv(out / "backtest_calibration.csv", index=False)
        self.runtime.to_csv(out / "backtest_runtime.csv", index=False)

    @classmethod
    def from_parquet(cls, directory) -> BacktestResults:
        from pathlib import Path

        out = Path(directory)
        predictions = pd.read_parquet(out / "backtest_predictions.parquet")
        for column in ("origin", "target_date"):
            if column in predictions:
                predictions[column] = pd.to_datetime(predictions[column])
        return cls(
            predictions=predictions,
            splits=pd.read_csv(out / "backtest_splits.csv"),
            calibration=pd.read_csv(out / "backtest_calibration.csv"),
            runtime=pd.read_csv(out / "backtest_runtime.csv"),
        )


def run_backtest(
    frame: pd.DataFrame,
    config: dict[str, Any],
    model_factories: dict[str, ModelFactory],
    *,
    targets: list[str] | None = None,
    horizons: list[int] | None = None,
    conformal: bool = True,
    splits: list[Split] | None = None,
    progress: bool = True,
) -> BacktestResults:
    """Fit, calibrate and evaluate every model on every split.

    The naive forecast is always computed alongside, because "is this better
    than a random walk?" is the only question that decides whether the rest of
    the pipeline earned its place.
    """
    target_names = targets or [t["name"] for t in config["targets"]]
    horizon_list = horizons or list(config["horizons"])
    split_list = splits or make_splits(
        pd.DatetimeIndex(frame.index), config, max_horizon=max(horizon_list)
    )

    rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    total = len(split_list) * len(target_names) * len(horizon_list) * len(model_factories)
    done = 0

    for split in split_list:
        for target in target_names:
            for horizon in horizon_list:
                # Reference forecast, shared by every model in this cell.
                naive = NaiveForecaster().fit(
                    frame, target, horizon, config, train_end=split.train_end
                )
                naive_prediction = naive.predict_origins(frame, split.test)
                actual = naive.actuals(frame, split.test)
                anchor = frame[target].reindex(split.test)

                for model_name, factory in model_factories.items():
                    started = time.perf_counter()
                    try:
                        block = _run_cell(
                            frame,
                            config,
                            split,
                            target,
                            horizon,
                            model_name,
                            factory,
                            conformal=conformal,
                            naive_prediction=naive_prediction,
                            actual=actual,
                            anchor=anchor,
                            calibration_rows=calibration_rows,
                        )
                        rows.append(block)
                    except Exception as exc:
                        logger.warning(
                            "split %d / %s / h=%d / %s failed: %s",
                            split.index,
                            target,
                            horizon,
                            model_name,
                            exc,
                        )
                    runtime_rows.append(
                        {
                            "split": split.index,
                            "target": target,
                            "horizon": horizon,
                            "model": model_name,
                            "seconds": time.perf_counter() - started,
                        }
                    )
                    done += 1
                    if progress and done % max(total // 20, 1) == 0:
                        logger.info("backtest %d/%d cells", done, total)

    if not rows:
        raise RuntimeError("every backtest cell failed; see the warnings above")

    predictions = pd.concat(rows, ignore_index=True)
    return BacktestResults(
        predictions=predictions,
        splits=splits_summary(split_list),
        calibration=pd.DataFrame(calibration_rows),
        runtime=pd.DataFrame(runtime_rows),
    )


def _run_cell(
    frame: pd.DataFrame,
    config: dict[str, Any],
    split: Split,
    target: str,
    horizon: int,
    model_name: str,
    factory: ModelFactory,
    *,
    conformal: bool,
    naive_prediction: pd.Series,
    actual: pd.Series,
    anchor: pd.Series,
    calibration_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """One (split, target, horizon, model) cell of the backtest."""
    base = factory()

    if conformal:
        model: ConformalPredictor = build_conformal(base, config)
        model.fit(frame, target, horizon, config, train_end=split.train_end)
        model.calibrate(frame, split.calibration)
        interval = model.predict_interval(frame, split.test)
        prediction = interval["prediction"]
        if model.summary is not None:
            calibration_rows.append(
                {
                    "split": split.index,
                    "target": target,
                    "horizon": horizon,
                    "model": model_name,
                    **_flatten_quantiles(model.summary.as_dict()),
                }
            )
    else:
        base.fit(frame, target, horizon, config, train_end=split.train_end)
        prediction = base.predict_origins(frame, split.test)
        interval = pd.DataFrame({"prediction": prediction}, index=split.test)

    block = pd.DataFrame(
        {
            "split": split.index,
            "model": model_name,
            "target": target,
            "horizon": horizon,
            "origin": split.test,
            "target_date": base.target_dates(frame, split.test)
            if not conformal
            else model.base_model.target_dates(frame, split.test),
            "anchor": anchor.to_numpy(),
            "prediction": prediction.to_numpy(),
            "actual": actual.to_numpy(),
            "naive": naive_prediction.to_numpy(),
        }
    )
    for column in interval.columns:
        if column.startswith(("lower_", "upper_", "alpha_")) or column == "scale":
            block[column] = interval[column].to_numpy()
    return block


def _flatten_quantiles(summary: dict[str, Any]) -> dict[str, Any]:
    flat = {k: v for k, v in summary.items() if not isinstance(v, dict)}
    for key in ("quantiles", "lower_quantiles", "upper_quantiles"):
        for level, value in (summary.get(key) or {}).items():
            flat[f"{key.removesuffix('s')}_{level}"] = value
    return flat


def default_model_factories(
    config: dict[str, Any], *, include_prophet: bool = True
) -> dict[str, ModelFactory]:
    """The model line-up compared throughout the project."""
    from mru_forecast.models.baseline import (
        DriftForecaster,
        MeanReversionForecaster,
        SeasonalNaiveForecaster,
    )
    from mru_forecast.models.prophet_model import build_prophet
    from mru_forecast.models.xgboost_model import XGBoostForecaster

    season = int(config["models"]["baseline"]["seasonal_naive"]["season_length"])
    factories: dict[str, ModelFactory] = {
        "naive": NaiveForecaster,
        "seasonal_naive": lambda: SeasonalNaiveForecaster(season),
        "drift": DriftForecaster,
        "mean_reversion": MeanReversionForecaster,
        "xgboost": XGBoostForecaster,
    }
    if include_prophet:
        factories["prophet"] = lambda: build_prophet(config)
    return factories
