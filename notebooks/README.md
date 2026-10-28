# Notebooks

Exploration and reporting only. Any logic that matters lives in `src/mru_forecast/`
and is covered by a test — notebooks are not importable and are not part of the
pipeline. They call the same library functions the scripts do, so an analysis here
can never drift from what the pipeline actually runs.

| Notebook | Topic |
|---|---|
| `01_data_exploration.ipynb` | provenance, raw vs cleaned, independent cross-check |
| `02_data_analysis.ipynb` | stationarity, volatility, ACF/PACF, seasonality, correlation |
| `03_baseline_models.ipynb` | backtest design, leakage check, model comparison |
| `04_model_evaluation.ipynb` | conformal mechanics, calibration, volatility regimes |

## Regenerating

The notebooks are **generated from a single definition** so their prose and code stay
in one place:

```bash
make notebooks
# equivalently:
python scripts/make_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Edit `scripts/make_notebooks.py`, not the `.ipynb` files — a hand-edit is overwritten
by the next regeneration. For the same reason `ruff format` skips this directory
(see `pyproject.toml`); `ruff check` still lints it.

They need the built artefacts:

```bash
python scripts/download_data.py && python scripts/build_dataset.py
python scripts/train.py && python scripts/backtest.py
```

Figures use Plotly's `notebook_connected` renderer — a CDN reference rather than an
embedded copy of plotly.js per figure — which keeps each notebook around 100-300 KB
instead of 4-5 MB.
