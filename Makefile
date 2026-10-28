# MRU-Forecast — common commands.  Run `make help` for the list.
.DEFAULT_GOAL := help
PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup install check lint format test test-fast coverage \
        data dataset eda train backtest backtest-fast predict all \
        notebooks notebook api dashboard mlflow docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# --- environment -------------------------------------------------------------
setup:  ## Create the virtual environment (Python 3.12) and install everything
	/usr/bin/python3.12 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e '.[dev]'
	@test -f .env || cp .env.example .env
	@echo "Done. Activate with: source .venv/bin/activate"

install:  ## Reinstall the project in editable mode
	$(PIP) install -e '.[dev]'

check:  ## Verify the environment is correctly installed
	$(PYTHON) scripts/check_environment.py

# --- pipeline ----------------------------------------------------------------
data:  ## Phase 2: fetch every configured series into data/interim
	$(PYTHON) scripts/download_data.py

dataset:  ## Phase 3: clean and validate into data/processed
	$(PYTHON) scripts/build_dataset.py

eda:  ## Phase 4: exploratory analysis + Plotly figures in docs/figures
	$(PYTHON) scripts/run_eda.py

train:  ## Phases 10 & 14: train + calibrate the production predictors, log to MLflow
	$(PYTHON) scripts/train.py

backtest:  ## Phases 9 & 11: full expanding-window backtest (~15 min, includes Prophet)
	$(PYTHON) scripts/backtest.py

backtest-fast:  ## Backtest without Prophet (~1 min)
	$(PYTHON) scripts/backtest.py --no-prophet

predict:  ## Forecast from the trained artefacts
	$(PYTHON) scripts/predict.py --horizon 7

all: data dataset eda train backtest-fast  ## Run the whole pipeline end to end

# --- quality -----------------------------------------------------------------
lint:  ## Run the linter
	.venv/bin/ruff check .

format:  ## Format the code and fix auto-fixable lint errors
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

test:  ## Run the full test suite
	$(PYTHON) -m pytest -m "not network"

test-fast:  ## Run tests, skipping the slow and network-bound ones
	$(PYTHON) -m pytest -m "not slow and not network"

coverage:  ## Test suite with a coverage report
	$(PYTHON) -m pytest -m "not network" --cov=mru_forecast --cov=api --cov-report=term-missing

# --- apps --------------------------------------------------------------------
notebooks:  ## Regenerate and execute the analysis notebooks
	$(PYTHON) scripts/make_notebooks.py
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb

notebook:  ## Start JupyterLab
	.venv/bin/jupyter lab

api:  ## Serve the FastAPI application on :8000
	.venv/bin/uvicorn api.main:app --reload --port 8000

dashboard:  ## Serve the Streamlit dashboard on :8501
	.venv/bin/streamlit run dashboard/app.py

mlflow:  ## Start the MLflow tracking UI on :5000
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# --- docker ------------------------------------------------------------------
docker-up:  ## Build and start the full stack (trainer -> api + dashboard)
	docker compose up --build

docker-down:  ## Stop the stack and remove the containers
	docker compose down

# --- housekeeping ------------------------------------------------------------
clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info .coverage htmlcov
