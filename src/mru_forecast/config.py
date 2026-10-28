"""Configuration loading.

Two distinct sources, deliberately kept apart:

* ``configs/config.yaml`` — reproducible, version-controlled parameters
  (paths, horizons, model hyper-parameters, evaluation settings).
* ``.env`` — machine-specific values and secrets (API keys, database
  credentials).  Never committed.

YAML values may reference environment variables with ``${VAR}``; they are
substituted at load time, which is how the two sources meet without hard-coding
a secret in a tracked file.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Marker that identifies the project root in every deployment shape.
_ROOT_MARKER = Path("configs") / "config.yaml"


def _find_project_root() -> Path:
    """Locate the directory holding ``configs/`` and the data tree.

    Three deployment shapes have to work, and walking up from ``__file__``
    alone only covers the first:

    * **editable install** — the package lives at ``<root>/src/mru_forecast/``;
    * **installed package** (the container) — the code sits in
      ``site-packages/`` while ``configs/``, ``data/`` and ``models/`` live in
      the working directory, so ``__file__`` says nothing useful about the root;
    * **explicit override** — ``MRU_PROJECT_ROOT``, for anything unusual.

    Resolution order: the override, then the working directory, then the
    ancestors of this file. Falling back to the source tree keeps a bare
    ``python -c "import mru_forecast"`` working from anywhere in the repo.
    """
    override = os.environ.get("MRU_PROJECT_ROOT")
    if override:
        return Path(override).resolve()

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / _ROOT_MARKER).exists():
            return candidate

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / _ROOT_MARKER).exists():
            return candidate

    # Nothing found: the src-layout default, so the error message that follows
    # names a plausible path rather than a site-packages directory.
    return here.parents[2]


PROJECT_ROOT = _find_project_root()

# Load .env once, at import time.  `override=False` means a variable already
# exported in the shell (or injected by Docker) wins over the file.
load_dotenv(PROJECT_ROOT / ".env", override=False)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class Settings(BaseSettings):
    """Environment-backed settings (secrets and deployment knobs)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- application ---
    mru_env: str = "dev"
    mru_config_path: str = "configs/config.yaml"
    log_level: str = "INFO"

    # --- data providers (empty string = provider unavailable, skipped) ---
    openexchangerates_app_id: str = ""
    exchangerate_host_api_key: str = ""
    alphavantage_api_key: str = ""
    fred_api_key: str = ""

    # --- mlflow ---
    # SQLite, not the file store: MLflow's filesystem backend is in maintenance
    # mode and raises on recent versions unless MLFLOW_ALLOW_FILE_STORE is set.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "mru-forecast"

    # --- api ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    def available_providers(self) -> list[str]:
        """Names of the providers for which a non-empty API key is configured."""
        candidates = {
            "openexchangerates": self.openexchangerates_app_id,
            "exchangerate_host": self.exchangerate_host_api_key,
            "alphavantage": self.alphavantage_api_key,
            "fred": self.fred_api_key,
        }
        return sorted(name for name, key in candidates.items() if key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, parsed once."""
    return Settings()


def _expand_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with environment values."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` as a plain dict, with ``${VAR}`` expansion.

    Parameters
    ----------
    path
        Config file to read.  Defaults to ``MRU_CONFIG_PATH`` (itself defaulting
        to ``configs/config.yaml``).  Relative paths resolve from the repo root,
        so the same call works from a notebook, a test or a container.
    """
    config_path = Path(path) if path is not None else Path(get_settings().mru_config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return _expand_env(raw)


def resolve_path(key: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve a ``paths.<key>`` entry from the config into an absolute path.

    The directory is created if it does not exist, so callers never have to
    guard writes with ``mkdir``.
    """
    cfg = config if config is not None else load_config()
    try:
        relative = cfg["paths"][key]
    except KeyError as exc:
        available = sorted(cfg.get("paths", {}))
        raise KeyError(f"Unknown path key '{key}'. Available: {available}") from exc

    path = PROJECT_ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    return path
