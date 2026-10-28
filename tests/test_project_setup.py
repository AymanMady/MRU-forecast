"""Phase 1 tests: the skeleton is importable and internally consistent.

These are cheap structural guarantees. They fail loudly if a later phase
removes a directory, drops a config section, or breaks the package layout.
"""

from __future__ import annotations

import pytest

import mru_forecast
from mru_forecast.config import Settings, get_settings, load_config, resolve_path


class TestPackage:
    def test_package_is_importable_and_versioned(self):
        assert mru_forecast.__version__ == "0.1.0"

    @pytest.mark.parametrize(
        "module",
        [
            "mru_forecast.data",
            "mru_forecast.features",
            "mru_forecast.models",
            "mru_forecast.evaluation",
            "mru_forecast.pipeline",
            "mru_forecast.utils",
        ],
    )
    def test_subpackages_are_importable(self, module):
        __import__(module)


class TestProjectLayout:
    @pytest.mark.parametrize(
        "relative",
        [
            "configs/config.yaml",
            "pyproject.toml",
            ".env.example",
            ".gitignore",
            "README.md",
            "data/raw",
            "data/processed",
            "data/external",
            "notebooks",
            "scripts",
            "api",
            "dashboard",
        ],
    )
    def test_expected_paths_exist(self, project_root, relative):
        assert (project_root / relative).exists(), f"missing: {relative}"

    def test_env_file_is_not_committed(self, project_root):
        """`.env` may exist locally, but .gitignore must exclude it."""
        ignored = (project_root / ".gitignore").read_text(encoding="utf-8")
        assert "\n.env\n" in ignored


class TestConfiguration:
    def test_top_level_sections_present(self, config):
        expected = {
            "project",
            "paths",
            "targets",
            "ingestion",
            "validation",
            "cleaning",
            "features",
            "horizons",
            "models",
            "backtesting",
            "conformal",
            "evaluation",
        }
        assert expected <= set(config)

    def test_targets_are_fully_specified(self, config):
        names = {t["name"] for t in config["targets"]}
        assert names == {"USD_MRU", "EUR_MRU"}
        for target in config["targets"]:
            # `rate_type`/`source` are "TBD" until Phase 2, but the keys must
            # exist: the project must never carry an undocumented rate type.
            assert {"base_currency", "quote_currency", "definition", "rate_type", "source"} <= set(
                target
            )
            assert target["quote_currency"] == "MRU"

    def test_horizons_are_positive_and_sorted(self, config):
        horizons = config["horizons"]
        assert horizons == sorted(horizons)
        assert all(isinstance(h, int) and h > 0 for h in horizons)

    def test_conformal_levels_are_valid_probabilities(self, config):
        levels = config["conformal"]["confidence_levels"]
        assert all(0 < lvl < 1 for lvl in levels)
        assert config["conformal"]["primary_level"] in levels

    def test_feature_lags_are_strictly_positive(self, config):
        """A lag of 0 would feed the model the value it is meant to predict."""
        assert all(lag >= 1 for lag in config["features"]["lags"])

    def test_backtesting_is_chronological(self, config):
        bt = config["backtesting"]
        assert bt["scheme"] in {"expanding", "rolling"}
        assert bt["initial_train_size"] > bt["calibration_size"] > 0
        assert bt["step_size"] > 0 and bt["n_splits"] > 0

    def test_config_is_cached(self):
        assert load_config() is load_config()

    def test_resolve_path_returns_absolute_existing_dir(self, project_root):
        path = resolve_path("data_processed")
        assert path.is_absolute() and path.is_dir()
        assert path.is_relative_to(project_root)

    def test_resolve_path_rejects_unknown_key(self):
        with pytest.raises(KeyError):
            resolve_path("no_such_path")


class TestSettings:
    def test_settings_defaults_are_usable_without_env_file(self):
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.mru_env in {"dev", "prod"}
        assert 0 < settings.api_port < 65536

    def test_missing_api_keys_are_reported_as_unavailable(self):
        """Providers without a key must be skipped, not fail at request time."""
        settings = Settings(openexchangerates_app_id="", fred_api_key="   ")
        assert "openexchangerates" not in settings.available_providers()
        assert "fred" not in settings.available_providers()

    def test_configured_key_marks_provider_available(self):
        settings = Settings(fred_api_key="abc123")
        assert "fred" in settings.available_providers()
