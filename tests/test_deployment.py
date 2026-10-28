"""Phase 15: the container configuration is internally consistent.

Static checks on `docker-compose.yml`, `Dockerfile` and the entrypoint. They are
cheap and they catch a class of mistake that a healthcheck cannot report and a
unit test never sees: a service that starts, answers nothing useful, and is
declared healthy anyway.
"""

from __future__ import annotations

import re

import pytest
import yaml

from mru_forecast.config import PROJECT_ROOT

COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
ENTRYPOINT_PATH = PROJECT_ROOT / "docker" / "entrypoint.sh"

# The port each service listens on *inside* the container.
CONTAINER_PORTS = {"api": 8000, "dashboard": 8501}


@pytest.fixture(scope="module")
def compose() -> dict:
    if not COMPOSE_PATH.exists():
        pytest.skip("docker-compose.yml not present")
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entrypoint() -> str:
    if not ENTRYPOINT_PATH.exists():
        pytest.skip("docker/entrypoint.sh not present")
    return ENTRYPOINT_PATH.read_text(encoding="utf-8")


class TestPortConsistency:
    """The published target, the listen port and the healthcheck must agree.

    Regression test: `API_PORT` once selected *both* the host port and the
    in-container listen port. A `.env` with `API_PORT=8002` then published
    host:8002 -> container:8000 while uvicorn bound container:8002 — nothing was
    served, and the healthcheck (which curls container:8000) reported unhealthy
    with no explanation of why.
    """

    @pytest.mark.parametrize("service", sorted(CONTAINER_PORTS))
    def test_published_target_is_the_container_port(self, compose, service):
        mappings = compose["services"][service]["ports"]
        targets = [int(str(m).rsplit(":", 1)[-1]) for m in mappings]
        assert CONTAINER_PORTS[service] in targets, (
            f"{service} publishes to {targets}, expected the container port "
            f"{CONTAINER_PORTS[service]}"
        )

    @pytest.mark.parametrize("service", sorted(CONTAINER_PORTS))
    def test_listen_port_is_pinned_not_inherited(self, compose, service):
        """`SERVER_PORT` must be set explicitly, so `.env` cannot move it."""
        environment = compose["services"][service].get("environment", {})
        assert "SERVER_PORT" in environment, (
            f"{service} does not pin SERVER_PORT; a value in .env would change "
            "the in-container listen port and silently break the mapping"
        )
        assert int(environment["SERVER_PORT"]) == CONTAINER_PORTS[service]

    @pytest.mark.parametrize("service", sorted(CONTAINER_PORTS))
    def test_healthcheck_targets_the_listen_port(self, compose, service):
        test = compose["services"][service]["healthcheck"]["test"]
        command = " ".join(test) if isinstance(test, list) else str(test)
        port = CONTAINER_PORTS[service]
        assert f":{port}" in command, f"{service} healthcheck does not probe :{port}: {command}"

    def test_entrypoint_uses_server_port_not_api_port(self, entrypoint):
        """`API_PORT` is for running the API directly; it must not reach the container."""
        assert "SERVER_PORT" in entrypoint
        assert "${API_PORT" not in entrypoint, (
            "entrypoint reads API_PORT, which also selects the host port"
        )
        assert "${DASHBOARD_PORT" not in entrypoint


class TestEntrypoint:
    @pytest.mark.parametrize("command", ["api", "dashboard", "pipeline", "backtest"])
    def test_every_documented_command_is_handled(self, entrypoint, command):
        assert re.search(rf"^\s*{command}\)", entrypoint, re.M), f"no case for '{command}'"

    def test_pipeline_is_idempotent(self, entrypoint):
        """A repeated `docker compose up` must not retrain from scratch."""
        assert "FORCE_RETRAIN" in entrypoint
        assert "models/predictors.pkl" in entrypoint

    def test_pipeline_produces_the_backtest_the_dashboard_needs(self, entrypoint):
        """Two of five dashboard pages read backtest reports."""
        assert "scripts/backtest.py" in entrypoint

    def test_servers_bind_all_interfaces(self, entrypoint):
        """Binding to localhost inside a container makes the port unreachable."""
        assert "0.0.0.0" in entrypoint

    def test_it_fails_fast(self, entrypoint):
        assert entrypoint.startswith("#!") and "set -euo pipefail" in entrypoint


class TestComposeStructure:
    def test_serving_services_wait_for_the_trainer(self, compose):
        for service in CONTAINER_PORTS:
            depends = compose["services"][service]["depends_on"]
            assert depends["trainer"]["condition"] == "service_completed_successfully"

    def test_artifacts_live_on_shared_volumes(self, compose):
        """The models trained once must be the models served."""
        declared = set(compose["volumes"])
        for service in (*CONTAINER_PORTS, "trainer"):
            mounts = {str(v).split(":")[0] for v in compose["services"][service].get("volumes", [])}
            assert {"mru-models", "mru-data"} <= (mounts & declared) | mounts

    def test_optional_services_are_behind_profiles(self, compose):
        """`docker compose up` must not start Postgres or MLflow unasked."""
        for service in ("mlflow", "postgres", "backtest"):
            assert compose["services"][service].get("profiles")

    def test_trainer_does_not_restart(self, compose):
        assert compose["services"]["trainer"]["restart"] == "no"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    if not DOCKERFILE_PATH.exists():
        pytest.skip("Dockerfile not present")
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


class TestDockerfile:
    def test_runs_as_a_non_root_user(self, dockerfile):
        assert re.search(r"^USER\s+(?!root)", dockerfile, re.M)

    def test_is_multi_stage(self, dockerfile):
        """The compiler prophet needs must not ship in the runtime image."""
        assert len(re.findall(r"^FROM ", dockerfile, re.M)) >= 2

    def test_copies_the_config_the_app_reads(self, dockerfile):
        assert "configs" in dockerfile

    def test_healthcheck_tool_is_installed(self, dockerfile):
        assert "curl" in dockerfile


class TestDockerignore:
    def test_secrets_and_artifacts_are_excluded(self):
        path = PROJECT_ROOT / ".dockerignore"
        if not path.exists():
            pytest.skip(".dockerignore not present")
        content = path.read_text(encoding="utf-8")
        for entry in (".env", ".venv/", "data/", "models/", ".git/"):
            assert entry in content, f"{entry} would be copied into the image"
