"""Phase 12: the dashboard renders without raising.

Streamlit's ``AppTest`` runs the real script in-process and records every
exception, which is the only way to catch "page crashes on click" without a
browser.  Each page is exercised through a tiny driver so that a failure names
the page it came from.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from mru_forecast.config import PROJECT_ROOT

PAGES = ["page_overview", "page_usd", "page_eur", "page_models", "page_uncertainty"]


class TestStreamlitEntryPoint:
    """The app must import under Streamlit's own sys.path, not pytest's.

    `streamlit run dashboard/app.py` sets sys.path[0] to the *script's
    directory* and does not put the working directory on the path. The driver
    scripts below add the repository root, which is what a test needs to import
    `dashboard.app` as a module — but it is also help the real entry point does
    not get. That difference once hid a crash that only appeared in the
    container, so the real entry point is exercised here on its own terms.
    """

    def test_app_runs_with_streamlit_path_setup(self):
        script = textwrap.dedent(
            f"""
            import pathlib, runpy, sys
            root = pathlib.Path({str(PROJECT_ROOT)!r})
            # Exactly what Streamlit's script runner does.
            sys.path.insert(0, str(root / "dashboard"))
            runpy.run_path(str(root / "dashboard" / "app.py"), run_name="__main__")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            # PYTHONSAFEPATH keeps the working directory off sys.path, which is
            # the condition that made this fail under `streamlit run`.
            env={**os.environ, "PYTHONSAFEPATH": "1"},
            cwd=tempfile.gettempdir(),
        )
        assert "ModuleNotFoundError" not in result.stderr, result.stderr[-2000:]
        assert result.returncode == 0, result.stderr[-2000:]


@pytest.fixture(scope="module")
def driver_dir(tmp_path_factory):
    """One tiny script per page, each calling a single page function."""
    directory = tmp_path_factory.mktemp("dashboard_drivers")
    for name in PAGES:
        (directory / f"{name}.py").write_text(
            textwrap.dedent(
                f"""
                import sys, warnings
                warnings.filterwarnings("ignore")
                sys.path.insert(0, {str(PROJECT_ROOT)!r})
                import streamlit as st
                st.set_page_config(layout="wide")
                import dashboard.app as app
                app.{name}()
                """
            ).strip()
        )
    return directory


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(driver_dir, page):
    app = AppTest.from_file(str(driver_dir / f"{page}.py"), default_timeout=300)
    app.run()
    errors = [str(e.value) for e in app.exception]
    assert not errors, f"{page} raised: {errors[0]}"


@pytest.mark.slow
def test_overview_shows_the_current_rates(driver_dir):
    app = AppTest.from_file(str(driver_dir / "page_overview.py"), default_timeout=300)
    app.run()
    labels = {m.label for m in app.metric}
    assert {"USD/MRU", "EUR/MRU"} <= labels


@pytest.mark.slow
def test_every_page_states_the_disclaimer(driver_dir):
    """Provenance and the experimental status must be on every page, not just one."""
    for page in PAGES:
        app = AppTest.from_file(str(driver_dir / f"{page}.py"), default_timeout=300)
        app.run()
        text = " ".join(c.value for c in app.caption)
        if not app.warning:  # a page short-circuits when artefacts are missing
            assert "Banque Centrale de Mauritanie" in text, f"{page} omits the disclaimer"


def test_components_expose_the_disclaimer_text():
    from dashboard.components import DISCLAIMER

    assert "Banque Centrale de Mauritanie" in DISCLAIMER
    assert "derived cross" in DISCLAIMER
    assert "Not investment advice" in DISCLAIMER
