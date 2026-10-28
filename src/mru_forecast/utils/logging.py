"""Project-wide logging setup.

One configuration function, called by scripts and services at start-up.  Library
modules only ever call ``get_logger(__name__)`` — they never configure handlers
themselves, which keeps log output under the control of the entry point.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def setup_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """Configure the root logger once, writing to stdout.

    Parameters
    ----------
    level
        Log level name or value.  Defaults to ``LOG_LEVEL`` from the environment.
    force
        Reconfigure even if ``setup_logging`` has already run.
    """
    global _configured
    if _configured and not force:
        return

    from mru_forecast.config import get_settings

    resolved = level if level is not None else get_settings().log_level
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

    # Third-party libraries are chatty at INFO; keep the signal readable.
    for noisy in ("cmdstanpy", "prophet", "matplotlib", "urllib3", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str, **kwargs: Any) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    setup_logging(**kwargs)
    return logging.getLogger(name)
