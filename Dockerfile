# syntax=docker/dockerfile:1
# =============================================================================
# MRU-Forecast — one image, three entry points (api | dashboard | cli).
#
# Multi-stage: the builder compiles wheels (prophet pulls in cmdstan/pystan and
# needs a toolchain); the runtime carries only the installed packages, so the
# compiler never ships to production.
# =============================================================================

# --- builder -----------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install into a virtualenv that is copied wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && pip install .

# --- runtime -----------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MRU_CONFIG_PATH=configs/config.yaml

# curl is used by the healthchecks in docker-compose.yml.
RUN apt-get update && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 mru

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=mru:mru pyproject.toml README.md ./
COPY --chown=mru:mru src ./src
COPY --chown=mru:mru configs ./configs
COPY --chown=mru:mru scripts ./scripts
COPY --chown=mru:mru api ./api
COPY --chown=mru:mru dashboard ./dashboard
COPY --chown=mru:mru docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Data, models and reports are bind-mounted or volume-mounted at run time; the
# directories must exist and be writable by the non-root user either way.
RUN mkdir -p data/raw data/interim data/processed data/external models reports mlruns \
    && chown -R mru:mru /app \
    && chmod +x /usr/local/bin/entrypoint.sh

USER mru
EXPOSE 8000 8501

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["api"]
