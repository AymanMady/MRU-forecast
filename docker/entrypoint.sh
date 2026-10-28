#!/usr/bin/env bash
# Single entrypoint, selected by the first argument.
#   api        -> FastAPI via uvicorn
#   dashboard  -> Streamlit
#   pipeline   -> download -> clean -> train, then exit (used by the `trainer` service)
#   backtest   -> full backtest, then exit
#   shell      -> anything else, run verbatim
set -euo pipefail

COMMAND="${1:-api}"
shift || true

wait_for_artifacts() {
  # The API and the dashboard are useless without trained models. Rather than
  # crash-looping, wait for the trainer service to produce them.
  local tries="${ARTIFACT_WAIT_TRIES:-60}"
  for _ in $(seq 1 "$tries"); do
    if [ -f models/predictors.pkl ]; then return 0; fi
    echo "waiting for models/predictors.pkl ..." >&2
    sleep 5
  done
  echo "artefacts not found after $((tries * 5))s; starting anyway (degraded mode)" >&2
}

case "$COMMAND" in
  api)
    wait_for_artifacts
    # SERVER_PORT, not API_PORT: inside a container the listen port is fixed and
    # docker-compose maps it to whatever host port the user wants. Reusing
    # API_PORT for both made a `.env` with API_PORT=8002 publish host:8002 ->
    # container:8000 while the app listened on container:8002 — nothing served,
    # and the healthcheck failed.
    exec uvicorn api.main:app \
      --host "${SERVER_HOST:-0.0.0.0}" --port "${SERVER_PORT:-8000}" "$@"
    ;;
  dashboard)
    wait_for_artifacts
    exec streamlit run dashboard/app.py \
      --server.address 0.0.0.0 \
      --server.port "${SERVER_PORT:-8501}" \
      --server.headless true \
      --browser.gatherUsageStats false "$@"
    ;;
  pipeline)
    # Idempotent: `docker compose up` is run repeatedly, and re-fetching and
    # retraining every time would cost minutes for nothing. Set FORCE_RETRAIN=1
    # to rebuild anyway (a new trading day, or a config change).
    if [ "${FORCE_RETRAIN:-0}" != "1" ] && [ -f models/predictors.pkl ] \
       && { [ "${SKIP_BACKTEST:-0}" = "1" ] || [ -f reports/metrics_point.csv ]; }; then
      echo "models/predictors.pkl already exists; skipping the pipeline." >&2
      echo "Set FORCE_RETRAIN=1 to rebuild." >&2
      exit 0
    fi
    python scripts/download_data.py "$@"
    python scripts/build_dataset.py
    python scripts/train.py
    # The Model-comparison and Uncertainty pages read backtest reports; without
    # them two of the dashboard's five pages come up empty. The fast variant
    # (no Prophet, ~70s) is enough to populate them. Run the full backtest with
    # `docker compose --profile backtest run backtest`.
    if [ "${SKIP_BACKTEST:-0}" != "1" ]; then
      python scripts/backtest.py --no-prophet
    fi
    ;;
  backtest)
    exec python scripts/backtest.py "$@"
    ;;
  *)
    exec "$COMMAND" "$@"
    ;;
esac
