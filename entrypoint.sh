#!/bin/sh
set -e

case "${1:-api}" in
  api)
    exec uvicorn run_ui:app --host 0.0.0.0 --port "${API_PORT:-8001}" --workers "${UVICORN_WORKERS:-1}"
    ;;
  cli)
    shift
    exec python run.py "$@"
    ;;
  test)
    exec pytest -q
    ;;
  *)
    exec "$@"
    ;;
esac
