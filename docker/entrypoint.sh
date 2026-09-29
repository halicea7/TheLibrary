#!/bin/sh
# The container's entrypoint.
#
#   serve     migrate, then the API (the default)
#   worker    the background reader: Tier 1/2, figures, OCR, threads
#   migrate   run the migrations and exit
#   anything else runs as given, e.g.
#     docker compose run --rm api ./library import /import --read
#     docker compose run --rm api ./library bench init
set -e
cd /app

wait_for_db() {
  i=0
  until alembic current >/dev/null 2>&1; do
    i=$((i + 1))
    [ "$i" -ge 60 ] && { echo "Postgres did not come up at $LIBRARY_DATABASE_URL" >&2; exit 1; }
    sleep 1
  done
}

case "${1:-serve}" in
  serve)
    wait_for_db
    alembic upgrade head
    exec uvicorn library_agent.api.app:app \
      --host "${LIBRARY_BIND:-0.0.0.0}" --port "${LIBRARY_PORT:-8077}" --log-level warning
    ;;
  worker)
    wait_for_db
    # The API migrates; wait until it has, so the worker never sees an old schema.
    until alembic current 2>/dev/null | grep -q "(head)"; do
      sleep 2
    done
    exec arq library_agent.worker.tasks.WorkerSettings
    ;;
  migrate)
    wait_for_db
    exec alembic upgrade head
    ;;
  *)
    exec "$@"
    ;;
esac
