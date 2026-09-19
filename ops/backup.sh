#!/usr/bin/env bash
# Back up the database and the content-addressed document store.
#
# Vectors are ~40% of the dump and are fully reproducible from the documents, but
# regenerating them means re-reading the corpus (hours at Tier 1), so they are included
# by default. Pass --no-vectors for a small, slow-to-restore backup.
set -euo pipefail
DEST="${1:-$HOME/library-agent-backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$DEST/$STAMP"
mkdir -p "$OUT"

EXCLUDE=""
if [ "${2:-}" = "--no-vectors" ]; then
  EXCLUDE="--exclude-table-data=embedding"
  echo "note: excluding vectors; restore requires re-embedding the corpus"
fi

echo "==> database"
pg_dump --format=custom $EXCLUDE library_agent > "$OUT/library_agent.dump"

echo "==> documents"
DOCS="$(python3 - <<'PY'
import pathlib, tomllib, os
print(os.environ.get("LIBRARY_STORAGE_DIR") or (pathlib.Path.home() / ".library-agent" / "documents"))
PY
)"
if [ -d "$DOCS" ]; then
  tar -czf "$OUT/documents.tar.gz" -C "$(dirname "$DOCS")" "$(basename "$DOCS")"
else
  echo "  (no document store at $DOCS)"
fi

du -sh "$OUT"/* 2>/dev/null
echo
echo "restore:"
echo "  createdb library_agent && psql -d library_agent -c 'CREATE EXTENSION vector; CREATE EXTENSION pg_trgm;'"
echo "  pg_restore -d library_agent $OUT/library_agent.dump"
echo "  tar -xzf $OUT/documents.tar.gz -C \$HOME/.library-agent/"
