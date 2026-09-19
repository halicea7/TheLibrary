#!/usr/bin/env bash
# Phase 0 bootstrap. Re-runnable.
#
# Prerequisite (needs sudo, run once by hand):
#   sudo xcodebuild -license accept
# Homebrew compiles pgvector against postgresql@16, and git itself is gated on the
# same license, so nothing below works until that is accepted.
set -euo pipefail

echo "==> pgvector"
# Homebrew's pgvector formula only ships builds for postgresql@17/@18; this box runs
# postgresql@16, so build from source against that pg_config instead.
PG_CONFIG="${PG_CONFIG:-/opt/homebrew/opt/postgresql@16/bin/pg_config}"
EXT_DIR="$($PG_CONFIG --sharedir)/extension"
if [ ! -f "$EXT_DIR/vector.control" ]; then
  tmp="$(mktemp -d)"
  git clone --depth 1 https://github.com/pgvector/pgvector.git "$tmp/pgvector"
  make -C "$tmp/pgvector" PG_CONFIG="$PG_CONFIG"
  make -C "$tmp/pgvector" install PG_CONFIG="$PG_CONFIG"
  rm -rf "$tmp"
fi

echo "==> database"
createdb library_agent 2>/dev/null || true
psql -d library_agent -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -d library_agent -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"

echo "==> models"
ollama list | grep -q "bge-m3"     || ollama pull bge-m3
ollama list | grep -q "qwen3:4b"   || ollama pull qwen3:4b

echo "==> migrations"
uv run alembic revision --autogenerate -m "initial schema" 2>/dev/null || true
uv run alembic upgrade head

echo "==> verify"
psql -d library_agent -tAc "select count(*) || ' tables' from information_schema.tables where table_schema='public';"
uv run python -m library_agent.healthcheck
