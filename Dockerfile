# The Library in a container: the API and the worker from one image.
#
#   docker compose up -d          # Postgres, Redis, the API and the worker; see docker-compose.yml
#
# Models are not in the image. Ollama runs where the GPU is -- on the host, another box, or
# the compose file's optional `ollama` service -- and the library reaches it over HTTP.
# The reranker (a small cross-encoder, CPU torch) is in the image unless built with
# --build-arg WITH_RERANKER=0, which saves ~1 GB; Find and Ask then go unreranked.

FROM python:3.12-slim-bookworm

ARG WITH_RERANKER=1

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl postgresql-client lsof \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

# Dependencies first, so a code change does not reinstall torch.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
      $( [ "$WITH_RERANKER" = "1" ] && echo "--extra ml" )

COPY alembic.ini ./
COPY alembic ./alembic
COPY src ./src
COPY web ./web
COPY docs ./docs
COPY bench ./bench
COPY library ./library
COPY docker/entrypoint.sh /usr/local/bin/entrypoint

# The project itself, installed in place: the app serves ./web relative to its source.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
      $( [ "$WITH_RERANKER" = "1" ] && echo "--extra ml" )

# Everything the library keeps outside Postgres lives under its home: ~/.library-agent
# (originals, figures, cartridges, providers, modules, classification) and the model
# caches. One volume at /data keeps all of it across rebuilds.
RUN useradd --create-home --home-dir /data --uid 1000 library \
 && chown -R library /app
USER library

ENV HOME=/data \
    PATH=/app/.venv/bin:$PATH \
    UV_NO_SYNC=1 \
    LIBRARY_BIND=0.0.0.0 \
    LIBRARY_PORT=8077

EXPOSE 8077
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -sf http://127.0.0.1:8077/api/health >/dev/null || exit 1

ENTRYPOINT ["entrypoint"]
CMD ["serve"]
