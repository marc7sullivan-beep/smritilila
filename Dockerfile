# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
# Smritilila in a container (2026-10-06). `docker compose up -d` builds this and runs it beside Postgres; see docker-compose.yml and README.md.
#
# The code is installed in place (`pip install -e .` at /app): Smritilila finds its migrations, scripts and example world beside its source.
# What a server writes (the registry of worlds, the witness logs, backups, the builder's conversations) goes to /data, a volume.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SMRITILILA_DATA_DIR=/data \
    HF_HOME=/data/huggingface

# pg_dump: Smritilila backs a world up before it migrates or clears one, and a client older than the server will not dump it, so the client
# is Postgres 17's, from Postgres's own repository. git: the witness log notes which commit a run was made at, if there is one.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release git \
 && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/pgdg.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/pgdg.gpg] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends postgresql-client-17 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (the embedding model needs it; the default wheel carries a GPU stack Smritilila does not use)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml alembic.ini ./
COPY src ./src
COPY migrations ./migrations
COPY scripts ./scripts
COPY worlds ./worlds
RUN pip install -e .

COPY docker/entrypoint.sh /usr/local/bin/smritilila-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/smritilila-entrypoint \
 && chmod +x /usr/local/bin/smritilila-entrypoint \
 && useradd --create-home --uid 1000 smritilila \
 && mkdir -p /data \
 && chown -R smritilila:smritilila /data /app

USER smritilila
EXPOSE 8198
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=10 \
  CMD curl -fsS http://127.0.0.1:8198/health || exit 1

ENTRYPOINT ["smritilila-entrypoint"]
