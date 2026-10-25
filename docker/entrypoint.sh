#!/bin/sh
# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
# Set the database up (idempotent: makes it if it is not there, brings it to the current schema, plants the example world if it is empty),
# then run the server. Postgres may still be starting, so setup is tried again for a while.
set -e
cd /app

tries=0
until python scripts/serve.py setup; do
  tries=$((tries + 1))
  if [ "$tries" -ge 20 ]; then
    echo "setup did not succeed after $tries tries: is Postgres up, and is DATABASE_URL right?" >&2
    exit 1
  fi
  echo "setup failed (try $tries); trying again in 3 s" >&2
  sleep 3
done

exec python scripts/serve.py run --host "${SMRITILILA_HOST:-0.0.0.0}" --port "${SMRITILILA_PORT:-8198}"
