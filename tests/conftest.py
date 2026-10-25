# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Shared test setup: the active-world marker (adhisthana._active_world_marker)
goes to a temp file, never the repo root -- a test selecting "Ravenmoor
Test" once steered the real 8198 server's next restart."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="smritilila-test-")
os.environ.setdefault("SMRITILILA_ACTIVE_WORLD_MARKER", os.path.join(_TMP, "active_world"))
# nor the real worlds.json or invitations (2026-09-30: a world's line is written to the registry)
os.environ.setdefault("SMRITILILA_WORLDS_CONFIG", os.path.join(_TMP, "worlds.json"))
os.environ.setdefault("SMRITILILA_INVITES_CONFIG", os.path.join(_TMP, "invites.json"))
# nor the database DATABASE_URL names (2026-10-05): a test on the plain client fixture reaches the real
# get_world_store, which falls back to it. Unset, the store is in memory; a test wanting one sets its own,
# and the live tests use TEST_DATABASE_URL alone.
os.environ.pop("DATABASE_URL", None)
# nor the ntfy topic (2026-10-05): it is set in Marc's user environment, and six tests reach the real _alert, which
# posts to it -- a run from a fresh terminal would push the tests' guests to his phone. Unset, nothing is sent.
os.environ.pop("SMRITILILA_NTFY_TOPIC", None)
# the Host allow-list (dvara._host_refusal): Starlette's TestClient and httpx's ASGI transport say "testserver" / "test"
os.environ.setdefault("SMRITILILA_ALLOWED_HOSTS", "testserver,test")
