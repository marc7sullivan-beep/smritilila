# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Backs up every registered Smritilila world (smritilila.worlds_registry) via
pg_dump -- built 2026-09-11 directly in response to a real incident:
POST /world/clear was run against the `smritilila` database (the "Frontier"
world's own real, months-long playthrough -- Amber, Jasper, Eliseo,
the Sioux brave, Easton) while testing an unrelated feature, and
clear_all's own full-table wipe (smritilila.persistence.repository.
clear_all, including delta_log, S11's own append-only source of truth)
destroyed it with no recovery path -- no pg_dump, no filesystem
snapshot, nothing. This module exists so that can never happen again
without a recovery point already sitting on disk first.

(delta_log is v2's; it was dropped with the other v2 tables on
2026-09-21 and nothing reads or writes it now -- see
persistence/models.py. The incident above is left as written because
it is why this module exists; in v3 the thing clear_all would destroy
is `memories`, which is append-only in exactly the same way and just
as unrecoverable.)

pg_dump's own custom format (-Fc): compressed, and one pg_restore can
selectively restore from (so can the directory and tar formats) -- a plain
SQL dump can only be replayed in full via psql. Backups land under Smritilila/backups/. Until
2026-09-15 that folder sat inside a cloud-synced folder, which was the
off-machine copy; the repo then moved out of it and
for three weeks nothing was off the machine (found 2026-10-07, after a
Windows update that looked like a factory reset). Now scripts/backup_offdisk.py
copies the dumps and the rest of what lives only on C: to a second disk,
and smritilila.nidhi seals them for a cloud-synced folder. Git-ignored: these are save data,
not source, and don't belong in the code backup repo.

Two callers, same function (it is also called, with their own `reason`, before a
migration -- adhisthana._bring_to_head -- by POST /worlds/{name}/backup in routes/loka.py,
and before a plant in routes/pratistha.py):
  - routes/loka.py's clear_world (POST /world/clear) calls backup_world for
    whichever world is currently active, BEFORE wiping it -- refuses to
    clear at all if the backup itself fails, rather than proceeding
    with no safety net.
  - scripts/backup_worlds.py (run on a schedule -- see
    scripts/register_backup_task.ps1) calls backup_all_worlds for
    everything in the registry, independent of whether anyone is
    actively clearing or even playing anything.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from smritilila.paths import data_dir
from smritilila.worlds_registry import WorldRegistryEntry, load_worlds_registry


_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # a console program started from a process with no console opens a visible window; this keeps it hidden (Marc, 2026-10-10)


def _run_hidden(*a, **k):
    return subprocess.run(*a, creationflags=k.pop("creationflags", _NOWIN), **k)

BACKUPS_DIR = data_dir() / "backups"
# Generous, not tight -- storage was explicitly upgraded to have room
# for this (2026-09-11). Per-world, not global: one world's own backup
# count never crowds out another's retention window.
KEEP_PER_WORLD = 30


def _pg_dump_path() -> str:
    return os.environ.get("PG_DUMP_PATH") or shutil.which("pg_dump") or r"C:\Program Files\PostgreSQL\17\bin\pg_dump.exe"


def _pg_dump_url(database_url: str) -> str:
    """pg_dump wants a plain postgresql:// DSN -- SQLAlchemy's own
    +asyncpg driver suffix (smritilila.persistence.engine's own convention)
    means nothing to it."""
    return re.sub(r"^postgresql\+\w+://", "postgresql://", database_url)


def _safe_dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name) or "world"


@dataclass(frozen=True)
class BackupResult:
    world_name: str
    path: Path | None
    error: str | None = None


def backup_world(entry: WorldRegistryEntry, *, reason: str = "scheduled", backups_dir: Path = BACKUPS_DIR) -> Path:
    """Runs pg_dump for one world, returns the backup file's path.
    Raises CalledProcessError if pg_dump itself fails -- a caller
    backing up before a destructive operation (clear_world) must let
    this propagate and abort, not proceed with no safety net."""
    world_dir = backups_dir / _safe_dirname(entry.name)
    world_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = world_dir / f"{_safe_dirname(entry.name)}_{timestamp}_{reason}.dump"
    _run_hidden(
        [_pg_dump_path(), "-Fc", "-f", str(dest), _pg_dump_url(entry.database_url)],
        check=True, capture_output=True, text=True,
    )
    _prune_old_backups(world_dir)
    return dest


def _prune_old_backups(world_dir: Path, keep: int = KEEP_PER_WORLD) -> None:
    backups = sorted(world_dir.glob("*.dump"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in backups[keep:]:
        stale.unlink()


def backup_all_worlds(*, reason: str = "scheduled", backups_dir: Path = BACKUPS_DIR) -> list[BackupResult]:
    """Best-effort across the whole registry -- one world's own pg_dump
    failure (a stopped Postgres service, a bad connection string) must
    not prevent every OTHER world from still getting backed up this
    run."""
    results = []
    for entry in load_worlds_registry():
        try:
            path = backup_world(entry, reason=reason, backups_dir=backups_dir)
            results.append(BackupResult(entry.name, path))
        except subprocess.CalledProcessError as e:
            results.append(BackupResult(entry.name, None, error=e.stderr))
    return results
