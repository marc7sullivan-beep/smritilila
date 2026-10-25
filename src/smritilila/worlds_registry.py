# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Registry of named worlds a client may select among -- Design decision,
2026-09-11: database-per-world, one world active in-process at a time,
backed by a small explicit config file rather than a shared-database
world_id column (no schema/migration touches any existing table -- a
world just IS a database, same pattern this project already uses for
`smritilila` vs. `smritilila_test`) or Postgres database introspection (no
elevated permissions needed, explicit and version-controllable over
discovered).

worlds.json lives at the repo root by default, resolved relative to
the package's own location (smritilila.paths), not the process's cwd -- the identical fix
scripts/dev_server.py already applies to SMRITILILA_SAKSHI_DIR, for the
identical reason: a Browser-pane-launched process's cwd is not
reliably the repo root, so a bare relative path can silently resolve
somewhere else entirely. SMRITILILA_WORLDS_CONFIG overrides the path for
anyone who wants a different location.

An entry is a name and a database url. It also carried `smrti_mode`
from 2026-09-18 until 2026-09-22: "run this world on the smrti loop
whatever the process default is", back when there was a v2 loop to run
instead. There is one loop now, nothing ever read the field, and a
switch that does nothing is worse than no switch -- removed, along with
SMRITILILA_SMRTI_MODE, SMRITILILA_JIJNASA_MODE and SMRITILILA_SHADOW_ASK in
scripts/dev_server.py, which no reader consulted either. An existing
worlds.json that still carries the key loads fine: unknown keys are
ignored.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from smritilila.paths import data_dir

_DEFAULT_CONFIG_PATH = data_dir() / "worlds.json"


@dataclass(frozen=True)
class WorldRegistryEntry:
    name: str
    database_url: str
    # Gods (Marc, 2026-09-30): Marc and Anu are gods of every world; `gods` names the others -- whoever
    # authored this world with them, by Marc's invitation. `port`: an author's world runs on a server of
    # its own, so making it never switches what a shared server shows (option 1).
    gods: tuple[str, ...] = ()
    port: int | None = None
    # one line drawn from the world's root (Marc, 2026-09-30: "so people have a good idea of what world they
    # are entering") -- kept with the list of worlds, for those outside deciding whether to enter
    description: str | None = None
    # when that line was last drawn (ISO, UTC): a world's line is drawn again once per real-time day (Marc, 2026-10-20)
    described_at: str | None = None
    # Pinned (Marc, 2026-10-01): the server on `port` shows this world and no other, and `url` is the world's own
    # public address -- choosing a world is going to its address, never switching what a server shows. `guests`:
    # open to guests (beside SMRITILILA_GUEST_WORLDS) -- true for any guest, or the emails of the only guests it is open
    # to (Marc, 2026-10-01: two friends into the Crossing, "they will both respect it").
    url: str | None = None
    guests: bool | tuple[str, ...] = False
    # A world's own Narrator mix (Marc, 2026-10-05: Austin, present day, where a character works real mathematics,
    # is narrated by the mid-size model alone -- Flash handed out a breakthrough its own pulse four earlier had ruled out):
    # the share of calls the secondary takes on this world's server, over SMRITILILA_ROTATE_SHARE. None: the server's.
    rotate_share: float | None = None


def load_worlds_registry(path: str | Path | None = None) -> list[WorldRegistryEntry]:
    """A missing config file returns an empty list, not an error -- a
    fresh checkout with no registry configured yet is a normal state
    (GET /worlds just reports no options; the app's existing single-
    DATABASE_URL behavior is otherwise untouched)."""
    config_path = Path(path) if path is not None else Path(os.environ.get("SMRITILILA_WORLDS_CONFIG", _DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        return []
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return [
        WorldRegistryEntry(name=w["name"], database_url=w["database_url"],
                           gods=tuple(g.strip().lower() for g in w.get("gods", []) if g.strip()), port=w.get("port"),
                           description=w.get("description"), described_at=w.get("description_at"), url=w.get("url"), guests=_guests(w.get("guests")),
                           rotate_share=_share(w.get("rotate_share")))
        for w in data.get("worlds", [])
    ]


def registry_stamp(path: str | Path | None = None) -> int | None:
    """When worlds.json last changed (its mtime in ns), or None if it is missing -- so a running server can see
    a world added after it started (Marc, 2026-10-05: Austin was missing from every older server's selector)."""
    config_path = Path(path) if path is not None else Path(os.environ.get("SMRITILILA_WORLDS_CONFIG", _DEFAULT_CONFIG_PATH))
    try:
        return config_path.stat().st_mtime_ns
    except OSError:
        return None


def _share(value) -> float | None:
    """A share between 0 and 1, or None (the server's own) for anything else -- a typo never takes a world down."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0.0 <= value <= 1.0 else None


def _guests(value) -> bool | tuple[str, ...]:
    if isinstance(value, list):
        return tuple(g.strip().lower() for g in value if isinstance(g, str) and g.strip())
    return bool(value)


_LOCK_STALE_SECONDS = 30
_LOCK_WAIT_SECONDS = 15


@contextlib.contextmanager
def _locked(config_path: Path):
    """One writer of worlds.json at a time, across the processes of this machine (2026-10-20): a world's server writes its own line once a
    day while another may be adding a world, and each rewrites the whole file, so two at once would lose one's change. The lock is a file
    made with O_EXCL beside it; one left behind by a process that died is taken over after 30 seconds."""
    lock = config_path.with_name(config_path.name + ".lock")
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except (FileExistsError, PermissionError):   # Windows says "permission denied", not "exists", for a lock file whose deletion is still pending
            try:
                if time.time() - lock.stat().st_mtime > _LOCK_STALE_SECONDS:
                    lock.unlink()
                    continue
            except OSError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"{lock} stayed held for {_LOCK_WAIT_SECONDS} s")
            time.sleep(0.02)
    try:
        yield
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _write_whole(config_path: Path, data: dict) -> None:
    """The file whole or not at all: written aside, moved into place, so a reader never sees half of it."""
    tmp = config_path.with_name(f"{config_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for attempt in range(25):
        try:
            os.replace(tmp, config_path)
            return
        except PermissionError:   # Windows refuses while a reader (a server loading the registry, a scanner) has the old file open: a moment, then again
            if attempt == 24:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.02)


def add_world(name: str, database_url: str, path: str | Path | None = None, gods: tuple[str, ...] = (),
              port: int | None = None) -> WorldRegistryEntry:
    """Register a world (POST /worlds/create): appended to worlds.json,
    refused if the name is taken. The file is rewritten whole, other
    entries untouched, under the lock."""
    config_path = Path(path) if path is not None else Path(os.environ.get("SMRITILILA_WORLDS_CONFIG", _DEFAULT_CONFIG_PATH))
    entry = {"name": name, "database_url": database_url}
    if gods:
        entry["gods"] = [g.strip().lower() for g in gods]
    if port is not None:
        entry["port"] = port
    with _locked(config_path):
        data = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {"worlds": []}
        if any(w.get("name", "").lower() == name.lower() for w in data.get("worlds", [])):
            raise ValueError(f"a world named {name!r} is already registered")
        data.setdefault("worlds", []).append(entry)
        _write_whole(config_path, data)
    return WorldRegistryEntry(name=name, database_url=database_url, gods=tuple(entry.get("gods", ())), port=port)


def set_world_description(name: str, description: str, path: str | Path | None = None, at: datetime | None = None) -> bool:
    """Keep `description` as world `name`'s one line, with the moment it was drawn (`at`, now by default; kept as `description_at`, ISO, UTC);
    False if no world has that name. Under the lock, the file written whole."""
    config_path = Path(path) if path is not None else Path(os.environ.get("SMRITILILA_WORLDS_CONFIG", _DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        return False
    with _locked(config_path):
        data = json.loads(config_path.read_text(encoding="utf-8"))
        entry = next((w for w in data.get("worlds", []) if w["name"] == name), None)
        if entry is None:
            return False
        entry["description"] = description
        entry["description_at"] = (at or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_whole(config_path, data)
    return True


# -- invitations (Marc, 2026-09-30: "only by a god's invitation"; "only i can invite for now") ----------
# Outside every world, beside worlds.json: the emails invited to author a world, each spent by the world
# it makes. Not committed (.gitignore): it holds people's emails.

_DEFAULT_INVITES_PATH = data_dir() / "invites.json"


def _invites_path() -> Path:
    return Path(os.environ.get("SMRITILILA_INVITES_CONFIG", _DEFAULT_INVITES_PATH))


def load_invites() -> list[str]:
    p = _invites_path()
    return list(json.loads(p.read_text(encoding="utf-8")).get("invites", [])) if p.exists() else []


def add_invite(email: str) -> None:
    email = email.strip().lower()
    invites = load_invites()
    if email and email not in invites:
        _invites_path().write_text(json.dumps({"invites": invites + [email]}, indent=2) + "\n", encoding="utf-8")


def take_invite(email: str) -> bool:
    """Spend an invitation: True if `email` held one (now gone)."""
    email = email.strip().lower()
    invites = load_invites()
    if email not in invites:
        return False
    _invites_path().write_text(json.dumps({"invites": [i for i in invites if i != email]}, indent=2) + "\n", encoding="utf-8")
    return True



def set_world_guests(name: str, guests: bool | list[str], path: str | Path | None = None) -> bool:
    """Keep `guests` as world `name`'s guest setting: true for any guest, false for none, or the emails of the only
    guests it is open to (the admin screen, 2026-10-06). False if no world has that name. Under the lock, the file written
    whole (2026-10-22: it had been the one writer that skipped both), other entries untouched; the servers see the change on
    their next look at the file."""
    config_path = Path(path) if path is not None else Path(os.environ.get("SMRITILILA_WORLDS_CONFIG", _DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        return False
    with _locked(config_path):
        data = json.loads(config_path.read_text(encoding="utf-8"))
        entry = next((w for w in data.get("worlds", []) if w["name"] == name), None)
        if entry is None:
            return False
        entry["guests"] = [g.strip().lower() for g in guests if g.strip()] if isinstance(guests, list) else bool(guests)
        _write_whole(config_path, data)
    return True
