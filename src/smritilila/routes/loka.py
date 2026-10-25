# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Loka (2026-10-05, Marc's sign-off): the worlds. Which worlds there are
and which this server shows (/worlds, /worlds/select), making one (its
database, migrated; a server of its own on a free port only for an invited
author, which dvara.GUESTS_MAY_CREATE_WORLDS now shuts, and for the builder,
routes/sthapati.py), inviting an author, backing a world up, asking whether
it exists, reading it whole (GET /world), and clearing it -- backed up first.

Moved whole from main.py, nothing inside changed. backup_world,
_spawn_world_server and _make_world_database are called by name from
here (the builder reaches the last two as loka.X), so a test fakes them
on this module."""

from __future__ import annotations

import asyncio
import re
import os
import sys
import subprocess

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila import dvara  # the gate, called through it for the same reason
from smritilila.adhisthana import WorldSession, _pinned_world, app
from smritilila.sakshi import log_event
from smritilila.backup import backup_world
from smritilila.paths import data_dir
from smritilila.worlds_registry import WorldRegistryEntry
from smritilila.schemas import (
    BackupNowResponse, EntityOut, InviteRequest, WorldCreateRequest, WorldCreateResponse, WorldExistsResponse,
    WorldInfo, WorldSelectRequest, WorldSnapshot,
)
from smritilila.dvara import INVITER, _open_to_guests


router = APIRouter()


def _snapshot(world: WorldSession) -> WorldSnapshot:
    return WorldSnapshot(
        world_time=world.canon.world_time,
        scene_owner_entity_id=world.scene_owner_entity_id,
        playable_entity_ids=list(world.playable_entity_ids),
        entities=[
            EntityOut(id=e.id, labels=list(e.labels), created_at_world_time=e.created_at_world_time, origin=e.origin)
            for e in world.canon.entities.values()
        ],
    )


@router.post("/world/clear", status_code=204)
async def clear_world() -> None:
    """The only way back to an empty canon: without it, genesis's own
    "canon is not empty" check would 409 forever. Wipes both the in-process
    session and the persisted store, so a subsequent POST /genesis with a
    new self-description actually goes through -- the web UI's "new
    world" button called this before showing the start screen, until
    its old forms left the page (2026-09-24).

    Backs up the currently-active world (smritilila.backup) FIRST if it's a
    registered one, and refuses to clear at all (503) if that backup
    fails -- added 2026-09-11 after this exact endpoint destroyed a
    real, months-long playthrough with no recovery path at all (see
    smritilila.backup's own docstring for the incident). An active database
    that isn't in the registry at all (no worlds.json, a bare
    DATABASE_URL, a test database) has no registry entry to back up
    under and proceeds exactly as this endpoint always did before."""
    entry = adhisthana._active_world_registry_entry()
    if entry is not None:
        try:
            await asyncio.to_thread(backup_world, entry, reason="pre-clear")
        except subprocess.CalledProcessError as e:
            raise HTTPException(503, f"Refusing to clear {entry.name!r} -- pre-clear backup failed: {e.stderr}")
    if hasattr(app.state, "world"):
        del app.state.world
    await adhisthana.get_world_store().clear()


@router.get("/worlds", response_model=list[WorldInfo])
def list_worlds(request: Request) -> list[WorldInfo]:
    """Every world registered in worlds.json (smritilila.worlds_registry), to an
    owner; a guest sees only the guest worlds (and those they are a god of) --
    an empty list is not an error, just a server with no registry
    configured yet, in which case the client's own world picker has
    nothing to offer and the app behaves exactly as it always did
    before this endpoint existed (a single DATABASE_URL-backed world,
    no picker rendered at all)."""
    active = adhisthana._active_world_registry_entry()
    guest = dvara._guest(request)
    return [WorldInfo(name=w.name, active=w is active, port=w.port, description=w.description, url=getattr(w, "url", None)) for w in adhisthana.get_worlds_registry()
            if guest is None or _open_to_guests(w, guest) or guest in w.gods]


@router.post("/worlds/invite")
async def invite_author(req: InviteRequest, request: Request) -> dict:
    """An invitation to author a world (Marc, 2026-09-30: "only by a god's
    invitation"; "only i can invite for now") -- Marc's own, through the
    door he is known at; not a guest's, not an author's, not this machine's.
    Its name: āvāhana (आवाहन), the invocation that invites a deity in (v5 §42)."""
    from smritilila.worlds_registry import add_invite
    email = (request.headers.get("cf-access-authenticated-user-email") or "").strip().lower()
    if email != INVITER:
        raise HTTPException(403, "only Marc invites, for now")
    if "@" not in req.email:
        raise HTTPException(422, "an invitation is to an email")
    add_invite(req.email)
    log_event("world.invited", invited=req.email.strip().lower())
    return {"invited": req.email.strip().lower()}


def _free_world_port() -> int:
    """A port for an author's world's own server: from 8200 up, none a
    registered world holds, none in use (8198 and 8199 are the shared ones)."""
    import socket
    held = {w.port for w in adhisthana.get_worlds_registry() if w.port} | {8198, 8199}
    port = 8200
    while True:
        if port not in held:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        port += 1


# The repo root, four steps up from this file (routes/ is a folder deeper than main.py was, where three steps were right; after the
# split of 2026-10-05 the marker and dev_server.py were looked for under src/, and no world could start its own server).
REPO_ROOT = Path(__file__).resolve().parents[3]


def _spawn_world_server(entry: WorldRegistryEntry) -> None:
    """An author's world, on a server of its own (option 1): its marker names
    the world, then the dev server starts detached on its port."""
    marker = data_dir(REPO_ROOT) / f".smritilila_active_world.{entry.port}"   # where the server for this port will look (SMRITILILA_DATA_DIR in a container)
    marker.write_text(entry.name, encoding="utf-8")
    env = {**os.environ, "SMRITILILA_DEV_PORT": str(entry.port)}
    env.pop("SMRITILILA_ACTIVE_WORLD_MARKER", None)
    # CREATE_NO_WINDOW, never DETACHED_PROCESS (scripts/spawn_hidden.py says why): detached, the venv launcher's real python.exe opens a visible terminal window
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
    subprocess.Popen([sys.executable, str(REPO_ROOT / "scripts" / "dev_server.py")],
                     env=env, creationflags=flags, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log_event("world.server_started", name=entry.name, port=entry.port)


def _db_slug(name: str) -> str:
    import unicodedata
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")[:40]
    return "smritilila_" + (slug or "world")


@router.post("/worlds/create", response_model=WorldCreateResponse)
async def create_world(req: WorldCreateRequest, request: Request = None) -> WorldCreateResponse:
    """A new world from the page's "new world" panel (2026-09-20): its
    own database on the same server as the active world's, migrated,
    registered in worlds.json, and selected -- empty, ready for
    /genesis. Database-per-world (2026-09-11) made by the app itself
    rather than by hand. (The page's builder makes its worlds through
    POST /builder/build now, not this route.)"""
    from smritilila.worlds_registry import add_world, load_invites, take_invite
    name = req.name.strip()
    guest = dvara._guest(request) if request is not None else None
    if guest is not None and not (dvara.GUESTS_MAY_CREATE_WORLDS and guest in load_invites()):
        log_event("guest.refused", guest=guest, method="POST", path="/worlds/create", reason="world_creation")
        raise HTTPException(403, "only the owner makes a world")
    if not name:
        raise HTTPException(422, "the world needs a name")
    if guest is None and _pinned_world() is not None:  # making it here would switch this server -- never on a pinned one
        raise HTTPException(409, f"this server keeps {_pinned_world().name}; make a new world from an unpinned server")
    if any(w.name.lower() == name.lower() for w in adhisthana.get_worlds_registry()):
        raise HTTPException(409, f"a world named {name!r} already exists")
    dbname, database_url = await _make_world_database(name)
    if guest is not None:
        # an invited author (option 1): their world on a server of its own; this one shows what it showed
        take_invite(guest)
        entry = add_world(name, database_url, gods=(guest,), port=_free_world_port())
        app.state.worlds_registry = None
        _spawn_world_server(entry)
        log_event("world.created", name=name, database=dbname, god=guest, port=entry.port)
        return WorldCreateResponse(name=name, database=dbname, port=entry.port)
    entry = add_world(name, database_url)
    app.state.worlds_registry = None  # reload from the file
    app.state.active_database_url = entry.database_url
    adhisthana._remember_active_world(entry.name)
    if hasattr(app.state, "world"):
        del app.state.world
    if hasattr(app.state, "world_store"):
        del app.state.world_store
    log_event("world.created", name=name, database=dbname)
    return WorldCreateResponse(name=name, database=dbname)


async def _make_world_database(name: str) -> tuple[str, str]:
    """A new world's own database on the active world's server, migrated: (its name, its url)."""
    from urllib.parse import urlsplit, urlunsplit
    import asyncpg
    base = getattr(app.state, "active_database_url", None) or os.environ.get("DATABASE_URL")
    if not base:
        raise HTTPException(503, "no database server to create the world on")
    parts = urlsplit(base)
    server_url = urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))
    taken: set[str] = set()
    conn = await asyncpg.connect(server_url.replace("postgresql+asyncpg", "postgresql"))
    try:
        taken = {r["datname"] for r in await conn.fetch("select datname from pg_database")}
        dbname = _db_slug(name)
        n = 2
        while dbname in taken:
            dbname = f"{_db_slug(name)}_{n}"
            n += 1
        await conn.execute(f'create database "{dbname}"')
    finally:
        await conn.close()
    database_url = urlunsplit((parts.scheme, parts.netloc, "/" + dbname, "", ""))
    try:
        await asyncio.to_thread(adhisthana._migrate_database, database_url)
    except Exception as exc:
        log_event("world.create_failed", name=name, database=dbname, error=repr(exc))
        raise HTTPException(500, f"the new database could not be migrated: {exc!r}")
    return dbname, database_url


@router.post("/worlds/select", response_model=WorldSnapshot)
async def select_world(req: WorldSelectRequest) -> WorldSnapshot:
    """Switches the single active world (Design decision, 2026-09-11:
    database-per-world, one active in-process at a time -- not a
    shared-database world_id column, not several worlds held
    concurrently) to a different registered database. Clears BOTH
    process-global singletons (app.state.world and .world_store), not
    just one -- world_store itself must be rebuilt against the newly
    selected database_url, not merely reloaded, since get_world_store's
    own lazy-singleton check would otherwise keep returning the OLD
    store untouched. get_world()'s own reload path then does exactly
    what it already does on a fresh process start: load what's there,
    or begin an empty world if the newly-selected database is empty."""
    entry = next((w for w in adhisthana.get_worlds_registry() if w.name == req.name), None)
    if entry is None:
        raise HTTPException(404, f"No registered world named {req.name!r}")
    pinned = _pinned_world()
    if pinned is not None and pinned.name != entry.name:
        where = f" -- it is at {entry.url}" if entry.url else ""
        raise HTTPException(409, f"this server keeps {pinned.name}; it never shows another world{where}")
    app.state.active_database_url = entry.database_url
    adhisthana._remember_active_world(entry.name)
    if hasattr(app.state, "world"):
        del app.state.world
    if hasattr(app.state, "world_store"):
        del app.state.world_store
    return _snapshot(await adhisthana.get_world())


@router.post("/worlds/{name}/backup", response_model=BackupNowResponse)
async def backup_world_now(name: str) -> BackupNowResponse:
    """On-demand pg_dump for one registered world (smritilila.backup),
    independent of the scheduled daily task and the automatic pre-clear
    backup (POST /world/clear) -- for exactly the case that actually
    caused the incident smritilila.backup exists to prevent: "I'm about
    to do something to this world I'm not fully sure about, back it up
    first." Runs immediately, not deferred to a background task, so a
    caller's own success/failure response is trustworthy."""
    entry = next((w for w in adhisthana.get_worlds_registry() if w.name == name), None)
    if entry is None:
        raise HTTPException(404, f"No registered world named {name!r}")
    try:
        path = await asyncio.to_thread(backup_world, entry, reason="manual")
    except subprocess.CalledProcessError as e:
        raise HTTPException(502, f"Backup failed for {name!r}: {e.stderr}")
    return BackupNowResponse(world_name=entry.name, backup_path=str(path))


@router.get("/world/exists", response_model=WorldExistsResponse)
async def world_exists(request: Request) -> WorldExistsResponse:
    """Read-only, side-effect-free existence check -- unlike GET /world
    (via get_world()), which builds a session even when nothing is
    persisted yet. A client's start screen needs to know whether to offer
    "continue as one of these characters" WITHOUT that check itself
    silently creating a world the visitor never asked for; that's exactly
    the bug the old client-side-only localStorage flag (now removed) was
    working around, unreliably, since it only remembered a world existed
    in the ONE browser profile that started it -- a restart, a different
    browser, or a cleared profile all lost that memory even though the
    world itself was still there on the server, with a real caller stuck
    unable to discover or resume it. Only peeks at the store (or the
    in-process cache, if already warm) -- never constructs or persists a
    WorldSession for an empty world (a non-empty one is loaded, as any
    request would load it), so an empty/nonexistent world is left exactly
    as empty as GET /world would otherwise have found it."""
    world = getattr(app.state, "world", None)
    if world is None:
        loaded = await adhisthana.get_world_store().load()
        if loaded is None:
            return WorldExistsResponse(exists=False)
    # A non-empty persisted canon (or an already-warm in-process one) is
    # exactly the condition under which get_world() loads rather than
    # beginning an empty world -- safe to call for real now, and reuses its
    # own already-correct session-config recovery rather than duplicating it.
    world = await adhisthana.get_world()
    guest = dvara._guest(request)
    return WorldExistsResponse(
        exists=True,
        playable_entities={
            eid: list(world.canon.entities[eid].labels)
            for eid in world.playable_entity_ids
            if eid in world.canon.entities and (guest is None or dvara._plays(world, guest, eid))
        },
    )


@router.get("/world", response_model=WorldSnapshot)
async def get_world_snapshot(request: Request) -> WorldSnapshot:
    world = await adhisthana.get_world()
    snap = _snapshot(world)
    guest = dvara._guest(request)
    if guest is not None:
        # a guest sees their own characters and what those characters have met -- never the rest of the
        # world's people and places (unmet ones would spoil it), nor anyone else's character
        own = [p for p in snap.playable_entity_ids if dvara._plays(world, guest, p)]
        met = set(own)
        for p in own:
            for mem in world.samskara().mem.by_entity.get(p, ()):
                met |= set(mem.subject_ids or ()) | ({mem.location_id} if mem.location_id else set())
        others = set(world.playable_entity_ids) - set(own)
        snap.playable_entity_ids = own
        snap.scene_owner_entity_id = own[0] if own else ""
        snap.entities = [e for e in snap.entities if e.id in met and e.id not in others]
    return snap
