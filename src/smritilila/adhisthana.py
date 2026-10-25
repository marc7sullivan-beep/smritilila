# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Adhiṣṭhāna (2026-10-05, Marc's sign-off): the seat the routes stand on.
In Vedānta the adhiṣṭhāna is the ground an appearance rests on, the rope
under the snake; Smritilila's world is Maya, and this is what it is loaded
onto and served from: the app object and its lifespan, which world this
port serves (the marker, the pin, the registry), the WorldSession held on
app.state, the store it is written to, the embedder and the generative
roles, and the pacing constants the session reads.

Moved here whole from main.py, nothing inside changed. One rule came with
the move: tests patch these services by name, so outside this module
they are called through it (`adhisthana.get_world()`), never imported
bare -- a bare import would be a second binding that a patch here does
not reach, and a missed patch on _resolve_roles is a real, paid model
call; on get_world_store, the database DATABASE_URL names.

From main.py's docstring, moved here with the code it explains (2026-10-05):

Scope, unchanged from before except for world SELECTION (2026-09-11,
smritilila.worlds_registry): exactly ONE world held on app.state at a time
-- still no several-worlds-concurrently routing -- but which database
that one world's data lives in is now switchable via POST
/worlds/select among a small registry (worlds.json), rather than fixed
to a single DATABASE_URL for the life of the process.

Who is present in a scene is read from containment (smrti/kosha.py,
v4) -- not from a caller-maintained roster.

Session config (scene_owner_entity_id, entry_point_location_ids,
playable_entity_ids) IS persisted (smritilila.persistence's session_config
table, one row -- WorldStore.save_session_config/load_session_config),
fixing the gap the old app had for scene_owner_actor_id: genesis and
/world/join save it, and get_world's reload path
prefers it over the starter-world literal-id heuristic, which now only
covers a Postgres world persisted before this existed. npc_entity_ids no
longer exists as session config at all -- NPC participation is derived
fresh from scene presence every pulse, so there was nothing to recover
on reload for that one anyway.

Generative roles are real (smritilila.llm.*), constructed lazily inside each
route's own body (see _resolve_roles), same Depends()-avoidance reasoning
as the old app: a live smoke test against a real running server showed
FastAPI resolves Depends() params before a route's own body validation
runs, so an eagerly-Depends()'d dependency that can fail (a missing API
key, an unreachable database) would mask a plain malformed-request 422
behind a 503.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable

from pathlib import Path

from fastapi import FastAPI, HTTPException

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.llm.embedder import SentenceTransformerEmbedder
from smritilila.llm import providers
from smritilila.llm.client import LLMClient, RotatingClient
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import NOWHERE, PARTICIPANTS, PARTICIPATION_HALF_LIFE, PARTICIPATION_WINDOW
from smritilila.smrti.samskara import Samskara, keep as keep_samskara
from smritilila.smrti.kosha import Kosha
from smritilila.smrti.kala import AGE_LEAD, chronicle_ids
from smritilila.smrti import prakriti
from smritilila.sakshi import bind_world, current_world, log_event, set_world
from smritilila.backup import backup_world
from smritilila.world_store import PostgresWorldStore, WorldStore, make_world_store
from smritilila.worlds_registry import WorldRegistryEntry, load_worlds_registry, registry_stamp
from smritilila.schemas import PulseResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warms the local embedder at process startup instead of on whichever
    live player's request happens to be first after a start/restart.

    Live investigation, 2026-09-08: get_embedder() (below) already caches
    SentenceTransformerEmbedder on app.state so it's only ever constructed
    ONCE per process -- but "once" used to mean "whenever the first /pulses
    request needs it," synchronously blocking that request. Measured at
    ~9.2s (mostly `import sentence_transformers` itself, ~5.1s, plus
    `import torch`, ~1.6s, both one-time per-process costs regardless of
    this fix -- see llm/embedder.py's own docstring for the separate fix
    that dropped this from ~9.2s to ~7s by removing an unwanted Hugging
    Face network round-trip from construction itself). Calling it here
    means a live player's first action after a restart pays nothing extra;
    the cost lands on deploy/restart instead, where it belongs. uvicorn
    (and therefore this app) doesn't accept connections until this
    generator's code before `yield` returns, so there's no request this
    could race against.

    get_embedder() is a plain synchronous call, not offloaded to a thread
    -- correct here specifically because nothing is being served yet (no
    concurrent request's own event-loop turn to block). Then recalls
    the world this port last selected, and starts the describe_daily task
    (routes/bija.py), cancelled at shutdown."""
    get_embedder()
    _recall_active_world()
    from smritilila.routes import bija as bija_routes   # at call time: the routes import this module
    daily = asyncio.create_task(bija_routes.describe_daily())   # the world's one line, drawn again once per real-time day
    try:
        yield
    finally:
        daily.cancel()


def _active_world_marker() -> Path:
    """Where this server remembers its last selected world -- per port, so
    two servers on one working tree (Marc's 8198, the crossing's 8199)
    never change each other's default. Kept beside worlds.json."""
    override = os.environ.get("SMRITILILA_ACTIVE_WORLD_MARKER")  # tests point this at a temp dir
    if override:
        return Path(override)
    port = os.environ.get("SMRITILILA_DEV_PORT", "8198")
    from smritilila.paths import data_dir
    return data_dir() / f".smritilila_active_world.{port}"


def _remember_active_world(name: str) -> None:
    """Both halves of "which world is this": the marker on disk, which a
    restart reads, and sakshi's own stamp, which every line written from
    here on carries. Kept in one function on purpose -- a world switch
    that updated only one of the two would leave a run's own record
    disagreeing with the server about where it was."""
    set_world(name)
    try:
        _active_world_marker().write_text(name, encoding="utf-8")
    except OSError as exc:
        log_event("world.remember_failed", error=repr(exc))


def _name_of_the_default_world() -> str | None:
    """What to call the world a process came up on when no marker chose
    one: its registered name if DATABASE_URL happens to be a registered
    world, otherwise the bare database name out of the URL.

    The fallback exists so no line sakshi writes is worldless. A run that
    started without a marker would otherwise carry `world: null` from
    first line to last -- which reads as "no world" when what actually
    happened is "the default one", and those are different answers to
    the question the stamp exists to settle.

    Swallows everything, on principle rather than defensively. This is
    called at startup, from lifespan() by way of _recall_active_world(),
    and its ONLY purpose is to put a name on
    sakshi's lines -- so a malformed worlds.json here raised
    JSONDecodeError out of get_worlds_registry() and the server did not
    start at all, which is the instrument breaking the engine it
    watches (v5 §44: the witness never affects what it witnesses).
    Measured on a hand-corrupted worlds.json, 2026-09-22, the same
    afternoon v5 §44 (then numbered §43) was written. An unnamed world is a worse answer than
    a named one and a far better one than no server."""
    from urllib.parse import urlsplit  # local, matching create_world's own (routes/loka.py)

    try:
        url = os.environ.get("DATABASE_URL")
        if not url:
            return None
        entry = next((w for w in get_worlds_registry() if w.database_url == url), None)
        if entry is not None:
            return entry.name
        tail = urlsplit(url).path.lstrip("/")
        return tail or None
    except Exception:
        return None


def _pinned_world():
    """The world whose own server this is (Marc, 2026-10-01: "I don't want
    people accidentally switching a world someone is playing in"): the
    registered world holding this server's port, or None -- a server no
    world holds may be switched, by its owner, as before."""
    try:
        port = int(os.environ.get("SMRITILILA_DEV_PORT", "0") or 0)
        return next((w for w in get_worlds_registry() if w.port == port), None) if port else None
    except Exception:
        return None


def _recall_active_world() -> None:
    """A restart comes back up on the world it was on (2026-09-19: after a
    restart the server defaulted to worlds.json's first entry while the
    page still believed itself in Ravenmoor, and every action 422'd on
    the player's id). Only a registered world's name is honoured; a
    process that recalls nothing still gets a world stamped on its
    lines, from the default."""
    set_world(_name_of_the_default_world())
    pinned = _pinned_world()
    if pinned is not None:  # this port is that world's own server: nothing else is ever shown here
        app.state.active_database_url = pinned.database_url
        _remember_active_world(pinned.name)
        log_event("world.pinned", name=pinned.name, url=pinned.url)
        return
    try:
        name = _active_world_marker().read_text(encoding="utf-8").strip()
    except OSError:
        return
    entry = next((w for w in get_worlds_registry() if w.name == name), None)
    if entry is None:
        return
    app.state.active_database_url = entry.database_url
    set_world(name)
    log_event("world.recalled", name=name)


app = FastAPI(title="Smritilila", lifespan=lifespan)


@dataclass
class WorldSession:
    canon: Maya
    scene_owner_entity_id: str
    entry_point_location_ids: list[str] = field(default_factory=list)
    # Every entity id a POST /pulses caller may act as via its entity_id
    # field -- always includes scene_owner_entity_id, but isn't limited to
    # it once /world/join has added a second, genuinely switchable
    # character. Defaulted empty so an existing direct WorldSession(...)
    # construction (tests, __post_init__ fallback below) keeps working;
    # see that fallback for what fills it when left empty.
    playable_entity_ids: list[str] = field(default_factory=list)
    # The exact setting description this world was seeded from, None for
    # a world born without one. Session-config bookkeeping, same category
    # as scene_owner_entity_id (persisted via save_session_config/
    # load_session_config below, not world canon) -- read by the world
    # builder (routes/bija.py: expand, the bhuvana, describe), so it
    # survives a process restart instead of only ever existing on this
    # one in-memory object. No pulse is handed it any more: the
    # Narrator's WORLD SETTING prompt section went with v2. (Named POST
    # /world/seed and /world/reset until 2026-09-22; both are v2's and
    # gone -- a v3 world is seeded at /genesis.)
    setting_description: str | None = None
    # bhuvana (2026-09-20): the world at large, fixed at birth; None for
    # worlds born before it or below wide breadth before 2026-10-05, and for one whose account failed to write
    bhuvana: str | None = None
    recent_memories: list[Memory] = field(default_factory=list)
    # The world's own store and name (item 6, 2026-09-30): every write asked the process which store was
    # current when it committed, so a turn begun here and committed after a /worlds/select wrote into the
    # other world's database. A world is written where it was loaded from. None: a world built in a test.
    store: Any = None
    name: str | None = None
    # (pending_activation_narrations and _promoted_entity_ids were here until 2026-09-30: a background turn's
    # telling, left for whichever player next pulsed near the doer -- decided when read, taken by the first,
    # an NPC's own telling in a guest's browser; the ids no page read. What a player sees of a turn is what
    # they witnessed, decided when it is written: pending_witnessed, below.)
    # v3 s19 (2026-09-20, Ravenmoor 88-94: Corwin demanded the transcript
    # four times to Nathaniel's face and Marc never saw it -- an NPC's
    # turn narrates in close third and the page files that on the NPC's
    # card, while "perceived, never voiced" means no tuple is written for
    # the player). What a player sees of a turn taken while they stood by
    # is the PLACE tuple: the impersonal record of what happened where
    # they stand, s5's shared observable, nothing of anyone's interior.
    # Keyed by player id; drained into the player's next pulse response.
    pending_witnessed: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Cadence (2026-09-21): the player tick at which the engaged draw last
    # took a scene, by place -- the scene's NPCs act on every SECOND
    # player pulse there, and the player sets the tempo.
    scene_draws: dict[str, int] = field(default_factory=dict)
    # v4 (2026-09-23): containment, name and action records (v5 §10, §16,
    # v5 §35) -- appended beside the memories, loaded with them.
    records: list[Record] = field(default_factory=list)
    # v5 §35: the suggestions the page last offered each player; an action
    # sent back unchanged is recorded as accepted, not typed (an exact
    # string comparison). Not persisted: a restart forgets what was offered.
    offered: dict[str, set[str]] = field(default_factory=dict)
    # v5 §33: one pulse per scene at a time.
    scene_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    # the world builder, one at a time: genesis, a join, an expansion, the world at large each check the
    # world and then plant into it, and a second one landing between the check and the plant must not
    builder_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # v5 §38: the tick an arrival last interrupted an engaged scene --
    # one turn in SCENE_QUIET_TICKS across all near scenes, not per scene
    last_arrival_interrupt: int | None = None
    # v5 §37 (Marc, 2026-09-23): the players who allow the world to act for
    # them (persisted), and who is online (runtime: explicit, with an idle
    # timeout as fallback -- any pulse of theirs counts as being here).
    autonomous: set[str] = field(default_factory=set)
    # the joins lately answered, by what was asked (entry and self-description): a retry gets its own newcomer
    # back (the Crossing, 2026-09-28: a join answered in 13 s was sent again twice -- three Nara Millwrights)
    recent_joins: dict = field(default_factory=dict)
    presence: dict[str, str] = field(default_factory=dict)
    last_seen: dict[str, float] = field(default_factory=dict)
    # Marc, 2026-09-24: who logged off (persisted -- an explicit log off;
    # from 2026-09-25 also thirty minutes without acting, _log_off_idle),
    # and who is back and not yet narrated so (returning, below)
    logged_off: set[str] = field(default_factory=set)
    played_by: dict[str, str] = field(default_factory=dict)  # the tether: which character each guest plays (Marc, 2026-09-30)
    # WAITING (Marc, 2026-10-01: "keep teo active, just don't take any turns until someone interacts with him"):
    # players the world acts for (also in autonomous) who take no turn of their own until touched -- in the world,
    # seen, answering in others' turns; drawn only after someone else's pulse writes them a tuple
    waiting: set[str] = field(default_factory=set)
    # a guest's retired characters, out of the world until a god reviews them (Marc, 2026-10-02: "creating gold
    # dragons is questionable"): not seen, not drawn, not voiced; released, they are the world's like anyone
    held_back: set[str] = field(default_factory=set)
    returning: set[str] = field(default_factory=set)
    # v5 §22: a pulse held for its player's consent (a long jump), by player -- a player is never held at a
    # locked scene (v5 §21: a place lived further along is open to them; what was lived there is out of reach)
    held: dict[str, dict] = field(default_factory=dict)
    # when this process took the world up: a player not seen since has been inactive since then
    started: float = field(default_factory=time.monotonic)
    # the world's kept reading of its history (smrti/samskara.py), read on at each use with what was
    # committed since -- not rebuilt from history each pulse (2026-09-25)
    _samskara: "Samskara | None" = None
    # (npc_cap, npc_from, npc_shown and npc_theirs were here until 2026-09-30: what a drawn turn was given,
    # left on the world for the turn to pick up once its scene was free -- and replaced meanwhile by any other
    # pulse's gate. A turn is now handed what it was given: _run_smrti_activation_in_background's arguments.)
    # saṅgama (Marc, 2026-09-27): a round open in a scene where players stand together, by scene --
    # {"actions": {player: text}, "sources": {player: "typed" | "suggested"}, "order": [first to act first],
    # "running": bool}; and each round's telling for its players, until their page collects it. Not persisted:
    # a restart forgets an open round (nothing was written) and an uncollected telling (the memory holds it).
    rounds: dict[str, dict] = field(default_factory=dict)
    round_told: dict[str, "PulseResponse"] = field(default_factory=dict)
    # the actions received by the page's own id, recent ones only: a retry joins the one already running or done
    client_pulses: dict[str, "asyncio.Future"] = field(default_factory=dict)
    typed: dict[str, float] = field(default_factory=dict)  # saṅgama: when each player's page last reported typing
    # saṅgama: each player's first suggestion, as last offered -- taken for them when a round goes on without their action
    first_suggestion: dict[str, str] = field(default_factory=dict)

    def absent(self) -> frozenset[str]:
        """Out of the world: logged off, and not allowing the world to act
        for them (Marc: "entirely suspended from the game, not present at
        all"), or held back. One who allows it stays in the world (waiting: no turn
        of their own until touched, v4.58)."""
        return frozenset(p for p in self.logged_off if p not in self.autonomous) | frozenset(self.held_back)

    def left_at(self, pid: str) -> int:
        """The last tick of a player's own stream: when they were last in the world."""
        last = self.samskara().mem.last_by.get(pid)
        return last.world_time if last is not None else self.canon.world_time

    def samskara(self) -> "Samskara":
        """The kept reading, as the world now stands: read on with what was
        committed since it was last asked for, or built afresh (logged, with why)."""
        start = time.perf_counter()
        sk, why = keep_samskara(self._samskara, self.canon, self.recent_memories, self.records, self.playable_entity_ids,
                                self.world_data.get("clock"), self.world_data.get("capabilities"))
        if why is not None or sk is not self._samskara:
            log_event("samskara.built" if why else "samskara.read_on", why=why, memories=len(self.recent_memories),
                      records=len(self.records), ms=round((time.perf_counter() - start) * 1000, 1))
        self._samskara = sk
        return sk

    def here(self, eid: str, absent: bool = True) -> tuple[str | None, list[str]]:
        """Where eid is and who is with it now, by containment (v5 §11) --
        from the kept reading."""
        sk = self.samskara()
        k = sk.kosha_for(self.canon, self.absent() if absent else ())
        t = (sk.mem.max_tick if sk.mem.max_tick >= 0 else self.canon.world_time) + 1
        scene = k.scene_of(eid, t)
        if scene is None:
            return None, []
        return scene, sorted(k.occupants(scene, t, exclude={eid}))

    def come_back(self, pid: str) -> bool:
        """A logged-off player is here again: their next pulse narrates the
        return. True if they had been logged off."""
        if pid not in self.logged_off:
            return False
        was_absent = pid in self.absent()
        self.logged_off.discard(pid)
        if was_absent:
            self.returning.add(pid)
        return True

    def online(self, pid: str) -> bool:
        """Its name: sānnidhya (सान्निध्य), presence, nearness (v5 §37)."""
        if self.presence.get(pid) == "offline":
            return False
        seen = self.last_seen.get(pid)
        return seen is not None and time.monotonic() - seen < PLAYER_IDLE_SECONDS

    def unvoiced(self) -> frozenset[str]:
        """Players the Narrator never voices in another's turn (v5 §37): all
        of them, except the offline who allow the world to act for them."""
        return frozenset(p for p in self.playable_entity_ids if self.online(p) or p not in self.autonomous)

    def movable(self) -> frozenset[str]:
        """Players another's turn may move (v5 §37): away, and allowing the world
        to act for them -- unvoiced()'s complement. It counted whoever had
        allowed it, online or not, until 2026-09-30: both Stonemarch players
        had, and each was moved by the other's solo turn 8 times while playing
        (19:06 on the 29th: Jakal pulled out of Corl's cabin into Dunstan's
        past at the keep gate)."""
        return frozenset(p for p in self.playable_entity_ids if p in self.autonomous and not self.online(p))

    # v5 §7-§8: the world's own view (None: the default), its library of
    # becomings, its world data (pacing, capability defaults) -- persisted
    darsana: str | None = None
    becomings: dict[str, str] = field(default_factory=dict)
    world_data: dict = field(default_factory=dict)

    def kosha(self) -> Kosha:
        """A reader's own fork of the kept containment index (not built from history)."""
        return self.samskara().kosha_for(self.canon, self.absent())

    def view(self) -> dict:
        return {"darsana": self.darsana, "becomings": dict(self.becomings), "world_data": dict(self.world_data)}

    def pace(self) -> dict:
        """v5 §7: pacing is the world's -- its world data over today's values."""
        return {"scene_quiet_ticks": SCENE_QUIET_TICKS, "quiet_turns": QUIET_TURNS, "chronicle_every": CHRONICLE_EVERY, "age_lead": AGE_LEAD,
                "far_every": FAR_EVERY, "near_window": NEAR_WINDOW, "scene_cadence": SCENE_CADENCE, "arrivals": True,
                "arrivals_on_traffic": True,
                "participants": PARTICIPANTS, "participation_window": PARTICIPATION_WINDOW,
                "participation_half_life": PARTICIPATION_HALF_LIFE,
                **(self.world_data.get("pacing") or {})}

    def scene_lock(self, scene: str | None) -> asyncio.Lock | None:
        if not scene or scene == NOWHERE:
            return None
        return self.scene_locks.setdefault(scene, asyncio.Lock())

    @asynccontextmanager
    async def in_scene(self, scene: str | None, source: str, entity_id: str | None = None):
        """v5 §33: one turn in a scene at a time -- held while a turn reads
        the scene and commits into it; a wait over 50 ms is witnessed."""
        lock = self.scene_lock(scene)
        waited = time.monotonic()
        if lock is not None:
            await lock.acquire()
        try:
            wait_ms = int((time.monotonic() - waited) * 1000)
            if wait_ms > 50:
                log_event("smrti.scene_lock_wait", source=source, entity_id=entity_id, wait_ms=wait_ms)
            yield
        finally:
            if lock is not None:
                lock.release()

    def __post_init__(self) -> None:
        if not self.playable_entity_ids and self.scene_owner_entity_id:
            self.playable_entity_ids = [self.scene_owner_entity_id]


def get_world_store() -> WorldStore:
    """Lazy singleton, same pattern as get_world's own in-process cache.
    NOT a FastAPI Depends() for the same body-validation-ordering reason
    as everything else in this module; tests monkeypatch this name
    directly to substitute a fake that records calls instead of touching
    a database.

    Uses app.state.active_database_url once POST /worlds/select has set
    one -- make_world_store's own fallback to the DATABASE_URL env var
    covers every caller before that ever happens, so this is a strict
    addition, not a behavior change for a server that never selects a
    world at all."""
    store = getattr(app.state, "world_store", None)
    if store is None:
        store = make_world_store(getattr(app.state, "active_database_url", None))
        app.state.world_store = store
    return store


def get_worlds_registry() -> list[WorldRegistryEntry]:
    """Loaded from worlds.json (smritilila.worlds_registry), and loaded again
    whenever the file changes -- each world has its own server, so a
    world added by one (Austin, 2026-10-05) was missing from every other
    server's selector until a restart. Cost: one stat per call. A reload
    that fails (a half-written file) keeps the list already held, and
    tries again on the next call. Tests monkeypatch this name directly,
    same as get_world_store, rather than touching the filesystem."""
    registry = getattr(app.state, "worlds_registry", None)
    stamp = registry_stamp()
    if registry is None or stamp != getattr(app.state, "worlds_registry_stamp", None):
        try:
            registry = load_worlds_registry()
        except Exception:
            if registry is None:
                raise
            return registry
        app.state.worlds_registry = registry
        app.state.worlds_registry_stamp = stamp
    return registry


def _active_world_registry_entry() -> WorldRegistryEntry | None:
    """Whichever registry entry's own database_url matches the
    currently-active one -- None if the active database isn't a
    registered world at all (e.g. a bare DATABASE_URL with no
    worlds.json, or a test database), in which case there's nothing
    for the registry-driven backup (routes/loka.py's clear_world) to name."""
    active_url = getattr(app.state, "active_database_url", None) or os.environ.get("DATABASE_URL")
    return next((w for w in get_worlds_registry() if w.database_url == active_url), None)


def get_embedder() -> SentenceTransformerEmbedder:
    """Lazy singleton, same pattern as get_world_store -- constructing a
    SentenceTransformerEmbedder loads a real local sentence-transformers
    model on first call (no external API key, but a real model load), not
    something to pay for on every request. Tests monkeypatch this name
    directly to substitute a fake, same as get_world_store."""
    embedder = getattr(app.state, "embedder", None)
    if embedder is None:
        embedder = SentenceTransformerEmbedder()
        app.state.embedder = embedder
    return embedder


# One load of the world at a time (2026-09-30): the page asks several things at once, and every request that
# found no world loaded loaded it whole -- 55 double loads in 143 runs, the last one to finish kept, and a write
# made meanwhile on one of the others would have been lost from memory while the store had it.
_world_load_lock = asyncio.Lock()


async def get_world() -> WorldSession:
    """In-process cache on app.state. The store is only consulted on the
    FIRST access after a process starts (or after a select): load() what's
    there, or begin an empty world waiting on /genesis. scene_owner_entity_id/entry_point_location_ids are
    recovered from the store's own saved session config (see
    save_session_config's callers) when one was saved; the starter-world
    literal-id heuristic below is now only a fallback for a Postgres world
    persisted before that existed, or one whose saved config points at an
    entity this canon no longer has. One request loads; the others wait
    for it and take what it loaded."""
    world = getattr(app.state, "world", None)
    if world is not None:
        bind_world(world.name)  # this request's lines, and its background turns', name the world it took
        await _log_off_idle(world)
        return world
    async with _world_load_lock:
        world = getattr(app.state, "world", None)
        if world is None:  # else loaded by the request this one waited on
            world = await _load_world()
        bind_world(world.name)
        return world


def _store(world: "WorldSession") -> WorldStore:
    """Where `world` is written: the store it was loaded from, never
    whichever the process has selected since (item 6)."""
    return world.store if world.store is not None else get_world_store()


async def _load_world() -> WorldSession:
    store = get_world_store()
    if isinstance(store, PostgresWorldStore):
        await _bring_to_head(getattr(app.state, "active_database_url", None) or os.environ["DATABASE_URL"])
    loaded = await store.load()
    if loaded is None:
        # v3: an empty store is an empty world, waiting on /genesis -- the
        # v2 starter world (player, Mara, a square, a sword) is history
        world = WorldSession(canon=Maya(world_time=0), scene_owner_entity_id="")
    else:
        canon = loaded
        saved_config = await store.load_session_config()
        # "" is a brief's world before anyone has arrived (no player yet),
        # not a missing config -- the else branch would crown an NPC
        if saved_config is not None and (saved_config[0] in canon.entities or saved_config[0] == ""):
            scene_owner_entity_id, entry_point_location_ids, playable_entity_ids, setting_description = saved_config
            # Defensive, same spirit as the scene_owner check above: drop
            # any playable id this canon no longer actually has (a stale
            # row surviving an incomplete /world/clear) rather than carry
            # forward a switch target that would 404/422 on every attempt.
            playable_entity_ids = ([eid for eid in playable_entity_ids if eid in canon.entities]
                                   or ([scene_owner_entity_id] if scene_owner_entity_id else []))
        else:
            scene_owner_entity_id = "player" if "player" in canon.entities else next(iter(canon.entities), "")
            entry_point_location_ids = ["square"] if "square" in canon.entities else []
            playable_entity_ids = [scene_owner_entity_id]
            setting_description = None
        world = WorldSession(
            canon=canon,
            scene_owner_entity_id=scene_owner_entity_id,
            entry_point_location_ids=entry_point_location_ids,
            playable_entity_ids=playable_entity_ids,
            setting_description=setting_description,
        )
        # The durable history, not an empty cache (live, 2026-09-16): every
        # reader of recent_memories -- retrieval, own_recent_memories, the
        # person-through-memory movement matcher, jijnasa, knows_way_to --
        # had been starting from nothing on each restart.
        world.recent_memories = await store.load_memories()
        world.records = await store.load_records()
        world.autonomous = set(await store.load_autonomy())
        world.logged_off = set(await store.load_logged_off())
        world.waiting = set(await store.load_waiting()) & set(world.playable_entity_ids)
        world.held_back = set(await store.load_held_back()) & set(canon.entities)
        # v4.58: a player the world may act for is waiting -- "active" is gone. One found allowed and not waiting
        # (set before) waits from this load: quiet from now, so only a touch after it wakes them
        newly = (world.autonomous & set(world.playable_entity_ids)) - world.waiting
        for pid in newly:
            if pid in canon.entities:
                canon.entities[pid].last_simulated_tick = canon.world_time
        if newly:
            world.waiting |= newly
            await store.save_waiting(sorted(world.waiting))
            log_event("player.waiting_from_active", players=sorted(newly))
        for pid in world.waiting:  # a waiting player has a quiet mark to be touched after: never none
            if pid in canon.entities and canon.entities[pid].last_simulated_tick is None:
                canon.entities[pid].last_simulated_tick = canon.world_time
        world.played_by = dict(await store.load_played_by())
        world.darsana, world.becomings, world.world_data = _reaped(canon, world.records, await store.load_view())
        world.bhuvana = await store.load_bhuvana()
        log_event("world.loaded", world_time=canon.world_time, entities=len(canon.entities), memories=len(world.recent_memories),
                  records=len(world.records))
    world.store, world.name = store, current_world()
    app.state.world = world
    return world


@dataclass
class GenerativeRoles:
    """The one model client the world runs on (v3: the Narrator is a
    prompt over it, not a role class; bija and the age write over the
    same client). Tests monkeypatch _resolve_roles to hand in a fake."""
    smrti_client: LLMClient | None = None


def build_roles(client: LLMClient) -> GenerativeRoles:
    return GenerativeRoles(smrti_client=client)


def _resolve_roles() -> GenerativeRoles:
    """The indirection point every route actually calls -- NOT wired up
    as a FastAPI Depends(). Confirmed via a live smoke test (a real
    running server, not TestClient) that FastAPI resolves ALL Depends()
    params for a route before that route's own body validation runs.
    Tests monkeypatch this name directly to substitute fakes.

    Which provider narrates: the first provider PRIMARY (the larger model; the small model since 2026-10-10), the second model secondary.
    Marc switched to the second model first on 2026-09-24 and back the same day on
    the evidence: replaying Ravenmoor 554, the second model gave the man a
    name another person bore ("Edward Lowell") 2/2 even when told whose it
    was, and truncated its JSON ~5 times in ~45 calls (3 of 7 on 554);
    the larger model, shown only the namesake line, 0/2. SMRITILILA_PRIMARY_PROVIDER=the second model
    puts the second model first. the second model goes through the JSON shape path, the
    configuration v5 §30 measured and the path THE PORT depends on.
    SMRITILILA_ROTATE_SHARE is the fraction of calls the SECONDARY gets
    (a pinned world's own `rotate_share` in worlds.json wins over it:
    Austin was 0, the mid-size model alone, Marc 2026-10-05, until taken out again the
    same day) on
    purpose (dev_server set 0.8 from 2026-09-26, Marc: the second model narrates
    four pulses in five, and sets 1 from 2026-10-13; default 0.0: the
    secondary only when the primary fails). Every call is stamped `route` on llm.call. With
    one key only, that provider narrates alone."""
    # (2026-10-17: the providers are llm/providers.py's: each is asked in its own order of preference, the one SMRITILILA_PRIMARY_PROVIDER names first.)
    available = providers.narrators(os.environ.get("SMRITILILA_PRIMARY_PROVIDER"))
    if not available:
        raise HTTPException(503, providers.nothing_configured())
    client: LLMClient = available[0]
    if len(available) >= 2:
        pinned = _pinned_world()
        if pinned is not None and pinned.rotate_share is not None:
            share = pinned.rotate_share  # this world's own mix (worlds.json `rotate_share`, Austin 2026-10-05)
        else:
            try:
                share = float(os.environ.get("SMRITILILA_ROTATE_SHARE", "0") or 0)
            except ValueError:
                return build_roles(client)  # a nonsense share leaves the primary alone rather than taking the server down (D5)
        client = RotatingClient(primary=available[0], secondary=available[1], share=share)
    return build_roles(client)


def _earth(canon: Maya) -> str | None:
    """The world's earth (v4.45): its one world at large, or None."""
    ids = chronicle_ids(canon)
    return next(iter(ids)) if len(ids) == 1 else None


def _reaped(canon: Maya, records: list[Record], birth: tuple[str | None, dict, dict]) -> tuple[str | None, dict, dict]:
    """v5 §15: the world's view as its earth bears it. A world whose earth
    bears no seed yet keeps the view it was given at birth -- logged, so an
    unsown world is seen."""
    earth = _earth(canon)
    if prakriti.sown(records, earth):
        return prakriti.reap(records, earth)
    log_event("prakriti.unsown", earth=earth, view=bool(birth[0] or birth[1] or birth[2]))
    return birth


def _stream_of(world: WorldSession, eid: str | None) -> list[Memory]:
    """An entity's own memories, oldest first, from the kept index (samskara's MemIndex) -- the same list, in
    the same order, as a pass over every memory the world holds, without the pass (the pass of 2026-10-05)."""
    return world.samskara().mem.by_entity.get(eid, []) if eid else []


async def _log_off_idle(world: WorldSession) -> list[str]:
    """v5 §37 (Marc, 2026-09-25): thirty minutes without acting logs a player
    off, exactly as the log off button does -- persisted, out of the world
    unless they allow the world to act for them, their return narrated when
    they act again. Activity is a pulse of theirs or coming back online;
    the page's own polling is not (an open tab is not a player). Not seen
    since this process started: inactive since then. Cost: the players, on
    each request."""
    now = time.monotonic()
    gone = [p for p in world.playable_entity_ids
            if p not in world.logged_off and now - world.last_seen.get(p, world.started) >= PLAYER_IDLE_SECONDS]
    if not gone:
        return []
    for p in gone:
        world.logged_off.add(p)
        world.presence[p] = "offline"
        log_event("player.auto_logoff", entity_id=p, idle_s=int(now - world.last_seen.get(p, world.started)),
                  absent=p in world.absent())
    await _store(world).save_logged_off(sorted(world.logged_off))
    return gone


def _alembic_config():
    from alembic.config import Config
    return Config(str(Path(__file__).resolve().parent.parent.parent / "alembic.ini"))


def _migrate_database(database_url: str) -> None:
    """alembic upgrade head against one database, in this process, the
    URL handed to env.py through config.attributes (never the process's
    DATABASE_URL). Called in a worker thread."""
    from alembic import command
    cfg = _alembic_config()
    cfg.attributes["database_url"] = database_url
    command.upgrade(cfg, "head")


async def _bring_to_head(database_url: str) -> None:
    """A world's database at the code's migration head before it is loaded
    (2026-09-30). Only /worlds/create migrated: a world made before a
    migration stayed behind it, and the first write that needed it failed
    (a sandhi record, refused by an older records_kind_check). Behind, it
    is backed up first -- the chain holds destructive steps -- then
    upgraded; at head, this is one query."""
    import asyncpg
    from alembic.script import ScriptDirectory
    head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
    conn = await asyncpg.connect(database_url.replace("postgresql+asyncpg", "postgresql"))
    try:
        at = await conn.fetchval("select version_num from alembic_version")
    except asyncpg.UndefinedTableError:
        at = None
    finally:
        await conn.close()
    if at == head:
        return
    entry = next((w for w in get_worlds_registry() if w.database_url == database_url), None)
    backup = await asyncio.to_thread(backup_world, entry, reason="pre-migrate") if entry is not None else None
    await asyncio.to_thread(_migrate_database, database_url)
    log_event("world.migrated", world=entry.name if entry else None, at=at, head=head, backup=str(backup) if backup else None)


SCENE_QUIET_TICKS = 12  # v5 §38: a quiet scene is asked "did anyone come?" at most this often (v3's ROAD_QUIET_TICKS)


PLAYER_IDLE_SECONDS = 30 * 60  # v5 §37: a player who has not acted this long is logged off (Marc, 2026-09-25; was: offline until they act)


QUIET_TURNS = 3  # v5 §38: a scene whose last this-many shared-record tuples were all use:none is quiet


CHRONICLE_EVERY = 24  # itihasa: on a far tick, the age moves if it has not moved in this many ticks -- ahead of any far actor


FAR_EVERY = 8  # v5 §38: an actor far from every player is drawn only on far ticks, tick % this in (0, 1), a two-tick window


NEAR_WINDOW = 20  # a player's last N tuples define the places that count as near


# The engaged draw fires on every SCENE_CADENCE-th of the player's own
# pulses at a place: 2 = every second one. Counted in player pulses, not
# ticks -- when the scene waits and nothing else in the world acts, no
# tick is spent, and a tick count would make it wait twice.
SCENE_CADENCE = 2


def _fold_in(world: WorldSession, entities: Iterable[Entity], world_time: int) -> None:
    """What a commit wrote, into the live world -- and nothing else: the
    entities it wrote (as the store has them) replace or join the live ones,
    and the clock never goes back. Every commit path folds in this way (the
    pulse's, an NPC's, the world builder's); none swaps a copy of the world
    in, since other turns commit while it awaits the store."""
    for ent in entities:
        world.canon.entities[ent.id] = ent
    world.canon.world_time = max(world.canon.world_time, world_time)


def _extend_history(world: WorldSession, memories: Iterable[Memory], records: Iterable[Record] = ()) -> None:
    """A commit's memories and records onto the live lists -- appended,
    never the lists replaced: another commit may have appended meanwhile
    (an NPC turn's snapshot replaced them, and an expansion's first tuples
    vanished from the running world while the store kept them)."""
    world.recent_memories.extend(memories)
    world.records.extend(records)
