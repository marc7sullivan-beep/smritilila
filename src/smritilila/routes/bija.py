# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The seed's routes (2026-10-05, Marc's sign-off; the engine is
smritilila.smrti.bija): how a world is born and grows. /genesis, joining
(entry points, the busiest places, /world/join and its arrival into a
live world), /world/expand, /world/describe, the bhuvana and the world's
view, the becomings and containments a god adds, and absorbing one entity
into another.

Moved whole from main.py, nothing inside changed. In tests this module is
imported as bija_routes: `bija` there is the engine.

From main.py's docstring, moved here with the code it explains (2026-10-05):

One way to bootstrap: POST /genesis. The first player's self-
description seeds the world through smrti/bija.py and smrti/genesis.py.
An empty store is an empty world waiting on /genesis -- the v2 starter
world behind POST /world/reset is gone (phase 0, 2026-09-21), and with
it the CommittedDelta path it seeded through.
A second player joins via GET /world/entry-points + POST /world/join
(locked decision, Appendix D item 5 option a: a bounded, engine-offered
set of entry locations, not free-form semantic-match binding). These
create a second player Entity in canon AND add it to
world.playable_entity_ids -- genuinely new ground the old app never had
at all (it only ever supported one hardcoded player), closed by giving
POST /pulses its own optional entity_id field: omitted, a pulse still
acts as world.scene_owner_entity_id exactly as before; supplied, it must
name one of playable_entity_ids and that entity acts instead. Switching
who acts is a per-request choice, not session-wide state -- there's
still no notion of "the currently active character," just "who does
THIS pulse belong to." scene_owner_entity_id itself is never reassigned
by a switch; it stays the world's original/primary character (used for
entry_point_location_ids bookkeeping and as the entity_id default).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import os
import re
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.ids import new_id
from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila import dvara  # the gate, called through it for the same reason
from smritilila.adhisthana import (
    GenerativeRoles, WorldSession, _earth, _extend_history, _fold_in, _store, _stream_of, app,
)
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import NOWHERE
from smritilila.smrti.pulse import mnemonic_id, player_named
from smritilila.smrti.genesis import ITIHASA
from smritilila.smrti.kala import apart
from smritilila.smrti import prakriti
from smritilila.sakshi import current_world, log_event
from smritilila.itihasa import _age_tuples, _at_large, _chronicle, _chronicle_id
from smritilila.schemas import (
    AbsorbResponse, BecomingRequest, BhuvanaRequest, BhuvanaResponse, ContainmentRequest, DescribeResponse,
    EntryPointOut, ExpandResponse, GenesisRequest, GenesisResponse, JoinRequest, JoinResponse, LocationOut,
    WorldView, _labelled,
)
from smritilila.routes.spanda import _queue_witnessed
from smritilila.routes import spanda  # its routes, and the names a test patches there, called through it


router = APIRouter()


async def _v3_genesis(req: GenesisRequest, world: WorldSession, roles: GenerativeRoles) -> GenesisResponse:
    """v3.0 s14 (bija, 2026-09-20): a world begins as tuples. The Narrator
    writes the seed from the self-description and setting; genesis.build
    plants it -- entities with opaque ids, first tuples, session config.
    No candidates, no relations, no provenance, no deltas: nothing v3
    would later have to settle. Same planter as an authored world file."""
    from smritilila.smrti.bija import brief_text, write_bhuvana, write_seed, write_world_seed
    from smritilila.smrti.genesis import build as plant
    pulse_id = new_id()
    # a world from its creator's brief has no player (the new world screen)
    from_brief = not req.self_description.strip()
    brief = brief_text(req.name or "", req.brief()) if from_brief else None
    log_event("pulse.player_action", kind="genesis", pulse_id=pulse_id, entity_id=None, world_time=0,
              action_text=brief or req.self_description, smrti=True, breadth=req.breadth, brief=from_brief)
    bhuvana: str | None = None
    try:
        if req.breadth == "wide":
            # the world at large first, fixed at birth; the seed is its starting region
            bhuvana = await asyncio.to_thread(write_bhuvana, roles.smrti_client, req.self_description, req.setting, brief)
        if from_brief:
            spec = await asyncio.to_thread(write_world_seed, _seeder(roles.smrti_client), req.name or "", req.brief(), req.breadth, bhuvana)
        else:
            spec = await asyncio.to_thread(write_seed, _seeder(roles.smrti_client), req.self_description, req.setting, req.breadth, bhuvana)
    except ValueError as exc:
        raise HTTPException(502, f"the seed could not be planted: {exc}")
    if bhuvana is None:  # below wide: the account follows the seed, so every world has one (Marc, 2026-10-05)
        bhuvana = await _account_after_seed(roles, req, brief, spec)
    entities, memories, config = plant(spec, adhisthana.get_embedder().embed, bhuvana=bhuvana)  # the earth with it (genesis.build)
    if world.canon.entities:
        raise HTTPException(409, "canon is not empty -- another genesis planted first")
    working_canon = Maya(world_time=0)
    for e in entities:
        working_canon.entities[e.id] = e
    player = config["scene_owner_entity_id"]
    opening = spec.get("opening") or next((m.experience for m in reversed(memories) if m.entity_id == player), "")
    # v5 §7-§8: the world's own view, written at birth -- empty where the default holds
    view = config["view"]
    darsana, becomings, world_data = view["darsana"], dict(view["becomings"]), dict(view["world_data"])
    if (req.source or "").strip():
        world_data["source"] = req.source.strip()  # every Narrator call leads with it (narrator.source_line)
    # v5 §15: the seed, laid at the root -- in the birth's own transaction
    earth = _earth(working_canon)
    sown = prakriti.sow(earth, prakriti.EMPTY, (darsana, becomings, world_data), 0, pulse_id) if earth is not None else []
    records = list(config["records"]) + sown
    store = _store(world)
    await store.commit(working_canon, memories, entities, records=records)
    world.records = records
    world.canon = working_canon
    world.scene_owner_entity_id = player
    world.playable_entity_ids = list(config["playable_entity_ids"])
    world.entry_point_location_ids = list(config["entry_point_location_ids"])
    world.setting_description = config["setting_description"] or None
    world.recent_memories = list(memories)
    app.state.world = world
    await store.save_session_config(player, world.entry_point_location_ids, world.playable_entity_ids, world.setting_description)
    world.bhuvana = bhuvana
    await store.save_bhuvana(bhuvana)
    world.darsana, world.becomings, world.world_data = darsana, becomings, world_data
    await store.save_view(darsana, becomings, world_data)  # what the gods gave at birth, kept as given
    log_event("bija.view", darsana=bool(world.darsana), becomings=len(world.becomings), world_data=world.world_data,
              clauses=sum(1 for r in config["records"] if r.kind == "clause"))
    log_event("bija.planted", pulse_id=pulse_id, name=spec["name"], entities=len(entities), tuples=len(memories), breadth=req.breadth)
    return GenesisResponse(player_entity_id=player, narration=opening, narrations={player: opening} if player else {})


async def _account_after_seed(roles: GenerativeRoles, req: GenesisRequest, brief: str | None, spec: dict) -> str | None:
    """The world at large for a world born below wide breadth (Marc, 2026-10-05: every world has one; option B of
    the scratch test -- written first, the account made the seed flatter and smaller, 11-14 places against 16-18
    and 10-12 as siblings under one region). It is written from the setting and the seed as seeded, so it agrees
    with what exists. Two attempts; a world whose account could not be written is still born -- logged, and
    POST /world/bhuvana is its remedy -- because the seed, which cost the more, is not to be thrown away for it."""
    from smritilila.smrti.bija import seeded_block, write_bhuvana
    seeded = seeded_block(spec)
    for attempt in (1, 2):
        try:
            return await asyncio.to_thread(write_bhuvana, roles.smrti_client, req.self_description, req.setting, brief,
                                           seeded=seeded)
        except ValueError as exc:
            log_event("bija.bhuvana_failed", attempt=attempt, error=str(exc)[:200])
    return None


JOIN_REPEAT_SECONDS = 600  # a join asked again within this is answered with the newcomer it made


async def _v3_join(req: JoinRequest, world: WorldSession, roles: GenerativeRoles) -> JoinResponse:
    """A newcomer into a live world (bija.write_arrival): their first
    tuples standing at the entry place, the place's tuple of the arrival,
    and they become playable. The entry place's first and latest tuples
    are the only record of the world it is shown (beside who stands there
    now, and the world's source, view and age) -- a stranger knows nothing
    of the world yet."""
    new_place = None
    if req.start_description.strip():
        place_id, new_place = await _described_start(req, world, roles)
    else:
        place_id = req.entry_location_id
    if new_place is not None:  # a new scene: its first tuple is all there is of it
        place_id = mnemonic_id(new_place["label"], new_id())
        label = new_place["label"]
        first = last = new_place["tuple"]
    else:
        place = world.canon.entities.get(place_id)
        label = place.labels[0] if place and place.labels else place_id
        at_place = _stream_of(world, place_id)
        first = at_place[0].experience if at_place else label
        last = at_place[-1].experience if at_place else first
    pulse_id = new_id()
    log_event("pulse.player_action", kind="join", pulse_id=pulse_id, entity_id=None, world_time=world.canon.world_time,
              action_text=req.self_description, smrti=True)
    async with world.builder_lock, world.in_scene(place_id, "join"):
        # asked again while the first was being written, or soon after: the same newcomer, not another (checked
        # under the lock, so a retry that waited on the first sees it)
        where = "start:" + " ".join(req.start_description.split()).casefold() if new_place is not None else place_id
        key = (where, " ".join(req.self_description.split()).casefold())
        now_s = time.monotonic()
        world.recent_joins = {k: v for k, v in world.recent_joins.items() if now_s - v[0] < JOIN_REPEAT_SECONDS}
        if key in world.recent_joins:
            log_event("join.repeated", pulse_id=pulse_id, entity_id=world.recent_joins[key][1].player_entity_id)
            return world.recent_joins[key][1]
        joined = await _v3_join_here(req, world, roles, place_id, label, first, last, pulse_id, new_place=new_place)
        world.recent_joins[key] = (time.monotonic(), joined)
        return joined


def _place_keys(world: WorldSession, kosha, place_ids: set[str], t: int,
                only: set[str] | None = None) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """The world's places as a writer is shown them: a short key for each (its label's words, made unique), and
    for each -- or for those in `only` -- (label, "scene|region, in <key>"). Shared by a described start and an
    expansion, which gather their places differently (the pass of 2026-10-05)."""
    from smritilila.smrti.bija import _key
    key_to_id: dict[str, str] = {}
    for eid in sorted(place_ids):
        e = world.canon.entities[eid]
        if not e.labels:
            continue
        k = _key(e.labels[0])
        while k in key_to_id:
            k += "_"
        key_to_id[k] = eid
    id_to_key = {v: k for k, v in key_to_id.items()}
    places: dict[str, tuple[str, str]] = {}
    for k, eid in key_to_id.items():
        if only is not None and eid not in only:
            continue
        into = kosha.container_at(eid, t)
        kind = "scene" if kosha.is_scene(eid) else "region"
        places[k] = (world.canon.entities[eid].labels[0], kind + (f", in {id_to_key[into]}" if into in id_to_key else ""))
    return key_to_id, places


async def _described_start(req: JoinRequest, world: WorldSession, roles: GenerativeRoles) -> tuple[str | None, dict | None]:
    """Where a newcomer starts, in their player's words (Marc, 2026-10-01): a
    listed scene that IS the place -- (its id, None) -- or one new scene in a
    listed place -- (None, {label, name, in: its container's id, tuple}),
    planted with the newcomer in one commit. The Narrator is shown the
    world's places only -- every scene and what holds it, up to the earth,
    never things or people -- so the list grows with geography, not history.
    Checked by structure: the container exists and is a place; a closed
    region within the world admits no stranger (v5 §15 -- a closed world does
    not refuse its players, who come from outside); a label a place already
    bears is that place. Cost: one model call, and the places and their
    chains."""
    from smritilila.smrti.bija import write_start
    kosha = world.kosha()
    t = world.canon.world_time + 1
    earth = _chronicle_id(world)
    place_ids = {eid for eid in world.canon.entities if kosha.is_scene(eid)}
    for eid in list(place_ids):
        c = kosha.container_at(eid, t)
        while c and c != earth and c in world.canon.entities and not kosha.is_perceiver(c) and c not in place_ids:
            place_ids.add(c)
            c = kosha.container_at(c, t)
    key_to_id, places = _place_keys(world, kosha, place_ids, t)
    try:
        out = await asyncio.to_thread(write_start, roles.smrti_client, req.start_description, req.self_description, places,
                                      source=world.world_data.get("source"), darsana=world.darsana)
    except ValueError as exc:
        raise HTTPException(502, f"the start could not be placed: {exc}")
    if "existing" in out:
        pid = key_to_id.get(out["existing"])
        if pid is None or not kosha.is_scene(pid):
            raise HTTPException(502, "the start named no scene of this world")
        log_event("join.start_placed", existing=pid, start=req.start_description[:200])
        return pid, None
    new = dict(out["new"])
    into = key_to_id.get(new["in"])
    if into is None:
        raise HTTPException(502, "the start was placed in nowhere this world has")
    same = next((eid for eid in place_ids if world.canon.entities[eid].labels[0].casefold() == new["label"].casefold()), None)
    if same is not None:  # a label a place already bears is that place
        if kosha.is_scene(same):
            log_event("join.start_placed", existing=same, start=req.start_description[:200], by="label")
            return same, None
        raise HTTPException(409, f"{new['label']} is a region, not a place to stand: describe where in it")
    if kosha.seed("pacing", into, t, field="arrivals") is False and (earth is None or kosha.seed("pacing", earth, t, field="arrivals") is not False):
        raise HTTPException(409, f"{world.canon.entities[into].labels[0]} admits no strangers: describe somewhere else")
    new["in"] = into
    log_event("join.start_placed", new=new["label"], into=into, start=req.start_description[:200])
    return None, new


async def _v3_join_here(req: JoinRequest, world: WorldSession, roles: GenerativeRoles, place_id: str, label: str,
                        first: str, last: str, pulse_id: str, new_place: dict | None = None) -> JoinResponse:
    """The arrival itself, holding the entry scene (v5 §33): who is there is
    read, the newcomer written and committed, with no turn in between."""
    from smritilila.smrti.bija import write_arrival
    from smritilila.smrti.genesis import plant_person
    # who is there now, by structure -- what they are, what they last did,
    # and whether they are a player away (present, still, silent)
    t_now = world.canon.world_time + 1
    present = []
    for pid in sorted(world.kosha().occupants(place_id, t_now)):
        e = world.canon.entities.get(pid)
        own = _stream_of(world, pid)
        present.append((e.labels[0] if e and e.labels else pid, own[-1].experience[:300] if own else "",
                        pid in world.playable_entity_ids and pid not in world.movable()))
    kala = world.samskara().kala
    # v5 §20 (Marc, 2026-09-25): never at the clock's start (a lodger joined at 1:00 where Danforth lived 2:08, and
    # stepped over him).
    # Every arrival, at a new place or one the world has (Marc, 2026-10-01: Aramus joined at the east track's own
    # clock, in hours all Millhaven had already lived -- out of reach, so no one in the village remembers the dragon):
    # was at the furthest along player's present, or the place's own time if the world has lived it further
    # (v5.24, Marc, 2026-10-07: that put Kypazec 25 minutes ahead of the two players in Stonemarch's common room, a third having
    # moved far away): now at the clock of the nearest online player in the tree, the place's own time a floor
    if kala is None:
        minute = None
    else:
        own = kala.scene_clock.get(new_place["in"] if new_place is not None else place_id, kala.start)
        near = spanda._nearest_players_clock(world, new_place["in"] if new_place is not None else place_id, new_place is not None)
        minute = own if near is None else max(near, own)
    try:
        arrival = await asyncio.to_thread(write_arrival, roles.smrti_client, req.self_description, label, first, last,
                                          _at_large(world, minute),
                                          present=present, source=world.world_data.get("source"), darsana=world.darsana)
    except ValueError as exc:
        raise HTTPException(502, f"the newcomer could not be written: {exc}")
    # a name someone in the world already bears is refused, as an expansion refuses one (v5 §17/§39; the Crossing,
    # 2026-09-28: three Nara Millwrights) -- the newcomer is not written
    declared = str(arrival.get("name") or arrival["label"]).strip()
    bearers = world.kosha().named(declared) if declared else set()
    if bearers:
        log_event("join.name_taken", pulse_id=pulse_id, name=declared, bearers=sorted(bearers))
        raise HTTPException(409, f"someone in this world is already called {declared}: choose another name")
    embed = adhisthana.get_embedder().embed
    now = world.canon.world_time + 1
    newcomer, memories, placed = plant_person(arrival["label"], arrival["tuples"], place_id, now, pulse_id, embed, minute=minute)
    born: list[Entity] = []
    born_records: list[Record] = []
    if new_place is not None:  # the place the player described, planted with them (v5 §10: placed by a record)
        born.append(Entity(place_id, [new_place["label"]], now, "genesis", avarana="scene", kosha=new_place["in"]))
        memories.insert(0, Memory(new_id(), place_id, frozenset(), new_place["tuple"], "AMBIENT", 5, place_id, now, pulse_id,
                                  embed(new_place["tuple"]), began=minute, ended=minute))
        born_records += [Record(new_id(), "kosha", place_id, new_place["in"], now, pulse_id, note="declared"),
                         Record(new_id(), "name", place_id, new_place["name"] or new_place["label"], now, pulse_id)]
    if arrival["place_tuple"]:
        # the arrival is the scene's record, witnessed by everyone there -- a player's copy is theirs (v5 §31)
        there = world.kosha().occupants(place_id, now) - world.absent()
        if kala is not None and minute is not None:  # whoever the arrival's moment is not the present of (kala.apart)
            there -= apart(kala, there, minute, world.playable_entity_ids, world.movable())
        present = frozenset(there | {newcomer.id})
        emb = embed(arrival["place_tuple"])
        memories.append(Memory(new_id(), place_id, present, arrival["place_tuple"], "AMBIENT", 5, place_id, now, pulse_id, emb,
                               began=minute, ended=minute))
        for x in sorted(there & set(world.playable_entity_ids)):
            memories.append(Memory(new_id(), x, present, arrival["place_tuple"], "PRIMARY", 5, place_id, now, pulse_id, emb))
    store = _store(world)
    records = [placed, Record(new_id(), "name", newcomer.id, arrival.get("name") or newcomer.labels[0], now, pulse_id)]  # v5 §16/§7
    # what every player who arrives begins as, if the world says (world data "newcomer_clauses": {key: clause}) --
    # written as their own becomings (v5 §8), so it holds whatever their description claims and whichever model
    # narrates (Marc, 2026-09-27, Stonemarch: "a level 1 new player character ... a solid rule, not optional")
    for key, clause in (world.world_data.get("newcomer_clauses") or {}).items():
        if str(clause or "").strip():
            records.append(Record(new_id(), "clause", newcomer.id, str(clause).strip(), now, pulse_id, note=f"self:{key}"))
    records = born_records + records
    await store.commit(world.canon, memories, born + [newcomer], records=records)
    _fold_in(world, born + [newcomer], now)
    _extend_history(world, memories, records)
    _queue_witnessed(world, memories, exclude=frozenset({newcomer.id}))  # meeting: the players there see who walked in
    # counted, as a pulse's are: a record naming a player in a turn not theirs (the arrival has no channel to move,
    # voice or write as a player -- only its prose can name one; nothing blocks, the rate is watched)
    player_named(world.kosha(), memories, newcomer.id, frozenset(world.playable_entity_ids), pulse_id)
    world.presence[newcomer.id] = "online"  # v5 §37: they are here -- not idle since the server started
    world.last_seen[newcomer.id] = time.monotonic()
    world.playable_entity_ids = list(world.playable_entity_ids) + [newcomer.id]
    if not world.scene_owner_entity_id:
        world.scene_owner_entity_id = newcomer.id  # a brief's world: the first to arrive is its first player
    await store.save_session_config(world.scene_owner_entity_id, world.entry_point_location_ids, world.playable_entity_ids, world.setting_description)
    log_event("bija.arrived", pulse_id=pulse_id, newcomer=newcomer.id, place=place_id, tuples=len(memories))
    return JoinResponse(player_entity_id=newcomer.id, narration=arrival["opening"], narrations={newcomer.id: arrival["opening"]})


@router.post("/entities/{keep}/absorb/{gone}", response_model=AbsorbResponse)
async def absorb_entity_route(keep: str, gone: str) -> AbsorbResponse:
    """s10 as a mechanism (2026-09-20): two ids that name one perceiver
    become one. `gone`'s tuples become `keep`'s (re-filed, never
    rewritten), `gone`'s label becomes an alias of `keep` (a name
    record, so item 5 binds either name; no tuple is written saying so),
    `gone`'s id is dropped, and the world is reloaded -- so no
    running server keeps a dead id in memory (Hillands, the second
    gatekeeper, deleted by hand under a live server and written back).
    Same kind only; never a playable character."""
    world = await adhisthana.get_world()
    a, b = world.canon.entities.get(keep), world.canon.entities.get(gone)
    if a is None or b is None:
        raise HTTPException(404, "both ids must exist")
    if keep == gone:
        raise HTTPException(422, "an entity cannot absorb itself")
    k = world.kosha()
    if k.is_scene(keep) != k.is_scene(gone):
        raise HTTPException(422, "a scene and a perceiver cannot be one")
    if gone in world.playable_entity_ids or keep in world.playable_entity_ids and gone in world.playable_entity_ids:
        raise HTTPException(422, "a playable character is never absorbed")
    alias = b.labels[0] if b.labels else gone
    store = _store(world)
    counts = await store.absorb(keep, gone)
    # the alias (v5 §16): a name record -- no English tuple written into memory
    alias_record = Record(new_id(), "name", keep, alias, world.canon.world_time, "absorb", note="alias")
    working = world.canon.fork()
    working.entities.pop(gone, None)
    await store.commit(working, [], [a], records=[alias_record])
    log_event("entity.absorbed", kept=keep, gone=gone, alias=alias, **counts)
    # reload: the store is the truth now
    if hasattr(app.state, "world"):
        del app.state.world
    return AbsorbResponse(kept=keep, gone=gone, label=a.labels[0], alias=alias, tuples=counts["tuples"], filed=counts["filed"], presence=counts["presence"],
                          records=counts.get("records", 0), contained=counts.get("contained", 0))


def _seeder(client):
    """Who writes a seed -- a birth's, an expansion's: the second model first, the other
    only if it fails (Marc, 2026-10-01). A single-provider world has only one."""
    from smritilila.llm import providers
    word = providers.seed_word()
    return client.prefer(word) if word and hasattr(client, "prefer") else client


@router.get("/world/view", response_model=WorldView)
async def get_world_view() -> WorldView:
    world = await adhisthana.get_world()
    return WorldView(darsana=world.darsana, becomings=world.becomings, world_data=world.world_data)


@router.put("/world/view", response_model=WorldView)
async def put_world_view(req: WorldView) -> WorldView:
    """Set the world's own view whole. Takes effect from the next pulse;
    nothing already written changes. Nothing of the engine's is in it
    (2026-09-30): who plays whom is the tether's, a reckoning is a record."""
    world = await adhisthana.get_world()
    new = (req.darsana.strip() if req.darsana and req.darsana.strip() else None,
           {str(k): str(v) for k, v in req.becomings.items() if str(v).strip()}, dict(req.world_data))
    earth = _earth(world.canon)
    if earth is None:  # a world with no earth yet keeps its view beside it, as before v5 §15
        await _store(world).save_view(*new)
        written = 0
    else:  # v5 §15: what changed, appended on the earth -- the whole seed, the first time
        old = (world.darsana, world.becomings, world.world_data) if prakriti.sown(world.records, earth) else prakriti.EMPTY
        sown = prakriti.sow(earth, old, new, world.canon.world_time, f"view:{new_id()}")
        await _store(world).commit(world.canon, [], [], records=sown)
        _extend_history(world, [], sown)
        written = len(sown)
    world.darsana, world.becomings, world.world_data = new
    log_event("world.view", darsana=bool(world.darsana), becomings=len(world.becomings), world_data=sorted(world.world_data),
              records=written)
    return WorldView(darsana=world.darsana, becomings=world.becomings, world_data=world.world_data)


@router.get("/world/bhuvana", response_model=BhuvanaResponse)
async def world_bhuvana() -> BhuvanaResponse:
    """The world at large, for the player to read -- what a literate
    person of this world would know of it -- and the chronicle since.
    None for a world born before 2026-10-05 below wide breadth (until
    POST /world/bhuvana), or one whose account could not be written."""
    world = await adhisthana.get_world()
    return BhuvanaResponse(bhuvana=world.bhuvana, chronicle=_chronicle(world, 50)[1:])


@router.post("/world/bhuvana", response_model=BhuvanaResponse)
async def plant_bhuvana(background_tasks: BackgroundTasks, req: BhuvanaRequest | None = None) -> BhuvanaResponse:
    """The world at large, given to a world already lived in (2026-09-21):
    every world born before bhuvana (the crossing, Tombstone, Rome,
    Bywater) had no age moving behind it -- roads carried no news, far
    ticks drew no chronicle. One bija call writes the bhuvana from the
    setting and the player's own first tuples; itihasa is planted with
    THE YEAR as its first tuple at tick 0, before everything, and takes
    the next far tick like any newborn chronicler (a world that already
    has its earth gives the earth the age as a new tuple instead). Once only: a world
    that has its age keeps it (fixed at birth, s0 for the engine).
    Live worlds only, as expand is."""
    from smritilila.smrti.bija import write_bhuvana
    from smritilila.smrti.genesis import plant_chronicle
    world = await adhisthana.get_world()
    if not world.canon.entities:
        raise HTTPException(409, "no world to give an age to -- begin one first")
    if world.bhuvana:
        raise HTTPException(409, "this world already has its age")
    roles = adhisthana._resolve_roles()
    req = req or BhuvanaRequest()
    setting = (req.setting or world.setting_description or "").strip() or None
    self_description = (req.self_description or "").strip()
    if not self_description:
        owner = world.canon.entities.get(world.scene_owner_entity_id)
        own = sorted(_stream_of(world, world.scene_owner_entity_id), key=lambda m: (m.world_time, m.id))[:3]
        self_description = "\n".join([f"I am {owner.labels[0]}." if owner and owner.labels else ""] + [m.experience for m in own]).strip()
    if not self_description and not setting:
        raise HTTPException(422, "nothing to write the world at large from: give a setting or a self_description")
    pulse_id = new_id()
    log_event("pulse.player_action", kind="bhuvana", pulse_id=pulse_id, entity_id=None, world_time=world.canon.world_time, action_text="", smrti=True)
    try:
        bhuvana = await asyncio.to_thread(write_bhuvana, roles.smrti_client, self_description, setting,
                                          source=world.world_data.get("source"), darsana=world.darsana)
    except ValueError as exc:
        raise HTTPException(502, f"the world at large could not be written: {exc}")
    earth = _chronicle_id(world)
    if earth is None:  # a world from before its earth: the age is planted as its earth
        ch_entity, ch_memory, ch_name = plant_chronicle(bhuvana, adhisthana.get_embedder().embed)
        new_entities, new_memories, new_records = [ch_entity], [ch_memory], [ch_name]
    else:  # the earth it has takes the age (Marc, 2026-09-30: earth is the world at large)
        head = bhuvana.split("THE REGIONS", 1)[0].strip() or bhuvana[:1200]
        new_entities, new_records = [], []
        new_memories = [Memory(new_id(), earth, frozenset(), head, "PRIMARY", 3, NOWHERE, world.canon.world_time, pulse_id,
                               adhisthana.get_embedder().embed(head))]
    async with world.builder_lock:
        if world.bhuvana:
            raise HTTPException(409, "this world already has its age")
        await _store(world).commit(world.canon, new_memories, new_entities, records=new_records)
        if new_entities:
            _fold_in(world, new_entities, world.canon.world_time)
        _extend_history(world, new_memories, new_records)
        world.bhuvana = bhuvana
        await _store(world).save_bhuvana(bhuvana)
    log_event("bija.bhuvana_planted", pulse_id=pulse_id, chronicler=earth or new_entities[0].id, words=len(bhuvana.split()))
    background_tasks.add_task(_describe, world, roles)  # the root has changed: its line is drawn again
    return BhuvanaResponse(bhuvana=bhuvana, chronicle=[])


def _root_of(world: WorldSession) -> str:
    """What the world grows from, as its earth carries it: the age, where the
    world keeps one; else the earth's first memory (its setting); else the setting.
    With the earth's newest entries after it (2026-10-20): the age moves, and a line drawn from the
    birth year alone said "April 27, 1928" of a world whose earth had reached July."""
    if world.bhuvana:
        base = world.bhuvana.split("THE REGIONS", 1)[0].strip() or world.bhuvana[:1200]
    else:
        earth = _chronicle_id(world)
        first = min(_stream_of(world, earth), key=lambda m: (m.world_time, m.id), default=None)
        base = first.experience if first is not None else (world.setting_description or "")
    newer = [m.experience.strip() for m in _age_tuples(world)[1:][-DESCRIBE_AGE_ENTRIES:] if m.experience.strip() and m.experience.strip() not in base]   # after the birth entry
    if not newer:
        return base
    return base + "\n\nTHE AGE SINCE (the world at large's newest entries: where and when the world has reached now):\n" + "\n\n".join(t[:1200] for t in newer)


async def _describe(world: WorldSession, roles: GenerativeRoles) -> str | None:
    """Draw the world's one line from its root and keep it with the list of
    worlds. A world outside the registry has no line to keep; a draw that
    fails leaves the line it had."""
    from smritilila.smrti.bija import describe_root
    from smritilila.worlds_registry import set_world_description
    entry = next((w for w in adhisthana.get_worlds_registry() if w.name == world.name), None) if world.name else None
    root = _root_of(world)
    if entry is None or not root.strip() or roles.smrti_client is None:
        return None
    try:
        line = await asyncio.to_thread(describe_root, roles.smrti_client, entry.name, root)
    except Exception as exc:
        log_event("world.describe_failed", name=entry.name, error=repr(exc))
        return None
    set_world_description(entry.name, line)
    app.state.worlds_registry = None  # reload from the file
    log_event("world.described", name=entry.name, description=line)
    return line


DESCRIBE_AGE_ENTRIES = 2        # how many of the earth's newest entries the line is drawn with, after the root
DESCRIBE_EVERY = 24 * 3600      # a world's line is drawn again once per real-time day (Marc, 2026-10-20)
DESCRIBE_FIRST_AFTER = 600      # seconds after a server starts before it first looks (a restart is not a reason to draw)
DESCRIBE_CHECK_EVERY = 1800     # seconds between looks
DESCRIBE_RETRY_AFTER = 3600     # a draw that failed is tried again after this long, not at the next look
_describe_attempted: dict[str, float] = {}


def _describe_due(entry, now: datetime | None = None) -> bool:
    """Its line has never been drawn with a time, or was drawn a real-time day ago or more (an unreadable time counts as never)."""
    try:
        at = datetime.strptime(entry.described_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return True
    return (now or datetime.now(timezone.utc)) - at >= timedelta(seconds=DESCRIBE_EVERY)


async def describe_if_due(now: datetime | None = None, clock=time.monotonic) -> str | None:
    """One look: the line of the world this server serves, drawn if a day has passed. The world is not loaded, nor a model asked, unless
    it is due and no draw has failed within the hour. Returns the new line, or None."""
    entry = adhisthana._active_world_registry_entry()
    if entry is None or not _describe_due(entry, now):
        return None
    if clock() - _describe_attempted.get(entry.name, -DESCRIBE_RETRY_AFTER) < DESCRIBE_RETRY_AFTER:
        return None
    _describe_attempted[entry.name] = clock()
    world = await adhisthana.get_world()
    line = await _describe(world, adhisthana._resolve_roles())
    if line is not None:
        _describe_attempted.pop(entry.name, None)
    return line


async def describe_daily() -> None:
    """Each world's one line, drawn again once per real-time day by the server that serves it (Marc, 2026-10-20: "the one sentence world
    description should be updated once per real time day"). Started with the server, stopped with it. A world with no entry of its own in the
    registry, or no root yet, keeps the line it has; a draw that fails leaves the line it had and is tried again within the hour."""
    port = os.environ.get("SMRITILILA_DEV_PORT", "8198")
    await asyncio.sleep(DESCRIBE_FIRST_AFTER + (int(port) % 10 if port.isdigit() else 0) * 20)   # six servers do not look in the same minute
    while True:
        try:
            await describe_if_due()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a failed look is tried again at the next one, never a reason to stop
            log_event("world.describe_failed", error=repr(exc))
        await asyncio.sleep(DESCRIBE_CHECK_EVERY)


@router.post("/world/describe", response_model=DescribeResponse)
async def describe_world() -> DescribeResponse:
    """A world's gods draw its one line again from its root."""
    world = await adhisthana.get_world()
    line = await _describe(world, adhisthana._resolve_roles())
    if line is None:
        raise HTTPException(409, "no line could be drawn: the world is unregistered, has no root yet, or the draw failed")
    return DescribeResponse(description=line)


@router.post("/world/expand", response_model=ExpandResponse)
async def expand_world() -> ExpandResponse:
    """Vistāra (2026-09-20): the world grows outward from its edges --
    what lies along each road off the map, the next holdings beyond the
    farthest, the people of the age not yet present -- as one seed call
    planted into the live world. Live worlds only; the engine learns
    nothing of what was added (s0)."""
    from smritilila.smrti.bija import expand
    from smritilila.smrti.genesis import plant_more
    world = await adhisthana.get_world()
    if not world.canon.entities:
        raise HTTPException(409, "no world to expand -- begin one first")
    roles = adhisthana._resolve_roles()
    # The OUTER RING only (Marc, 2026-09-20): the Narrator is shown what
    # the most recent expansion planted (or the seed, before any), not
    # the whole world -- the listing stays bounded as rings accumulate,
    # and the region grows from its own edge rather than echoing the
    # player's seed back outward (four rings in, the fjord had three
    # drownings). Names taken stay world-wide: that is what keeps a
    # second Grim out.
    # A ring is one expansion's planting, marked by its pulse id
    # ("vistara:<n>"); the outer ring is the last one planted, the highest
    # n (below). (Four rings planted at one world tick
    # were indistinguishable by tick, and a person's first tuple lands
    # at now-20 -- the first cut of this listed 46 places and no one.)
    # ("vistara:<n>", n counting up; the bare "vistara" of the first
    # rings counts as 0. Log order is not planting order when rings share
    # a tick -- the store returns tuples by tick, then id.)
    def ring_no(pulse_id: str) -> int | None:
        if pulse_id == "vistara":
            return 0
        if pulse_id.startswith("vistara:") and pulse_id[8:].isdigit():
            return int(pulse_id[8:])
        return None
    ring_nos = {ring_no(m.pulse_id) for m in world.recent_memories} - {None}
    if ring_nos:
        last = max(ring_nos)
        outer_ring = "vistara" if last == 0 else f"vistara:{last}"
        outer = {m.entity_id for m in world.recent_memories if m.pulse_id == outer_ring}
    else:
        last = 0
        outer_ring = "genesis"
        outer = {m.entity_id for m in world.recent_memories if m.pulse_id == "genesis"}
    # v4 (v5 §11, §14): a place is a scene, or a container that holds one --
    # a region -- never a perceiver
    kosha = world.kosha()
    t_now = world.canon.world_time + 1
    place_ids = {eid for eid in world.canon.entities if kosha.is_scene(eid)}
    place_ids |= {c for c in (kosha.container_at(eid, t_now) for eid in list(place_ids))
                  if c and c in world.canon.entities and not kosha.is_perceiver(c)}
    key_to_id, places = _place_keys(world, kosha, place_ids, t_now, only=outer)
    id_to_key = {v: k for k, v in key_to_id.items()}
    standing: list[tuple[str, str]] = []  # whoever and whatever stands at the edge, unsorted (v5 §1)
    for e in world.canon.entities.values():
        if e.id not in place_ids and e.labels and e.id in outer and e.origin != ITIHASA:
            place, _ = world.here(e.id)
            standing.append((e.labels[0], id_to_key.get(place or "", "nowhere")))
    log_event("bija.outer_ring", ring=outer_ring, places=len(places), standing=len(standing))
    pulse_id = new_id()
    log_event("pulse.player_action", kind="expand", pulse_id=pulse_id, entity_id=None, world_time=world.canon.world_time, action_text="", smrti=True)
    # names taken: people's own names, and every name the memory carries
    # (a place's name is not a person's: "Unnr of Kross" is Unnr)
    # v5 §16/§39: the names this world's people carry, declared or derived, exact
    names_taken = frozenset(kosha._name_text.get(n, n) for eid in world.canon.entities if eid not in place_ids for n in kosha.names_of(eid))
    labels_taken = frozenset(l for e in world.canon.entities.values() if e.id in place_ids for l in e.labels[:1])
    new = await asyncio.to_thread(expand, _seeder(roles.smrti_client), world.setting_description, places, standing, names_taken, labels_taken,
                                  _at_large(world, spanda._players_latest(world)), frozenset(key_to_id), source=world.world_data.get("source"),
                                  darsana=world.darsana, titles=world.world_data.get("name_titles"))
    if not new["places"] and not new["people"]:
        raise HTTPException(502, "the expansion added nothing")
    async with world.builder_lock:
        entities, memories, planted = plant_more(new, key_to_id, adhisthana.get_embedder().embed, world.canon.world_time, pulse_id=f"vistara:{last + 1}",
                                                 earth=_chronicle_id(world))
        await _store(world).commit(world.canon, memories, entities, records=planted)
        _fold_in(world, entities, world.canon.world_time)
        _extend_history(world, memories, planted)
    log_event("bija.vistara", pulse_id=pulse_id, entities=len(entities), tuples=len(memories))
    return ExpandResponse(places=[p["label"] for p in new["places"].values()], people=[q["label"] for q in new["people"].values()])


# One genesis at a time (2026-09-30). A seed takes minutes and the tunnel gives up on a request at 100s: the
# page then said it failed and offered Build again while the first was still seeding. Each request with no world
# loaded checked its own empty session, so the second passed, and both seeds would have gone into one world.
_genesis_lock = asyncio.Lock()


@router.post("/genesis", response_model=GenesisResponse)
async def genesis(req: GenesisRequest, background_tasks: BackgroundTasks = None) -> GenesisResponse:
    if not req.self_description.strip():
        if not (req.where_when or "").strip():
            raise HTTPException(422, "give a self_description, or a world's where_when")
        if not (req.name or "").strip():
            raise HTTPException(422, "a world from a brief needs its name")
        if not (req.source or "").strip():
            raise HTTPException(422, "a world from a brief needs its source: history, a book or a film, and which")
    if _genesis_lock.locked():
        raise HTTPException(409, "a world is already being seeded here -- it will open when it is done")
    async with _genesis_lock:
        world = getattr(app.state, "world", None)
        if world is None:
            store = adhisthana.get_world_store()
            loaded = await store.load()
            if loaded is not None and loaded.entities:
                raise HTTPException(409, "canon is not empty -- use POST /world/join instead")
            world = WorldSession(canon=loaded if loaded is not None else Maya(world_time=0), scene_owner_entity_id="",
                                 store=store, name=current_world())
        elif world.canon.entities:
            raise HTTPException(409, "canon is not empty -- use POST /world/join instead")

        roles = adhisthana._resolve_roles()  # 503 after emptiness/validation checks above
        born = await _v3_genesis(req, world, roles)
    if background_tasks is not None:  # called in-process (a test): no line drawn
        background_tasks.add_task(_describe, world, roles)  # its root is laid: its line, for those deciding to enter
    return born


def _current_entry_point_ids(world: WorldSession) -> list[str]:
    """Where a new character joining would actually land -- computed
    fresh from wherever the playable characters in the world stand RIGHT
    NOW (at first, wherever world.scene_owner_entity_id, the primary,
    currently-active character, was), not the static
    entry_point_location_ids set once at genesis and never updated. A
    real, observed problem this fixes: the active story moves on (a
    fresh genesis, travel, a manual correction), but the static list
    kept offering joins into whatever location happened to be current
    back at genesis -- a real session ended up offering the disconnected
    starter-world "square" as the only entry point long after the actual
    story had moved to a camp nowhere near it.

    "For now" fix, by explicit request: entry_point_location_ids itself
    (WorldSession field, session_config's persisted column) is left in
    place rather than torn out with its own migration -- it's consulted
    now only when no playable character stands anywhere. A real
    multi-entry-point design is future work; every location any playable
    character currently occupies is offered (at first, only the primary
    one's), which covers the common single-active-story case the
    sessions so far have actually hit."""
    # v3: every place a playable character now stands, from tuples
    # (s11: where someone is, is the place whose tuples last record
    # them); a stranger arrives where a story already is.
    out: list[str] = []
    for pid in world.playable_entity_ids:
        if pid in world.absent():
            continue
        place, _ = world.here(pid)
        if place and place != NOWHERE and place not in out:
            out.append(place)
    return out or list(world.entry_point_location_ids)


# Marc, 2026-09-24: a newcomer chooses where to arrive -- the places the
# most has happened, and wherever a player now stands.
BUSIEST_LOCATIONS = 3


def _join_locations(world: WorldSession) -> list[LocationOut]:
    """Where a newcomer may arrive: the BUSIEST_LOCATIONS places with the
    most tuples formed in them across all history, then any place a
    player stands that is not already among them. A count, not a
    quality (s0): each tuple is one perceiver's moment there.

    Cost: one pass over every memory held (all of history is loaded at
    startup), on each call -- called only when someone opens the new
    character panel or joins, never per pulse."""
    traffic = Counter(m.location_id for m in world.recent_memories
                      if m.location_id != NOWHERE and m.location_id in world.canon.entities)
    here: dict[str, list[str]] = {}
    for pid in world.playable_entity_ids:
        if pid in world.absent():
            continue  # out of the world: stands nowhere a newcomer could meet them
        place, _ = world.here(pid)
        if place and place != NOWHERE:
            here.setdefault(place, []).append(_labelled(world.canon, pid))
    ids = [loc for loc, _ in sorted(traffic.items(), key=lambda kv: (-kv[1], kv[0]))[:BUSIEST_LOCATIONS]]
    ids += [loc for loc in here if loc not in ids]
    if not ids:
        ids = list(world.entry_point_location_ids)
    return [LocationOut(location_id=loc, label=_labelled(world.canon, loc), memories=traffic.get(loc, 0),
                        players_here=here.get(loc, [])) for loc in ids]


@router.post("/world/becoming/{eid}", status_code=204)
async def add_becoming(eid: str, req: BecomingRequest) -> None:
    """A becoming appended by the world's owner (v5 §8) -- a gravestone repair for what play missed, never an edit:
    insert-only, the newest under its key holds, as the Narrator's own do. Marc, 2026-09-28: Jakal's first
    expedition (t72-122) finished five goals before his measure began at t146, and was never awarded. Owner only."""
    world = await adhisthana.get_world()
    key, clause = req.key.strip(), req.clause.strip()
    if eid not in world.canon.entities or not key or not clause or not re.fullmatch(r"[a-z0-9_]+", key):
        raise HTTPException(422, "an existing being, a snake_case key and a clause")
    now = world.canon.world_time
    r = Record(new_id(), "clause", eid, clause, now, f"owner:{new_id()}", note=f"self:{key}")
    await _store(world).commit(world.canon, [], [], records=[r])
    _extend_history(world, [], [r])
    log_event("world.becoming_added", entity_id=eid, key=key, clause=clause[:200], why=req.why[:300], world_time=now)


@router.post("/world/containment/{eid}", status_code=204)
async def add_containment(eid: str, req: ContainmentRequest) -> None:
    """A containment record appended by the world's owner -- a gravestone repair, insert-only: where eid is from
    now on, never where it was. Marc, 2026-09-28: Turuko stood in no scene since a lane was minted as a thing beside
    the scene of that name. Checked for structure only (v5 §10): both exist, no cycle, never a player's body. Owner only."""
    world = await adhisthana.get_world()
    k = world.kosha()
    t = world.canon.world_time + 1
    if eid not in world.canon.entities or req.into not in world.canon.entities or eid == req.into:
        raise HTTPException(422, "two existing entities")
    if k.inside(req.into, eid, t):
        raise HTTPException(422, "a cycle: it holds what it would go into")
    if eid == _chronicle_id(world):
        raise HTTPException(422, "the earth is never moved: it holds all, and nothing holds it")
    if eid in world.playable_entity_ids:
        raise HTTPException(422, "a player's body is their own")
    r = Record(new_id(), "kosha", eid, req.into, world.canon.world_time, f"owner:{new_id()}", note="owner")
    await _store(world).commit(world.canon, [], [], records=[r])
    _extend_history(world, [], [r])
    log_event("world.containment_added", entity_id=eid, into=req.into, why=req.why[:300], world_time=world.canon.world_time)


@router.get("/world/entry-points", response_model=list[EntryPointOut])
async def entry_points() -> list[EntryPointOut]:
    world = await adhisthana.get_world()
    return [EntryPointOut(location_id=loc_id) for loc_id in _current_entry_point_ids(world)]


@router.get("/world/locations", response_model=list[LocationOut])
async def join_locations() -> list[LocationOut]:
    """Where a new character may arrive (_join_locations)."""
    world = await adhisthana.get_world()
    return _join_locations(world)


@router.post("/world/join", response_model=JoinResponse)
async def join_world(req: JoinRequest, request: Request) -> JoinResponse:
    if not req.self_description.strip():
        raise HTTPException(422, "self_description must not be empty")
    world = await adhisthana.get_world()
    if not req.start_description.strip():
        valid_entry_points = [loc.location_id for loc in _join_locations(world)]
        if req.entry_location_id not in valid_entry_points:
            raise HTTPException(422, f"entry_location_id must be one of {valid_entry_points}, or describe where you start")

    roles = adhisthana._resolve_roles()
    joined = await _v3_join(req, world, roles)
    guest = dvara._guest(request)
    if guest:  # the character a guest makes is theirs to play (and only theirs)
        world.played_by[joined.player_entity_id] = guest  # the tether (Marc, 2026-09-30): not the world's view
        await _store(world).save_played_by(world.played_by)
        log_event("guest.joined", guest=guest, entity_id=joined.player_entity_id)
    return joined
