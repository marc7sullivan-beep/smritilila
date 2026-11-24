# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Udbhava (उद्भव, "arising, coming into being"; Marc's sign-off, 2026-11-10): a world arising, and a newcomer arising in it.
A world's genesis from its first player's self-description or its creator's brief (the seed written, the world at large with
it, births for the builder's people, all planted in one commit); a newcomer's join into a live world, starting in their own
words or at a place offered, the arrival written while its scene is held; the places a newcomer may arrive at; and who writes
a seed (seeder). The writers are smritilila.smrti.bija's, the planter smritilila.smrti.genesis's.

Moved whole from routes/bija.py on 2026-11-10 (stage 2 of moving the orchestration out of the routes), nothing inside
changed: the routes keep their handlers and the request's own checks, and call what is here through this module
(udbhava.X), never imported bare, so a test's patch here reaches every caller.

From main.py's docstring, moved with the code it explains (2026-10-05):

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
import time

from fastapi import HTTPException

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.ids import new_id
from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila.adhisthana import (
    GenerativeRoles, WorldSession, earth_of, extend_history, fold_in, store_of, stream_of, app,
)
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import NOWHERE
from smritilila.smrti.pulse import mnemonic_id, player_named
from smritilila.smrti.kala import apart
from smritilila.smrti import prakriti
from smritilila.sakshi import log_event
from smritilila.itihasa import at_large, chronicle_id
from smritilila.schemas import (
    GenesisRequest, GenesisResponse, JoinRequest, JoinResponse, LocationOut, labelled,
)
from smritilila.spanda import queue_witnessed
from smritilila import spanda  # the pulse's services, and the names a test patches there, called through it
from smritilila import abhasa  # ābhāsa: a newcomer's portrait (Marc, 2026-11-11)


async def v3_genesis(req: GenesisRequest, world: WorldSession, roles: GenerativeRoles) -> GenesisResponse:
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
            spec = await asyncio.to_thread(write_world_seed, seeder(roles.smrti_client), req.name or "", req.brief(), req.breadth, bhuvana)
        else:
            spec = await asyncio.to_thread(write_seed, seeder(roles.smrti_client), req.self_description, req.setting, req.breadth, bhuvana)
    except ValueError as exc:
        raise HTTPException(502, f"the seed could not be planted: {exc}")
    if bhuvana is None:  # below wide: the account follows the seed, so every world has one (Marc, 2026-10-05)
        bhuvana = await _account_after_seed(roles, req, brief, spec)
    births = await births_of(roles, spec["people"], lambda k: (spec["places"].get(k) or {}).get("label", k), bhuvana,
                              (spec.get("view") or {}).get("darsana"), req.source, skip=frozenset(spec["playable"]))
    entities, memories, config = plant(spec, adhisthana.get_embedder().embed, bhuvana=bhuvana, births=births)  # the earth with it (genesis.build)
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
    earth = earth_of(working_canon)
    sown = prakriti.sow(earth, prakriti.EMPTY, (darsana, becomings, world_data), 0, pulse_id) if earth is not None else []
    records = list(config["records"]) + sown
    store = store_of(world)
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


async def births_of(roles: GenerativeRoles, people: dict, place_label, bhuvana: str | None, darsana: str | None, source: str | None,
                     skip: frozenset[str] = frozenset()) -> dict[str, list[dict]]:
    """A birth memory for each of a builder's people (bija.write_births; Marc, 2026-11-06: every first memory is preceded by one), except
    `skip` -- a player, whose own description is their first memory. {person key: tuples}, for those written; a call that fails leaves its
    being without one and never fails the build."""
    from smritilila.smrti.bija import birth_asks, write_births
    try:
        return await asyncio.to_thread(write_births, roles.smrti_client, birth_asks(people, place_label, skip), bhuvana, darsana,
                                       (source or "").strip() or None)
    except Exception as exc:  # noqa: BLE001 -- a world built without births is still built
        log_event("bija.births_failed", error=str(exc)[:200])
        return {}


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


async def v3_join(req: JoinRequest, world: WorldSession, roles: GenerativeRoles) -> JoinResponse:
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
        at_place = stream_of(world, place_id)
        first = at_place[0].experience if at_place else label
        last = at_place[-1].experience if at_place else first
    pulse_id = new_id()
    log_event("pulse.player_action", kind="join", pulse_id=pulse_id, entity_id=None, world_time=world.canon.world_time,
              action_text=req.told(), smrti=True)
    async with world.builder_lock, world.in_scene(place_id, "join"):
        # asked again while the first was being written, or soon after: the same newcomer, not another (checked
        # under the lock, so a retry that waited on the first sees it)
        where = "start:" + " ".join(req.start_description.split()).casefold() if new_place is not None else place_id
        key = (where, " ".join(req.told().split()).casefold())
        now_s = time.monotonic()
        world.recent_joins = {k: v for k, v in world.recent_joins.items() if now_s - v[0] < JOIN_REPEAT_SECONDS}
        if key in world.recent_joins:
            log_event("join.repeated", pulse_id=pulse_id, entity_id=world.recent_joins[key][1].player_entity_id)
            return world.recent_joins[key][1]
        joined = await _v3_join_here(req, world, roles, place_id, label, first, last, pulse_id, new_place=new_place)
        world.recent_joins[key] = (time.monotonic(), joined)
        return joined


def place_keys(world: WorldSession, kosha, place_ids: set[str], t: int,
                only: set[str] | None = None) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """The world's places as a writer is shown them: a short key for each (its label's words, made unique), and
    for each -- or for those in `only` -- (label, "scene|region, in <key>"). Shared by a described start and an
    expansion, which gather their places differently (the pass of 2026-10-05)."""
    from smritilila.smrti.bija import label_key
    key_to_id: dict[str, str] = {}
    for eid in sorted(place_ids):
        e = world.canon.entities[eid]
        if not e.labels:
            continue
        k = label_key(e.labels[0])
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
    earth = chronicle_id(world)
    place_ids = {eid for eid in world.canon.entities if kosha.is_scene(eid)}
    for eid in list(place_ids):
        c = kosha.container_at(eid, t)
        while c and c != earth and c in world.canon.entities and not kosha.is_perceiver(c) and c not in place_ids:
            place_ids.add(c)
            c = kosha.container_at(c, t)
    key_to_id, places = place_keys(world, kosha, place_ids, t)
    try:
        out = await asyncio.to_thread(write_start, roles.smrti_client, req.start_description, req.told(), places,
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
        own = stream_of(world, pid)
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
        near = spanda.nearest_players_clock(world, new_place["in"] if new_place is not None else place_id, new_place is not None)
        minute = own if near is None else max(near, own)
    try:
        arrival = await asyncio.to_thread(write_arrival, roles.smrti_client, req.told(), label, first, last,
                                          at_large(world, minute),
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
    store = store_of(world)
    records = [placed, Record(new_id(), "name", newcomer.id, arrival.get("name") or newcomer.labels[0], now, pulse_id)]  # v5 §16/§7
    # what every player who arrives begins as, if the world says (world data "newcomer_clauses": {key: clause}) --
    # written as their own becomings (v5 §8), so it holds whatever their description claims and whichever model
    # narrates (Marc, 2026-09-27, Stonemarch: "a level 1 new player character ... a solid rule, not optional")
    for key, clause in (world.world_data.get("newcomer_clauses") or {}).items():
        if str(clause or "").strip():
            records.append(Record(new_id(), "clause", newcomer.id, str(clause).strip(), now, pulse_id, note=f"self:{key}"))
    records = born_records + records
    await store.commit(world.canon, memories, born + [newcomer], records=records)
    fold_in(world, born + [newcomer], now)
    extend_history(world, memories, records)
    queue_witnessed(world, memories, exclude=frozenset({newcomer.id}))  # meeting: the players there see who walked in
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
    abhasa.after_join(world, newcomer.id, req.appearance, req.self_description)  # their portrait, in the background (Marc, 2026-11-11)
    return JoinResponse(player_entity_id=newcomer.id, narration=arrival["opening"], narrations={newcomer.id: arrival["opening"]})


def seeder(client):
    """Who writes a seed -- a birth's, an expansion's: the second model first, the other
    only if it fails (Marc, 2026-10-01). A single-provider world has only one."""
    from smritilila.llm import providers
    word = providers.seed_word()
    return client.prefer(word) if word and hasattr(client, "prefer") else client


def current_entry_point_ids(world: WorldSession) -> list[str]:
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


def join_locations(world: WorldSession) -> list[LocationOut]:
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
            here.setdefault(place, []).append(labelled(world.canon, pid))
    ids = [loc for loc, _ in sorted(traffic.items(), key=lambda kv: (-kv[1], kv[0]))[:BUSIEST_LOCATIONS]]
    ids += [loc for loc in here if loc not in ids]
    if not ids:
        ids = list(world.entry_point_location_ids)
    return [LocationOut(location_id=loc, label=labelled(world.canon, loc), memories=traffic.get(loc, 0),
                        players_here=here.get(loc, [])) for loc in ids]
