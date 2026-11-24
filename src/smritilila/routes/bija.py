# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The seed's routes (2026-10-05, Marc's sign-off; the engine is smritilila.smrti.bija): how a world is born and grows, as
requests. /genesis (one at a time, _genesis_lock), joining (/world/entry-points, /world/locations, /world/join), /world/expand,
/world/describe, the bhuvana and the world's view, the becomings and containments a god adds, and absorbing one entity into
another.

What a genesis and a join do -- the seed planted, the newcomer written and committed -- is smritilila/udbhava.py's, and the world's
one line smritilila/itihasa.py's (both moved there 2026-11-10, nothing inside changed); called through those modules (udbhava.X,
itihasa.X) so a test's patch there reaches them. In tests this module is imported as bija_routes: `bija` there is the engine.
"""

from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from smritilila.graph.canon import Maya
from smritilila.ids import new_id
from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila import dvara  # the gate, called through it for the same reason
from smritilila.adhisthana import (
    WorldSession, earth_of, extend_history, fold_in, store_of, stream_of, app,
)
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import NOWHERE
from smritilila.smrti.genesis import ITIHASA
from smritilila.smrti import prakriti
from smritilila.sakshi import current_world, log_event
from smritilila.itihasa import at_large, chronicle, chronicle_id
from smritilila.schemas import (
    AbsorbResponse, BecomingRequest, BhuvanaRequest, BhuvanaResponse, ContainmentRequest, DescribeResponse,
    EntryPointOut, ExpandResponse, GenesisRequest, GenesisResponse, JoinRequest, JoinResponse, LocationOut,
    WorldView,
)
from smritilila import spanda  # the pulse's services, and the names a test patches there, called through it
from smritilila import udbhava  # genesis and joining, called through it so a test's patch there reaches them
from smritilila import itihasa  # the world's one line (describe), called through it for the same reason


router = APIRouter()


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
    store = store_of(world)
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
    earth = earth_of(world.canon)
    if earth is None:  # a world with no earth yet keeps its view beside it, as before v5 §15
        await store_of(world).save_view(*new)
        written = 0
    else:  # v5 §15: what changed, appended on the earth -- the whole seed, the first time
        old = (world.darsana, world.becomings, world.world_data) if prakriti.sown(world.records, earth) else prakriti.EMPTY
        sown = prakriti.sow(earth, old, new, world.canon.world_time, f"view:{new_id()}")
        await store_of(world).commit(world.canon, [], [], records=sown)
        extend_history(world, [], sown)
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
    return BhuvanaResponse(bhuvana=world.bhuvana, chronicle=chronicle(world, 50)[1:])


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
    roles = adhisthana.resolve_roles()
    req = req or BhuvanaRequest()
    setting = (req.setting or world.setting_description or "").strip() or None
    self_description = (req.self_description or "").strip()
    if not self_description:
        owner = world.canon.entities.get(world.scene_owner_entity_id)
        own = sorted(stream_of(world, world.scene_owner_entity_id), key=lambda m: (m.world_time, m.id))[:3]
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
    earth = chronicle_id(world)
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
        await store_of(world).commit(world.canon, new_memories, new_entities, records=new_records)
        if new_entities:
            fold_in(world, new_entities, world.canon.world_time)
        extend_history(world, new_memories, new_records)
        world.bhuvana = bhuvana
        await store_of(world).save_bhuvana(bhuvana)
    log_event("bija.bhuvana_planted", pulse_id=pulse_id, chronicler=earth or new_entities[0].id, words=len(bhuvana.split()))
    background_tasks.add_task(itihasa.describe, world, roles)  # the root has changed: its line is drawn again
    return BhuvanaResponse(bhuvana=bhuvana, chronicle=[])


@router.post("/world/describe", response_model=DescribeResponse)
async def describe_world() -> DescribeResponse:
    """A world's gods draw its one line again from its root."""
    world = await adhisthana.get_world()
    line = await itihasa.describe(world, adhisthana.resolve_roles())
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
    roles = adhisthana.resolve_roles()
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
    key_to_id, places = udbhava.place_keys(world, kosha, place_ids, t_now, only=outer)
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
    new = await asyncio.to_thread(expand, udbhava.seeder(roles.smrti_client), world.setting_description, places, standing, names_taken, labels_taken,
                                  at_large(world, spanda.players_latest(world)), frozenset(key_to_id), source=world.world_data.get("source"),
                                  darsana=world.darsana, titles=world.world_data.get("name_titles"))
    if not new["places"] and not new["people"]:
        raise HTTPException(502, "the expansion added nothing")
    births = await udbhava.births_of(roles, new["people"], lambda k: (new["places"].get(k) or {}).get("label") or (places.get(k) or (k,))[0], world.bhuvana,
                              world.darsana, world.world_data.get("source"))
    async with world.builder_lock:
        entities, memories, planted = plant_more(new, key_to_id, adhisthana.get_embedder().embed, world.canon.world_time, pulse_id=f"vistara:{last + 1}",
                                                 earth=chronicle_id(world), births=births)
        await store_of(world).commit(world.canon, memories, entities, records=planted)
        fold_in(world, entities, world.canon.world_time)
        extend_history(world, memories, planted)
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

        roles = adhisthana.resolve_roles()  # 503 after emptiness/validation checks above
        born = await udbhava.v3_genesis(req, world, roles)
    if background_tasks is not None:  # called in-process (a test): no line drawn
        background_tasks.add_task(itihasa.describe, world, roles)  # its root is laid: its line, for those deciding to enter
    return born


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
    await store_of(world).commit(world.canon, [], [], records=[r])
    extend_history(world, [], [r])
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
    if eid == chronicle_id(world):
        raise HTTPException(422, "the earth is never moved: it holds all, and nothing holds it")
    if eid in world.playable_entity_ids:
        raise HTTPException(422, "a player's body is their own")
    r = Record(new_id(), "kosha", eid, req.into, world.canon.world_time, f"owner:{new_id()}", note="owner")
    await store_of(world).commit(world.canon, [], [], records=[r])
    extend_history(world, [], [r])
    log_event("world.containment_added", entity_id=eid, into=req.into, why=req.why[:300], world_time=world.canon.world_time)


@router.get("/world/entry-points", response_model=list[EntryPointOut])
async def entry_points() -> list[EntryPointOut]:
    world = await adhisthana.get_world()
    return [EntryPointOut(location_id=loc_id) for loc_id in udbhava.current_entry_point_ids(world)]


@router.get("/world/locations", response_model=list[LocationOut])
async def join_locations() -> list[LocationOut]:
    """Where a new character may arrive (join_locations)."""
    world = await adhisthana.get_world()
    return udbhava.join_locations(world)


@router.post("/world/join", response_model=JoinResponse)
async def join_world(req: JoinRequest, request: Request) -> JoinResponse:
    if not req.self_description.strip():
        raise HTTPException(422, "self_description must not be empty")
    world = await adhisthana.get_world()
    if not req.start_description.strip():
        valid_entry_points = [loc.location_id for loc in udbhava.join_locations(world)]
        if req.entry_location_id not in valid_entry_points:
            raise HTTPException(422, f"entry_location_id must be one of {valid_entry_points}, or describe where you start")

    roles = adhisthana.resolve_roles()
    joined = await udbhava.v3_join(req, world, roles)
    guest = dvara.guest_of(request)
    if guest:  # the character a guest makes is theirs to play (and only theirs)
        world.played_by[joined.player_entity_id] = guest  # the tether (Marc, 2026-09-30): not the world's view
        await store_of(world).save_played_by(world.played_by)
        log_event("guest.joined", guest=guest, entity_id=joined.player_entity_id)
    return joined
