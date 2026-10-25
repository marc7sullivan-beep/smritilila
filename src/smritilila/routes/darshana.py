# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Darśana (2026-10-05, Marc's sign-off): what is seen. The reads of one
entity: the scene it stands in, its memories, anusmṛti (its character as
the world knows it), the recap, and what it has witnessed since it last
asked.

Moved whole from main.py, nothing inside changed."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila.adhisthana import _store
from smritilila.smrti.context import NOWHERE
from smritilila.schemas import (
    Anusmrti, AnusmrtiOut, EntityMemories, EntityMemoryOut, EntityRecap, EntityScene, WitnessedResponse,
)


router = APIRouter()


@router.get("/entities/{entity_id}/scene", response_model=EntityScene)
async def entity_scene(entity_id: str) -> EntityScene:
    world = await adhisthana.get_world()
    if entity_id not in world.canon.entities:
        raise HTTPException(404, f"no such entity: {entity_id}")
    # Where you stand and who is there come from containment (v5 §11),
    # read from the tuples -- the graph's answer froze at the cutover
    # (Marc, Tombstone, 2026-09-18: "a lot of entities present from the
    # town"). No adjacency is returned: adjacent is always empty.
    place, present = world.here(entity_id)
    return EntityScene(location_id=place, present=present, adjacent=[])


@router.get("/entities/{entity_id}/memories", response_model=EntityMemories)
async def entity_memories(entity_id: str, limit: int = 8) -> EntityMemories:
    world = await adhisthana.get_world()
    if entity_id not in world.canon.entities:
        raise HTTPException(404, f"no such entity: {entity_id}")
    own = sorted(world.samskara().mem.by_entity.get(entity_id, ()), key=lambda m: (m.world_time, m.id))
    return EntityMemories(memories=[EntityMemoryOut(world_time=m.world_time, experience=m.experience, salience=m.salience)
                                    for m in own[-max(1, min(limit, 50)):]])


@router.get("/entities/{entity_id}/anusmrti", response_model=Anusmrti)
async def entity_anusmrti(entity_id: str) -> Anusmrti:
    """What this being would call to mind on its own turn -- its weighty and
    key memories, what followed them, its latest -- by the same retrieval the
    Narrator is given, filtered to its own stream. An agent player reads it
    in place of the last few memories, so it knows its character as the
    world does (Marc, 2026-10-02). Read-only. Cost: one retrieval."""
    from smritilila.smrti.narrator import SALIENCE_NAMES
    from smritilila.smrti.pulse import anusmrti
    world = await adhisthana.get_world()
    if entity_id not in world.canon.entities:
        raise HTTPException(404, f"no such entity: {entity_id}")
    rows = await asyncio.to_thread(anusmrti, world.samskara(), world.canon, entity_id, world.canon.world_time + 1,
                                   adhisthana.get_embedder().embed, players=world.unvoiced() | frozenset(world.playable_entity_ids),
                                   absent=world.absent(), view=world.view())
    return Anusmrti(memories=[AnusmrtiOut(world_time=t.chronology, experience=t.memory,
                                          weight=str(SALIENCE_NAMES.get(t.salience, t.salience)), use=t.use or "none", why=why)
                              for t, why in rows])


@router.get("/entities/{entity_id}/recap", response_model=EntityRecap)
async def entity_recap(entity_id: str) -> EntityRecap:
    """Lets a client re-orient after switching to (or resuming) a
    character: where they currently are, and the last thing they
    themselves experienced. Reads the world's kept reading (world.here)
    for location (always live/correct regardless of switching) and the
    persistence layer directly for the narration (world.recent_memories
    was an in-process cache, empty on every fresh restart until new
    pulses accumulated -- it is seeded from the store on load since
    2026-09-16 -- the exact gap session_config's own persistence already
    had to fix once for scene_owner_entity_id)."""
    world = await adhisthana.get_world()
    if entity_id not in world.canon.entities:
        raise HTTPException(404, f"no such entity: {entity_id}")
    location_id, _ = world.here(entity_id, absent=False)  # v3: where their tuples last record them
    if location_id == NOWHERE:
        location_id = None
    last_memory = await _store(world).last_memory_for(entity_id)
    return EntityRecap(location_id=location_id, last_narration=last_memory.experience if last_memory is not None else None)


@router.get("/entities/{entity_id}/witnessed", response_model=WitnessedResponse)
async def entity_witnessed(entity_id: str) -> WitnessedResponse:
    """What this player has witnessed since they last looked -- the place
    tuples written where they stood while they were not acting (s19) --
    handed over and drained, so the page can show them AS THEY HAPPEN
    (Marc, 2026-09-21: the tuple is now the player's own the moment it
    is written; it must reach the player then, not with their next
    action). Whichever asks first, this or the next pulse's response,
    gets them; nothing is shown twice."""
    world = await adhisthana.get_world()
    if entity_id not in world.playable_entity_ids:
        raise HTTPException(422, f"entity_id must be one of {world.playable_entity_ids}")
    return WitnessedResponse(witnessed=world.pending_witnessed.pop(entity_id, []))
