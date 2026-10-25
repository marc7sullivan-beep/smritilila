# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Read/write for the v3 schema (persistence/models.py: entities,
memories, session_config) and v4's records. memories (and records) are
insert-only, but for absorb_entity below, which re-points ids on both when
one entity is merged into another; entities is upserted;
session_config is one row. The v2 functions that lived here -- candidates,
the delta log, current relations, field state, samskara -- are history
(2026-09-21).
"""

from __future__ import annotations

import logging

from sqlalchemy import case, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.persistence.models import EntityRow, MemoryRow, RecordRow, SessionConfigRow

logger = logging.getLogger(__name__)

SESSION_CONFIG_ID = "singleton"  # single-WorldSession app -- see SessionConfigRow's own docstring


# -- Entity -------------------------------------------------------------


def entity_to_row(entity: Entity) -> EntityRow:
    return EntityRow(
        id=entity.id,
        labels=list(entity.labels),
        created_at_world_time=entity.created_at_world_time,
        origin=entity.origin,
        last_simulated_tick=entity.last_simulated_tick,
        avarana=entity.avarana,
        kosha=entity.kosha,
    )


def row_to_entity(row: EntityRow) -> Entity:
    return Entity(
        id=row.id,
        labels=list(row.labels),
        created_at_world_time=row.created_at_world_time,
        origin=row.origin,  # type: ignore[arg-type]
        last_simulated_tick=row.last_simulated_tick,
        avarana=row.avarana,  # type: ignore[arg-type]
        kosha=row.kosha,
    )


async def upsert_entities(session: AsyncSession, entities: list[Entity]) -> None:
    """A real upsert (INSERT ... ON CONFLICT DO UPDATE). Live, 2026-09-17
    (tick 201): a background activation pulse and the player's pulse
    each promoted the same candidate; merge() SELECTed before the other
    transaction committed, found nothing, INSERTed, and the second commit
    died on entities_pkey -- the player's pulse lost to a 500."""
    if not entities:
        return
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    # Always in id order, so every transaction takes its row locks in the
    # same order and a cycle cannot form (2026-09-22). An ON CONFLICT DO
    # UPDATE takes an exclusive row lock, and two pulses commit
    # concurrently by design -- the player's and the deferred activation's
    # (routes/spanda.py's background task). The order came from
    # `set(touched_ids)` in world_store.commit, and a Python set orders by
    # hash modulo its table size, so two DIFFERENT sets that share two
    # entities routinely iterate them oppositely: measured, 12 of 12
    # realistic pairings from the crossing flipped. That is a deadlock
    # window and Postgres found it -- one v3 pulse lost, 2026-09-22:
    #   Process 9804 waits for ShareLock on transaction 10718;
    #   blocked by process 14804.
    #   Process 14804 waits for ShareLock on transaction 10719;
    #   blocked by process 9804.
    # Sorting is the whole fix: no retry (which would re-run a commit
    # whose narration already happened) and no serialising the two
    # commits (which would give back the concurrency the deferred
    # activation exists to buy). The memories written just below need no
    # such ordering -- their FK takes KEY SHARE, and two KEY SHAREs on a
    # row do not conflict with each other.
    for entity in sorted(entities, key=lambda e: e.id):
        row = entity_to_row(entity)
        values = {
            "id": row.id, "labels": row.labels, "created_at_world_time": row.created_at_world_time, "origin": row.origin,
            "last_simulated_tick": row.last_simulated_tick,
            "avarana": row.avarana, "kosha": row.kosha,
        }
        stmt = pg_insert(EntityRow).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[EntityRow.id], set_={k: getattr(stmt.excluded, k) for k in values if k != "id"}
        )
        await session.execute(stmt)
    # memories.entity_id FKs to entities.id: flush before anything that
    # references a freshly promoted id.
    await session.flush()


async def load_entities(session: AsyncSession) -> dict[str, Entity]:
    rows = (await session.execute(select(EntityRow))).scalars().all()
    return {row.id: row_to_entity(row) for row in rows}


async def load_canon(session: AsyncSession) -> Maya:
    """The world as the engine holds it: the entity registry and the
    clock. v3's clock is the last tuple's tick (a world whose pulses
    append only memories), or chronology restarts at 1 on every reload
    and lands before what was written."""
    entities = await load_entities(session)
    max_memory_time = (await session.execute(select(func.max(MemoryRow.world_time)))).scalar()
    return Maya(world_time=max_memory_time or 0, entities=entities)


# -- Session config -------------------------------------------------------


async def save_session_config(
    session: AsyncSession,
    scene_owner_entity_id: str,
    entry_point_location_ids: list[str],
    playable_entity_ids: list[str],
    setting_description: str | None = None,
) -> None:
    await session.merge(
        SessionConfigRow(
            id=SESSION_CONFIG_ID,
            scene_owner_entity_id=scene_owner_entity_id,
            entry_point_location_ids=list(entry_point_location_ids),
            playable_entity_ids=list(playable_entity_ids),
            setting_description=setting_description,
        )
    )
    await session.flush()


async def absorb_entity(session: AsyncSession, keep: str, gone: str) -> dict[str, int]:
    """s10's merge as a mechanism (2026-09-20): every tuple of `gone`
    becomes `keep`'s, every tuple filed at `gone` (a place) is filed at
    `keep`, every presence list naming `gone` names `keep`, and `gone`'s
    row is dropped.
    Its name: aikya (ऐक्य), oneness (v5 §6).
    Nothing is rewritten but the ids. Returns counts."""
    from sqlalchemy import text as sql
    out: dict[str, int] = {}
    r = await session.execute(sql("update memories set entity_id = :keep where entity_id = :gone"), {"keep": keep, "gone": gone})
    out["tuples"] = r.rowcount
    r = await session.execute(sql("update memories set location_id = :keep where location_id = :gone"), {"keep": keep, "gone": gone})
    out["filed"] = r.rowcount
    r = await session.execute(sql("update memories set subject_ids = replace(subject_ids::text, :gone, :keep)::jsonb where subject_ids::text like :pat"),
                              {"keep": keep, "gone": gone, "pat": f"%{gone}%"})
    out["presence"] = r.rowcount
    # v4's records (2026-09-24, Ravenmoor's Elias copy): this merge predates
    # them, and would have left the gone id's containment, name and clause
    # records -- and anything contained IN it -- naming an entity that no
    # longer exists. Its records become keep's; what was in it is in keep.
    r = await session.execute(sql("update records set entity_id = :keep where entity_id = :gone"), {"keep": keep, "gone": gone})
    out["records"] = r.rowcount
    r = await session.execute(sql("update records set value = :keep where kind = 'kosha' and value = :gone"), {"keep": keep, "gone": gone})
    out["contained"] = r.rowcount
    # 2026-09-25 review: a boundary joined to the gone scene (sandhi, v5 §13) joins keep; and a seed's place
    # whose containment was the entity's own field (the old way, before records) is in keep, not in nothing
    r = await session.execute(sql("update records set value = :keep where kind = 'sandhi' and value = :gone"), {"keep": keep, "gone": gone})
    out["joined"] = r.rowcount
    r = await session.execute(sql("update entities set kosha = :keep where kosha = :gone"), {"keep": keep, "gone": gone})
    out["field"] = r.rowcount
    # 2026-10-03: a line the gone one said is keep's line -- its speaker is the record's note (Stonemarch's second
    # Ostler and second Jaroo, Ravenmoor's grey-faced copy: 21 lines left naming no one)
    r = await session.execute(sql("update records set note = :keep where kind = 'said' and note = :gone"), {"keep": keep, "gone": gone})
    out["speaker"] = r.rowcount
    r = await session.execute(sql("delete from entities where id = :gone"), {"gone": gone})
    out["dropped"] = r.rowcount
    return out


async def save_bhuvana(session: AsyncSession, text: str | None) -> None:
    """The world at large, on the session row (which must exist)."""
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    if row is not None:
        row.bhuvana = text


async def load_bhuvana(session: AsyncSession) -> str | None:
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    return row.bhuvana if row is not None else None


async def save_view(session: AsyncSession, darsana: str | None, becomings: dict, world_data: dict) -> None:
    """v5 §7-§8: the world's own view, its library of becomings, its world data."""
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    if row is not None:
        row.darsana, row.becomings, row.world_data = darsana, dict(becomings), dict(world_data)


async def load_view(session: AsyncSession) -> tuple[str | None, dict, dict]:
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    if row is None:
        return None, {}, {}
    return row.darsana, dict(row.becomings or {}), dict(row.world_data or {})


async def save_autonomy(session: AsyncSession, player_ids: list[str]) -> None:
    """v5 §37: who allows the world to act for them, on the session row."""
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    if row is not None:
        row.autonomous_players = sorted(set(player_ids))


async def load_autonomy(session: AsyncSession) -> list[str]:
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    return list(row.autonomous_players or []) if row is not None else []


async def save_logged_off(session: AsyncSession, player_ids: list[str]) -> None:
    """Marc, 2026-09-24: who logged off, on the session row."""
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    if row is not None:
        row.logged_off_players = sorted(set(player_ids))


async def load_logged_off(session: AsyncSession) -> list[str]:
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    return list(row.logged_off_players or []) if row is not None else []


async def save_waiting(session: AsyncSession, player_ids: list[str]) -> None:
    """Marc, 2026-10-01: who waits -- in the world, no turn of their own until touched."""
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    if row is not None:
        row.waiting_players = sorted(set(player_ids))


async def load_waiting(session: AsyncSession) -> list[str]:
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    return list(row.waiting_players or []) if row is not None else []


async def save_held_back(session: AsyncSession, ids: list[str]) -> None:
    """Marc, 2026-10-02: a guest's retired character, out of the world until a god reviews it."""
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    if row is not None:
        row.held_back = sorted(set(ids))


async def load_held_back(session: AsyncSession) -> list[str]:
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    return list(row.held_back or []) if row is not None else []


async def save_played_by(session: AsyncSession, played_by: dict[str, str]) -> None:
    """Marc, 2026-09-30: the tether -- which character each guest plays."""
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    if row is not None:
        row.played_by = dict(played_by)


async def load_played_by(session: AsyncSession) -> dict[str, str]:
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    return dict(row.played_by or {}) if row is not None else {}


async def load_session_config(session: AsyncSession) -> tuple[str, list[str], list[str], str | None] | None:
    row = await session.get(SessionConfigRow, SESSION_CONFIG_ID)
    if row is None:
        return None
    return (
        row.scene_owner_entity_id, list(row.entry_point_location_ids), list(row.playable_entity_ids),
        row.setting_description,
    )


# -- Memory --------------------------------------------------------------


def memory_to_row(memory: Memory) -> MemoryRow:
    return MemoryRow(
        id=memory.id,
        entity_id=memory.entity_id,
        subject_ids=list(memory.subject_ids),
        experience=memory.experience,
        tier=memory.tier,
        salience=memory.salience,
        location_id=memory.location_id,
        world_time=memory.world_time,
        pulse_id=memory.pulse_id,
        embedding=memory.embedding,
        use=memory.use,
        began=memory.began,
        ended=memory.ended,
    )


def row_to_memory(row: MemoryRow) -> Memory:
    return Memory(
        id=row.id,
        entity_id=row.entity_id,
        subject_ids=frozenset(row.subject_ids),
        experience=row.experience,
        tier=row.tier,  # type: ignore[arg-type]
        salience=row.salience,
        location_id=row.location_id,
        world_time=row.world_time,
        pulse_id=row.pulse_id,
        embedding=row.embedding,
        use=row.use,
        began=row.began,
        ended=row.ended,
    )


async def save_memories(session: AsyncSession, memories: list[Memory]) -> None:
    for memory in memories:
        session.add(memory_to_row(memory))


async def load_all_memories(session: AsyncSession) -> list[Memory]:
    """Every memory in the world, oldest first -- what WorldSession.
    recent_memories must START as on a process restart, not an empty
    list (live, 2026-09-16: the Rome run had been played for ten pulses
    after a restart with `memory.retrieval` firing zero times -- the
    Narrator, own_recent_memories, _last_shared_location, jijnasa and
    knows_way_to all read that in-process list, and nothing refilled it).
    Cost: one table scan at world load, the rows and their embeddings;
    retrieval reads the kept index (WorldSession.samskara(), read on with
    only what was committed since), so this adds nothing per pulse. When a world's
    memory table outgrows a single load, bound this per entity (last N
    plus every embedded row) rather than here."""
    rows = (await session.execute(select(MemoryRow).order_by(MemoryRow.world_time, MemoryRow.id))).scalars().all()
    return [row_to_memory(row) for row in rows]


async def load_last_memory(session: AsyncSession, entity_id: str) -> Memory | None:
    """The single most recent memory an entity has of its own experience
    -- for a client resuming a character (routes/darshana.py's GET /entities/{id}/
    recap) to show "here's what you were last doing" rather than a blank
    log. world.recent_memories (the in-process cache) is NOT a
    substitute for this: it was empty on every fresh process start (seeded from the store on load since 2026-09-16, adhisthana._load_world) until
    new pulses accumulate, same gap session_config's own persistence
    fixed for scene_owner_entity_id -- this needs the real, durable
    history, not whatever's happened to accumulate since the last restart."""
    stmt = select(MemoryRow).where(MemoryRow.entity_id == entity_id).order_by(MemoryRow.world_time.desc()).limit(1)
    row = (await session.execute(stmt)).scalars().first()
    return row_to_memory(row) if row is not None else None


async def clear_all(session: AsyncSession) -> None:
    """Wipes every row from the four tables -- the persistence side of
    "start a genuinely new world" (routes/loka.py's POST /world/clear). memories
    references entities, so it goes first; a leftover session_config row
    must not survive to be misread as the NEXT world's."""
    await session.execute(delete(RecordRow))
    await session.execute(delete(MemoryRow))
    await session.execute(delete(EntityRow))
    await session.execute(delete(SessionConfigRow))
    await session.flush()


# -- Records (v4) ---------------------------------------------------------


def record_to_row(r: Record) -> RecordRow:
    return RecordRow(id=r.id, kind=r.kind, entity_id=r.entity_id, value=r.value, note=r.note,
                     world_time=r.world_time, pulse_id=r.pulse_id)


def row_to_record(row: RecordRow) -> Record:
    return Record(id=row.id, kind=row.kind, entity_id=row.entity_id, value=row.value,  # type: ignore[arg-type]
                  world_time=row.world_time, pulse_id=row.pulse_id, note=row.note)


async def save_records(session: AsyncSession, records: list[Record]) -> None:
    for r in records:
        session.add(record_to_row(r))


async def load_all_records(session: AsyncSession) -> list[Record]:
    """Every record, oldest first. Cost as load_all_memories: one scan at
    world load; records are far fewer than tuples."""
    # Within one tick, a pulse writes a thing's `declared` placement before
    # any move it makes that pulse; ids are random, so ordering by id alone
    # put the two in a coin-flip order on every reload -- the crossing, 587:
    # a bowl declared on the hearthstone and carried out to the tin-man came
    # back on the hearthstone. `declared` first within a tick restores the
    # order it was written in, for every record already stored.
    declared_first = case((RecordRow.note == "declared", 0), else_=1)
    rows = (await session.execute(select(RecordRow).order_by(RecordRow.world_time, declared_first, RecordRow.id))).scalars().all()
    return [row_to_record(row) for row in rows]
