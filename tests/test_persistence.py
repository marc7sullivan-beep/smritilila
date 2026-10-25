# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Row mappings for the v3 schema: entities and memories round-trip."""
from __future__ import annotations

import pytest

from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.persistence.repository import entity_to_row, memory_to_row, row_to_entity, row_to_memory


def test_entity_round_trips() -> None:
    entity = Entity(id="mara", labels=["Mara", "the guard"], created_at_world_time=5, origin="smrti", last_simulated_tick=3)
    row = entity_to_row(entity)
    assert row_to_entity(row) == entity


def test_entity_round_trips_with_null_scheduling_fields() -> None:
    entity = Entity(id="mara", labels=["Mara"], created_at_world_time=0, origin="genesis")
    row = entity_to_row(entity)
    assert row_to_entity(row) == entity
    assert row.last_simulated_tick is None


def test_memory_round_trips() -> None:
    memory = Memory(
        id="m1", entity_id="mara", subject_ids=frozenset({"mara", "player"}),
        experience="I felt the ground shake.", tier="PRIMARY", salience=2,
        location_id="square", world_time=10, pulse_id="p1",
    )
    row = memory_to_row(memory)
    assert row_to_memory(row) == memory
    assert row.embedding is None  # no Embedder supplied when this memory was written


def test_memory_round_trips_with_a_real_embedding() -> None:
    memory = Memory(
        id="m1", entity_id="mara", subject_ids=frozenset({"mara"}),
        experience="I felt the ground shake.", tier="PRIMARY", salience=2,
        location_id="square", world_time=10, pulse_id="p1", embedding=[0.1, -0.2, 0.3],
    )
    row = memory_to_row(memory)
    assert row.embedding == [0.1, -0.2, 0.3]
    assert row_to_memory(row) == memory


@pytest.mark.asyncio
async def test_entities_are_upserted_in_id_order_whatever_order_they_arrive_in() -> None:
    """The deadlock of 2026-09-22. ON CONFLICT DO UPDATE takes an
    exclusive row lock, and two pulses commit concurrently by design --
    the player's and the deferred activation's (routes/spanda.py's background
    task). The order came from `set(touched_ids)` in world_store.commit,
    and a set iterates by hash modulo its table size, so two DIFFERENT
    sets sharing two entities routinely order them oppositely: 12 of 12
    realistic pairings from the crossing flipped. Postgres found the
    cycle and one v3 pulse was lost. One global lock order is the whole
    fix -- no retry, no serialising the two commits."""
    from smritilila.persistence.repository import upsert_entities

    seen: list[str] = []

    class _Session:
        async def execute(self, stmt) -> None:
            seen.append(stmt.compile().params["id"])

        async def flush(self) -> None:
            return None

    arrived = [Entity(eid, [eid], 0, "genesis")
               for eid in ("sela_pedlar_b21c", "ilo_boy_2f27", "teo_ferryman_e5cf", "hollow_dead_oak")]
    await upsert_entities(_Session(), arrived)

    assert seen == ["hollow_dead_oak", "ilo_boy_2f27", "sela_pedlar_b21c", "teo_ferryman_e5cf"]
