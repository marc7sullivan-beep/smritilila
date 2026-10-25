# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Māyā -- the world as the engine holds it in memory: a clock and the
entity registry, nothing else (v3, 2026-09-21: the v2 delta log, the
relation graph, candidates and field state are history). Everything the
world IS lives in tuples (smritilila.memory.memory.Memory) and, from v4, records
(smritilila.memory.record.Record); an entity here is
an id with a label and the scheduler's bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smritilila.graph.entity import Entity


@dataclass
class Maya:
    world_time: int
    entities: dict[str, Entity] = field(default_factory=dict)

    def fork(self) -> "Maya":
        """A pulse's working world: its own clock and its own entity table
        -- what it mints is added there, a rerun takes its mints back out --
        over the same entity objects. Exact because a pulse never changes
        an entity in place: where a thing is, what it is called, what it
        has become and when are records (v4); an entity is an id, its
        labels and the scheduler's stamp, which only the live world writes.
        tests/test_smrti.py holds a pulse to it. (Was a deep copy per pulse,
        ~7 us an entity: 2026-09-25, Marc.)"""
        return Maya(world_time=self.world_time, entities=dict(self.entities))
