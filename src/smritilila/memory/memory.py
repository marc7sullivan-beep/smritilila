# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The tuple: one perceiver's memory of one pulse (v1.4 S8, S9, S11; v3
s2). Immutable once written: the weight is assigned once by the Narrator
and never revised ("there is no reinforcement or decay of an existing
salience value"); only absorb_entity (persistence/repository.py) re-points
the ids -- entity_id, location_id, subject_ids -- when one entity is merged
into another. v3 writes them in smrti/pulse.py; retrieval is
smrti/context.py. The v2 writers that lived here are history (2026-09-21).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Tier = Literal["PRIMARY", "SECONDARY", "AMBIENT"]


@dataclass(frozen=True)
class Memory:
    id: str
    entity_id: str  # the perceiver this memory belongs to
    subject_ids: frozenset[str]  # entities this memory is "about", for retrieval
    experience: str  # the tier-appropriate perspectival content
    tier: Tier
    salience: int  # 1-7, 1 = most salient; assigned once, immutable
    location_id: str
    world_time: int
    pulse_id: str
    # Computed once at write time (if an embed function is supplied) and
    # persisted alongside the memory, not recomputed on every retrieval --
    # None for memories written before an Embedder existed, or when none
    # was supplied. Similarity is computed in smrti/context.py.
    embedding: list[float] | None = None
    # v5 §3 (2026-09-23): what the moment GIVES its perceiver -- key,
    # useful or none -- beside salience, which is what it DID to them.
    # None on every tuple written before v4, read as none.
    use: str | None = None
    # v5 §19 kāla, a world that keeps a clock: the minutes of its pulse this
    # perceiver lived -- where the pulse ends, all of it; elsewhere, its
    # first moment (ended == began). Marc, 2026-09-24: "aren't time records
    # just another field associated with a memory?" None: no clock, or
    # written before one.
    began: int | None = None
    ended: int | None = None
