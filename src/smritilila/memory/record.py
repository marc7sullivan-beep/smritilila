# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""v4's structural records (Design Principles v4 draft, 2026-09-23):
append-only facts that are not anyone's experience, kept apart from the
memory log so that nothing reading memories as experience ever sees them.

  kosha  -- *at world_time t, entity is in value* (v5 §10). value is the
            container's id, or None for *in nothing*. A move is a new
            record; nothing is ever edited.
  name   -- entity is called value (v5 §16): the name declared at mint,
            and every alias declared after. Exact, never taken away.
  action -- a player's own words for a pulse (v5 §35): value is the
            text as sent, note is "typed", "suggested" (the page's
            suggestion, sent unchanged), "autonomous" (v5 §37) or
            "default" (a round closed on the first suggestion of a
            player who had not acted, v5 §36).
  clause -- entity bears value, a clause of a view (v5 §7-§8), under note
            "<reach>:<key>" -- reach "within" (holds for everything in
            the bearer) or "self" (the bearer alone); value None is the
            clause withdrawn. A clause authored in play is always "self".
  capability -- entity is given (note "on") or loses (note "off") the
            capability named in value: "perceives" (a scene with a
            perspective of its own), "reaches" (perception beyond a
            scene and its neighbours). A reach, as a clause has (v5 §15):
            "on"/"off" is the bearer's own; "within:on"/"within:off"
            holds for everything inside it -- a seed's, never play's.
  prakriti -- the world's nature (v5 §15): entity, a branch or the earth,
            bears value (JSON) under the key in note -- pacing, daiva,
            clock, phala, source, ...; value None is it withdrawn. Read
            from the tree: the nearest that says, else the earth.
  said   -- the line spoken aloud in a scene's record for a pulse: entity
            is the scene, value the line, note the speaker. Structured, so
            no language is written into memory text by the engine.
  closes -- (v5.33) at world_time t, entity's experience at t CLOSES
            entity's own earlier experience at tick `value` (an integer, the
            head): the Narrator wrote that the matter that memory opened is
            over. Retrieval shows a head and its tail, the latest closer,
            together, and nothing between.
  follows -- (v5.30, retired v5.33) rows written by the chains the spec no
            longer has. Kept (records are insert-only), never read.
  mark   -- a fact about a memory of that pulse the engine must not write
            into it: value "as_narrated" (an actor tuple stored as the
            narration, not rewritten).
  time   -- (v5 §19) the time a pulse took: entity's clock reaches minute
            `value` (an integer), note "began|origin|dest". Written now
            only for the world at large's chronicle (note "began||"); the
            per-person ones of the first clock build are still read.
  sandhi -- (v5 §13) at world_time t, entity, a boundary between scenes, is
            also in the scene `value`; note "open", or "closed" once shut.
  reckoned -- (v5 §8, phala) entity, a doer, was last reckoned at world
            time `value` (an integer): where their next reckoning counts
            from. Kept out of the world's view.
  secret -- (v5 §2) entity is kept secret from every god outside its
            holder, `value`: a proposal that names it is withheld from the
            others, and what a secret proposal plants is born secret.

A record is never updated or deleted: where an entity is at tick t is
its latest kosha record before t; what it is called is every name
record it has. The one exception is absorb_entity (v5 §6; persistence/
repository.py), which re-points the ids when one entity is merged into
another: a record's entity_id, a kosha or sandhi value, a said note.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# As the store's records_kind_check accepts them (persistence/models.py): the
# two must be changed together. Pratyabhijñā's own store also writes
# "supersedes", which no Smritilila world holds.
RecordKind = Literal["kosha", "name", "action", "clause", "capability", "said", "mark", "time", "sandhi", "reckoned", "prakriti", "secret", "follows", "closes"]


@dataclass(frozen=True)
class Record:
    id: str
    kind: RecordKind
    entity_id: str
    value: str | None
    world_time: int
    pulse_id: str
    note: str | None = None
