# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The world's own turn, chosen (Appendix A.1, A.5): each tick at most one
entity is granted an activation episode -- the oldest in the exploration
pool, never-run first.

ACTIVATION_PULSE_BUDGET was locked at 0 (inert) through Phase 7 by
explicit user decision (see the project_smritilila_activation_episode_pulse_
budget memory); raised to 1 by explicit user instruction. This module
itself stays a PURE selection function -- it decides WHO is activated and
how many pulses (1) that grants, but does not run them. The caller
(routes/spanda.py's _after_smrti_commit) is responsible for actually spending the budget:
it hands pulses_run to _run_smrti_activation_in_background, which runs
the same v3 pulse a player gets with no action supplied and commits the
result through the same world store path. Selection stays here,
execution stays with the pulse -- this module has no LLM/persistence
dependencies and none should be added to it.

(v2's heap of scheduled wakes and its dormancy debt went on 2026-09-25:
nothing had scheduled a wake since v2, so the heap was always empty and
the exploration pool always chose.)
"""

from __future__ import annotations

from dataclasses import dataclass

from smritilila.graph.canon import Maya
from smritilila.sakshi import log_event

ACTIVATION_PULSE_BUDGET = 1


@dataclass(frozen=True)
class WorldTickResult:
    activated_entity_id: str | None
    pulses_run: int


def _select_from_exploration_pool(exploration_pool: list[str], canon: Maya) -> str | None:
    """Oldest-last_simulated_tick-first, never-run entities treated as
    maximally overdue and going first -- NOT the doc's full
    "differentiated entities compete individually, shallow entities
    compete collectively through one bounded bucket" split (Appendix D
    open items 2/3 already mark that split as something to watch in
    testing, not something this phase needs to nail)."""
    candidates = [eid for eid in exploration_pool if eid in canon.entities]
    if not candidates:
        return None

    def key(entity_id: str) -> tuple[int, int, str]:
        entity = canon.entities[entity_id]
        never_run = entity.last_simulated_tick is None
        return (0 if never_run else 1, entity.last_simulated_tick or 0, entity_id)

    return min(candidates, key=key)


def advance_world_tick(canon: Maya, current_tick: int, exploration_pool: list[str]) -> WorldTickResult:
    """One world tick: at most one entity is granted an activation episode
    (S18), stamped with this tick so the next choice is someone older."""
    entity_id = _select_from_exploration_pool(exploration_pool, canon)
    log_event("tick.source", tick=current_tick, source="exploration", entity_id=entity_id)
    if entity_id is None:
        return WorldTickResult(activated_entity_id=None, pulses_run=0)
    # Selection only -- the caller spends ACTIVATION_PULSE_BUDGET pulses
    # against this entity itself, see this module's own docstring.
    canon.entities[entity_id].last_simulated_tick = current_tick
    return WorldTickResult(activated_entity_id=entity_id, pulses_run=ACTIVATION_PULSE_BUDGET)
