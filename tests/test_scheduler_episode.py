# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.scheduler.episode import ACTIVATION_PULSE_BUDGET, advance_world_tick


def make_canon(**entities: Entity) -> Maya:
    return Maya(world_time=0, entities=dict(entities))


def test_last_simulated_tick_updates_to_current_tick_after_an_episode() -> None:
    canon = make_canon(storm=Entity("storm", ["Storm"], 0, "genesis"))
    advance_world_tick(canon, current_tick=5, exploration_pool=["storm"])
    assert canon.entities["storm"].last_simulated_tick == 5


def test_pulses_run_is_always_the_configured_budget() -> None:
    canon = make_canon(storm=Entity("storm", ["Storm"], 0, "genesis"))
    result = advance_world_tick(canon, current_tick=5, exploration_pool=["storm"])
    assert result.pulses_run == ACTIVATION_PULSE_BUDGET == 1


def test_exploration_pool_picks_a_never_run_entity_first() -> None:
    canon = make_canon(
        village=Entity("village", ["Village"], 0, "genesis", last_simulated_tick=1),
        forest=Entity("forest", ["Forest"], 0, "genesis", last_simulated_tick=None),
    )
    result = advance_world_tick(canon, current_tick=5, exploration_pool=["village", "forest"])
    assert result.activated_entity_id == "forest"


def test_exploration_pool_picks_the_oldest_last_simulated_tick_when_all_have_run() -> None:
    canon = make_canon(
        village=Entity("village", ["Village"], 0, "genesis", last_simulated_tick=3),
        forest=Entity("forest", ["Forest"], 0, "genesis", last_simulated_tick=1),
    )
    result = advance_world_tick(canon, current_tick=5, exploration_pool=["village", "forest"])
    assert result.activated_entity_id == "forest"


def test_an_empty_pool_activates_nothing() -> None:
    canon = make_canon()
    result = advance_world_tick(canon, current_tick=5, exploration_pool=[])
    assert result.activated_entity_id is None and result.pulses_run == 0


def test_the_pool_ignores_an_id_the_canon_does_not_hold() -> None:
    canon = make_canon(forest=Entity("forest", ["Forest"], 0, "genesis"))
    assert advance_world_tick(canon, current_tick=5, exploration_pool=["gone", "forest"]).activated_entity_id == "forest"


def test_tick_source_is_logged(caplog) -> None:
    canon = make_canon(forest=Entity("forest", ["Forest"], 0, "genesis"))

    with caplog.at_level(logging.INFO, logger="smritilila.sakshi"):
        advance_world_tick(canon, current_tick=5, exploration_pool=["forest"])

    events = [json.loads(r.message) for r in caplog.records if r.name == "smritilila.sakshi"]
    # sakshi's three stamps ride on every line (2026-09-22) -- asserted
    # in tests/test_sakshi.py, dropped here so this test keeps asserting
    # what the scheduler itself logs.
    for e in events:
        assert e.pop("when") and e.pop("run")
        e.pop("world")
    assert events == [{"event": "tick.source", "tick": 5, "source": "exploration", "entity_id": "forest"}]
