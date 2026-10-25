# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""smrti/janasruti.py -- hearsay: a heavy moment travels out along the tree on the world's clock (v4.62)."""

from __future__ import annotations

import json

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.janasruti import heard, steps_between


def _world(close_hearsay: bool = False):
    """A village (farm, inn) and a far valley (mill), under one land."""
    canon = Maya(world_time=60, entities={
        "land": Entity("land", ["the land"], 0, "genesis"),
        "village": Entity("village", ["the village"], 0, "genesis"),
        "farm": Entity("farm", ["the farm"], 0, "genesis", avarana="scene"),
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene"),
        "valley": Entity("valley", ["the far valley"], 0, "genesis"),
        "mill": Entity("mill", ["the valley mill"], 0, "genesis", avarana="scene"),
        "hobb": Entity("hobb", ["Hobb"], 50, "smrti"),            # minted in the pulse he died: that pulse is news
        "seeded": Entity("seeded", ["a planted priest"], 55, "genesis"),
    })
    recs = [Record("r1", "kosha", "village", "land", 0, "p"), Record("r2", "kosha", "farm", "village", 0, "p"),
            Record("r3", "kosha", "inn", "village", 0, "p"), Record("r4", "kosha", "valley", "land", 0, "p"),
            Record("r5", "kosha", "mill", "valley", 0, "p"), Record("r6", "kosha", "hobb", "farm", 50, "p50"),
            Record("r7", "kosha", "seeded", "inn", 55, "p55")]
    if close_hearsay:
        recs.append(Record("r8", "prakriti", "valley", json.dumps({"hearsay": False}), 0, "p", "pacing"))
    mems = [
        Memory("m1", "hobb", frozenset(), "Raiders cut me down at my own door.", "PRIMARY", 1, "farm", 50, "p50", None, began=1000, ended=1001),
        Memory("m2", "farm", frozenset({"hobb"}), "Raiders broke into the farmhouse and cut down the farmer.", "AMBIENT", 5, "farm", 50, "p50", None,
               began=1000, ended=1001),
        Memory("m3", "seeded", frozenset(), "I am the planted priest, and I serve the deep.", "PRIMARY", 1, "inn", 55, "p55", None, began=1100, ended=1100),
        Memory("m4", "inn", frozenset({"seeded"}), "A priest in grey sat by the inn fire.", "AMBIENT", 5, "inn", 55, "p55", None, began=1100, ended=1100),
    ]
    return canon, mems, recs


def _kosha(canon, mems, recs):
    from smritilila.smrti.samskara import Samskara
    return Samskara.build(canon, mems, recs, players=[], clock=None, defaults=None).log_for(canon).kosha


def test_a_heavy_moment_is_heard_next_door_once_its_time_has_passed_and_farther_later() -> None:
    canon, mems, recs = _world()
    k = _kosha(canon, mems, recs)
    assert steps_between(k, "farm", "inn", 61) == 2 and steps_between(k, "farm", "mill", 61) == 4
    assert heard(k, canon.entities, mems, "inn", 61, 1030) == []                        # not yet: two steps take 90 minutes
    told = heard(k, canon.entities, mems, "inn", 61, 1200)
    assert [(o, w) for o, _t, _a, w in told] == [("farm", 1)] and "cut down the farmer" in told[0][1]
    assert heard(k, canon.entities, mems, "farm", 61, 1200) == []                       # its own record already holds it
    assert heard(k, canon.entities, mems, "mill", 61, 1100) == []                       # four steps: not yet
    assert [o for o, *_ in heard(k, canon.entities, mems, "mill", 61, 1400)] == ["farm"]
    assert heard(k, canon.entities, mems, "mill", 61, 1001 + 4 * 24 * 60) == []         # old news after the horizon


def test_a_planted_birth_is_never_news_and_a_closed_branch_hears_nothing() -> None:
    canon, mems, recs = _world()
    k = _kosha(canon, mems, recs)
    assert "inn" not in [o for o, *_ in heard(k, canon.entities, mems, "farm", 61, 1500)]  # the planted priest's birth: who he is
    canon2, mems2, recs2 = _world(close_hearsay=True)
    assert heard(_kosha(canon2, mems2, recs2), canon2.entities, mems2, "mill", 61, 1400) == []  # pacing {"hearsay": false}
