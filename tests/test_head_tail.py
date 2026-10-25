# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""v5.33 -- the Narrator says when a tail closes a head; retrieval shows the head and its tail together, and nothing between
(Marc, 2026-10-13, from his "Head and tail"; replaces v5.30's causal chains, which walked every step)."""

from __future__ import annotations

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import PAIR_CAP, Smrti, Tuple, head_tail_company, retrieve
from smritilila.smrti.log import build_log, extend_log
from smritilila.smrti.narrator import _SCHEMA_FACES, SCHEMA
from smritilila.smrti.pulse import closes_records


def _stream() -> Smrti:
    """Fil's days: a fall (10), a bad night (11), a fever (12), a broth (13), a healing (20); years of road (21-60); now (61)."""
    log = Smrti()
    seq = 0

    def add(p, t, text, place=None, sal=4):
        nonlocal seq
        seq += 1
        log.append(Tuple(p, seq, t, text, sal, place, None))
    add("road", 0, "The high road", "road")
    add("fil", 0, "Filonius Applegate")
    add("fil", 10, "I fell and cracked my ribs.", sal=2)
    add("fil", 11, "A bad night in the hay.", sal=6)
    add("fil", 12, "Fever, and a carter.", sal=6)
    add("fil", 13, "The hayloft; a broth.", sal=6)
    add("fil", 20, "My ribs healed; I left the crutch behind.", sal=7)
    for i in range(21, 61):
        add("fil", i, f"Walking, tick {i}.", sal=5)
    add("fil", 61, "The road again.")
    return log


class _Ctx(dict):
    """A bare context of (perceiver, seq) -> Tuple, with a phase, as retrieve's is."""

    def __init__(self, log, picks):
        super().__init__()
        self.phase = "x"
        self.source = {}
        for who, tick in picks:
            for t in log._at[(who, tick)]:
                self[(t.perceiver, t.seq)] = t

    def add(self, t):
        self[(t.perceiver, t.seq)] = t


def _ticks(ctx) -> set[int]:
    return {t.chronology for t in ctx.values()}


def test_nothing_is_added_without_a_closing() -> None:
    ctx = retrieve(_stream(), "fil", 62, None, k_place=0, k_sim=0)
    assert ctx is not None and not any(src == "head_tail" for src in ctx.sources.values())


def test_a_head_shown_brings_its_tail() -> None:
    log = _stream()
    base = retrieve(log, "fil", 62, None, k_place=0, k_sim=0)
    assert any(t.chronology == 10 for t in base.tuples) and not any(t.chronology == 20 for t in base.tuples)  # the fall is shown for its weight; the healing is not
    log.close("fil", 20, 10)
    ctx = retrieve(log, "fil", 62, None, k_place=0, k_sim=0)
    healing = [t for t in ctx.tuples if t.chronology == 20]
    assert healing and ctx.sources[(healing[0].perceiver, healing[0].seq)] == "head_tail"


def test_a_tail_shown_brings_its_head() -> None:
    log = _stream()
    log.close("fil", 20, 10)
    ctx = _Ctx(log, [("fil", 20)])
    assert head_tail_company(log, ctx, "fil", 62) == 1 and _ticks(ctx) == {10, 20}


def test_nothing_between_is_ever_added() -> None:
    log = _stream()
    log.close("fil", 20, 10)                       # the bad night, the fever and the broth close nothing and are never named
    for shown in (10, 20):
        ctx = _Ctx(log, [("fil", shown)])
        head_tail_company(log, ctx, "fil", 62)
        assert _ticks(ctx) == {10, 20}


def test_a_matter_closed_twice_has_the_latest_closer_for_its_tail() -> None:
    log = _stream()
    log.close("fil", 13, 10)                       # the Narrator closed it at the broth ...
    log.close("fil", 20, 10)                       # ... and, healed for good, again at the healing
    ctx = _Ctx(log, [("fil", 10)])
    head_tail_company(log, ctx, "fil", 62)
    assert _ticks(ctx) == {10, 20}
    ctx = _Ctx(log, [("fil", 13)])                 # shown a closer that is not the latest: the head and the latest tail come
    head_tail_company(log, ctx, "fil", 62)
    assert _ticks(ctx) == {10, 13, 20}


def test_a_moment_that_closes_two_matters_brings_both_heads() -> None:
    log = _stream()
    log.close("fil", 20, 10)
    log.close("fil", 20, 12)
    ctx = _Ctx(log, [("fil", 20)])
    head_tail_company(log, ctx, "fil", 62)
    assert _ticks(ctx) == {10, 12, 20}


def test_the_loop_a_second_hurt_is_a_new_head_and_is_open() -> None:
    """Hurt, healed, hurt again: the healing closes the first hurt; the second hurt opens its own matter and has no tail yet."""
    log = _stream()
    log.close("fil", 20, 10)
    ctx = _Ctx(log, [("fil", 30)])                 # the second hurt, shown: an open head brings nothing
    assert head_tail_company(log, ctx, "fil", 62) == 0
    ctx = _Ctx(log, [("fil", 10)])                 # the first hurt, shown: its tail is the healing, not the second hurt
    head_tail_company(log, ctx, "fil", 62)
    assert _ticks(ctx) == {10, 20}


def test_a_pair_is_added_whole_or_not_at_all() -> None:
    log = _stream()
    log.close("fil", 13, 10)
    log.close("fil", 20, 10)
    ctx = _Ctx(log, [("fil", 13)])                 # shown the middle closer: the head AND the latest tail are both absent, two to add
    assert head_tail_company(log, ctx, "fil", 62, cap=1) == 0 and _ticks(ctx) == {13}        # one slot does not take half a pair
    assert head_tail_company(log, ctx, "fil", 62, cap=2) == 2 and _ticks(ctx) == {10, 13, 20}


def test_the_rule_adds_at_most_the_cap_whatever_the_closings() -> None:
    log = _stream()
    for k in range(21, 61):
        log.close("fil", k, 10)                  # forty closings of the fall: only the latest is its tail
    ctx = _Ctx(log, [("fil", 10)])
    assert head_tail_company(log, ctx, "fil", 62) == 1 and _ticks(ctx) == {10, 60}
    many = _Ctx(log, [("fil", i) for i in range(21, 41)])            # twenty closers shown: the head and the latest tail, once
    assert head_tail_company(log, many, "fil", 62) == 2 <= PAIR_CAP


def test_a_closing_reaches_only_what_has_happened() -> None:
    log = _stream()
    log.close("fil", 61, 10)
    ctx = _Ctx(log, [("fil", 10)])
    assert head_tail_company(log, ctx, "fil", 61) == 0     # now is 61: the closer at 61 is not yet part of the past
    log.close("fil", 20, 20)                               # a closing of itself or a later one is ignored
    assert not log._closes.get(("fil", 20))


def test_closings_are_read_from_records_and_survive_extend_log() -> None:
    canon = Maya(world_time=70, entities={"earth": Entity("earth", ["the world"], 0, "genesis"), "fil": Entity("fil", ["Fil"], 0, "genesis")})

    def mem(i, t, text):
        return Memory(f"m{i}", "fil", frozenset(), text, "PRIMARY", 4, "earth", t, f"p{i}", None, "useful")
    mems = [mem(1, 10, "I fell."), mem(2, 20, "I healed.")]
    recs = [Record("r1", "closes", "fil", "10", 20, "p2")]
    log, _ = build_log(canon, mems, recs)
    assert log._closes[("fil", 20)] == [10] and log._closed_by[("fil", 10)] == [20]
    more = extend_log(log, canon, log.kosha, [mem(3, 30, "I fell again."), mem(4, 40, "Healed again.")], [Record("r2", "closes", "fil", "30", 40, "p4")])
    assert more is not None and more._closes[("fil", 40)] == [30]
    assert ("fil", 40) not in log._closes and ("fil", 30) not in log._closed_by           # the older version is unchanged


def test_a_retired_follows_row_is_never_read() -> None:
    canon = Maya(world_time=70, entities={"earth": Entity("earth", ["the world"], 0, "genesis"), "fil": Entity("fil", ["Fil"], 0, "genesis")})

    def mem(i, t, text):
        return Memory(f"m{i}", "fil", frozenset(), text, "PRIMARY", 4, "earth", t, f"p{i}", None, "useful")
    log, _ = build_log(canon, [mem(1, 10, "I fell."), mem(2, 20, "I healed.")], [Record("r1", "follows", "fil", "10", 20, "p2")])
    assert not log._closes and not log._closed_by


def test_closes_records_keep_only_real_earlier_ticks_of_that_perceiver() -> None:
    log = _stream()
    rs = closes_records([10, 11, 999, "20", 61, "x"], "fil", 61, "pz", log)
    assert sorted(r.value for r in rs) == ["10", "11", "20"] and all(r.kind == "closes" and r.entity_id == "fil" and r.world_time == 61 for r in rs)
    assert closes_records([10], "someone_else", 61, "pz", log) == []             # not their own tuple
    assert closes_records(None, "fil", 61, "pz", log) == [] and closes_records("10", "fil", 61, "pz", log) == []
    many = closes_records([10, 11, 20, 21, 22, 23], "fil", 61, "pz", log)
    assert len(many) == 3 and sorted(int(r.value) for r in many) == [21, 22, 23]    # a short list: the newest three


def test_the_schema_offers_closes_with_unlike_examples_and_the_second_model_must_answer_it() -> None:
    item = SCHEMA["properties"]["perceiver_tuples"]["items"]
    assert item["properties"]["closes"]["items"] == {"type": "integer"} and "closes" not in item["required"]
    assert "follows" not in item["properties"]
    desc = item["properties"]["closes"]["description"]
    assert "INJURY" in desc and "PROMISE" in desc and "BUCKET OF WATER" in desc               # three matters unlike one another
    faces = _SCHEMA_FACES["properties"]["perceiver_tuples"]["items"]
    assert "closes" in faces["required"] and "follows" not in faces["required"]
