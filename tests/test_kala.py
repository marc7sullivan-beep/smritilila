# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Kāla: people carry time, and every jump is consented to (spec v5 §19-§23)."""
from __future__ import annotations

from fastapi import BackgroundTasks
from smritilila import schemas
from smritilila import pravrtti
from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.kala import Kala, changes_in, mirror, time_line
from smritilila.smrti.kosha import Kosha
from smritilila import adhisthana
from smritilila.routes import spanda

CLOCK = {"epoch": "1928-04-27T00:00", "start": 25 * 60}  # Saturday 28 April 1928, 1:00 am
H = 60
ONE = 25 * H  # 1:00 am Saturday


def _span(eid, start, end, origin, dest, tick):
    return Record(f"s{eid}{tick}", "time", eid, str(end), tick, f"p{tick}", f"{start}|{origin or ''}|{dest or ''}")


def _ids():
    n = iter(range(10**6))
    return lambda: f"n{next(n)}"


def test_a_world_without_a_clock_keeps_none() -> None:
    assert Kala.of(None, []) is None and Kala.of({}, []) is None


def test_people_carry_time_and_the_time_is_said_as_a_person_says_it() -> None:
    k = Kala.of(CLOCK, [_span("racer", ONE, ONE + 6 * H + 30, "room", "room", 5)])
    assert k.clock("racer") == ONE + 6 * H + 30 and k.clock("laggard") is None
    assert k.now("laggard") == ONE and k.now("racer") == ONE + 6 * H + 30
    assert k.now("laggard", wait_until=ONE + 3 * H) == ONE + 3 * H      # agreed to wait for a locked scene
    assert k.say(ONE) == "Saturday, 28 April 1928, 1:00 at night"
    assert k.say(24 * H) == "Saturday, 28 April 1928, midnight"
    assert k.say(ONE + 3 * H + 50) == "Saturday, 28 April 1928, 4:50 at night"          # before 5 is night
    assert k.say(ONE + 4 * H) == "Saturday, 28 April 1928, 5:00 in the morning"          # from 5, morning (Marc)
    # a world without our calendar counts its days, and never dates them
    d = Kala.of({"calendar": "days", "start": 23 * H}, [])
    assert d.say(23 * H) == "the first day, 11:00 at night" and d.say(29 * H) == "the second day, 5:00 in the morning"
    assert d.say(10 * 24 * H + 12 * H) == "day 11, noon" and d.now("anyone") == 23 * H
    assert "the time between has passed with them waiting" in time_line(k, ONE + 3 * H, ONE)
    # a world's turn is told the span it covers, and to fill it (Marc, 2026-09-30): not a ceiling
    turn = time_line(k, ONE, ONE, max_elapsed=20)
    assert "THIS TURN COVERS THE NEXT 20 minutes -- give `elapsed_minutes` 20" in turn
    assert "how long this pulse takes" not in turn and "THE LONGEST" not in turn
    assert "THE NEXT 1 day, 3 hours, 5 minutes" in time_line(k, ONE, ONE, max_elapsed=24 * H + 3 * H + 5)


def test_a_lived_moment_locks_a_scene_until_they_left_and_passing_through_does_not() -> None:
    """The racer slept in the room 1:00-7:30 and left; he crossed the square
    at 7:40. A laggard at 3:00 finds the room closed until 7:30 and the
    square open; after 7:30 the room is open to him."""
    k = Kala.of(CLOCK, [
        _span("racer", ONE, ONE + 6 * H + 30, "room", "room", 5),                  # asleep, 1:00-7:30
        _span("racer", ONE + 6 * H + 30, ONE + 6 * H + 40, "room", "square", 6),   # walks out, into the square at 7:40
        _span("racer", ONE + 6 * H + 40, ONE + 6 * H + 45, "square", "street", 7),  # through, and gone
    ])
    three = ONE + 2 * H
    assert k.where_at("racer", three) == "room" and k.where_at("racer", ONE + 6 * H + 42) == "square"
    assert k.locked_until("room", three, "laggard") == ONE + 6 * H + 30
    assert k.locked_until("square", three, "laggard") is None
    assert k.locked_until("room", ONE + 7 * H, "laggard") is None
    assert k.locked_until("room", three, "racer") is None                          # never locked to oneself
    # present_at is for someone the containment puts in a scene now: the racer is in the street since 7:40
    assert k.present_at("racer", "street", ONE + 6 * H + 50) and not k.present_at("racer", "street", three)
    # the mirror: a pulse from 2:00 to 4:00 in the room passes through his sleep
    assert k.lived_in("room", ONE + H, ONE + 3 * H, "laggard") == [(ONE + H, "racer")]
    assert k.lived_in("square", ONE + H, ONE + 3 * H, "laggard") == []


def test_the_mirror_shows_what_was_lived_here_after_your_now_not_who_is_here_with_you() -> None:
    """v5 §21, the mirror (2026-09-30): the racer slept in the room 1:00-7:30 and
    is gone; a laggard's pulse there 2:00-4:00 holds his sleep -- a span
    running through the laggard's now, by someone no longer here. The groom
    came in at 3:00: after the laggard's now, shown though he is here now.
    Wen sat in the room 1:00-5:00 and is here now: a companion whose clock
    ran ahead is not the future but company, already shown as present --
    Stonemarch's willow, where Calmert sat with the party at the very minute
    the mirror said he "was here after your now", and a correct telling was
    told again (34 of 42 mirror reruns)."""
    k = Kala.of(CLOCK, [
        _span("racer", ONE, ONE + 6 * H + 30, "room", "room", 5),
        _span("racer", ONE + 6 * H + 30, ONE + 6 * H + 40, "room", "street", 6),
        _span("groom", ONE + 2 * H, ONE + 3 * H, "room", "room", 7),
        _span("wen", ONE, ONE + 4 * H, "room", "room", 8),
    ])
    here_now = {"groom", "wen"}
    assert mirror(k, ["room"], ONE + H, ONE + 3 * H, "laggard", here_now) == [(ONE + H, "racer"), (ONE + 2 * H, "groom")]
    assert mirror(k, ["room", None], ONE + H, ONE + 3 * H, "laggard", set()) == [
        (ONE + H, "racer"), (ONE + H, "wen"), (ONE + 2 * H, "groom")]  # no one here: all of it
    assert mirror(k, ["street"], ONE + H, ONE + 3 * H, "laggard", here_now) == []
    # out of reach (v4.31): the racer a player whose time is his own -- and what he holds, gone with him
    k = Kala.of(CLOCK, [_span("racer", ONE, ONE + 6 * H + 30, "room", "room", 5),
                        _span("loaf", ONE, ONE + 6 * H + 30, "room", "room", 6)])
    held = {"loaf": ["racer", "street"]}
    assert mirror(k, ["room"], ONE + H, ONE + 3 * H, "laggard", set()) == [(ONE + H, "loaf"), (ONE + H, "racer")]
    assert mirror(k, ["room"], ONE + H, ONE + 3 * H, "laggard", set(), out_of_reach=frozenset({"racer"}),
                  holders=lambda who: held.get(who, [])) == []


def test_apart_is_whoever_this_moment_is_not_the_present_of() -> None:
    """Each perceiver's time is their own (Marc, 2026-09-30). Apart from a
    moment: anyone who has lived past it -- it is their past, out of reach --
    and a player behind it, not there yet, unless the world carries them
    (away, allowing it). An NPC behind is not apart: NPCs follow."""
    from smritilila.smrti.kala import apart
    k = Kala.of(CLOCK, [_span("racer", ONE, ONE + 3 * H, "yard", "yard", 1), _span("laggard", ONE, ONE + H, "yard", "yard", 2),
                        _span("cook", ONE, ONE + 2 * H, "yard", "yard", 3), _span("boy", ONE, ONE + H, "yard", "yard", 4)])
    now = ONE + 2 * H
    who = ["racer", "laggard", "cook", "boy", "nobody"]
    assert apart(k, who, now, players={"racer", "laggard"}) == {"racer", "laggard"}
    assert apart(k, who, now, players={"racer", "laggard"}, carried={"laggard"}) == {"racer"}


def test_a_world_turn_writes_nothing_for_whoever_it_is_not_the_present_of(monkeypatch) -> None:
    """The gap v4.27 left (Stonemarch, 2026-09-29, t537: Dunstan's turn, 58
    minutes behind, moved Jakal and Spugnoir out of Corl's cabin; 33 NPCs
    voiced ahead; Jakal given a memory of a moment 17 hours before he came).
    What was lived further along in a scene was hidden from a PLAYER's pulse
    only, and only from being shown. Now, in any pulse, whoever this moment
    is not the present of is not moved, voiced, heard, made to become,
    listed present, or given a witnessed copy: the groom here has lived to
    5:00, Wentworth (a player) to 6:00, and the cook's turn is at 1:00."""
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda *a, **k: None)
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "stable": Entity("stable", ["the stable"], 0, "genesis", avarana="scene"),
        "cook": Entity("cook", ["the cook"], 0, "genesis"),
        "groom": Entity("groom", ["the groom"], 0, "genesis"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis")})
    mems = [Memory("m1", "cook", frozenset(), "I scrubbed pots in the yard.", "PRIMARY", 4, "yard", 2, "p2", None, None, ONE, ONE),
            Memory("m2", "groom", frozenset(), "I curried the horses in the yard.", "PRIMARY", 4, "yard", 3, "p3", None, None, ONE, ONE + 4 * H),
            Memory("m3", "wen", frozenset(), "I waited in the yard.", "PRIMARY", 4, "yard", 4, "p4", None, None, ONE, ONE + 5 * H)]
    recs = [Record("r1", "kosha", "cook", "yard", 0, "p0", None), Record("r2", "kosha", "groom", "yard", 0, "p0", None),
            Record("r3", "kosha", "wen", "yard", 0, "p0", None), Record("r4", "kosha", "stable", "yard", 0, "p0", None)]

    class C:
        def complete_with_tool(self, system, user, **kw):
            return {"narration": "The cook scrubbed; the groom went to the stable.", "elapsed_minutes": 30,
                    "perceiver_tuples": [{"perceiver": "cook", "text": "I scrubbed the big pot.", "salience": "ordinary"},
                                         {"perceiver": "groom", "text": "I went out to the stable.", "salience": "ordinary"}],
                    "scene_tuples": [{"scene": "yard", "text": "The cook scrubbed; the groom said good morning.",
                                      "present": ["cook", "groom", "wen"], "said": "Good morning.", "said_by": "groom"}],
                    "moves": [{"entity": "groom", "into": "stable"}],
                    "becomings": [{"entity": "groom", "key": "tired", "clause": "Tired from the early horses."}],
                    "suggested_next_action": ""}
    r = run_smrti_pulse(canon, "cook", None, 6, "p6", mems, C(), None, records=recs, view={"world_data": {"clock": CLOCK}},
                        players=frozenset({"wen"}), playable=frozenset({"wen"}), max_elapsed=30)
    assert any(m.entity_id == "cook" for m in r.memories_written)                          # the turn's own, written
    assert not any(m.entity_id in ("groom", "wen") for m in r.memories_written)            # not voiced; no witnessed copy
    assert not any(x.entity_id == "groom" for x in r.records_written if x.kind in ("kosha", "clause"))  # not moved, not become
    assert not any(x.kind == "said" and x.note == "groom" for x in r.records_written)       # not heard
    row = next(m for m in r.memories_written if m.entity_id == "yard")
    assert "groom" not in row.subject_ids and "wen" not in row.subject_ids                  # not listed present


def test_a_change_that_still_shows_locks_and_what_was_carried_away_does_not() -> None:
    """At 8:00 the racer took the revolver from the drawer, left a glove,
    and carried away a ticket he was given. At 3:00 the study is closed
    until 8 -- for the revolver and the glove, not the ticket."""
    canon = Maya(world_time=10, entities={
        "study": Entity("study", ["the study"], 0, "genesis", avarana="scene"),
        "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
        "racer": Entity("racer", ["the racer"], 0, "genesis"),
        "drawer": Entity("drawer", ["the drawer"], 0, "genesis"),
        "revolver": Entity("revolver", ["the revolver"], 0, "genesis"),
        "glove": Entity("glove", ["a glove"], 6, "smrti"), "ticket": Entity("ticket", ["a ticket"], 6, "smrti")})
    recs = [Record("k1", "kosha", "drawer", "study", 0, "p0", None), Record("k2", "kosha", "revolver", "drawer", 0, "p0", None),
            Record("k3", "kosha", "racer", "study", 5, "p5", None),
            Record("k4", "kosha", "revolver", "racer", 6, "p6", None), Record("k5", "kosha", "glove", "study", 6, "p6", None),
            Record("k6", "kosha", "ticket", "racer", 6, "p6", None), Record("k7", "kosha", "racer", "hall", 7, "p7", None),
            _span("racer", ONE + 6 * H, ONE + 7 * H, "hall", "study", 6), _span("racer", ONE + 7 * H, ONE + 7 * H + 5, "study", "hall", 7)]
    mems = [Memory("m1", "racer", frozenset(), "I stood in the study.", "PRIMARY", 4, "study", 5, "p5", None)]
    kosha = Kosha(canon, mems, recs)
    k = Kala.of(CLOCK, recs)
    three = ONE + 2 * H
    ch = changes_in(kosha, k, "study", three, 11, {"racer"})
    assert sorted(e for _m, e in ch) == ["glove", "revolver"]                       # the ticket went with him: no lock
    assert k.locked_until("study", three, "laggard", ch) == ONE + 7 * H
    assert changes_in(kosha, k, "study", ONE + 8 * H, 11, {"racer"}) == []          # after 8: nothing is later


def test_a_repair_to_the_tree_is_not_a_lived_change_and_locks_nothing() -> None:
    """Marc, 2026-10-01 ("a repair isn't a lived change"): the Crossing's far bank
    took the farmhouse out of the west road, where old nesting had written
    "beside" as "in". Read as lived, the west road was locked to anyone behind
    the repair's minute. A god's repair (a seed, a gravestone) locks nothing;
    the same move lived in a pulse still does."""
    canon = Maya(world_time=10, entities={
        "road": Entity("road", ["the west road"], 0, "genesis", avarana="road"),
        "farm": Entity("farm", ["the farmhouse"], 0, "smrti", avarana="building"),
        "bank": Entity("bank", ["the far bank"], 9, "genesis"),
        "cart": Entity("cart", ["a cart"], 0, "genesis")})
    recs = [Record("k1", "kosha", "farm", "road", 0, "p0", "declared"), Record("k2", "kosha", "cart", "road", 0, "p0", "declared"),
            Record("k3", "kosha", "bank", None, 6, "seed4", "seed4"), Record("k4", "kosha", "farm", "bank", 6, "seed4", "seed4"),
            Record("k5", "kosha", "cart", "bank", 6, "p6", "moved"),
            _span("racer", ONE + 5 * H, ONE + 6 * H, "road", "road", 6)]
    kosha = Kosha(canon, [], recs)
    k = Kala.of(CLOCK, recs)
    ch = changes_in(kosha, k, "road", ONE, 11, set())
    assert [e for _m, e in ch] == ["cart"]                                          # the cart was taken; the farmhouse was not
    assert not changes_in(Kosha(canon, [], [r for r in recs if r.id != "k5"]), k, "road", ONE, 11, set())


def test_a_pulse_shows_the_time_hides_what_is_later_and_asks_before_a_long_jump(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "dan": Entity("dan", ["Danforth"], 0, "genesis"),
        "lantern": Entity("lantern", ["a lantern"], 0, "genesis")})
    mems = [Memory("m1", "wen", frozenset(), "I stood in the yard.", "PRIMARY", 4, "yard", 3, "p3", None),
            Memory("m2", "yard", frozenset(), "At dawn a man came into the yard.", "AMBIENT", 5, "yard", 4, "p4", None),
            Memory("m3", "dan", frozenset(), "I came into the yard at dawn.", "PRIMARY", 4, "yard", 4, "p4", None)]
    recs = [Record("r1", "kosha", "wen", "yard", 0, "p0", None), Record("r2", "kosha", "dan", "yard", 4, "p4", None),
            Record("r0", "kosha", "lantern", "yard", 0, "p0", None),  # a thing in the yard: it does not carry time
            _span("wen", ONE, ONE, "yard", "yard", 3), _span("dan", ONE + 5 * H, ONE + 5 * H, "street", "yard", 4)]
    seen = []

    def client(elapsed):
        class C:
            def complete_with_tool(self, system, user, **kw):
                seen.append(user)
                return {"narration": "You wait.", "elapsed_minutes": elapsed,
                        "perceiver_tuples": [{"perceiver": "wen", "text": "I waited in the yard a good while.", "salience": "ordinary"}],
                        "scene_tuples": [{"scene": "yard", "text": "A man waited in the yard.", "present": ["wen"]}],
                        "suggested_next_action": ""}
        return C()
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda *a, **k: None)
    view = {"world_data": {"clock": CLOCK}}
    r = run_smrti_pulse(canon, "wen", "waits", 6, "p6", mems, client(5), None, records=recs, view=view,
                        players=frozenset({"wen", "dan"}), playable=frozenset({"wen", "dan"}))
    assert "THE TIME HERE: Saturday, 28 April 1928, 1:00 at night" in seen[0]
    assert "At dawn a man came into the yard" not in seen[0] and "I came into the yard at dawn" not in seen[0]  # later than 1:00
    assert r.pending is None
    assert not [x for x in r.records_written if x.kind == "time"]                       # the minutes are on the memories
    assert {(m.began, m.ended) for m in r.memories_written} == {(ONE, ONE + 5)}
    after = Kala.of(CLOCK, recs, mems + r.memories_written)
    assert after.clock("wen") == ONE + 5 and after.clock("lantern") is None and after.clock("yard") is None
    assert after.scene_clock["yard"] == ONE + 5
    # a 45-minute wait asks first; consented, it does not
    r = run_smrti_pulse(canon, "wen", "waits", 6, "p6", mems, client(45), None, records=recs, view=view,
                        players=frozenset({"wen", "dan"}), playable=frozenset({"wen", "dan"}))
    assert r.pending["kind"] == "long" and r.pending["prompt"] == "You are about to jump ahead 45 min (to Saturday, 28 April 1928, 1:45 at night). Proceed?"
    r = run_smrti_pulse(canon, "wen", "waits", 6, "p6", mems, client(45), None, records=recs, view=view,
                        players=frozenset({"wen", "dan"}), playable=frozenset({"wen", "dan"}), confirmed=True)
    assert r.pending is None
    # an autonomous turn covers the span it is given, and never asks: past it or short of it, it fills it
    # (Stonemarch: world turns used a median 23% of their gap, and their NPCs stayed a day behind)
    r = run_smrti_pulse(canon, "wen", None, 6, "p6", mems, client(300), None, records=recs, view=view,
                        players=frozenset({"dan"}), playable=frozenset({"wen", "dan"}), max_elapsed=20)
    assert r.pending is None and "THIS TURN COVERS THE NEXT 20 minutes" in seen[-1]
    assert {m.ended for m in r.memories_written} == {ONE + 20}
    r = run_smrti_pulse(canon, "wen", None, 6, "p6", mems, client(5), None, records=recs, view=view,
                        players=frozenset({"dan"}), playable=frozenset({"wen", "dan"}), max_elapsed=20)
    assert {m.ended for m in r.memories_written} == {ONE + 20}
    # a world without a clock: no line, no time records, no prompt
    seen.clear()
    r = run_smrti_pulse(canon, "wen", "waits", 6, "p6", mems, client(600), None, records=recs,
                        players=frozenset({"wen", "dan"}), playable=frozenset({"wen", "dan"}))
    assert "THE TIME HERE" not in seen[0] and r.pending is None and {m.ended for m in r.memories_written} == {None}


def test_rates_compose_up_the_branches_and_time_is_never_stopped() -> None:
    """v5 §19 (Marc, 2026-10-01): time is kept through the branches; a branch's rate is
    relative to what holds it, multiplied up the chain."""
    canon = Maya(world_time=6, entities={
        "earth": Entity("earth", ["the world at large"], 0, "genesis"), "hill": Entity("hill", ["the fey hill"], 0, "genesis"),
        "hall": Entity("hall", ["the hall under the hill"], 0, "genesis", avarana="scene"),
        "road": Entity("road", ["the road"], 0, "genesis", avarana="scene")})
    recs = [Record("k1", "kosha", "hill", "earth", 0, "p0", None), Record("k2", "kosha", "hall", "hill", 0, "p0", None),
            Record("k3", "kosha", "road", "earth", 0, "p0", None)]
    k = Kosha(canon, [], recs)
    assert k.rate("hall", 7) == 1.0 and k.rate("road", 7) == 1.0                      # nothing said: the world's own
    k.add(Record("t1", "prakriti", "hill", '{"rate": 4}', 1, "p1", "time"))
    k.add(Record("t2", "prakriti", "hall", '{"rate": 0.5}', 1, "p1", "time"))
    assert k.rate("hall", 7) == 2.0 and k.rate("hill", 7) == 4.0 and k.rate("road", 7) == 1.0  # composed: 4 x 0.5
    k.add(Record("t3", "prakriti", "hall", '{"rate": 0}', 2, "p2", "time"))           # stopped: says nothing
    assert k.rate("hall", 7) == 4.0
    k.add(Record("t4", "prakriti", "hill", None, 3, "p3", "time"))                    # withdrawn
    assert k.rate("hall", 7) == 1.0


def test_a_pulse_in_a_branch_lives_its_own_time_and_the_world_moves_by_the_rate(monkeypatch) -> None:
    """v5 §19: the Narrator gives the time lived here, as long as it says (no cap, v5.31); the world moves
    as far as the rate says; the consent names both; a world's turn is told its span as
    lived here. Rate 1 everywhere is today's clock to the minute."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "hollow": Entity("hollow", ["the fey hollow"], 0, "genesis"),
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis")})
    mems = [Memory("m1", "wen", frozenset(), "I stood in the yard.", "PRIMARY", 4, "yard", 3, "p3", None)]
    base = [Record("r0", "kosha", "yard", "hollow", 0, "p0", None), Record("r1", "kosha", "wen", "yard", 0, "p0", None),
            _span("wen", ONE, ONE, "yard", "yard", 3)]
    seen = []

    def client(elapsed):
        class C:
            def complete_with_tool(self, system, user, **kw):
                seen.append(user)
                return {"narration": "You wait.", "elapsed_minutes": elapsed,
                        "perceiver_tuples": [{"perceiver": "wen", "text": "I waited in the yard a good while.", "salience": "ordinary"}],
                        "scene_tuples": [{"scene": "yard", "text": "A man waited in the yard.", "present": ["wen"]}],
                        "suggested_next_action": ""}
        return C()
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda *a, **k: None)
    view = {"world_data": {"clock": CLOCK}}
    kw = dict(view=view, players=frozenset({"wen"}), playable=frozenset({"wen"}))

    def run(elapsed, rate, **more):
        recs = base + ([Record("t1", "prakriti", "hollow", f'{{"rate": {rate}}}', 0, "p0", "time")] if rate is not None else [])
        return run_smrti_pulse(canon, "wen", more.pop("action", "waits"), 6, "p6", mems, client(elapsed), None, records=recs, **kw, **more)

    # rate 1 said aloud reads exactly as no rate at all
    r0 = run(5, None)
    r1 = run(5, 1)
    assert seen[-1] == seen[-2] and {m.ended for m in r1.memories_written} == {m.ended for m in r0.memories_written} == {ONE + 5}
    # twice the world's pace: five minutes lived here, ten outside
    r = run(5, 2)
    assert "TIME RUNS AT ITS OWN PACE HERE: an hour lived here is 2 hours in the world beyond" in seen[-1]
    assert {(m.began, m.ended) for m in r.memories_written} == {(ONE, ONE + 10)} and r.pending is None
    # the consent names both
    r = run(45, 2)
    assert r.pending["prompt"] == ("You are about to spend 45 min here -- 1 h 30 min in the world outside "
                                   "(to Saturday, 28 April 1928, 2:30 at night). Proceed?")
    # no cap (v5.31): 3000 minutes lived here, 6000 outside, as the Narrator said
    r = run(3000, 2, confirmed=True)
    assert {m.ended for m in r.memories_written} == {ONE + 2 * 3000}
    # a world's turn is told its span as lived here, and covers the world's span
    r = run(5, 2, action=None, max_elapsed=120)
    assert "THIS TURN COVERS THE NEXT 1 hour --" in seen[-1] and {m.ended for m in r.memories_written} == {ONE + 120}


def _gate_world():
    import time as _time
    from smritilila import main
    canon = Maya(world_time=8, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "station": Entity("station", ["the station"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "dan": Entity("dan", ["Danforth"], 0, "genesis"),
        "barman": Entity("barman", ["the barman"], 0, "genesis"),
        "clerk": Entity("clerk", ["the clerk"], 0, "genesis"),
        "porter": Entity("porter", ["the porter"], 0, "genesis")})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="wen", entry_point_location_ids=["yard"],
                              playable_entity_ids=["wen", "dan"])
    world.world_data = {"clock": CLOCK}
    world.records = [Record("r1", "kosha", "wen", "yard", 0, "p0", None), Record("r2", "kosha", "barman", "yard", 0, "p0", None),
                     Record("r3", "kosha", "clerk", "yard", 0, "p0", None), Record("r4", "kosha", "dan", "station", 0, "p0", None),
                     Record("r5", "kosha", "porter", "station", 0, "p0", None),
                     _span("wen", ONE, ONE + H, "yard", "yard", 3), _span("dan", ONE, ONE + 5 * H, "station", "station", 4),
                     _span("barman", ONE, ONE + 30, "yard", "yard", 5), _span("clerk", ONE, ONE + 90, "yard", "yard", 6)]
    for p in ("wen", "dan"):
        world.last_seen[p] = _time.monotonic()
    return main, world


def _hall_world():
    """Danforth lived the hall 1:00-3:00 (the barman there with him), then
    went to the station until 9:00. Wentworth lived the hall 7:00-8:00."""
    import time as _time
    from smritilila import main
    canon = Maya(world_time=6, entities={
        "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
        "station": Entity("station", ["the station"], 0, "genesis", avarana="scene"),
        "stable": Entity("stable", ["the stable"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "dan": Entity("dan", ["Danforth"], 0, "genesis"),
        "barman": Entity("barman", ["the barman"], 0, "genesis"),
        "groom": Entity("groom", ["the groom"], 0, "genesis")})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="wen", entry_point_location_ids=["hall"],
                              playable_entity_ids=["wen", "dan"])
    world.world_data = {"clock": CLOCK}
    world.records = [Record(f"k{i}", "kosha", e, into, 0, "p0", None)
                     for i, (e, into) in enumerate([("groom", "hall"), ("barman", "hall"), ("dan", "station"), ("wen", "hall")])]
    world.recent_memories = [
        _m(0, "barman", "hall", 0, None, None),
        _m(1, "groom", "hall", 1, ONE, ONE + 30),
        _m(2, "dan", "hall", 2, ONE, ONE + 2 * H), _m(3, "hall", "hall", 2, ONE, ONE + 2 * H, "AMBIENT", {"dan", "barman"}),
        _m(4, "dan", "station", 3, ONE + 2 * H, ONE + 2 * H + 10),
        _m(5, "dan", "station", 4, ONE + 2 * H + 10, ONE + 8 * H),
        _m(6, "wen", "hall", 5, ONE + 6 * H, ONE + 7 * H)]
    for p in ("wen", "dan"):
        world.last_seen[p] = _time.monotonic()
    return main, world


def test_the_worlds_turns_follow_the_players_and_never_narrate_over_what_was_lived() -> None:
    """The groom, last in the hall at 1:30, begins when Danforth is gone (3:00)
    and must end before Wentworth comes (7:00); the barman, carried to 3:00
    by the hall's row, the same. The stable has no time of its own: it
    begins the hour before its reference, not at 1:00. The oldest wake first."""
    main, world = _hall_world()
    three, seven = ONE + 2 * H, ONE + 6 * H
    assert pravrtti._kala_gate(world, ["groom", "barman", "stable"]) == (
        ["barman", "groom"], {"groom": (three, 4 * H), "barman": (three, 4 * H), "stable": (seven, H)})
    assert pravrtti._kala_gate(world, ["stable"]) == (["stable"], {"stable": (seven, H)})
    # a world without a clock: the pool as it was
    world.world_data = {}
    assert pravrtti._kala_gate(world, ["groom"]) == (["groom"], {})


def test_a_drawn_turn_keeps_its_cap_when_another_pulse_gates_before_it_starts(monkeypatch) -> None:
    """The draw-cap race (found reading the tree, 2026-09-30): the gate wrote
    each drawn turn's cap and begin into dicts the whole world shared, and the
    turn read them only once its scene was free. Another player's pulse
    committing meanwhile ran its own gate, which replaced the dicts: the turn
    ran uncapped, from its own clock. Never seen in 620 logged turns, but
    nothing prevented it. A drawn turn carries what its gate gave it."""
    import asyncio
    from fastapi import BackgroundTasks
    from smritilila.smrti.pulse import SmrtiPulseResult
    main, world = _hall_world()
    seen = {}

    def run(canon, actor, action, tick, pid, memories, client, embed, **kw):
        seen[actor] = (kw.get("wait_until"), kw.get("max_elapsed"))
        raise RuntimeError("the turn is not narrated here")  # the activation logs run_failed and returns

    monkeypatch.setattr(spanda, "run_smrti_pulse", run)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda text: None)})())
    monkeypatch.setattr(spanda, "_smrti_exploration_pool", lambda w: ["groom"])
    for hold in ("_fold_absorbed", "_hold_for_away", "_hold_for_rounds"):
        monkeypatch.setattr(spanda, hold, lambda w, pool: pool)
    roles = type("R", (), {"smrti_client": None})()
    tasks = BackgroundTasks()
    result = SmrtiPulseResult(narration="You wait.", memories_written=[], new_entities=[], suggested_next_action="",
                              context_tuples=0, bound_ids=[])
    spanda._after_smrti_commit(world, roles, tasks, result, "wen", "p9", 0.0)
    pravrtti._kala_gate(world, ["stable"])  # another player's pulse commits before the groom's scene is free
    asyncio.run(tasks())
    assert seen == {"groom": (ONE + 2 * H, 4 * H)}  # begins when Danforth is gone, ends before Wentworth comes


def test_a_world_turn_stops_at_an_offline_players_present_unless_the_world_may_act_for_them() -> None:
    """v5 §23 v4.31: every span stops at the first present it touches -- a world
    turn's too. Ravenmoor's parlour ran 1507 -> 1534 past a player standing in
    it at 1500, offline: the gate asked only online players' clocks, and the
    online one was elsewhere and later. A player whose time is their own holds
    the turns near them whether online or not; one who lets the world act for
    them while away (v5 §37) is carried."""
    import time as _time
    from smritilila import main
    canon = Maya(world_time=6, entities={
        "parlour": Entity("parlour", ["the parlour"], 0, "genesis", avarana="scene"),
        "study": Entity("study", ["the study"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "mara": Entity("mara", ["Mara"], 0, "genesis"),
        "aunt": Entity("aunt", ["the aunt"], 0, "genesis")})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="wen", entry_point_location_ids=["study"],
                              playable_entity_ids=["wen", "mara"])
    world.world_data = {"clock": CLOCK}
    world.records = [Record(f"k{i}", "kosha", e, into, 0, "p0", None)
                     for i, (e, into) in enumerate([("aunt", "parlour"), ("mara", "parlour"), ("wen", "study")])]
    world.recent_memories = [_m(1, "aunt", "parlour", 1, ONE - 10, ONE), _m(2, "mara", "parlour", 2, ONE - 10, ONE),
                             _m(3, "wen", "study", 3, ONE, ONE + 2 * H)]
    world.last_seen["wen"] = _time.monotonic()  # Wentworth online, in the study, later; Mara offline in the parlour
    assert pravrtti._kala_gate(world, ["aunt"])[0] == []              # the parlour waits at Mara's present
    world.autonomous.add("mara")                                  # she lets the world act for her while away: carried
    assert pravrtti._kala_gate(world, ["aunt"])[0] == ["aunt"]


def _held_bot_world():
    """Three far scenes: Wentworth (a person, playing) in the study at ONE+2H, a waiting bot kept online by its beat in the cellar at ONE, and an
    orc in the yard at ONE+H -- between them, far from both. Both are online."""
    import time as _time
    scenes = {s: Entity(s, [f"the {s}"], 0, "genesis", avarana="scene") for s in ("study", "cellar", "yard")}
    canon = Maya(world_time=6, entities={**scenes, "wen": Entity("wen", ["Wentworth"], 0, "genesis"), "bot": Entity("bot", ["Dunstan"], 0, "genesis"),
                                         "orc": Entity("orc", ["the orc"], 0, "genesis")})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="wen", entry_point_location_ids=["study"], playable_entity_ids=["wen", "bot"])
    world.world_data = {"clock": CLOCK}
    world.records = [Record(f"k{i}", "kosha", e, into, 0, "p0", None) for i, (e, into) in enumerate([("orc", "yard"), ("bot", "cellar"), ("wen", "study")])]
    world.recent_memories = [_m(1, "orc", "yard", 1, ONE, ONE + H), _m(2, "bot", "cellar", 2, ONE - 10, ONE), _m(3, "wen", "study", 3, ONE, ONE + 2 * H)]
    world.autonomous.add("bot")
    world.waiting.add("bot")                                      # as a stopped agent is left: waiting (player_tool stop)
    world.last_seen["wen"] = world.last_seen["bot"] = _time.monotonic()
    return world


def test_a_waiting_player_present_only_by_the_beat_does_not_hold_the_worlds_present() -> None:
    """Marc, 2026-10-28 ("Exclude waiting players"). Stonemarch's present stood at Dunstan's 42,399 for a day while Nagresh played at 89,515: Dunstan is a
    held bot, kept online by a beat that posts no action, and the gate waited on the earliest online clock -- so no chieftain, no troll and no priest
    had a turn (the 29 draws since the redraw were things, one the second model call each). A waiting player is the world's to carry; present by the beat
    alone they take no turn, and the world does not wait on their clock. Acting, they are a player like any other (an agent playing; a person back
    at a page they left at waiting): online outranks the setting, as movable() has it since 2026-09-30."""
    import time as _time
    from smritilila.adhisthana import PLAYER_IDLE_SECONDS
    world = _held_bot_world()
    assert world.online("bot") and world.acting("wen") and not world.acting("bot")
    assert pravrtti._kala_gate(world, ["orc"])[0] == ["orc"]          # the present is Wentworth's, the orc is behind it
    world.last_acted["bot"] = _time.monotonic()                      # the bot acts: its clock is a present like any other
    assert world.acting("bot") and pravrtti._kala_gate(world, ["orc"])[0] == []
    world.last_acted["bot"] = _time.monotonic() - PLAYER_IDLE_SECONDS - 1   # it acted, long ago: a beat since
    assert not world.acting("bot") and pravrtti._kala_gate(world, ["orc"])[0] == ["orc"]
    world.waiting.discard("bot")                                     # not waiting: online is online, acted or not
    assert world.acting("bot") and pravrtti._kala_gate(world, ["orc"])[0] == []


def test_the_world_still_waits_on_a_beat_when_no_one_is_acting() -> None:
    """The fallback is as it was: with no acting player the world's present is the earliest of the online, and with none online of all. An offline
    Oldric, earlier than the bot, and an orc between them tell the three apart."""
    world = _held_bot_world()
    world.canon.entities.update({"attic": Entity("attic", ["the attic"], 0, "genesis", avarana="scene"), "old": Entity("old", ["Oldric"], 0, "genesis")})
    world.playable_entity_ids.append("old")
    world.records.append(Record("kold", "kosha", "old", "attic", 0, "p0", None))
    world.recent_memories[0] = _m(1, "orc", "yard", 1, ONE, ONE + 30)           # the orc, at ONE+30
    world.recent_memories[1] = _m(2, "bot", "cellar", 2, ONE, ONE + H)          # the bot, at ONE+H
    world.recent_memories.append(_m(4, "old", "attic", 4, ONE - 20, ONE))       # Oldric, offline, at ONE (the world's first minute)
    del world.last_seen["wen"]                                       # Wentworth's page is closed: only the bot's beat is left
    assert not world.acting("bot") and world.online("bot")
    assert pravrtti._kala_gate(world, ["orc"])[0] == ["orc"]         # the earliest of the online is the bot's ONE+H: the orc is behind it
    del world.last_seen["bot"]
    assert pravrtti._kala_gate(world, ["orc"])[0] == []              # nobody online: the earliest of all, Oldric's ONE: the orc is past it


def _far_world(age_ahead: int):
    """A clocked world, tick 80 (a far tick), the age overdue and `age_ahead` minutes past the player's present; a monk in the abbey,
    far from the hall where the player stands and online."""
    import time as _time
    from smritilila.smrti.genesis import ITIHASA
    canon = Maya(world_time=80, entities={
        "keep": Entity("keep", ["the keep"], 0, "genesis", avarana="scene"),
        "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene", kosha="keep"),
        "cloister": Entity("cloister", ["the cloister"], 0, "genesis", avarana="scene"),
        "abbey": Entity("abbey", ["the abbey"], 0, "genesis", avarana="scene", kosha="cloister"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "monk": Entity("monk", ["the monk"], 0, "genesis"),
        "age": Entity("age", ["the world at large"], 0, ITIHASA)})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="wen", entry_point_location_ids=["hall"], playable_entity_ids=["wen"])
    world.world_data = {"clock": CLOCK, "pacing": {"arrivals_on_traffic": False}}
    world.bhuvana = "THE YEAR. 1928."
    present = ONE + 2 * H
    world.records = [Record("k1", "kosha", "wen", "hall", 0, "p0", None), Record("k2", "kosha", "monk", "abbey", 0, "p0", None),
                     Record("t1", "time", "age", str(present + age_ahead), 0, "p0", f"{ONE}||")]  # the age has reached `age_ahead` past the player
    world.recent_memories = [_m(0, "age", "nowhere", 0, None, None), _m(1, "monk", "abbey", 10, ONE, ONE + H), _m(2, "wen", "hall", 70, ONE, present)]
    world.last_seen["wen"] = _time.monotonic()
    return world


def test_an_age_the_gate_would_hold_does_not_take_the_far_tick_from_the_far_world() -> None:
    """Stonemarch, 2026-10-27: the priests of the Temple had no turn in 400 ticks. A far being is drawn only on a far tick, and on every far tick
    the overdue age took the draw ahead of them -- alone -- and the gate (NPCs follow and never lead) held it, the age being a season ahead of the
    players: nobody was drawn, and an age never drawn is overdue again at the next far tick (444 of 462 far ticks in Stonemarch's log, since tick 72).
    The pool asks the gate's own question before it gives the age the tick."""
    from smritilila.smrti.kala import AGE_LEAD
    world = _far_world(AGE_LEAD + H)
    assert pravrtti._age_held(world) is True
    pool = pravrtti._smrti_exploration_pool(world)
    assert pool != ["age"] and "monk" in pool                          # the far being has its tick
    drawn, turns = pravrtti._kala_gate(world, pool)
    assert "monk" in drawn and "monk" in turns and "age" not in drawn   # the gate lets it run (the far scenes are asked too, as ever); the held age waits, as before
    world.canon.world_time = 82                                          # not a far tick: far beings wait for theirs, held age or none
    assert "monk" not in pravrtti._smrti_exploration_pool(world)


def test_an_age_the_gate_would_let_run_still_takes_the_far_tick_first() -> None:
    """The age is not ahead of the players by a season: it takes the draw ahead of the far actors, alone, as v5 §38's age moves."""
    from smritilila.smrti.kala import AGE_LEAD
    world = _far_world(AGE_LEAD - H)
    assert pravrtti._age_held(world) is False
    assert pravrtti._smrti_exploration_pool(world) == ["age"]
    assert pravrtti._kala_gate(world, ["age"])[0] == ["age"]
    world.world_data = {}                                                # a world keeping no clock has nothing to hold: the age is never held
    assert pravrtti._age_held(world) is False


def _engaged_far_world(guard_held: bool):
    """`_far_world` with the age held, and the player's own scene given two of the world's own: the guard, who wrote a tuple in the player's last
    pulse (so the engaged scene's draw is his alone), and the porter, who did not. The guard stands at the player's present, in the scene the
    player lived to it, so the gate holds him (NPCs follow and never lead); with `guard_held` false the world keeps no clock, so there is no
    gate and no age, and the engaged scene draws him. All three are kept (a `useful` tuple), so nothing folds them away."""
    import dataclasses
    from smritilila.smrti.kala import AGE_LEAD
    world = _far_world(AGE_LEAD + H)
    world.world_data["pacing"]["arrivals"] = False                                    # no quiet scene is asked whether anyone came: the engaged scene has the draw
    if not guard_held:
        world.world_data.pop("clock")
        world.bhuvana = None
    present = ONE + 2 * H
    for eid in ("guard", "porter"):
        world.canon.entities[eid] = Entity(eid, [f"the {eid}"], 0, "genesis")
        world.records.append(Record(f"k_{eid}", "kosha", eid, "hall", 0, "p0", None))
    world.recent_memories += [_m(3, "guard", "hall", 70, ONE, present), _m(4, "porter", "hall", 20, ONE, ONE + H)]
    world.recent_memories = [dataclasses.replace(m, use="useful") if m.entity_id in ("monk", "guard", "porter") else m for m in world.recent_memories]
    return world


def test_an_engaged_scene_the_gate_holds_gives_the_draw_to_the_rest_of_the_world() -> None:
    """Marc, 2026-10-28 ("When we have that kind of block we should redraw the autonomous action instead of skipping"). Stonemarch's far windows at
    ticks 2536 and 2537 drew nobody: the player (Nagresh) stood in a new scene each pulse, so the cadence never gave the draw away; the engaged
    scene's only candidate (his companion) stood at the player's present, the gate held him (smrti.world_waits), and nobody else was asked.
    The engaged scene's draw that leaves nobody is an off-beat: the rest of the world is drawn, the player's scene left out."""
    world = _engaged_far_world(guard_held=True)
    assert pravrtti._smrti_exploration_pool(world) == ["guard"]                       # the engaged scene has the draw, alone
    assert pravrtti._kala_gate(world, ["guard"]) == ([], {})                          # and the gate holds him: before, nobody was drawn
    rest = pravrtti._smrti_exploration_pool(world, redraw=True)
    assert "monk" in rest and "guard" not in rest and "porter" not in rest            # the off-beat's pool: the world, the player's scene left out
    drawn, turns = spanda._draw(world)
    assert "monk" in drawn and "monk" in turns                                        # the far being has the far tick
    assert "guard" not in drawn and "porter" not in drawn and "age" not in drawn      # the player's scene keeps its beat; the held age waits as before
    world.canon.world_time = 82                                                       # not a far tick: no far being has a tick to take
    drawn, _ = spanda._draw(world)
    assert "monk" not in drawn and "guard" not in drawn and "porter" not in drawn     # drawn again, and nobody there: nobody


def test_an_engaged_scene_that_draws_someone_is_not_drawn_past() -> None:
    """The redraw is for a draw that left nobody: a guard the gate lets run has the turn, and the far world waits for its own."""
    world = _engaged_far_world(guard_held=False)
    drawn, _ = spanda._draw(world)
    assert drawn == ["guard"]


def test_a_redraw_without_an_engaged_scene_changes_nothing() -> None:
    world = _far_world(0)
    assert pravrtti._smrti_exploration_pool(world, redraw=True) == pravrtti._smrti_exploration_pool(world)


def test_a_redrawn_draw_is_witnessed(monkeypatch) -> None:
    """smrti.redrawn: what was blocked, how many were left, how many drawn -- so a draw rule is scored on the log."""
    seen = []
    monkeypatch.setattr(spanda, "log_event", lambda event, **kw: seen.append((event, kw)))
    world = _engaged_far_world(guard_held=True)
    spanda._draw(world)
    events = [kw for event, kw in seen if event == "smrti.redrawn"]
    assert len(events) == 1 and events[0]["tick"] == 80 and events[0]["blocked"] == 1
    assert events[0]["drawn"] >= 1 and events[0]["rest"] >= events[0]["drawn"]
    seen.clear()
    spanda._draw(_engaged_far_world(guard_held=False))
    assert not [event for event, _ in seen if event == "smrti.redrawn"]


def test_a_rounds_key_is_the_scene_and_what_a_boundary_joins_to_it() -> None:
    """Marc, 2026-09-28: "Dunstan and I are still locked in rounds. Shouldn't we be going separately now?" -- keyed
    by the container directly above (v4.31), the church and the smithy of Millhaven shared a round. A round is its
    scene, and the scenes a boundary thing (a door, a gate) joins to it: the courtyard and the hall through their
    door are one; the causeway, beside them in the same ruin, is its own; a scene in nothing is its own."""
    from smritilila import main
    from smritilila.memory.record import Record
    ents = {"ruin": Entity("ruin", ["the ruin"], 0, "genesis"),
            "courtyard": Entity("courtyard", ["the courtyard"], 0, "genesis", avarana="scene", kosha="ruin"),
            "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene", kosha="ruin"),
            "causeway": Entity("causeway", ["the causeway"], 0, "genesis", avarana="scene", kosha="ruin"),
            "door": Entity("door", ["the hall door"], 0, "genesis", kosha="courtyard"),
            "heath": Entity("heath", ["the heath"], 0, "genesis", avarana="scene")}
    k = Kosha(Maya(world_time=3, entities=ents), [], [Record("s1", "sandhi", "door", "hall", 1, "p1", "open")])
    from smritilila.sangama import _reach_key  # moved from main (2026-10-05)
    key = lambda s: _reach_key(k, s, 4)  # noqa: E731
    assert key("courtyard") == key("hall") and key("causeway") == "causeway" and key("causeway") != key("courtyard")
    assert key("heath") == "heath" and key(None) is None


def test_a_declined_jump_writes_nothing(monkeypatch) -> None:
    from fastapi.testclient import TestClient
    main, world = _gate_world()
    tags = []
    monkeypatch.setattr(spanda, "log_discarded", lambda **kw: tags.append(kw))
    client = TestClient(main.app)
    client.app.state.world = world
    monkeypatch.setattr(adhisthana, "get_world", lambda: _async(world))
    assert client.post("/pulses/confirm", json={"entity_id": "wen", "proceed": False}).status_code == 409
    world.held["wen"] = {"pulse_id": "p9", "kind": "long", "tick": 8}
    before = (len(world.records), len(world.recent_memories), world.canon.world_time)
    r = client.post("/pulses/confirm", json={"entity_id": "wen", "proceed": False})
    assert r.status_code == 200 and r.json()["narrations"] == {} and r.json()["confirm"] is None
    assert (len(world.records), len(world.recent_memories), world.canon.world_time) == before and "wen" not in world.held
    assert tags == [{"pulse_id": "p9", "reason": "declined", "detail": "long"}]


async def _async(x):
    return x


def test_going_into_a_scene_someone_has_lived_ahead_lets_the_player_in_with_them_out_of_reach(monkeypatch) -> None:
    """Danforth lived the hall 1:00-6:00. Wentworth, at 1:00, walks into it.
    v4.27 (Marc, 2026-09-27: "I couldn't change anything in the scene that's
    been future locked ... the player who locked shouldn't be seen either"):
    no question -- he goes in at his own time, and Danforth, already lived
    further along there, is out of his reach. A telling that touched him is
    told once more with him hidden; the Narrator is never told why."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "dan": Entity("dan", ["Danforth"], 0, "genesis")})
    mems = [Memory("m1", "wen", frozenset(), "I stood in the yard.", "PRIMARY", 4, "yard", 3, "p3", None),
            Memory("m2", "dan", frozenset(), "I sat in the hall all night.", "PRIMARY", 4, "hall", 4, "p4", None)]
    recs = [Record("r1", "kosha", "wen", "yard", 0, "p0", None), Record("r2", "kosha", "dan", "hall", 0, "p0", None),
            Record("r3", "kosha", "hall", "yard", 0, "p0", None),
            _span("wen", ONE, ONE, "yard", "yard", 3), _span("dan", ONE, ONE + 5 * H, "hall", "hall", 4)]
    calls, seen = [], []
    greet = {"on": True}

    class C:
        def complete_with_tool(self, system, user, **kw):
            calls.append(1)
            seen.append(user)
            tuples = [{"perceiver": "wen", "text": "I went into the hall.", "salience": "ordinary"}]
            if greet["on"]:  # a first telling that has Danforth look up
                tuples.append({"perceiver": "dan", "text": "Wentworth came into the hall.", "salience": "ordinary"})
            return {"narration": "You go into the hall.", "elapsed_minutes": 2, "perceiver_tuples": tuples,
                    "scene_tuples": [], "moves": [{"entity": "wen", "into": "hall"}], "suggested_next_action": ""}
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: events.append((e, k)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
    view = {"world_data": {"clock": CLOCK}}
    args = dict(records=recs, view=view, players=frozenset({"wen", "dan"}), playable=frozenset({"wen", "dan"}))
    r = run_smrti_pulse(canon, "wen", "I go into the hall", 6, "p6", mems, C(), None, **args)
    assert r.pending is None                                                       # no question: he goes in
    assert len(calls) == 2 and any(e == "smrti.lived_later_rerun" for e, _ in events)  # told once more, Danforth hidden
    assert "dan" not in {m.entity_id for m in r.memories_written}                   # nothing written for him
    assert {m.ended for m in r.memories_written if m.entity_id == "wen"} == {ONE + 2}  # at Wentworth's own time
    assert "CLOSED" not in seen[0] and "hall, until" not in seen[0] and "Danforth" not in seen[-1]
    # a telling that leaves him be is not told again
    greet["on"] = False
    calls.clear()
    r = run_smrti_pulse(canon, "wen", "I go into the hall", 6, "p6", mems, C(), None, **args)
    assert r.pending is None and len(calls) == 1
    # a world's own turn into it still may not go in: the question is kept for the engine's own use, with why
    from smritilila.smrti.pulse import locked_prompt
    from smritilila.smrti.kala import Kala
    k = Kala.of(CLOCK, recs, mems, frozenset({"wen", "dan"}), frozenset())
    assert locked_prompt(k, canon, "hall", ONE + 5 * H, ONE, [("present", "dan", ONE + 5 * H)], frozenset({"wen"})).startswith(
        "The hall is already lived later than your present: someone is there until")


def test_a_person_of_the_world_lived_later_is_not_hidden_by_a_second_telling(monkeypatch) -> None:
    """v5.50 (Marc, 2026-10-19: "lived-later people who are not a player's character stay seen but are not written for"). Three of the 36 lived-later reruns
    recorded since 09-22 made a twin (Jaroo Ashstaff 09-30, Miller Mytch 10-16, Tad 10-18): the first telling had the real person and offended only by writing a tuple
    for him; the second, shown no one there, invented him again. Here Danforth, lived later in the hall and NOT a player's character, is touched by the first
    telling (a tuple, and a move): the telling stands and is the only one asked for, nothing is written for him and nothing of his moves, and the witness counts
    it (smrti.lived_later_kept). Replayed offline on the 36 recorded first tellings (no model call): no twin, none written for or moved. A player's character in
    the same place is still hidden by a second telling (the test above)."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "dan": Entity("dan", ["Danforth"], 0, "genesis")})
    mems = [Memory("m1", "wen", frozenset(), "I stood in the yard.", "PRIMARY", 4, "yard", 3, "p3", None),
            Memory("m2", "dan", frozenset(), "I sat in the hall all night.", "PRIMARY", 4, "hall", 4, "p4", None)]
    recs = [Record("r1", "kosha", "wen", "yard", 0, "p0", None), Record("r2", "kosha", "dan", "hall", 0, "p0", None),
            Record("r3", "kosha", "hall", "yard", 0, "p0", None),
            _span("wen", ONE, ONE, "yard", "yard", 3), _span("dan", ONE, ONE + 5 * H, "hall", "hall", 4)]
    calls = []

    class C:
        def complete_with_tool(self, system, user, **kw):
            calls.append(user)
            return {"narration": "You go into the hall. Danforth looks up from his bench.", "elapsed_minutes": 2,
                    "perceiver_tuples": [{"perceiver": "wen", "text": "I went into the hall; Danforth looked up.", "salience": "ordinary"},
                                         {"perceiver": "dan", "text": "Wentworth came into the hall.", "salience": "ordinary"}],
                    "scene_tuples": [], "moves": [{"entity": "wen", "into": "hall"}, {"entity": "dan", "into": "yard"}], "suggested_next_action": ""}
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: events.append((e, k)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
    r = run_smrti_pulse(canon, "wen", "I go into the hall", 6, "p6", mems, C(), None, records=recs, view={"world_data": {"clock": CLOCK}},
                        players=frozenset({"wen"}), playable=frozenset({"wen"}))
    kinds = {e: k for e, k in events}
    assert r.pending is None and len(calls) == 1                                    # the first telling stands: one call
    assert "smrti.lived_later_rerun" not in kinds and kinds["smrti.lived_later_kept"]["ids"] == ["dan"]
    assert r.narration.startswith("You go into the hall. Danforth looks up")        # seen, as the actor would see him
    assert "dan" not in {m.entity_id for m in r.memories_written}                   # nothing written for him
    assert "dan" not in {x.entity_id for x in r.records_written if x.kind == "kosha"}   # and he is not moved
    assert {m.ended for m in r.memories_written if m.entity_id == "wen"} == {ONE + 2}
    assert not r.new_entities                                                       # nobody is made again


def test_every_span_stops_at_the_first_present_it_touches(monkeypatch) -> None:
    """v5 §20 v4.31 (Marc, 2026-09-28: "It feels like it should stop where they
    meet"). Stonemarch t154: Dunstan walked to the east track, where Jakal's
    present stood at 5:27. Here Danforth's present is the hall at 1:07 and
    Wentworth walks in at 1:00. The bound is shown before the telling -- where
    and when, never who (v4.27). A telling that reaches it ends there and they
    meet; one that runs past it is told once more, and ends where they meet;
    one that stops short meets nobody, and Danforth stays out of reach. The
    ten-minute wait (v4.28) is withdrawn. The same three cases hold when the present is a WAITING player's (offline, letting the world
    act for them, so carried while behind): a waiting player who is ahead is met at their present like any other (Stonemarch 1577-1590,
    Marc 2026-10-13: Jakal "caught up with" Dunstan, eight days on, and no jump was offered)."""
    from smritilila.smrti.pulse import run_smrti_pulse
    mems = [Memory("m1", "wen", frozenset(), "I stood in the yard.", "PRIMARY", 4, "yard", 3, "p3", None),
            Memory("m2", "dan", frozenset(), "I waited in the hall.", "PRIMARY", 4, "hall", 4, "p4", None)]
    recs = [Record("r1", "kosha", "wen", "yard", 0, "p0", None), Record("r2", "kosha", "dan", "hall", 0, "p0", None),
            Record("r3", "kosha", "hall", "yard", 0, "p0", None),
            _span("wen", ONE, ONE, "yard", "yard", 3), _span("dan", ONE, ONE + 7, "hall", "hall", 4)]
    for waiting, (minutes, expect_end, expect_calls) in [(w, c) for w in (False, True) for c in ((7, ONE + 7, 1), (12, ONE + 7, 2), (2, ONE + 2, 2))]:
        canon = Maya(world_time=6, entities={
            "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
            "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
            "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
            "dan": Entity("dan", ["Danforth"], 0, "genesis")})
        calls = []

        class C:
            def complete_with_tool(self, system, user, **kw):
                calls.append(user)
                return {"narration": "You go into the hall.", "elapsed_minutes": minutes,
                        "perceiver_tuples": [{"perceiver": "wen", "text": "I went into the hall.", "salience": "ordinary"},
                                             {"perceiver": "dan", "text": "Wentworth came into the hall.", "salience": "ordinary"}],
                        "scene_tuples": [], "moves": [{"entity": "wen", "into": "hall"}], "suggested_next_action": ""}
        events = []
        monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: events.append((e, k)))
        monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
        r = run_smrti_pulse(canon, "wen", "I go into the hall", 6, "p6", mems, C(), None, records=recs,
                            view={"world_data": {"clock": CLOCK}}, players=frozenset({"wen"} if waiting else {"wen", "dan"}),
                            playable=frozenset({"wen", "dan"}), movable=frozenset({"dan"} if waiting else ()))
        kinds = {e: k for e, k in events}
        assert "PRESENTS WITHIN REACH" in calls[0] and "at the hall, as of" in calls[0]
        assert "Danforth" not in calls[0].split("PRESENTS WITHIN REACH")[1]           # where and when, never who
        assert {m.ended for m in r.memories_written if m.entity_id == "wen"} == {expect_end}
        assert len(calls) == expect_calls and "smrti.met_waiting" not in kinds
        if minutes == 7:
            assert kinds["smrti.met"]["at"] == ONE + 7 and not kinds["smrti.met"]["clamped"]
            assert "smrti.lived_later_hidden" not in kinds and "smrti.arrival_rerun" not in kinds
        elif minutes == 12:
            assert kinds["smrti.meeting_rerun"]["meet_at"] == ONE + 7 and kinds["smrti.met"]["clamped"]
            assert "THIS PULSE REACHES THEM" in calls[1] and r.pending is None
        else:
            assert "smrti.met" not in kinds and "smrti.lived_later_rerun" in kinds  # his minutes there are out of reach


def test_a_telling_that_writes_for_someone_elsewhere_is_told_again_shown_where_they_are(monkeypatch) -> None:
    """Stonemarch, 2026-09-28, the east track: Jakal walked out and "found Elmo
    and Dunstan crouched in the frosted brush" -- both stood at the inn's
    kitchen postern. The engine dropped the telling's writes for them, the
    narration kept them, and Jakal talked to shadowy, not-yet-here Elmo. The
    telling's own writes show who it thought was here: one written for who
    stands outside the reach of where the pulse began and ended makes it told
    again, shown NOT HERE and where they are. Replayed before building (the
    live prompt, the second model, 3 runs): Dunstan and Elmo stayed at the postern 3/3."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "village": Entity("village", ["the village"], 0, "genesis"),
        "inn": Entity("inn", ["the inn"], 0, "genesis", kosha="village"),
        "knoll": Entity("knoll", ["the knoll"], 0, "genesis", avarana="scene", kosha="village"),
        "track": Entity("track", ["the east track"], 0, "genesis", avarana="scene", kosha="village"),
        "postern": Entity("postern", ["the kitchen postern"], 0, "genesis", avarana="scene", kosha="inn"),
        "jakal": Entity("jakal", ["Jakal Hierophant"], 0, "genesis"),
        "dunstan": Entity("dunstan", ["Dunstan Mere"], 0, "genesis")})
    mems = [Memory("m1", "jakal", frozenset(), "I stood on the knoll.", "PRIMARY", 4, "knoll", 3, "p3", None),
            Memory("m2", "dunstan", frozenset(), "I waited at the postern.", "PRIMARY", 4, "postern", 4, "p4", None)]
    recs = [Record("r1", "kosha", "jakal", "knoll", 0, "p0", None), Record("r2", "kosha", "dunstan", "postern", 0, "p0", None)]
    seen, events = [], []

    class C:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            shown = "NOT HERE" in user
            tuples = [{"perceiver": "jakal", "text": "I walked to the east track.", "salience": "ordinary"}]
            if not shown:  # the first telling puts him in the brush
                tuples.append({"perceiver": "dunstan", "text": "Jakal found me in the brush.", "salience": "ordinary"})
            return {"narration": "You walk to the track.", "perceiver_tuples": tuples, "scene_tuples": [],
                    "moves": [{"entity": "jakal", "into": "track"}], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: events.append((e, k)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
    run_smrti_pulse(canon, "jakal", "I set out to the east track", 6, "p6", mems, C(), None, records=recs,
                    players=frozenset({"jakal", "dunstan"}), playable=frozenset({"jakal", "dunstan"}))
    kinds = {e: k for e, k in events}
    assert len(seen) == 2 and kinds["smrti.shadow_rerun"]["not_here"] == ["dunstan"]
    assert 'NOT HERE' in seen[1] and 'dunstan "Dunstan Mere", at the kitchen postern' in seen[1]


def test_a_telling_that_leaves_someone_in_no_scene_is_told_again(monkeypatch) -> None:
    """Stonemarch, 2026-09-27 19:19, the ditch: the Narrator minted "the lane to the Welcome Wench" as a thing in the
    village beside the scene of that name, and moved Kobort into it -- in the village, which is no scene, he stood in
    none, and a night later a look-alike was minted for him. Anyone a telling writes for or moves, left in no scene,
    has it told again, shown IN NO SCENE. A far place declared in the village with nothing written for it does not."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "village": Entity("village", ["the village"], 0, "genesis"),
        "knoll": Entity("knoll", ["the knoll"], 0, "genesis", avarana="scene", kosha="village"),
        "jakal": Entity("jakal", ["Jakal Hierophant"], 0, "genesis"),
        "kobort": Entity("kobort", ["Kobort the Strong"], 0, "genesis")})
    mems = [Memory("m1", "jakal", frozenset(), "I stood on the knoll.", "PRIMARY", 4, "knoll", 3, "p3", None),
            Memory("m2", "kobort", frozenset(), "I loomed on the knoll.", "PRIMARY", 4, "knoll", 4, "p4", None)]
    recs = [Record("r1", "kosha", "jakal", "knoll", 0, "p0", None), Record("r2", "kosha", "kobort", "knoll", 0, "p0", None)]

    def run(first):
        seen, events = [], []

        class C:
            def complete_with_tool(self, system, user, **kw):
                seen.append(user)
                extra = first if "IN NO SCENE" not in user else {"moves": [{"entity": "kobort", "into": "knoll"}]}
                return {"narration": "Kobort strides off.", "scene_tuples": [], "suggested_next_action": "",
                        "perceiver_tuples": [{"perceiver": "jakal", "text": "I watched him go.", "salience": "ordinary"},
                                             {"perceiver": "kobort", "text": "I walked off.", "salience": "ordinary"}],
                        **extra}
        monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: events.append((e, k)))
        monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
        run_smrti_pulse(Maya(world_time=6, entities=dict(canon.entities)), "jakal", "I watch him", 6, "p6", mems, C(),
                        None, records=list(recs), players=frozenset({"jakal"}), playable=frozenset({"jakal"}))
        return seen, events
    seen, events = run({"new_entities": [{"id": "NEW:lane", "name": "the lane", "in": "village", "scene": False}],
                        "moves": [{"entity": "kobort", "into": "NEW:lane"}]})
    assert len(seen) == 2 and [k["no_scene"] for e, k in events if e == "smrti.no_scene_rerun"] == [["kobort"]]
    assert 'IN NO SCENE' in seen[1] and 'kobort "Kobort the Strong", in the lane' in seen[1]
    seen, events = run({"new_entities": [{"id": "NEW:city", "name": "the far city", "in": "village", "scene": False}]})
    assert len(seen) == 1 and not any(e == "smrti.no_scene_rerun" for e, _ in events)  # declared far off: no rerun


def test_what_the_telling_itself_carries_out_of_reach_is_not_a_shadow(monkeypatch) -> None:
    """Stonemarch, 2026-09-28, the fen: the purse Jakal held was dropped into a new bog hole the telling left in the
    region above the scene -- after the telling it stood in no scene within reach, and read as a shadow; the round
    was told again. A shadow stood elsewhere before the pulse as well as after it."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "village": Entity("village", ["the village"], 0, "genesis"),
        "knoll": Entity("knoll", ["the knoll"], 0, "genesis", avarana="scene", kosha="village"),
        "jakal": Entity("jakal", ["Jakal Hierophant"], 0, "genesis"),
        "purse": Entity("purse", ["a stitched purse"], 0, "genesis")})
    mems = [Memory("m1", "jakal", frozenset(), "I stood on the knoll.", "PRIMARY", 4, "knoll", 3, "p3", None),
            Memory("m2", "purse", frozenset(), "I hung at the scholar's belt.", "PRIMARY", 4, "knoll", 4, "p4", None)]
    recs = [Record("r1", "kosha", "jakal", "knoll", 0, "p0", None), Record("r2", "kosha", "purse", "jakal", 0, "p0", None)]
    seen, events = [], []

    class C:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            return {"narration": "The purse slips from your belt into the bog.", "scene_tuples": [],
                    "perceiver_tuples": [{"perceiver": "jakal", "text": "I lost the purse.", "salience": "ordinary"},
                                         {"perceiver": "purse", "text": "I sank into black water.", "salience": "ordinary"}],
                    "new_entities": [{"id": "NEW:bog_hole", "name": "a bog hole", "in": "village", "scene": False}],
                    "moves": [{"entity": "purse", "into": "NEW:bog_hole"}], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: events.append((e, k)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
    run_smrti_pulse(canon, "jakal", "I cross the fen", 6, "p6", mems, C(), None, records=recs,
                    players=frozenset({"jakal"}), playable=frozenset({"jakal"}))
    assert not any(e == "smrti.shadow_rerun" for e, _ in events)
    # ... but the purse, carried into a bog hole left in the region, stands in no scene: that rerun instead
    assert [k["no_scene"] for e, k in events if e == "smrti.no_scene_rerun"] == [["purse"]]


def test_a_player_behind_is_not_there_yet(monkeypatch) -> None:
    """v5 §20 v4.31: a player whose present is behind the actor's now, within
    reach, is not there yet -- not shown, never written for, no arrival rerun
    for them. They meet the actor when their own span reaches the actor's
    present. Stonemarch t154 (the Welcome Wench): the arrival rerun had brought
    Jakal, six minutes behind, into Dunstan's telling."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "dan": Entity("dan", ["Danforth"], 0, "genesis")})
    mems = [Memory("m1", "wen", frozenset(), "I stood in the yard.", "PRIMARY", 4, "yard", 3, "p3", None),
            Memory("m2", "dan", frozenset(), "I sat in the hall.", "PRIMARY", 4, "hall", 4, "p4", None)]
    recs = [Record("r1", "kosha", "wen", "yard", 0, "p0", None), Record("r2", "kosha", "dan", "hall", 0, "p0", None),
            Record("r3", "kosha", "hall", "yard", 0, "p0", None),
            _span("wen", ONE, ONE + 30, "yard", "yard", 3), _span("dan", ONE, ONE + 6, "hall", "hall", 4)]
    calls, events = [], []

    class C:
        def complete_with_tool(self, system, user, **kw):
            calls.append(user)
            return {"narration": "You go into the hall.", "elapsed_minutes": 3,
                    "perceiver_tuples": [{"perceiver": "wen", "text": "I went into the hall.", "salience": "ordinary"}],
                    "scene_tuples": [], "moves": [{"entity": "wen", "into": "hall"}], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: events.append((e, k)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
    run_smrti_pulse(canon, "wen", "I go into the hall", 6, "p6", mems, C(), None, records=recs,
                    view={"world_data": {"clock": CLOCK}}, players=frozenset({"wen", "dan"}), playable=frozenset({"wen", "dan"}))
    kinds = {e: k for e, k in events}
    assert kinds["smrti.not_yet_here"]["ids"] == ["dan"] and "smrti.arrival_rerun" not in kinds and len(calls) == 1
    assert "PRESENTS WITHIN REACH" not in calls[0]


def _m(i, who, loc, tick, began, ended, tier="PRIMARY", present=()):
    return Memory(f"m{i}", who, frozenset(present), f"memory {i}", tier, 4, loc, tick, f"p{tick}", None, None, began, ended)


def test_time_is_read_from_the_memories_minutes() -> None:
    """The racer sleeps in the room 1:00-7:30 with the barman there, unvoiced;
    walks out to the square at 7:40. Read from the memories alone: the
    racer's and the barman's clocks, where the racer was, the room locked
    to a laggard at 3:00, the scenes' clocks, the tick's minute."""
    mems = [_m(1, "racer", "room", 1, None, None), _m(2, "barman", "room", 1, None, None),        # before the clock
            _m(3, "racer", "room", 5, ONE, ONE + 6 * H + 30),
            _m(4, "room", "room", 5, ONE, ONE + 6 * H + 30, "AMBIENT", {"racer", "barman", "lamp"}),
            _m(5, "racer", "square", 6, ONE + 6 * H + 30, ONE + 6 * H + 40),
            _m(6, "square", "square", 6, ONE + 6 * H + 30, ONE + 6 * H + 40, "AMBIENT", {"racer"})]
    k = Kala.of(CLOCK, [], mems)
    assert k.clock("racer") == ONE + 6 * H + 40 and k.clock("barman") == ONE + 6 * H + 30   # present, unvoiced: lived it too
    assert k.clock("lamp") is None and k.clock("room") is None                              # a thing that perceives nothing; a scene
    assert k.where_at("racer", ONE + 2 * H) == "room" and k.where_at("racer", ONE + 6 * H + 35) == "room"
    assert k.where_at("racer", ONE + 6 * H + 40) == "square"
    assert k.locked_until("room", ONE + 2 * H, "laggard") == ONE + 6 * H + 30
    assert k.scene_clock == {"room": ONE + 6 * H + 40, "square": ONE + 6 * H + 40}          # left at 7:40: the room saw it
    assert k.tick_minute == {5: ONE + 6 * H + 30, 6: ONE + 6 * H + 40}
    # the first build's records still read, for their pulses
    old = Kala.of(CLOCK, [_span("racer", ONE, ONE + 6 * H + 30, "room", "room", 5)], mems[:2])
    assert old.clock("racer") == ONE + 6 * H + 30


def test_a_worlds_turn_is_shown_what_players_lived_next_door(monkeypatch) -> None:
    """The cook, in the yard at 1:00, may take five hours. Danforth lived
    the hall next door 1:00-6:00: the turn is shown it, not stopped by it."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
        "cook": Entity("cook", ["the cook"], 0, "genesis"),
        "dan": Entity("dan", ["Danforth"], 0, "genesis")})
    mems = [_m(1, "cook", "yard", 3, None, None),
            _m(2, "dan", "hall", 4, ONE, ONE + 5 * H), _m(3, "hall", "hall", 4, ONE, ONE + 5 * H, "AMBIENT", {"dan"})]
    recs = [Record("r1", "kosha", "cook", "yard", 0, "p0", None), Record("r2", "kosha", "dan", "hall", 0, "p0", None),
            Record("r3", "kosha", "hall", "yard", 0, "p0", None)]
    seen = []

    class C:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            return {"narration": "The cook scrubbed pots.", "elapsed_minutes": 90,
                    "perceiver_tuples": [{"perceiver": "cook", "text": "I scrubbed the pots.", "salience": "ordinary"}],
                    "scene_tuples": [], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda *a, **k: None)
    r = run_smrti_pulse(canon, "cook", None, 6, "p6", mems, C(), None, records=recs, view={"world_data": {"clock": CLOCK}},
                        players=frozenset({"dan"}), playable=frozenset({"dan"}), max_elapsed=5 * H)
    assert "ALREADY LIVED HERE AND NEARBY" in seen[0] and "from Saturday, 28 April 1928, 1:00 at night, dan in hall" in seen[0]
    assert r.pending is None and {m.ended for m in r.memories_written} == {ONE + 5 * H}  # its whole span, not the 90 it said


def _landing_world(before: tuple = ()):
    """Teo waits at the landing 5:00-5:45: a long jump held for his consent. The vale and the road are next door; in this flow they take no
    turn, they only RECORD -- `before` is what they recorded earlier, told before Teo's pulse."""
    from smritilila.smrti.pulse import SmrtiPulseResult
    canon = Maya(world_time=9, entities={
        "landing": Entity("landing", ["the landing"], 0, "genesis", avarana="scene"),
        "vale": Entity("vale", ["the vale"], 0, "genesis", avarana="scene"),
        "road": Entity("road", ["the road"], 0, "genesis", avarana="scene"),
        "teo": Entity("teo", ["Teo"], 0, "genesis")})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["landing"], playable_entity_ids=["teo"])
    world.world_data = {"clock": CLOCK}
    five = ONE + 4 * H
    world.records = [Record("k1", "kosha", "teo", "landing", 0, "p0", None), Record("k2", "kosha", "landing", "vale", 0, "p0", None),
                     Record("k3", "kosha", "road", "vale", 0, "p0", None)]
    world.recent_memories = [_m(1, "teo", "landing", 8, five - 10, five)]
    for mem, rec in before:
        world.recent_memories.append(mem)
        world.records += [rec] if rec else []
    held_mem = _m(9, "teo", "landing", 10, five, five + 45)
    result = SmrtiPulseResult(narration="You wait.", memories_written=[held_mem], new_entities=[], bound_ids=["teo", "passenger", "road"],
                              suggested_next_action="", context_tuples=0)
    world.held["teo"] = {"req": schemas.PulseRequest(action_text="I wait for the bell", entity_id="teo"), "result": result,
                         "working_canon": copy_canon(canon, 10), "pulse_id": "p10", "scene": "landing", "tick": 9,
                         "kind": "long", "minutes": 45, "from": five, "to": five + 45, "origin": "landing", "dest": "landing",
                         "prompt": "..."}
    return world, five


def _teo_says_yes(monkeypatch, world, roll="real"):
    """Teo agrees. What would cost a model call is a stub that keeps what it was given: a scene's turn (which must never run now), the
    commit, the telling told again. `roll`: the die -- None (nothing interrupts), (minute, scene), or "real" (as it is)."""
    import asyncio
    turns, commits, reruns, rolls = [], [], [], []

    async def turn(*a, **k):
        turns.append((a, k))

    async def commit(world_, working, result_, actor, pid):
        commits.append((working.world_time, [m.world_time for m in result_.memories_written]))

    async def rerun(req, world_, actor, roles, bt, start, confirmed=False, wait_until=None, window=None, setting=None, interrupt=None):
        reruns.append({"window": window, "setting": setting, "interrupt": interrupt})
        return schemas.PulseResponse(narrations={}, promoted_entity_ids=[],
                                  suggested_next_action="", suggested_next_action_from_memory="")
    if roll != "real":
        def die(world_, held_, actor_):
            rolls.append(held_["from"])
            return roll
        monkeypatch.setattr(spanda, "_interruption", die)
    monkeypatch.setattr(spanda, "_run_smrti_activation_in_background", turn)
    monkeypatch.setattr(spanda, "_commit_smrti_result", commit)
    monkeypatch.setattr(spanda, "_submit_smrti_pulse", rerun)
    monkeypatch.setattr(spanda, "_after_smrti_commit", lambda *a, **k: schemas.PulseResponse(
        narrations={}, promoted_entity_ids=[], suggested_next_action="", suggested_next_action_from_memory=""))
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: None)
    monkeypatch.setattr(adhisthana, "get_world", lambda: _async(world))
    monkeypatch.setattr(spanda, "log_discarded", lambda **k: None)
    asyncio.run(spanda.confirm_pulse(schemas.ConfirmRequest(entity_id="teo", proceed=True), BackgroundTasks()))
    return turns, commits, reruns, rolls


def copy_canon(canon, t):
    import copy
    c = copy.deepcopy(canon)
    c.world_time = t
    return c


def test_a_long_jump_agreed_to_with_nothing_in_its_way_is_committed_as_it_was_told(monkeypatch) -> None:
    """Marc, 2026-10-27: "It's the wrong flow." After a Yes the world took up to three turns next door, one after another, and the telling was
    thrown away and told again: a median 30 s after Yes over 173 jumps, 67% of them with nothing found. Now one die is rolled, and if nothing
    interrupts, the telling the player was asked about is the one committed: no scene takes a turn, no model call is made."""
    world, five = _landing_world()
    turns, commits, reruns, rolls = _teo_says_yes(monkeypatch, world, roll=None)
    assert rolls == [five] and turns == [] and reruns == []
    assert commits == [(10, [10])]                                    # the held telling, committed at the tick it lands at


def test_an_interrupted_long_jump_is_told_again_once_with_the_interruption(monkeypatch) -> None:
    world, five = _landing_world()
    turns, commits, reruns, rolls = _teo_says_yes(monkeypatch, world, roll=(five + 20, "landing"))
    assert turns == [] and commits == []                              # no scene's turn; the held telling is not the one lived
    assert reruns == [{"window": None, "setting": None, "interrupt": (five + 20, "landing")}]   # one telling, told where it stops


def test_what_next_door_already_recorded_is_an_event_and_no_die_is_rolled(monkeypatch) -> None:
    """Stonemarch, 2026-10-01: a dragon came down on the green at 11:40, told before Filonius sat down next door at 11:41 to wait three hours;
    nothing showed it to him. A neighbour's recorded row whose pulse moved someone not the actor's is an event, and the jump ends there --
    found with no model call, so a jump still stops at what was already written, and a real event needs no die."""
    five = ONE + 4 * H
    landed = Memory("g7", "vale", frozenset(), "A winged beast came down on the vale; horns sounded.", "AMBIENT", 5, "vale", 7, "p7",
                    None, None, five - 1, five + 5)
    world, _five = _landing_world(before=[(landed, Record("gk7", "kosha", "beast", "vale", 7, "p7", None))])
    turns, commits, reruns, rolls = _teo_says_yes(monkeypatch, world, roll=None)
    assert turns == [] and commits == [] and rolls == []
    assert reruns == [{"window": [(five, "vale", "A winged beast came down on the vale; horns sounded.")], "setting": None, "interrupt": None}]


def test_what_next_door_recorded_quietly_neither_stops_nor_retells_the_long_jump(monkeypatch) -> None:
    """The counter-case: recorded rows with no move, mark or line by anyone are the world around, never a stop (listed as events they cut 4 of
    8 replayed waits short) -- and, now that no turn is run for them, never a reason to tell the jump again: the die is rolled and the held
    telling stands."""
    five = ONE + 4 * H
    quiet = Memory("g7", "vale", frozenset(), "Carts went by on the vale road.", "AMBIENT", 5, "vale", 7, "p7",
                   None, None, five + 5, five + 30)
    world, _five = _landing_world(before=[(quiet, None)])
    turns, commits, reruns, rolls = _teo_says_yes(monkeypatch, world, roll=None)
    assert turns == [] and reruns == [] and rolls == [five] and commits == [(10, [10])]


def test_a_row_recorded_next_door_that_names_the_actor_is_an_event(monkeypatch) -> None:
    """Told not to, a neighbour's row says Teo sat on the post: by his name, the engine's own string. It stands as an event, so his pulse is
    told again and holds it (the check a fresh turn's row had too)."""
    five = ONE + 4 * H
    named = Memory("g8", "road", frozenset(), "Teo sat on the post by the landing.", "AMBIENT", 5, "road", 8, "p8", None, None, five + 15, five + 20)
    world, _five = _landing_world(before=[(named, None)])
    turns, commits, reruns, rolls = _teo_says_yes(monkeypatch, world, roll=None)
    assert turns == [] and commits == [] and rolls == []
    assert reruns == [{"window": [(five + 15, "road", "Teo sat on the post by the landing.")], "setting": None, "interrupt": None}]


def test_an_interrupted_jump_is_told_under_the_same_sky_as_the_quiet_rows_next_door(monkeypatch) -> None:
    """v4.32 (the Crossing: the held night had rain and a flood, next door frost): when the jump is told again anyway, what next door recorded
    quietly in the same hours is shown with the interruption, as it was with an event."""
    five = ONE + 4 * H
    quiet = Memory("g7", "vale", frozenset(), "Carts went by on the vale road.", "AMBIENT", 5, "vale", 7, "p7",
                   None, None, five + 5, five + 30)
    world, _five = _landing_world(before=[(quiet, None)])
    turns, commits, reruns, rolls = _teo_says_yes(monkeypatch, world, roll=(five + 30, "landing"))
    assert reruns == [{"window": None, "setting": [("vale", "Carts went by on the vale road.")], "interrupt": (five + 30, "landing")}]


def test_the_chance_of_an_interruption_grows_with_the_length_of_the_jump() -> None:
    """Marc, 2026-10-27: a probability of an interrupting event as a function of the jump's length; 96 hours the mean time between them."""
    from smritilila.smrti.kala import INTERRUPTION_HOURS, interruption_chance
    assert INTERRUPTION_HOURS == 96
    assert interruption_chance(0, 96) == 0 and interruption_chance(60, None) == 0 and interruption_chance(60, 0) == 0
    odds = {m: round(interruption_chance(m, 96), 3) for m in (30, 60, 120, 480, 24 * 60, 48 * 60, 7 * 24 * 60, 14 * 24 * 60)}
    assert odds == {30: 0.005, 60: 0.01, 120: 0.021, 480: 0.08, 1440: 0.221, 2880: 0.393, 10080: 0.826, 20160: 0.97}
    chances = [interruption_chance(m, 96) for m in (10, 100, 1000, 10000)]
    assert chances == sorted(chances) and chances[-1] < 1.0 + 1e-12


def test_a_worlds_own_scale_sets_the_chance_and_the_pace_holds_the_default() -> None:
    world, _five = _landing_world()
    assert world.pace()["interruption_hours"] == 96
    world.world_data["pacing"] = {"interruption_hours": 12}
    assert world.pace()["interruption_hours"] == 12


def test_the_die_is_the_pulses_own_and_a_retry_meets_the_same_one() -> None:
    """Like the fortune (Marc, 2026-09-27, Jakal declining a closed scene three times): whose roll and when, never the attempt. Declining and
    asking again at the same moment meets the same die, and a longer jump is interrupted whenever a shorter one was."""
    world, five = _landing_world()
    held = world.held["teo"]
    world.world_data["pacing"] = {"interruption_hours": 0.001}          # a chance of 1 for any jump: the die is the minute
    a = spanda._interruption(world, held, "teo")
    assert a is not None and a == spanda._interruption(world, held, "teo")
    minute, scene = a
    assert five < minute < five + 45 and scene == "landing"
    world.world_data["pacing"] = {"interruption_hours": 0}
    assert spanda._interruption(world, held, "teo") is None                # a world that sets none is never interrupted
    world.world_data["pacing"] = {"interruption_hours": 2}
    for tick in range(1, 120):
        w = copy_canon(world.canon, tick)
        short, long_ = dict(held, working_canon=w, to=five + 30), dict(held, working_canon=w, to=five + 600)
        if spanda._interruption(world, short, "teo") is not None:
            assert spanda._interruption(world, long_, "teo") is not None


def test_a_closed_scene_is_never_interrupted() -> None:
    """The pacing's `arrivals` (v5 §15): a closed valley is closed -- nothing comes in, so nothing interrupts. Sure to be interrupted otherwise."""
    import json
    world, five = _landing_world()
    world.world_data["pacing"] = {"interruption_hours": 0.001}
    held = world.held["teo"]
    assert spanda._interruption(world, held, "teo") is not None
    world.records.append(Record("pz", "prakriti", "landing", json.dumps({"arrivals": False}), 0, "p0", "pacing"))
    world._samskara = None
    assert spanda._interruption(world, held, "teo") is None


def test_the_die_over_many_jumps_follows_the_chance_and_the_minute_is_uniform_over_the_span() -> None:
    world, five = _landing_world()
    held = world.held["teo"]
    day = dict(held, to=five + 24 * 60)
    hits, where = 0, []
    for tick in range(1, 2001):
        r = spanda._interruption(world, dict(day, working_canon=copy_canon(world.canon, tick)), "teo")
        if r is not None:
            hits += 1
            where.append((r[0] - five) / (24 * 60))
    assert 0.18 < hits / 2000 < 0.26                                      # 22.1% for a day at a mean of 96 hours
    assert 0.45 < sum(where) / len(where) < 0.55 and 0 < min(where) and max(where) < 1
    assert sum(1 for w in where if w < 0.5) / len(where) > 0.4 and sum(1 for w in where if w >= 0.5) / len(where) > 0.4


def test_the_interruption_is_shown_to_the_telling_which_stops_there(monkeypatch) -> None:
    """The interruption rides the window's own path: the action is lived only up to the minute, the Narrator says what comes, and the
    pulse ends there (`elapsed_minutes` 20). The engine says only that something does: it knows no kind (s0)."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis")})
    seen = []

    class C:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            return {"narration": "A rider comes into the yard; you stand.", "elapsed_minutes": 20,
                    "perceiver_tuples": [{"perceiver": "wen", "text": "A rider came into the yard and I stood.", "salience": "ordinary"}],
                    "scene_tuples": [], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda *a, **k: None)
    r = run_smrti_pulse(canon, "wen", "I wait for the bell", 6, "p6", [_m(1, "wen", "yard", 3, None, None)], C(), None,
                        records=[Record("r1", "kosha", "wen", "yard", 0, "p0", None)], view={"world_data": {"clock": CLOCK}},
                        players=frozenset({"wen"}), playable=frozenset({"wen"}), confirmed=True,
                        interrupt=(ONE + 20, "yard"))
    assert len(seen) == 1 and "SOMETHING INTERRUPTS THIS ACTION at Saturday, 28 April 1928, 1:20 at night, in yard" in seen[0]
    assert "THESE HAPPEN" not in seen[0]                                   # not a recorded event: the Narrator's to say
    assert r.pending is None and {m.ended for m in r.memories_written} == {ONE + 20}


def test_the_rerun_is_shown_the_window_and_stops_there(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis")})
    seen = []

    class C:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            return {"narration": "The bell rings; you stand.", "elapsed_minutes": 20,
                    "perceiver_tuples": [{"perceiver": "wen", "text": "I stood when the bell rang.", "salience": "ordinary"}],
                    "scene_tuples": [], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda *a, **k: None)
    r = run_smrti_pulse(canon, "wen", "I wait for the bell", 6, "p6", [_m(1, "wen", "yard", 3, None, None)], C(), None,
                        records=[Record("r1", "kosha", "wen", "yard", 0, "p0", None)], view={"world_data": {"clock": CLOCK}},
                        players=frozenset({"wen"}), playable=frozenset({"wen"}), confirmed=True,
                        window=[(ONE + 20, "yard", "A traveller rang the bell.")])
    assert len(seen) == 1 and "IN THE TIME THIS ACTION WOULD TAKE, THESE HAPPEN" in seen[0]
    assert "at Saturday, 28 April 1928, 1:20 at night, in yard: A traveller rang the bell." in seen[0]
    assert r.pending is None and {m.ended for m in r.memories_written} == {ONE + 20}


def test_the_rerun_is_shown_the_world_around_and_it_never_stops_the_action(monkeypatch) -> None:
    """v4.32: the quiet turns' rows are shown as the world around in the
    action's time -- same sky -- and do not stop it (the telling runs its
    whole 45 minutes)."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis")})
    seen = []

    class C:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            return {"narration": "You wait out the frost.", "elapsed_minutes": 45,
                    "perceiver_tuples": [{"perceiver": "wen", "text": "I waited in the frost.", "salience": "ordinary"}],
                    "scene_tuples": [], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda *a, **k: None)
    r = run_smrti_pulse(canon, "wen", "I wait for the bell", 6, "p6", [_m(1, "wen", "yard", 3, None, None)], C(), None,
                        records=[Record("r1", "kosha", "wen", "yard", 0, "p0", None)], view={"world_data": {"clock": CLOCK}},
                        players=frozenset({"wen"}), playable=frozenset({"wen"}), confirmed=True,
                        setting=[("yard", "The yard lay white with frost.")])
    assert len(seen) == 1 and "THE WORLD AROUND, IN THIS ACTION'S TIME" in seen[0]
    assert "in the yard: The yard lay white with frost." in seen[0] and "THESE HAPPEN" not in seen[0]
    assert r.pending is None and {m.ended for m in r.memories_written} == {ONE + 45}


def test_an_idle_player_keeps_their_own_clock() -> None:
    """Danforth stands idle in his back room at 1:00. The floorboard takes a
    turn there, 1:00-1:20, and he is given its row as his own witnessed
    tuple: his clock stays at 1:00. A turn the world takes FOR him (a tuple
    of his own) moves it."""
    row = "The floorboard creaked under the rug."
    mems = [_m(1, "dan", "back", 1, ONE - 10, ONE),
            _m(2, "board", "back", 2, ONE, ONE + 20),
            _m(3, "back", "back", 2, ONE, ONE + 20, "AMBIENT", {"dan", "board"}),
            Memory("m4", "dan", frozenset({"dan", "board"}), row, "PRIMARY", 5, "back", 2, "p2", None, None, ONE, ONE + 20)]
    mems[2] = dataclasses_replace(mems[2], experience=row)
    assert Kala.of(CLOCK, [], mems).clock("dan") == ONE + 20                      # without knowing he is a player: carried
    k = Kala.of(CLOCK, [], mems, players={"dan"})
    assert k.clock("dan") == ONE and k.clock("board") == ONE + 20                 # idle: his own clock; the board's turn is the board's
    own = mems + [_m(5, "dan", "back", 3, ONE + 20, ONE + 35)]                     # the world acts for him: a tuple of his own
    assert Kala.of(CLOCK, [], own, players={"dan"}).clock("dan") == ONE + 35


def dataclasses_replace(m, **kw):
    import dataclasses
    return dataclasses.replace(m, **kw)


def test_the_counters_for_a_meeting_across_time_and_a_voice_from_ahead(monkeypatch) -> None:
    """Wentworth (a player, idle at 1:00) is in the yard Danforth walks into
    at 3:00; the barman there has lived to 5:00 and is voiced at 3:02."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "dan": Entity("dan", ["Danforth"], 0, "genesis"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "barman": Entity("barman", ["the barman"], 0, "genesis")})
    three = ONE + 2 * H
    mems = [_m(1, "wen", "yard", 1, ONE - 5, ONE), _m(2, "dan", "yard", 2, three - 5, three),
            _m(3, "barman", "yard", 3, three, ONE + 4 * H)]
    recs = [Record("r1", "kosha", "wen", "yard", 0, "p0", None), Record("r2", "kosha", "dan", "yard", 0, "p0", None),
            Record("r3", "kosha", "barman", "yard", 0, "p0", None)]
    events = []

    class C:
        def complete_with_tool(self, system, user, **kw):
            return {"narration": "You nod to the barman.", "elapsed_minutes": 2,
                    "perceiver_tuples": [{"perceiver": "dan", "text": "I nodded to the barman.", "salience": "ordinary"},
                                         {"perceiver": "barman", "text": "The young man nodded to me.", "salience": "ordinary"}],
                    "scene_tuples": [], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: events.append((e, k)))
    run_smrti_pulse(canon, "dan", "I nod to the barman", 6, "p6", mems, C(), None, records=recs,
                    view={"world_data": {"clock": CLOCK}}, players=frozenset({"dan", "wen"}), playable=frozenset({"dan", "wen"}))
    kinds = {e: k for e, k in events if e in ("smrti.meeting_behind", "smrti.voiced_ahead", "smrti.time",
                                                 "smrti.lived_later_hidden", "smrti.lived_later_voiced")}
    assert kinds["smrti.meeting_behind"]["player"] == "wen" and kinds["smrti.meeting_behind"]["gap"] == 2 * H + 2
    # v4.27: the barman has lived the yard to 5:00 -- out of Danforth's reach at 3:00: not shown, not written for
    assert "barman" in kinds["smrti.lived_later_hidden"]["ids"] and kinds["smrti.lived_later_voiced"]["perceiver"] == "barman"
    assert "smrti.voiced_ahead" not in kinds
    assert "kala_ms" in kinds["smrti.time"]
    # in a round with Danforth, Wentworth lives the pulse's span with him: behind only by what the round's
    # start pulled him forward, not by the round's own length (Stonemarch 2026-09-27: 5 of 7 counts were that)
    for wen_at, counted in ((ONE, 2 * H), (three - 5, None)):
        events.clear()
        m = [_m(1, "wen", "yard", 1, wen_at - 5, wen_at)] + mems[1:]
        run_smrti_pulse(canon, "dan", "I nod to the barman", 6, "p6", m, C(), None, records=recs,
                        view={"world_data": {"clock": CLOCK}}, players=frozenset({"dan", "wen"}),
                        playable=frozenset({"dan", "wen"}), sangama={"wen": "I wait by the gate."}, wait_until=three)
        behind = [k for e, k in events if e == "smrti.meeting_behind"]
        assert ([(k["gap"], k["in_round"]) for k in behind] == [(counted, True)]) if counted else not behind


def test_thirty_minutes_without_acting_logs_a_player_off(monkeypatch) -> None:
    """Marc, 2026-09-25: auto log off after 30 minutes of inactivity -- as the
    button does. Seen 31 minutes ago: off. Seen a minute ago: not. Never seen
    since the server started 31 minutes ago: off. One who lets the world act
    for them is logged off and stays in the world."""
    import asyncio
    import time as _time
    from smritilila import main
    canon = Maya(world_time=1, entities={p: Entity(p, [p], 0, "genesis") for p in ("a", "b", "c")})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="a", entry_point_location_ids=[], playable_entity_ids=["a", "b", "c"])
    now = _time.monotonic()
    world.started = now - 31 * 60
    world.last_seen = {"a": now - 31 * 60, "b": now - 60}
    world.autonomous = {"c"}
    saved = []

    class Store:
        async def save_logged_off(self, ids):
            saved.append(ids)
    monkeypatch.setattr(adhisthana, "get_world_store", lambda: Store())
    monkeypatch.setattr(adhisthana, "log_event", lambda *a, **k: None)
    assert asyncio.run(adhisthana._log_off_idle(world)) == ["a", "c"]
    assert world.logged_off == {"a", "c"} and saved == [["a", "c"]]
    assert world.absent() == frozenset({"a"})                                    # c lets the world act for them: still here
    assert asyncio.run(adhisthana._log_off_idle(world)) == [] and len(saved) == 1      # once
    assert world.come_back("a") and "a" in world.returning                       # acting again: back, and narrated


def test_what_goes_with_the_actor_lives_the_journey_whole(monkeypatch) -> None:
    """Austin t32 (2026-10-05): Franklin left his apartment for the street, phone in his pocket. The phone, written in
    no scene like him, was cut to its first minute as a room left behind is; a pulse behind its carrier, its own turn
    (t33) was shown nothing of leaving and it lit on the desk. What moved with the actor lives his span."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "earth": Entity("earth", ["the world at large"], 0, "itihasa"),
        "city": Entity("city", ["the city"], 0, "genesis", kosha="earth"),  # in the earth, which perceives: no scene
        "flat": Entity("flat", ["the apartment"], 0, "genesis", avarana="scene", kosha="city"),
        "frank": Entity("frank", ["Franklin"], 0, "genesis"),
        "phone": Entity("phone", ["Franklin's phone"], 0, "genesis"),
        "book": Entity("book", ["the Gita"], 0, "genesis")})
    mems = [Memory("m0", "earth", frozenset(), "A boom city under an October sun.", "PRIMARY", 3, "earth", 0, "p0", None, None, ONE - 60, ONE - 60),
            Memory("m1", "frank", frozenset(), "I sat at my desk.", "PRIMARY", 4, "flat", 3, "p3", None, None, ONE, ONE + 10),
            Memory("m2", "phone", frozenset(), "I lay dark on the desk.", "PRIMARY", 6, "flat", 3, "p3", None, None, ONE, ONE + 10),
            Memory("m3", "book", frozenset(), "I lay closed on the desk.", "PRIMARY", 6, "flat", 3, "p3", None, None, ONE, ONE + 10)]
    recs = [Record("r1", "kosha", "frank", "flat", 0, "p0", None), Record("r2", "kosha", "phone", "flat", 0, "p0", None),
            Record("r3", "kosha", "book", "flat", 0, "p0", None)]

    class C:
        def complete_with_tool(self, system, user, **kw):
            return {"narration": "You pocket the phone and go down to the street.", "elapsed_minutes": 10,
                    "perceiver_tuples": [{"perceiver": "frank", "text": "I took my phone and went down to the street.", "salience": "ordinary"},
                                         {"perceiver": "phone", "text": "I rode in a pocket down to the street.", "salience": "passing"},
                                         {"perceiver": "book", "text": "I lay closed as the door shut.", "salience": "passing"}],
                    "scene_tuples": [{"scene": "flat", "text": "Franklin takes his phone and leaves.", "present": []}],
                    "moves": [{"entity": "frank", "into": "city"}, {"entity": "phone", "into": "city"}], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: None)
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
    res = run_smrti_pulse(canon, "frank", "I take my phone and go down to the street", 6, "p6", mems, C(), None, records=recs,
                          view={"world_data": {"clock": CLOCK}}, players=frozenset({"frank"}), playable=frozenset({"frank"}))
    spans = {m.entity_id: (m.began, m.ended) for m in res.memories_written}
    assert spans["frank"] == (ONE + 10, ONE + 20)
    assert spans["phone"] == spans["frank"]        # it went with him: the journey, whole
    # left behind: its first minute only (v5 §19: a room left behind does not live the traveller's journey). Until
    # 2026-10-05 `whole` fell back to the scene LEFT when the actor went into no scene, and the room and what stayed
    # in it lived the walk -- Austin t32's apartment, notebook and Upanishads rows span 7:17-7:27 while Franklin
    # walked out (the pass's P1; replayed on every world: Austin alone, 2 pulses of 47, no clock anyone read moved)
    assert spans["book"] == spans["flat"] == (ONE + 10, ONE + 10)


def test_an_actors_own_last_moment_is_never_hidden_as_later_than_now(monkeypatch) -> None:
    """Austin t33, as the store held it: the phone's t32 cut to 7:17-7:17, Franklin's 7:17-7:27, the pulse's clock
    7:27. On the phone's turn at 7:17, 'nothing later than now' hid every t32 tuple, its own leaving included, and it
    narrated the desk. The actor's last moment, and whoever is in transit with it, are shown whatever the clock."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=33, entities={
        "earth": Entity("earth", ["the world at large"], 0, "itihasa"),
        "city": Entity("city", ["the city"], 0, "genesis", kosha="earth"),  # in the earth, so the city is no scene
        "flat": Entity("flat", ["the apartment"], 0, "genesis", avarana="scene", kosha="city"),
        "frank": Entity("frank", ["Franklin"], 0, "genesis"),
        "phone": Entity("phone", ["Franklin's phone"], 0, "genesis")})
    t = ONE
    mems = [Memory("m0", "earth", frozenset(), "A boom city under an October sun.", "PRIMARY", 3, "earth", 0, "p0", None, None, t - 60, t - 60),
            # the earth perceives (itihasa), so a chain ending in it reaches no scene -- as Austin's does
            Memory("m1", "frank", frozenset(), "I sat at my desk.", "PRIMARY", 4, "flat", 31, "p31", None, None, t - 10, t),
            Memory("m2", "phone", frozenset(), "I lay dark on the desk.", "PRIMARY", 6, "flat", 31, "p31", None, None, t - 10, t),
            Memory("m3", "frank", frozenset(), "I took my phone and rode the elevator down to the street.", "PRIMARY", 5, "nowhere", 32, "p32", None, None, t, t + 10),
            Memory("m4", "phone", frozenset(), "I rode in a pocket out of the apartment.", "PRIMARY", 6, "nowhere", 32, "p32", None, None, t, t),
            Memory("m5", "flat", frozenset(), "Franklin leaves, phone in hand.", "AMBIENT", 5, "flat", 32, "p32", None, None, t, t)]
    recs = [Record("r1", "kosha", "frank", "flat", 0, "p0", None), Record("r2", "kosha", "phone", "flat", 0, "p0", None),
            Record("r3", "kosha", "frank", "city", 32, "p32", "moved"), Record("r4", "kosha", "phone", "city", 32, "p32", "moved")]
    seen = []

    class C:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            return {"narration": "The phone rides dark in a pocket down the street.", "elapsed_minutes": 10,
                    "perceiver_tuples": [{"perceiver": "phone", "text": "I rode dark in a pocket along the street.", "salience": "passing"}],
                    "scene_tuples": [], "moves": [], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: None)
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
    run_smrti_pulse(canon, "phone", None, 33, "p33", mems, C(), None, records=recs,
                    view={"world_data": {"clock": CLOCK}}, players=frozenset({"frank"}), playable=frozenset({"frank"}))
    assert seen and "I rode in a pocket out of the apartment." in seen[0]            # its own last moment
    assert "rode the elevator down to the street" in seen[0]                          # its carrier's, in transit with it
    assert "SCENE: none" in seen[0] and "PRESENT (in the scene, by what they are in): frank" in seen[0]


def test_a_pulse_is_as_long_as_the_narrator_says_and_a_jump_of_days_is_worded_in_days() -> None:
    """v5.31 (Marc, 2026-10-11): the cap of a day is gone; 3% of the Narrator's declared durations were a day or more (Filonius's 'two cold days' was written as 1440)."""
    from smritilila.smrti import kala as K
    from smritilila.smrti.pulse import _duration
    assert K._minutes(2880) == 2880 and K._minutes(131_400) == 131_400                      # days, a season: as declared
    assert K._minutes(-5) == 0 and K._minutes("x") == 1 and K._minutes(None) == 1            # a negative, a word or nothing: as before
    assert K._minutes(1e30) == K.MAX_ELAPSED and K.MAX_ELAPSED > 100 * 525_600              # only the integer the column holds bounds it, far past a century
    assert K.lived_span(10_080, 2.0) == 5_040 and K.lived_span(10_080, 1.0) == 10_080        # a week at half pace, no day cap on either branch
    assert _duration(45) == "45 min" and _duration(85) == "1 h 25 min" and _duration(7 * 60) == "7 h"   # a day's wording is unchanged below a day
    assert _duration(1440) == "1 day" and _duration(2880 + 180) == "2 days, 3 hours"


def test_a_named_player_lived_past_this_moment_in_the_actors_own_scene_is_told_where_the_actor_is(monkeypatch) -> None:
    """Stonemarch 1577-1579 (Marc, 2026-10-13, "Let's try A"): Jakal stood at the keep, in the scene Dunstan -- a carried player, eight days
    ahead -- lived in at that minute. He was present, so never in players_named; the engine hid him as lived later and the Narrator was
    told nothing, so a freshly minted guard told Jakal what Dunstan was doing. Named in the action, he now gets the LIVED PAST line, with
    the place being where the actor is; not named, nothing is said of him."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=6, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "dan": Entity("dan", ["Danforth"], 0, "genesis"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "barman": Entity("barman", ["the barman"], 0, "genesis")})
    three = ONE + 2 * H
    mems = [_m(1, "wen", "yard", 1, ONE, ONE + 4 * H), _m(2, "dan", "yard", 2, three - 5, three),
            _m(3, "barman", "yard", 3, three - 30, three)]
    recs = [Record("r1", "kosha", "wen", "yard", 0, "p0", None), Record("r2", "kosha", "dan", "yard", 0, "p0", None),
            Record("r3", "kosha", "barman", "yard", 0, "p0", None)]
    seen = []

    class C:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            return {"narration": "The barman shrugs.", "elapsed_minutes": 2,
                    "perceiver_tuples": [{"perceiver": "dan", "text": "I asked the barman.", "salience": "ordinary"}],
                    "scene_tuples": [], "suggested_next_action": ""}

    def go(action):
        seen.clear()
        run_smrti_pulse(canon, "dan", action, 6, "p6", mems, C(), None, records=recs, view={"world_data": {"clock": CLOCK}},
                        players=frozenset({"dan"}), playable=frozenset({"dan", "wen"}), movable=frozenset({"wen"}))
        return seen[0]
    named = go("I ask the barman where Wentworth is")
    assert 'LIVED PAST THIS MOMENT' in named and 'wen "Wentworth", at the yard, where the actor is now' in named
    assert "MOVABLE" not in named                                                  # nor is he offered as movable
    assert "LIVED PAST" not in go("I ask the barman for a drink")                  # not named: nothing is said of him
