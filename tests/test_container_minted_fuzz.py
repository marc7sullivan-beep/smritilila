# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Tier 0 of the v5.41 test scope (2026-10-15): odd Narrator moves against the engine, no model call.

Found by a fuzz of 25 odd moves and by the training log (notes/container_mint_tier0_2026-10-15/): the first nine tests state what the design requires; the
last group are gaps the first cut had (a truncated person handle made a second person, a sentence became a label, `!!!` made a container, the id was cut
at 48 characters) and the guard on what the engine makes from words closed (Marc: "do A"). Cases from real answers are named by their training-log line.
"""

from __future__ import annotations

import time

from test_container_minted import _made, _run, _world

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.smrti.kosha import Kosha
from smritilila.smrti.pulse import run_smrti_pulse

PEOPLE = {
    "marren-who-keeps-the-inn": Entity("marren-who-keeps-the-inn", ["Marren, who keeps the inn"], 0, "genesis", kosha="inn"),
    "mia": Entity("mia", ["Mia"], 0, "genesis", kosha="inn"),
}


def test_a_chain_of_two_undeclared_containers_is_two_mentions_and_nothing_stands(monkeypatch) -> None:
    r, k, refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": "a pouch"}, {"entity": "a pouch", "into": "a sack"}]})
    assert len(_made(r, "a pouch")) == 1 and len(_made(r, "a sack")) == 1 and len(undeclared) == 2 and not refused
    assert k.container_at("key", 4) == "inn"                                  # no holder anywhere in the chain: the key is put back


def test_an_undeclared_container_put_into_itself_is_refused_as_self(monkeypatch) -> None:
    r, _k, refused, _u = _run(monkeypatch, {"moves": [{"entity": "a bowl", "into": "a bowl"}]})
    assert len(_made(r, "a bowl")) == 1 and [x["reason"] for x in refused] == ["self"]


def test_the_hash_tail_the_narrator_copies_onto_its_own_new_row_finds_the_row(monkeypatch) -> None:
    """Training log 1866 (Stonemarch, 2026-09-30): the row said NEW:loosened_iron_bracket_spike and the move said `loosened_iron_bracket_spike_8f7a21`, the tail
    it expected the engine to add. The engine dropped the move; the branch resolves it onto the row (one entity, the declared name)."""
    r, k, refused, _u = _run(monkeypatch, {
        "new_entities": [{"id": "NEW:loosened_spike", "name": "a loosened bracket spike", "in": "inn", "scene": False}],
        "moves": [{"entity": "loosened_spike_8f7a21", "into": "wen"}]})
    spike = _made(r, "a loosened bracket spike")
    assert len(spike) == 1 and len(r.new_entities) == 1 and not refused
    assert k.container_at(spike[0].id, 4) == "wen"


def test_twenty_distinct_undeclared_containers_cost_twenty_mentions_and_move_nothing(monkeypatch) -> None:
    t0 = time.perf_counter()
    r, k, refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": f"jar number {i}"} for i in range(20)]})
    assert len(r.new_entities) == 20 and len(undeclared) == 20 and not refused
    assert k.container_at("key", 4) == "inn" and time.perf_counter() - t0 < 5


def test_the_actor_into_an_undeclared_container_is_what_a_bare_new_handle_always_was(monkeypatch) -> None:
    """Parity with the old path: `NEW:cellar` with no row was already made by the Binder, and the actor in it is its own scene (v5 §11), left to the place rerun
    to say what holds it. Words for the same cellar do the same; the branch adds no new way to lose the actor."""
    said = {}
    for how in ("NEW:cellar", "a cellar"):
        r, k, _refused, _u = _run(monkeypatch, {"moves": [{"entity": "wen", "into": how}]})
        cellar = [e for e in r.new_entities if "cellar" in e.labels[0]]
        said[how] = (len(cellar), k.container_at("wen", 4) == (cellar[0].id if cellar else None), k.container_at(cellar[0].id, 4) if cellar else "none")
    assert said["NEW:cellar"] == said["a cellar"] == (1, True, None)


def test_a_player_who_is_not_the_actor_stays_put_and_the_container_made_for_the_move_is_empty(monkeypatch) -> None:
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", False)
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **kw: None)
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **kw: events.append((e, kw)))
    ents, mems = _world()
    ents.update(PEOPLE)

    class C:
        def complete_with_tool(self, system, user, **kw):
            return {"narration": "x", "perceiver_tuples": [{"perceiver": "wen", "text": "I rowed.", "salience": "ordinary"}], "scene_tuples": [],
                    "suggested_next_action": "", "moves": [{"entity": "mia", "into": "a rowboat"}]}
    canon = Maya(world_time=3, entities=dict(ents))
    r = run_smrti_pulse(canon, "wen", "I row", 3, "p3", mems, C(), None, players=frozenset({"mia"}),
                        view={"darsana": None, "becomings": {}, "world_data": {"sandhi": True}})
    k = Kosha(canon, mems + r.memories_written, r.records_written)
    refused = [x for e, f in events if e == "smrti.move_refused" for x in f["refused"]]
    assert [(x["entity"], x["reason"]) for x in refused] == [("mia", "player_not_actor")]
    assert k.container_at("mia", 4) == "inn" and len(_made(r, "a rowboat")) == 1 and not k.holds(_made(r, "a rowboat")[0].id, 4)


def test_nothing_with_a_full_stop_and_empty_words_keep_their_old_readings(monkeypatch) -> None:
    r, k, refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": "Nothing."}, {"entity": "mug", "into": "   "}]})
    assert not r.new_entities and not undeclared
    assert [(x["entity"], x["reason"]) for x in refused] == [("key", "unknown_container")]     # said as a word: refused, as before
    assert k.container_at("mug", 4) is None                                                    # empty: in nothing, as the contract says


def test_unicode_words_make_a_container(monkeypatch) -> None:
    r, _k, refused, _u = _run(monkeypatch, {"moves": [{"entity": "key", "into": "कोश के भीतर"}]})
    assert len(_made(r, "कोश के भीतर")) == 1 and not refused


def test_a_new_marker_inside_words_is_read_as_before(monkeypatch) -> None:
    """`new chest` and `a brand new` were always read as NEW markers (Vadhu tick 211): pre-existing, pinned so a change to it is seen."""
    r, _k, _refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": "new chest"}]})
    assert [e.labels[0] for e in r.new_entities] == ["chest"] and not undeclared


# ---- found by Tier 0 and closed by the guard on what the engine makes from words (Marc, 2026-10-15: "do A") ----

def test_a_truncated_handle_of_an_existing_person_is_not_made_a_second_one(monkeypatch) -> None:
    """Vadhu, 2026-10-05 (training log 3033): the Narrator wrote `marren-innkeeper` for `marren-who-keeps-the-inn`. The old engine dropped the move; the first cut
    of the branch made a thing labelled `marren-innkeeper` and placed it in the inn, a stray second Marren. A hyphened or underscored token is a copy of an id,
    so one with no row is a slip: not made, counted `smrti.move_not_made`, the move dropped as before."""
    r, k, _refused, _u = _run(monkeypatch, {"moves": [{"entity": "marren-innkeeper", "into": "inn"}]}, extra_entities=PEOPLE)
    assert not r.new_entities and k.container_at("marren-who-keeps-the-inn", 4) == "inn"


def test_what_is_not_words_is_refused_as_it_was_before(monkeypatch) -> None:
    cases = ["marren-innkeeper", "!!!", "x" * 61, "one two three four five six seven eight nine", "the coffer, its lid cracked",
             "jakal_ash_staff_510bb1, jakal_spellbook_f8aa73", "jar_7"]
    for said in cases:
        r, _k, refused, _u = _run(monkeypatch, {"moves": [{"entity": "key", "into": said}]})
        assert not r.new_entities, said
        assert [x["reason"] for x in refused] == ["unknown_container"], said


def test_the_refusal_reasons_reach_the_witness(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **kw: seen.append((e, kw)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **kw: None)
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", False)
    ents, mems = _world()
    ents.update(PEOPLE)

    class C:
        def complete_with_tool(self, system, user, **kw):
            return {"narration": "x", "perceiver_tuples": [{"perceiver": "wen", "text": "I did it.", "salience": "ordinary"}], "scene_tuples": [],
                    "suggested_next_action": "", "moves": [{"entity": "marren-innkeeper", "into": "inn"}, {"entity": "key", "into": "!!!"}]}
    run_smrti_pulse(Maya(world_time=3, entities=dict(ents)), "wen", "I did it.", 3, "p3", mems, C(), None)
    assert sorted((f["said"], f["role"], f["why"]) for e, f in seen if e == "smrti.move_not_made") == [("!!!", "container", "no_letter"), ("marren-innkeeper", "thing", "handle_like")]


def test_a_sentence_or_a_very_long_string_is_not_a_container_name(monkeypatch) -> None:
    for said in ("the cedar coffer that stood by the north wall under the window where the old man had slept, its lid cracked.", "x" * 400):
        r, _k, _refused, _u = _run(monkeypatch, {"moves": [{"entity": "key", "into": said}]})
        assert not r.new_entities


def test_words_with_no_letter_in_them_are_not_a_container(monkeypatch) -> None:
    r, _k, _refused, _u = _run(monkeypatch, {"moves": [{"entity": "key", "into": "!!!"}, {"entity": "mug", "into": "???"}]})
    assert not r.new_entities


def test_two_phrases_that_differ_only_after_the_48th_character_are_two_containers(monkeypatch) -> None:
    """The id used to be cut at 48 characters, so two phrases with the same first 48 were one container. Within the bound (60 characters, 8 words) they are two."""
    one, two = "weathered oak coffer beneath the northern window east", "weathered oak coffer beneath the northern window west"
    assert one[:48] == two[:48] and len(one) <= 60 and len(one.split()) <= 8
    r, _k, _refused, _u = _run(monkeypatch, {"moves": [{"entity": "key", "into": one}, {"entity": "mug", "into": two}]})
    assert len(r.new_entities) == 2


def test_the_bounds_are_sixty_characters_and_eight_words_and_an_abbreviation_is_not_a_sentence(monkeypatch) -> None:
    ok = ["one two three four five six seven eight", "a" * 60, "Dr. Armitage's desk", "pouch", "St. Cuthbert's cell"]
    for said in ok:
        r, _k, _refused, _u = _run(monkeypatch, {"moves": [{"entity": "key", "into": said}]})
        assert len(r.new_entities) == 1, said


def test_the_narrators_own_row_is_found_by_its_words_with_a_hyphen_for_an_underscore(monkeypatch) -> None:
    """A move that says `tin-cup` for the row NEW:tin_cup is onto the row: exact, the same slug (the fold that keeps log 1866's repair, for hyphen worlds too)."""
    r, k, refused, undeclared = _run(monkeypatch, {"new_entities": [{"id": "NEW:tin_cup", "name": "a rusted tin cup", "in": "inn", "scene": False}],
                                                   "moves": [{"entity": "tin-cup", "into": "wen"}]})
    cup = _made(r, "a rusted tin cup")
    assert len(cup) == 1 and len(r.new_entities) == 1 and not refused and not undeclared and k.container_at(cup[0].id, 4) == "wen"
