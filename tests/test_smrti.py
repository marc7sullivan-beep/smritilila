# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""smrti -- the v3 substrate: s7 retrieval over tuples."""

from __future__ import annotations

import pytest

import re
import json

import numpy as np

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.smrti.context import Smrti, Tuple, retrieve
from smritilila.smrti.log import build_log


def _log() -> Smrti:
    log = Smrti()
    seq = 0
    def add(p, t, text, place=None, sal=4, emb=None):
        nonlocal seq
        seq += 1
        log.append(Tuple(p, seq, t, text, sal, place, emb))
    add("house", 0, "Family Town House on the Palatine", "house")
    add("curia", 0, "The Curia Julia", "curia")
    add("senator", 0, "Quintus Marcius Rufus")
    add("wife", 0, "his wife")
    add("labienus", 0, "Titus Labienus")
    add("wife", 5, "I take his arm in the garden court.", "house")
    add("senator", 5, "My wife takes my arm; the fountain murmurs.", "house")
    add("labienus", 6, "I wait on the Curia floor for the session.", "curia")
    for i in range(7, 30):
        add("senator", i, f"An ordinary evening at home, tick {i}.", "house")
    return log


def test_structural_items_always_supplied_and_rooms_are_prose() -> None:
    ctx = retrieve(_log(), "senator", 30, None, k_place=30, k_sim=0)
    assert ctx is not None and ctx.place == "house"
    ids = {t.perceiver for t in ctx.tuples}
    # the actor's last tuple, the place's FIRST tuple (its name), the place's last-k, attention's last tuples
    assert "house" in ids and any(t.memory == "Family Town House on the Palatine" for t in ctx.tuples)
    assert any(t.chronology == 29 and t.perceiver == "senator" for t in ctx.tuples)
    assert "wife" in ids and "wife" in ctx.present
    # nothing from a place the actor is not in
    assert "labienus" not in ids
    # and with a short k the wife, last seen 25 ticks ago, is no longer "present"
    short = retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0)
    assert "wife" not in short.present


def test_a_place_witnesses_a_departure_and_nothing_after() -> None:
    """Vadhu tick 283: the river town inn's window still held the three
    guests from the night they lodged there; item 3 handed its innkeeper
    their last tuples at the crossing, a day's walk off, and she narrated
    Ilo pressing Sela at her own hearth."""
    log = Smrti()
    seq = 0
    def add(p, t, text, place=None, sal=4):
        nonlocal seq
        seq += 1
        log.append(Tuple(p, seq, t, text, sal, place, None))
    add("rinn", 0, "the river town inn", "rinn"); add("cinn", 0, "the inn at the crossing", "cinn"); add("yard", 0, "the well yard", "yard")
    add("keeper", 0, "the river town innkeeper"); add("ilo", 0, "Ilo, the ferryman"); add("marren", 0, "Marren, who keeps the inn")
    add("ilo", 10, "I took a bed at the river town inn.", "rinn")
    add("keeper", 10, "An old man took a bed.", "rinn")
    add("rinn", 10, "An old man lodged.", "rinn")
    add("ilo", 11, "I left the river town at first light.", "rinn")
    add("keeper", 11, "The old man left at first light; he said someone named Marren keeps a ferry.", "rinn")
    add("rinn", 11, "The old man left.", "rinn")
    add("marren", 12, "I kneaded bread at the crossing.", "cinn")
    add("ilo", 40, "I told Sela I knew she was lying, at the crossing.", "cinn")
    add("marren", 40, "I heard Ilo press the pedlar woman.", "cinn")
    ctx = retrieve(log, "keeper", 41, None, k_place=12, k_sim=0)
    got = {(t.perceiver, t.chronology) for t in ctx.tuples}
    assert ("ilo", 0) in got and ("ilo", 40) not in got          # who he is, not what he is doing elsewhere
    assert ("marren", 40) not in got    # not her hearth (item 5's names need a declared name, v5 §16: test_kosha.py)
    # v5 §26 item 3: the NAME of a scene someone in attention has been in, and nothing of its record
    assert all(t.chronology == 0 for t in ctx.tuples if t.perceiver == "cinn")
    # tick 154's shape still holds: a step out of here is witnessed, and the place stepped to is named
    add("ilo", 42, "I came back to the river town inn.", "rinn")
    add("rinn", 42, "The old man came back.", "rinn")
    add("ilo", 43, "I stepped out to the well yard.", "yard")
    add("rinn", 43, "The old man stepped out to the yard.", "rinn")  # the moving pulse's departure tuple
    ctx = retrieve(log, "keeper", 44, None, k_place=12, k_sim=0)
    got = {(t.perceiver, t.chronology) for t in ctx.tuples}
    assert ("ilo", 43) in got and ("yard", 0) in got


def test_item_5_is_bounded_by_the_actors_own_names_not_the_worlds() -> None:
    log = _log()
    # a one-word generic ("house") does not match a two-word label on its own
    ctx = retrieve(log, "senator", 30, None, k_place=3, k_sim=0, action="Stays at the house.")
    assert "curia" not in {t.perceiver for t in ctx.tuples}


def test_similarity_fill_uses_embeddings_when_present() -> None:
    log = Smrti()
    log.append(Tuple("a", 1, 0, "a", 4, None, None))
    log.append(Tuple("a", 2, 1, "the horse was lost at the Circus", 2, "p", (1.0, 0.0)))
    log.append(Tuple("a", 3, 2, "supper in the garden", 4, "p", (0.0, 1.0)))
    log.append(Tuple("a", 4, 3, "an ordinary night", 6, "p", (0.0, 1.0)))
    ctx = retrieve(log, "a", 4, np.asarray([1.0, 0.0]), k_place=1, k_sim=1)
    assert any("horse" in t.memory for t in ctx.tuples)


def _canon() -> Maya:
    canon = Maya(world_time=10, entities={
        "hill": Entity("hill", ["The Palatine Hill"], 0, "macro_seed", avarana="district"),
        "house": Entity("house", ["Family Town House on the Palatine"], 0, "minted_geometry", avarana="building"),
        "court": Entity("court", ["the garden court"], 0, "promoted_path2", avarana="building"),
        "senator": Entity("senator", ["Quintus Marcius Rufus"], 0, "genesis"),
        "wife": Entity("wife", ["his wife"], 0, "genesis"),
        "gnaeus": Entity("gnaeus", ["old Gnaeus"], 0, "genesis"),
    })
    return canon


def _mem(i, who, t, text, loc="house") -> Memory:
    return Memory(f"m{i}", who, frozenset(), text, "PRIMARY", 3, loc, t, f"p{t}", None)


class _FakeClient:
    def __init__(self, out: dict) -> None:
        self.out = out
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, *, max_tokens=8192):  # pragma: no cover
        raise NotImplementedError

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
        self.calls.append((system, user))
        return self.out


def test_run_smrti_pulse_appends_one_tuple_per_perceiver_and_one_per_place(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    mems = [_mem(1, "senator", 5, "I find my wife in the garden court."), _mem(2, "wife", 5, "He comes to me in the court.")]
    client = _FakeClient({
        "narration": "You tell her.",
        "perceiver_tuples": [
            {"perceiver": "senator", "text": "I tell my wife about Gnaeus.", "salience": 2},
            {"perceiver": "wife", "text": "He tells me about the old groom.", "salience": 3},
            {"perceiver": "NEW:household_slave_at_door", "text": "The master speaks low to the mistress; I keep to the door.", "salience": 6},
        ],
        "scene_tuples": [{"scene": "house", "text": "The senator and his wife talk in the garden court; a slave waits at the door.", "present": ["senator", "wife", "NEW:household_slave_at_door"]}],
        "suggested_next_action": "Ask her what she would do.",
    })
    ids = iter(["slave1", "m1", "m2", "m3", "m4"])
    r = run_smrti_pulse(canon, "senator", "Tells my wife about Gnaeus.", 6, "p6", mems, client, None, id_factory=lambda: next(ids))
    assert r.narration == "You tell her." and r.suggested_next_action
    sid = "household_slave_at_door_slave1"
    assert [e.id for e in r.new_entities] == [sid]
    assert canon.entities[sid].labels == ["household slave at door"]  # v5 §9: no kinds at mint
    assert [m.entity_id for m in r.memories_written] == ["senator", "wife", sid, "house"]
    assert all(m.location_id == "house" for m in r.memories_written)
    assert r.memories_written[-1].tier == "AMBIENT" and r.memories_written[0].tier == "PRIMARY"
    assert r.memories_written[0].subject_ids == frozenset({"senator", "wife", sid})
    # the prompt carried the opaque-id contract and the action
    system, user = client.calls[0]
    assert "OPAQUE IDENTIFIER" in system and "ACTION: Tells my wife about Gnaeus." in user


def test_a_new_handle_for_someone_shown_folds_by_the_name_it_declares(monkeypatch) -> None:
    """A NEW written for someone already shown folds onto them by the name it
    declares (v5 §16), not by its label's words: v5 §17 retired that fold, and in
    all logged play it folded once -- the Crossing's donkey into its owner."""
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    mems = [_mem(1, "senator", 5, "I speak with old Gnaeus in the garden court."), _mem(2, "gnaeus", 5, "The master speaks to me.")]
    client = _FakeClient({
        "narration": "x",
        "new_entities": [{"id": "NEW:old_gnaeus_02", "name": "Gnaeus"}],
        "perceiver_tuples": [{"perceiver": "senator", "text": "I nod to old Gnaeus across the court.", "salience": 4},
                             {"perceiver": "NEW:old_gnaeus_02", "text": "The master nods to me across the court.", "salience": 5}],
        "scene_tuples": [{"scene": "house", "text": "Two men exchange a nod in the garden court.", "present": ["senator", "NEW:old_gnaeus_02"]}],
        "suggested_next_action": "",
    })
    r = run_smrti_pulse(canon, "senator", "Nods to Gnaeus.", 6, "p6", mems, client, None)
    assert r.new_entities == [] and r.resolved_new == {"NEW:old_gnaeus_02": "gnaeus"}
    assert [m.entity_id for m in r.memories_written] == ["senator", "gnaeus", "house"]


def test_an_autonomous_pulse_supplies_no_action_and_says_so(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    from smritilila.smrti.narrator import AUTONOMOUS_ACTION
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    client = _FakeClient({"narration": "x", "perceiver_tuples": [{"perceiver": "wife", "text": "I see to the household accounts by the fountain.", "salience": 6}],
                          "scene_tuples": [{"scene": "house", "text": "A quiet afternoon in the garden court.", "present": ["wife"]}], "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "wife", None, 6, "p6", [_mem(1, "wife", 5, "The court is quiet.")], client, None)
    assert AUTONOMOUS_ACTION in client.calls[0][1] and r.memories_written


def test_item_6_the_actors_recent_places_are_named() -> None:
    log = _log()
    seq = 1000
    # the senator walks out to the street; his last tuples are there, the house is where he lately was
    log.append(Tuple("street", seq, 0, "the street below the house", 6, "street", None))
    log.append(Tuple("senator", seq + 1, 30, "I step out into the street.", 4, "street", None))
    ctx = retrieve(log, "senator", 31, None, k_place=3, k_sim=0, action="Walks back home.")
    assert ctx.place == "street"
    assert any(t.perceiver == "house" and t.memory == "Family Town House on the Palatine" for t in ctx.tuples)


def test_genesis_builds_entities_and_first_tuples_and_no_relations() -> None:
    from smritilila.smrti.genesis import build
    spec = {
        "name": "t", "setting": "s", "scene_owner": "a", "playable": ["a"], "entry_points": ["p"],
        "places": {"p": {"label": "the landing", "kind": "open", "tuple": "A landing."}},
        "people": {"a": {"label": "A", "place": "p", "tuples": ["I came here.", "I am here now."]}},
    }
    ents, mems, config = build(spec, lambda text: [0.0])
    earth = config["earth"]                                                       # born with its earth (2026-09-30)
    ents, mems = [e for e in ents if e.id != earth], [m for m in mems if m.entity_id != earth]
    assert all(e.origin == "genesis" for e in ents)  # the planter writes no kinds
    place = next(e for e in ents if e.avarana is not None); actor = next(e for e in ents if e.avarana is None)
    assert place.avarana == "scene"  # a v3 kind plants as the scene bit (2026-10-03)
    assert [(m.entity_id, m.world_time, m.tier) for m in mems] == [(place.id, 0, "AMBIENT"), (actor.id, -10, "PRIMARY"), (actor.id, 0, "PRIMARY")]
    assert all(m.location_id == place.id for m in mems)
    assert config["scene_owner_entity_id"] == actor.id and config["entry_point_location_ids"] == [place.id]


def test_worlds_registry_ignores_a_key_it_no_longer_knows(tmp_path) -> None:
    """smrti_mode was removed 2026-09-22; a worlds.json still carrying it
    must load, not raise -- someone else's file is not ours to migrate."""
    from smritilila.worlds_registry import load_worlds_registry
    p = tmp_path / "worlds.json"
    p.write_text('{"worlds": [{"name": "a", "database_url": "x"}, {"name": "b", "database_url": "y", "smrti_mode": "live"}]}', encoding="utf-8")
    a, b = load_worlds_registry(p)
    assert (a.name, a.database_url) == ("a", "x") and (b.name, b.database_url) == ("b", "y")


def test_present_means_whose_own_last_tuple_is_here() -> None:
    log = _log()
    # the wife walked out: her last tuple is elsewhere, though the house's window still holds her tick-5 tuple
    log.append(Tuple("street", 900, 0, "the street", 6, "street", None))
    log.append(Tuple("wife", 901, 29, "I go out into the street.", 4, "street", None))
    ctx = retrieve(log, "senator", 30, None, k_place=40, k_sim=0)
    assert "wife" not in ctx.present
    assert any(t.perceiver == "wife" and t.chronology == 29 for t in ctx.tuples)  # but her last tuple is still shown


def test_a_perceiver_no_place_tuple_lists_stays_where_their_own_stream_has_them(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    canon.entities["yard"] = Entity("yard", ["the yard"], 0, "genesis", avarana="open")
    mems = [_mem(1, "senator", 5, "I cross to the yard."), _mem(2, "wife", 5, "I watch him go from the court.")]
    client = _FakeClient({
        "narration": "x",
        "perceiver_tuples": [{"perceiver": "senator", "text": "I cross the court and reach the yard.", "salience": 4},
                             {"perceiver": "wife", "text": "From the court I watch him reach the yard.", "salience": 5}],
        "scene_tuples": [{"scene": "yard", "text": "A man arrives in the yard from the house.", "present": ["senator"]}],  # only the arrival
        "suggested_next_action": "",
    })
    r = run_smrti_pulse(canon, "senator", "Crosses to the yard.", 6, "p6", mems, client, None)
    by = {m.entity_id: m.location_id for m in r.memories_written}
    assert by["senator"] == "yard" and by["wife"] == "house"  # she stayed where her own stream has her


def test_a_garbled_id_resolves_to_the_one_shown_id_it_is_near() -> None:
    from smritilila.smrti.pulse import nearest_id
    shown = {"e5cfa5652a244f2f878cae7b41509d07", "05f00b495b534e19a3acb53948926af5"}
    assert nearest_id("e5cfa5652a204f2f878cae7b41509d07", shown) == "e5cfa5652a244f2f878cae7b41509d07"
    assert nearest_id("ffffffffffffffffffffffffffffffff", shown) is None
    assert nearest_id("teo", shown) is None


def test_minted_ids_carry_the_mnemonic(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    client = _FakeClient({"narration": "x", "perceiver_tuples": [{"perceiver": "senator", "text": "I see a drover with a mule at the gate.", "salience": 4},
                                                                  {"perceiver": "NEW:drover_with_a_mule", "text": "The man at the gate looks at me a moment.", "salience": 5}],
                          "scene_tuples": [{"scene": "house", "text": "Two men stand at the gate of the house.", "present": ["senator", "NEW:drover_with_a_mule"]}], "suggested_next_action": ""})
    ids = iter(["abcdef0123456789", "m1", "m2", "m3"])
    r = run_smrti_pulse(canon, "senator", "Looks.", 6, "p6", [_mem(1, "senator", 5, "I wait.")], client, None, id_factory=lambda: next(ids))
    assert r.new_entities[0].id == "drover_with_a_mule_abcdef"


def test_every_perceiver_in_attention_brings_their_first_tuple_too() -> None:
    ctx = retrieve(_log(), "senator", 30, None, k_place=30, k_sim=0)
    assert any(t.perceiver == "wife" and t.memory == "his wife" for t in ctx.tuples)  # who she is
    assert any(t.perceiver == "wife" and t.chronology == 5 for t in ctx.tuples)       # what she last did


def test_malformed_narrator_rows_are_dropped_not_fatal(monkeypatch) -> None:
    from smritilila.smrti.narrator import narrate
    monkeypatch.setattr("smritilila.smrti.narrator.log_event", lambda event, **f: None)
    client = _FakeClient({"narration": "x", "perceiver_tuples": ["just a string", {"perceiver": "a", "text": "The court is quiet and the fountain runs.", "salience": 4}],
                          "scene_tuples": [{"place": "p"}], "suggested_next_action": None})
    ctx = retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0)
    out = narrate(client, ctx, "waits", set())
    assert [p["perceiver"] for p in out.perceiver_tuples] == ["a"] and out.place_tuples == [] and out.suggested_next_action == ""


def test_a_truncated_tail_still_yields_its_list_and_the_place_tuples_before_the_cut(monkeypatch) -> None:
    # Vadhu tick 116: same shape as 106, the string cut short inside the
    # last value; place_tuples present but EMPTY at the top level
    from smritilila.smrti.narrator import narrate
    events = []
    monkeypatch.setattr("smritilila.smrti.narrator.log_event", lambda event, **f: events.append((event, f)))
    tail = ('[{"perceiver": "a", "text": "I put down the comb and went to the tin-man myself.", "salience": 3}], '
            '"scene_tuples": [{"scene": "p", "text": "The old ferryman rose from the stair and crossed the room.", "present": ["a"]}], '
            '"suggested_next_action": "Wait on the answer, or press him once mo')
    client = _FakeClient({"narration": "x", "perceiver_tuples": tail, "scene_tuples": [], "sara": "You ask."})
    out = narrate(client, retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0), "waits", set())
    assert [p["perceiver"] for p in out.perceiver_tuples] == ["a"] and out.place_tuples[0]["present"] == ["a"]
    assert out.suggested_next_action.startswith("Wait on the answer") and out.sara == "You ask."
    assert events[0] == ("smrti.shape_recovered", {"field": "perceiver_tuples", "filled": ["perceiver_tuples", "scene_tuples", "suggested_next_action"]})
    # cut inside the list itself: the rows that closed survive, nothing fatal
    cut = '[{"perceiver": "a", "text": "I put down the comb and went to the tin-man myself.", "salience": 3}, {"perceiver": "b", "text": "I saw him cro'
    out = narrate(_FakeClient({"narration": "x", "perceiver_tuples": cut, "scene_tuples": []}), retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0), "waits", set())
    assert [p["perceiver"] for p in out.perceiver_tuples] == ["a"] and out.place_tuples == []
    # not a list at all: dropped, not fatal
    out = narrate(_FakeClient({"narration": "x", "perceiver_tuples": "no rows here", "scene_tuples": []}), retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0), "waits", set())
    assert out.perceiver_tuples == []


def test_a_list_field_that_arrives_as_the_tail_of_the_object_is_unfolded(monkeypatch) -> None:
    # Vadhu tick 106: perceiver_tuples was a string holding the rest of
    # the object, place_tuples absent at the top level
    from smritilila.smrti.narrator import narrate
    events = []
    monkeypatch.setattr("smritilila.smrti.narrator.log_event", lambda event, **f: events.append((event, f)))
    tail = ('[{"perceiver": "a", "text": "I put down the comb and went to the tin-man myself.", "salience": 3}], '
            '"scene_tuples": [{"scene": "p", "text": "The old ferryman rose from the stair and crossed the room.", "present": ["a"]}], '
            '"suggested_next_action": "Wait for the answer."')
    client = _FakeClient({"narration": "x", "perceiver_tuples": tail})
    out = narrate(client, retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0), "waits", set())
    assert [p["perceiver"] for p in out.perceiver_tuples] == ["a"]
    assert out.place_tuples[0]["present"] == ["a"] and out.suggested_next_action == "Wait for the answer."
    assert events == [("smrti.shape_recovered", {"field": "perceiver_tuples", "filled": ["perceiver_tuples", "scene_tuples", "suggested_next_action"]})]
    # a plain stringified list unfolds too; garbage is dropped, not fatal
    client = _FakeClient({"narration": "x", "perceiver_tuples": json.dumps([{"perceiver": "a", "text": "The court is quiet and the fountain runs.", "salience": 4}]), "scene_tuples": "not json at all"})
    out = narrate(client, retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0), "waits", set())
    assert [p["perceiver"] for p in out.perceiver_tuples] == ["a"] and out.place_tuples == []


def test_a_bare_head_noun_in_the_action_does_not_reach_a_place_the_actor_never_knew() -> None:
    # Vadhu tick 176: "find the market" in a river town pulled the hill market into context
    log = Smrti()
    log.append(Tuple("court", 0, 0, "the garden court", 5, "court"))
    log.append(Tuple("hill_market", 1, 1, "hill market", 5, "hill_market"))
    log.append(Tuple("senator", 2, 2, "I walked all morning and came over a rise above a town.", 4, "court"))
    ctx = retrieve(log, "senator", 3, None, k_place=3, k_sim=0, action="find the market and ask after her")
    assert "hill_market" not in {t.perceiver for t in ctx.tuples}
    # ... but the place his own last tuple stands in is where he is, whatever the action says
    log.append(Tuple("senator", 3, 3, "I sold wool at the market yesterday.", 4, "hill_market"))
    ctx = retrieve(log, "senator", 4, None, k_place=3, k_sim=0, action="find the market and ask after her")
    assert "hill_market" in {t.perceiver for t in ctx.tuples}


def test_what_can_perceive_is_the_worlds_to_say(monkeypatch) -> None:
    # s0 (Marc, 2026-09-19): an engine rule that a thing is never a perceiver was withdrawn the day it
    # was written -- a world whose wind has a stream keeps the wind's tuples like anyone's
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    canon.entities["wind"] = Entity("wind", ["the wind off the river"], 0, "genesis")
    mems = [_mem(1, "senator", 5, "I stood on the open grass and asked the unseen.", loc="house"), _mem(2, "wind", 5, "I came up off the water.", loc="house")]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "senator", "text": "I asked which way, and the wind turned into my face.", "salience": "weighty"},
                                {"perceiver": "wind", "text": "I turned and brought him the smell of water.", "salience": "passing"}],
           "scene_tuples": [{"scene": "house", "text": "A man stood asking; the wind shifted.", "present": ["senator", "wind"]}],
           "suggested_next_action": ""}
    r = run_smrti_pulse(canon, "senator", "asks the unseen", 6, "p6", mems, _FakeClient(out), None)
    assert {m.entity_id for m in r.memories_written} == {"senator", "wind", "house"}
    assert all("wind" in m.subject_ids for m in r.memories_written)


def test_a_player_is_perceived_never_voiced_in_anothers_pulse(monkeypatch) -> None:
    # s19, the crossing's tick 244: Ilo's turn wrote Teo's tuple "I have a brother"
    from smritilila.smrti.pulse import run_smrti_pulse
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    canon = _canon()
    mems = [_mem(1, "senator", 5, "I sat by the fire.", loc="house"), _mem(2, "wife", 5, "I sat beside him.", loc="house")]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "wife", "text": "I told him at last that there had been a son.", "salience": "indelible"},
                                {"perceiver": "senator", "text": "She told me there was a son; I could not answer.", "salience": "indelible"}],
           "scene_tuples": [{"scene": "house", "text": "By the fire the woman told the man of a son.", "present": ["wife", "senator"]}],
           "suggested_next_action": ""}
    # the wife's pulse: the player (senator) is present in the place tuple; the Narrator's tuple FOR him is
    # refused (his interior is his) -- and what he witnessed, the place's record, is filed as his own (2026-09-21)
    r = run_smrti_pulse(canon, "wife", None, 6, "p6", mems, _FakeClient(out), None, players=frozenset({"senator"}))
    assert [m.entity_id for m in r.memories_written] == ["wife", "house", "senator"]
    witnessed = r.memories_written[-1]
    assert witnessed.experience == "By the fire the woman told the man of a son." and witnessed.tier == "PRIMARY" and witnessed.location_id == "house"
    assert witnessed.subject_ids == frozenset({"wife", "senator"}) and witnessed.world_time == 6
    assert not any(m.entity_id == "senator" and "could not answer" in m.experience for m in r.memories_written)
    assert ("smrti.player_not_voiced", {"pulse_id": "p6", "player": "senator", "actor": "wife"}) in events
    assert ("smrti.player_witnessed",
            {"pulse_id": "p6", "player": "senator", "place": "house", "use": "none"}) in events
    # the player's own pulse: his tuple is his
    r = run_smrti_pulse(canon, "senator", "sits with it", 7, "p7", mems, _FakeClient(out), None, players=frozenset({"senator"}))
    assert {m.entity_id for m in r.memories_written} == {"wife", "senator", "house"}


def test_the_new_marker_is_read_wherever_the_narrator_put_it_and_a_bare_mnemonic_is_new() -> None:
    from smritilila.smrti.pulse import new_handle, run_smrti_pulse
    assert new_handle("river_town_low_hills_track_NEW") == "NEW:river_town_low_hills_track"
    assert new_handle("NEW_low_hills") == "NEW:low_hills" and new_handle("new: low hills") == "NEW:low hills" and new_handle("NEW:x") == "NEW:x"
    assert new_handle("e5cfa5652a244f2f878cae7b41509d07") == "e5cfa5652a244f2f878cae7b41509d07"
    events = []
    canon = _canon()
    import smritilila.smrti.pulse as P
    P.log_event = lambda event, **f: events.append((event, f))
    client = _FakeClient({"narration": "x",
                          "perceiver_tuples": [{"perceiver": "senator", "text": "I climbed into the low hills along her line.", "salience": "notable"}],
                          "scene_tuples": [{"scene": "house", "text": "The man left the house for the hills.", "present": []},
                                           {"scene": "low_hills_track_NEW", "kind": "open", "text": "A track climbs between gorse and stone.", "present": ["senator"]}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "senator", "climbs", 6, "p6", [_mem(1, "senator", 5, "I stood in the house.", loc="house")], client, None)
    assert [e.labels[0] for e in r.new_entities] == ["low hills track"] and r.dropped == []
    assert {m.entity_id: m.location_id for m in r.memories_written}["senator"] == r.new_entities[0].id
    # a bare mnemonic, unmarked, is new; a garbled hex id is still dropped
    client = _FakeClient({"narration": "x", "perceiver_tuples": [{"perceiver": "senator", "text": "I sat a while and watched the hills.", "salience": "passing"}],
                          "scene_tuples": [{"scene": "gorse_hollow", "kind": "open", "text": "A hollow in the gorse above the track, out of the wind.", "present": ["senator", "0123456789abcdef0123456789abcdef"]}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "senator", "sits", 7, "p7", [_mem(1, "senator", 5, "I stood in the house.", loc="house")], client, None)
    assert [e.labels[0] for e in r.new_entities] == ["gorse hollow"] and r.dropped == ["0123456789abcdef0123456789abcdef"]
    assert ("smrti.bare_mnemonic", {"pulse_id": "p7", "handle": "gorse_hollow"}) in events


def test_a_perceiver_in_attention_brings_the_name_of_the_place_they_now_stand_in() -> None:
    # Vadhu tick 154: Ilo out at the well yard; Marren's turn minted a second yard
    log = Smrti()
    log.append(Tuple("house", 0, 0, "the inn", 5, "house"))
    log.append(Tuple("yard", 1, 1, "the well yard", 5, "yard"))
    log.append(Tuple("marren", 2, 2, "Marren, who keeps the inn", 5, "house"))
    log.append(Tuple("ilo", 3, 3, "Ilo, the ferryman", 5, "house"))
    log.append(Tuple("house", 4, 4, "The ferryman sat by the fire, then said he'd walk out to the well.", 5, "house"))
    log.append(Tuple("marren", 5, 4, "I let him go out to the well.", 4, "house"))
    log.append(Tuple("ilo", 6, 4, "I stepped out to the well to have something to do with my hands.", 4, "yard"))
    ctx = retrieve(log, "marren", 5, None, k_place=3, k_sim=0)
    assert ("yard", 1) in {(t.perceiver, t.seq) for t in ctx.tuples}


def test_the_narrator_weighs_a_memory_in_words_and_the_store_keeps_the_number(monkeypatch) -> None:
    from smritilila.smrti.narrator import SALIENCE_NAMES, render, salience_of
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    assert salience_of("indelible") == 1 and salience_of("Habitual") == 7 and salience_of(3) == 3 and salience_of("9") == 7
    assert salience_of(None) == 4 and salience_of("vivid") == 4 and salience_of(True) == 4
    canon = _canon()
    client = _FakeClient({"narration": "x",
                          "perceiver_tuples": [{"perceiver": "senator", "text": "I said her name at last, and it changed me.", "salience": "indelible"},
                                               {"perceiver": "wife", "text": "I heard him say it and kept kneading the dough.", "salience": "passing"}],
                          "scene_tuples": [{"scene": "house", "text": "A name was spoken by the fire.", "present": ["senator", "wife"]}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "senator", "says her name", 6, "p6", [_mem(1, "senator", 5, "I sat by the fire.", loc="house"), _mem(2, "wife", 5, "I kneaded.", loc="house")], client, None)
    assert {m.entity_id: m.salience for m in r.memories_written if m.tier == "PRIMARY"} == {"senator": 1, "wife": 6}
    # and the Narrator is shown the words, not the numbers
    ctx = retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0)
    text = render(ctx, "waits", set())
    assert all(SALIENCE_NAMES[n] in text for n in {t.salience for t in ctx.tuples}) and ", 3]" not in text


def test_the_actors_own_most_salient_tuples_are_always_present() -> None:
    # Vadhu tick 180: the name said at 167 (salience 2) absent, said again differently
    log = Smrti()
    log.append(Tuple("court", 0, 0, "the garden court", 5, "court"))
    log.append(Tuple("senator", 1, 1, "Quintus, a senator", 5, "court"))
    log.append(Tuple("senator", 2, 2, "I said her name at last -- Sela.", 2, "court"))
    for i in range(3, 40):
        log.append(Tuple("senator", i, i, f"I walked and looked at the stalls, tick {i}.", 4, "court"))
    log.append(Tuple("senator", 40, 40, "I built the shrine myself, the winter after.", 1, "court"))
    ctx = retrieve(log, "senator", 41, None, k_place=3, k_sim=0)
    mine = {t.chronology for t in ctx.tuples if t.perceiver == "senator"}
    assert {2, 40} <= mine and len(mine) <= 2 + 2 + 6 * (1 + 2)  # first, last, at most OWN_SALIENT more, each followed WEIGHT_FOLLOW_UP


def test_an_own_tuple_retrieved_brings_the_one_that_followed_it() -> None:
    # Vadhu tick 139: the question at 101 in context, the answer at 102 not
    log = Smrti()
    log.append(Tuple("court", 0, 0, "the garden court", 5, "court", np.zeros(3)))
    q = np.array([1.0, 0.0, 0.0])
    log.append(Tuple("senator", 1, 1, "I sat by the fountain a while.", 4, "court", np.array([0.0, 1.0, 0.0])))
    log.append(Tuple("senator", 2, 2, "The boy asked me whether I had ever seen the pedlar woman.", 4, "court", q))
    log.append(Tuple("senator", 3, 3, "I told him plain: three days back, on the ridge path.", 4, "court", np.array([0.0, 0.0, 1.0])))
    for i in range(4, 12):
        log.append(Tuple("senator", i, i, f"I ate bread and watched the fire, tick {i}.", 4, "court", np.array([0.0, 1.0, 0.0])))
    ctx = retrieve(log, "senator", 12, q, k_place=3, k_sim=1)
    seqs = {t.seq for t in ctx.tuples if t.perceiver == "senator"}
    assert 2 in seqs and 3 in seqs  # the question by similarity, the answer because it followed
    # all the same weight as the rest, so nothing else brings the answer (at salience 3 both were always in, by weight, whatever similarity did)
    ctx = retrieve(log, "senator", 12, q, k_place=3, k_sim=0)
    assert 3 not in {t.seq for t in ctx.tuples if t.perceiver == "senator"}


def test_a_non_actor_the_memory_places_elsewhere_is_not_present_in_someone_elses_pulse(monkeypatch) -> None:
    # Vadhu tick 129: the tin peddler asleep at the inn, written present at
    # the hill market by the drover's turn
    from smritilila.smrti.pulse import run_smrti_pulse
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    canon = _canon()
    canon.entities["market"] = Entity("market", ["the hill market"], 0, "genesis", avarana="open")
    canon.entities["drover"] = Entity("drover", ["a drover"], 0, "genesis")
    canon.entities["peddler"] = Entity("peddler", ["a tin peddler"], 0, "genesis")
    canon.entities["companion"] = Entity("companion", ["a companion"], 0, "genesis")
    mems = [_mem(1, "market", 0, "A market on the ridge, stalls and pens.", loc="market"),
            _mem(2, "drover", 5, "I sold my wool at the market.", loc="market"),
            _mem(3, "companion", 5, "I stood with the drover among the stalls.", loc="market"),
            _mem(4, "peddler", 6, "I slept by the inn's woodpile.", loc="house")]
    client = _FakeClient({"narration": "x",
                          "perceiver_tuples": [{"perceiver": "drover", "text": "I wandered over to a spread of tin pots with my companion.", "salience": 4},
                                               {"perceiver": "companion", "text": "I followed the drover across to the tin stall.", "salience": 5},
                                               {"perceiver": "peddler", "text": "A drover came and looked over my pots a long while.", "salience": 4}],
                          "scene_tuples": [{"scene": "market", "text": "The drover and his companion crossed to a tin stall.", "present": ["drover", "companion", "peddler"]}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "drover", None, 7, "p7", mems, client, None)
    assert r.elsewhere == ["peddler"]
    assert {m.entity_id for m in r.memories_written} == {"drover", "companion", "market"}
    assert all(set(m.subject_ids) == {"drover", "companion"} for m in r.memories_written)
    assert ("smrti.elsewhere", {"pulse_id": "p7", "ids": ["peddler"], "scenes": ["market"]}) in events
    # the companion, who stood where the actor began, may move with him
    client = _FakeClient({"narration": "x",
                          "perceiver_tuples": [{"perceiver": "drover", "text": "We walked down off the ridge to the house.", "salience": 4},
                                               {"perceiver": "companion", "text": "I walked down with the drover to the house.", "salience": 4}],
                          "scene_tuples": [{"scene": "market", "text": "The drover and his companion left down the road.", "present": []},
                                           {"scene": "house", "text": "The drover and his companion came in.", "present": ["drover", "companion"]}],
                          "moves": [{"entity": "drover", "into": "house"}, {"entity": "companion", "into": "house"}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "drover", "walks to the house", 8, "p8", mems, client, None)
    assert r.elsewhere == [] and {m.entity_id: m.location_id for m in r.memories_written if m.entity_id in ("drover", "companion")} == {"drover": "house", "companion": "house"}
    # tick 157: a person steps back, unnarrated, to where they lately were
    mems2 = mems + [_mem(5, "peddler", 6, "I stepped out to the yard.", loc="market"),
                    _mem(6, "peddler", 7, "I stood in the market a while.", loc="market"),
                    _mem(7, "peddler", 8, "I went back to the house.", loc="house"),
                    _mem(8, "peddler", 9, "I sat by the fire in the house.", loc="house"),
                    _mem(9, "peddler", 10, "I stepped out to the well.", loc="market")]
    client = _FakeClient({"narration": "x",
                          "perceiver_tuples": [{"perceiver": "drover", "text": "I came in and the peddler was back at the fire.", "salience": 4},
                                               {"perceiver": "peddler", "text": "I had just come back in from the well when he arrived.", "salience": 4}],
                          "scene_tuples": [{"scene": "house", "text": "The drover came in; the peddler was back inside.", "present": ["drover", "peddler"]}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "drover", "walks in", 11, "p11", mems2, client, None)
    # v5 §10: the roster adds a move only from a neighbouring scene; the
    # peddler's own record has him at the market, and a step back must be a move
    assert r.elsewhere == ["peddler"]
    # tick 231: the actor lately came from the non-actor's place -- they could have followed
    canon.entities["crest"] = Entity("crest", ["the crest"], 0, "genesis", avarana="open")
    canon.entities["valley"] = Entity("valley", ["the valley road"], 0, "genesis", avarana="open")
    canon.entities["ilo"] = Entity("ilo", ["an old man"], 0, "genesis")
    mems3 = [_mem(1, "crest", 0, "A windy crest above a valley.", loc="crest"), _mem(2, "valley", 0, "A thin road through the valley.", loc="valley"),
             _mem(3, "ilo", 5, "I rested on the crest.", loc="crest"), _mem(4, "drover", 5, "I stood on the crest with the old man.", loc="crest"),
             _mem(5, "drover", 6, "I ran down to the valley road.", loc="valley")]
    client = _FakeClient({"narration": "x",
                          "perceiver_tuples": [{"perceiver": "drover", "text": "I climbed back and stood by the old man as she came up.", "salience": 4},
                                               {"perceiver": "ilo", "text": "The boy came back up to me and a woman climbed behind him.", "salience": 2}],
                          "scene_tuples": [{"scene": "valley", "text": "The young man climbed back to the old man; a woman followed.", "present": ["drover", "ilo"]}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "drover", "climbs back up", 7, "p7b", mems3, client, None)
    assert r.elsewhere == ["ilo"]  # v4: on the crest by his own record; the valley row cannot move him


def test_sara_rides_beside_the_narration_and_is_cleaned_like_it(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    client = _FakeClient({"narration": "You wait by the stalls a long while and the light goes.",
                          "sara": "You wait; the light goes.",
                          "suggested_next_action_from_memory": "Go and find the freedman you promised to see.",
                          "perceiver_tuples": [{"perceiver": "senator", "text": "I waited by the stalls until the light went.", "salience": 5}],
                          "scene_tuples": [{"scene": "house", "text": "A man waited by the stalls until dusk.", "present": ["senator"]}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "senator", "Waits.", 6, "p6", [_mem(1, "senator", 5, "I wait by the stalls a while.")], client, None)
    assert r.sara == "You wait; the light goes." and r.narration.startswith("You wait by the stalls")
    # the structured sara: what, a verbatim said (checked), a turn; and what the footer showed (present, at, new), now only on the result
    client = _FakeClient({"narration": "You wait. \u201cThe light is going,\u201d she says, and takes your arm. Nobody answers.",
                          "sara": {"what": "You wait; she takes your arm.", "said": "The light is going,", "turn": "nobody answers yet"},
                          "perceiver_tuples": [{"perceiver": "senator", "text": "I waited and she took my arm by the stalls.", "salience": "notable"},
                                               {"perceiver": "wife", "text": "I told him the light was going and took his arm.", "salience": "ordinary"}],
                          "scene_tuples": [{"scene": "house", "text": "A man waited; a woman took his arm.", "present": ["senator", "wife"]}],
                          "suggested_next_action": ""})
    r2 = run_smrti_pulse(canon, "senator", "Waits.", 8, "p8", [_mem(1, "senator", 5, "I wait by the stalls a while.", loc="house"),
                                                                _mem(2, "wife", 5, "I came out to the stalls.", loc="house")], client, None)
    assert (r2.sara, r2.said, r2.turn) == ("You wait; she takes your arm.", "The light is going,", "nobody answers yet")
    assert r2.scene == "house" and r2.new_entities == []
    from smritilila.smrti.kosha import Kosha
    assert Kosha(canon, [_mem(2, "wife", 5, "I came out to the stalls.", loc="house")] + r2.memories_written).occupants("house", 9, exclude={"senator"}) == {"wife"}
    # a said that is not in the narration is dropped
    client = _FakeClient({"narration": "You wait a while.", "sara": {"what": "You wait.", "said": "Come with me.", "turn": ""},
                          "perceiver_tuples": [{"perceiver": "senator", "text": "I waited by the stalls a while longer.", "salience": "passing"}],
                          "scene_tuples": [], "suggested_next_action": ""})
    r3 = run_smrti_pulse(canon, "senator", "Waits.", 9, "p9", [_mem(1, "senator", 5, "I wait by the stalls a while.", loc="house")], client, None)
    assert r3.said == ""
    assert r.suggested_next_action_from_memory == "Go and find the freedman you promised to see."
    # absent: empty, never the narration repeated
    client = _FakeClient({"narration": "You wait.", "perceiver_tuples": [], "scene_tuples": [], "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "senator", "Waits.", 7, "p7", [_mem(1, "senator", 5, "I wait by the stalls a while.")], client, None)
    assert r.sara == "" and r.suggested_next_action_from_memory == ""


def test_a_placeholder_tuple_is_not_a_memory(monkeypatch) -> None:
    from smritilila.smrti.narrator import narrate
    monkeypatch.setattr("smritilila.smrti.narrator.log_event", lambda event, **f: None)
    client = _FakeClient({"narration": "x", "perceiver_tuples": [], "scene_tuples": [{"scene": "house", "text": "arrival pending", "present": []},
                          {"scene": "house", "text": "The senator came in from the street and the household looked up.", "present": []}], "suggested_next_action": ""})
    out = narrate(client, retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0), "goes in", {"house"})
    assert len(out.place_tuples) == 1 and out.place_tuples[0]["text"].startswith("The senator")


def test_a_bare_mnemonic_in_present_resolves_to_the_handle_minted_this_pulse(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    client = _FakeClient({"narration": "x",
                          "perceiver_tuples": [{"perceiver": "senator", "text": "I see a cloth seller at her stall.", "salience": 4},
                                               {"perceiver": "NEW:cloth_seller_woman", "text": "A man stops at my stall and asks after cloth.", "salience": 5}],
                          "scene_tuples": [{"scene": "house", "text": "A man and a cloth seller talk at a stall.", "present": ["senator", "cloth_seller_woman"]}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "senator", "Looks.", 6, "p6", [_mem(1, "senator", 5, "I wait by the stalls a while.")], client, None)
    assert r.dropped == [] and r.new_entities[0].id in r.memories_written[-1].subject_ids


def test_a_missing_actor_tuple_is_the_narration(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    client = _FakeClient({"narration": "You watch the road all afternoon and nothing comes.", "perceiver_tuples": [],
                          "scene_tuples": [{"scene": "house", "text": "The court lay quiet through the afternoon.", "present": ["senator"]}], "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "senator", "Watches.", 6, "p6", [_mem(1, "senator", 5, "I sit in the court and wait.")], client, None)
    assert r.memories_written[0].entity_id == "senator" and r.memories_written[0].experience.startswith("You watch")  # v5 §32: as narrated, marked


def test_scene_of_answers_from_last_tuples_not_the_graph() -> None:
    from smritilila.smrti.kosha import Kosha
    canon = _canon()
    canon.entities["yard"] = Entity("yard", ["the yard"], 0, "genesis", avarana="open")
    mems = [_mem(1, "senator", 5, "In the court."), _mem(2, "wife", 5, "In the court too."), _mem(3, "gnaeus", 6, "Out in the yard.", loc="yard"),
            _mem(4, "wife", 7, "I go out to the yard.", loc="yard")]
    k = Kosha(canon, mems)  # where someone is: their containment, their last tuple where no record says
    assert k.scene_of("senator", 8) == "house" and k.occupants("house", 8, exclude={"senator"}) == set()  # the wife's LAST tuple is in the yard
    assert k.scene_of("gnaeus", 8) == "yard" and k.occupants("yard", 8, exclude={"gnaeus"}) == {"wife"}


def test_narration_is_cut_at_leaked_markup_and_ids_become_labels(monkeypatch) -> None:
    from smritilila.smrti.pulse import clean_narration
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    canon.entities["a955c4aed94d4179b7628bfb4a9683f7"] = Entity("a955c4aed94d4179b7628bfb4a9683f7", ["Wind on the Mountain"], 0, "genesis")
    bad = 'You walk north.</narration>\n<parameter name="perceiver_tuples">[{"perceiver": "a955c4aed94d4179b7628bfb4a9683f7"'
    assert clean_narration(bad, canon, "p") == "You walk north."
    assert clean_narration("a955c4aed94d4179b7628bfb4a9683f7 looks up.", canon, "p") == "Wind on the Mountain looks up."
    # a minted id gives its own words, never a label that is a name the reader may not have heard (v4.55's rule)
    canon.entities["grey_faced_young_man_88c865"] = Entity("grey_faced_young_man_88c865", ["Elias Lowell"], 0, "smrti")
    assert clean_narration("grey_faced_young_man_88c865 looks up.", canon, "p") == "grey faced young man looks up."


def test_a_minted_id_keeps_its_words_and_takes_a_fresh_tail() -> None:
    from smritilila.smrti.pulse import mnemonic_id
    assert mnemonic_id("river_fork_pine_country_9f3a21", "bf2c47deadbeef") == "river_fork_pine_country_bf2c47"


def test_item_6_brings_who_stands_at_a_recent_place() -> None:
    log = _log()
    log.append(Tuple("road", 900, 0, "the east road", 6, "road", None))
    log.append(Tuple("senator", 901, 25, "I walk the east road a way and turn back.", 4, "road", None))
    log.append(Tuple("senator", 902, 26, "Back at the house.", 4, "house", None))
    log.append(Tuple("peddler", 903, 0, "a tin peddler", 6, None, None))
    log.append(Tuple("peddler", 904, 27, "I come limping down the east road toward the water.", 4, "road", None))
    ctx = retrieve(log, "senator", 28, None, k_place=3, k_sim=0)
    ids = {t.perceiver for t in ctx.tuples}
    assert "road" in ids and "peddler" in ids  # the road he lately walked, and the man now standing on it


def test_a_handle_with_glued_punctuation_still_resolves(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    client = _FakeClient({"narration": "x", "perceiver_tuples": [{"perceiver": "senator", "text": "I sit with my wife in the court.", "salience": 4}],
                          "scene_tuples": [{"scene": "house", "text": "Two people sit together in the court.", "present": ["senator, ", "wife,"]}], "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "senator", "Sits.", 6, "p6", [_mem(1, "senator", 5, "I sit down in the court a while."), _mem(2, "wife", 5, "I sat.")], client, None)
    assert r.dropped == [] and r.memories_written[-1].subject_ids == frozenset({"senator", "wife"})


def test_a_thing_whose_stream_stands_here_is_shown_present(monkeypatch) -> None:
    """s0 (2026-09-20): the present-list fallback reads streams, not kinds.
    A v2 'object' whose last tuple is at the actor's place is here."""
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    canon.entities["purse"] = Entity("purse", ["his purse"], 0, "smrti")
    client = _FakeClient({"narration": "You wait a while.", "sara": "You wait.",
                          "perceiver_tuples": [{"perceiver": "senator", "text": "I waited by the stalls.", "salience": "ordinary"}],
                          "scene_tuples": [{"scene": "house", "text": "A man waited.", "present": []}],
                          "suggested_next_action": ""})
    mems = [_mem(1, "senator", 5, "I wait by the stalls a while.", loc="house"), _mem(2, "purse", 5, "I sat heavy at his belt.", loc="house")]
    r = run_smrti_pulse(canon, "senator", "Waits.", 8, "p8", mems, client, None)
    from smritilila.smrti.kosha import Kosha
    assert Kosha(canon, mems + r.memories_written).occupants("house", 9, exclude={"senator"}) == {"purse"}  # a thing that perceives is present


def test_the_road_prompt_names_no_world_and_no_sky() -> None:
    """s0 (2026-09-20): the road's turn said 'toward the crossing' in every
    world, and told the Narrator to write the weather, the hour and the
    season -- the engine asserting a sky. Only the road's own tuples now."""
    from smritilila.smrti.narrator import SCENE_ARRIVAL_ACTION, SYSTEM
    rule_b = SYSTEM[SYSTEM.index("B. A QUIET SCENE"):SYSTEM.index("C. THE WORLD AT LARGE")].lower()  # the turn's text lives in rule B now
    for low in (SCENE_ARRIVAL_ACTION.lower(), rule_b):
        assert "crossing" not in low
        assert not any(w in low for w in ("weather", "season", "the hour"))
    assert "scene's own tuples" in rule_b and "rule b" in SCENE_ARRIVAL_ACTION.lower()  # v5 §38: a quiet scene's turn, as a road's was


def test_the_second_suggestions_thread_is_chosen_by_the_engine_a_person_not_returned_to(monkeypatch) -> None:
    """Ravenmoor tick 79 and the crossing 375-408: the Narrator, choosing the
    thread itself from six indelible questions, offered ones settled twenty
    ticks back. Now the engine picks: the person the actor's weightier
    tuples are most about who is not here and not in the actor's last
    THREAD_RECENT tuples; else such a place; else nothing. The Narrator
    only phrases, from the THREAD FROM MEMORY line."""
    from smritilila.smrti.narrator import SYSTEM
    from smritilila.smrti.pulse import THREAD_RECENT, run_smrti_pulse, thread_from_memory
    assert "grounded ONLY in the THREAD FROM MEMORY line" in SYSTEM
    canon = _canon()
    canon.entities["sela"] = Entity("sela", ["Sela, a pedlar"], 0, "genesis")
    canon.entities["landing"] = Entity("landing", ["the east landing"], 0, "genesis", avarana="open")
    mems = []
    # long ago: weighty tuples with Sela at the landing; with Gnaeus, one passing tuple
    mems.append(Memory("a1", "senator", frozenset({"sela"}), "I asked Sela straight out whether she was my mother.", "PRIMARY", 1, "landing", 10, "p10", None))
    mems.append(Memory("a2", "senator", frozenset({"sela"}), "She said the name at last, and I held it.", "PRIMARY", 2, "landing", 12, "p12", None))
    mems.append(Memory("a3", "senator", frozenset({"gnaeus"}), "Gnaeus passed me on the stair.", "PRIMARY", 6, "house", 14, "p14", None))
    # lately: THREAD_RECENT tuples with the wife at the house, Sela nowhere in them
    for i in range(THREAD_RECENT):
        mems.append(Memory(f"r{i}", "senator", frozenset({"wife"}), f"My wife and I talked, tick {20 + i}.", "PRIMARY", 5, "house", 20 + i, f"p{20 + i}", None))
    mems.append(Memory("w1", "wife", frozenset({"senator"}), "We talked.", "PRIMARY", 5, "house", 34, "p34", None))
    mems.append(Memory("s1", "sela", frozenset({"senator"}), "He asked me.", "PRIMARY", 5, "landing", 10, "p10", None))  # Sela has a stream
    # a v2 leftover present in the old tuples' subject_ids, with no stream of its own: never a thread
    canon.entities["wallet"] = Entity("wallet", ["a wallet with $60 in it"], 0, "promoted_path2")
    mems[0] = Memory("a1", "senator", frozenset({"sela", "wallet"}), "I asked Sela straight out whether she was my mother.", "PRIMARY", 1, "landing", 10, "p10", None)
    t = thread_from_memory(canon, mems, "senator", present={"wife"}, here="house")
    assert t == "sela (Sela, a pedlar) -- [10, indelible] I asked Sela straight out whether she was my mother."
    # Sela present: not a thread; Gnaeus's passing tuple never is (weight > 3); the place stood in falls back --
    # but not on a tuple about someone standing here: the landing's tuples are all with Sela, so nothing
    assert thread_from_memory(canon, mems, "senator", present={"wife", "sela"}, here="house") == ""
    mems.append(Memory("a6", "senator", frozenset(), "I stood alone at the landing and watched the water.", "PRIMARY", 3, "landing", 11, "p11", None))
    assert thread_from_memory(canon, mems, "senator", present={"wife", "sela"}, here="house") == "landing (the east landing) -- [11, weighty] I stood alone at the landing and watched the water."
    mems.pop()
    # too little memory: nothing
    assert thread_from_memory(canon, mems[:5], "senator", present=set(), here="house") == ""
    # a tuple that NAMES the person anchors ahead of an equally weighty, later one where they were merely present
    mems.append(Memory("a4", "senator", frozenset({"sela"}), "I sat by the fire a long while and said nothing.", "PRIMARY", 1, "landing", 13, "p13", None))
    assert thread_from_memory(canon, mems, "senator", present={"wife"}, here="house") == "sela (Sela, a pedlar) -- [10, indelible] I asked Sela straight out whether she was my mother."
    mems.pop()
    # the line reaches the Narrator, on a player's pulse (and on a being's own turn: the next test)
    client = _FakeClient({"narration": "x", "perceiver_tuples": [{"perceiver": "senator", "text": "I sat a while longer by the fire.", "salience": "passing"}],
                          "scene_tuples": [{"scene": "house", "text": "The man sat on by the fire a while.", "present": ["senator", "wife"]}], "suggested_next_action": ""})
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    run_smrti_pulse(canon, "senator", "sits", 40, "p40", mems, client, None)
    assert "THREAD FROM MEMORY: sela (Sela, a pedlar) -- [10, indelible]" in client.calls[0][1]
    # a person merely present in weighty tuples, never named, is furniture: not a thread
    canon.entities["stall"] = Entity("stall", ["a cloth stallholder"], 0, "smrti")
    mems.append(Memory("s2", "stall", frozenset({"senator"}), "Two men asked me about a pedlar.", "PRIMARY", 5, "market", 9, "p9", None))
    mems.append(Memory("a5", "senator", frozenset({"stall", "wife"}), "I asked my wife about the road, hard.", "PRIMARY", 1, "market", 9, "p9", None))
    assert thread_from_memory(canon, mems, "senator", present={"wife", "sela"}, here="house") == ""
    run_smrti_pulse(canon, "senator", None, 41, "p41", mems, client, None)
    assert "THREAD FROM MEMORY: sela (Sela, a pedlar) -- [10, indelible]" in client.calls[1][1]  # the senator's own turn: the thread, which it was never shown before 2026-10-27
    assert "stall" not in client.calls[1][1].split("THREAD FROM MEMORY:")[1].split("\n")[0]       # a person merely present in weighty tuples, never named, is still furniture


def test_a_plan_named_only_in_words_is_a_thread_on_a_beings_own_turn(monkeypatch) -> None:
    """Stonemarch, 2026-10-27: the Temple's two priests held plans against each other and did nothing outward in 34 turns. Their seed lists no one
    (subject_ids []), their later tuples list only the priest and the medallion in his hand, and a being's own turn was shown THREAD FROM MEMORY:
    (none) whatever it held. On a being's own turn the thread now also reads whom a weighty tuple names in its words (`named`); a player's second
    suggestion is as it was."""
    from smritilila.smrti.narrator import SYSTEM
    from smritilila.smrti.pulse import THREAD_RECENT, run_smrti_pulse, thread_from_memory
    assert "THREAD FROM MEMORY, when it names someone or somewhere" in SYSTEM
    canon = Maya(world_time=60, entities={
        "fane": Entity("fane", ["the fane of earth"], 0, "genesis", avarana="building"),
        "cavern": Entity("cavern", ["the cavern pool"], 0, "genesis", avarana="building"),
        "romag": Entity("romag", ["Romag, the earth priest"], 0, "genesis"),
        "belsornig": Entity("belsornig", ["Belsornig, the water priest"], 0, "genesis"),
        "davl": Entity("davl", ["Rannos Davl"], 0, "genesis"),          # named in a tuple, and has no stream of his own: never a thread
        "medallion": Entity("medallion", ["a bronze medallion"], 0, "genesis"),
    })

    def m(i, who, t, text, sal, subjects=(), loc="fane"):
        return Memory(f"m{i}", who, frozenset(subjects), text, "PRIMARY", sal, loc, t, f"p{t}", None)
    mems = [m(1, "romag", 0, "I am Romag, curate of the Earth Temple. I crush bone upon the altar.", 1),
            m(2, "romag", 0, "I deal with Rannos Davl in Millhaven for arms.", 3),
            m(3, "romag", 0, "I know Belsornig buys captives, and I scheme to crush his water-spawn before the Great Temple awakens.", 2)]
    for i in range(THREAD_RECENT + 3):
        mems.append(m(10 + i, "romag", 10 + i, "I ground bone and paid the bugbear sentries.", 5, {"romag", "medallion"}))
    mems.append(m(50, "belsornig", 1, "I tend the green braziers.", 5, {"belsornig"}, "cavern"))
    # as it was: a tuple is about whom it lists; the plan lists no one, so there is no thread
    assert thread_from_memory(canon, mems, "romag", set(), "fane") == ""
    # asked for what the tuple names: the matter he has been leaving, with the plan's own words
    t = thread_from_memory(canon, mems, "romag", set(), "fane", named=True)
    assert t.startswith("belsornig (Belsornig, the water priest) -- [0, marking] I know Belsornig buys captives")
    # present: not a thread; with the actor lately (listed in his recent tuples): not left; a name with no stream: never one
    assert thread_from_memory(canon, mems, "romag", {"belsornig"}, "fane", named=True) == ""
    lately = mems + [m(99, "romag", 55, "Belsornig came to the fane and we spoke.", 5, {"romag", "belsornig"})]
    assert thread_from_memory(canon, lately, "romag", set(), "fane", named=True) == ""
    # a place named in words is not a person: never a thread of this kind (the scene fallback is as it was)
    mems.append(m(60, "romag", 20, "I swore at the cavern pool that it would fall.", 2))
    assert thread_from_memory(canon, mems, "romag", set(), "fane", named=True).startswith("belsornig ")
    # the line reaches the Narrator on the being's own turn, and on a player's pulse it is as it was
    client = _FakeClient({"narration": "x", "perceiver_tuples": [{"perceiver": "romag", "text": "I ground the bone.", "salience": "passing"}],
                          "scene_tuples": [{"scene": "fane", "text": "The priest ground bone.", "present": ["romag"]}], "suggested_next_action": ""})
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    run_smrti_pulse(canon, "romag", None, 61, "p61", mems, client, None)
    assert "THREAD FROM MEMORY: belsornig (Belsornig, the water priest) -- [0, marking]" in client.calls[0][1]
    run_smrti_pulse(canon, "romag", "grinds the bone", 62, "p62", mems, client, None)
    assert "THREAD FROM MEMORY: (none)" in client.calls[1][1]


def test_rule_10_an_action_is_an_attempt_and_memory_shapes_the_outcome() -> None:
    """Marc, 2026-09-20: nothing told the Narrator an action might not
    work out. Memory sets the shape of what can follow; within it one
    outcome happens, not always the attempted one. (First draft; to be
    refined by play.)"""
    from smritilila.smrti.narrator import DEFAULT_VIEW, SYSTEM, system_for
    assert "ACTION is what the actor ATTEMPTS, not what happens" in SYSTEM
    assert "sets the shape of what can follow" in SYSTEM
    assert "which may be to refuse" in SYSTEM
    # how things turn out is the default view's (v5 §7), and still in every prompt unless a world says otherwise
    assert "not always the outcome attempted, nor the likeliest" in DEFAULT_VIEW and DEFAULT_VIEW in system_for(None)


def test_whoever_was_written_nowhere_in_the_same_pulse_is_in_the_boat() -> None:
    """Vadhu tick 341: Teo and the tin peddler both stood nowhere after
    339; Teo's context held no one present and the Narrator minted the
    tin man a second time on the far bank."""
    log = Smrti()
    seq = 0
    def add(p, t, text, place=None, sal=4):
        nonlocal seq
        seq += 1
        log.append(Tuple(p, seq, t, text, sal, place, None))
    add("east", 0, "the east landing", "east"); add("teo", 0, "Teo"); add("tin", 0, "the tin peddler"); add("other", 0, "a drover"); add("label_only", 8, "a stranger")
    add("teo", 5, "I brought the cart woman in.", "east")
    add("tin", 5, "I waited on the west bank.", "east")
    add("other", 7, "I walked out into the hills alone.", None)  # nowhere, a different pulse
    add("teo", 9, "I took the tin man aboard and pushed off.", None)
    add("tin", 9, "I climbed aboard and held my pots.", None)
    ctx = retrieve(log, "teo", 10, None, k_place=12, k_sim=0)
    assert ctx.place is None
    assert ctx.present == {"tin"}
    got = {(t.perceiver, t.chronology) for t in ctx.tuples}
    assert ("tin", 0) in got and ("tin", 9) in got
    assert ("other", 7) not in got and "label_only" not in {t.perceiver for t in ctx.tuples}


def test_what_those_present_cannot_forget_is_in_the_context() -> None:
    """Vadhu tick 345: Ilo's indelible (306, pulled Teo from the
    flood-wrack) was not in Teo's context; Ilo's next answer gave Teo a
    second past. The Narrator voices those present, so their most
    salient tuples ride along, bounded."""
    log = Smrti()
    seq = 0
    def add(p, t, text, place=None, sal=4):
        nonlocal seq
        seq += 1
        log.append(Tuple(p, seq, t, text, sal, place, None))
    add("landing", 0, "the east landing", "landing"); add("teo", 0, "Teo"); add("ilo", 0, "Ilo"); add("far", 0, "a drover")
    add("ilo", 10, "I pulled him from the flood-wrack and kept him.", "landing", sal=1)
    for i in range(11, 40):
        add("ilo", i, f"An ordinary day on the water, tick {i}.", "landing", sal=6)
    add("far", 12, "I will never forget the fire.", "hills", sal=1)
    add("teo", 40, "I sat on the stone and asked for the rest.", "landing")
    add("ilo", 40, "The boy asked me for the rest of it.", "landing")
    add("landing", 40, "The two sat on the stone.", "landing")
    ctx = retrieve(log, "teo", 41, None, k_place=3, k_sim=0)
    got = {(t.perceiver, t.chronology) for t in ctx.tuples}
    assert ("ilo", 10) in got            # what Ilo cannot forget
    assert ("far", 12) not in got        # not someone elsewhere
    assert sum(1 for p, c in got if p == "ilo" and 11 <= c <= 39) <= 2 * (1 + 2)  # bounded: PRESENT_SALIENT, each followed WEIGHT_FOLLOW_UP
    assert {("ilo", 11), ("ilo", 12)} <= got  # what came of Ilo's indelible (2026-09-21): 308 beside 306


def test_bija_seed_becomes_the_world_file_shape_and_plants() -> None:
    """bija (2026-09-20): the Narrator writes the seed; as_spec normalises
    it to what genesis.build plants -- keys, places, people standing in
    places that exist, the player among them, weights as salience."""
    from smritilila.smrti.bija import as_spec
    from smritilila.smrti.genesis import build
    out = {"name": "The Mill Ford", "setting": "A mill on a river, high summer.", "opening": "You stand on the ford.",
           "places": [{"key": "Mill Ford", "label": "the mill ford", "kind": "open", "tuple": "A shallow ford below the mill, the north road going off toward the hills."},
                      {"key": "north_road", "label": "the north road", "kind": "road", "tuple": "A cart track climbing north out of the valley; a trader a week, a rider now and then."},
                      {"key": "bad", "label": "", "kind": "open", "tuple": "x"}],
           "people": [{"key": "Wren", "label": "Wren, the miller's daughter", "place": "mill_ford",
                       "tuples": [{"text": "I was born in the mill and have never slept a night away from the sound of the wheel.", "weight": "marking"},
                                  {"text": "This morning I came down to the ford to wait for the north trader.", "weight": "ordinary"},
                                  {"text": "short", "weight": "ordinary"}]},
                      {"key": "ghost", "label": "a man nobody placed", "place": "nowhere_such", "tuples": [{"text": "I stood somewhere the seed never named, long enough.", "weight": "ordinary"}]}],
           "player": "wren"}
    spec = as_spec(out)
    assert set(spec["places"]) == {"mill_ford", "north_road"} and set(spec["people"]) == {"wren"}
    assert spec["scene_owner"] == "wren" and spec["playable"] == ["wren"] and spec["entry_points"] == ["mill_ford"]
    assert len(spec["people"]["wren"]["tuples"]) == 2  # the placeholder dropped
    entities, memories, config = build(spec, lambda t: None)
    kinds = {e.labels[0]: e.avarana for e in entities if e.avarana is not None}
    assert kinds == {"the mill ford": "scene", "the north road": "scene"}  # v3 kinds are the scene bit (2026-10-03)
    wren = next(e for e in entities if e.labels[0].startswith("Wren") or e.id.startswith("wren"))
    own = [m for m in memories if m.entity_id == wren.id]
    assert [m.salience for m in own] == [2, 5] and [m.world_time for m in own] == [-10, 0]  # marking, ordinary
    assert config["scene_owner_entity_id"] == wren.id and config["setting_description"] == "A mill on a river, high summer."
    # an authored seed's plain strings still take the file's own weights
    entities, memories, _ = build({"places": spec["places"], "people": {"wren": {"label": "Wren", "place": "mill_ford", "tuples": ["I waited by the ford a long while today.", "I saw the trader on the road above."]}},
                                   "scene_owner": "wren", "playable": ["wren"], "entry_points": ["mill_ford"]}, lambda t: None)
    assert [m.salience for m in memories if m.tier == "PRIMARY" and m.entity_id != _["earth"]] == [4, 3]


def test_bija_refuses_a_seed_with_no_place_or_no_player() -> None:
    from pytest import raises
    from smritilila.smrti.bija import as_spec
    with raises(ValueError):
        as_spec({"places": [], "people": [], "player": "x"})
    with raises(ValueError):
        as_spec({"places": [{"key": "a", "label": "a yard", "kind": "open", "tuple": "A yard with a well in it and a gate."}],
                 "people": [{"key": "b", "label": "b", "place": "a", "tuples": [{"text": "I have stood in this yard all my life.", "weight": "ordinary"}]}],
                 "player": "nobody"})


def test_bija_extends_a_seed_that_comes_in_under_its_floors() -> None:
    """A short seed is extended, not redrawn: what was written stays, new
    places and people are added beyond it, repeats are refused."""
    from smritilila.smrti import bija
    calls: list[tuple[str, str]] = []
    first = {"name": "Eyri", "setting": "A fjord.", "opening": "You stand at the racks.", "player": "halla",
             "places": [{"key": "racks", "label": "the drying racks at Eyri", "kind": "open", "tuple": "Rows of split fish above the shore; the shore path runs north and south."},
                        {"key": "shore_path", "label": "the shore path", "kind": "road", "tuple": "A track along the fjord from farm to farm; a rider most days."}],
             "people": [{"key": "halla", "label": "Halla", "place": "racks", "tuples": [{"text": "I keep the racks at Eyri since my husband drowned.", "weight": "indelible"}]}]}
    more = {"name": "x", "setting": "x", "opening": "x", "player": "halla",
            "places": [{"key": "racks", "label": "The Drying Racks at Eyri", "kind": "open", "tuple": "a repeat under another key, refused"},
                       {"key": "hof", "label": "Hof, the chieftain's farm", "kind": "building", "tuple": "The largest hall on the fjord, north along the shore path from Eyri."}],
            "people": [{"key": "ossur", "label": "Ossur the godi", "place": "hof", "tuples": [{"text": "I keep the peace on this fjord by memory more than law.", "weight": "weighty"}]},
                       {"key": "ghost", "label": "a man nowhere", "place": "nowhere_such", "tuples": [{"text": "I stood where the seed never named a place for me.", "weight": "ordinary"}]}]}

    class Client:
        def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
            calls.append((system, user))
            return first if len(calls) == 1 else more

    spec = bija.write_seed(Client(), "I am Halla.", "Iceland, 1120.", "around")
    assert len(calls) == 2 and "EXTENDING" in calls[1][0] and "ALREADY IN THE SEED" in calls[1][1]
    assert set(spec["places"]) == {"racks", "shore_path", "hof"}          # the repeat refused, the new farm added
    assert set(spec["people"]) == {"halla", "ossur"}                      # the unplaced man dropped
    assert spec["scene_owner"] == "halla" and spec["places"]["racks"]["label"] == "the drying racks at Eyri"
    # a seed at its floors makes one call
    calls.clear()
    spec = bija.write_seed(Client(), "I am Halla.", None, "here")
    assert len(calls) == 1 and "SIZE: the scene only" in calls[0][1]


def test_a_world_from_a_brief_has_no_player_and_begins_somewhere() -> None:
    """The new world screen (2026-09-24): a seed from the creator's brief
    -- no player, no opening; the world takes the creator's name; where it
    begins is its entry point; a blank box is the builder's to fill; an
    extension keeps `begins`, not a player; a `begins` that is no scene
    falls back to the most crowded scene."""
    from smritilila.smrti import bija
    calls: list[tuple[str, str, dict]] = []
    first = {"name": "the Narrator's name", "setting": "Tombstone, Arizona Territory, October 1881.", "begins": "oriental",
             "places": [{"key": "tombstone", "label": "Tombstone", "in": "", "scene": False, "tuple": "A silver camp on a mesa; the Earps keep the law."},
                        {"key": "oriental", "label": "the Oriental Saloon", "in": "tombstone", "scene": True, "tuple": "A long bar on Allen Street, faro tables at the back; the street door opens on Allen and Fifth."}],
             "people": [{"key": "wyatt", "label": "Wyatt Earp", "place": "oriental", "tuples": [{"text": "I deal faro at the Oriental and keep an eye on the Clantons.", "weight": "weighty"}]}]}
    more = {"name": "x", "setting": "x", "begins": "oriental",
            "places": [{"key": "ok_corral", "label": "the O.K. Corral", "in": "tombstone", "scene": True, "tuple": "A livery yard off Allen Street, its back lot on Fremont."}],
            "people": [{"key": "ike", "label": "Ike Clanton", "place": "ok_corral", "tuples": [{"text": "I came into town to settle what the Earps owe my family.", "weight": "weighty"}]}]}

    class Client:
        def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
            calls.append((system, user, input_schema))
            return first if len(calls) == 1 else more

    spec = bija.write_world_seed(Client(), "Tombstone 1881", {"where_when": "Tombstone, October 1881", "begins": ""}, "around")
    system, user, schema = calls[0]
    assert system == bija.BRIEF_SYSTEM and schema is bija.BRIEF_SCHEMA and "player" not in schema["properties"]
    assert "WORLD NAME: Tombstone 1881" in user and "WHERE AND WHEN: Tombstone, October 1881" in user
    assert "WHERE IT BEGINS: not given -- choose the scene where a story in this world most likely opens" in user
    assert "WHAT IS DIFFERENT HERE: not given -- as the record or the source has it" in user and "SELF-DESCRIPTION" not in user
    assert "EXTENDING" in calls[1][0] and "Set `begins` to oriental" in calls[1][1] and calls[1][2] is bija.BRIEF_SCHEMA
    assert spec["name"] == "Tombstone 1881" and spec["scene_owner"] == "" and spec["playable"] == [] and spec["opening"] == ""
    assert spec["entry_points"] == ["oriental"] and set(spec["people"]) == {"wyatt", "ike"}
    # a `begins` that is a region, not a scene: the scene with the most people in it
    spec = bija.as_spec({**first, "begins": "tombstone"}, begins=True)
    assert spec["entry_points"] == ["oriental"]
    # the player path is untouched: its prompt still has its player
    assert "which person the player is" in bija.SYSTEM and "which person the player is" not in bija.BRIEF_SYSTEM
    # the world at large, from the brief
    asked: list[tuple[str, str]] = []

    class Writer:
        def complete(self, system, user, *, max_tokens=8192):
            asked.append((system, user))
            return FINISHED

    FINISHED = ("THE YEAR. 1881, October. " * 10 + "THE REGIONS. The San Pedro valley. THE CENTRAL FIGURES. The Earps. "
                "THE WAYS. The Benson road. THE STARTING REGION. Tombstone.")
    brief = bija.brief_text("Tombstone 1881", {"where_when": "Tombstone, October 1881"})
    bija.write_bhuvana(Writer(), "", None, brief)
    assert asked == [(bija.BHUVANA_BRIEF_SYSTEM, brief)] and "player" not in bija.BHUVANA_BRIEF_SYSTEM.replace("never address the player", "")
    # an unfinished one is refused (Ravenmoor's first stopped mid-sentence at the old 4096 cap), and the source leads
    for cut in (FINISHED[:-1] + " The world beyond New England", FINISHED.split("THE WAYS")[0]):
        FINISHED_BAK, FINISHED = FINISHED, cut
        with pytest.raises(ValueError, match="unfinished"):
            bija.write_bhuvana(Writer(), "I am Nathaniel.", "Arkham.", source="Fiction: Ravenmoor")
        FINISHED = FINISHED_BAK
    bija.write_bhuvana(Writer(), "I am Nathaniel.", "Arkham.", source="Fiction: Ravenmoor")
    assert asked[-1][1].startswith("THIS WORLD'S SOURCE: Fiction: Ravenmoor") and "SOURCE's own register" in asked[-1][0]
    # prose, not markdown: the page shows the world at large as text
    assert bija.plain("**THE YEAR**\n\n*Kingsport.* A town; 2*3 stays.") == "THE YEAR\n\nKingsport. A town; 2*3 stays."


def test_a_title_is_not_a_taken_name() -> None:
    """Eyri, second expansion: "Brother Magnús" refused because "Brother
    Jón" had taken "Brother". v4 (v5 §16/§39): names are declared and matched
    exactly -- a title is never a name, because nobody declares it one."""
    from smritilila.smrti import bija
    out = {"places": [{"key": "school", "label": "the field-school", "kind": "building", "tuple": "A turf schoolroom behind the priest's house, on the causeway road."}],
           "people": [{"key": "magnus", "label": "Brother Magnús, who teaches the boys", "name": "Magnús Ketilsson", "place": "school", "tuples": [{"text": "I teach the farmers' boys their letters through the winter.", "weight": "ordinary"}]},
                      {"key": "jon2", "label": "Jón the younger", "name": "Jón the younger", "place": "school", "tuples": [{"text": "I came to the school this spring to learn the psalter.", "weight": "ordinary"}]}]}
    class Client:
        def complete_with_tool(self, *a, **k):
            return out
    new = bija.expand(Client(), None, {"church": ("the church at Hofsstadir", "building")}, [("Brother Jón, a young priest", "church")],
                      names_taken=frozenset({"Brother", "Jón", "Halla"}))
    assert set(new["people"]) == {"magnus"}  # Magnús kept; a second Jón refused (his name begins with Jón)
    # a refusal cascades to whoever is defined by the refused name
    out["people"] += [{"key": "grim2", "label": "Grim of Stakkar", "name": "Grim of Stakkar", "place": "school", "tuples": [{"text": "I farm Stakkar on the moor road with my wife.", "weight": "ordinary"}]},
                      {"key": "ranveig", "label": "Ranveig, Grim's wife", "name": "Ranveig Ormsdóttir", "place": "school", "tuples": [{"text": "I keep house at Stakkar for Grim and our three.", "weight": "ordinary"}]}]
    new = bija.expand(Client(), None, {"church": ("the church at Hofsstadir", "building")}, [], names_taken=frozenset({"Grim", "Jón"}))
    assert set(new["people"]) == {"magnus"}


def test_an_expansion_names_whole_and_refuses_a_name_it_begins_with() -> None:
    """Marc, 2026-09-25: a person's name is made whole (two words at least);
    and a new name that BEGINS with a name the world carries is that person
    ("Grim of Stakkar" is Grim), while one that only holds it is not
    ("Unnr of Kross" beside Kross; "Walter Pike's apprentice")."""
    from smritilila.smrti import bija

    def person(key, label, name=None):
        return {"key": key, "label": label, **({"name": name} if name else {}), "place": "school",
                "tuples": [{"text": "I keep to my own work here.", "weight": "ordinary"}]}
    out = {"places": [{"key": "school", "label": "the field-school", "kind": "building", "tuple": "A turf schoolroom."}],
           "people": [person("wei", "Li Wei"), person("ruth2", "Ruth Pell"), person("appr", "Walter Pike's apprentice", "Sam Dunne"),
                      person("pike2", "Walter Pike's apprentice"), person("one", "Wenna"), person("mr", "Mr. Brinsmade"),
                      person("widow", "the smith's widow"), person("unnr", "Unnr of Kross")]}

    class Client:
        def complete_with_tool(self, *a, **k):
            return out
    new = bija.expand(Client(), None, {"church": ("the church", "building")}, [],
                      names_taken=frozenset({"Li Na", "Ruth", "Walter Pike", "Kross"}))
    # Li Wei is not Li Na; Ruth Pell begins with Ruth (the accepted catch); a
    # possessive is someone else; one word, or a title and one word, is no whole name
    assert set(new["people"]) == {"wei", "appr", "pike2", "widow", "unnr"}


def test_words_and_names_are_unicode() -> None:
    """Eyri: `[a-z]` read Skálholt as "lholt" and Jón as nothing."""
    from smritilila.text import units
    assert "skálholt" in units("the bishop's seat at Skálholt") and "jón" in units("Jón rode out")


def test_a_place_the_world_holds_is_not_planted_again() -> None:
    from smritilila.smrti import bija
    out = {"places": [{"key": "kross", "label": "the farm of Kross", "kind": "building", "tuple": "A farm the world already holds, minted again from an outer ring."},
                      {"key": "kross_field", "label": "the home-field at Kross", "kind": "open", "tuple": "The hayfield below the farm of Kross, walled in turf."}],
           "people": [{"key": "unnr", "label": "Unnr of Kross", "place": "kross_field", "tuples": [{"text": "I keep the house at Kross for my husband Teitur.", "weight": "ordinary"}]}]}
    class Client:
        def complete_with_tool(self, *a, **k):
            return out
    new = bija.expand(Client(), None, {"edge": ("the east moor road", "road")}, [], names_taken=frozenset({"Teitur", "Kross"}), labels_taken=frozenset({"the farm of Kross"}))
    assert set(new["places"]) == {"kross_field"}   # the farm refused as already held; the field near it kept
    assert set(new["people"]) == {"unnr"}          # Kross is her farm, not her name


def test_hex_ids_are_shown_as_handles_and_handles_come_back_as_ids(monkeypatch) -> None:
    # 2026-09-21: the genesis cast carried 32-hex ids into every prompt (976 of
    # the crossing's 1281 tuples); tick 34 garbled one by a character. Encoded
    # at the boundary by llm/handles.py, as it was for the v2 path before that went.
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    teo, ilo, inn = "e5cfa5652a244f2f878cae7b41509d07", "8cfe75fa407f4ccf92d6222fb307ceff", "b5609f75d8d64d21a571566a001b3444"
    canon = Maya(world_time=10, entities={
        teo: Entity(teo, ["Teo, the ferryman's apprentice"], 0, "genesis"),
        ilo: Entity(ilo, ["Ilo, the ferryman"], 0, "genesis"),
        inn: Entity(inn, ["the inn at the crossing"], 0, "genesis", avarana="building"),
        "sela_pedlar_blue_pack_b21cf6": Entity("sela_pedlar_blue_pack_b21cf6", ["sela pedlar blue pack"], 5, "smrti"),
    })
    mems = [_mem(1, teo, 5, "I sat by the fire.", loc=inn), _mem(2, ilo, 5, "I watched the boy.", loc=inn),
            _mem(3, "sela_pedlar_blue_pack_b21cf6", 5, "I kept to the bench.", loc=inn), _mem(4, inn, 5, "Three by the fire.", loc=inn)]
    client = _FakeClient({
        "narration": "You speak.",
        "perceiver_tuples": [{"perceiver": "teo-the-ferrymans-apprentice", "text": "one edit off the shown handle: I told Ilo I would cross at first light.", "salience": "weighty"},
                             {"perceiver": "Ilo-the-Ferryman", "text": "The boy said he would cross at first light.", "salience": "notable"},
                             {"perceiver": "sela_pedlar_blue_pack_b21cf6", "text": "I heard the boy say he would cross.", "salience": "notable"},
                             {"perceiver": "ilo-the-feryman", "text": "garbled by a letter: still Ilo, folded onto his row above", "salience": "passing"}],
        "scene_tuples": [{"scene": "inn-at-the-crossing", "text": "The boy said by the fire that he would cross at first light.", "present": ["teo-the-ferrymans-apprentice", "ilo-the-ferryman", "sela_pedlar_blue_pack_b21cf6"]}],
        "actor_stands": "inn-at-the-crossing", "came_along": [], "suggested_next_action": "",
    })
    r = run_smrti_pulse(canon, teo, "I say I'll cross at first light.", 6, "p6", mems, client, None)
    system, user = client.calls[0]
    assert not re.search(r"[0-9a-f]{32}", user), "no hex id reaches the Narrator"
    assert "[teo-the-ferryman-s-apprentice, 5, weighty, none]" in user and "[inn-at-the-crossing [scene], 5" in user
    assert "ACTOR: teo-the-ferryman-s-apprentice" in user and "SCENE: inn-at-the-crossing" in user
    # 2026-10-02, one rule for every id: a minted id is shown by its own words, its tail dropped -- and the raw id
    # the model may still copy back (as this fake does) is accepted
    assert "sela_pedlar_blue_pack_b21cf6" not in user and "[sela-pedlar-blue-pack, 5," in user
    assert r.new_entities == [] and r.dropped == []
    assert {m.entity_id for m in r.memories_written} == {teo, ilo, "sela_pedlar_blue_pack_b21cf6", inn}
    assert all(m.location_id == inn for m in r.memories_written)


def test_a_pulse_never_changes_an_entity_in_place(monkeypatch) -> None:
    """Marc, 2026-09-25: a pulse's working world shares the live world's
    entity objects (Maya.fork, not a deep copy). Exact only while no pulse
    changes an entity in place -- it mints, and records. A pulse that
    mints, moves, names, aliases, becomes and is narrated again leaves
    every entity it was given exactly as it was, and the live table too."""
    import dataclasses
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    live = Maya(world_time=10, entities={
        "teo": Entity("teo", ["Teo, the ferryman's apprentice"], 0, "genesis", kosha="inn"),
        "ilo": Entity("ilo", ["Ilo, the ferryman"], 0, "genesis", kosha="inn"),
        "inn": Entity("inn", ["the inn at the crossing"], 0, "genesis", avarana="building"),
        "rope": Entity("rope", ["a coil of rope"], 2, "smrti", kosha="inn"),
    })
    before = {i: dataclasses.asdict(e) for i, e in live.entities.items()}
    mems = [_mem(1, "teo", 5, "I sat by the fire.", loc="inn"), _mem(2, "ilo", 5, "I watched the boy.", loc="inn"),
            _mem(3, "inn", 5, "Two by the fire; a coil of rope by the door.", loc="inn")]
    client = _FakeClient({
        "narration": "You take the rope; a stranger named Ilo Varn comes in with a lamp.",
        "perceiver_tuples": [{"perceiver": "teo", "text": "I took up the rope and met Ilo Varn at the door.", "salience": "notable"}],
        "scene_tuples": [{"scene": "inn", "text": "The boy took the rope; a stranger with a lamp came in.", "present": ["teo", "ilo", "NEW:varn"]}],
        "new_entities": [{"id": "NEW:varn", "name": "Ilo Varn", "in": "inn", "scene": False},
                         {"id": "NEW:lamp", "name": "the stranger's lamp", "in": "NEW:varn", "scene": False}],
        "moves": [{"entity": "rope", "into": "teo"}],
        "aliases": [{"entity": "ilo", "name": "the old ferryman"}],
        "becomings": [{"entity": "inn", "key": "wakes", "clause": "The inn remembers who comes in."}],
        "capabilities": [{"entity": "inn", "capability": "perceives", "on": True}],
        "actor_stands": "inn", "came_along": [], "suggested_next_action": "",
    })
    work = live.fork()
    r = run_smrti_pulse(work, "teo", "I take up the rope.", 11, "p11", mems, client, None)
    assert len(client.calls) == 2  # narrated again: "Ilo Varn" shares a name with Ilo
    assert r.new_entities and all(e.id not in live.entities for e in r.new_entities)
    assert set(live.entities) == set(before)                                   # the live table: nothing added
    assert {i: dataclasses.asdict(e) for i, e in live.entities.items()} == before  # nor any entity changed
    assert all(work.entities[i] is live.entities[i] for i in before)           # shared, not copied


def test_a_handle_is_decoded_exactly_or_within_two_edits_never_a_minted_id() -> None:
    from smritilila.llm.handles import HandleMap
    from smritilila.smrti.narrator import decode_handle
    h = HandleMap.build(["e5cfa5652a244f2f878cae7b41509d07"], {"e5cfa5652a244f2f878cae7b41509d07": ["the east landing"]})
    assert decode_handle("east-landing", h) == "e5cfa5652a244f2f878cae7b41509d07"
    assert decode_handle("East_Landing", h) == "e5cfa5652a244f2f878cae7b41509d07"
    assert decode_handle("east-lnding", h) == "e5cfa5652a244f2f878cae7b41509d07"
    assert decode_handle("NEW:east-landing", h) == "NEW:east-landing"
    assert decode_handle("east_landing_a1b2c3", h) == "east_landing_a1b2c3"  # a minted id, never fuzzed
    assert decode_handle("west-road", h) == "west-road"



def test_genesis_ids_carry_the_label_and_a_short_tail() -> None:
    from smritilila.smrti.genesis import build, plant_chronicle
    spec = {
        "name": "t", "setting": "s", "scene_owner": "a", "playable": ["a"], "entry_points": ["p"],
        "places": {"p": {"label": "the east landing", "kind": "open", "tuple": "A landing."}},
        "people": {"a": {"label": "Teo, the ferryman's apprentice", "place": "p", "tuples": ["I came here."]}},
    }
    ents, _, _ = build(spec, lambda text: [0.0])
    place, person = next(e.id for e in ents if e.avarana), next(e.id for e in ents if not e.avarana)
    assert re.fullmatch(r"the_east_landing_[0-9a-f]{6}", place) and re.fullmatch(r"teo_the_ferryman_s_apprentice_[0-9a-f]{6}", person)
    e, _, name = plant_chronicle("THE YEAR is 3018.", lambda text: [0.0])
    assert (name.kind, name.entity_id, name.value) == ("name", e.id, "the world at large")  # it declares its name
    assert re.fullmatch(r"the_world_at_large_[0-9a-f]{6}", e.id)


def test_every_retrieved_tuple_names_the_rule_that_supplied_it() -> None:
    # 2026-09-21: the block is logged by source, so a cap is cut on measurement
    from smritilila.smrti.pulse import context_sections
    log = _log()
    ctx = retrieve(log, "senator", 30, None, k_place=3, k_sim=0)
    assert ctx.sources and set(ctx.sources) == {(t.perceiver, t.seq) for t in ctx.tuples}
    assert ctx.sources[(ctx.actor, max(t.seq for t in ctx.tuples if t.perceiver == ctx.actor))] == "last"
    sections = context_sections(ctx)
    assert sum(d["tuples"] for d in sections.values()) == len(ctx.tuples) and "unsourced" not in sections
    assert all(d["tokens"] > 0 for d in sections.values())
    # the age reads its own stream only, and says so
    own = retrieve(log, "senator", 30, None, k_place=0, k_sim=0, own_only=True)
    assert set(own.sources.values()) <= {"last", "own_salient", "own_recent"}


def test_a_retrieved_tuple_brings_the_name_of_the_place_it_stands_in() -> None:
    # Ravenmoor tick 58: seven own tuples at the house shown, the house's own
    # first tuple not -- "drove home" minted a second house
    log = Smrti()
    log.append(Tuple("house", 0, -1, "a modest house not far from campus", 6, "house"))
    log.append(Tuple("campus", 1, -1, "miskatonic campus grounds", 6, "campus"))
    log.append(Tuple("nat", 2, -1, "Nathaniel Thompson", 6, None))
    log.append(Tuple("nat", 3, 0, "I stand in the front room of my house, the crate of dig tools by the door.", 2, "house"))
    for i in range(1, 30):  # a long stretch elsewhere: the house falls out of "lately"
        log.append(Tuple("nat", 3 + i, i, f"I worked on campus, day {i}.", 5, "campus"))
        log.append(Tuple("campus", 40 + i, i, f"The professor worked on the grounds, day {i}.", 5, "campus"))
    ctx = retrieve(log, "nat", 30, None, k_place=3, k_sim=0)
    assert any(t.perceiver == "nat" and t.place == "house" for t in ctx.tuples)  # item 8 carried the weighted tuple
    house = next(t for t in ctx.tuples if t.perceiver == "house")
    assert house.memory == "a modest house not far from campus" and ctx.sources[("house", 0)] == "stood_in"


def test_a_place_keeps_the_line_said_in_it_verbatim_and_a_player_present_hears_it(monkeypatch) -> None:
    # Marc, 2026-09-21: what the walls heard -- the one line said aloud, the narration's own words or nothing,
    # in the place's record and so in the witnessed tuple, where the player hears the sentence, not a precis
    from smritilila.smrti.pulse import run_smrti_pulse, split_heard
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    monkeypatch.setattr("smritilila.smrti.narrator.log_event", lambda event, **f: events.append((event, f)))
    canon = _canon()
    mems = [_mem(1, "senator", 5, "I sat by the fire.", loc="house"), _mem(2, "wife", 5, "I sat beside him.", loc="house")]
    out = {"narration": "She crosses to him and sets a hand on his shoulder. “You didn't fail, boy,” she says, and nobody speaks.",
           "perceiver_tuples": [{"perceiver": "wife", "text": "I told him plain that he had not failed.", "salience": "weighty"}],
           "scene_tuples": [{"scene": "house", "text": "The woman crossed to the man and told him he had not failed.", "said": "You didn't fail, boy",
                             "said_by": "wife", "present": ["wife", "senator"]}],
           "suggested_next_action": ""}
    r = run_smrti_pulse(canon, "wife", None, 6, "p6", mems, _FakeClient(out), None, players=frozenset({"senator"}))
    by = {m.entity_id: m for m in r.memories_written}
    # v5 §31 audit: the line is a record, not English written into the memory
    assert by["house"].experience == "The woman crossed to the man and told him he had not failed."
    assert by["senator"].experience == by["house"].experience  # the player witnessed the scene's record
    assert [(x.kind, x.entity_id, x.value, x.note) for x in r.records_written if x.kind == "said"] == [("said", "house", "You didn't fail, boy", "wife")]
    # ... and the Narrator is shown it with the record, the witness's copy too
    log, _ = build_log(canon, mems + r.memories_written, r.records_written)
    assert any(t.memory.endswith('-- said by wife: "You didn\'t fail, boy"') for t in log.stream("house"))
    assert any(t.memory.endswith('-- said by wife: "You didn\'t fail, boy"') for t in log.stream("senator"))
    assert split_heard('x -- Said: "a line"') == ("x", "a line", "") and split_heard('a -- b -- c said: "d"') == ("a -- b", "d", "c") and split_heard("no line here") == ("no line here", "", "")
    # a line the narration does not hold is dropped, not kept
    out["scene_tuples"][0]["said"] = "Something she never said"
    r = run_smrti_pulse(canon, "wife", None, 7, "p7", mems, _FakeClient(out), None, players=frozenset({"senator"}))
    assert not any(x.kind == "said" for x in r.records_written)
    # stamped with the pulse that dropped it -- p7, not p6: run_smrti_pulse
    # binds its own id, so the stamp cannot be a leftover from the last one
    assert ("smrti.place_said_dropped", {"pulse_id": "p7", "said": "Something she never said"}) in events


def test_a_quoted_line_interrupted_by_its_tag_is_still_the_narrations_own_words(monkeypatch) -> None:
    # the crossing, tick 395: "Yes. I'm his mother. I gave him away because I was young..." was in the
    # narration in two pieces around `she says`, and the whole was dropped
    from smritilila.smrti.narrator import verified_said
    events = []
    monkeypatch.setattr("smritilila.smrti.narrator.log_event", lambda event, **f: events.append((event, f)))
    narration = 'She looks up at last. "Yes. I\'m his mother," she says, her voice level. "I gave him away because I was young and frightened." Nobody moves.'
    assert verified_said("Yes. I'm his mother. I gave him away because I was young and frightened.", narration, "x") == "Yes. I'm his mother. I gave him away because I was young and frightened."
    assert verified_said("Yes. I'm his mother. I never told anyone.", narration, "x") == ""  # a sentence not in the prose: the whole goes
    assert verified_said("I'm his mother", narration, "x") == "I'm his mother"
    # the crossing 397: the tag inside one sentence -- `"Whatever that name was," you say, low, "he's Teo now..."`
    marren = "You set down bread, still warm. \"Whatever that name was,\" you say, low, \"he's Teo now, and he's fed, and it's late.\" You touch the boy's shoulder."
    assert verified_said("Whatever that name was, he's Teo now, and he's fed, and it's late.", marren, "x") == "Whatever that name was, he's Teo now, and he's fed, and it's late."
    assert verified_said("Whatever that name was, go to bed.", marren, "x") == ""  # words not in the prose
    assert verified_said("late, and he's fed", marren, "x") == ""  # the prose's words out of order
    # pulse_id rides on the drop (2026-09-22), so a drop can be
    # attributed to a turn without going by its position in the record.
    # Bound explicitly here because these are direct calls, outside any
    # pulse -- the binding is a ContextVar that outlasts any call and this test must not depend
    # on which pulse some earlier test left bound.
    from smritilila.llm.client import bind_pulse_id
    bind_pulse_id("q1")
    events.clear()
    verified_said("Yes. I'm his mother. I never told anyone.", narration, "x")
    assert [e for e in events if e[0] == "x"][0] == (
        "x", {"pulse_id": "q1", "said": "Yes. I'm his mother. I never told anyone."})


def test_a_tuple_under_watch_is_logged_with_the_rule_that_brought_it(monkeypatch) -> None:
    from smritilila.smrti.pulse import run_smrti_pulse, watched_tuples
    monkeypatch.setenv("SMRITILILA_WATCH", "senator:5, wife:-9 ,bad")
    assert watched_tuples() == frozenset({("senator", 5), ("wife", -9)})
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    canon = _canon()
    mems = [_mem(1, "senator", 5, "I sat by the fire.", loc="house"), _mem(2, "wife", 5, "I sat beside him.", loc="house")]
    out = {"narration": "x", "perceiver_tuples": [{"perceiver": "senator", "text": "I sat on by the fire a while longer.", "salience": "passing"}],
           "scene_tuples": [{"scene": "house", "text": "The man sat on by the fire.", "present": ["senator", "wife"]}], "suggested_next_action": ""}
    run_smrti_pulse(canon, "senator", "sits", 6, "p6", mems, _FakeClient(out), None)
    watched = [f for e, f in events if e == "smrti.watched"]
    assert watched == [{"pulse_id": "p6", "world_time": 6, "actor": "senator", "perceiver": "senator", "tick": 5, "source": "last"}]


def test_a_seam_the_narrator_wrote_into_its_own_text_is_stripped_before_ours(monkeypatch) -> None:
    # Ravenmoor 195: `-- aunt_ruth_ipswich_12f9b0 said: "..."` inside the place text, the line then twice
    from smritilila.smrti.pulse import strip_seams
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    text = 'She offered them coffee. -- aunt_ruth_ipswich_12f9b0 said: "There\'s coffee on the stove." The men stood.'
    assert strip_seams(text, "p1") == "She offered them coffee. The men stood."
    assert strip_seams('x -- Said: "y"', "p1") == "x" and strip_seams("no seam -- just a dash", "p1") == "no seam -- just a dash"
    assert events == [("smrti.seam_imitated", {"pulse_id": "p1", "count": 1}), ("smrti.seam_imitated", {"pulse_id": "p1", "count": 1})]


def test_a_player_standing_here_gets_the_place_s_account_off_the_roster(monkeypatch) -> None:
    """The 66 (2026-09-22). `present` is the NARRATOR's roster, so a
    pulse that voiced a player and left them off it shut both doors at
    once -- v5 §37 drops the voiced tuple and v5 §31's remedy never ran, and
    the player's record of the pulse did not exist. The crossing at 364:
    the place tuple named the ferryman, four others took tuples there,
    and Teo has 363 and 365 and nothing between. The engine knows where
    he stood."""
    from smritilila.smrti.pulse import run_smrti_pulse
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    canon = _canon()
    # the player's own last tuple stands at the house, by the engine's own record
    mems = [_mem(1, "senator", 5, "I sat by the fire.", loc="house"),
            _mem(2, "wife", 5, "I sat beside him.", loc="house")]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "wife", "text": "I told him at last that there had been a son.", "salience": "indelible"},
                                {"perceiver": "senator", "text": "She told me there was a son.", "salience": "indelible"}],
           # the roster omits the player the Narrator just voiced -- the 66
           "scene_tuples": [{"scene": "house", "text": "By the fire the woman told the man of a son.", "present": ["wife"]}],
           "suggested_next_action": ""}

    r = run_smrti_pulse(canon, "wife", None, 6, "p6", mems, _FakeClient(out), None,
                        players=frozenset({"senator"}))

    assert ("smrti.player_not_voiced", {"pulse_id": "p6", "player": "senator", "actor": "wife"}) in events
    witnessed = [m for m in r.memories_written if m.entity_id == "senator"]
    assert len(witnessed) == 1, "the player who stood here is owed the place's account"
    assert witnessed[0].experience == "By the fire the woman told the man of a son."
    assert witnessed[0].tier == "PRIMARY" and witnessed[0].location_id == "house"
    # present is containment (v5 §11), not the roster: he is by the fire
    assert witnessed[0].subject_ids == frozenset({"wife", "senator"})
    # and which door they came through stays measurable
    assert ("smrti.player_witnessed",
            {"pulse_id": "p6", "player": "senator", "place": "house", "use": "none"}) in events
    # the Narrator's own words for him are still refused: v5 §37 is untouched
    assert not any("She told me" in m.experience for m in r.memories_written)


def test_a_player_who_is_somewhere_else_is_not_handed_the_place_s_account(monkeypatch) -> None:
    """The mirror. `occupants` answers from the player's own last tuple
    and is allowed to say "not here" -- it must not become a license to
    file every place's record on every player."""
    from smritilila.smrti.pulse import run_smrti_pulse
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    canon = _canon()
    canon.entities["yard"] = Entity("yard", ["the yard"], 0, "genesis", avarana="scene")
    mems = [_mem(1, "senator", 5, "I stood out in the yard.", loc="yard"),
            _mem(2, "wife", 5, "I sat by the fire.", loc="house")]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "wife", "text": "I said it to the empty room.", "salience": "weighty"},
                                {"perceiver": "senator", "text": "I heard it from the yard.", "salience": "weighty"}],
           "scene_tuples": [{"scene": "house", "text": "The woman spoke to an empty room.", "present": ["wife"]}],
           "suggested_next_action": ""}

    r = run_smrti_pulse(canon, "wife", None, 6, "p6", mems, _FakeClient(out), None,
                        players=frozenset({"senator"}))

    assert not any(e == "smrti.player_witnessed" for e, _ in events)
    assert not any(m.entity_id == "senator" for m in r.memories_written)


def test_a_person_is_never_promoted_to_a_place(monkeypatch) -> None:
    """The mirror of became_a_place, deleted in v4 (2026-09-23): a place row
    naming a person by mistake must not turn that person into ground."""
    from smritilila.smrti.kosha import Kosha
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    mems = [_mem(1, "senator", 5, "I sat with my wife.", loc="house")]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "senator", "text": "I sat a while.", "salience": "ordinary"}],
           "scene_tuples": [{"scene": "wife", "text": "Someone was here.", "present": ["senator"]}],
           "suggested_next_action": ""}

    run_smrti_pulse(canon, "senator", "sit", 6, "p6", mems, _FakeClient(out), None)

    assert not Kosha(canon).is_scene("wife")


# 2026-09-23, v4: deleted with the mechanisms they tested -- label-word
# and head-noun guards (same_label, resolve_handle, named-place binding,
# "a person is not named by where they stand"), actor_stands, came_along,
# roads, things and became_a_place, and item 9's text rule (v5 §9, §17).
# Their scenarios -- Vadhu 149, 323, 337, 339, Ravenmoor 212-275 and
# 292-306 -- are restated against moves and containment in test_kosha.py.


def test_the_narrators_prompt_names_no_world() -> None:
    """Live, the crossing 479 (2026-09-23): the v4 prompt's example
    "behind the old sheep pen" -- Ravenmoor's -- came back as a sheep pen
    in Aldwick. The prompt serves every world; its examples name none."""
    from smritilila.smrti.narrator import CHRONICLE_ACTION, SCENE_ARRIVAL_ACTION, SYSTEM
    text = (SYSTEM + SCENE_ARRIVAL_ACTION + CHRONICLE_ACTION).lower()
    for name in ("sheep pen", "corwin", "st. mary", "ipswich", "creek elbow", "nathaniel", "hartley", "ruth", "arkham",
                 "miskatonic", "teo", "ilo", "sela", "marren", "vadh", "bree", "shire", "tombstone", "palatine", "aldwick", "miller", "mill path", "storehouse", "overhang", "quarry"):
        assert not re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", text), name


def test_a_missing_narration_is_the_pages_own_summary(monkeypatch) -> None:
    """Live, the crossing 480: the model left the required narration out
    on the pulse Bren was found; the player read nothing."""
    from smritilila.smrti.narrator import narrate
    events = []
    monkeypatch.setattr("smritilila.smrti.narrator.log_event", lambda event, **f: events.append((event, f)))
    client = _FakeClient({"perceiver_tuples": [], "scene_tuples": [],
                          "sara": {"what": "You walk on past the mill and a man looks up.", "said": "", "turn": ""}})
    out = narrate(client, retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0), "walks", set())
    assert out.narration == "You walk on past the mill and a man looks up."
    assert any(e == "smrti.narration_missing" and f["fallback"] == "sara" for e, f in events)


def test_a_line_split_by_a_long_tag_is_still_the_line_and_indirect_speech_is_not(monkeypatch) -> None:
    """The crossing 496: Bren's line, quoted exactly around a seven-word
    tag, was dropped as unheard; 488 and 491, told in indirect speech,
    rightly were."""
    from smritilila.smrti.narrator import verified_said
    monkeypatch.setattr("smritilila.smrti.narrator.log_event", lambda event, **f: None)
    narr = ('He comes to the opening. "I’ll come," he says, plain, no dressing on it. "Thirty years of not asking is enough '
            'not-asking. I want to see the man who raised you." He asks only if you mean to leave.')
    said = "I'll come. Thirty years of not asking is enough not-asking. I want to see the man who raised you."
    assert verified_said(said, narr, "x") == said
    indirect = "he says he needs the night to think on it, that thirty years doesn't unknot itself between one lane and one morning."
    assert verified_said("I need the night to think on it. Thirty years doesn't unknot itself between one lane and one morning.", indirect, "x") == ""


def test_one_broken_row_does_not_cost_the_good_ones(monkeypatch) -> None:
    """The crossing 515: perceiver_tuples came back as a string with one
    broken row; the whole list was dropped, and four people present lost
    their tuples of the evening Bren came to the inn."""
    from smritilila.smrti.narrator import narrate
    events = []
    monkeypatch.setattr("smritilila.smrti.narrator.log_event", lambda event, **f: events.append((event, f)))
    rows = ('[{"perceiver": "a", "text": "I stood back and let the boy say it for all of us.", "salience": "notable"}, '
            '{"perceiver": "b, ": "text_placeholder", "salience": "notable"}, '
            '{"perceiver": "c", "text": "I set down my cloth and started pulling out the chairs.", "salience": "notable"}]')
    out = narrate(_FakeClient({"narration": "x", "perceiver_tuples": rows, "scene_tuples": []}),
                  retrieve(_log(), "senator", 30, None, k_place=3, k_sim=0), "waits", set())
    assert [p["perceiver"] for p in out.perceiver_tuples] == ["a", "c"]
    assert ("smrti.rows_salvaged", {"field": "perceiver_tuples", "kept": 2}) in events


def test_an_apostrophe_inside_a_word_is_never_an_opening_quote() -> None:
    """The crossing, 623: a spoken line holding a nested quotation lost 21
    words -- the closing quote after "again." and the apostrophe in "he'd"
    were read as a dialogue tag's two ends. A tag still goes."""
    from smritilila.smrti.narrator import strip_tag
    line = "I told him, 'Take your pots west, Hollis, and don't come asking at the mill again.' I said it hard, so he'd believe me."
    assert strip_tag(line) == line
    assert strip_tag("No stone,' he says slowly, 'means nothing here.") == "No stone, means nothing here."
    assert strip_tag("No stone,’ he says slowly, ‘means nothing here.") == "No stone, means nothing here."


def test_a_first_tuple_carries_the_narrators_use() -> None:
    """2026-09-25 review: the builder asked for weight and never for use, so
    every founding memory was stored with none -- ranked below any "useful"
    pulse memory, and never key. An authored seed's plain strings still have
    none."""
    from smritilila.smrti.bija import _tuples
    from smritilila.smrti.genesis import plant_person
    tuples = _tuples([{"text": "I came to Arkham to find my brother, who wrote from here in March.", "weight": "weighty", "use": "key"},
                      {"text": "I have a room on Garrison Street and a week of money left.", "weight": "ordinary"}])
    assert [t["use"] for t in tuples] == ["key", "none"]  # unanswered: none
    _, mems, _ = plant_person("Hob", tuples + ["I have always been a light sleeper, all my life."], "ford", 5, "p5", lambda t: None)
    assert [m.use for m in mems] == ["key", "none", None]


def test_a_provider_with_its_own_figures_sends_its_model_those_in_place_of_the_shared_ones(monkeypatch) -> None:
    """Marc, 2026-09-26: length suggestions by the model called, for the memories mainly. A provider (llm/providers.py) may offer swaps for its model;
    the Narrator's prompt carries them as a variant, resolved by the client that answers, and a model with none is sent the shared prompt."""
    from smritilila.llm import providers
    from smritilila.llm.client import resolve_for
    from smritilila.smrti.narrator import NARRATION_LENGTH, SYSTEM, TUPLE_LENGTH, system_for
    assert NARRATION_LENGTH in SYSTEM and TUPLE_LENGTH in SYSTEM  # else the swap would silently do nothing
    swaps = {NARRATION_LENGTH: "130 to 170 words", TUPLE_LENGTH: "Each is two to three sentences (the actor's runs to four or five); a bystander's is one."}
    providers.all_providers()
    monkeypatch.setitem(providers._PROVIDERS, "fake", providers.Provider(
        name="fake", narrate_rank=99, best_rank=99, make_narrator=lambda: None, make_best=lambda model: None, describe=lambda: None, hint="X",
        swaps=lambda: {"fakemodel": swaps}))
    view = "Here a song runs 80 to 150 words."
    prompt = system_for(view, "Fiction: the stories of H. P. Lovecraft")
    plain, which = resolve_for(prompt, "some-other-model")
    assert which == "default" and plain == str(prompt)
    own, which = resolve_for(prompt, "fakemodel-1")
    assert which == "fakemodel" and "130 to 170 words" in own and "(the actor's runs to four or five)" in own
    back = own
    for old, new in swaps.items():
        back = back.replace(new, old)
    assert back == plain  # the figures, and nothing else
    assert view in own  # the world's own words are its own


def _echo_log(past: list[str]):
    canon = Maya(world_time=20, entities={
        "stairs": Entity("stairs", ["the stairs"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis")})
    mems = [Memory(f"m{i}", "wen", frozenset(), text, "PRIMARY", 4, "stairs", 10 + i, f"p{10 + i}", None) for i, text in enumerate(past)]
    return build_log(canon, mems)[0]


def test_pratidhvani_tells_a_dead_echo_and_names_the_tic(monkeypatch) -> None:
    """Wentworth, Ravenmoor 554-590: "the count ran on under my thoughts"
    tuple after tuple, taking nothing from the room -- an echo, and past
    ECHO_TIC in a row, a tic. An instrument: it only witnesses."""
    from smritilila.smrti import pulse
    events = []
    monkeypatch.setattr(pulse, "log_event", lambda e, **f: events.append((e, f)))
    log = _echo_log(["The count ran on under my thoughts: two, two.", "The count ran on under my thoughts: six, nine.",
                     "The count ran on under my thoughts: four, one."])
    written = [Memory("n1", "wen", frozenset(), "The count ran on under my thoughts: nine, three.", "PRIMARY", 4, "stairs", 20, "p20", None),
               Memory("n2", "stairs", frozenset(), "Lowell whispered by the window.", "AMBIENT", 4, "stairs", 20, "p20", None)]
    found = pulse.pratidhvani(written, log, "p20")
    assert len(found) == 1 and found[0]["alive"] is False and found[0]["streak"] == 4
    assert [e for e, _ in events] == ["smrti.echo", "smrti.echo_tic"]


def test_pratidhvani_an_echo_that_takes_from_the_present_is_alive(monkeypatch) -> None:
    """The ticking in Frye: a phrase that comes back, but brings the room
    with it -- alive, and no tic however often."""
    from smritilila.smrti import pulse
    monkeypatch.setattr(pulse, "log_event", lambda e, **f: None)
    log = _echo_log(["I heard the ticking under the floor, slow and steady."] * 4)
    written = [Memory("n1", "wen", frozenset(), "I heard the ticking under the floor, slow and steady, and the window rattled.",
                      "PRIMARY", 4, "stairs", 20, "p20", None),
               Memory("n2", "stairs", frozenset(), "The window rattled in its frame.", "AMBIENT", 4, "stairs", 20, "p20", None),
               Memory("n3", "wen", frozenset(), "A woman in a grey wrapper opened the green door.", "PRIMARY", 4, "stairs", 20, "p20", None)]
    found = pulse.pratidhvani(written, log, "p20")
    assert len(found) == 1 and found[0]["alive"] is True  # the fresh sentence is no echo at all


def test_a_round_is_one_pulse_its_players_each_act_in_and_are_each_told(monkeypatch) -> None:
    """Saṅgama (Marc, 2026-09-27, Stonemarch: "combine both of our actions
    and create a combined narrative for each of us"). Two players stand
    together; each acts; one pulse resolves both. The other player of the
    round is an actor of it -- their own tuple, their own move, their own
    words, their own fortune -- and is told the moment in their own
    narration, not handed the scene's account."""
    from smritilila.smrti.pulse import run_smrti_pulse
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    canon = _canon()
    mems = [_mem(1, "senator", 5, "I sat by the fire.", loc="house"), _mem(2, "wife", 5, "I sat beside him.", loc="house")]
    client = _FakeClient({
        "narration": "You call for old Gnaeus as your wife rises and goes out to the garden court.",
        "perceiver_tuples": [{"perceiver": "senator", "text": "I called for Gnaeus while my wife went out to the court.", "weight": "ordinary"},
                             {"perceiver": "wife", "text": "I rose and went out to the garden court as he called for Gnaeus.", "weight": "ordinary"}],
        "scene_tuples": [{"scene": "house", "text": "The man called for a servant; the woman went out to the court.", "said": "",
                          "present": ["senator"]}],
        "moves": [{"entity": "wife", "into": "court"}],
        "round_narrations": [{"player": "wife", "narration": "You rise and step out into the garden court; behind you he calls for Gnaeus.",
                              "suggested_next_action": "Wait for him by the fountain."}],
        "suggested_next_action": "",
    })
    r = run_smrti_pulse(canon, "senator", "I call for Gnaeus.", 6, "p6", mems, client, None,
                        players=frozenset({"senator", "wife"}), playable=frozenset({"senator", "wife"}),
                        sangama={"wife": "I go out to the garden court."})
    user = client.calls[0][1]
    assert "SAṄGAMA" in user and "I go out to the garden court." in user and "IF THEIR ACTION HOLDS ANYTHING IN DOUBT:" in user  # her own fortune
    assert "THE OTHERS HERE ACT TOO" in user  # a round holds the scene's world turns: those present act in it, unasked
    assert r.round_narrations == {"wife": {"narration": "You rise and step out into the garden court; behind you he calls for Gnaeus.",
                                           "suggested_next_action": "Wait for him by the fountain."}}
    hers = [m for m in r.memories_written if m.entity_id == "wife"]
    assert [m.experience for m in hers] == ["I rose and went out to the garden court as he called for Gnaeus."]  # hers, not the scene's copy
    assert any(x.kind == "kosha" and x.entity_id == "wife" and x.value == "court" for x in r.records_written)  # her action moved her
    assert any(x.kind == "action" and x.entity_id == "wife" and x.value == "I go out to the garden court." for x in r.records_written)
    assert not any(e == "smrti.player_not_voiced" for e, _ in events)


def test_a_player_in_a_round_who_did_not_act_is_written_for_and_never_moved(monkeypatch) -> None:
    """v5 §36-§37 in a round: one who has not acted when the round is written is
    still theirs -- never moved -- but perceives, and is told it; a telling
    the Narrator left out is their own tuple of the moment."""
    from smritilila.smrti.pulse import run_smrti_pulse
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    canon = _canon()
    mems = [_mem(1, "senator", 5, "I sat by the fire.", loc="house"), _mem(2, "wife", 5, "I sat beside him.", loc="house")]
    client = _FakeClient({
        "narration": "You call for old Gnaeus.",
        "perceiver_tuples": [{"perceiver": "senator", "text": "I called for old Gnaeus across the house.", "weight": "ordinary"},
                             {"perceiver": "wife", "text": "I heard him call for old Gnaeus across the house.", "weight": "passing"}],
        "scene_tuples": [{"scene": "house", "text": "The man called for a servant.", "present": ["senator", "wife"]}],
        "moves": [{"entity": "wife", "into": "court"}],  # not hers to be moved: she did not act
        "suggested_next_action": "",
    })
    r = run_smrti_pulse(canon, "senator", "I call for Gnaeus.", 6, "p6", mems, client, None,
                        players=frozenset({"senator", "wife"}), playable=frozenset({"senator", "wife"}), sangama={"wife": ""})
    assert [m.experience for m in r.memories_written if m.entity_id == "wife"] == ["I heard him call for old Gnaeus across the house."]
    assert not any(x.kind == "kosha" and x.entity_id == "wife" for x in r.records_written)
    assert not any(x.kind == "action" and x.entity_id == "wife" for x in r.records_written)
    assert r.round_narrations["wife"]["narration"] == "I heard him call for old Gnaeus across the house."
    assert ("smrti.round_narration_missing", {"pulse_id": "p6", "player": "wife", "fallback": "tuple"}) in events


def test_a_name_in_a_player_s_narration_they_were_never_given_is_counted(monkeypatch) -> None:
    """Stonemarch (2026-09-27): Dunstan's narration named Turuko, Kobort and
    Zert, whom he knew only by description. Counted, never blocked: a name
    the player's own tuples never carried; one they carry is not."""
    from smritilila.smrti.pulse import run_smrti_pulse
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    out = {"narration": "You sit by the fire; old Gnaeus watches from the door.",
           "perceiver_tuples": [{"perceiver": "senator", "text": "I sat by the fire while an old groom watched from the door.", "weight": "ordinary"}],
           "scene_tuples": [{"scene": "house", "text": "A man sat by the fire; an old groom watched.", "present": ["senator"]}],
           "suggested_next_action": ""}
    mems = [_mem(1, "senator", 5, "I sat by the fire.", loc="house"), _mem(2, "gnaeus", 5, "I watched from the door.", loc="house")]
    run_smrti_pulse(_canon(), "senator", "sits", 6, "p6", mems, _FakeClient(out), None,
                    players=frozenset({"senator"}), playable=frozenset({"senator"}))
    hits = [f for e, f in events if e == "smrti.narration_named_unknown"]
    assert hits and hits[0]["player"] == "senator" and hits[0]["parts"] == ["Gnaeus"]
    events.clear()
    mems[0] = _mem(1, "senator", 5, "I sat by the fire and nodded to Gnaeus.", loc="house")
    run_smrti_pulse(_canon(), "senator", "sits", 6, "p6", mems, _FakeClient(out), None,
                    players=frozenset({"senator"}), playable=frozenset({"senator"}))
    assert not [f for e, f in events if e == "smrti.narration_named_unknown"]


def test_a_turn_that_moves_someone_it_was_not_shown_is_told_again_with_them_shown(monkeypatch) -> None:
    """Stonemarch t148: the Narrator wrote Elmo "waiting at the east track" from
    the plan in memory; the record had him at the trading post, so his move
    was refused and the narration kept him -- the players then saw a man who
    was not there. Now the pulse is narrated again with him shown where he
    is, so the telling can bring him properly or leave him."""
    from smritilila.smrti.pulse import run_smrti_pulse
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))

    class _Twice(_FakeClient):
        def __init__(self, outs):
            super().__init__(outs[0])
            self.outs = outs

        def complete_with_tool(self, system, user, **kw):
            self.calls.append((system, user))
            return self.outs[min(len(self.calls), len(self.outs)) - 1]

    first = {"narration": "You find old Gnaeus waiting by the fire.",
             "perceiver_tuples": [{"perceiver": "senator", "text": "I found old Gnaeus waiting for me by the fire.", "weight": "ordinary"}],
             "scene_tuples": [{"scene": "house", "text": "The man found the old groom by the fire.", "present": ["senator", "gnaeus"]}],
             "moves": [{"entity": "gnaeus", "into": "house"}], "suggested_next_action": ""}
    second = {"narration": "You wait by the fire; the old groom has not come.",
              "perceiver_tuples": [{"perceiver": "senator", "text": "I waited by the fire, and the old groom did not come.", "weight": "ordinary"}],
              "scene_tuples": [{"scene": "house", "text": "A man waited alone by the fire.", "present": ["senator"]}],
              "moves": [], "suggested_next_action": ""}
    client = _Twice([first, second])
    mems = [_mem(1, "senator", 5, "I sat by the fire.", loc="house"), _mem(2, "gnaeus", 4, "I brushed down the horses.", loc="court")]
    r = run_smrti_pulse(_canon(), "senator", "I wait by the fire, as we agreed.", 6, "p6", mems, client, None,
                        players=frozenset({"senator"}), playable=frozenset({"senator"}))
    assert any(e == "smrti.unshown_rerun" for e, _ in events)
    assert len(client.calls) == 2 and "NOT HERE, SHOWN WHERE THEY ARE" in client.calls[1][1]
    assert r.narration == "You wait by the fire; the old groom has not come."


def test_the_narrator_states_what_is_in_doubt_and_its_better_and_worse_before_it_narrates() -> None:
    """Marc, 2026-09-28: "That will provide an anchor for it to use in interpreting the luck draw". The answer's
    first field after sankalpa is in_doubt -- what each acting player attempts that is uncertain, and what better and worse would
    look like within it -- and the fortune section says the draw chooses between them. Replayed on the second model, judged
    blind: invented setbacks on bad draws 9/28 -> 3-6/25; good draws in doubt better 2/26 -> 5/25."""
    from smritilila.smrti.narrator import SCHEMA, SYSTEM, round_schema
    for schema in (SCHEMA, round_schema(SCHEMA)):
        # v5 §28: saṅkalpa first of all, then in_doubt -- and each row commits what happens under its own fortune
        assert list(schema["properties"])[:2] == ["sankalpa", "in_doubt"] and {"sankalpa", "in_doubt"} <= set(schema["required"])
        item = schema["properties"]["in_doubt"]["items"]
        assert set(item["required"]) == {"player", "uncertain", "better", "worse", "fortune", "so_it_goes"}
    assert "Before the narration, fill `in_doubt`" in SYSTEM and "The fortune chooses between those" in SYSTEM


def test_a_world_is_born_with_its_earth_beneath_every_place() -> None:
    """Earth is the world at large (Marc, 2026-09-30): the root beneath every
    place, holding all that can come from it, a perceiver from birth -- its
    first memory the bhuvana, or the setting in a world that keeps no age."""
    from smritilila.smrti.genesis import ITIHASA, build
    spec = {
        "name": "t", "setting": "A ferry crossing on a slow river.", "scene_owner": "a", "playable": ["a"], "entry_points": ["p"],
        "places": {"vale": {"label": "the river vale", "scene": False, "tuple": "A long green vale."},
                   "p": {"label": "the landing", "scene": True, "in": "vale", "tuple": "A landing."}},
        "people": {"a": {"label": "A", "place": "p", "tuples": ["I came here."]}},
    }
    ents, mems, config = build(spec, lambda text: [0.0])
    earth = next(e for e in ents if e.origin == ITIHASA)
    assert config["earth"] == earth.id
    assert any(m.entity_id == earth.id and m.tier == "PRIMARY" and "ferry crossing" in m.experience for m in mems)
    placed = {r.entity_id: r.value for r in config["records"] if r.kind == "kosha"}
    vale = next(e.id for e in ents if e.labels[0] == "the river vale")
    assert placed[vale] == earth.id and earth.id not in placed                    # the top place in the earth; the earth in nothing
    ents, mems, config = build(spec, lambda text: [0.0], bhuvana="THE YEAR. 1928, spring. THE REGIONS. Essex County.")
    earth = next(e for e in ents if e.origin == ITIHASA)
    assert any(m.entity_id == earth.id and m.experience.startswith("THE YEAR. 1928") for m in mems)


def test_what_a_pulse_leaves_in_nothing_is_in_the_earth(monkeypatch) -> None:
    """Nothing is declared in nothing any more: a world's earth holds whatever
    a telling places nowhere -- still in no scene (the earth perceives, and is
    no one's scene), so a telling that leaves someone there is told again."""
    from smritilila.smrti.genesis import ITIHASA
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: None)
    canon = _canon()
    canon.entities["earth"] = Entity("earth", ["the world at large"], 0, ITIHASA)
    mems = [_mem(1, "senator", 5, "I stand in the garden court."),
            Memory("e0", "earth", frozenset(), "A Roman house in the Augustan age.", "PRIMARY", 3, "nowhere", 0, "genesis", None)]
    client = _FakeClient({"narration": "x", "perceiver_tuples": [{"perceiver": "senator", "text": "I thought of the far hills.", "salience": 4}],
                          "new_entities": [{"id": "NEW:far_hills", "name": "the far hills", "in": ""}],
                          "scene_tuples": [{"scene": "house", "text": "A man looked out over the court.", "present": ["senator"]}],
                          "suggested_next_action": ""})
    r = run_smrti_pulse(canon, "senator", "Looks out.", 6, "p6", mems, client, None)
    hills = next(e.id for e in r.new_entities if e.labels[0] == "the far hills")
    assert [x.value for x in r.records_written if x.kind == "kosha" and x.entity_id == hills] == ["earth"]


def test_a_world_is_born_with_a_clock() -> None:
    """2026-09-30, the sample world: nothing gave a world a clock at birth --
    the live worlds had theirs set by hand -- so a new world (an author's
    too) kept no time at all. The seed says when the world opens: a day of
    our calendar only where the world is dated in it, and the hour; the
    engine makes the clock. Undated, its days are counted; no hour, 8:00."""
    from smritilila.smrti.bija import as_spec
    from smritilila.smrti.kala import CONFIRM_MINUTES
    base = {"name": "The Mill Ford", "setting": "A mill.", "opening": "You stand on the ford.", "player": "wren",
            "places": [{"key": "ford", "label": "the mill ford", "in": "", "scene": True, "tuple": "A shallow ford below the mill, a road going north."}],
            "people": [{"key": "wren", "label": "Wren Miller", "place": "ford",
                        "tuples": [{"text": "I came down to the ford this morning to wait for the trader.", "weight": "ordinary", "use": "none"}]}]}
    clock = lambda **k: as_spec({**base, **k})["view"]["world_data"]["clock"]
    assert clock(date="1120-06-21", hour=19) == {"epoch": "1120-06-21T00:00", "start": 19 * 60, "confirm_minutes": CONFIRM_MINUTES}
    assert clock(hour=23) == {"calendar": "days", "start": 23 * 60, "confirm_minutes": CONFIRM_MINUTES}   # not dated in ours
    assert clock(date="the ninetieth year after the Sundering") == {"calendar": "days", "start": 8 * 60, "confirm_minutes": CONFIRM_MINUTES}
    assert clock(date="1928-04-27", hour=31)["start"] == 8 * 60                                          # no such hour: morning
    assert clock()["calendar"] == "days"


def test_a_new_world_asks_before_a_jump_of_more_than_thirty_minutes():
    """Marc, 2026-10-16: "let's adjust the jump ahead to moves greater than 20 minutes for all worlds", then "actually let's change all to 30 to match
    stonemarch". The threshold is `elapsed > confirm_minutes`, so a pulse of exactly thirty minutes goes through and one of thirty-one asks. A world is born
    with it (bija's clock); a live world keeps the value in its own clock, which this does not reach (Stonemarch's was already 30, the others' 10)."""
    from smritilila.smrti.bija import as_spec
    from smritilila.smrti.kala import CONFIRM_MINUTES
    assert CONFIRM_MINUTES == 30
    born = as_spec({"name": "The Mill Ford", "setting": "A mill.", "opening": "You stand on the ford.", "player": "wren",
                    "places": [{"key": "ford", "label": "the mill ford", "in": "", "scene": True, "tuple": "A shallow ford below the mill."}],
                    "people": [{"key": "wren", "label": "Wren Miller", "place": "ford",
                                "tuples": [{"text": "I came down to the ford this morning.", "weight": "ordinary", "use": "none"}]}]})
    assert born["view"]["world_data"]["clock"]["confirm_minutes"] == 30
    assert not (30 > CONFIRM_MINUTES) and (31 > CONFIRM_MINUTES)         # the rule asks only when elapsed is greater


def test_a_worlds_line_must_come_back_whole() -> None:
    """The sample world's line came back cut short when a model's thinking spent
    its budget ("You walk broken islands held by wary ferry"), and was kept.
    A line that does not end as a sentence ends is refused."""
    from smritilila.smrti.bija import describe_root

    class Client:
        def __init__(self, text): self.text, self.max_tokens = text, None
        def complete(self, system, user, *, max_tokens=8192):
            self.max_tokens = max_tokens
            return self.text
    with pytest.raises(ValueError):
        describe_root(Client("You walk broken islands held by wary ferry"), "The Braided Isles", "An archipelago.")
    c = Client("Broken islands, each with its own law, and the ferrymen who know the crossings.")
    assert describe_root(c, "The Braided Isles", "An archipelago.").endswith("crossings.") and c.max_tokens >= 2000


def _laws(places):
    from smritilila.smrti.genesis import clause_records
    ids = {k: f"id_{k}" for k in places}
    return {r.entity_id[3:]: r.value for r in clause_records(places, ids, 0, "genesis")}


def test_a_law_every_place_in_a_region_carries_is_the_regions() -> None:
    """The sample world (2026-09-30): Thessel's drift was written on each of
    its scenes, never on the island -- so a landing made in play, inside the
    island, stood under no law, and the nested scenes showed it twice. A law
    that every place directly in a region carries (two or more) is the
    region's; a clause its container already carries is inherited, not
    repeated."""
    drift = "Things left unwatched slowly drift from their form."
    thessel = {"island": {"in": "", "scene": False}, "folds": {"in": "island", "clause": drift},
               "precinct": {"in": "island", "clause": drift}, "sanctum": {"in": "precinct", "clause": drift},
               "landing": {"in": "precinct", "clause": drift}}
    assert _laws(thessel) == {"island": drift}
    # one place in the island carries it: the engine cannot tell the island's law from that place's -- it stays there,
    # and what that place holds inherits it rather than repeating it (Corran: the square, its hall and quay)
    weight = "Weight follows the giver's intent."
    corran = {"island": {"in": "", "scene": False}, "square": {"in": "island", "clause": weight},
              "hall": {"in": "square", "clause": weight}, "quay": {"in": "square", "clause": weight}}
    assert _laws(corran) == {"square": weight}


def test_a_places_own_law_stays_its_own() -> None:
    """Not every clause in a region is the region's: the haunted chapel in a
    village, or two scenes with laws of their own, keep them."""
    village = {"village": {"in": "", "scene": False}, "chapel": {"in": "village", "clause": "The dead speak here."},
               "green": {"in": "village"}}
    assert _laws(village) == {"chapel": "The dead speak here."}
    reach = {"reach": {"in": "", "scene": False}, "crag": {"in": "reach", "clause": "Salt pools strip tallow."},
             "cairn": {"in": "reach", "clause": "The white stone drinks the damp."}}
    assert _laws(reach) == {"crag": "Salt pools strip tallow.", "cairn": "The white stone drinks the damp."}
    # a place already under its own different law keeps it beneath the region's
    mixed = {"isle": {"in": "", "scene": False}, "a": {"in": "isle", "clause": "X holds."}, "b": {"in": "isle", "clause": "X holds."},
             "c": {"in": "a", "clause": "Y holds here."}}
    assert _laws(mixed) == {"isle": "X holds.", "c": "Y holds here."}


def test_the_edge_is_shown_unsorted() -> None:
    """2026-10-03 (the nirguna audit): an expansion listed everything at the
    edge that was not a place under THE PEOPLE -- an island, a wallet. The
    engine does not say which are people (v5 §1): the edge is who and what stands."""
    from smritilila.smrti import bija
    asked: list[str] = []

    class Client:
        def complete_with_tool(self, system, user, **k):
            asked.append(user)
            return {"places": [], "people": []}
    bija.expand(Client(), None, {"quay": ("the long quay", "scene")}, [("the Island of Harrow", "nowhere"), ("Jago Kane", "quay")])
    assert "WHO AND WHAT STANDS AT THAT EDGE:\n  the Island of Harrow @ nowhere\n  Jago Kane @ quay" in asked[0]
    assert "THE PEOPLE AT THAT EDGE" not in asked[0]
