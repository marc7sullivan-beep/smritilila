# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Kośa: what is in what (Design Principles v5 §3, §9-§12, §16-§18, §26, §29, §31-§36, §38).

Replaces v3.10's tests (a place's scene was the root of its kośa chain;
v4 makes containment one relation for everything, recorded, and the
scene the nearest container up the chain that is one). The scenarios
v3 tested through actor_stands, came_along, things and the label guards
-- the crossing's 323, 337 and 339, Ravenmoor 212-275 and 342-384 -- are
restated here against moves, records and declared names.
"""
from __future__ import annotations

import numpy as np

from smritilila import adhisthana
from smritilila import pravrtti
from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import NOWHERE, retrieve
from smritilila.smrti.kosha import Kosha
from smritilila.smrti.log import build_log


def _mem(i, who, t, text, loc, tier="PRIMARY", subjects=(), use=None) -> Memory:
    return Memory(f"m{i}", who, frozenset(subjects), text, tier, 4, loc, t, f"p{t}", None, use)


def _rec(i, eid, into, t, kind="kosha", note=None) -> Record:
    return Record(f"r{i}", kind, eid, into, t, f"p{t}", note)


def _hospital() -> Maya:
    return Maya(world_time=384, entities={
        "arkham": Entity("arkham", ["Arkham"], 0, "genesis"),                                   # a region: no scene bit
        "hospital": Entity("hospital", ["St. Mary's Hospital"], 254, "smrti", avarana="scene"),
        "ward": Entity("ward", ["st marys hospital ward"], 342, "smrti", avarana="scene"),
        "bed": Entity("bed", ["a hospital bed"], 342, "smrti"),                                  # not a scene
        "nathaniel": Entity("nathaniel", ["Nathaniel Thompson"], 0, "genesis"),
        "corwin": Entity("corwin", ["corwin biologist"], 77, "smrti"),
        "heart": Entity("heart", ["Corwin's heart"], 342, "smrti"),
    })


def _hospital_records() -> list[Record]:
    return [_rec(1, "hospital", "arkham", 254), _rec(2, "ward", "hospital", 342), _rec(3, "bed", "ward", 342),
            _rec(4, "corwin", "bed", 344), _rec(5, "nathaniel", "ward", 383), _rec(6, "heart", "corwin", 342)]


def _hospital_memories() -> list[Memory]:
    return [_mem(1, "corwin", 344, "I lay back in the bed.", "ward"), _mem(2, "nathaniel", 383, "I came back to tell Corwin.", "ward")]


# -- the index -------------------------------------------------------------


def test_the_scene_is_the_nearest_container_up_the_chain_that_is_one() -> None:
    k = Kosha(_hospital(), _hospital_memories(), _hospital_records())
    assert k.chain("corwin", 385) == ["bed", "ward"]
    assert k.scene_of("corwin", 385) == "ward" and k.scene_of("nathaniel", 385) == "ward"
    assert k.scene_of("hospital", 385) == "hospital"          # a scene is its own
    assert k.scene_of("arkham", 385) is None                   # a region holds scenes and is none
    # Corwin in the bed and Nathaniel beside it share the ward (v5 §11) -- Ravenmoor 384
    assert k.occupants("ward", 385) == {"corwin", "nathaniel"}
    # the heart is in the ward's scene through Corwin, and never present: it has no experience tuple (v5 §4)
    assert k.scene_of("heart", 385) == "ward" and "heart" not in k.occupants("ward", 385)
    # neighbours are the tree (v5 §12): the ward's parent is the hospital
    assert k.neighbours("ward", 385) == ["hospital"] and k.neighbours("hospital", 385) == ["ward"]


def test_a_move_is_a_later_record_and_carrying_is_the_chain() -> None:
    recs = _hospital_records() + [_rec(7, "bed", "corridor", 390)]
    canon = _hospital()
    canon.entities["corridor"] = Entity("corridor", ["the corridor"], 390, "smrti", avarana="scene")
    k = Kosha(canon, _hospital_memories(), recs)
    assert k.scene_of("corwin", 389) == "ward"
    assert k.scene_of("corwin", 391) == "corridor"      # the bed wheeled out takes him with it; nothing written for him
    assert k.scene_of("nathaniel", 391) == "ward"


def test_a_cycle_is_refused_by_inside() -> None:
    k = Kosha(_hospital(), _hospital_memories(), _hospital_records())
    assert k.inside("corwin", "ward", 385)       # the ward is up Corwin's chain: moving the ward into him is a cycle
    assert not k.inside("ward", "corwin", 385)


def test_legacy_worlds_are_read_by_derivation_without_writes() -> None:
    """v5 Appendix B, corrected while building: a perceiver with no record is where
    its latest tuple is; v3.10's entity.kosha is a container; every v3
    place is a scene. Nothing is written."""
    canon = Maya(world_time=10, entities={
        "entrance": Entity("entrance", ["st marys hospital entrance"], 0, "smrti", avarana="building"),
        "ward": Entity("ward", ["st marys hospital ward"], 0, "smrti", avarana="building", kosha="entrance"),
        "yard": Entity("yard", ["the yard"], 0, "smrti", avarana="scene"),
        "corwin": Entity("corwin", ["corwin biologist"], 0, "smrti"),
    })
    k = Kosha(canon, [_mem(1, "corwin", 5, "In bed.", "ward"), _mem(2, "corwin", 3, "Earlier, outside.", "yard")])
    assert k.is_scene("ward") and k.is_scene("yard") and not k.is_scene("corwin")
    assert k.scene_of("corwin", 6) == "ward" and k.scene_of("corwin", 4) == "yard"
    assert k.parent_scene("ward", 6) == "entrance"
    # a record outranks the derivation; a later tuple somewhere else outranks the record
    k = Kosha(canon, [_mem(1, "corwin", 5, "In bed.", "ward"), _mem(3, "corwin", 8, "Out in the yard.", "yard")],
              [_rec(1, "corwin", "ward", 6)])
    assert k.scene_of("corwin", 7) == "ward" and k.scene_of("corwin", 9) == "yard"


def test_names_are_exact_and_a_hyphen_joins_a_name() -> None:
    canon = _hospital()
    canon.entities["ruth"] = Entity("ruth", ["Ruth Marsh-Corwin"], 0, "genesis")
    k = Kosha(canon, records=[_rec(9, "corwin", "Corwin", 80, kind="name"), _rec(10, "corwin", "Doc", 90, kind="name")])
    assert k.named("corwin") == {"corwin"} and k.named("DOC") == {"corwin"}
    # legacy names: the label whole, and its capitalised proper names
    assert "ruth marsh-corwin" in k.names_of("ruth") and "ruth" in k.names_of("ruth")
    assert k.names_in("I asked Ruth Marsh-Corwin about the ledger.", ["corwin", "ruth"]) == {"ruth": {"ruth", "ruth marsh-corwin", "marsh-corwin"}} or \
        set(k.names_in("I asked Ruth Marsh-Corwin about the ledger.", ["corwin", "ruth"])) == {"ruth"}
    assert set(k.names_in("Doc said Corwin's bed was empty.", ["corwin", "ruth"])) == {"corwin"}


# -- the pulse ---------------------------------------------------------------


class _FakeClient:
    def __init__(self, out: dict) -> None:
        self.out = out
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, *, max_tokens=8192):  # pragma: no cover
        raise NotImplementedError

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
        self.calls.append((system, user))
        return self.out


def _crossing() -> Maya:
    return Maya(world_time=10, entities={
        "east": Entity("east", ["the east landing"], 0, "genesis", avarana="open"),
        "west": Entity("west", ["the west landing"], 0, "genesis", avarana="open"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
        "ilo": Entity("ilo", ["Ilo, the ferryman"], 0, "genesis"),
        "trav": Entity("trav", ["a woman traveler"], 0, "smrti"),
    })


def _pulse(monkeypatch, canon, actor, action, t, mems, out, **kw):
    from smritilila.smrti.pulse import run_smrti_pulse
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    client = _FakeClient(out)
    r = run_smrti_pulse(canon, actor, action, t, f"p{t}", mems, client, None, **kw)
    return r, events, client


def _where(canon, mems, recs, who, t):
    return Kosha(canon, mems, recs).scene_of(who, t) or NOWHERE


def test_the_ferry_crossing_moves_who_was_moved_and_the_watcher_stays(monkeypatch) -> None:
    """Vadhu 323: Teo poled the traveler across; the Narrator listed Ilo,
    watching from the east bank, present at the west landing. v4: moves
    say who went; a roster entry from the scene the actor left is refused."""
    canon = _crossing()
    mems = [_mem(1, "teo", 5, "I ran down to the landing.", "east"), _mem(2, "ilo", 5, "I followed the boy down.", "east"),
            _mem(3, "trav", 5, "I waited by the stone.", "east")]
    out = {"narration": "You pole across.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "I poled us across above the middle stone.", "salience": "ordinary", "use": "none"},
                                {"perceiver": "trav", "text": "The boy poled me across quick and clean.", "salience": "ordinary"},
                                {"perceiver": "ilo", "text": "I watched the boy push off and take the line I taught him.", "salience": "ordinary"}],
           "moves": [{"entity": "teo", "into": "west"}, {"entity": "trav", "into": "west"}],
           "scene_tuples": [{"scene": "west", "text": "The ferry came in and tied off; the ferryman watched from across the water.", "present": ["teo", "trav", "ilo"]},
                            {"scene": "east", "text": "The ferry pushed off with the boy and a woman; the old man stayed on the bank.", "present": ["ilo"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "teo", "poles the traveler across", 6, mems, out)
    recs = r.records_written
    after = mems + r.memories_written
    assert _where(canon, after, recs, "teo", 7) == "west" and _where(canon, after, recs, "trav", 7) == "west"
    assert _where(canon, after, recs, "ilo", 7) == "east"
    assert {m.entity_id: m.location_id for m in r.memories_written if m.tier == "PRIMARY"} == {"teo": "west", "trav": "west", "ilo": "east"}
    assert any(e == "smrti.roster_from_origin_refused" and f["entity"] == "ilo" for e, f in events)
    rows = {(m.entity_id, frozenset(m.subject_ids)) for m in r.memories_written if m.tier == "AMBIENT"}
    assert ("west", frozenset({"teo", "trav"})) in rows and ("east", frozenset({"ilo"})) in rows   # present is containment
    assert r.scene == "west"


def test_moving_into_nothing_leaves_the_bank_behind(monkeypatch) -> None:
    """Vadhu 337: Teo pushed off alone -- in nothing, nobody present; Ilo's
    own tuple is filed at the east bank where he is."""
    canon = _crossing()
    mems = [_mem(1, "teo", 5, "I brought the cart woman in.", "east"), _mem(2, "ilo", 5, "I caught the line.", "east")]
    out = {"narration": "You push off alone.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "I pushed off alone into the dark.", "salience": "ordinary"},
                                {"perceiver": "ilo", "text": "I watched the boat go and stayed on the bank.", "salience": "ordinary"}],
           "moves": [{"entity": "teo", "into": ""}],
           "scene_tuples": [{"scene": "east", "text": "The apprentice pushed off alone; the ferryman stayed on the bank.", "present": ["ilo"]}],
           "suggested_next_action": ""}
    r, _, _ = _pulse(monkeypatch, canon, "teo", "poles back out alone", 6, mems, out)
    by = {m.entity_id: m for m in r.memories_written}
    assert by["teo"].location_id == NOWHERE and by["teo"].subject_ids == frozenset()
    assert by["ilo"].location_id == "east"
    assert r.scene is None  # the page says what "nowhere" is, not the engine (v5 §7)


def test_travelling_together_is_being_in_the_same_boat(monkeypatch) -> None:
    """Vadhu 339 restated: the tin peddler aboard with Teo mid-river. v3
    kept them together by a same-pulse rule; v4 by a container -- the
    ferry, minted as a scene, both moved into it, and the ferry's own
    moves carry them."""
    canon = _crossing()
    canon.entities["tin"] = Entity("tin", ["the tin peddler"], 0, "smrti")
    mems = [_mem(1, "teo", 5, "I held the ferry at the bank.", "east"), _mem(2, "tin", 5, "I waited on my knee.", "east")]
    out = {"narration": "You take him aboard and push off.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "I held the ferry and he climbed aboard; I pushed off.", "salience": "ordinary"},
                                {"perceiver": "tin", "text": "I climbed aboard and held my pots.", "salience": "ordinary"}],
           "new_entities": [{"id": "NEW:the_ferry", "name": "", "in": "", "scene": True}],
           "moves": [{"entity": "teo", "into": "NEW:the_ferry"}, {"entity": "tin", "into": "NEW:the_ferry"}],
           "scene_tuples": [{"scene": "east", "text": "The ferry took the tin-worker aboard and pushed off.", "present": []},
                            {"scene": "NEW:the_ferry", "text": "Two aboard, pushing out into the current.", "present": ["teo", "tin"]}],
           "suggested_next_action": ""}
    r, _, _ = _pulse(monkeypatch, canon, "teo", "takes the tin peddler aboard and pushes off", 6, mems, out)
    ferry = r.resolved_new["NEW:the_ferry"]
    assert canon.entities[ferry].avarana == "scene"
    after, recs = mems + r.memories_written, r.records_written
    assert _where(canon, after, recs, "teo", 7) == ferry == _where(canon, after, recs, "tin", 7)
    k = Kosha(canon, after, recs)
    assert k.occupants(ferry, 7) == {"teo", "tin"}
    # the ferry lands: one record for the boat, and both are at the west landing's scene? no -- in the boat, tied up there
    k.add(_rec(50, ferry, "west", 8))
    assert k.chain("tin", 9) == [ferry] and k.parent_scene(ferry, 9) == "west"


def test_the_engine_checks_structure_only(monkeypatch) -> None:
    canon = _hospital()
    mems = _hospital_memories()
    recs = _hospital_records()
    canon.entities["far"] = Entity("far", ["a man in Boston"], 0, "genesis")
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "nathaniel", "text": "I sat on the edge of Corwin's bed.", "salience": "ordinary"}],
           "moves": [{"entity": "nathaniel", "into": "bed"},       # a person into a bed: the Narrator's to judge, allowed
                     {"entity": "ward", "into": "bed"},             # the ward into a bed inside it: a cycle
                     {"entity": "far", "into": "ward"},             # never shown to this pulse
                     {"entity": "corwin", "into": "NOTANID"}],      # into nothing that exists: the container is minted first (v5.41)
           "scene_tuples": [{"scene": "ward", "text": "A visitor sat on the edge of the bed.", "present": ["nathaniel", "corwin"]}],
           "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", False)
    view = {"darsana": None, "becomings": {}, "world_data": {"sandhi": True}}    # as every live world (the put-back is `_mentions`)
    r, events, _ = _pulse(monkeypatch, canon, "nathaniel", "sits on the bed", 385, mems, out, records=recs, view=view)
    refused = next(f["refused"] for e, f in events if e == "smrti.move_refused")
    assert {(x["entity"], x["reason"]) for x in refused} == {("ward", "cycle"), ("far", "not_shown")}     # no unknown_container for a name (v5.41): it is minted
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.container_at("nathaniel", 386) == "bed" and k.scene_of("nathaniel", 386) == "ward"
    made = [e for e in r.new_entities if e.labels == ["NOTANID"]]
    assert len(made) == 1                                                       # the move made the container (v5.41), so the name exists
    assert [kw for e, kw in events if e == "smrti.move_undeclared" and kw["id"] == made[0].id]
    assert k.container_at("corwin", 386) == "bed"                               # nothing said what holds it, so Corwin was put back where he was
    assert k.container_at(made[0].id, 386) is None                              # and no holder was guessed for it


def test_an_npcs_turn_does_not_move_a_player(monkeypatch) -> None:
    """Refused and logged unless the player allows it (v5 §37, the ruling on the
    ontology flag of 2026-09-23): may another's turn carry a player along?"""
    canon = _crossing()
    mems = [_mem(1, "teo", 5, "I sat on the bank.", "east"), _mem(2, "ilo", 5, "I sat with the boy.", "east")]
    out = {"narration": "Ilo takes the boy across.",
           "perceiver_tuples": [{"perceiver": "ilo", "text": "I took the boy across myself.", "salience": "ordinary"}],
           "moves": [{"entity": "ilo", "into": "west"}, {"entity": "teo", "into": "west"}],
           "scene_tuples": [{"scene": "west", "text": "The old ferryman brought the boy across.", "present": ["ilo", "teo"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "ilo", None, 6, mems, out, players=frozenset({"teo"}))
    refused = next(f["refused"] for e, f in events if e == "smrti.move_refused")
    assert ("teo", "player_not_actor") in {(x["entity"], x["reason"]) for x in refused}
    assert _where(canon, mems + r.memories_written, r.records_written, "teo", 7) == "east"


def test_the_roster_adds_from_a_neighbour_and_never_overrides_a_move(monkeypatch) -> None:
    canon = Maya(world_time=10, entities={
        "yard": Entity("yard", ["the farmyard"], 0, "genesis", avarana="scene"),
        "house": Entity("house", ["the farmhouse"], 0, "genesis", avarana="scene"),
        "ruth": Entity("ruth", ["Ruth"], 0, "genesis"),
        "hartley": Entity("hartley", ["Hartley"], 0, "genesis"),
        "nathaniel": Entity("nathaniel", ["Nathaniel"], 0, "genesis"),
    })
    recs = [_rec(1, "house", "yard", 0)]
    mems = [_mem(1, "nathaniel", 5, "I sat at Ruth's table.", "house"), _mem(2, "ruth", 5, "I fed the hens.", "yard"),
            _mem(3, "hartley", 5, "I sat at the table too.", "house")]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "nathaniel", "text": "Ruth came in from the yard and Hartley went out.", "salience": "ordinary"}],
           "moves": [{"entity": "hartley", "into": "yard"}],
           "scene_tuples": [{"scene": "house", "text": "The woman came in from the yard; one man went out.", "present": ["nathaniel", "ruth", "hartley"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "nathaniel", "waits", 6, mems, out, records=recs)
    after, allrecs = mems + r.memories_written, recs + r.records_written
    assert _where(canon, after, allrecs, "ruth", 7) == "house"      # from the yard, a neighbour (the house's parent)
    assert _where(canon, after, allrecs, "hartley", 7) == "yard"    # the move stands; the roster does not undo it
    assert any(e == "smrti.roster_added" and f["entity"] == "ruth" for e, f in events)
    assert any(e == "smrti.roster_disagrees" and f["entity"] == "hartley" for e, f in events)


def test_a_location_named_as_existing_takes_an_id_and_a_container(monkeypatch) -> None:
    """v5 §10, Ravenmoor 206: "behind the old sheep pen, past the north tree
    line" -- an id and a container at first naming, a name declared."""
    canon = Maya(world_time=10, entities={
        "farm": Entity("farm", ["Ruth's farm"], 0, "genesis"),
        "kitchen": Entity("kitchen", ["Ruth's kitchen"], 0, "genesis", avarana="scene"),
        "ruth": Entity("ruth", ["Ruth"], 0, "genesis"), "nathaniel": Entity("nathaniel", ["Nathaniel"], 0, "genesis"),
    })
    recs = [_rec(1, "kitchen", "farm", 0)]
    mems = [_mem(1, "nathaniel", 5, "I asked Ruth where.", "kitchen"), _mem(2, "ruth", 5, "He asked me where.", "kitchen")]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "ruth", "text": "I told him: behind the old sheep pen, past the north tree line.", "salience": "weighty", "use": "key"},
                                {"perceiver": "nathaniel", "text": "She said behind the old sheep pen.", "salience": "weighty", "use": "key"}],
           "new_entities": [{"id": "NEW:old_sheep_pen", "name": "the old sheep pen", "in": "farm", "scene": False}],
           "scene_tuples": [{"scene": "kitchen", "text": "The woman named a place behind the old sheep pen.", "present": ["ruth", "nathaniel"]}],
           "suggested_next_action": ""}
    r, _, _ = _pulse(monkeypatch, canon, "nathaniel", "asks where", 6, mems, out, records=recs)
    pen = r.resolved_new["NEW:old_sheep_pen"]
    assert canon.entities[pen].avarana is None and canon.entities[pen].labels == ["the old sheep pen"]
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.container_at(pen, 7) == "farm" and k.named("The Old Sheep Pen") == {pen}
    # the use rides on the tuples, and the scene's record carries the most useful of the pulse (v5 §3, §31)
    assert {m.entity_id: m.use for m in r.memories_written} == {"ruth": "key", "nathaniel": "key", "kitchen": "key"}


def test_a_declared_name_folds_onto_someone_known_and_only_them(monkeypatch) -> None:
    """v5 §17, Ravenmoor 381-384: "Professor Corwin" minted beside `corwin
    biologist`. Declared names, matched exactly, within acquaintance."""
    canon = _hospital()
    recs = _hospital_records() + [_rec(20, "corwin", "Professor Corwin", 100, kind="name")]
    mems = _hospital_memories() + [_mem(3, "nathaniel", 300, "Corwin read the transcript.", "ward", subjects=("corwin",))]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "nathaniel", "text": "Professor Corwin opened his eyes.", "salience": "weighty"},
                                {"perceiver": "NEW:professor_corwin", "text": "I opened my eyes on the ward.", "salience": "weighty"}],
           "new_entities": [{"id": "NEW:professor_corwin", "name": "Professor Corwin", "in": "bed"}],
           "scene_tuples": [{"scene": "ward", "text": "The patient opened his eyes.", "present": ["nathaniel", "NEW:professor_corwin"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "nathaniel", "sits by the bed", 385, mems, out, records=recs)
    assert r.new_entities == [] and r.resolved_new["NEW:professor_corwin"] == "corwin"
    assert ("smrti.name_fold", {"pulse_id": "p385", "handle": "NEW:professor_corwin", "name": "Professor Corwin", "onto": "corwin"}) in events
    # someone the actor never met is not folded onto: two Johns can exist -- flagged only
    canon2 = _hospital()
    r2, events2, _ = _pulse(monkeypatch, canon2, "nathaniel", "sits by the bed", 385, _hospital_memories()[1:],
                            dict(out, perceiver_tuples=out["perceiver_tuples"][:1]), records=recs[:3] + [recs[4], recs[6]])
    assert r2.resolved_new["NEW:professor_corwin"] != "corwin"
    assert any(e == "smrti.name_taken_elsewhere" for e, _ in events2)


def test_any_rerun_carries_the_names_the_first_telling_gave_that_another_bears(monkeypatch) -> None:
    """Stonemarch t133 (Marc: "The shadow is two ostlers!"): Dunstan walked from
    the trading post into the taproom, where Jakal stood unshown and Ostler
    Gundigoot stood too. The first telling declared a NEW Ostler Gundigoot;
    the arrival rerun spent the pulse's one rerun, and the second minted him
    again. Now the one rerun carries the borne name as well: shown it, the
    Narrator uses the Ostler already there. Two Johns can still exist -- the
    engine folds no person by name; the Narrator, shown, decides."""
    from smritilila.smrti.pulse import run_smrti_pulse
    canon = Maya(world_time=10, entities={
        "taproom": Entity("taproom", ["the taproom"], 0, "genesis", avarana="scene"),
        "post": Entity("post", ["the trading post"], 0, "genesis", avarana="scene"),
        "ostler": Entity("ostler", ["Ostler Gundigoot"], 0, "genesis"),
        "jakal": Entity("jakal", ["Jakal Hierophant"], 0, "genesis"),
        "dunstan": Entity("dunstan", ["Dunstan Mere"], 0, "genesis")})
    recs = [_rec(1, "ostler", "taproom", 0), _rec(2, "ostler", "Ostler Gundigoot", 0, kind="name"), _rec(3, "dunstan", "post", 0),
            _rec(4, "jakal", "taproom", 0)]
    mems = [_mem(1, "ostler", 5, "I drew ale in my taproom.", "taproom"), _mem(2, "dunstan", 5, "I paid at the post.", "post"),
            _mem(3, "jakal", 5, "I sat in the taproom.", "taproom")]
    seen, events = [], []

    class C:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            borne = "NAMES ALREADY BORNE" in user
            ostler = "ostler" if borne else "NEW:ostler_gundigoot"
            return {"narration": "You walk to the inn.",
                    "perceiver_tuples": [{"perceiver": "dunstan", "text": "I walked in and the innkeeper named his price.", "salience": "ordinary"},
                                         {"perceiver": ostler, "text": "A sellsword came in.", "salience": "ordinary"}],
                    "new_entities": [] if borne else [{"id": "NEW:ostler_gundigoot", "name": "Ostler Gundigoot", "in": "taproom", "scene": False}],
                    "moves": [{"entity": "dunstan", "into": "taproom"}], "scene_tuples": [], "suggested_next_action": ""}
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **k: events.append((e, k)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **k: None)
    r = run_smrti_pulse(canon, "dunstan", "I walk to the inn", 10, "p10", mems, C(), None, records=recs,
                        players=frozenset({"dunstan", "jakal"}), playable=frozenset({"dunstan", "jakal"}))
    kinds = {e: k for e, k in events}
    assert len(seen) == 2 and "smrti.arrival_rerun" in kinds and kinds["smrti.arrival_rerun"]["taken"]
    assert "NAMES ALREADY BORNE" in seen[1] and "ostler" in seen[1].split("NAMES ALREADY BORNE")[1]
    assert r.new_entities == []                                                       # no second Ostler minted
    assert [e for e in canon.entities if "gundigoot" in canon.entities[e].labels[0].lower()] == ["ostler"]


def test_aliases_are_appended_as_name_records(monkeypatch) -> None:
    canon = _hospital()
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "nathaniel", "text": "I called him Doc and he almost smiled.", "salience": "ordinary"}],
           "aliases": [{"entity": "corwin", "name": "Doc"}],
           "scene_tuples": [{"scene": "ward", "text": "The visitor called the patient Doc.", "present": ["nathaniel", "corwin"]}],
           "suggested_next_action": ""}
    r, _, _ = _pulse(monkeypatch, canon, "nathaniel", "calls him Doc", 385, _hospital_memories(), out, records=_hospital_records())
    assert [(x.kind, x.entity_id, x.value, x.note) for x in r.records_written if x.kind == "name"] == [("name", "corwin", "Doc", "alias")]


def test_a_players_own_words_are_kept_with_their_provenance(monkeypatch) -> None:
    """v5 §35: the action as sent, typed or the page's suggestion, beside
    the player's tuple, and shown back to the Narrator next time."""
    canon = _hospital()
    out = {"narration": "You tell him.",
           "perceiver_tuples": [{"perceiver": "nathaniel", "text": "I told Corwin the creek elbow was done.", "salience": "weighty"}],
           "scene_tuples": [{"scene": "ward", "text": "The visitor spoke low to the patient.", "present": ["nathaniel", "corwin"]}],
           "suggested_next_action": ""}
    r, _, _ = _pulse(monkeypatch, canon, "nathaniel", "tell Corwin it's done", 385, _hospital_memories(), out,
                     records=_hospital_records(), players=frozenset({"nathaniel"}), action_source="suggested")
    act = [x for x in r.records_written if x.kind == "action"]
    assert [(x.entity_id, x.value, x.note, x.world_time) for x in act] == [("nathaniel", "tell Corwin it's done", "suggested", 385)]
    mems = _hospital_memories() + r.memories_written
    _, _, client = _pulse(monkeypatch, canon, "nathaniel", "waits", 386, mems, out,
                          records=_hospital_records() + r.records_written, players=frozenset({"nathaniel"}))
    user = client.calls[0][1]
    assert "their own words that pulse (the page's suggestion, accepted): \"tell Corwin it's done\"" in user


def test_no_one_elses_turn_speaks_for_a_player_and_a_witness_inherits_use(monkeypatch) -> None:
    """v5 §33 C (Ravenmoor 301: "Hartley, get over here"), and v5 §31: the player
    who overheard is handed the scene's record with the most useful USE
    the others gave, at an ordinary weight."""
    canon = _hospital()
    out = {"narration": "Corwin speaks.",
           "perceiver_tuples": [{"perceiver": "corwin", "text": "I told him the third site is under the creek elbow.", "salience": "weighty", "use": "key"}],
           "scene_tuples": [{"scene": "ward", "text": "The patient spoke of a site under the creek elbow; the visitor answered.",
                             "said": "Tell me more", "said_by": "nathaniel", "present": ["corwin", "nathaniel"]}],
           "suggested_next_action": ""}
    out["narration"] = 'Corwin speaks. "Tell me more," Nathaniel says.'
    r, events, _ = _pulse(monkeypatch, canon, "corwin", None, 385, _hospital_memories(), out,
                          records=_hospital_records(), players=frozenset({"nathaniel"}))
    assert any(e == "smrti.player_said_dropped" for e, _ in events)
    witnessed = next(m for m in r.memories_written if m.entity_id == "nathaniel")
    assert witnessed.use == "key" and witnessed.salience == 5 and "Tell me more" not in witnessed.experience


def test_a_missing_actor_tuple_is_the_narration_as_narrated_and_marked(monkeypatch) -> None:
    """v5 history wrote the fallback in the first person by an English table, which
    also rewrote others' quoted speech ("Did you see it?" became "Did I
    see it?"). v5 §32 audit: stored as narrated, never rewritten, and marked
    so the Narrator reads it for what it is."""
    canon = _hospital()
    out = {"narration": "You sit by the bed and watch him sleep.", "perceiver_tuples": [],
           "scene_tuples": [{"scene": "ward", "text": "The visitor sat by the bed while the patient slept.", "present": ["nathaniel", "corwin"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "nathaniel", "sits", 385, _hospital_memories(), out, records=_hospital_records())
    assert r.memories_written[0].experience == "You sit by the bed and watch him sleep."
    assert [(x.kind, x.entity_id, x.value) for x in r.records_written if x.kind == "mark"] == [("mark", "nathaniel", "as_narrated")]
    log, _ = build_log(canon, _hospital_memories() + r.memories_written, _hospital_records() + r.records_written)
    assert any(t.memory.startswith("[as narrated to them") for t in log.stream("nathaniel"))


def test_retrieval_reaches_a_declared_name_only_among_those_met() -> None:
    """v5 §17 item 5: a name in the action brings its bearer -- someone the
    actor's stream has met, never a world-wide search."""
    canon = Maya(world_time=40, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene"),
        "market": Entity("market", ["the hill market"], 0, "genesis", avarana="scene"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
        "sela": Entity("sela", ["a pedlar with a blue pack"], 0, "genesis"),
        "sela2": Entity("sela2", ["a woman at the market"], 0, "genesis"),
    })
    recs = [_rec(1, "sela", "Sela", 0, kind="name"), _rec(2, "sela2", "Sela", 0, kind="name")]
    mems = [_mem(1, "teo", 10, "The pedlar and I spoke.", "inn", subjects=("sela",)), _mem(2, "sela", 10, "The boy spoke to me.", "inn"),
            _mem(3, "sela", 20, "I walked to the market.", "market"), _mem(4, "sela2", 20, "I sold eggs.", "market"),
            _mem(5, "teo", 30, "I sat alone.", "inn")]
    log, _ = build_log(canon, mems, recs)
    ctx = retrieve(log, "teo", 31, None, k_place=3, k_sim=0, action="Go and look for Sela.")
    ids = {t.perceiver for t in ctx.tuples}
    assert "sela" in ids and "sela2" not in ids
    assert "market" in ids   # item 3: the name of the scene she is now in


def test_a_thing_declared_and_moved_into_the_same_container_is_recorded_once(monkeypatch) -> None:
    """Live, the crossing 466: the tin cup declared in Teo and moved into
    Teo in one pulse took two records at one tick."""
    canon = _crossing()
    mems = [_mem(1, "teo", 5, "I crouched by the fire-ring.", "east")]
    out = {"narration": "You pocket the cup.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "I slipped the rusted cup into my coat.", "salience": "ordinary"}],
           "new_entities": [{"id": "NEW:rusted_tin_cup", "name": "", "in": "teo"}],
           "moves": [{"entity": "NEW:rusted_tin_cup", "into": "teo"}],
           "scene_tuples": [{"scene": "east", "text": "The boy pocketed an old cup.", "present": ["teo"]}],
           "suggested_next_action": ""}
    r, _, _ = _pulse(monkeypatch, canon, "teo", "pockets the cup", 6, mems, out)
    cup = r.resolved_new["NEW:rusted_tin_cup"]
    assert [(x.entity_id, x.value) for x in r.records_written if x.kind == "kosha"] == [(cup, "teo")]


def test_a_scenes_own_record_never_says_where_the_scene_is() -> None:
    """Live, the crossing 469: the overhang (in the quarry track by its
    record) was asked whether anyone came, and its own tuples, filed at
    itself, outranked the record -- it became its own container, lost its
    parent, left the track's neighbours, and was minted again at 470."""
    canon = Maya(world_time=470, entities={
        "track": Entity("track", ["old quarry track"], 0, "smrti", avarana="scene"),
        "overhang": Entity("overhang", ["the overhang"], 467, "smrti", avarana="scene"),
    })
    mems = [_mem(1, "overhang", 469, "The overhang sat quiet.", "overhang", tier="AMBIENT"),
            _mem(2, "overhang", 469, "I sat quiet above the cut.", "overhang")]
    k = Kosha(canon, mems, [_rec(1, "overhang", "track", 467)])
    assert k.container_at("overhang", 471) == "track"
    assert k.neighbours("track", 471) == ["overhang"]


def test_a_new_scene_is_not_asked_whether_anyone_came_the_tick_it_is_minted() -> None:
    from smritilila import main
    canon = Maya(world_time=469, entities={
        "track": Entity("track", ["old quarry track"], 0, "smrti", avarana="scene"),
        "overhang": Entity("overhang", ["the overhang"], 467, "smrti", avarana="scene"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
    })
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["track"], playable_entity_ids=["teo"])
    world.world_data["pacing"] = {"arrivals_on_traffic": False}  # asked on the clock: the ask itself, not its traffic gate
    world.recent_memories = [_mem(1, "teo", 468, "I walked the track.", "track")]
    world.records = [_rec(1, "overhang", "track", 467)]
    assert "overhang" not in pravrtti._smrti_exploration_pool(world)
    canon.world_time = 480
    assert "overhang" in pravrtti._smrti_exploration_pool(world)


def test_a_place_declared_by_the_same_name_in_the_same_container_is_the_one_there(monkeypatch) -> None:
    """The crossing 497: Bren's turn minted "the mill path" in the road a
    second time -- he had never been on the first, so acquaintance could
    not fold it. A thing or place has no stream to tell two apart."""
    canon = Maya(world_time=497, entities={
        "road": Entity("road", ["the road"], 0, "smrti", avarana="scene"),
        "lane": Entity("lane", ["the lane"], 0, "smrti", avarana="scene"),
        "path": Entity("path", ["the mill path"], 489, "smrti"),
        "bren": Entity("bren", ["Bren"], 480, "smrti"),
    })
    recs = [_rec(1, "lane", "road", 480), _rec(2, "path", "road", 489), _rec(3, "path", "the mill path", 489, kind="name")]
    mems = [_mem(1, "bren", 490, "I worked the tin.", "lane")]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "bren", "text": "I walked up the mill path toward the storehouse.", "salience": "ordinary"}],
           "new_entities": [{"id": "NEW:mill_path", "name": "the mill path", "in": "road", "scene": False}],
           "moves": [{"entity": "bren", "into": "NEW:mill_path"}],
           "scene_tuples": [{"scene": "lane", "text": "The tin-worker walked up the lane.", "present": []}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "bren", None, 497, mems, out, records=recs)
    assert r.new_entities == [] and r.resolved_new["NEW:mill_path"] == "path"
    # the road holds the lane, so the path is shown as held by it and known (v5 §26)
    assert any(e == "smrti.name_fold" for e, _ in events)
    # beyond what is shown as held, the same container still folds it
    monkeypatch.setattr("smritilila.smrti.context.HELD", 0)
    r, events, _ = _pulse(monkeypatch, canon, "bren", None, 497, mems, out, records=recs)
    assert r.new_entities == [] and r.resolved_new["NEW:mill_path"] == "path"
    assert any(e == "smrti.name_container_fold" for e, _ in events)
    monkeypatch.setattr("smritilila.smrti.context.HELD", 8)
    # a person by the same name in the same place is not folded: two Johns can exist
    # (the lane is on its own here, so the road's John is not shown to this pulse)
    canon.entities["john"] = Entity("john", ["John"], 0, "smrti")
    recs2 = recs[1:] + [_rec(4, "john", "road", 400)]
    mems2 = mems + [_mem(2, "john", 400, "I stood on the road.", "road")]
    out2 = dict(out, new_entities=[{"id": "NEW:john", "name": "John", "in": "road", "scene": False}], moves=[],
                perceiver_tuples=out["perceiver_tuples"] + [{"perceiver": "NEW:john", "text": "I nodded to the tin-worker on the road.", "salience": "passing"}])
    r2, _, _ = _pulse(monkeypatch, canon, "bren", None, 498, mems2, out2, records=recs2)
    assert r2.resolved_new["NEW:john"] != "john"


def test_a_chain_that_reaches_no_scene_has_one_its_topmost_container(monkeypatch) -> None:
    """v5 §11 (Marc, 2026-09-23): the crossing 513 -- four travellers moved
    into a path minted `scene: false` in nothing were each alone. The
    root of a scene-less chain is its occupants' scene; a perceiver is
    never someone else's scene."""
    canon = _crossing()
    canon.entities["cup"] = Entity("cup", ["a tin cup"], 0, "smrti")
    mems = [_mem(1, "teo", 5, "I stood on the bank.", "east"), _mem(2, "ilo", 5, "I stood with the boy.", "east"),
            _mem(3, "trav", 5, "I waited on the bank.", "east")]
    out = {"narration": "You walk up the path together.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "We walked up the path from the landing together.", "salience": "ordinary"},
                                {"perceiver": "ilo", "text": "I walked up beside the boy.", "salience": "ordinary"}],
           "new_entities": [{"id": "NEW:path_up", "name": "the path up", "in": "", "scene": False}],
           "moves": [{"entity": "teo", "into": "NEW:path_up"}, {"entity": "ilo", "into": "NEW:path_up"}],
           "scene_tuples": [{"scene": "east", "text": "Two walked up from the landing.", "present": ["trav"]},
                            {"scene": "NEW:path_up", "text": "Two came up the path together.", "present": ["teo", "ilo"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "teo", "walks up", 6, mems, out)
    path = r.resolved_new["NEW:path_up"]
    k = Kosha(canon, mems + r.memories_written, r.records_written)
    assert k.scene_of("teo", 7) == path and k.occupants(path, 7) == {"teo", "ilo"}
    assert {m.entity_id: m.location_id for m in r.memories_written if m.tier == "PRIMARY"} == {"teo": path, "ilo": path}
    assert any(m.entity_id == path and m.tier == "AMBIENT" for m in r.memories_written)  # the path keeps their record
    assert any(e == "smrti.root_as_scene" for e, _ in events)
    # a perceiver is no one's scene: the cup in Teo, Teo in nothing
    k.add(_rec(40, "cup", "teo", 7))
    k.add(_rec(41, "teo", None, 8))
    assert k.scene_of("cup", 9) is None and k.scene_of("teo", 9) is None


def test_another_turn_may_move_a_player_who_allows_it_and_their_autonomous_turn_is_marked(monkeypatch) -> None:
    """v5 §37: movable players are moved like anyone; an offline player's
    turn taken by the Narrator is recorded `autonomous` and shown to the
    Narrator later as not their words."""
    canon = _crossing()
    mems = [_mem(1, "teo", 5, "I sat on the bank.", "east"), _mem(2, "ilo", 5, "I sat with the boy.", "east")]
    out = {"narration": "Ilo takes the boy across.",
           "perceiver_tuples": [{"perceiver": "ilo", "text": "I took the boy across myself.", "salience": "ordinary"}],
           "moves": [{"entity": "ilo", "into": "west"}, {"entity": "teo", "into": "west"}],
           "scene_tuples": [{"scene": "west", "text": "The old ferryman brought the boy across.", "present": ["ilo", "teo"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "ilo", None, 6, mems, out, players=frozenset({"teo"}), movable=frozenset({"teo"}))
    assert not any(e == "smrti.move_refused" for e, _ in events)
    assert _where(canon, mems + r.memories_written, r.records_written, "teo", 7) == "west"
    # Teo offline, allowing it: the Narrator acts for him (not in `players`: voiced)
    out2 = {"narration": "Teo coils the rope.",
            "perceiver_tuples": [{"perceiver": "teo", "text": "I coiled the ferry rope and watched the far bank.", "salience": "passing"}],
            "scene_tuples": [{"scene": "west", "text": "The boy coiled the rope.", "present": ["teo", "ilo"]}], "suggested_next_action": ""}
    mems2 = mems + r.memories_written
    r2, events2, _ = _pulse(monkeypatch, canon, "teo", None, 7, mems2, out2, records=r.records_written,
                            players=frozenset(), playable=frozenset({"teo"}))
    act = [x for x in r2.records_written if x.kind == "action"]
    assert [(x.entity_id, x.note) for x in act] == [("teo", "autonomous")]
    _, _, client = _pulse(monkeypatch, canon, "teo", "look across", 8, mems2 + r2.memories_written, out2,
                          records=r.records_written + r2.records_written, players=frozenset({"teo"}), playable=frozenset({"teo"}))
    assert "not their words: the Narrator acted for them that pulse, while they were away" in client.calls[0][1]


def test_a_seed_states_its_tree_and_genesis_plants_it() -> None:
    """v5 §39 (bija rewritten 2026-09-23): a seed place says what it is in
    and whether it is a scene; a region is a container that is not one; a
    bad `in` or a loop is dropped, never guessed at."""
    from smritilila.smrti.bija import as_spec
    from smritilila.smrti.genesis import build
    out = {"name": "t", "setting": "s", "opening": "o", "player": "halla",
           "places": [
               {"key": "fjord", "label": "the head of the fjord", "in": "", "scene": False, "tuple": "A fjord's head under steep fells, three farms along its shore."},
               {"key": "farm", "label": "Halla's farm", "in": "fjord", "scene": False, "tuple": "A turf-walled farm at the fjord's head."},
               {"key": "hall", "label": "the hall at Halla's farm", "in": "farm", "scene": True, "tuple": "A long turf hall, the fire down its middle."},
               {"key": "yard", "label": "the home-field", "in": "farm", "scene": True, "tuple": "The home-field before the hall, walled in turf."},
               {"key": "ghost", "label": "a place in nowhere", "in": "no_such_key", "scene": True, "tuple": "A place whose container the seed never made."},
               {"key": "a", "label": "loop a", "in": "b", "scene": True, "tuple": "One half of a loop the seed should never hold."},
               {"key": "b", "label": "loop b", "in": "a", "scene": True, "tuple": "The other half of the loop the seed should never hold."},
           ],
           "people": [{"key": "halla", "label": "Halla", "place": "hall", "tuples": [{"text": "I held my dead husband's farm alone.", "weight": "marking"}]}]}
    spec = as_spec(out)
    assert spec["places"]["hall"] == {"label": "the hall at Halla's farm", "tuple": "A long turf hall, the fire down its middle.", "scene": True, "in": "farm"}
    assert "in" not in spec["places"]["ghost"]
    assert not ("in" in spec["places"]["a"] and "in" in spec["places"]["b"])  # the loop is broken
    ents, mems, _ = build(spec, lambda t: None)
    by = {e.labels[0]: e for e in ents}
    canon = Maya(world_time=1, entities={e.id: e for e in ents})
    k = Kosha(canon, mems)
    hall, yard, farm = by["the hall at Halla's farm"].id, by["the home-field"].id, by["Halla's farm"].id
    assert k.is_scene(hall) and not k.is_scene(farm) and by["Halla's farm"].avarana is None
    assert k.container_at(hall, 2) == farm and k.container_at(farm, 2) == by["the head of the fjord"].id
    assert k.scene_of(by["Halla"].id, 2) == hall
    # v4.29 (Marc, 2026-09-28): two scenes directly in one container are within reach of each other,
    # whatever the container is -- the hall and the home-field of one farm
    assert k.neighbours(hall, 2) == [yard] and k.neighbours(yard, 2) == [hall]


def test_within_reach_is_the_container_directly_above() -> None:
    """Stonemarch t180: the courtyard and the causeway are both directly in the
    ruined moathouse, which is not a scene, in the Viscounty, which is not one
    either -- and the courtyard was shown no neighbours at all, so the step back
    out through the gate arch was told blind and split the party. v4.29: scenes
    directly in one container are within reach of each other; a scene and a
    scene directly in it are too; a scene in another container is not, however
    near the containers stand. The present index and the past's scan agree."""
    ents = {
        "viscounty": Entity("viscounty", ["the Viscounty"], 0, "genesis"),
        "moathouse": Entity("moathouse", ["the ruined moathouse"], 0, "genesis", kosha="viscounty"),
        "millhaven": Entity("millhaven", ["Millhaven"], 0, "genesis", kosha="viscounty"),
        "courtyard": Entity("courtyard", ["the courtyard"], 0, "genesis", avarana="scene", kosha="moathouse"),
        "causeway": Entity("causeway", ["the causeway"], 0, "genesis", avarana="scene", kosha="moathouse"),
        "cellar": Entity("cellar", ["the cellar"], 0, "genesis", avarana="scene", kosha="courtyard"),
        "green": Entity("green", ["the green"], 0, "genesis", avarana="scene", kosha="millhaven"),
    }
    k = Kosha(Maya(world_time=3, entities=ents), [])
    for t in (2, 10**6):  # the past's scan, and the present index
        k.use_now = t > 2
        assert k.neighbours("courtyard", t) == ["cellar", "causeway"]   # what it holds, then its siblings
        assert k.neighbours("causeway", t) == ["courtyard"]
        assert k.neighbours("cellar", t) == ["courtyard"]                # what it is directly in, being a scene
        assert "green" not in k.neighbours("courtyard", t) and k.neighbours("green", t) == []


def _realm_world():
    ents = {
        "county": Entity("county", ["the county"], 0, "genesis"),
        "village": Entity("village", ["the village"], 0, "genesis", kosha="county"),
        "ruin": Entity("ruin", ["the ruin"], 0, "genesis", kosha="county"),
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene", kosha="village"),
        "green": Entity("green", ["the green"], 0, "genesis", avarana="scene", kosha="village"),
        "yard": Entity("yard", ["the ruin's yard"], 0, "genesis", avarana="scene", kosha="ruin"),
        "mug": Entity("mug", ["a mug"], 0, "genesis", kosha="inn"),
        "wen": Entity("wen", ["Wen"], 0, "genesis", kosha="inn"),
    }
    mems = [Memory("m1", "wen", frozenset(), "I sat in the inn.", "PRIMARY", 4, "inn", 1, "p1", None),
            Memory("m2", "village", frozenset(), "A village of thatch.", "PRIMARY", 4, "village", 0, "p0", None)]
    return ents, mems


def test_the_tree_is_the_scene_and_what_holds_it_with_the_places_each_holds() -> None:
    """v4.30 (Marc, 2026-09-28: "making sure the narrator has the tree to reason
    with"): the scene and what holds it, from the top down, each with the places
    it directly holds -- a region planted with a first tuple is a place by what
    it holds; a mug or a person in the inn is not a place."""
    ents, mems = _realm_world()
    k = Kosha(Maya(world_time=3, entities=ents), mems)
    assert k.tree("inn", 3) == [("county", ["ruin"]), ("village", ["green"]), ("inn", [])]
    assert k.is_place("village", 3) and k.is_place("ruin", 3) and not k.is_place("mug", 3) and not k.is_place("wen", 3)


def test_a_place_that_holds_the_top_grows_the_tree_and_a_loose_root_is_a_signal(monkeypatch) -> None:
    """v4.30: a realm named around the county takes the county into it -- the
    tree grows upward (Stonemarch, replayed 3/3: the Flanaess took the Viscounty
    in). A place left in nothing beside the top is a loose root: logged, never
    repaired (Marc: "If that case occurs we're missing something"). A thing
    left in nothing is neither. The prompt shows the tree."""
    from smritilila.smrti.pulse import run_smrti_pulse
    answers = {
        "grow": {"new_entities": [{"id": "NEW:realm", "name": "the realm", "in": "", "scene": False}],
                 "moves": [{"entity": "county", "into": "NEW:realm"}]},
        "loose": {"new_entities": [{"id": "NEW:far_land", "name": "the far land", "in": "", "scene": False},
                                   {"id": "NEW:far_hall", "name": "the far hall", "in": "NEW:far_land", "scene": True}],
                  "moves": []},
        "thing": {"new_entities": [{"id": "NEW:far_light", "name": "a far light", "in": "", "scene": False}], "moves": []},
    }
    seen = []
    for case, extra in answers.items():
        ents, mems = _realm_world()
        events = []

        class C:
            def complete_with_tool(self, system, user, **kw):
                seen.append((system, user))
                return {"narration": "You look out.", "perceiver_tuples": [{"perceiver": "wen", "text": "I looked out.",
                                                                            "salience": "ordinary"}],
                        "scene_tuples": [], "suggested_next_action": "", **extra}
        monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **kw: events.append((e, kw)))
        monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **kw: None)
        run_smrti_pulse(Maya(world_time=3, entities=ents), "wen", "I look out", 3, "p3", mems, C(), None)
        kinds = {e: kw for e, kw in events if e in ("smrti.root_grew", "smrti.loose_root")}
        if case == "grow":
            assert kinds["smrti.root_grew"]["was"] == "county" and "smrti.loose_root" not in kinds
        elif case == "loose":
            assert kinds["smrti.loose_root"]["top"] == "county" and "smrti.root_grew" not in kinds
        else:
            assert kinds == {}
    system, user = seen[0]
    assert '- county "the county" (the world\'s top) holds: ruin "the ruin"' in user
    assert "THE TREE" in user and "the smallest place in THE TREE that holds it" in system


def test_a_part_leaves_its_whole_and_comes_back_into_it(monkeypatch) -> None:
    """Decomposition is containment (Marc, 2026-09-28: the world builder decomposes by putting places in places): a
    dart thrown from Jakal's bundle is declared in the bundle, then moved out, and its first container says where it
    came from. A thing goes where the narrative puts it (Marc, 2026-10-13: "Everything is a container ... If the
    narrative says something merged with or goes into something that should happen"): moved into its whole it is in
    the whole; moved to whoever holds the whole, or to anyone else, it is there as said, and the engine does not send
    it anywhere it was not told (the guess from who holds the whole, v4.34's smrti.rejoined, is gone)."""
    from smritilila.smrti.pulse import run_smrti_pulse
    import json
    ents, mems = _realm_world()
    ents["darts"] = Entity("darts", ["iron throwing darts"], 0, "genesis", kosha="wen")
    ents["kell"] = Entity("kell", ["Kell Marrow"], 0, "genesis", kosha="inn")
    ents["mace"] = Entity("mace", ["Kell's mace"], 0, "genesis", kosha="kell")
    answers = [
        {"new_entities": [{"id": "NEW:dart", "name": "a thrown iron dart", "in": "darts", "scene": False}],
         "moves": [{"entity": "NEW:dart", "into": "inn"}]},
        {"moves": [{"entity": "DART", "into": "darts"}, {"entity": "mace", "into": "wen"}]},
        {"moves": [{"entity": "DART", "into": "wen"}]},
        {"moves": [{"entity": "DART", "into": "kell"}]},
    ]
    events, results = [], []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **kw: events.append((e, kw)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **kw: None)
    records, dart = [], None
    for i, extra in enumerate(answers):
        extra = json.loads(json.dumps(extra).replace("DART", dart or "DART"))

        class C:
            def complete_with_tool(self, system, user, **kw):
                return {"narration": "You throw.", "perceiver_tuples": [{"perceiver": "wen", "text": "I threw.",
                                                                       "salience": "ordinary"}],
                        "scene_tuples": [], "suggested_next_action": "", **extra}
        r = run_smrti_pulse(Maya(world_time=3 + i, entities=dict(ents)), "wen", "I throw a dart", 3 + i, f"p{3 + i}",
                            mems, C(), None, records=list(records))
        records += r.records_written
        results.append({x.entity_id: x.value for x in r.records_written if x.kind == "kosha"})
        if i == 0:
            dart = r.resolved_new["NEW:dart"]
            ents[dart] = Entity(dart, ["a thrown iron dart"], 3, "smrti")
    k = Kosha(Maya(world_time=6, entities=ents), mems, records)
    assert results[0][dart] == "inn" and "darts" not in results[0]      # the part went; the whole stayed with Wen
    assert k.origin(dart) == "darts" and k.origin("mace") == "kell"
    assert results[1][dart] == "darts" and results[1]["mace"] == "wen"  # told into its whole: in it; the mace as said
    assert results[2][dart] == "wen"                                     # told onto Wen, who holds the whole: onto Wen, as said
    assert results[3][dart] == "kell"                                    # told to Kell: with Kell
    assert not [e for e, kw in events if e == "smrti.rejoined"]          # the guess is gone


def test_a_being_becomes_in_play_and_it_reaches_the_bearer_alone(monkeypatch) -> None:
    """v5 §8: becoming is native. Authored in play, a clause holds for its
    bearer only; from the library by key; ended by a later record; a
    player becomes in another's turn only if they allow it (v5 §37)."""
    canon = _crossing()
    mems = [_mem(1, "teo", 5, "I sat on the bank.", "east"), _mem(2, "ilo", 5, "I sat with the boy.", "east"),
            _mem(3, "trav", 5, "I waited by the stone.", "east")]
    out = {"narration": "The water goes still.",
           "perceiver_tuples": [{"perceiver": "ilo", "text": "The river went still and I saw the drowned standing in it.", "salience": "indelible"}],
           "becomings": [{"entity": "ilo", "key": "the_sight", "clause": "Ilo sees the drowned where they stand."},
                         {"entity": "trav", "key": "the_mark", "clause": ""},
                         {"entity": "teo", "key": "the_sight", "clause": "Teo sees them too."}],
           "scene_tuples": [{"scene": "east", "text": "The river went still for a moment.", "present": ["ilo", "teo", "trav"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "ilo", None, 6, mems, out, players=frozenset({"teo"}),
                          view={"becomings": {"the_mark": "The traveler bears a mark that the dead can see."}})
    clauses = [(x.entity_id, x.note, x.value) for x in r.records_written if x.kind == "clause"]
    assert ("ilo", "self:the_sight", "Ilo sees the drowned where they stand.") in clauses
    assert ("trav", "self:the_mark", "The traveler bears a mark that the dead can see.") in clauses   # from the library
    assert not any(e == "teo" for e, _n, _v in clauses)                                              # a player, not allowing it
    assert any(e == "smrti.becoming_refused" and f["reason"] == "player_not_actor" for e, f in events)
    k = Kosha(canon, mems + r.memories_written, r.records_written)
    assert k.view_of("ilo", 7)[-1] == ("ilo", "the_sight", "Ilo sees the drowned where they stand.")
    assert all(bearer != "ilo" for bearer, _k, _t in k.view_of("trav", 7))  # self-scoped: reaches nobody else
    k.add(Record("r90", "clause", "ilo", None, 8, "p8", "self:the_sight"))
    assert k.view_of("ilo", 9) == []                                        # ended by a later record


def test_a_house_that_perceives_keeps_its_own_tuple_and_a_reacher_is_written_from_afar(monkeypatch) -> None:
    """v5 §7-§8: capabilities -- a world default, or given to one being."""
    canon = _crossing()
    canon.entities["far"] = Entity("far", ["a seer far upriver"], 0, "genesis")
    mems = [_mem(1, "teo", 5, "I sat on the bank.", "east"), _mem(2, "ilo", 5, "I sat with the boy.", "east"),
            _mem(3, "far", 5, "I dreamed of the crossing.", "west", subjects=("teo",)),
            _mem(4, "teo", 5, "I once met a seer from upriver.", "east", subjects=("far",))]
    out = {"narration": "You speak.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "I spoke to the landing and it seemed to listen.", "salience": "notable"},
                                {"perceiver": "east", "text": "The boy spoke to me and I kept his words.", "salience": "notable"},
                                {"perceiver": "far", "text": "Far off, I heard the boy speak at the crossing.", "salience": "notable"}],
           "scene_tuples": [{"scene": "east", "text": "The boy spoke to the empty landing.", "present": ["teo", "ilo"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "teo", "speak to the landing", 6, mems, out)
    assert {m.entity_id for m in r.memories_written if m.tier == "PRIMARY"} == {"teo"}  # by default: neither
    recs = [Record("c1", "capability", "east", "perceives", 0, "p0", "on"), Record("c2", "capability", "far", "reaches", 0, "p0", "on"),
            Record("n1", "name", "far", "Oda", 0, "p0")]
    # reaching covers anyone the pulse SHOWED: Teo, who has met Oda, speaks of her
    r, events, client = _pulse(monkeypatch, canon, "teo", "speak to the landing, thinking of Oda", 6, mems, out, records=recs)
    wrote = {m.entity_id: m for m in r.memories_written if m.tier == "PRIMARY"}
    assert set(wrote) == {"teo", "east", "far"} and wrote["east"].location_id == "east"
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert "east" not in k.occupants("east", 7) and k.is_perceiver("east", 7)  # never present inside itself
    assert "PERCEIVING (scenes with a perspective of their own)" in client.calls[0][1]


def test_the_views_that_hold_are_shown_outermost_first_and_each_ones_own_last(monkeypatch) -> None:
    """v5 §7: clauses on containers, nearest wins; v5 §8: each present being's own."""
    canon = _crossing()
    canon.entities["vale"] = Entity("vale", ["the vale"], 0, "genesis")
    mems = [_mem(1, "teo", 5, "I sat on the bank.", "east"), _mem(2, "ilo", 5, "I sat with the boy.", "east")]
    recs = [_rec(1, "east", "vale", 0),
            Record("v1", "clause", "vale", "In the vale the dead walk and remember.", 0, "p0", "within:dead_walk"),
            Record("v2", "clause", "east", "The landing forgets nothing said on it.", 0, "p0", "within:landing"),
            Record("v3", "clause", "ilo", "Ilo hears the river speak.", 0, "p0", "self:river_voice"),
            Record("v4", "clause", "vale", "Only the vale itself knows this.", 0, "p0", "self:vale_self")]
    out = {"narration": "x", "perceiver_tuples": [{"perceiver": "teo", "text": "I listened to the water a while.", "salience": "passing"}],
           "scene_tuples": [{"scene": "east", "text": "The boy listened to the water.", "present": ["teo", "ilo"]}], "suggested_next_action": ""}
    _, _, client = _pulse(monkeypatch, canon, "teo", "listen", 6, mems, out, records=recs,
                          view={"darsana": "Rivers here are persons.", "becomings": {}, "world_data": {}})
    system, user = client.calls[0]
    assert system.endswith("THIS WORLD'S OWN VIEW -- it wins over the default wherever they differ:\nRivers here are persons.")
    assert user.index("In the vale the dead walk") < user.index("The landing forgets nothing")
    assert "Only the vale itself knows this." not in user          # a self clause reaches no one inside
    assert "WHAT HOLDS FOR EACH ONE PRESENT" in user and "Ilo hears the river speak." in user


def test_a_quiet_scene_is_asked_who_came_only_if_someone_could_have(monkeypatch) -> None:
    """v5 §38 (Marc, 2026-09-28, "c sounds right"): Stonemarch's places took 51 of 99 world turns, and 2 moved or minted
    anything -- empty rooms telling that they were empty. A quiet scene is asked only if something not a player left
    a neighbouring scene since it was last asked, for it or for no scene recorded: replayed, 6 of the 51 kept, both
    that gave anything among them."""
    from smritilila import main, pravrtti
    events = []
    monkeypatch.setattr(pravrtti, "log_event", lambda e, **kw: events.append((e, kw)))  # the draw logs from its own module
    canon = Maya(world_time=40, entities={
        "village": Entity("village", ["the village"], 0, "genesis"),
        "green": Entity("green", ["the green"], 0, "genesis", avarana="scene", kosha="village"),
        "lane": Entity("lane", ["the lane"], 0, "genesis", avarana="scene", kosha="village"),
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene", kosha="village"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
        "drover": Entity("drover", ["a drover"], 0, "genesis")})

    def pool(records):
        world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["green"],
                                  playable_entity_ids=["teo"])
        world.recent_memories = [_mem(1, "teo", 30, "I stood on the green.", "green"),
                                 _mem(2, "teo", 5, "I crossed the yard.", "yard"),
                                 _mem(3, "drover", 5, "I drove through the lane.", "lane")]
        world.records = [_rec(1, "teo", "green", 0), _rec(2, "drover", "lane", 0), *records]
        return pravrtti._smrti_exploration_pool(world)
    assert "yard" not in pool([]) and any(e == "smrti.nobody_near" for e, _ in events)  # nobody could have come
    assert "yard" in pool([_rec(3, "drover", "yard", 25)])                 # the drover left the lane for the yard
    assert "yard" in pool([_rec(3, "drover", None, 25)])                   # ... or for nowhere recorded
    assert "yard" not in pool([_rec(3, "drover", "green", 25)])            # ... he went to the green instead
    assert "yard" not in pool([_rec(3, "teo", "lane", 3), _rec(4, "teo", "yard", 25)])  # a player's going is their own


def test_a_worlds_view_is_read_and_set_whole_and_a_closed_world_is_never_asked_who_came(monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from smritilila import adhisthana, main
    saved = {}

    class _Store:
        async def save_view(self, d, b, w):
            saved["v"] = (d, b, w)

    monkeypatch.setattr(adhisthana, "get_world_store", lambda: _Store())
    canon = Maya(world_time=40, entities={"inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene"),
                                          "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
                                          "teo": Entity("teo", ["Teo"], 0, "genesis")})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.world_data["pacing"] = {"arrivals_on_traffic": False}  # asked on the clock: the ask itself, not its traffic gate
    world.recent_memories = [_mem(1, "teo", 30, "I sat at the inn.", "inn"), _mem(2, "teo", 20, "I crossed the yard.", "yard")]
    client = TestClient(main.app)
    client.app.state.world = world
    try:
        assert "yard" in pravrtti._smrti_exploration_pool(world)
        r = client.put("/world/view", json={"darsana": "  An island; no one comes.  ", "becomings": {"the_sight": "sees the drowned", "x": " "},
                                            "world_data": {"pacing": {"arrivals": False}}})
        assert r.status_code == 200 and r.json()["darsana"] == "An island; no one comes."
        assert r.json()["becomings"] == {"the_sight": "sees the drowned"} and saved["v"][0] == "An island; no one comes."
        assert client.get("/world/view").json()["world_data"] == {"pacing": {"arrivals": False}}
        assert "yard" not in pravrtti._smrti_exploration_pool(world)
    finally:
        del client.app.state.world


def test_a_planted_world_declares_every_name_and_derives_none() -> None:
    """v5 §16/§39: genesis writes a name record for everything it plants --
    the seed's `name`, else the whole label -- so a v4-born world is never
    named by the v3 capital-letter derivation (in German every noun, in
    Chinese nothing)."""
    from smritilila.smrti.bija import as_spec
    from smritilila.smrti.genesis import build
    out = {"name": "t", "setting": "s", "opening": "o", "player": "li",
           "places": [{"key": "yard", "label": "Der Hof am Fluss", "name": "", "in": "", "scene": True, "tuple": "Ein Hof am Fluss, die Mühle dahinter."}],
           "people": [{"key": "li", "label": "李白，诗人", "name": "李白", "place": "yard", "tuples": [{"text": "我在河边的院子里等了很久。", "weight": "notable"}]},
                      {"key": "anna", "label": "Anna Müller, die Müllerin", "name": "Anna", "place": "yard", "tuples": [{"text": "Ich habe die Mühle seit zwanzig Jahren geführt.", "weight": "notable"}]}]}
    spec = as_spec(out)
    ents, mems, config = build(spec, lambda t: None)
    by = {e.labels[0]: e.id for e in ents}
    assert {(r.entity_id, r.value) for r in config["records"] if r.kind == "name"} == {(by["Der Hof am Fluss"], "Der Hof am Fluss"),
                                                                    (by["李白，诗人"], "李白"), (by["Anna Müller, die Müllerin"], "Anna"),
                                                                    (config["earth"], "the world at large")}  # and its earth
    canon = Maya(world_time=1, entities={e.id: e for e in ents})
    k = Kosha(canon, mems, config["records"])
    assert k.named("Anna") == {by["Anna Müller, die Müllerin"]} and not k.named("Müller") and not k.named("Hof")  # nothing derived
    assert set(k.names_in("我见到李白了。", [by["李白，诗人"]])) == {by["李白，诗人"]}
    assert all(m.experience for m in mems)  # a 13-character Chinese memory is a memory, not a placeholder


def test_a_seed_writes_its_own_view_and_its_places_carry_their_physics() -> None:
    """v5 §7: bija writes, at birth, only where the world differs from the
    default -- the view, a place's clause holding within it, the library
    of becomings, the switches. Planted: a `within` clause record on the
    place; the view handed to the world."""
    from smritilila.smrti.bija import as_spec
    from smritilila.smrti.genesis import build
    from smritilila.smrti.kala import CONFIRM_MINUTES
    out = {"name": "t", "setting": "s", "opening": "o", "player": "eel",
           "view": "", "becomings": [{"key": "The Drowned Sight", "text": "The one who has it sees the drowned wherever water stands."}],
           "places_perceive": False, "perception_reaches": False, "closed": True,
           "places": [{"key": "fen", "label": "the fen country", "in": "", "scene": False, "tuple": "Flat wet country, dykes and reeds to the sea."},
                      {"key": "marsh", "label": "Gallows Marsh", "in": "fen", "scene": True,
                       "clause": "Here the drowned walk at night and remember who drowned them.", "tuple": "A marsh of black pools below the old gallows."},
                      {"key": "quay", "label": "the eel quay", "in": "fen", "scene": True, "tuple": "A plank quay where the eel boats tie up."}],
           "people": [{"key": "eel", "label": "the eel-catcher", "name": "Wat", "place": "quay",
                       "tuples": [{"text": "I have worked the marsh edge since I could walk.", "weight": "notable"}]}]}
    spec = as_spec(out)
    assert spec["view"] == {"darsana": None, "becomings": {"the_drowned_sight": "The one who has it sees the drowned wherever water stands."},
                            "world_data": {"capabilities": {"perceives": False, "reaches": False}, "pacing": {"arrivals": False},
                                           "clock": {"calendar": "days", "start": 480, "confirm_minutes": CONFIRM_MINUTES},  # born keeping time
                                           "sandhi": True}}  # and seeing across what joins two scenes
    ents, mems, config = build(spec, lambda t: None)
    by = {e.labels[0]: e.id for e in ents}
    clauses = [(r.entity_id, r.note, r.value) for r in config["records"] if r.kind == "clause"]
    assert clauses == [(by["Gallows Marsh"], "within:marsh", "Here the drowned walk at night and remember who drowned them.")]
    assert config["view"] is spec["view"]
    k = Kosha(Maya(world_time=1, entities={e.id: e for e in ents}), mems, config["records"])
    assert k.view_of(by["the eel-catcher"], 2) == []                                    # the quay: the default holds
    k.add(Record("m1", "kosha", by["the eel-catcher"], by["Gallows Marsh"], 2, "p2"))
    assert k.view_of(by["the eel-catcher"], 3)[0][2].startswith("Here the drowned walk")  # walk into the marsh, and it holds


def test_a_memory_built_from_anothers_is_flagged_and_a_shared_moment_is_not(monkeypatch) -> None:
    """v5 §5 / the contract, measured: B's new tuple repeating a long run of
    A's tuple from the context, where B was not there -- flagged, never
    blocked. The same words from a moment both were in are not a leak."""
    canon = _crossing()
    mems = [_mem(1, "teo", 5, "I sat on the bank.", "east"), _mem(2, "ilo", 5, "I sat with the boy.", "east"),
            _mem(3, "ilo", 3, "Alone at night I buried the gold under the third stone past the old mill race.", "west"),
            _mem(4, "trav", 4, "I watched the boy mend the net under the willow by the landing post.", "east"),
            _mem(5, "teo", 4, "I mended the net under the willow while the traveler watched.", "east")]
    out = {"narration": "x",
           "perceiver_tuples": [
               {"perceiver": "teo", "text": "I knew then that the gold was buried under the third stone past the old mill race.", "salience": "weighty"},
               {"perceiver": "ilo", "text": "The boy looked at me strangely, as if he knew about the gold.", "salience": "notable"}],
           "scene_tuples": [{"scene": "east", "text": "The boy watched the boy mend the net under the willow by the landing post.", "present": ["teo", "ilo"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "teo", "think", 6, mems, out)
    leaks = [f for e, f in events if e == "smrti.memory_leak_suspected"]
    assert [(f["perceiver"], f["source"], f["source_tick"]) for f in leaks] == [("teo", "ilo", 3)]
    assert any(e == "smrti.pulse" and f["leaks"] == 1 for e, f in events)
    assert {m.entity_id for m in r.memories_written} >= {"teo", "ilo"}  # measured, never blocked


def test_an_actor_who_reaches_is_shown_the_distant_perceiver_nearest_their_mind() -> None:
    """v5 §8 (the telepath playtest): the engine permitted a reaching
    perceiver's tuple and never showed one. An actor who reaches is shown
    the few perceivers anywhere whose recent tuples lie nearest the action
    and their own latest memory, lifted by weight. An actor who does not
    reach is shown no one new."""
    canon = Maya(world_time=10, entities={
        "office": Entity("office", ["the office"], 0, "genesis", avarana="scene"),
        "pocket": Entity("pocket", ["the pocket"], 0, "genesis", avarana="scene"),
        "canteen": Entity("canteen", ["the canteen"], 0, "genesis", avarana="scene"),
        "ines": Entity("ines", ["Ines"], 0, "genesis"), "piet": Entity("piet", ["Piet"], 0, "genesis"),
        "cook": Entity("cook", ["the cook"], 0, "genesis")})
    def m(i, who, t, text, loc, emb, sal=4):
        return Memory(f"m{i}", who, frozenset(), text, "PRIMARY", sal, loc, t, f"p{t}", list(emb))
    mems = [m(1, "ines", 5, "I heard a man counting his air below the rock.", "office", (1.0, 0.0, 0.0)),
            m(2, "piet", 5, "I am thinking as loud as I can; let someone hear.", "pocket", (0.9, 0.1, 0.0), sal=1),
            m(3, "cook", 5, "I scrubbed the pans and banked the stove.", "canteen", (0.0, 0.0, 1.0))]
    log, _ = build_log(canon, mems, [], defaults={"reaches": True})
    ctx = retrieve(log, "ines", 6, np.asarray([1.0, 0.0, 0.0]), k_place=12, k_sim=0, action="reach down")
    reached = {t.perceiver for t in ctx.tuples if ctx.sources.get((t.perceiver, t.seq)) == "reaching"}
    assert "piet" in reached
    log2, _ = build_log(canon, mems, [])
    ctx2 = retrieve(log2, "ines", 6, np.asarray([1.0, 0.0, 0.0]), k_place=12, k_sim=0, action="reach down")
    assert not any(v == "reaching" for v in ctx2.sources.values())


def _mine() -> tuple[Maya, list[Memory], list[Record]]:
    """The telepath playtest's headframe yard, reduced: the yard in the
    mine, the mine on the moon; the shaft in the mine and the cage in the
    shaft, silent; the workings beside the yard with a station in it."""
    canon = Maya(world_time=10, entities={
        "moon": Entity("moon", ["Ossery"], 0, "genesis"),
        "mine": Entity("mine", ["Number Four"], 0, "genesis"),
        "yard": Entity("yard", ["the collar"], 0, "genesis", avarana="scene"),
        "workings": Entity("workings", ["the workings"], 0, "genesis"),
        "station": Entity("station", ["the 900 level station"], 0, "genesis", avarana="scene"),
        "shaft": Entity("shaft", ["the Number Four shaft"], 4, "smrti"),
        "cage": Entity("cage", ["the cage"], 4, "smrti"),
        "lamp": Entity("lamp", ["the cage lamp"], 4, "smrti"),
        "phone": Entity("phone", ["the collar telephone"], 7, "smrti"),
        "maren": Entity("maren", ["Maren"], 0, "genesis"),
        "joss": Entity("joss", ["Joss"], 0, "genesis"),
    })
    recs = [_rec(1, "mine", "moon", 0), _rec(2, "yard", "mine", 0), _rec(3, "workings", "mine", 0), _rec(4, "station", "workings", 0),
            _rec(5, "shaft", "mine", 4), _rec(6, "cage", "shaft", 4), _rec(7, "lamp", "cage", 4), _rec(8, "phone", "yard", 7),
            _rec(9, "maren", "yard", 0), _rec(10, "joss", "station", 0)]
    for i, (eid, name) in enumerate([("moon", "Ossery"), ("mine", "Number Four"), ("yard", "the collar"), ("workings", "the workings"),
                                      ("station", "the 900 level station"), ("shaft", "the Number Four shaft"), ("cage", "the cage"),
                                      ("lamp", "the cage lamp"), ("phone", "the collar telephone"), ("maren", "Maren"), ("joss", "Joss")]):
        recs.append(_rec(20 + i, eid, name, 0, kind="name"))
    mems = [_mem(1, "moon", 0, "A cold moon of ice and grey rock.", "moon"),
            _mem(2, "mine", 0, "The Company's working shaft, cut into the blueshale.", "mine"),
            _mem(3, "yard", 0, "A shed around the shaft collar.", "yard"),
            _mem(4, "maren", 5, "I sent the cage down to Eleven.", "yard"),
            _mem(5, "joss", 5, "I waited at the station gate.", "station")]
    return canon, mems, recs


def test_what_holds_the_scene_is_shown_to_the_root_and_what_it_holds_silently_by_name() -> None:
    """The telepath playtest: the shaft named from the yard was minted in
    nothing (the mine above the yard never shown), and the cage, silent in
    it, was minted again. What holds the scene is shown to the root, with
    first tuples; what the nearest holder holds, perceiving nothing, is
    listed by declared name -- two steps inside it at most, never what is
    in the scene itself, never a perceiver."""
    from smritilila.smrti.narrator import render
    canon, mems, recs = _mine()
    log, _ = build_log(canon, mems, recs)
    ctx = retrieve(log, "maren", 8, None, k_place=6, k_sim=0, action="Step into the cage.")
    assert ctx.above == ["mine", "moon"]
    assert {t.perceiver for t in ctx.tuples if ctx.sources.get((t.perceiver, t.seq)) == "above"} == {"mine", "moon"}
    held = {eid: (name, c) for eid, name, c in ctx.held}
    assert held["cage"] == ("the cage", "shaft") and held["shaft"] == ("the Number Four shaft", "mine")
    assert "workings" in held and "station" in held
    assert not {"lamp", "phone", "joss", "yard", "mine", "maren"} & set(held)  # three steps in; shown as the scene's contents; a perceiver; shown
    text = render(ctx, "Step into the cage.", {"yard", "station"})
    assert "ACTOR IS IN: yard -> mine -> moon" in text
    assert 'cage "the cage", in shaft' in text and "HELD HERE AND IN WHAT HOLDS THE SCENE" in text


def test_what_the_holder_holds_is_bounded_nearest_first() -> None:
    canon, mems, recs = _mine()
    k = Kosha(canon, mems, recs)
    # the scene's own first; then each holder's, latest active first, nearer breaking a tie
    assert [h[0] for h in k.held("yard", 8)] == ["phone", "station", "shaft", "cage", "workings"]
    assert [h[0] for h in k.held("yard", 8, cap=2)] == ["phone", "station"]
    # a place something happened in lately comes before one only recorded lately
    k2 = Kosha(canon, mems + [_mem(9, "joss", 7, "I walked the workings.", "workings")], recs)
    assert [h[0] for h in k2.held("yard", 8, cap=3)] == ["phone", "workings", "station"]
    # the workings hold nothing else, so the mine's are next: one step in or two, latest active first
    assert [h[0] for h in k.held("station", 8)] == ["phone", "yard", "shaft", "cage"]
    assert k.held(None, 8) == []


def test_the_silent_cage_is_the_one_stepped_into(monkeypatch) -> None:
    """The cage declared again by its name folds onto the one held by
    the scene's holder: it is shown, so it is known (v5 §17)."""
    canon, mems, recs = _mine()
    out = {"narration": "You step into the cage.",
           "perceiver_tuples": [{"perceiver": "maren", "text": "I stepped into the cage.", "salience": "ordinary"}],
           "scene_tuples": [], "moves": [{"entity": "maren", "into": "NEW:the_cage"}],
           "new_entities": [{"id": "NEW:the_cage", "name": "the cage", "in": "yard", "scene": False}],
           "suggested_next_action": ""}
    r, events, client = _pulse(monkeypatch, canon, "maren", "Step into the cage.", 8, mems, out, records=recs)
    assert 'cage "the cage", in shaft' in client.calls[0][1]
    assert not any(e == "smrti.minted" for e, _ in events)
    assert [(x.entity_id, x.value) for x in r.records_written if x.kind == "kosha"] == [("maren", "cage")]


def _market() -> tuple[Maya, list[Memory], list[Record]]:
    """A crowded scene: three people busy lately, three things handled
    once, long ago -- the Ravenmoor hollow, reduced."""
    ents = {"market": Entity("market", ["the market"], 0, "genesis", avarana="scene"),
            "teo": Entity("teo", ["Teo"], 0, "genesis")}
    recs = [_rec(1, "teo", "market", 0)]
    mems = [_mem(1, "market", 0, "A square of stalls.", "market"), _mem(2, "teo", 30, "I came into the market.", "market"),
            _mem(3, "teo", 39, "I looked about.", "market")]
    for i, (eid, name, ticks) in enumerate([("ana", "Ana", (32, 36, 38)), ("ben", "Ben", (30, 37)), ("cai", "Cai", (35,)),
                                             ("o1", "the crate", (2, 3)), ("o2", "the jar", (2, 4)), ("o3", "the lamp", (1, 5))]):
        ents[eid] = Entity(eid, [name], 0, "genesis")
        recs += [_rec(10 + i, eid, "market", 0), _rec(20 + i, eid, name, 0, kind="name")]
        mems += [_mem(100 + 10 * i + j, eid, t, f"{name} at {t}.", "market") for j, t in enumerate(ticks)]
    return Maya(world_time=40, entities=ents), mems, recs


def test_past_the_limit_the_least_active_present_are_quiet_and_known_by_who_they_are() -> None:
    """The participation limit (Marc, 2026-09-23): computational, never a
    prohibition. Past PARTICIPANTS present, the least active bring only
    their first tuple -- who they are -- and are marked quiet; the actor,
    the players and anyone the action names always take part."""
    from smritilila.smrti.narrator import render
    canon, mems, recs = _market()
    log, _ = build_log(canon, mems, recs)
    ctx = retrieve(log, "teo", 40, None, k_place=12, k_sim=0, action="Look at the stalls.")
    assert ctx.present == {"ana", "ben", "cai", "o1", "o2", "o3"} and ctx.quiet == {"o2", "o3"}
    # who they are, and nothing the pulse brings per member; the scene's own
    # recent window is the scene's, bounded by itself, and may still hold them
    per_member = {"present_salient", "followed", "their_scenes"}
    for q in ("o2", "o3"):
        mine = [t for t in ctx.tuples if t.perceiver == q]
        assert log.stream(q)[0] in mine
        assert not any(ctx.sources.get((t.perceiver, t.seq)) in per_member for t in mine)
    assert any(t.perceiver == "o1" and ctx.sources.get((t.perceiver, t.seq)) == "present_salient" for t in ctx.tuples)
    assert "PRESENT AND QUIET THIS PULSE" in render(ctx, "Look at the stalls.", {"market"})
    # named in the action: it takes part
    assert retrieve(log, "teo", 40, None, k_place=12, k_sim=0, action="Pick up the lamp.").quiet == {"o2"}
    # a player present always takes part
    assert retrieve(log, "teo", 40, None, k_place=12, k_sim=0, action="Look.", keep=frozenset({"o3"})).quiet == {"o2"}  # K counts the rest
    # the world's pacing sets the limit; 0 is none
    assert retrieve(log, "teo", 40, None, k_place=12, k_sim=0, action="Look.", pacing={"participants": 0}).quiet == set()
    assert retrieve(log, "teo", 40, None, k_place=12, k_sim=0, action="Look.", pacing={"participants": 2}).quiet == {"cai", "o1", "o2", "o3"}


def test_the_quiet_may_still_be_written_for(monkeypatch) -> None:
    """Quiet is a bound on what a pulse carries, not on what happens: a
    tuple for a quiet member is kept like anyone's (Ravenmoor 377, replayed:
    the dynamite went into the quiet sinkhole, and it perceived it)."""
    canon, mems, recs = _market()
    out = {"narration": "Teo knocks the lamp over.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "I knocked the lamp over.", "salience": "ordinary"},
                                {"perceiver": "o3", "text": "I tipped off the stall and struck the stones.", "salience": "passing"}],
           "scene_tuples": [], "suggested_next_action": ""}
    r, _, client = _pulse(monkeypatch, canon, "teo", "Walk between the stalls.", 40, mems, out, records=recs)
    assert "PRESENT AND QUIET THIS PULSE" in client.calls[0][1] and "o3" in client.calls[0][1]
    assert any(m.entity_id == "o3" and m.experience == "I tipped off the stall and struck the stones." for m in r.memories_written)


def test_a_carried_thing_that_perceives_goes_with_its_carrier() -> None:
    """The crossing, 580: Sela held the comb (a record into her at 565); the
    comb's own tuple at 566 was filed at the inn, where she was; she walked
    out to the yard, and the comb stayed at the inn. A later tuple overrides
    a record only if it disagrees with where the record's chain reached AT
    THE TUPLE'S TICK -- a v3-style move -- never because the carrier moved."""
    canon = Maya(world_time=20, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene"),
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "sela": Entity("sela", ["Sela"], 0, "genesis"), "comb": Entity("comb", ["the bone comb"], 5, "smrti")})
    recs = [_rec(1, "sela", "inn", 0), _rec(2, "comb", "sela", 5), _rec(3, "sela", "yard", 10)]
    mems = [_mem(1, "sela", 4, "I sat by the fire.", "inn"), _mem(2, "comb", 6, "I was held in a warm hand.", "inn"),
            _mem(3, "sela", 10, "I went out to the yard.", "yard")]
    k = Kosha(canon, mems, recs)
    assert k.container_at("comb", 11) == "sela" and k.scene_of("comb", 11) == "yard"
    # ... and its tuple of the very pulse she carried it out in, filed at the
    # inn the pulse began in (the crossing, 580)
    k1 = Kosha(canon, mems + [_mem(5, "comb", 10, "I was carried out into colder air.", "inn")], recs)
    assert k1.scene_of("comb", 11) == "yard"
    # a v3-style move is still told by a tuple: a later tuple elsewhere, no record
    k2 = Kosha(canon, mems + [_mem(4, "comb", 12, "I lay on the inn floor.", "inn")], recs)
    assert k2.scene_of("comb", 13) == "inn"
    # v4 files a tuple where the pulse ENDED (Ravenmoor 666): voiced on the walk out,
    # filed in the yard -- still hers, and with her when she goes on to the road
    road = {**canon.entities, "road": Entity("road", ["the road"], 0, "genesis", avarana="scene"),
            "county": Entity("county", ["the county"], 0, "genesis")}
    c3 = Maya(world_time=20, entities=road)
    k3 = Kosha(c3, mems + [_mem(6, "comb", 10, "I was carried out into colder air.", "yard")], recs + [_rec(4, "sela", "road", 15)])
    assert k3.container_at("comb", 11) == "sela" and k3.scene_of("comb", 16) == "road"
    # and a tuple filed where no scene is (the root, standing in) moves nothing
    k4 = Kosha(c3, mems + [_mem(7, "comb", 12, "I lay still.", "county")], recs)
    assert k4.container_at("comb", 13) == "sela"


def test_the_last_time_here_is_shown_for_each_participant() -> None:
    """The crossing, 594-596: Tam, asked about this room, had slept against
    its wall 200 ticks before -- unshown -- and a past was invented over the
    recorded one. For the actor and each present participant, the last tuple
    they filed at this scene before this visit comes into view."""
    canon = Maya(world_time=60, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene"),
        "road": Entity("road", ["the road"], 0, "genesis", avarana="scene"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"), "tam": Entity("tam", ["Tam"], 0, "genesis"),
        "new": Entity("new", ["a stranger"], 0, "genesis")})
    recs = [_rec(1, "teo", "inn", 0), _rec(2, "tam", "inn", 0), _rec(3, "tam", "road", 11), _rec(4, "tam", "inn", 50),
            _rec(5, "new", "inn", 50)]
    def heavy(i, who, t, text, loc):  # what he cannot forget happened elsewhere
        return Memory(f"h{i}", who, frozenset(), text, "PRIMARY", 1, loc, t, f"p{t}", None, "key")
    mems = [_mem(1, "teo", 5, "I kept the fire.", "inn"), _mem(2, "tam", 9, "I slept against the wall by the hearth.", "inn"),
            _mem(3, "tam", 10, "I woke and saw the old man cross the room.", "inn"),
            heavy(4, "tam", 12, "I walked the road east and wept.", "road"), heavy(5, "tam", 13, "I buried my father by the road.", "road"),
            _mem(6, "tam", 55, "I sat at the table.", "inn"),
            _mem(7, "new", 55, "I came in from the rain.", "inn"), _mem(8, "teo", 58, "I asked the tinner his name.", "inn")]
    log, _ = build_log(canon, mems, recs)
    shown = lambda: {(t.perceiver, t.chronology) for t in retrieve(log, "teo", 60, None, k_place=2, k_sim=0, action="Ask him about the room.").tuples}
    assert ("tam", 10) in shown()                    # the end of his last visit, not this one
    assert ("tam", 9) not in shown()                 # one tuple: the last of it
    import smritilila.smrti.context as C
    old, C.LAST_HERE = C.LAST_HERE, 0
    try:
        assert ("tam", 10) not in shown()            # nothing else brought it
    finally:
        C.LAST_HERE = old


def _landing(nat_last: str = "I asked the boatman about the landing.", bren_last: str = "I walked beside the boy.") -> tuple[Maya, list[Memory], list[Record]]:
    """Mauna's shapes, reduced: Thompson at the landing carrying his watch and
    shotgun (Ravenmoor 685-697); Bren beside him carrying the shrine cloth
    (the crossing, 533-536); a player sitting in the Model T."""
    names = {"landing": "the landing", "nat": "Nathaniel Thompson", "watch": "the pocket watch",
             "gun": "the double-barreled shotgun", "bren": "Bren", "cloth": "the cloth from the shrine",
             "car": "the Model T", "wen": "Wentworth"}
    ents = {e: Entity(e, [n], 0, "genesis") for e, n in names.items()}
    ents["landing"] = Entity("landing", ["the landing"], 0, "genesis", avarana="scene")
    recs = [_rec(1, "nat", "landing", 0), _rec(2, "watch", "nat", 0), _rec(3, "gun", "nat", 0), _rec(4, "bren", "landing", 0),
            _rec(5, "cloth", "bren", 0), _rec(6, "car", "landing", 0), _rec(7, "wen", "car", 0)]
    recs += [_rec(20 + i, e, n, 0, kind="name") for i, (e, n) in enumerate(names.items())]
    mems = [_mem(1, "landing", 1, "Mud and a tarred shanty under the bridge.", "landing"),
            _mem(2, "watch", 5, "I ticked in the waistcoat pocket.", "landing"),
            _mem(3, "watch", 7, "I ticked on in the pocket.", "landing"),
            _mem(4, "gun", 6, "I lay pressed along his ribs.", "landing"),
            _mem(5, "gun", 8, "I stayed pressed along his ribs.", "landing"),
            _mem(6, "cloth", 5, "I was folded in a man's coat.", "landing"),
            _mem(7, "cloth", 8, "I stayed folded in the coat.", "landing"),
            _mem(8, "car", 4, "I stood on the sand.", "landing"),
            _mem(9, "wen", 8, "I sat in the car and waited.", "landing"),
            _mem(10, "bren", 9, bren_last, "landing"),
            _mem(11, "nat", 9, nat_last, "landing")]
    return Maya(world_time=10, entities=ents), mems, recs


def test_what_is_held_is_quiet_unless_its_name_is_in_play() -> None:
    """Mauna (Marc, 2026-09-26): a thing held by the actor, or by someone
    present the action names, is present and quiet -- its first tuple only --
    unless a word of its name is in the action or its holder's latest tuple.
    A carried thing written last pulse was written again 76% of the time
    (Ravenmoor, from 423): shown its own last tuple, asked a tuple of every
    perceiver present, it copied its own past."""
    from smritilila.smrti.narrator import render
    canon, mems, recs = _landing()
    log, _ = build_log(canon, mems, recs)
    off = {"participants": 0}  # the participation limit aside

    def quiet(action, lg=log):
        return retrieve(lg, "nat", 10, None, k_place=12, k_sim=0, action=action, pacing=off, keep=frozenset({"wen"})).quiet

    ctx = retrieve(log, "nat", 10, None, k_place=12, k_sim=0, action="Look along the river.", pacing=off, keep=frozenset({"wen"}))
    assert ctx.quiet == {"watch", "gun"}  # Bren is not named: his cloth is his own business
    for q in ("watch", "gun"):
        assert [t for t in ctx.tuples if t.perceiver == q] == [log.stream(q)[0]]  # who it is, and nothing the window brought
    assert "PRESENT AND QUIET THIS PULSE" in render(ctx, "Look along the river.", {"landing"})
    # a word of its name in the action: it speaks
    assert quiet("Check my watch.") == {"gun"}
    # ... or in its holder's latest tuple
    canon2, mems2, recs2 = _landing(nat_last="I kept the shotgun under my coat.")
    assert quiet("Look along the river.", build_log(canon2, mems2, recs2)[0]) == {"watch"}
    # held by someone the action names: quiet; the one named takes part
    assert quiet("Ask Bren about the flood.") == {"watch", "gun", "cloth"}
    canon3, mems3, recs3 = _landing(bren_last="I touched the cloth in my coat.")
    assert quiet("Ask Bren about the flood.", build_log(canon3, mems3, recs3)[0]) == {"watch", "gun"}
    # a player is never held quiet, whatever holds them
    assert "wen" not in quiet("Knock on the Model T.")


def test_mauna_can_be_switched_off(monkeypatch) -> None:
    canon, mems, recs = _landing()
    log, _ = build_log(canon, mems, recs)
    monkeypatch.setattr("smritilila.smrti.context.MAUNA", False)
    assert retrieve(log, "nat", 10, None, k_place=12, k_sim=0, action="Look along the river.",
                    pacing={"participants": 0}).quiet == set()


def test_what_is_held_quiet_may_still_be_written_for(monkeypatch) -> None:
    """Quiet is what a pulse is shown, not what may happen: Ravenmoor 697,
    "time?" -- the watch named nowhere, shown by name only, was used and
    written 6/6 (the larger model and the second model), never minted again."""
    canon, mems, recs = _landing()
    out = {"narration": "You thumb the watch open: twenty past eight.",
           "perceiver_tuples": [{"perceiver": "nat", "text": "I read my watch: twenty past eight.", "salience": "ordinary"},
                                {"perceiver": "watch", "text": "I was drawn out and opened to the grey light.", "salience": "passing"}],
           "scene_tuples": [], "suggested_next_action": ""}
    r, _, client = _pulse(monkeypatch, canon, "nat", "time?", 10, mems, out, records=recs)
    prompt = client.calls[0][1]
    assert "PRESENT AND QUIET THIS PULSE" in prompt and "the pocket watch" in prompt
    assert "I ticked on in the pocket." not in prompt and "I ticked in the waistcoat pocket." not in prompt  # its name, not its voice
    assert any(m.entity_id == "watch" and m.experience == "I was drawn out and opened to the grey light." for m in r.memories_written)


def _room_114() -> tuple[Maya, list[Memory], list[Record]]:
    """Clara's lecture hall at 707, reduced: five things handled lately, and
    two long quiet -- the schedule card and a cloth from the crossing."""
    things = {"register": "the class register", "rag": "the wet rag", "board": "the blackboard", "desk": "the lecture desk",
              "chair": "the chair at the front", "card": "the schedule card on the door frame",
              "cloth": "the cloth from the shrine stone"}
    ents = {"hall": Entity("hall", ["the lecture hall"], 0, "genesis", avarana="scene"),
            "clara": Entity("clara", ["Clara Whitcombe"], 0, "genesis")}
    recs = [_rec(1, "clara", "hall", 0)]
    mems = [_mem(1, "hall", 0, "Benches in tiers, a lecture desk.", "hall"), _mem(2, "clara", 39, "I came in early.", "hall")]
    for i, (eid, name) in enumerate(things.items()):
        ents[eid] = Entity(eid, [name], 0, "genesis")
        recs += [_rec(10 + i, eid, "hall", 0), _rec(30 + i, eid, name, 0, kind="name")]
        t = 2 if eid in ("card", "cloth") else 36 + i % 3
        mems.append(_mem(100 + i, eid, t, f"{name} at {t}.", "hall"))
    return Maya(world_time=40, entities=ents), mems, recs


def test_the_action_names_what_it_paraphrases_by_two_words() -> None:
    """Paraphrase should be allowed (Marc, 2026-09-26). Clara's 707: "the
    schedule card PINNED on the door frame" named nothing whole; the card
    stayed quiet and its own text went unshown. Two words of a label, rare
    among the world's labels, reach it; one does not -- at one, "from" woke
    both of the crossing's cloths."""
    canon, mems, recs = _room_114()
    log, _ = build_log(canon, mems, recs)

    def quiet(action):
        return retrieve(log, "clara", 40, None, k_place=12, k_sim=0, action=action).quiet

    assert {"card", "cloth"} <= quiet("Look about the hall.")
    assert "card" not in quiet("Read the schedule card pinned on the door frame.")  # schedule, card, frame
    assert "cloth" in quiet("Ask about the peddler from the hills.")  # "from" alone
    assert "card" in quiet("Straighten the card.")  # one word of it
    assert "card" not in quiet("Straighten the schedule card on the door frame.")  # the whole name, as before


def _peddler(teo_last: str = "I met a man at the inn.") -> tuple[Maya, list[Memory], list[Record]]:
    """The crossing, reduced: Teo at the landing; the tin peddler, met at
    the inn, standing there still."""
    names = {"landing": "the east landing", "inn": "the inn at the crossing", "teo": "Teo",
             "peddler": "the tin peddler off the hill roads", "post": "the mooring post", "cup": "the rusted cup"}
    ents = {e: Entity(e, [n], 0, "genesis") for e, n in names.items()}
    for s in ("landing", "inn"):
        ents[s] = Entity(s, [names[s]], 0, "genesis", avarana="scene")
    recs = [_rec(1, "teo", "landing", 0), _rec(2, "peddler", "inn", 0), _rec(3, "post", "landing", 0), _rec(4, "cup", "inn", 0)]
    recs += [_rec(20 + i, e, n, 0, kind="name") for i, (e, n) in enumerate(names.items())]
    mems = [_mem(1, "landing", 0, "Gravel and a post.", "landing"), _mem(2, "inn", 0, "A low room with a hearth.", "inn"),
            _mem(3, "teo", 5, "I met a man with pots at the inn.", "inn", subjects=("peddler",)),
            _mem(4, "peddler", 5, "I sold a boy nothing and ate my soup.", "inn"),
            _mem(5, "teo", 9, teo_last, "landing")]
    return Maya(world_time=10, entities=ents), mems, recs


def test_item_5_reaches_what_the_action_paraphrases_and_only_the_action() -> None:
    """Item 5 under "paraphrase should be allowed": an acquainted being is
    brought by two words of its label, rare among the world's labels, in the
    ACTION -- not in the actor's own tuples, where two words woke 16-22
    beings on nearly every pulse (measured, both worlds)."""
    canon, mems, recs = _peddler()
    log, _ = build_log(canon, mems, recs)

    def named(action, lg=log):
        ctx = retrieve(lg, "teo", 10, None, k_place=12, k_sim=0, action=action)
        return {t.perceiver for t in ctx.tuples if ctx.sources.get((t.perceiver, t.seq)) == "named"}

    assert "peddler" in named("Ask Ilo about the peddler from the hill country.")  # peddler, hill
    assert "peddler" in named("Ask about the tin peddler off the hill roads.")  # whole, as before
    assert "peddler" not in named("Ask about the peddler.")  # one word
    canon2, mems2, recs2 = _peddler(teo_last="I thought of the peddler from the hills and his roads.")
    assert "peddler" not in named("Wait by the post.", build_log(canon2, mems2, recs2)[0])  # his own words, not the action's


def test_participation_counts_what_matters_not_how_often() -> None:
    """Weighted participation (the crossing, 596): the man at the centre of
    the scene, with few heavy memories, outranks a cup written every pulse."""
    from smritilila.smrti.context import participation
    canon = Maya(world_time=30, entities={e: Entity(e, [e], 0, "genesis") for e in ("inn", "teo", "tam", "cup", "a", "b", "c")})
    canon.entities["inn"] = Entity("inn", ["the inn"], 0, "genesis", avarana="scene")
    def m(i, who, t, sal):
        return Memory(f"m{i}", who, frozenset(), f"{who} at {t} was here and did a thing.", "PRIMARY", sal, "inn", t, f"p{t}", None, "none")
    mems = [m(1, "tam", 28, 1), m(2, "tam", 25, 2)] + [m(10 + i, w, 29 - j, 4) for i, (w, j) in enumerate(
        [(w, j) for w in ("a", "b", "c") for j in range(3)])] + [m(40 + j, "cup", 29 - j, 7) for j in range(6)]
    log, _ = build_log(canon, mems, [])
    quiet = participation(log, {"tam", "cup", "a", "b", "c"}, 30, "", set(), 4)
    assert quiet == {"cup"}


def test_what_a_pulse_declared_is_known_to_everyone_who_was_in_it() -> None:
    """The crossing, 600-602: Tam named his father's soldering iron in the
    frame out in the yard; two ticks later Teo asked for "the soldering
    iron" by that very name, and it was minted again -- never a subject of
    his tuples, so never acquainted, so never retrieved by name. Minting is
    meeting: whatever a pulse declared is known to all who wrote in it."""
    canon = Maya(world_time=10, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene"),
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "frame": Entity("frame", ["the frame"], 0, "genesis"), "teo": Entity("teo", ["Teo"], 0, "genesis"),
        "tam": Entity("tam", ["Tam"], 0, "genesis"), "iron": Entity("iron", ["the soldering iron"], 8, "smrti")})
    recs = [_rec(1, "teo", "inn", 0), _rec(2, "tam", "inn", 0), _rec(3, "frame", "yard", 0),
            Record("r4", "kosha", "iron", "frame", 8, "p8", "declared"), Record("r5", "name", "iron", "the soldering iron", 8, "p8", None)]
    mems = [_mem(1, "teo", 8, "Tam said his father's iron was out in his frame.", "inn"),
            _mem(2, "tam", 8, "I told the boy about my father's iron.", "inn")]
    log, _ = build_log(canon, mems, recs)
    assert "iron" in log.acquainted["teo"] and "iron" in log.acquainted["tam"]
    ctx = retrieve(log, "teo", 10, None, k_place=6, k_sim=0, action="Ask Tam to bring the soldering iron in.")
    assert "iron" in {t.perceiver for t in ctx.tuples}


def test_a_thing_known_and_placed_nowhere_is_listed_by_name() -> None:
    """The crossing, 523 -> 543: the first half of the blanket, named as
    having gone downriver and placed nowhere, was minted again when found in
    Cass's chest. Silent things the actor knows of, in nothing, minted
    lately, are listed by declared name -- not placed things, not perceivers,
    not old ones."""
    from smritilila.smrti.narrator import render
    canon = Maya(world_time=80, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"), "ilo": Entity("ilo", ["Ilo"], 0, "genesis"),
        "half": Entity("half", ["the torn half"], 60, "smrti"), "cup": Entity("cup", ["the cup"], 60, "smrti"),
        "old": Entity("old", ["the old trough"], 5, "smrti"), "cat": Entity("cat", ["the cat"], 60, "smrti")})
    recs = [_rec(1, "teo", "inn", 0), _rec(2, "ilo", "inn", 0), _rec(3, "cup", "inn", 60)]
    recs += [Record(f"n{i}", "name", e, n, t, f"p{t}", None) for i, (e, n, t) in enumerate(
        [("half", "the torn half", 60), ("cup", "the cup", 60), ("old", "the old trough", 5), ("cat", "the cat", 60)])]
    mems = [_mem(1, "teo", 5, "Ilo spoke of the trough.", "inn"), _mem(2, "teo", 60, "Ilo said the other part of it went downriver.", "inn"),
            _mem(3, "ilo", 60, "I told the boy.", "inn"), _mem(4, "cat", 61, "I slept by the fire, far off.", "inn"),
            _mem(5, "teo", 79, "I sat down.", "inn")]
    log, _ = build_log(canon, mems, recs)
    ctx = retrieve(log, "teo", 80, None, k_place=1, k_sim=0, action="Sit.")
    assert ctx.known == [("half", "the torn half")]   # the cup is placed, the cat perceives, the trough is old
    assert 'half "the torn half"' in render(ctx, "Sit.", {"inn"})


def test_a_newcomer_sees_who_is_there_and_a_player_away_is_still() -> None:
    """Ravenmoor 422-423: Walter Pike's arrival called the lecture hall empty
    while Nathaniel -- a player, away -- stood in it, and that became Walter's
    own first memory. The arrival is shown who is there now (as seen, never by
    name); a pulse is shown the players present, whose bodies are their own."""
    from smritilila.smrti.bija import write_arrival
    from smritilila.smrti.pulse import view_block

    class Cap:
        def complete_with_tool(self, system, user, **kw):
            self.user = user
            return {"label": "Walter Pike", "tuples": [{"text": "I came up to the university to find him.", "weight": "notable"}],
                    "place_tuple": "A surveyor came in.", "opening": "You come in."}
    cap = Cap()
    write_arrival(cap, "A surveyor.", "the lecture hall", "A hall.", "The lecture ended.",
                  present=[("Nathaniel Thompson", "I pocketed my sketch.", True), ("the students", "We packed up.", False)])
    assert "WHO IS THERE NOW" in cap.user and "Nathaniel Thompson (STILL" in cap.user and "the students --" in cap.user
    canon = Maya(world_time=10, entities={"hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
                                          "nat": Entity("nat", ["Nathaniel"], 0, "genesis"), "wal": Entity("wal", ["Walter"], 0, "genesis")})
    k = Kosha(canon, [_mem(1, "nat", 5, "I stood by the lectern.", "hall"), _mem(2, "wal", 6, "I came in.", "hall")],
              [_rec(1, "nat", "hall", 0), _rec(2, "wal", "hall", 6)])
    from smritilila.smrti.context import Context
    ctx = Context(actor="wal", place="hall", tuples=[], present={"nat"})
    assert "PLAYERS HERE" in view_block(k, ctx, 10, {}, frozenset(), set(), players=frozenset({"nat", "wal"}))
    assert "PLAYERS HERE" not in view_block(k, ctx, 10, {}, frozenset({"nat"}), set(), players=frozenset({"nat", "wal"}))  # movable: the world may act


def test_walking_into_a_scene_with_a_player_is_narrated_again_shown_them(monkeypatch) -> None:
    """Ravenmoor 478: Danforth walked from the library steps into the lecture
    hall; the pulse was shown the steps' present, not the hall's, and wrote
    the hall empty while Nathaniel -- a player, away -- stood in it. A pulse
    that takes its actor into a scene holding a player it was not shown is
    narrated once more, shown them; the first pass's mints leave the canon.
    Movable players (the world may act for them) and non-players do not
    trigger it."""
    def setup():
        canon = _crossing()
        mems = [_mem(1, "teo", 5, "I stood on the east landing.", "east"), _mem(2, "ilo", 5, "I sat on the west landing.", "west")]
        return canon, mems
    out = {"narration": "You cross; the west landing is empty.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "I crossed and found the west landing empty.", "salience": "ordinary"}],
           "moves": [{"entity": "teo", "into": "west"}],
           "new_entities": [{"id": "NEW:coil_of_rope", "name": "a coil of rope", "in": "west"}],
           "scene_tuples": [{"scene": "west", "text": "A boy came across to the empty landing.", "present": ["teo"]}],
           "suggested_next_action": ""}
    canon, mems = setup()
    tags: list[dict] = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **f: tags.append(f))
    r, events, client = _pulse(monkeypatch, canon, "teo", "goes across to the west landing", 6, mems, out,
                               players=frozenset({"teo", "ilo"}))
    assert len(client.calls) == 2
    assert [(t["pulse_id"], t["reason"]) for t in tags] == [("p6", "arrival_rerun")]   # the first answer's training example, tagged
    assert "WHERE THIS ACTION LEADS" not in client.calls[0][1] and "WHERE THIS ACTION LEADS" in client.calls[1][1]
    assert any(e == "smrti.arrival_rerun" and f["players"] == ["ilo"] for e, f in events)
    assert sum(1 for e in canon.entities.values() if e.labels == ["a coil of rope"]) == 1   # the first pass's mint is gone
    assert {e.id for e in r.new_entities} <= set(canon.entities)
    # no second pass: the player there is movable, or is no player at all, or the actor stays put
    for kw in ({"players": frozenset({"teo", "ilo"}), "movable": frozenset({"ilo"})}, {"players": frozenset({"teo"})}):
        canon, mems = setup()
        _, _, client = _pulse(monkeypatch, canon, "teo", "goes across to the west landing", 6, mems, out, **kw)
        assert len(client.calls) == 1
    canon, mems = setup()
    stays = {**out, "moves": [], "new_entities": [], "scene_tuples": [{"scene": "east", "text": "A boy waited on the landing.", "present": ["teo"]}]}
    _, _, client = _pulse(monkeypatch, canon, "teo", "waits", 6, mems, stays, players=frozenset({"teo", "ilo"}))
    assert len(client.calls) == 1


def test_a_thing_known_and_placed_where_this_scene_does_not_reach_is_listed_where_it_is() -> None:
    """Ravenmoor 524-552: Wentworth, in the yard, looked through a window into
    the reading room (no neighbour of the yard); each pulse saw nothing of the
    man the last had minted there, and Elias Lowell became four entities. What
    the actor met lately and the world placed somewhere unshown is listed by
    name and container -- never a player, never what is already shown, never
    what KNOWN lists (placed nowhere), never the old."""
    from smritilila.smrti.narrator import render
    canon = Maya(world_time=80, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "room": Entity("room", ["the reading room"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"), "nat": Entity("nat", ["Nathaniel"], 0, "genesis"),
        "writer": Entity("writer", ["the writer at the table"], 60, "smrti"),
        "half": Entity("half", ["the torn half"], 60, "smrti"),
        "barrel": Entity("barrel", ["the ash barrel"], 60, "smrti"),
        "old": Entity("old", ["the old lamp"], 5, "smrti")})
    recs = [_rec(1, "wen", "yard", 0), _rec(2, "nat", "room", 0), _rec(3, "writer", "room", 60), _rec(4, "barrel", "yard", 60),
            _rec(5, "old", "room", 5)]
    recs += [Record(f"n{i}", "name", e, n, t, f"p{t}", None) for i, (e, n, t) in enumerate(
        [("writer", "the writer at the table", 60), ("half", "the torn half", 60), ("barrel", "the ash barrel", 60),
         ("old", "the old lamp", 5), ("nat", "Nathaniel", 60)])]
    mems = [_mem(1, "wen", 5, "I saw a lamp through the glass.", "yard"),
            _mem(2, "wen", 60, "Through the window a man wrote on the floor; I rolled the barrel over.", "yard"),
            _mem(3, "wen", 79, "I crouched on the landing.", "yard")]
    log, _ = build_log(canon, mems, recs)
    ctx = retrieve(log, "wen", 80, None, k_place=1, k_sim=0, action="Look again.", keep=frozenset({"wen", "nat"}))
    assert ctx.elsewhere == [("writer", "the writer at the table", "room")]  # the barrel is here, the half is KNOWN's, the lamp is old, Nathaniel a player
    assert ctx.known == [("half", "the torn half")]
    assert 'writer "the writer at the table", in room' in render(ctx, "Look again.", {"yard", "room"})


def test_someone_the_memory_shows_standing_elsewhere_is_listed_where_the_world_has_them() -> None:
    """Ravenmoor 669-670 (Marc, 2026-09-25: solve by what is shown): Hartley
    stood, by his last memory, in the Thompson house; Thompson drove to
    Hartley's own house, the Narrator voiced him there without moving him,
    and both his tuples were dropped. Someone named in the action or the
    memory, standing in a scene this pulse does not reach, is listed with
    where they are -- those in the action first; never a player, never one
    here or next door, never a root (standing in no scene is no where)."""
    from smritilila.smrti.narrator import render
    canon = Maya(world_time=80, entities={
        "county": Entity("county", ["the county"], 0, "genesis"),
        "quad": Entity("quad", ["the quad"], 0, "genesis", avarana="scene"),
        "gate": Entity("gate", ["the gate lodge"], 0, "genesis", avarana="scene"),
        "house": Entity("house", ["the Thompson house"], 0, "genesis", avarana="scene"),
        "campus": Entity("campus", ["the campus"], 0, "genesis"), "town": Entity("town", ["the town"], 0, "genesis"),
        "nat": Entity("nat", ["Nathaniel Thompson"], 0, "genesis"), "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "hartley": Entity("hartley", ["Professor Hartley"], 0, "genesis"),
        "porter": Entity("porter", ["the porter"], 0, "genesis"),
        "cap": Entity("cap", ["the tweed cap"], 60, "smrti")})
    recs = [_rec(8, "campus", "county", 0), _rec(9, "town", "county", 0),
            _rec(1, "quad", "campus", 0), _rec(2, "gate", "quad", 0), _rec(3, "house", "town", 0), _rec(4, "nat", "quad", 0),
            _rec(5, "wen", "house", 0), _rec(6, "porter", "gate", 0), _rec(7, "cap", "county", 60)]
    recs += [Record(f"n{i}", "name", e, n, 0, "p0", None) for i, (e, n) in enumerate(
        [("hartley", "Hartley"), ("wen", "Wentworth"), ("porter", "the porter"), ("cap", "the tweed cap")])]
    mems = [_mem(1, "hartley", 5, "I sat at the kitchen table waiting for Thompson's car.", "house"),
            _mem(2, "nat", 60, "The porter at the gate lodge nodded; Wentworth had gone home; I left the tweed cap on the steps.", "quad",
                 subjects=("hartley", "porter", "wen", "cap")),  # met: Hartley too (the acquaintance a name reaches)
            _mem(3, "porter", 60, "I nodded to the professor.", "gate"), _mem(4, "wen", 60, "I went home.", "house"),
            _mem(5, "nat", 79, "I stood on the quad.", "quad")]
    log, _ = build_log(canon, mems, recs)
    ctx = retrieve(log, "nat", 80, None, k_place=1, k_sim=0, action="Drive to Hartley's house and tell him.",
                   keep=frozenset({"nat", "wen"}))
    # Hartley, named in the action, is where his last memory has him; the porter is next door, Wentworth
    # a player, the cap in no scene (placed in the county root)
    assert [(e, w) for e, _n, w in ctx.elsewhere] == [("hartley", "house")]
    assert "gate" in ctx.neighbours and "house" not in ctx.neighbours
    line = render(ctx, "Drive to Hartley's house and tell him.", {"quad", "gate", "house", "county"})
    assert 'hartley "Hartley", in house' in line and "only if `moves` brings them" in line


def test_a_player_out_of_the_world_is_in_no_scene_and_never_written_for(monkeypatch) -> None:
    """Marc, 2026-09-24: logged off and frozen is "entirely suspended from the
    game, not present at all". Ilo, out of the world at the east landing, is
    in no one's present; the Narrator is told he is not in the world (memory
    still holds him there); nothing is written for him -- a player is never
    voiced in another's turn (v5 §37), and the scene's witnessed copy goes only
    to those present. Back, his first pulse is told to narrate the return."""
    canon = _crossing()
    mems = [_mem(1, "teo", 5, "I stood on the east landing.", "east"), _mem(2, "ilo", 5, "I sat on the east landing, mending.", "east")]
    out = {"narration": "You look round the landing.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "I looked round the east landing.", "salience": "ordinary"},
                                {"perceiver": "ilo", "text": "I watched the boy look round.", "salience": "ordinary"}],
           "scene_tuples": [{"scene": "east", "text": "A boy looked round the landing.", "present": ["teo", "ilo"]}],
           "suggested_next_action": ""}
    r, events, client = _pulse(monkeypatch, canon, "teo", "looks round", 6, mems, out,
                               players=frozenset({"teo", "ilo"}), absent=frozenset({"ilo"}))
    prompt = client.calls[0][1]
    assert "NOT IN THE WORLD NOW" in prompt and "PLAYERS HERE" not in prompt and "RETURNING" not in prompt
    assert [m.entity_id for m in r.memories_written] == ["teo", "east"]   # no tuple, no witnessed copy
    assert all("ilo" not in m.subject_ids for m in r.memories_written)
    # the same world, Ilo back: present again, and his return narrated
    canon = _crossing()
    _, _, client = _pulse(monkeypatch, canon, "ilo", "stands up", 6, mems, {**out, "perceiver_tuples": []},
                          players=frozenset({"teo", "ilo"}), absent=frozenset({"ilo"}), returning=5)
    assert "RETURNING: the actor has been out of the world since tick 5" in client.calls[0][1]
    assert "NOT IN THE WORLD NOW" not in client.calls[0][1]


def test_a_name_part_brings_in_a_stranger_of_that_name_and_marks_a_player(monkeypatch) -> None:
    """Ravenmoor playtest (Marc, 2026-09-24): Wentworth read "Lowell, E.",
    never met Edmund Lowell, and a second E. Lowell was made; "Professor
    Thompson" matched nothing of Nathaniel, and other players' NPCs gave him
    a crime. A capitalised part of a person's own name, found whole, brings a
    stranger in by name and marks a player; a description ("the door of Room
    214"), a title, an abbreviation, and anyone the actor has met do not.
    Every tuple naming another player is counted."""
    from smritilila.smrti.narrator import render
    canon = Maya(world_time=80, entities={
        "yard": Entity("yard", ["the yard"], 0, "genesis", avarana="scene"),
        "hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "nat": Entity("nat", ["Nathaniel Thompson"], 0, "genesis"),
        "edm": Entity("edm", ["the Lowell boy"], 0, "smrti"),
        "door": Entity("door", ["the door of Room 214"], 0, "smrti"),
        "abby": Entity("abby", ["Abigail Mercer"], 0, "smrti")})
    recs = [_rec(1, "wen", "yard", 0), _rec(2, "nat", "hall", 0), _rec(3, "edm", "hall", 0), _rec(4, "door", "hall", 0),
            _rec(5, "abby", "yard", 0)]
    recs += [Record(f"n{i}", "name", e, n, 1, "p1", None) for i, (e, n) in enumerate(
        [("nat", "Asst. Prof. N. Thompson"), ("edm", "Edmund Lowell"), ("abby", "Abigail Mercer")])]
    mems = [_mem(1, "edm", 1, "I drove the roadster north.", "hall"), _mem(2, "door", 1, "A door with a frosted pane.", "hall"),
            _mem(3, "abby", 2, "I kept the green door.", "yard"), _mem(4, "nat", 3, "I sat at the front.", "hall"),
            _mem(5, "wen", 60, "I read the register: Lowell, E., and the cover said Thompson. Mercer was a name I knew; Room 214 was on the list.", "yard",
                 subjects=("abby",))]
    log, _ = build_log(canon, mems, recs)
    ctx = retrieve(log, "wen", 80, None, k_place=1, k_sim=0, action="Ask about Lowell.", keep=frozenset({"wen", "nat"}))
    assert ctx.namesakes == [("edm", "Edmund Lowell", "Lowell")]   # not Abigail (met), not the door (a description)
    assert all(t.perceiver not in ("edm", "nat") for t in ctx.tuples)   # by name only: their own tuples are theirs
    assert ctx.players_named == [("nat", "Asst. Prof. N. Thompson")]
    prompt = render(ctx, "Ask about Lowell.", {"yard", "hall"})
    assert 'SOMEONE OF THAT NAME IS IN THE WORLD' in prompt and 'A PLAYER IS NAMED HERE, NOT PRESENT' in prompt
    assert log.kosha.name_parts("nat") == {"Nathaniel", "Thompson"}
    # the counter: a tuple in another's turn naming a player
    from smritilila.smrti import pulse as P
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    written = [_mem(9, "yard", 81, "The janitor said Thompson went up Tuesday.", "yard", tier="AMBIENT"),
               _mem(10, "wen", 81, "I asked about the crock.", "yard")]
    assert P.player_named(log.kosha, written, "wen", frozenset({"wen", "nat"}), "p81") == 1
    assert [(f["player"], f["part"], f["perceiver"]) for e, f in events if e == "smrti.player_named"] == [("nat", "Thompson", "yard")]


def test_a_name_another_person_bears_is_narrated_again_told_whose_it_is(monkeypatch) -> None:
    """Ravenmoor 554 (Marc: "the Narrator is reusing almost identical
    names, confusing to players"): the man in the reading room was named
    Elias Lowell while Edmund Lowell existed. A pulse that gives someone a
    name sharing a name part with a different person already in the world
    is narrated once more, told whose name it is -- once only; the first
    answer's training example is tagged; a name no one bears, or a
    description, does not trigger it."""
    def setup():
        canon = _crossing()
        canon.entities["edm"] = Entity("edm", ["the Lowell boy"], 0, "smrti")
        recs = [_rec(1, "teo", "east", 0), _rec(2, "trav", "east", 0), _rec(3, "edm", "west", 0),
                Record("n1", "name", "edm", "Edmund Lowell", 0, "p0", None)]
        mems = [_mem(1, "teo", 5, "I stood on the east landing.", "east"), _mem(2, "trav", 5, "I waited by the stone.", "east"),
                _mem(3, "edm", 5, "I drove north.", "west")]
        return canon, recs, mems
    def out_naming(name):
        return {"narration": "The traveler gives her name.",
                "perceiver_tuples": [{"perceiver": "teo", "text": "The traveler told me her name.", "salience": "ordinary"}],
                "aliases": [{"entity": "trav", "name": name}],
                "scene_tuples": [{"scene": "east", "text": "The traveler said her name.", "present": ["teo", "trav"]}],
                "suggested_next_action": ""}
    tags: list[dict] = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **f: tags.append(f))
    canon, recs, mems = setup()
    r, events, client = _pulse(monkeypatch, canon, "teo", "asks her name", 6, mems, out_naming("Edith Lowell"), records=recs)
    assert len(client.calls) == 2   # the fake answers the same twice: a rerun happens once, never again
    assert "NAMES ALREADY BORNE" not in client.calls[0][1] and 'Lowell -- edm "Edmund Lowell"' in client.calls[1][1]
    # Stonemarch t2367 (2026-10-22): told only that "Hobb" was taken, the rerun came back with the sentry "Brand", one word;
    # the rerun says again that a person's new name is whole (Marc: "a name should be multiple words always")
    rerun_line = next(ln for ln in client.calls[1][1].splitlines() if ln.startswith("NAMES ALREADY BORNE"))
    assert "two words at least" in rerun_line and "a given name and a family name, or a byname" in rerun_line
    assert any(e == "smrti.name_rerun" and f["taken"] == {"Lowell": "edm"} for e, f in events)
    assert [t["reason"] for t in tags] == ["name_rerun"]
    for name in ("Edith Marsh", "the woman from the stone"):
        canon, recs, mems = setup()
        _, _, client = _pulse(monkeypatch, canon, "teo", "asks her name", 6, mems, out_naming(name), records=recs)
        assert len(client.calls) == 1


def test_the_world_s_source_leads_every_narrator_prompt() -> None:
    """Marc, 2026-09-24: "we should always include the source up front.
    historical or fictional, that's what grounds the narrator in the
    correct context" -- Ravenmoor's view never said it was Ravenmoor's.
    The source is the system prompt's first line; a newcomer's arrival and
    an expansion carry it too; a world without one is unchanged."""
    from smritilila.smrti.narrator import SYSTEM, system_for
    from smritilila.smrti import bija
    src = "Fiction: the stories of H. P. Lovecraft"
    assert system_for("A view.", src).startswith("THIS WORLD'S SOURCE: Fiction: the stories of H. P. Lovecraft\n")
    assert system_for("A view.").startswith(SYSTEM[:40])
    seen = []

    class Cap:
        def complete_with_tool(self, system, user, **kw):
            seen.append(user)
            if kw.get("tool_name") == "write_arrival":
                return {"label": "a stranger", "tuples": [{"text": "I came up the road from Innsmouth this morning.", "weight": "notable"}],
                        "place_tuple": "A stranger came in.", "opening": "You come in."}
            return {"places": [], "people": []}
    bija.write_arrival(Cap(), "A stranger.", "the hall", "A hall.", "Quiet.", source=src)
    assert seen[-1].startswith("THIS WORLD'S SOURCE: " + src)
    bija.expand(Cap(), "Arkham.", {"hall": ("the hall", "scene")}, [], source=src)
    assert seen[-1].startswith("THIS WORLD'S SOURCE: " + src)


def _carrying():
    canon = _crossing()
    canon.entities["key"] = Entity("key", ["the gate key"], 0, "genesis")
    canon.entities["bag"] = Entity("bag", ["Teo's bag"], 0, "genesis")
    mems = [_mem(1, "teo", 5, "I stood on the east landing with my bag and the key.", "east")]
    recs = [_rec(1, "teo", "east", 1), _rec(2, "key", "teo", 1), _rec(3, "bag", "teo", 1), _rec(4, "ilo", "east", 1)]
    return canon, mems, recs


def _crossing_out(moves):
    return {"narration": "You cross.",
            "perceiver_tuples": [{"perceiver": "teo", "text": "I crossed to the west landing.", "salience": "ordinary"}],
            "moves": moves,
            "scene_tuples": [{"scene": "west", "text": "The boy came up onto the west landing.", "present": ["teo"]}],
            "suggested_next_action": ""}


def test_a_carried_thing_moved_where_its_carrier_went_stays_carried(monkeypatch) -> None:
    """Collision run, 2026-09-25 (Danforth, Ravenmoor 662; 26 things in 11
    moves across the worlds): the Narrator reported where carried things
    now were, and the engine took them out of their carrier's pocket."""
    canon, mems, recs = _carrying()
    out = _crossing_out([{"entity": "key", "into": "west"}, {"entity": "teo", "into": "west"}, {"entity": "bag", "into": "west"}])
    r, events, _ = _pulse(monkeypatch, canon, "teo", "crosses to the west landing", 6, mems, out, records=recs)
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.scene_of("teo", 7) == "west"
    assert k.container_at("key", 7) == "teo" and k.container_at("bag", 7) == "teo"
    assert sorted(f["entity"] for e, f in events if e == "smrti.carried_stays") == ["bag", "key"]


def test_letting_go_says_from_and_is_honoured(monkeypatch) -> None:
    canon, mems, recs = _carrying()
    out = _crossing_out([{"entity": "teo", "into": "west"}, {"entity": "bag", "into": "west", "from": "teo"}])
    r, events, _ = _pulse(monkeypatch, canon, "teo", "crosses and drops the bag on the landing", 6, mems, out, records=recs)
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.container_at("bag", 7) == "west" and k.container_at("key", 7) == "teo"
    assert not [e for e, _ in events if e == "smrti.carried_stays"]


def test_a_carried_thing_moves_if_its_carriers_own_move_was_refused(monkeypatch) -> None:
    # teo, a player, moved in ilo's turn: refused -- so the key is not "with" a carrier that went nowhere
    canon, mems, recs = _carrying()
    out = _crossing_out([{"entity": "teo", "into": "west"}, {"entity": "key", "into": "west"}])
    r, events, _ = _pulse(monkeypatch, canon, "ilo", None, 6, mems, out, records=recs, players=frozenset({"teo"}))
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.scene_of("teo", 7) == "east"
    assert not [e for e, _ in events if e == "smrti.carried_stays"]


def test_a_thing_named_for_its_holder_is_not_a_namesake(monkeypatch) -> None:
    """Collision run, Ravenmoor 662: "Walter Pike's matchbox", declared into
    Walter Pike, was narrated again as a namesake of the man holding it --
    the one false rerun in history. A part its holder bears is skipped; a
    new person bearing another's part is still one (667, Mrs. Pickering)."""
    def setup():
        canon = _crossing()
        recs = [_rec(1, "teo", "east", 0), _rec(2, "trav", "east", 0), Record("n1", "name", "teo", "Teo Varr", 0, "p0", None)]
        return canon, recs, [_mem(1, "teo", 5, "I stood on the east landing.", "east")]
    def out_minting(name, into):
        return {"narration": "x", "perceiver_tuples": [{"perceiver": "teo", "text": "I felt in my coat.", "salience": "ordinary"}],
                "new_entities": [{"id": "NEW:it", "name": name, "in": into}],
                "scene_tuples": [{"scene": "east", "text": "The boy felt in his coat.", "present": ["teo"]}],
                "suggested_next_action": ""}
    canon, recs, mems = setup()
    _, events, client = _pulse(monkeypatch, canon, "teo", "feels in his coat", 6, mems, out_minting("Teo Varr's knife", "teo"), records=recs)
    assert len(client.calls) == 1 and not [e for e, _ in events if e == "smrti.name_rerun"]
    canon, recs, mems = setup()
    _, events, client = _pulse(monkeypatch, canon, "teo", "looks up", 6, mems, out_minting("Old Varr the smith", "east"), records=recs)
    assert len(client.calls) == 2 and any(e == "smrti.name_rerun" and f["taken"] == {"Varr": "teo"} for e, f in events)


def _town():
    canon = Maya(world_time=10, entities={
        "arkham": Entity("arkham", ["Arkham"], 0, "genesis"),
        "garrison": Entity("garrison", ["Garrison Street"], 0, "genesis"),
        "salt": Entity("salt", ["Saltonstall Street"], 0, "genesis"),
        "house": Entity("house", ["the green-door house"], 0, "genesis"),
        "parlour": Entity("parlour", ["the parlour"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"),
        "aunt": Entity("aunt", ["the aunt"], 0, "genesis"),
        "waders": Entity("waders", ["the rubber waders"], 0, "genesis"),
    })
    recs = [_rec(1, "garrison", "arkham", 0), _rec(2, "salt", "arkham", 0), _rec(3, "house", "garrison", 0),
            _rec(4, "parlour", "house", 0), _rec(5, "wen", "parlour", 0), _rec(6, "aunt", "parlour", 0), _rec(7, "waders", "parlour", 0)]
    mems = [_mem(1, "wen", 5, "I sat in the aunt's parlour.", "parlour"), _mem(2, "aunt", 5, "The man sat in my parlour.", "parlour")]
    return canon, recs, mems


def _parlour_out(new, moves, extra_tuples=()):
    return {"narration": "x",
            "perceiver_tuples": [{"perceiver": "wen", "text": "She told me where he rooms.", "salience": "ordinary"},
                                 {"perceiver": "aunt", "text": "I told him where my nephew rooms.", "salience": "ordinary"}, *extra_tuples],
            "new_entities": new, "moves": moves,
            "scene_tuples": [{"scene": "parlour", "text": "The aunt told the visitor where her nephew rooms.", "present": ["wen", "aunt"]}],
            "suggested_next_action": ""}


def test_a_place_only_spoken_of_is_a_mention_and_nothing_is_put_in_it(monkeypatch) -> None:
    """Collision run, Ravenmoor 665 (question 4): the aunt said Elias took the
    waders to his rooms; the pulse minted the rooms on Saltonstall Street,
    where nobody was, and moved the waders into them. Now the rooms are a
    name placed nowhere, and the waders stay where they were seen."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    canon, recs, mems = _town()
    out = _parlour_out([{"id": "NEW:elias_rooms", "name": "Elias's rooms", "in": "salt", "scene": True}],
                       [{"entity": "waders", "into": "NEW:elias_rooms"}])
    r, events, _ = _pulse(monkeypatch, canon, "wen", "asks where he rooms", 6, mems, out, records=recs)
    rooms = r.resolved_new["NEW:elias_rooms"]
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.container_at(rooms, 7) is None and k.named("Elias's rooms") == {rooms}   # a name, placed nowhere
    assert k.container_at("waders", 7) == "parlour"
    mention = next(f for e, f in events if e == "smrti.mention")
    assert mention["ids"] == [rooms] and mention["held_back"] == ["waders"]


def test_a_mention_is_placed_when_someone_reaches_it(monkeypatch) -> None:
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    canon, recs, mems = _town()
    canon.entities["rooms"] = Entity("rooms", ["Elias's rooms"], 6, "smrti", avarana="scene")
    recs.append(Record("n9", "name", "rooms", "Elias's rooms", 6, "p6", None))
    mems.append(_mem(3, "wen", 6, "She said he rooms on Saltonstall Street.", "parlour", subjects=("rooms",)))  # met, placed nowhere: KNOWN
    out = {"narration": "x", "perceiver_tuples": [{"perceiver": "wen", "text": "I climbed to his rooms.", "salience": "ordinary"}],
           "moves": [{"entity": "rooms", "into": "salt"}, {"entity": "wen", "into": "rooms"}],
           "scene_tuples": [{"scene": "rooms", "text": "A visitor came up to the rooms.", "present": ["wen"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "wen", "goes to Elias's rooms", 8, mems, out, records=recs)
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.container_at("rooms", 9) == "salt" and k.scene_of("wen", 9) == "rooms"
    assert not [e for e, _ in events if e == "smrti.mention"]


def test_a_person_put_where_nobody_has_been_is_a_mention(monkeypatch) -> None:
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    # Ravenmoor 524 and 530: the grey-faced young man, put into a reading room nobody had reached, twice
    canon, recs, mems = _town()
    canon.entities["rooms"] = Entity("rooms", ["Elias's rooms"], 6, "smrti", avarana="scene")
    out = _parlour_out([{"id": "NEW:young_man", "name": "a grey-faced young man", "in": "rooms", "scene": False}], [])
    r, events, _ = _pulse(monkeypatch, canon, "wen", "asks who is there", 8, mems, out, records=recs)
    man = r.resolved_new["NEW:young_man"]
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.container_at(man, 9) is None and k.container_at("rooms", 9) is None
    assert next(f for e, f in events if e == "smrti.mention")["ids"] == sorted([man, "rooms"])  # the room is one too


# -- a mention is in the earth, and a telling that makes it new is shown it (Marc, 2026-10-04) --

class _Telling(_FakeClient):
    """A first telling, then what the rerun answers."""
    def __init__(self, *outs: dict) -> None:
        super().__init__(outs[0])
        self.outs = list(outs)

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
        self.calls.append((system, user))
        return self.outs[min(len(self.calls), len(self.outs)) - 1]


def _town_with_earth(mill_known: bool = False, mill_in_earth: bool = True):
    canon, recs, mems = _town()
    canon.entities["earth"] = Entity("earth", ["the world at large"], 0, "itihasa")
    canon.entities["mill"] = Entity("mill", ["the old mill"], 6, "smrti", avarana="scene")
    recs += [_rec(80, "arkham", "earth", 0, note="earth"), Record("n80", "name", "mill", "the old mill", 6, "p6", None)]
    if mill_in_earth:  # else: a mention from before mentions were filed in the earth -- in nothing, and not shown
        recs.append(_rec(81, "mill", "earth", 6, note="mention"))
    if mill_known:
        mems.append(_mem(3, "wen", 6, "The aunt said the old mill stands empty.", "parlour", subjects=("mill",)))
    return canon, recs, mems


def _walk_to(mill: str, new: list[dict]) -> dict:
    return {"narration": "x", "perceiver_tuples": [{"perceiver": "wen", "text": "I walked out to the old mill.", "salience": "ordinary"}],
            "new_entities": new, "moves": [{"entity": "wen", "into": mill}],
            "scene_tuples": [{"scene": mill, "text": "A visitor came to the old mill.", "present": ["wen"]}],
            "suggested_next_action": ""}


def test_a_mention_is_filed_in_the_earth(monkeypatch) -> None:
    """Marc, 2026-10-04: a thing named in passing goes in the earth like anything unplaced, "otherwise it's
    unreachable" -- in the tree, still in no scene, and nothing put into it; no rerun for it."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    canon, recs, mems = _town_with_earth()
    out = _parlour_out([{"id": "NEW:elias_rooms", "name": "Elias's rooms", "in": "salt", "scene": True}],
                       [{"entity": "waders", "into": "NEW:elias_rooms"}])
    r, events, client = _pulse(monkeypatch, canon, "wen", "asks where he rooms", 7, mems, out, records=recs)
    rooms = r.resolved_new["NEW:elias_rooms"]
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.container_at(rooms, 8) == "earth" and k.root(rooms, 8) == "earth"  # in the tree; a scene of its own, in none
    assert any(x.kind == "kosha" and x.entity_id == rooms and x.value == "earth" and x.note == "mention" for x in r.records_written)
    assert k.container_at("waders", 8) == "parlour"
    assert len(client.calls) == 1 and not [e for e, _ in events if e.endswith("_rerun")]


def test_a_mention_in_the_earth_is_shown_so_a_stranger_finds_it(monkeypatch) -> None:
    """In the earth, a mention stands on THE TREE's top line, shown to every pulse -- what anchors an unplaced thing's
    identity (the is_place replay, 2026-10-03: hidden, Harold Wheelock was minted again 3 of 3). So a stranger who
    walks out to "the old mill" folds onto it at once (v5 §17: what is shown is acquainted), and it is placed there."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    canon, recs, mems = _town_with_earth()
    out = _walk_to("NEW:mill2", [{"id": "NEW:mill2", "name": "the old mill", "in": "salt", "scene": True}])
    r, events, client = _pulse(monkeypatch, canon, "wen", "walks out to the old mill", 7, mems, out, records=recs)
    assert '- earth "the world at large" (the world\'s top) holds: mill "the old mill"' in client.calls[0][1]
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert len(client.calls) == 1 and k.container_at("mill", 8) == "salt" and k.scene_of("wen", 8) == "mill"
    assert k.named("the old mill") == {"mill"}  # no second mill


def test_a_stranger_who_makes_an_unshown_mention_new_is_told_again_shown_it(monkeypatch) -> None:
    """The mention replay (2026-10-04): a mention no pulse shows (from before mentions went into the earth) is
    minted again by its name. The telling is asked once more, shown NAMED IN PASSING, NOT YET FOUND; told so, it
    uses the mention's id, and the mill is placed where it is found. The engine itself never folds on the name."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    canon, recs, mems = _town_with_earth(mill_in_earth=False)
    first = _walk_to("NEW:mill2", [{"id": "NEW:mill2", "name": "the old mill", "in": "salt", "scene": True}])
    second = _walk_to("mill", [])
    second["moves"] = [{"entity": "mill", "into": "salt"}, {"entity": "wen", "into": "mill"}]
    client = _Telling(first, second)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    from smritilila.smrti.pulse import run_smrti_pulse
    r = run_smrti_pulse(canon, "wen", "walks out to the old mill", 7, "p7", mems, client, None, records=recs)
    assert len(client.calls) == 2
    rerun = next(f for e, f in events if e == "smrti.mention_rerun")
    assert rerun["mentions"] == {"NEW:mill2": "mill"}
    assert "NAMED IN PASSING, NOT YET FOUND" in client.calls[1][1] and "NAMED IN PASSING" not in client.calls[0][1]
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.container_at("mill", 8) == "salt" and k.scene_of("wen", 8) == "mill"
    assert "mill2" not in str(sorted(canon.entities)) and k.named("the old mill") == {"mill"}  # no second mill


def test_a_mention_declared_again_by_its_own_id_is_placed_where_it_is_found(monkeypatch) -> None:
    """The replay's form slip: told of the mention, the Narrator sometimes lists its id among the new, with where
    it is. The binder resolves it to the mention; its first placement contradicts no record, so it is placed."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    canon, recs, mems = _town_with_earth(mill_in_earth=False)
    first = _walk_to("NEW:mill2", [{"id": "NEW:mill2", "name": "the old mill", "in": "salt", "scene": True}])
    second = _walk_to("mill", [{"id": "mill", "name": "the old mill", "in": "salt", "scene": True}])
    client = _Telling(first, second)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    from smritilila.smrti.pulse import run_smrti_pulse
    r = run_smrti_pulse(canon, "wen", "walks out to the old mill", 7, "p7", mems, client, None, records=recs)
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert len(client.calls) == 2 and k.container_at("mill", 8) == "salt" and k.scene_of("wen", 8) == "mill"
    assert any(e == "smrti.move" and f["why"] == "mention_placed" and f["entity"] == "mill" for e, f in events)


def test_one_who_met_the_mention_folds_onto_it_and_it_is_placed(monkeypatch) -> None:
    """Acquainted, the declared name folds onto the mention at once (v5 §17) -- and the fold must place it where the
    telling says, or the actor walks into a scene that stands in no scene. No rerun."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    canon, recs, mems = _town_with_earth(mill_known=True)
    out = _walk_to("NEW:mill2", [{"id": "NEW:mill2", "name": "the old mill", "in": "salt", "scene": True}])
    r, events, client = _pulse(monkeypatch, canon, "wen", "walks out to the old mill", 7, mems, out, records=recs)
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert len(client.calls) == 1 and k.container_at("mill", 8) == "salt" and k.scene_of("wen", 8) == "mill"
    assert any(e == "smrti.name_fold" and f["onto"] == "mill" for e, f in events)


# -- sandhi: a boundary thing is in both scenes it joins (collision run, question 4) --

def _house():
    canon = Maya(world_time=10, entities={
        "arkham": Entity("arkham", ["Arkham"], 0, "genesis"),
        "house": Entity("house", ["the house"], 0, "genesis"),
        "bedroom": Entity("bedroom", ["the bedroom"], 0, "genesis", avarana="scene"),
        "hall": Entity("hall", ["the front hall"], 0, "genesis", avarana="scene"),
        "window": Entity("window", ["the bedroom window"], 0, "genesis"),
        "nat": Entity("nat", ["Nathaniel"], 0, "genesis"),
    })
    recs = [_rec(1, "house", "arkham", 0), _rec(2, "bedroom", "house", 0), _rec(3, "hall", "house", 0),
            _rec(4, "window", "bedroom", 0), _rec(5, "nat", "bedroom", 0)]
    mems = [_mem(1, "nat", 5, "I woke in the bedroom.", "bedroom"), _mem(2, "bedroom", 5, "The bedroom was cold.", "bedroom", tier="AMBIENT")]
    return canon, recs, mems


def test_a_sandhi_joins_two_scenes_both_ways_and_can_close() -> None:
    canon, recs, mems = _house()
    canon.entities["street"] = Entity("street", ["the street"], 6, "smrti", avarana="scene")
    recs += [_rec(6, "street", "arkham", 6), Record("s1", "sandhi", "window", "street", 6, "p6", "open")]
    k = Kosha(canon, mems, recs)
    assert k.joins("window", 7) == {"bedroom", "street"} and k.joins("window", 6) == set()  # from the next tick, as containment
    assert k.across("bedroom", 7) == [("window", "street")] and k.across("street", 7) == [("window", "bedroom")]
    assert k.neighbours("bedroom", 7)[0] == "street"          # across a boundary first, then the tree's
    assert "bedroom" in k.neighbours("street", 7)
    f = k.clone(fork=True)
    f.add(Record("s2", "sandhi", "window", "street", 8, "p8", "closed"))
    assert f.across("bedroom", 9) == [] and k.across("bedroom", 9) == [("window", "street")]  # a fork's own, the world's untouched


def test_a_boundary_looked_across_joins_the_far_side_and_places_what_is_seen(monkeypatch) -> None:
    """Ravenmoor 662: Nathaniel at the bedroom window saw the milkman in the
    street. The window was the bedroom's alone, the street was minted beside
    it, and under the mention rule the street and the milkman were placed
    nowhere -- seeing is not being told. The window is in both now."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    canon, recs, mems = _house()
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "nat", "text": "I looked down at the milkman in the street.", "salience": "ordinary"}],
           "new_entities": [{"id": "NEW:street", "name": "the street outside", "in": "arkham", "scene": True},
                            {"id": "NEW:milkman", "name": "the milkman", "in": "NEW:street", "scene": False}],
           "boundaries": [{"thing": "window", "also_in": "NEW:street"}],
           "scene_tuples": [{"scene": "bedroom", "text": "A man looked out of the bedroom window.", "present": ["nat"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "nat", "looks down at the street", 6, mems, out, records=recs)
    street, milkman = r.resolved_new["NEW:street"], r.resolved_new["NEW:milkman"]
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.joins("window", 7) == {"bedroom", street}
    assert k.container_at(street, 7) == "arkham" and k.container_at(milkman, 7) == street   # seen across: placed
    assert not [e for e, _ in events if e == "smrti.mention"]
    assert r.route == ["bedroom", street]


def test_a_route_through_a_door_reaches_the_room_passed_through(monkeypatch) -> None:
    """Ravenmoor 580: Wentworth broke Room 214's glass, went in, saw the desk,
    and ended on the front steps -- the engine knew only where he ended."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", True)
    canon, recs, mems = _house()
    canon.entities["door"] = Entity("door", ["the hall door"], 0, "genesis")
    canon.entities["study"] = Entity("study", ["the study"], 0, "smrti", avarana="scene")
    recs += [_rec(6, "door", "bedroom", 0), _rec(7, "study", "house", 0)]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "nat", "text": "I went through the study and out into the hall.", "salience": "ordinary"}],
           "new_entities": [{"id": "NEW:desk", "name": "the desk", "in": "study", "scene": False}],
           "boundaries": [{"thing": "door", "also_in": "study"}],
           "moves": [{"entity": "nat", "into": "hall"}],
           "scene_tuples": [{"scene": "hall", "text": "A man came into the hall.", "present": ["nat"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "nat", "goes through the study to the hall", 6, mems, out, records=recs)
    desk = r.resolved_new["NEW:desk"]
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert r.route == ["bedroom", "study", "hall"] and k.container_at(desk, 7) == "study"


def test_the_next_pulse_is_shown_what_is_across(monkeypatch) -> None:
    canon, recs, mems = _house()
    canon.entities["street"] = Entity("street", ["the street"], 6, "smrti", avarana="scene")
    canon.entities["milk"] = Entity("milk", ["the milkman"], 6, "smrti")
    recs += [_rec(6, "street", "arkham", 6), _rec(7, "milk", "street", 6), Record("s1", "sandhi", "window", "street", 6, "p6", "open")]
    mems += [_mem(3, "milk", 6, "I set the bottles on the step.", "street")]
    out = {"narration": "x", "perceiver_tuples": [{"perceiver": "nat", "text": "I watched.", "salience": "ordinary"}],
           "scene_tuples": [{"scene": "bedroom", "text": "A man watched from the window.", "present": ["nat"]}],
           "suggested_next_action": ""}
    _, _, client = _pulse(monkeypatch, canon, "nat", "watches the street", 8, mems, out, records=recs)
    prompt = client.calls[0][1]
    assert "ACROSS A BOUNDARY OF THIS SCENE" in prompt and "I set the bottles on the step." in prompt


def test_a_sandhi_is_refused_for_a_player_or_a_side_that_is_not_a_scene(monkeypatch) -> None:
    # a thing that perceives may still be a boundary: a door knocked on has tuples of its own (live, collide 671)
    canon, recs, mems = _house()
    mems.append(_mem(9, "window", 5, "Something tapped on me.", "bedroom"))
    out = {"narration": "x", "perceiver_tuples": [{"perceiver": "nat", "text": "I stood in the doorway.", "salience": "ordinary"}],
           "boundaries": [{"thing": "nat", "also_in": "hall"}, {"thing": "window", "also_in": "house"}],
           "scene_tuples": [{"scene": "bedroom", "text": "A man stood in the doorway.", "present": ["nat"]}],
           "suggested_next_action": ""}
    out["boundaries"].append({"thing": "window", "also_in": "hall"})
    r, events, _ = _pulse(monkeypatch, canon, "nat", "stands in the doorway", 6, mems, out, records=recs, players=frozenset({"nat"}))
    refused = next(f["refused"] for e, f in events if e == "smrti.move_refused")
    assert {(x["entity"], x["reason"]) for x in refused} == {("nat", "a_player"), ("window", "not_a_scene")}
    assert [(x.entity_id, x.value) for x in r.records_written if x.kind == "sandhi"] == [("window", "hall")]  # a perceiver, and a boundary


def test_the_roster_never_moves_a_boundary_into_the_scene_it_already_joins(monkeypatch) -> None:
    """Live, collide 678: a lodger went through Lowell's door; the scene row
    listed the door present in the room -- where it is, by its sandhi -- and
    the roster moved it off the landing, which lost its door."""
    canon, recs, mems = _house()
    recs.append(Record("s1", "sandhi", "window", "hall", 5, "p5", "open"))
    out = {"narration": "x", "perceiver_tuples": [{"perceiver": "nat", "text": "I climbed through into the hall.", "salience": "ordinary"}],
           "moves": [{"entity": "nat", "into": "hall"}],
           "scene_tuples": [{"scene": "hall", "text": "A man climbed in through the window.", "present": ["nat", "window"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "nat", "climbs through the window", 6, mems, out, records=recs)
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    assert k.container_at("window", 7) == "bedroom" and k.joins("window", 7) == {"bedroom", "hall"}
    assert k.scene_of("nat", 7) == "hall"


def test_sandhi_is_a_world_s_own_switch(monkeypatch) -> None:
    """Marc, 2026-09-25: on in Ravenmoor and the Crossing -- world data
    "sandhi": true, with the server switch off; off in every other world."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", False)
    out = _parlour_out([{"id": "NEW:elias_rooms", "name": "Elias's rooms", "in": "salt", "scene": True}],
                       [{"entity": "waders", "into": "NEW:elias_rooms"}])
    for on in (True, False):
        canon, recs, mems = _town()
        view = {"darsana": None, "becomings": {}, "world_data": {"sandhi": True} if on else {}}
        r, events, client = _pulse(monkeypatch, canon, "wen", "asks where he rooms", 6, mems, out, records=recs, view=view)
        assert ("15. BOUNDARIES" in client.calls[0][0]) == on and "<<BOUNDARIES>>" not in client.calls[0][0]
        assert bool([e for e, _ in events if e == "smrti.mention"]) == on


def test_a_second_narration_is_the_same_call_with_only_its_change(monkeypatch) -> None:
    """Each rerun copied a 17-argument call by hand, and all three dropped
    `shown` (a window turn's "never describe the player" lines) and
    `window`. Now the call's own arguments, captured at entry, with only
    what the rerun changes."""
    import smritilila.smrti.pulse as P
    real = P.run_smrti_pulse
    again: list[dict] = []

    def watch(*a, **kw):
        again.append(kw)
        return real(*a, **kw)
    canon = _crossing()
    canon.entities["edm"] = Entity("edm", ["the Lowell boy"], 0, "smrti")
    recs = [_rec(1, "teo", "east", 0), _rec(2, "trav", "east", 0), _rec(3, "edm", "west", 0),
            Record("n1", "name", "edm", "Edmund Lowell", 0, "p0", None)]
    mems = [_mem(1, "teo", 5, "I stood on the east landing.", "east"), _mem(2, "trav", 5, "I waited by the stone.", "east"),
            _mem(3, "edm", 5, "I drove north.", "west")]
    out = {"narration": "The traveler gives her name.",
           "perceiver_tuples": [{"perceiver": "teo", "text": "The traveler told me her name.", "salience": "ordinary"}],
           "aliases": [{"entity": "trav", "name": "Edith Lowell"}],
           "scene_tuples": [{"scene": "east", "text": "The traveler said her name.", "present": ["teo", "trav"]}],
           "suggested_next_action": ""}
    monkeypatch.setattr(P, "run_smrti_pulse", watch)
    _pulse(monkeypatch, canon, "teo", "asks her name", 6, mems, out, records=recs, shown=["never describe the boy"], wait_until=3)
    assert len(again) == 2  # the pulse, and its name rerun once (a pulse given a window never reruns)
    first, kw = again
    assert kw["shown"] == ["never describe the boy"] and kw["wait_until"] == 3 and kw["action_text"] == "asks her name"
    assert set(kw["name_taken"]) == {"Lowell"} and "name_taken" not in first


def test_energy_measures_where_a_pulse_stands_relative_to_the_players() -> None:
    """Marc, 2026-09-26: "Players inject energy into the world." A pulse's
    distance from the nearest player, in containment-tree steps, and the
    ticks since a player was in its scene -- read against the echo rate."""
    from smritilila.smrti.pulse import player_distance, ticks_since_player
    scene = lambda i, n: Entity(i, [n], 0, "genesis", avarana="scene")
    canon = Maya(world_time=20, entities={
        "county": Entity("county", ["the county"], 0, "genesis"), "town": Entity("town", ["the town"], 0, "genesis"),
        "inn": scene("inn", "the inn"), "yard": scene("yard", "the yard"), "moor": scene("moor", "the moor"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"), "cup": Entity("cup", ["the cup"], 0, "genesis")})
    recs = [_rec(1, "town", "county", 0), _rec(2, "inn", "town", 0), _rec(3, "yard", "town", 0),
            _rec(4, "teo", "yard", 0), _rec(5, "cup", "inn", 0)]
    mems = [_mem(1, "inn", 0, "A low room.", "inn"), _mem(2, "yard", 0, "Mud and a gate.", "yard"),
            _mem(3, "teo", 5, "I sat by the fire.", "inn"), _mem(4, "teo", 9, "I went out to the yard.", "yard"),
            _mem(5, "cup", 12, "I stood on the table.", "inn")]
    log, _ = build_log(canon, mems, recs)
    assert player_distance(log, "yard", 13, {"teo"}) == (0, "teo")      # the same scene
    assert player_distance(log, "inn", 13, {"teo"}) == (2, "teo")       # inn -> town <- yard
    assert player_distance(log, "moor", 13, {"teo"}) == (None, None)    # no tree they share
    assert player_distance(log, "inn", 13, set()) == (None, None)
    assert ticks_since_player(log, "inn", 13, {"teo"}) == 8             # his tuple at 5
    assert ticks_since_player(log, "moor", 13, {"teo"}) is None


def test_every_pulse_says_what_brought_the_present_in_and_what_echoed(monkeypatch) -> None:
    """smrti.energy, once a pulse, zeros included: the denominator the
    echo rate needs over time."""
    canon, mems, recs = _landing()
    out = {"narration": "You look along the river.",
           "perceiver_tuples": [{"perceiver": "nat", "text": "I looked along the river at the mist.", "salience": "ordinary"}],
           "scene_tuples": [], "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "nat", "Look along the river.", 10, mems, out, records=recs,
                          playable=frozenset({"nat", "wen"}))
    energy = [f for e, f in events if e == "smrti.energy"]
    assert len(energy) == 1
    f = energy[0]
    assert f["player_action"] is True and f["autonomous"] is False and f["steps_to_player"] == 0
    assert f["tuples"] == 1 and f["echoes"] == 0 and f["dead"] == 0 and f["tics"] == 0 and f["sentences"] >= 1


def test_daiva_draws_the_spread_by_the_pulse() -> None:
    """Daiva (Marc's name): the engine supplies chance, judging nothing.
    The same pulse draws the same word (a held pulse's second pass, a
    replay); the spread holds; a world may reshape it or switch it off."""
    from collections import Counter
    from smritilila.smrti.narrator import DAIVA_WORDS
    from smritilila.smrti.pulse import DAIVA_SPREAD, daiva
    assert daiva("pulse-a") == daiva("pulse-a")
    n = 20000
    got = Counter(daiva(f"p{i}") for i in range(n))
    for word, share in zip(DAIVA_WORDS, DAIVA_SPREAD):
        assert abs(got[word] / n - share / sum(DAIVA_SPREAD)) < 0.015, word
    assert daiva("pulse-a", {"daiva": {"off": True}}) is None
    assert {daiva(f"p{i}", {"daiva": {"spread": [0, 0, 0, 1, 0, 0, 0]}}) for i in range(200)} == {"even"}
    assert daiva("pulse-a", {"daiva": {"spread": [1, 2]}}) == daiva("pulse-a")  # malformed: the default spread


def test_the_balanced_spread_is_centred_and_ruinous_is_one_in_a_hundred() -> None:
    """Marc, 2026-09-30, after Jakal's moathouse fight went easy on a run of good draws: "back to the balanced spread
    with the second model ... more centered around the midpoint, fewer extremes at either end." The further a word is from even the
    rarer it is, and the five words from grim to fortunate are symmetric. 2026-10-17 ("Actually 1%"): ruinous, the one
    word that costs for good, is drawn one time in a hundred, rarer than its opposite; the shares still sum to a hundred."""
    from smritilila.smrti.pulse import DAIVA_SPREAD
    assert sum(DAIVA_SPREAD) == 100 and DAIVA_SPREAD[0] == 1 and DAIVA_SPREAD[0] < DAIVA_SPREAD[-1]
    assert tuple(DAIVA_SPREAD[1:6]) == tuple(reversed(DAIVA_SPREAD[1:6]))      # grim .. fortunate, symmetric
    assert all(a < b for a, b in zip(DAIVA_SPREAD[:3], DAIVA_SPREAD[1:4]))   # rising to the middle
    assert all(a > b for a, b in zip(DAIVA_SPREAD[3:6], DAIVA_SPREAD[4:7]))  # and falling from it
    total = sum(DAIVA_SPREAD)
    assert (DAIVA_SPREAD[0] + DAIVA_SPREAD[-1]) / total <= 0.06 and DAIVA_SPREAD[3] / total >= 1 / 3  # few extremes, a heavy middle


def test_daiva_leans_by_the_model_that_narrates() -> None:
    """Marc, 2026-09-28 ("a different distribution of luck for each individual llm"): measured blind, the bad words
    bite and the good barely help, so the spread leans by the narrating model; a world's own spread still wins, and
    an unknown model keeps the even-handed one. A rotating client plans its pick, and the call uses that pick.
    (The table is empty since 2026-09-30: every model draws the balanced spread.)"""
    from collections import Counter
    from smritilila.llm.client import RotatingClient
    from smritilila.smrti.narrator import DAIVA_WORDS
    from smritilila.smrti.pulse import DAIVA_SPREAD, DAIVA_SPREAD_BY_MODEL, daiva, daiva_spread
    assert daiva_spread("model-b") == DAIVA_SPREAD  # Marc, 2026-09-30: the second model back on the balanced spread
    assert daiva_spread("model-a") == DAIVA_SPREAD  # and the first model, the same evening: every pulse the same luck
    assert DAIVA_SPREAD_BY_MODEL == {}
    assert daiva_spread("some-open-model") == DAIVA_SPREAD and daiva_spread(None) == DAIVA_SPREAD
    n = 20000
    got = Counter(daiva(f"p{i}", None, "model-b") for i in range(n))
    spread = DAIVA_SPREAD
    for word, share in zip(DAIVA_WORDS, spread):
        assert abs(got[word] / n - share / sum(spread)) < 0.015, word
    assert {daiva(f"p{i}", {"daiva": {"spread": [0, 0, 0, 1, 0, 0, 0]}}, "model-b") for i in range(100)} == {"even"}

    class M:
        def __init__(self, model):
            self.model, self.calls = model, 0

        def complete(self, *a, **k):
            self.calls += 1
            return self.model
    a, b = M("model-a"), M("model-b")
    rc = RotatingClient(a, b, share=0.5)
    for _ in range(6):
        planned = rc.plan()
        assert rc.complete("s", "u") == planned  # the planned model is the one that answers


def test_the_narrator_is_shown_the_pulses_fortune_and_its_scale(monkeypatch) -> None:
    from smritilila.smrti.narrator import DAIVA_WORDS, SYSTEM
    assert "<<DAIVA>>" not in SYSTEM and all(f"  {w} -- " in SYSTEM for w in DAIVA_WORDS)
    canon, mems, recs = _landing()
    out = {"narration": "You look along the river.",
           "perceiver_tuples": [{"perceiver": "nat", "text": "I looked along the river at the mist.", "salience": "ordinary"}],
           "scene_tuples": [], "suggested_next_action": ""}
    r, events, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 10, mems, out, records=recs)
    drawn = [f for e, f in events if e == "smrti.daiva" and f["actor"] == "nat"]
    assert len(drawn) == 1 and drawn[0]["word"] in DAIVA_WORDS
    # v5 §28: each one else present and lately active draws a fortune of their own, by who and when
    others = {f["actor"]: f["word"] for e, f in events if e == "smrti.daiva" and f["actor"] != "nat"}
    assert others and all(w in DAIVA_WORDS for w in others.values()) and all(f.get("sankalpa") for e, f in events
                                                                             if e == "smrti.daiva" and f["actor"] != "nat")
    assert f"FORTUNE THIS PULSE, IF THE ACTION HOLDS ANYTHING IN DOUBT: {drawn[0]['word']}." in client.calls[0][1]
    assert next(f for e, f in events if e == "smrti.energy")["daiva"] == drawn[0]["word"]
    # a world with it off: no line, no draw
    r, events, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 10, mems, out, records=recs,
                               view={"world_data": {"daiva": {"off": True}}})
    assert "FORTUNE THIS PULSE" not in client.calls[0][1] and not any(e == "smrti.daiva" for e, _ in events)


def test_a_fortune_is_whose_and_when_so_declining_cannot_reroll_it() -> None:
    """Stonemarch t205 (2026-09-27): three declined attempts at one moment drew
    adverse, providential, adverse. The draw is now by who and when: taken
    back and tried again at the same tick, the same fortune."""
    from smritilila.smrti.pulse import daiva, fortune_seed
    assert daiva(fortune_seed("jakal", 205)) == daiva(fortune_seed("jakal", 205))
    assert len({daiva(fortune_seed("jakal", t)) for t in range(200, 260)}) > 1  # the moment moves, the fortune may too
    assert len({daiva(fortune_seed(p, 205)) for p in [f"p{i}" for i in range(40)]}) > 1  # each one's own


def test_a_quiet_thing_that_never_kept_a_memory_is_folded_out_of_the_draw() -> None:
    """Absorption (b), Marc 2026-09-29: quiet for ABSORBED_QUIET_TICKS and never
    having kept a memory, a candidate is not drawn on its own. A person who once
    kept one (Jaroo, by his teaching) stays, however quiet; a mention wakes."""
    from smritilila import main
    canon = Maya(world_time=500, entities={
        "lair": Entity("lair", ["the ogre's lair"], 0, "smrti", avarana="scene"),
        "chest": Entity("chest", ["an iron-banded chest"], 360, "smrti"),
        "jaroo": Entity("jaroo", ["Jaroo Ashstaff"], 0, "genesis"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
    })
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["lair"], playable_entity_ids=["teo"])
    world.recent_memories = [_mem(1, "chest", 371, "Hands pried at my hasp.", "lair", use="none"),
                             _mem(2, "jaroo", 391, "I taught the scholar the old herbs.", "lair", use="key"),
                             _mem(3, "teo", 495, "I walked the road.", "lair")]
    world.records = [_rec(1, "chest", "lair", 360), _rec(2, "jaroo", "lair", 0)]
    assert pravrtti._fold_absorbed(world, ["chest", "jaroo"]) == ["jaroo"]
    world.recent_memories.append(_mem(4, "teo", 499, "I looked at an iron-banded chest.", "lair"))
    assert pravrtti._fold_absorbed(world, ["chest", "jaroo"]) == ["chest", "jaroo"]  # mentioned lately: awake
    canon.world_time = 400  # within the window of its own last tuple: not quiet
    world.recent_memories = world.recent_memories[:3]
    assert "chest" in pravrtti._fold_absorbed(world, ["chest", "jaroo"])
    # the world at large is the age's chronicler: never kept, never folded (Stonemarch t521 folded it)
    canon.world_time = 500
    canon.entities["age"] = Entity("age", ["the world at large"], 0, "itihasa")
    world.recent_memories.append(_mem(9, "age", 300, "The season turned in the Viscounty.", "age", use="none"))
    assert "age" in pravrtti._fold_absorbed(world, ["chest", "age"]) and "chest" not in pravrtti._fold_absorbed(world, ["chest", "age"])


def test_those_with_the_actor_lately_and_not_here_now_are_shown() -> None:
    """Marc, 2026-09-29: noticeable departures belong in the telling. Those who
    shared the actor's scene in their last few moments and are not here now
    are shown -- whoever ever kept a memory; not a rope, not who is here."""
    from smritilila.smrti.narrator import render
    canon = Maya(world_time=40, entities={
        "guardroom": Entity("guardroom", ["the guardroom"], 0, "genesis", avarana="scene"),
        "passage": Entity("passage", ["the narrow passage"], 0, "genesis", avarana="scene"),
        "jakal": Entity("jakal", ["Jakal"], 0, "genesis"),
        "dunstan": Entity("dunstan", ["Dunstan"], 0, "genesis"),
        "elmo": Entity("elmo", ["Elmo"], 0, "genesis"),
        "rope": Entity("rope", ["a hemp rope"], 0, "genesis"),
    })
    recs = [_rec(1, "jakal", "guardroom", 0), _rec(2, "dunstan", "guardroom", 0), _rec(3, "elmo", "guardroom", 0),
            _rec(4, "rope", "guardroom", 0), _rec(5, "jakal", "passage", 31), _rec(6, "dunstan", "passage", 31)]
    mems = [_mem(1, "jakal", 30, "We bound the gnolls.", "guardroom"), _mem(2, "elmo", 30, "I kept my axe on them.", "guardroom", use="useful"),
            _mem(3, "rope", 30, "I was knotted tight.", "guardroom", use="none"), _mem(4, "dunstan", 30, "I tied the knots.", "guardroom", use="useful"),
            _mem(5, "jakal", 31, "I followed Dunstan into the passage.", "passage")]
    log, _ = build_log(canon, mems, recs)
    ctx = retrieve(log, "jakal", 32, None, k_place=3, k_sim=0, action="I look around.")
    assert [e[0] for e in ctx.lately] == ["elmo"]  # Dunstan is here; the rope never kept a memory
    assert 'WITH YOU LATELY, NOT HERE NOW' in render(ctx, "I look around.", set())


# -- 2026-09-30: the kinds v4 replaced, retired where they still decided -------------------------------


def test_a_scene_is_the_scene_bit_and_nothing_a_tag_says() -> None:
    """v5 §11/§1: no type may tell a scene from anything else. svabhava was read
    beside the bit, and decided nothing in the four live worlds that the bit
    did not; it is read no more."""
    canon = Maya(world_time=1, entities={
        "hall": Entity("hall", ["the hall"], 0, "genesis"),
        "hut": Entity("hut", ["the hut"], 0, "genesis", avarana="scene")})
    k = Kosha(canon)
    assert not k.is_scene("hall") and k.is_scene("hut")


def test_a_name_is_derived_only_for_what_came_before_a_world_declared_names() -> None:
    """v5 §16, Appendix B: names are declared; derivation from a label's capitals stays for
    worlds born before names were -- not for a thing minted, unnamed, after
    (6 such in the Crossing and Ravenmoor were given names nobody declared)."""
    canon = Maya(world_time=9, entities={
        "smith": Entity("smith", ["Brother Smyth at his forge"], 0, "genesis"),
        "bren": Entity("bren", ["the miller's girl"], 3, "smrti"),
        "cup": Entity("cup", ["the Old Mill cup"], 5, "smrti"),
        "arkham": Entity("arkham", ["Arkham"], 6, "genesis")})
    k = Kosha(canon, [], [Record("n1", "name", "bren", "Bren Hale", 3, "p3", None)])
    assert "smyth" in k.names_of("smith")                   # before the world declared a name: derived, as ever
    assert k.names_of("bren") == {"bren hale"}              # declared
    assert k.names_of("cup") == set()                       # minted in play after, never named: it has no name
    assert "arkham" in k.names_of("arkham")                 # authored by a seed after: its label is the author's name


def test_a_name_is_made_whole_by_the_worlds_own_titles() -> None:
    """v5 §16 v4.7 (Marc, 2026-09-25) kept; its English titles become world data
    (v5 §16), today's list the default."""
    from smritilila.smrti.kosha import whole_name
    assert not whole_name("Brother Smyth") and whole_name("Ruth Pell")
    assert whole_name("Brother Smyth", titles=frozenset())
    assert not whole_name("Hrafn Gothi", titles=frozenset({"gothi"}))


def test_a_new_thing_is_not_folded_by_its_label_words(monkeypatch) -> None:
    """v5 §17 retired `same_label`'s word-set fold. It ran on, and folded once in
    all recorded play: the Crossing's laden donkey, into "the man with the
    laden donkey" -- "man" is an English stopword. A new thing is minted;
    only a declared name folds (v5 §17)."""
    canon = _crossing()
    canon.entities["donkeyman"] = Entity("donkeyman", ["the man with the laden donkey"], 0, "smrti")
    recs = [_rec(1, "teo", "east", 0), _rec(2, "donkeyman", "east", 0)]
    mems = [_mem(1, "teo", 5, "I watched the man with the laden donkey.", "east"),
            _mem(2, "donkeyman", 5, "I led my donkey down to the landing.", "east")]
    out = {"narration": "A laden donkey stamps.", "perceiver_tuples": [{"perceiver": "teo", "text": "I patted the donkey.", "salience": "ordinary"}],
           "new_entities": [{"id": "NEW:laden_donkey", "in": "east"}],
           "scene_tuples": [{"scene": "east", "text": "A laden donkey stamped on the landing.", "present": ["teo", "NEW:laden_donkey"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "teo", "pats the donkey", 6, mems, out, records=recs)
    assert not [e for e, _ in events if e == "smrti.label_fold"]
    assert any(e.labels[0] != "the man with the laden donkey" for e in r.new_entities)  # minted, not folded onto the man


def test_a_namesake_is_one_who_perceives_not_one_a_tag_calls_a_person(monkeypatch) -> None:
    """v5 §17: no type may tell them apart. Namesakes took, beside perceivers,
    whatever svabhava called an actor -- in the live worlds a wallet, a
    derringer and "my horse" among ten. A name another PERCEIVER bears is
    shown; a tag makes no one a person."""
    canon = _crossing()
    canon.entities["purse"] = Entity("purse", ["a purse of Varr silver"], 0, "genesis")
    canon.entities["old"] = Entity("old", ["an old smith"], 0, "genesis")
    recs = [_rec(1, "teo", "east", 0), _rec(2, "purse", "west", 0), _rec(3, "old", "west", 0),
            Record("n1", "name", "purse", "Varr Purse", 0, "p0", None), Record("n2", "name", "old", "Old Varr", 0, "p0", None)]
    mems = [_mem(1, "teo", 5, "I stood on the east landing.", "east"), _mem(2, "old", 5, "I hammered at my anvil.", "west")]
    out = {"narration": "x", "perceiver_tuples": [{"perceiver": "teo", "text": "I asked after Varr.", "salience": "ordinary"}],
           "scene_tuples": [{"scene": "east", "text": "The boy asked after Varr.", "present": ["teo"]}], "suggested_next_action": ""}
    _, _, client = _pulse(monkeypatch, canon, "teo", "asks after Varr", 6, mems, out, records=recs)
    shown = client.calls[0][1]
    assert '"Old Varr"' in shown and '"Varr Purse"' not in shown


def test_what_is_present_is_shown_by_what_holds_it() -> None:
    """The sample world (2026-09-30): PRESENT promised "by what they are in"
    and listed everyone flat -- the rope in the trader's hands, the pilgrim in
    the skiff, each looked like it lay in the scene. So the Narrator "left"
    them by not moving them, and never gave the move a let-go needs; Stonemarch's
    carried gear was listed behind in the room it left 248 times. Replayed
    with holders shown: set-downs 0/3 -> 3/3, 0/3 -> 2/3, 0/3 -> 3/3, and no
    gear left behind. What is present is shown under what holds it, and what
    the actor holds is named as the actor's."""
    from smritilila.smrti.narrator import render
    canon = Maya(world_time=20, entities={
        "dock": Entity("dock", ["the landing"], 0, "genesis", avarana="scene"),
        "ansel": Entity("ansel", ["Ansel Dray"], 0, "genesis"), "rope": Entity("rope", ["a coil of rope"], 0, "genesis"),
        "crate": Entity("crate", ["a crate of salt"], 0, "genesis"), "skiff": Entity("skiff", ["the Grey Tern"], 0, "genesis"),
        "elspeth": Entity("elspeth", ["Elspeth Thorne"], 0, "genesis"), "keir": Entity("keir", ["Keir Vane"], 0, "genesis"),
        "tender": Entity("tender", ["a shrine-tender"], 0, "genesis")})
    recs = [_rec(1, "ansel", "dock", 0), _rec(2, "rope", "ansel", 0), _rec(3, "crate", "ansel", 0), _rec(4, "skiff", "dock", 0),
            _rec(5, "elspeth", "skiff", 0), _rec(6, "keir", "skiff", 0), _rec(7, "tender", "dock", 0)]
    mems = [_mem(i, who, 10, f"I was at the landing ({who}).", "dock") for i, who in enumerate(
        ["ansel", "rope", "skiff", "elspeth", "keir", "tender"], start=1)]
    log, _ = build_log(canon, mems, recs)
    ctx = retrieve(log, "ansel", 20, None, k_place=6, k_sim=0, action="I leave the rope on the bollard.")
    line = next(l for l in render(ctx, "I leave the rope on the bollard.", {"dock"}).split("\n") if l.startswith("PRESENT (in the scene"))
    listed = line.split(": ", 1)[1]
    assert "skiff (holding: elspeth, keir)" in listed
    assert listed.endswith("held by the actor, ansel: crate, rope")
    roots = listed.split("; held by the actor")[0].split("; ")
    assert sorted(r.split(" (")[0] for r in roots) == ["skiff", "tender"]     # the pilgrim and the rope are not loose in the scene


def _isles():
    canon = Maya(world_time=10, entities={
        "sea": Entity("sea", ["the channels"], 0, "genesis"), "lane": Entity("lane", ["the crossing lane"], 0, "genesis", avarana="scene"),
        "isle": Entity("isle", ["Thessel"], 0, "genesis"), "landing": Entity("landing", ["the pilgrim landing"], 0, "genesis", avarana="scene"),
        "folds": Entity("folds", ["the sheepfolds"], 0, "genesis", avarana="scene"), "bare": Entity("bare", ["a bare rock"], 0, "genesis"),
        "teo": Entity("teo", ["Teo"], 0, "genesis")})
    recs = [_rec(1, "lane", "sea", 0), _rec(2, "landing", "isle", 0), _rec(3, "folds", "isle", 0), _rec(4, "teo", "lane", 0)]
    mems = [_mem(1, "teo", 5, "I sat in the boat on the lane.", "lane"), _mem(2, "landing", 0, "A stone landing below the Head.", "landing"),
            _mem(3, "folds", 0, "Sheepfolds on the high ground.", "folds"), _mem(4, "lane", 0, "A lane of water between the isles.", "lane")]
    return canon, mems, recs


def _landing_out(region):
    return {"narration": "You come ashore.",
            "perceiver_tuples": [{"perceiver": "teo", "text": "I stepped ashore on a stone landing.", "salience": "ordinary"}],
            "new_entities": [{"id": "NEW:head_landing", "name": "the landing at the Head", "in": region, "scene": True}],
            "moves": [{"entity": "teo", "into": "NEW:head_landing"}],
            "scene_tuples": [{"scene": "NEW:head_landing", "text": "A boy came ashore.", "present": ["teo"]}],
            "suggested_next_action": ""}


def test_a_place_made_beyond_the_actors_branch_is_asked_again_shown_what_is_there(monkeypatch) -> None:
    """The sample world (2026-09-30): arriving on another island, the Narrator
    knew the island but none of its places -- the tree shows only the actor's
    own branch -- and minted a second landing beside the first, three
    arrivals of three. A scene made in a region off the actor's branch that
    already holds scenes is asked once more, shown them; the first pass's
    mint leaves the canon. Replayed on the kept prompts: 8 of 9 found the
    existing place, and places that did not exist were still made."""
    canon, mems, recs = _isles()
    r, events, client = _pulse(monkeypatch, canon, "teo", "I land at the pilgrim landing", 6, mems, _landing_out("isle"), records=recs)
    assert len(client.calls) == 2
    assert "PLACES ALREADY IN isle" not in client.calls[0][1] and "PLACES ALREADY IN isle" in client.calls[1][1]
    assert 'landing "the pilgrim landing"' in client.calls[1][1] and 'folds "the sheepfolds"' in client.calls[1][1]
    assert any(e == "smrti.place_rerun" and f["regions"] == ["isle"] for e, f in events)
    assert sum(1 for e in canon.entities.values() if e.labels == ["the landing at the Head"]) == 1   # the first pass's mint is gone


def test_a_place_made_where_the_tree_shows_or_where_nothing_is_is_not_asked_again(monkeypatch) -> None:
    canon, mems, recs = _isles()
    _r, _e, client = _pulse(monkeypatch, canon, "teo", "I tie up in a new cove off the lane", 6, mems, _landing_out("sea"), records=recs)
    assert len(client.calls) == 1   # the actor's own region: its places were shown in the tree
    canon, mems, recs = _isles()
    _r, _e, client = _pulse(monkeypatch, canon, "teo", "I land on the bare rock", 6, mems, _landing_out("bare"), records=recs)
    assert len(client.calls) == 1   # a region holding no scene: nothing to have missed


def test_a_rerun_for_something_else_still_shows_the_places_it_missed(monkeypatch) -> None:
    """Stonemarch t29, replayed under today's code (2026-10-04): a telling that gave a player an intention AND made a
    place beside places it was not shown spent its one rerun on the intention, and the second telling minted a
    second moathouse approach beside the seed's. One rerun carries everything the first telling got wrong: the
    places it missed go with it, whichever rerun fires."""
    canon, mems, recs = _isles()
    canon.entities["mara"] = Entity("mara", ["Mara"], 0, "genesis")
    recs.append(_rec(5, "mara", "lane", 0))
    mems.append(_mem(5, "mara", 5, "I rowed beside the boy.", "lane"))
    out = _landing_out("isle")
    out["sankalpa"] = [{"who": "mara", "intends": "wave to the boy"}]  # a player's deeds are their own (rule 10)
    r, events, client = _pulse(monkeypatch, canon, "teo", "I land at the pilgrim landing", 6, mems, out, records=recs,
                               players=frozenset({"teo", "mara"}), playable=frozenset({"teo", "mara"}))
    assert len(client.calls) == 2
    rerun = next(f for e, f in events if e == "smrti.sankalpa_rerun")
    assert rerun["beyond"] == ["isle"] and not any(e == "smrti.place_rerun" for e, _ in events)
    assert "PLACES ALREADY IN isle" in client.calls[1][1] and 'landing "the pilgrim landing"' in client.calls[1][1]


def test_a_new_house_around_a_new_room_asks_again_of_the_region_that_was_there(monkeypatch) -> None:
    """Live, the same evening: the Narrator made "Ansgar House" and a counting
    parlour in it, on the island where the parlour already stood. The region
    asked about is the first one that was there before the pulse."""
    canon, mems, recs = _isles()
    out = _landing_out("NEW:head_house")
    out["new_entities"] = [{"id": "NEW:head_house", "name": "the Head House", "in": "isle", "scene": False}] + out["new_entities"]
    _r, events, client = _pulse(monkeypatch, canon, "teo", "I go up to the house at the Head", 6, mems, out, records=recs)
    assert len(client.calls) == 2 and "PLACES ALREADY IN isle" in client.calls[1][1]
    assert any(e == "smrti.place_rerun" and f["regions"] == ["isle"] for e, f in events)


# -- saṅkalpa (v5 §28, Marc 2026-10-01) ---------------------------------------


def test_who_may_act_is_who_is_here_or_across_and_lately_active_never_a_player_or_a_carried_thing() -> None:
    from smritilila.smrti.context import Context
    from smritilila.smrti.pulse import who_may_act
    canon, mems, recs = _landing()
    canon.entities["boat"] = Entity("boat", ["the boatman"], 0, "genesis")
    canon.entities["shanty"] = Entity("shanty", ["the tarred shanty"], 0, "genesis", avarana="scene")
    canon.entities["door"] = Entity("door", ["the shanty door"], 0, "genesis")
    recs += [_rec(30, "shanty", "landing", 0), _rec(31, "boat", "shanty", 0), _rec(32, "door", "landing", 0)]
    mems += [_mem(20, "boat", 9, "I watched from my doorway.", "shanty"), _mem(21, "door", 2, "I hung on my strap.", "landing")]
    log, _places = build_log(canon, mems, recs)
    ctx = Context(actor="nat", place="landing", tuples=[], present={"nat", "bren", "watch", "gun", "cloth", "car", "wen", "door"})
    players = frozenset({"wen"})
    may = who_may_act(log, ctx, "nat", 10, "Look along the river.", players)
    # the watch and gun are in his hand, and wen is a player: no part. The cloth in Bren's coat has one -- only the actor,
    # a player or someone the action names is a holder (mauna); structure cannot tell a man holding a cloth from a cart
    # holding a man, and a man in a cart must be free to act
    assert set(may) == {"bren", "car", "door", "cloth"}
    assert "cloth" not in who_may_act(log, ctx, "nat", 10, "Ask Bren what he carries.", players)  # named: his things are his
    # across an open boundary: the boatman in his shanty, once the shanty door is in both (sandhi)
    assert "boat" not in may
    log.kosha.add(Record("s1", "sandhi", "door", "shanty", 3, "p3", "open"))
    assert "boat" in who_may_act(log, ctx, "nat", 10, "Look along the river.", players)
    assert "boat" not in who_may_act(log, ctx, "nat", 10, "x", players, out_of_reach=frozenset({"boat"}))  # lived later
    # long quiet and not named: no part in this moment; named, a part
    assert "door" not in who_may_act(log, ctx, "nat", 40, "Look along the river.", players)
    assert "door" in who_may_act(log, ctx, "nat", 40, "Knock on the shanty door.", players)


def test_this_moment_is_in_an_order_no_one_chose_and_the_same_for_a_rerun() -> None:
    from smritilila.smrti.narrator import moment_lines
    entries = [("dunstan", "I hold the line.", "grim"), ("jakal", "I cast.", "adverse"), ("ogre", None, "providential"),
               ("calmert", None, "ruinous")]
    a = moment_lines(entries, "corridor|367")
    assert a == moment_lines(list(reversed(entries)), "corridor|367")    # the order handed in, or pressed, means nothing
    assert a[0].startswith("THIS MOMENT, IN NO ORDER OF PRECEDENCE")
    assert "- ogre -- may act (their saṅkalpa) -- FORTUNE, IF IT HOLDS ANYTHING IN DOUBT: providential" in a
    orders = {tuple(l.split(" -- ")[0] for l in moment_lines(entries, f"corridor|{t}")[1:]) for t in range(30)}
    assert len(orders) > 5                                                # the moment draws different orders


def test_an_intention_for_a_player_is_told_again_and_then_dropped(monkeypatch) -> None:
    """v5 §28: only those listed may intend; a player's deeds are their own (rule 10)."""
    canon, mems, recs = _landing()
    out = {"narration": "You look along the river.", "sankalpa": [{"who": "bren", "intends": "Watch the water."},
                                                                  {"who": "wen", "intends": "Climb out of the car."}],
           "perceiver_tuples": [{"perceiver": "nat", "text": "I looked along the river at the mist.", "salience": "ordinary"}],
           "scene_tuples": [], "suggested_next_action": ""}
    r, events, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 10, mems, out, records=recs,
                               players=frozenset({"wen"}), playable=frozenset({"nat", "wen"}))
    assert [f["who"] for e, f in events if e == "smrti.sankalpa_rerun"] == [["wen"]]
    assert len(client.calls) == 2 and "TOLD AGAIN: these may not intend anything this moment -- wen (a player" in client.calls[1][1]
    assert [f["who"] for e, f in events if e == "smrti.sankalpa_refused"] == [["wen"]]   # still there: dropped, counted
    kept = next(f for e, f in events if e == "smrti.sankalpa")
    assert kept["intents"] == [("bren", "Watch the water.")]


def _sankalpa_out(who: str, intends: str) -> dict:
    return {"narration": "You look along the river.", "sankalpa": [{"who": "bren", "intends": "Watch the water."}, {"who": who, "intends": intends}],
            "perceiver_tuples": [{"perceiver": "nat", "text": "I looked along the river at the mist.", "salience": "ordinary"}],
            "scene_tuples": [], "suggested_next_action": ""}


def test_the_rule_is_shown_on_the_first_pass_to_every_player_who_has_given_an_action(monkeypatch) -> None:
    """v5.25 (Marc, 2026-10-08). 15% of recent Narrator calls were discarded and rerun, 62% of those for `sankalpa_rerun`; all 76 recorded
    cases flagged the actor or a player acting in the same moment. Replayed on the first prompts: 124 of 266 answers gave a player an
    intention; with this line said up front, 0 of 269, and the intentions for those who may act unchanged."""
    canon, mems, recs = _landing()
    out = _sankalpa_out("nat", "Look along the river.")
    _r, _events, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 10, mems, out, records=recs,
                                 players=frozenset({"nat"}), playable=frozenset({"nat", "wen"}))
    first = client.calls[0][1]
    assert "NOT TO INTEND: nat (a player with an ACTION above: their deeds are their own)" in first
    assert "wen" not in first.split("NOT TO INTEND:")[1].split("\n")[0]       # a player present and not acting is not named: they have no action
    assert "TOLD AGAIN" not in first and "Only those listed as able to act (may act) form an intention this moment." in first
    # counter-test: an actor who is not a player has no such line (their own turn, or an NPC's action)
    canon, mems, recs = _landing()
    _r, _e, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 10, mems, _sankalpa_out("bren", "Watch."), records=recs,
                            players=frozenset({"wen"}), playable=frozenset({"wen"}))
    assert "NOT TO INTEND" not in client.calls[0][1]


def test_a_player_who_has_acted_and_is_written_an_intention_is_not_told_again(monkeypatch) -> None:
    """v5.25: the entry restates their own action. It is dropped and counted, with no second call; a rerun mended nothing (replayed, the
    narration gave the player something substantial in 3% of slipped answers and 5% of clean ones) and cost a whole call, 13 s."""
    canon, mems, recs = _landing()
    out = _sankalpa_out("nat", "Look along the river.")      # the actor, a player, written an intention for the action they gave
    r, events, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 10, mems, out, records=recs,
                               players=frozenset({"nat"}), playable=frozenset({"nat", "wen"}))
    assert len(client.calls) == 1 and not any(e == "smrti.sankalpa_rerun" for e, _ in events)
    refused = next(f for e, f in events if e == "smrti.sankalpa_refused")
    assert refused["who"] == ["nat"] and refused["acting"] == ["nat"]                # dropped, counted, and said to be one who acted
    assert next(f for e, f in events if e == "smrti.sankalpa")["intents"] == [("bren", "Watch the water.")]   # only the one who may act
    # counter-test: a player present who has NOT acted is still told again (rule 10; the existing rerun)
    canon, mems, recs = _landing()
    out = _sankalpa_out("wen", "Climb out of the car.")
    r, events, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 10, mems, out, records=recs,
                               players=frozenset({"nat"}), playable=frozenset({"nat", "wen"}))
    assert len(client.calls) == 2 and [f["who"] for e, f in events if e == "smrti.sankalpa_rerun"] == [["wen"]]


def test_a_player_alone_who_is_written_an_intention_is_not_told_again(monkeypatch) -> None:
    """v5.25 for the pulse with no moment block (Stonemarch, 2026-10-13, Jakal at 1522): nobody else could act, so there was no THIS MOMENT
    and the actor was not among those who had acted; the Narrator wrote an intention for the actor, it counted as a player not acting, and
    the pulse was told again with its first telling thrown away (the oak it had made went with it). 8 of the 9 saṅkalpa reruns since
    10-08 were the actor alone. Every earlier test had Bren present and able to act, so a moment always existed."""
    canon, mems, recs = _landing()
    out = {"narration": "You look along the river.", "sankalpa": [{"who": "nat", "intends": "Look along the river."}],
           "perceiver_tuples": [{"perceiver": "nat", "text": "I looked along the river at the mist.", "salience": "ordinary"}],
           "scene_tuples": [], "suggested_next_action": ""}
    # tick 40: Bren's last tuple (9) is long past the participation window, so no one else may act and the moment is empty
    r, events, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 40, mems, out, records=recs,
                               players=frozenset({"nat"}), playable=frozenset({"nat", "wen"}))
    assert "THIS MOMENT" not in client.calls[0][1]                                    # the shape under test: no moment block
    assert len(client.calls) == 1 and not any(e == "smrti.sankalpa_rerun" for e, _ in events)
    refused = next(f for e, f in events if e == "smrti.sankalpa_refused")
    assert refused["who"] == ["nat"] and refused["acting"] == ["nat"]                 # dropped, counted, and said to be one who acted
    # counter-test: a player present who has NOT acted is still told again, moment block or none (rule 10)
    canon, mems, recs = _landing()
    out["sankalpa"] = [{"who": "wen", "intends": "Climb out of the car."}]
    r, events, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 40, mems, out, records=recs,
                               players=frozenset({"nat"}), playable=frozenset({"nat", "wen"}))
    assert len(client.calls) == 2 and [f["who"] for e, f in events if e == "smrti.sankalpa_rerun"] == [["wen"]]


def test_an_intention_for_someone_else_off_the_list_is_dropped_and_counted_without_a_rerun(monkeypatch) -> None:
    """v5 §28 (Marc, 2026-10-01: "rerun only for players"): at the Welcome Wench the telling gave intentions to the
    lived-later, and a rerun never mended it -- so anyone off the list who is not a player costs no second call."""
    canon, mems, recs = _landing()
    out = {"narration": "You look along the river.", "sankalpa": [{"who": "bren", "intends": "Watch the water."},
                                                                  {"who": "ghost", "intends": "Drift by."}],
           "perceiver_tuples": [{"perceiver": "nat", "text": "I looked along the river at the mist.", "salience": "ordinary"}],
           "scene_tuples": [], "suggested_next_action": ""}
    r, events, client = _pulse(monkeypatch, canon, "nat", "Look along the river.", 10, mems, out, records=recs,
                               players=frozenset({"wen"}), playable=frozenset({"nat", "wen"}))
    assert len(client.calls) == 1 and not any(e == "smrti.sankalpa_rerun" for e, _ in events)
    assert [f["who"] for e, f in events if e == "smrti.sankalpa_refused"] == [["ghost"]]


def test_anusmrti_recalls_a_being_as_its_own_turn_would_and_nothing_of_anyone_else() -> None:
    """Marc, 2026-10-02: an agent player read only Dunstan's last 8 memories, so the
    wounds the world remembered him by were out of its sight. Anusmṛti is his own
    stream as the world's retrieval selects it for his own turn: the old marking
    wound comes back beside the latest, and no one else's tuple, nor the scene's."""
    from smritilila.smrti.pulse import anusmrti
    from smritilila.smrti.samskara import Samskara
    canon = _crossing()
    mems = [Memory("m0", "teo", frozenset(), "The rope burned through my palm and laid it open to the bone.", "PRIMARY", 2, "east", 1, "p1", None, "key")]
    mems += [_mem(i, "teo", i, f"I coiled rope on the landing, morning {i}.", "east") for i in range(2, 20)]
    mems += [_mem(30, "ilo", 19, "I watched the boy coil rope and said nothing.", "east"),
             _mem(31, "east", 19, "The boy coiled rope while the old man watched.", "east")]
    sk = Samskara.build(canon, mems, [], players=["teo"], clock=None, defaults=None)
    rows = anusmrti(sk, canon, "teo", 20, None, players=["teo"])
    texts = [t.memory for t, _ in rows]
    assert any(x.startswith("The rope burned through my palm") for x in texts)  # 18 later tuples did not bury it
    assert "I coiled rope on the landing, morning 19." in texts        # the latest is there too
    assert all(t.perceiver == "teo" for t, _ in rows)                  # nothing of Ilo's, nor the scene's
    assert len(texts) < 19 and all(why for _, why in rows)             # a selection, each with its rule
    assert [t.memory for t, _ in anusmrti(sk, canon, "trav", 20, None)] == ["a woman traveler"]  # nothing lived: who she is


def test_a_chain_held_one_inside_the_next_is_read_in_order_and_is_otherwise_quiet() -> None:
    """v4.60 (Marc, 2026-10-02: "a page at a time ... each page nested within the container of the book"): leaves held
    one inside the next are read in order -- named, the words of every leaf not yet read are shown; a leaf a pulse of
    the reader's wrote to is read; reading on, the rest; otherwise the leaves are quiet, their words in no prompt."""
    from smritilila.smrti.context import read_chains
    from smritilila.smrti.pulse import _context_for
    from smritilila.smrti.samskara import Samskara
    canon = _crossing()
    canon.entities["satchel"] = Entity("satchel", ["a waxed satchel"], 0, "genesis")
    for i in (1, 2, 3):
        canon.entities[f"leaf{i}"] = Entity(f"leaf{i}", [f"leaf {i} of the satchel letter"], 0, "genesis")
    words = {1: "Dear Teo, the river is rising.", 2: "Cross only at the middle stone.", 3: "Your father, Ilo."}
    mems = [_mem(1, "teo", 5, "I sat on the landing with the satchel.", "east")]
    mems += [Memory(f"w{i}", f"leaf{i}", frozenset(), words[i], "PRIMARY", 3, "nowhere", 0, "p0", None, "none") for i in (1, 2, 3)]
    recs = [_rec(1, "satchel", "teo", 1), _rec(2, "leaf1", "satchel", 1), _rec(3, "leaf2", "leaf1", 1), _rec(4, "leaf3", "leaf2", 1)]

    def prompt_words(memories, action, now):
        sk = Samskara.build(canon, memories, recs, players=["teo"], clock=None, defaults=None)
        ctx = _context_for(canon, sk.log_for(canon), "teo", action, now, None, keep=frozenset({"teo"}))
        text = "\n".join(t.memory for t in ctx.tuples)
        return [i for i in (1, 2, 3) if words[i] in text], ctx.reading

    assert prompt_words(mems, "I mend my net.", 6) == ([], [])                       # quiet: no leaf speaks
    shown, reading = prompt_words(mems, "I open the waxed satchel and read the letter.", 6)
    assert shown == [1, 2, 3] and reading == [("satchel", [], ["leaf1", "leaf2", "leaf3"])]
    # the first leaf read: a tuple of it on the reader's tick
    read1 = mems + [Memory("r1", "leaf1", frozenset({"teo"}), "The boy read me.", "PRIMARY", 5, "east", 6, "p6", None),
                    Memory("r2", "teo", frozenset(), "I read the first leaf of the letter.", "PRIMARY", 4, "east", 6, "p6", None)]
    shown, reading = prompt_words(read1, "I turn the leaf.", 7)                         # reading on: what is left
    assert shown == [2, 3] and reading == [("satchel", ["leaf1"], ["leaf2", "leaf3"])]
    assert read_chains(Samskara.build(canon, mems, recs, players=["teo"], clock=None, defaults=None).log_for(canon),
                       "ilo", 6, "I read the letter.", set(), []) == []                                       # not held, not named: nothing


def test_a_chain_is_walked_however_deep_it_runs() -> None:
    """2026-10-03: CHAIN_DEPTH = 16 called a deeper chain a loop, but v4.60's chains
    nest one inside the next -- Stonemarch's book, page twelve, was 17 deep and read its
    root as a county, short of the earth. `inside`, capped at 32, could have let a long
    chain's cycle through. A walk never revisits what it passed, so it needs no cap."""
    ents = {"earth": Entity("earth", ["the world at large"], 0, "itihasa"),
            "county": Entity("county", ["the county"], 0, "genesis")}
    recs = [Record("r0", "kosha", "county", "earth", 0, "p0")]
    prev = "county"
    for i in range(40):  # forty pages, each inside the one before
        ents[f"p{i}"] = Entity(f"p{i}", [f"page {i}"], 0, "genesis")
        recs.append(Record(f"r{i + 1}", "kosha", f"p{i}", prev, 0, "p0"))
        prev = f"p{i}"
    k = Kosha(Maya(world_time=1, entities=ents), (), recs)
    assert k.root("p39", 1) == "earth"
    assert k.inside("p39", "county", 1)  # moving the county into the last page would be a cycle


def test_the_earth_is_never_moved(monkeypatch) -> None:
    """Marc, 2026-10-03: "make the earth unmovable". The earth holds all that can come from it and nothing holds it,
    so the root and the earth never part: a realm minted around the county is refused the earth, never given it."""
    canon = _crossing()
    canon.entities["earth"] = Entity("earth", ["the world at large"], 0, "itihasa")
    canon.entities["county"] = Entity("county", ["the river county"], 0, "genesis")
    recs = [_rec(90, "county", "earth", 0), _rec(91, "east", "county", 0), _rec(92, "west", "county", 0), _rec(93, "teo", "east", 0)]
    mems = [_mem(1, "teo", 5, "I stood on the east landing.", "east")]
    out = {"narration": "x",
           "perceiver_tuples": [{"perceiver": "teo", "text": "I thought of the old kingdom.", "salience": "ordinary"}],
           "new_entities": [{"id": "NEW:the_old_kingdom", "name": "the old kingdom", "in": "", "scene": False}],
           "moves": [{"entity": "earth", "into": "NEW:the_old_kingdom"}],
           "scene_tuples": [{"scene": "east", "text": "A man stood thinking.", "present": ["teo"]}],
           "suggested_next_action": ""}
    r, events, _ = _pulse(monkeypatch, canon, "teo", "thinks of the old kingdom", 6, mems, out, records=recs)
    refused = next(f["refused"] for e, f in events if e == "smrti.move_refused")
    assert [(x["entity"], x["reason"]) for x in refused] == [("earth", "earth")]
    k = Kosha(canon, mems + r.memories_written, recs + r.records_written)
    kingdom = r.resolved_new["NEW:the_old_kingdom"]
    assert k.container_at("earth", 7) is None and k.root("teo", 7) == "earth" and k.container_at(kingdom, 7) == "earth"


def test_who_may_act_is_held_to_the_worlds_own_participation_window_not_a_fixed_twenty() -> None:
    """Marc, 2026-10-10: the participation window of a world drives who may act, as it drives who stays quiet. The same action at tick 40, the door last active at tick 2 (38 ticks ago):
    outside the default window of 20, inside a world's own window of 50 -- and a window of 5 shuts out someone who was inside 20."""
    from smritilila.smrti.context import Context
    from smritilila.smrti.pulse import who_may_act
    canon, mems, recs = _landing()
    canon.entities["door"] = Entity("door", ["the shanty door"], 0, "genesis")
    recs += [_rec(32, "door", "landing", 0)]
    mems += [_mem(40, "door", 2, "I hung on my strap.", "landing"), _mem(41, "bren", 30, "I stood by the rail.", "landing")]   # (id, who, the tick it was written)
    log, _places = build_log(canon, mems, recs)
    ctx = Context(actor="nat", place="landing", tuples=[], present={"nat", "bren", "door"})
    players = frozenset()
    assert "door" not in who_may_act(log, ctx, "nat", 40, "Look along the river.", players)                               # 38 ticks: past the default 20
    assert "door" in who_may_act(log, ctx, "nat", 40, "Look along the river.", players, pacing={"participation_window": 50})   # the world's window is wider
    assert "bren" in who_may_act(log, ctx, "nat", 40, "Look along the river.", players)                                     # 10 ticks: inside the default
    assert "bren" not in who_may_act(log, ctx, "nat", 40, "Look along the river.", players, pacing={"participation_window": 5})   # the world's window is narrower
    assert "door" not in who_may_act(log, ctx, "nat", 40, "Look along the river.", players, pacing={"participation_window": None})   # null is the default again


def test_the_narrators_schema_says_an_empty_into_means_in_the_earth_as_the_engine_does() -> None:
    from smritilila.smrti.narrator import SCHEMA
    into = SCHEMA["properties"]["moves"]["items"]["properties"]["into"]["description"]
    assert "in the earth" in into and "in nothing" not in into


def test_a_perceiver_with_more_than_two_tuples_in_one_pulse_is_counted_never_dropped() -> None:
    """v5 §3: one tuple, and at most one more for a moment that stands alone. The engine counts what breaks it (smrti.moment_over) and writes everything as before."""
    from smritilila.smrti.pulse import over_one_moment
    t = lambda who, text="x": {"perceiver": who, "text": text}  # noqa: E731
    assert over_one_moment([]) == {}
    assert over_one_moment([t("nat"), t("bren")]) == {}                                   # one each
    assert over_one_moment([t("nat"), t("nat"), t("bren")]) == {}                         # two: the moment that stands alone is allowed
    assert over_one_moment([t("nat"), t("nat"), t("nat"), t("bren")]) == {"nat": 3}       # a third is counted
    assert over_one_moment([t("nat"), t("nat"), t("nat", "")]) == {}                      # an empty tuple is not a tuple
    assert over_one_moment([t(" nat "), t("nat"), t("nat")]) == {"nat": 3}                # as written, trimmed
    assert over_one_moment([{"text": "x"}, {"text": "x"}, {"text": "x"}]) == {}           # no perceiver named: nothing to count


def _two_levels(teo_in_the_city_once=True):
    """A world with its top, a county, a walled city inside the county holding two scenes, and a yard filed in the top beside the county: Teo stood at the gate, then came to the yard."""
    canon = Maya(world_time=10, entities={
        "earth": Entity("earth", ["the world at large"], 0, "genesis"), "county": Entity("county", ["the county"], 0, "genesis"),
        "city": Entity("city", ["the walled city"], 0, "genesis"), "gate": Entity("gate", ["the south gate"], 0, "genesis", avarana="scene"),
        "hall": Entity("hall", ["the great hall"], 0, "genesis", avarana="scene"), "yard": Entity("yard", ["the hospice yard"], 0, "genesis", avarana="scene"),
        "teo": Entity("teo", ["Teo"], 0, "genesis")})
    recs = [_rec(1, "county", "earth", 0), _rec(2, "city", "county", 0), _rec(3, "gate", "city", 0), _rec(4, "hall", "city", 0), _rec(5, "yard", "earth", 0), _rec(6, "teo", "yard", 0)]
    mems = [_mem(3, "gate", 0, "A gate in the wall.", "gate"), _mem(4, "hall", 0, "A great hall.", "hall"), _mem(5, "yard", 0, "A yard.", "yard"),
            _mem(6, "earth", 0, "The world at large.", "earth"), _mem(8, "teo", 8, "I came to the hospice yard.", "yard")]
    if teo_in_the_city_once:
        mems.append(_mem(1, "teo", 3, "I stood at the south gate.", "gate"))
    return canon, mems, recs


def test_the_region_the_actor_last_stood_in_is_found_when_the_tree_shows_them_none() -> None:
    """Stonemarch, 2026-10-10 (Filonius): the city was one level down in the county, the hospice he reached stood in the world at large, and the tree listed only the world at large."""
    from smritilila.smrti.context import last_region_stood
    canon, mems, recs = _two_levels()
    log, _ = build_log(canon, mems, recs)
    k = log.kosha

    def tree_of(place):
        return [(h, k.label(h), [(x, k.label(x)) for x in xs]) for h, xs in k.tree(place, 10)]
    region, name, ago, rows = last_region_stood(log, k, "teo", "yard", 10, tree_of("yard"))
    assert (region, name, ago) == ("city", "the walled city", 7)                              # he stood at the gate at 3; now is 10
    assert {r[0] for r in rows} == {"gate", "hall"} and all(r[2] is None for r in rows)       # the city's scenes, by id and name
    # not asked when the tree already shows the region (he is inside the city), when he never stood in one, or when there is nothing to compare with
    assert last_region_stood(log, k, "teo", "gate", 10, tree_of("gate")) is None
    # in one scene of the city having stood in another: the tree lists the city, so it is not "the region they last stood in" -- the guard that alone stops that
    recs3 = [r for r in recs if r.entity_id != "teo"] + [_rec(7, "teo", "hall", 0)]
    log3, _ = build_log(canon, mems, recs3)
    k3 = log3.kosha
    tree3 = [(h, k3.label(h), [(x, k3.label(x)) for x in xs]) for h, xs in k3.tree("hall", 10)]
    assert "city" in {h for h, _n, _p in tree3} and last_region_stood(log3, k3, "teo", "hall", 10, tree3) is None
    canon2, mems2, recs2 = _two_levels(teo_in_the_city_once=False)
    log2, _ = build_log(canon2, mems2, recs2)
    assert last_region_stood(log2, log2.kosha, "teo", "yard", 10, [(h, "", []) for h, _x in log2.kosha.tree("yard", 10)]) is None
    assert last_region_stood(log, k, "teo", None, 10, tree_of("yard")) is None and last_region_stood(log, k, "teo", "yard", 10, []) is None


def test_the_last_region_reaches_the_narrators_prompt_only_when_it_is_missing_from_the_tree(monkeypatch) -> None:
    canon, mems, recs = _two_levels()
    out = {"narration": "You walk on.", "perceiver_tuples": [{"perceiver": "teo", "text": "I walked on.", "salience": "ordinary"}],
           "scene_tuples": [{"scene": "yard", "text": "A yard.", "present": ["teo"]}], "new_entities": [], "moves": [], "suggested_next_action": ""}
    r, events, client = _pulse(monkeypatch, canon, "teo", "I walk on", 9, mems, out, records=recs)
    prompt = client.calls[0][1]
    assert 'PLACES ALREADY IN city "the walled city" (the region the actor last stood in, 6 moments ago; the actor now stands outside it' in prompt
    assert 'gate "the south gate"' in prompt and 'hall "the great hall"' in prompt and "belongs inside the region" in prompt
    canon2, mems2, recs2 = _two_levels(teo_in_the_city_once=False)
    r2, events2, client2 = _pulse(monkeypatch, canon2, "teo", "I walk on", 9, mems2, out, records=recs2)
    assert "the region the actor last stood in" not in client2.calls[0][1]                     # he never stood in one: nothing is added


def test_a_player_lived_past_this_moment_is_not_listed_movable_and_is_told_where_they_were() -> None:
    """Stonemarch 1533-1573 (Marc, 2026-10-13): Dunstan, eight days further on than Jakal, was listed MOVABLE though every move of him is
    refused as apart in time; the Narrator moved him onto the church steps five times. He is no longer listed movable; the prompt says where
    he was at this moment -- the place only."""
    from smritilila.smrti.pulse import view_block
    from smritilila.smrti.context import Context
    canon = Maya(world_time=10, entities={"hall": Entity("hall", ["the hall"], 0, "genesis", avarana="scene"),
                                          "nat": Entity("nat", ["Nathaniel"], 0, "genesis"), "wal": Entity("wal", ["Walter"], 0, "genesis"),
                                          "dan": Entity("dan", ["Dunstan"], 0, "genesis")})
    k = Kosha(canon, [_mem(1, "wal", 6, "I came in.", "hall")], [_rec(1, "wal", "hall", 6)])
    ctx = Context(actor="wal", place="hall", tuples=[], present=set())
    players = frozenset({"wal", "dan", "nat"})
    text = view_block(k, ctx, 10, {}, frozenset({"dan", "nat"}), set(), players=players,
                      out_of_time=frozenset({"dan"}), lived_at={"dan": ("Dunstan Mere", "the keep bailey")})
    assert "MOVABLE (players another's turn may move): nat" in text and "dan" not in text.split("MOVABLE")[1].split("\n")[0]
    assert 'LIVED PAST THIS MOMENT' in text and 'dan "Dunstan Mere", at the keep bailey' in text
    plain = view_block(k, ctx, 10, {}, frozenset({"dan"}), set(), players=players)
    assert "MOVABLE (players another's turn may move): dan" in plain and "LIVED PAST" not in plain
