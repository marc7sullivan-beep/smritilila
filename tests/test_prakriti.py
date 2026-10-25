# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Prakṛti: the seed lies at the root and grows through the branches
(spec v4.53, v5 §15). Written before the backfill, as its counter-tests:
a branch that says nothing reads as the world does; a leaf carried out of
a branch stops standing under it; a becoming's capability reaches only its
bearer; the earth's view is shown once, where it always was."""
from __future__ import annotations

from smritilila import adhisthana
from smritilila import pravrtti
from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.record import Record
from smritilila.smrti import prakriti
from smritilila.smrti.kosha import Kosha
from smritilila.smrti.narrator import DAIVA_WORDS

from test_kosha import _crossing, _mem, _pulse, _rec

STONEMARCH_LIKE = (
    "Gods are real and answer prayers.",
    {"lycanthropy": "Bitten, they turn at the full moon."},
    {"capabilities": {"perceives": False, "reaches": False}, "clock": {"epoch": "576-03-01", "start": 480},
     "newcomer_clauses": {"level": "First level."}, "pacing": {"arrivals": True}, "phala": {"ask": True},
     "source": "the World of Stonemarch", "daiva": {"spread": [1, 2, 3, 4, 3, 2, 1]}},
)


def _world():
    """The crossing under an earth, with a marsh branch between: earth > marsh > east; west directly in the earth."""
    canon = _crossing()
    canon.entities["earth"] = Entity("earth", ["the world at large"], 0, "genesis")
    canon.entities["marsh"] = Entity("marsh", ["the gallows marsh"], 0, "genesis")
    canon.entities["rope"] = Entity("rope", ["a coil of rope"], 0, "genesis")
    recs = [_rec(1, "marsh", "earth", 0), _rec(2, "east", "marsh", 0), _rec(3, "west", "earth", 0),
            _rec(4, "rope", "east", 0), _rec(5, "teo", "east", 0), _rec(6, "ilo", "east", 0),
            _rec(8, "trav", "west", 0)]  # nothing is in nothing (v4.45)
    return canon, recs


# -- sowing and reaping -----------------------------------------------------


def test_the_seed_sown_whole_is_reaped_as_it_was_given() -> None:
    sown = prakriti.sow("earth", prakriti.EMPTY, STONEMARCH_LIKE, 0, "p0")
    assert all(r.entity_id == "earth" for r in sown)
    assert prakriti.sown(sown, "earth") and not prakriti.sown(sown, "elsewhere")
    assert prakriti.reap(sown, "earth") == STONEMARCH_LIKE


def test_a_god_changing_the_seed_appends_only_what_changed() -> None:
    sown = prakriti.sow("earth", prakriti.EMPTY, STONEMARCH_LIKE, 0, "p0")
    d, b, w = STONEMARCH_LIKE
    changed = (d, b, {**{k: v for k, v in w.items() if k != "phala"}, "source": "Stonemarch, the 1983 boxed set"})
    more = prakriti.sow("earth", STONEMARCH_LIKE, changed, 5, "p5")
    assert sorted((r.kind, r.note, r.value) for r in more) == [
        ("prakriti", "phala", None), ("prakriti", "source", '"Stonemarch, the 1983 boxed set"')]
    assert prakriti.reap(sown + more, "earth") == changed
    assert prakriti.sow("earth", changed, changed, 6, "p6") == []          # nothing changed, nothing written


def test_a_world_with_no_view_sows_nothing() -> None:
    assert prakriti.sow("earth", prakriti.EMPTY, prakriti.EMPTY, 0, "p0") == []


# -- reading from the tree ----------------------------------------------------


def test_the_nearest_that_says_wins_and_a_branch_that_says_nothing_reads_as_the_world() -> None:
    canon, recs = _world()
    recs += prakriti.sow("earth", prakriti.EMPTY, STONEMARCH_LIKE, 0, "p0")
    k = Kosha(canon, [], recs)
    assert k.seed("daiva", "east", 1) == {"spread": [1, 2, 3, 4, 3, 2, 1]}   # the marsh says nothing: the earth's
    assert k.seed("daiva", "west", 1) == k.seed("daiva", "east", 1)
    k.add(Record("m1", "prakriti", "marsh", '{"spread": [7, 0, 0, 0, 0, 0, 0]}', 2, "p2", "daiva"))
    assert k.seed("daiva", "east", 3) == {"spread": [7, 0, 0, 0, 0, 0, 0]}   # the marsh's, for what it holds
    assert k.seed("daiva", "west", 3) == {"spread": [1, 2, 3, 4, 3, 2, 1]}   # and for nothing outside it
    assert k.seed("daiva", "east", 2) == {"spread": [1, 2, 3, 4, 3, 2, 1]}   # not before it was said
    k.add(Record("m2", "prakriti", "marsh", None, 4, "p4", "daiva"))
    assert k.seed("daiva", "east", 5) == {"spread": [1, 2, 3, 4, 3, 2, 1]}   # withdrawn: the marsh says nothing again
    assert k.seed("nothing_said", "east", 5) is None                         # nobody says: the engine's own


def test_a_leaf_carried_out_of_a_branch_stops_standing_under_it() -> None:
    canon, recs = _world()
    recs.append(Record("m1", "prakriti", "marsh", '{"rate": 2}', 0, "p0", "time"))
    k = Kosha(canon, [], recs)
    assert k.seed("time", "rope", 1) == {"rate": 2}
    k.add(_rec(9, "rope", "west", 3))                                        # carried off to the west landing
    assert k.seed("time", "rope", 4) is None


# -- capabilities have a reach (v5 §15, §8) -----------------------------------


def test_a_seeds_capability_reaches_within_and_a_becomings_reaches_only_its_bearer() -> None:
    canon, recs = _world()
    canon.entities["cellar"] = Entity("cellar", ["the cellar"], 0, "genesis", avarana="scene")
    recs.append(_rec(7, "cellar", "east", 0))
    k = Kosha(canon, [], recs)
    assert not k.capable("east", "perceives", 1) and not k.capable("cellar", "perceives", 1)
    k.add(Record("c1", "capability", "east", "perceives", 1, "p1", "on"))     # the landing wakes (a becoming): self
    assert k.capable("east", "perceives", 2) and not k.capable("cellar", "perceives", 2)
    k.add(Record("c2", "capability", "marsh", "perceives", 2, "p2", "within:on"))  # the marsh's seed: within
    assert k.capable("cellar", "perceives", 3) and k.capable("marsh", "perceives", 3)
    assert not k.capable("west", "perceives", 3)                             # outside the marsh
    k.add(Record("c3", "capability", "cellar", "perceives", 3, "p3", "off"))  # its own wins over the branch's
    assert not k.capable("cellar", "perceives", 4)


def test_the_earths_capabilities_read_as_the_worlds_defaults_did() -> None:
    canon, recs = _world()
    seed = (None, {}, {"capabilities": {"perceives": True, "reaches": False}})
    with_defaults = Kosha(canon, [], recs, defaults=seed[2]["capabilities"])
    sown = Kosha(canon, [], recs + prakriti.sow("earth", prakriti.EMPTY, seed, 0, "p0"))
    for eid in canon.entities:
        for cap in ("perceives", "reaches"):
            assert sown.capable(eid, cap, 1) == with_defaults.capable(eid, cap, 1), (eid, cap)


# -- what the Narrator is shown ------------------------------------------------


def _out():
    return {"narration": "x", "perceiver_tuples": [{"perceiver": "teo", "text": "I listened to the water a while.", "salience": "passing"}],
            "scene_tuples": [{"scene": "east", "text": "The boy listened to the water.", "present": ["teo", "ilo"]}],
            "suggested_next_action": ""}


def test_the_earths_view_is_shown_once_where_it_always_was(monkeypatch) -> None:
    canon, recs = _world()
    mems = [_mem(1, "teo", 5, "I sat on the bank.", "east")]
    view = ("Rivers here are persons.", {}, {})
    recs += prakriti.sow("earth", prakriti.EMPTY, view, 0, "p0")
    recs.append(Record("v1", "clause", "marsh", "In the marsh the drowned walk.", 0, "p0", "within:drowned"))
    _, _, client = _pulse(monkeypatch, canon, "teo", "listen", 6, mems, _out(), records=recs,
                          view={"darsana": view[0], "becomings": {}, "world_data": {}})
    system, user = client.calls[0]
    assert system.endswith("THIS WORLD'S OWN VIEW -- it wins over the default wherever they differ:\nRivers here are persons.")
    assert "Rivers here are persons." not in user                           # not again in the views below it
    assert "In the marsh the drowned walk." in user                          # a branch's view still holds here


def test_the_dice_are_read_where_the_attempt_happens(monkeypatch) -> None:
    canon, recs = _world()
    mems = [_mem(1, "teo", 5, "I sat on the bank.", "east")]
    world = {"daiva": {"off": True}}
    recs += prakriti.sow("earth", prakriti.EMPTY, (None, {}, world), 0, "p0")
    view = {"darsana": None, "becomings": {}, "world_data": world}
    _, events, _ = _pulse(monkeypatch, canon, "teo", "listen", 6, mems, _out(), records=list(recs), view=view)
    assert not [f for e, f in events if e == "smrti.daiva"]                  # the world's dice are off
    recs.append(Record("m1", "prakriti", "marsh", '{"spread": [0, 0, 0, 1, 0, 0, 0]}', 0, "p0", "daiva"))
    _, events, _ = _pulse(monkeypatch, canon, "teo", "listen", 6, mems, _out(), records=recs, view=view)
    assert [f["word"] for e, f in events if e == "smrti.daiva"] == [list(DAIVA_WORDS)[3]]  # the marsh casts its own


# -- born with it (v5 §15: branch givens are for worlds born with them) ----------


def _born():
    from smritilila.smrti.bija import as_spec
    from smritilila.smrti.genesis import build
    out = {"name": "t", "setting": "s", "opening": "o", "player": "wat",
           "places": [{"key": "dale", "label": "the dale", "in": "", "scene": False, "tuple": "A long green dale under the fells."},
                      {"key": "hill", "label": "the fairy hill", "in": "dale", "scene": False, "time_rate": 1095, "rooms_remember": True,
                       "tuple": "A round green hill no sheep will graze."},
                      {"key": "hall", "label": "the hall under the hill", "in": "hill", "scene": True, "time_rate": 0,
                       "tuple": "A long hall lit by no fire."},
                      {"key": "valley", "label": "the hidden valley", "in": "dale", "scene": False, "no_strangers": True, "time_rate": 1,
                       "tuple": "A valley behind the falls that no road finds."},
                      {"key": "farm", "label": "the valley farm", "in": "valley", "scene": True, "tuple": "One farm, walled in stone."},
                      {"key": "inn", "label": "the dale inn", "in": "dale", "scene": True, "no_strangers": False, "tuple": "An inn on the dale road."}],
           "people": [{"key": "wat", "label": "Wat the shepherd", "name": "Wat", "place": "inn",
                       "tuples": [{"text": "I have kept sheep in this dale all my life.", "weight": "notable"}]}]}
    spec = as_spec(out)
    ents, mems, config = build(spec, lambda t: None)
    return spec, ents, mems, config, {e.labels[0]: e.id for e in ents}


def test_a_world_born_with_branches_bears_their_nature_where_the_seed_put_it() -> None:
    spec, ents, mems, config, by = _born()
    assert spec["places"]["hill"]["prakriti"] == {"rate": 1095.0, "perceives": True}
    assert "prakriti" not in spec["places"]["hall"] and "prakriti" not in spec["places"]["inn"]  # 0 and false say nothing
    assert spec["places"]["valley"]["prakriti"] == {"closed": True}                              # a rate of 1 is no rate
    nature = sorted((r.kind, r.entity_id, r.note, r.value) for r in config["records"] if r.kind in ("prakriti", "capability"))
    assert nature == sorted([("prakriti", by["the fairy hill"], "time", '{"rate": 1095.0}'),
                             ("capability", by["the fairy hill"], "within:on", "perceives"),
                             ("prakriti", by["the hidden valley"], "pacing", '{"arrivals": false}')])
    k = Kosha(Maya(world_time=1, entities={e.id: e for e in ents}), mems, config["records"])
    assert k.rate(by["the hall under the hill"], 2) == 1095.0 and k.rate(by["the dale inn"], 2) == 1.0
    assert k.capable(by["the hall under the hill"], "perceives", 2) and not k.capable(by["the dale inn"], "perceives", 2)
    assert k.seed("pacing", by["the valley farm"], 2, field="arrivals") is False
    assert k.seed("pacing", by["the dale inn"], 2, field="arrivals") is None                  # the world's own


def test_a_closed_valley_is_never_asked_who_came_and_the_dale_around_it_is(monkeypatch) -> None:
    from smritilila import main
    spec, ents, mems, config, by = _born()
    canon = Maya(world_time=40, entities={e.id: e for e in ents})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id=by["Wat the shepherd"], entry_point_location_ids=[by["the dale inn"]],
                              playable_entity_ids=[by["Wat the shepherd"]])
    world.world_data["pacing"] = {"arrivals_on_traffic": False}  # asked on the clock: the ask itself, not its traffic gate
    world.records = list(config["records"])
    world.recent_memories = list(mems)
    pool = pravrtti._smrti_exploration_pool(world)
    assert by["the valley farm"] not in pool
    world.records.append(Record("o1", "prakriti", by["the hidden valley"], None, 3, "p3", "pacing"))  # the valley opened
    world._samskara = None
    assert by["the valley farm"] in pravrtti._smrti_exploration_pool(world)


def test_a_nature_copied_onto_every_scene_is_the_regions_and_a_copied_rate_does_not_compose() -> None:
    """The replay of 2026-10-01: the mid-size model put Ashgrove Hall's 'perceives' on each of its rooms,
    not the hall. Settled as laws are: what every place directly in a region carries is the
    region's; what a place's container already carries is inherited -- or a hall copying its
    hill's rate would run at the rate squared."""
    from smritilila.smrti.genesis import prakriti_records
    places = {"hill": {"label": "the hill", "prakriti": {"rate": 852.0}},
              "hall": {"label": "the hall", "in": "hill", "prakriti": {"rate": 852.0}},
              "manor": {"label": "the manor"},
              "great": {"label": "the great hall", "in": "manor", "prakriti": {"perceives": True}},
              "morning": {"label": "the morning room", "in": "manor", "prakriti": {"perceives": True}}}
    ids = {k: f"id_{k}" for k in places}
    got = sorted((r.kind, r.entity_id, r.note, r.value) for r in prakriti_records(places, ids, 0, "genesis"))
    assert got == [("capability", "id_manor", "within:on", "perceives"), ("prakriti", "id_hill", "time", '{"rate": 852.0}')]


def test_a_seed_that_cannot_be_planted_is_asked_for_once_more(monkeypatch) -> None:
    """Marc, 2026-10-01: a broken seed call (the mid-size model's `places` as text or markup) is the model's
    slip -- asked once more; a second failure fails the birth as before."""
    from smritilila.smrti import bija
    events = []
    monkeypatch.setattr(bija, "log_event", lambda e, **f: events.append((e, f)))
    good = {"name": "t", "setting": "s", "begins": "quay", "people": [],
            "places": [{"key": "quay", "label": "the quay", "in": "", "scene": True, "tuple": "A plank quay."}]}

    class Broken:
        def __init__(self, n):
            self.left = n

        def complete_with_tool(self, *a, **k):
            self.left -= 1
            return {**good, "places": '<parameter name="key">quay'} if self.left >= 0 else good

    spec = bija.write_world_seed(Broken(1), "t", {"source": "x"}, breadth="here")
    assert list(spec["places"]) == ["quay"] and [e for e, _ in events].count("bija.retry") == 1
    import pytest
    with pytest.raises(ValueError, match="names no place"):
        bija.write_world_seed(Broken(2), "t", {"source": "x"}, breadth="here")
