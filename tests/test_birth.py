# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Birth memories (Marc, 2026-11-06: every time an entity is assigned its first memory it is also given a birth memory that comes before it):
the writer (smrti/bija.py), the planter (genesis.plant_person) and the builder's three routes -- a world's seed, its outward rings, a proposal."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from smritilila import adhisthana
from smritilila.smrti import bija
from smritilila.smrti.bija import BIRTH_SCHEMA, BIRTH_SYSTEM, birth_asks, write_birth, write_births
from smritilila.smrti.genesis import build, plant_more, plant_person
from smritilila.smrti.pratistha import give_births, plant, summary
from test_main import _BijaClient, _SpyNarrator, _fake_roles, client, fake_store  # noqa: F401 -- the shared fixtures

BIRTH = [{"text": "I was a drowned tinker's daughter, raised by the reed-cutters, and I count who comes because someone must.", "weight": "marking", "use": "useful"},
         {"text": "I believe the fen keeps an account and that every debt to it falls due; I cannot abide a liar at the water.", "weight": "notable", "use": "useful"},
         {"text": "I live so that the fen's account is kept honestly and no one is given to it unmourned.", "weight": "indelible", "use": "key"}]


class _Writer:
    """A model that answers write_birth, and fails for any being whose name is in `fail`."""

    def __init__(self, fail: tuple[str, ...] = (), tuples: list | None = None) -> None:
        self.users: list[str] = []
        self.fail = fail
        self.tuples = BIRTH if tuples is None else tuples

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
        assert tool_name == "write_birth" and system == BIRTH_SYSTEM and input_schema is BIRTH_SCHEMA
        self.users.append(user)
        if any(f in user for f in self.fail):
            raise RuntimeError("the model slipped")
        return {"tuples": json.loads(json.dumps(self.tuples))}


def test_the_writer_is_shown_who_where_and_what_the_being_already_remembers() -> None:
    w = _Writer()
    got = write_birth(w, "Ossa Wend", "the Reed Chapel", ["I keep the chapel for those who pay the fen its due.", "The quiet men bring me silver."],
                      bhuvana="THE YEAR. A fen under a grey sky. THE REGIONS. The fen.", source="Fiction: the fens")
    assert [t["text"] for t in got] == [b["text"] for b in BIRTH] and [t["use"] for t in got] == ["useful", "useful", "key"]
    user = w.users[0]
    assert "THE BEING: Ossa Wend" in user and "WHERE IT IS: the Reed Chapel" in user
    assert "  - I keep the chapel for those who pay the fen its due.\n  - The quiet men bring me silver." in user    # what its beginning comes before
    assert "THIS WORLD'S SOURCE: Fiction: the fens" in user and "A fen under a grey sky" in user
    assert "THE DEFAULT VIEW" in user                                                                          # the world's physics, as any writer into a world is shown


def test_the_writers_wording_keeps_the_three_fixes_the_second_replay_tested() -> None:
    """Replayed 2026-11-08 (18 calls, model-b, the six recorded cases): the last tuple a purpose and never a re-telling or a plan, no
    person named that the memory did not give, no shared past claimed with one who would not remember it. Each is a sentence of this prompt."""
    assert "LIFE PURPOSE" in BIRTH_SYSTEM and "is never a duty restated" in BIRTH_SYSTEM
    assert "re-tell none of it, state no plan for the day, and write nothing that happens after it" in BIRTH_SYSTEM
    assert "Name a person only if the memory they already have names them" in BIRTH_SYSTEM
    assert "never give a named person a past with them that the memory does not state" in BIRTH_SYSTEM
    assert "two or three tuples" in BIRTH_SYSTEM and "A creature writes as what it is" in BIRTH_SYSTEM


def test_an_answer_under_two_tuples_is_refused_and_a_long_one_is_cut_to_three() -> None:
    with pytest.raises(ValueError):
        write_birth(_Writer(tuples=BIRTH[:1]), "Ossa", "the chapel", ["I keep it."])
    with pytest.raises(ValueError):
        write_birth(_Writer(tuples=[BIRTH[0], {"text": "Hm.", "weight": "ordinary"}]), "Ossa", "the chapel", ["I keep it."])   # one too short to be a tuple
    assert [t["text"] for t in write_birth(_Writer(tuples=BIRTH + BIRTH), "Ossa", "the chapel", ["I keep it."])] == [b["text"] for b in BIRTH]


def test_a_failed_call_leaves_one_being_without_a_birth_and_fails_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple] = []
    monkeypatch.setattr(bija, "log_event", lambda name, **kw: events.append((name, kw)))
    beings = [{"key": k, "label": k.title(), "place": "the chapel", "memory": ["I keep it, as I always have."]} for k in ("ossa", "wend", "brae")]
    got = write_births(_Writer(fail=("THE BEING: Wend",)), beings)
    assert set(got) == {"ossa", "brae"} and all(len(v) == 3 for v in got.values())
    assert ("bija.birth_failed", {"key": "wend", "error": "the model slipped"}) in events and ("bija.births", {"asked": 3, "written": 2}) in events
    assert write_births(_Writer(), []) == {}


def test_a_player_and_a_being_with_no_memory_are_not_asked_for() -> None:
    people = {"wren": {"label": "Wren", "place": "ford", "tuples": ["I was born in the mill."]},
              "ossur": {"label": "Ossur", "place": "hof", "tuples": [{"text": "I keep the peace.", "weight": "weighty"}, "  "]},
              "blank": {"label": "Nobody", "place": "ford", "tuples": []}}
    asks = birth_asks(people, lambda k: {"ford": "the mill ford", "hof": "Hof"}[k], skip=frozenset({"wren"}))
    assert asks == [{"key": "ossur", "label": "Ossur", "place": "Hof", "memory": ["I keep the peace."]}]


def test_plant_person_plants_the_birth_first_in_one_tick_with_its_own_weights() -> None:
    first = [{"text": "I keep the Reed Chapel for those who pay the fen its due.", "weight": "weighty", "use": "useful"},
             {"text": "The quiet men bring me silver every new moon.", "weight": "notable", "use": "key"}]
    _, mems, _ = plant_person("Ossa", first, "chapel", 40, "p40", lambda t: None, birth=BIRTH)
    assert [m.experience for m in mems] == [b["text"] for b in BIRTH] + [f["text"] for f in first]
    assert [m.id for m in mems] == sorted(m.id for m in mems) and {m.world_time for m in mems} == {40}             # ids sort in order, in the one tick
    assert [(m.salience, m.use) for m in mems] == [(2, "useful"), (4, "useful"), (1, "key"), (3, "useful"), (4, "key")]
    assert {m.pulse_id for m in mems} == {"p40"} and {m.location_id for m in mems} == {"chapel"}                 # ordinary tuples: nothing marks them
    _, plain, _ = plant_person("Ossa", first, "chapel", 40, "p40", lambda t: None)
    assert [m.experience for m in plain] == [f["text"] for f in first]                                           # no birth: as before


def test_a_backdated_seed_runs_the_birth_up_to_the_first_tuples_without_moving_them() -> None:
    first = ["I keep the Reed Chapel for those who pay the fen its due.", "The quiet men bring me silver every new moon."]
    _, plain, _ = plant_person("Ossa", first, "chapel", 0, "genesis", lambda t: None, backdate=True)
    _, born, _ = plant_person("Ossa", first, "chapel", 0, "genesis", lambda t: None, backdate=True, birth=BIRTH)
    assert [m.world_time for m in plain] == [-10, 0] and [m.world_time for m in born] == [-40, -30, -20, -10, 0]
    assert [m.experience for m in born[3:]] == first and [m.salience for m in born[3:]] == [m.salience for m in plain]


SPEC = {"name": "The Mill Ford", "setting": "A mill on a river.", "scene_owner": "wren", "entry_points": ["ford"], "playable": ["wren"],
        "places": {"ford": {"label": "the mill ford", "tuple": "A shallow ford below the mill; the north road climbs away from it."}},
        "people": {"wren": {"label": "Wren", "place": "ford", "tuples": ["I was born in the mill and have never slept away from the wheel.",
                                                                          "This morning I came down to the ford to wait for the north trader."]},
                   "ossur": {"label": "Ossur the godi", "place": "ford", "tuples": ["I keep the peace on this water by memory more than by law."]}}}


def test_a_seed_plants_a_birth_for_whoever_it_was_given_to_and_leaves_the_rest() -> None:
    ents, mems, config = build(SPEC, lambda t: None, births={"ossur": BIRTH})
    ids = {e.labels[0]: e.id for e in ents}
    ossur = sorted((m for m in mems if m.entity_id == ids["Ossur the godi"]), key=lambda m: (m.world_time, m.id))
    assert [m.experience for m in ossur] == [b["text"] for b in BIRTH] + SPEC["people"]["ossur"]["tuples"]
    wren = [m for m in mems if m.entity_id == ids["Wren"]]
    assert len(wren) == 2 and [m.world_time for m in wren] == [-10, 0]                                            # the player: their own description, as it was


def test_an_expansion_plants_births_before_its_people_first_tuples() -> None:
    new = {"places": {"hof": {"label": "Hof", "tuple": "The largest hall on the water, north of the mill ford."}},
           "people": {"ossur": {"label": "Ossur the godi", "place": "hof", "tuples": ["I keep the peace on this water by memory more than by law."]}}}
    ents, mems, _ = plant_more(new, {}, lambda t: None, 50, pulse_id="vistara:1", births={"ossur": BIRTH})
    ossur = next(e.id for e in ents if e.labels[0] == "Ossur the godi")
    own = sorted((m for m in mems if m.entity_id == ossur), key=lambda m: (m.world_time, m.id))
    assert [m.experience for m in own] == [b["text"] for b in BIRTH] + new["people"]["ossur"]["tuples"] and {m.world_time for m in own} == {50}


def test_a_proposal_gives_each_person_a_birth_the_god_reads_and_the_plant_carries() -> None:
    spec = {"title": "The Reed Chapel", "why": "the talk names it",
            "places": [{"key": "chapel", "label": "a chapel of reeds", "in": "fen", "scene": True, "tuple": "A chapel woven of green reeds stands on stilts over the water."}],
            "people": [{"key": "keeper", "label": "a priest in a wet grey robe", "name": "Ossa Wend", "in": "chapel",
                        "tuples": [{"text": "I keep the Reed Chapel for those who pay the fen its due.", "weight": "weighty", "use": "useful"}]},
                       {"key": "boy", "label": "a boy with a punt", "name": "Tam", "in": "fen",
                        "tuples": [{"text": "I pole the quiet men across at night and ask nothing.", "weight": "ordinary", "use": "none"}]}]}
    w = _Writer(fail=("THE BEING: Tam",))
    out = give_births(w, spec, lambda x: {"chapel": "a chapel of reeds"}.get(x, "the black fen"), "THE YEAR. A fen.", None, "Fiction: the fens")
    assert out is spec and [r["birth"] for r in spec["people"] if "birth" in r] == [BIRTH] and "birth" not in spec["people"][1]   # Tam's call failed: none, no fault
    assert any("Ossa Wend" in w_ and "WHERE IT IS: a chapel of reeds" in w_ for w_ in w.users) and any("THE BEING: Tam, a boy with a punt" in w_ for w_ in w.users)
    assert any("a birth memory of 3" in line for line in summary(spec))
    _, mems, _, ids = plant(spec, 12, None, "pratistha:x", lambda t: None)
    own = sorted((m for m in mems if m.entity_id == ids["keeper"]), key=lambda m: (m.world_time, m.id))
    assert [m.experience for m in own][:3] == [b["text"] for b in BIRTH] and len(own) == 4
    assert len([m for m in mems if m.entity_id == ids["boy"]]) == 1


def test_a_proposal_whose_births_blow_up_is_still_a_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route's guard, over give_births itself (write_births already isolates a failed call)."""
    import asyncio
    from types import SimpleNamespace
    from smritilila.routes import pratistha as routes
    events: list[str] = []
    monkeypatch.setattr(routes, "log_event", lambda name, **kw: events.append(name))
    monkeypatch.setattr("smritilila.smrti.pratistha.give_births", lambda *a, **k: 1 / 0)
    world = SimpleNamespace(canon=SimpleNamespace(entities={}), bhuvana=None, darsana=None, world_data={})
    spec = {"places": [], "people": [{"key": "keeper", "label": "a priest", "in": "chapel", "tuples": []}]}
    assert asyncio.run(routes._give_births(world, SimpleNamespace(smrti_client=None), spec)) is spec and events == ["pratistha.births_failed"]


class _BirthClient(_BijaClient):
    """The mill-ford seed with a second person beside the player (not the godi: the ring adds him), and a birth memory for whoever is asked."""

    fail = False

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
        if tool_name == "write_birth":
            self.births.append(user)
            if self.fail:
                raise RuntimeError("the model slipped")
            return {"tuples": json.loads(json.dumps(BIRTH))}
        out = super().complete_with_tool(system, user, tool_name=tool_name, tool_description=tool_description, input_schema=input_schema)
        if tool_name == "write_seed" and "EXPANDING" not in system:
            out["people"].append({"key": "brae", "label": "Brae the ferrywright", "place": "ford",
                                  "tuples": [{"text": "I keep the ford crossing in repair, as my father did before me.", "weight": "weighty"}]})
        return out

    def __init__(self) -> None:
        super().__init__()
        self.births: list[str] = []


def _roles(monkeypatch: pytest.MonkeyPatch, writer: _BirthClient):
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = writer
    monkeypatch.setattr(adhisthana, "resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    return roles


def test_a_built_world_gives_its_people_births_but_not_its_player(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    writer = _BirthClient()
    _roles(monkeypatch, writer)
    client.post("/world/clear")
    r = client.post("/genesis", json={"self_description": "I am Wren.", "setting": "A mill on a river, high summer.", "breadth": "here", "source": "Fiction: the mill"})
    assert r.status_code == 200, r.text
    world = client.app.state.world
    by = {e.labels[0]: e.id for e in world.canon.entities.values()}
    brae = sorted((m for m in world.recent_memories if m.entity_id == by["Brae the ferrywright"]), key=lambda m: (m.world_time, m.id))
    assert [m.experience for m in brae] == [b["text"] for b in BIRTH] + ["I keep the ford crossing in repair, as my father did before me."]
    assert [m.world_time for m in brae] == [-30, -20, -10, 0]                                                          # the birth runs up to the tuple they were written with
    assert [m.salience for m in world.recent_memories if m.entity_id == by["Wren, the miller's daughter"]] == [2, 5]      # the player: only what she wrote
    assert len(writer.births) == 1 and "THE BEING: Brae the ferrywright" in writer.births[0] and "WHERE IT IS: the mill ford" in writer.births[0]
    assert "THIS WORLD'S SOURCE: Fiction: the mill" in writer.births[0]


def test_a_world_whose_births_all_fail_is_still_built(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    writer = _BirthClient()
    writer.fail = True
    _roles(monkeypatch, writer)
    client.post("/world/clear")
    r = client.post("/genesis", json={"self_description": "I am Wren.", "setting": "A mill on a river, high summer.", "breadth": "here"})
    assert r.status_code == 200, r.text
    world = client.app.state.world
    brae = next(e.id for e in world.canon.entities.values() if e.labels[0] == "Brae the ferrywright")
    assert [m.experience for m in world.recent_memories if m.entity_id == brae] == ["I keep the ford crossing in repair, as my father did before me."]


def test_an_outward_ring_gives_each_new_person_a_birth(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    writer = _BirthClient()
    _roles(monkeypatch, writer)
    client.post("/world/clear")
    assert client.post("/genesis", json={"self_description": "I am Wren.", "setting": "A mill on a river, high summer.", "breadth": "here"}).status_code == 200
    world = client.app.state.world
    before = set(world.canon.entities)
    writer.births.clear()
    r = client.post("/world/expand")
    assert r.status_code == 200, r.text
    added = {e.labels[0]: e for e in world.canon.entities.values() if e.id not in before}
    people = [e for label, e in added.items() if label not in ("Hof, the chieftain's hall", "the sea road")]
    assert {e.labels[0] for e in people} >= {"Ossur the godi", "Gudrun Hallsdóttir, Wren's aunt"}
    assert len(writer.births) == len(people) and any("WHERE IT IS: Hof, the chieftain's hall" in u for u in writer.births)   # a place the ring added
    assert any("WHERE IT IS: the mill ford" in u for u in writer.births)                                              # a place the world already held, by its key
    for p in people:
        own = sorted((m for m in world.recent_memories if m.entity_id == p.id), key=lambda m: (m.world_time, m.id))
        assert [m.experience for m in own][:3] == [b["text"] for b in BIRTH] and len(own) == 4
    assert not any(m.experience in {b["text"] for b in BIRTH} for m in world.recent_memories if m.entity_id in {added["the sea road"].id, added["Hof, the chieftain's hall"].id})
