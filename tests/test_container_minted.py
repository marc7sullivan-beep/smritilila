# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""A move mints what it names, and the Narrator declares what holds it (Marc, 2026-10-15: "If something is attempted to be moved into an unknown container, a
container MUST be minted first"; "The narrator has to be the one to declare it"; an unknown thing too).

A thing put into something that is not an entity cannot be put anywhere. The engine used to refuse the move (`unknown_container`), so a key pushed into
"a hollow in the roof beam" stayed where it was and the hollow never existed. Now the move makes whatever it names that has no id, thing or container, so
the name exists. NO HOLDER IS GUESSED: the Narrator declares what holds a new container in new_entities, in the same answer (rule 3), and then the move
stands. One it leaves undeclared is a mention, in the earth, and anything put into it is put back where it was (`_mentions`), so the move does not stand
until the Narrator says what holds the container; `smrti.move_undeclared` counts them. Everything is a container, so nothing else is needed.
"""

from __future__ import annotations

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.smrti.kosha import Kosha
from smritilila.smrti.pulse import run_smrti_pulse


def _world():
    ents = {
        "county": Entity("county", ["the county"], 0, "genesis"),
        "village": Entity("village", ["the village"], 0, "genesis", kosha="county"),
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene", kosha="village"),
        "mug": Entity("mug", ["a mug"], 0, "genesis", kosha="inn"),
        "key": Entity("key", ["a brass key"], 0, "genesis", kosha="inn"),
        "wen": Entity("wen", ["Wen"], 0, "genesis", kosha="inn"),
    }
    mems = [Memory("m1", "wen", frozenset(), "I sat in the inn.", "PRIMARY", 4, "inn", 1, "p1", None),
            Memory("m2", "village", frozenset(), "A village of thatch.", "PRIMARY", 4, "village", 0, "p0", None)]
    return ents, mems


def _run(monkeypatch, extra: dict, extra_entities: dict | None = None, sandhi: bool = True):
    """Every live world has sandhi on (Marc, 2026-10-01), and the put-back of a move into an undeclared container is `_mentions`, which runs only there."""
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", False)
    ents, mems = _world()
    ents.update(extra_entities or {})
    events = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **kw: events.append((e, kw)))
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **kw: None)

    class C:
        def complete_with_tool(self, system, user, **kw):
            return {"narration": "You put it away.", "perceiver_tuples": [{"perceiver": "wen", "text": "I put it away.", "salience": "ordinary"}],
                    "scene_tuples": [], "suggested_next_action": "", **extra}
    canon = Maya(world_time=3, entities=dict(ents))
    view = {"darsana": None, "becomings": {}, "world_data": {"sandhi": True} if sandhi else {}}
    r = run_smrti_pulse(canon, "wen", "I put it away.", 3, "p3", mems, C(), None, view=view)
    k = Kosha(canon, mems + r.memories_written, r.records_written)
    refused = [x for e, f in events if e == "smrti.move_refused" for x in f["refused"]]
    undeclared = [kw for e, kw in events if e == "smrti.move_undeclared"]
    return r, k, refused, undeclared


def _made(r, label):
    return [e for e in r.new_entities if e.labels == [label]]


def test_a_move_into_a_container_nobody_declared_mints_it_and_does_not_stand(monkeypatch) -> None:
    r, k, refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": "a hollow in the roof beam"}]})
    made = _made(r, "a hollow in the roof beam")
    assert len(made) == 1                                           # the container exists: the move made it
    assert [u["role"] for u in undeclared] == ["container"] and undeclared[0]["id"] == made[0].id
    assert not [x for x in refused if x["reason"] == "unknown_container"]
    assert k.container_at("key", 4) == "inn"                        # nothing said what holds it, so the key was put back where it was
    assert k.container_at(made[0].id, 4) is None                    # and no holder was guessed for the hollow


def test_without_sandhi_the_container_is_made_but_nothing_puts_the_thing_back(monkeypatch) -> None:
    """Pinned, not wished: `_mentions` is a sandhi world's (v5 §10), so where sandhi is off the thing goes into a container that holds nothing. No live world is such."""
    r, k, _refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": "a hollow in the roof beam"}]}, sandhi=False)
    made = _made(r, "a hollow in the roof beam")
    assert len(made) == 1 and len(undeclared) == 1
    assert k.container_at("key", 4) == made[0].id and k.container_at(made[0].id, 4) is None


def test_a_container_the_narrator_declares_with_its_holder_takes_the_thing(monkeypatch) -> None:
    r, k, _refused, undeclared = _run(monkeypatch, {
        "new_entities": [{"id": "NEW:hollow", "name": "a hollow in the roof beam", "in": "inn", "scene": False}],
        "moves": [{"entity": "key", "into": "NEW:hollow"}]})
    made = _made(r, "a hollow in the roof beam")
    assert len(made) == 1 and not undeclared                        # declared: nothing is left undeclared
    assert k.container_at(made[0].id, 4) == "inn" and k.container_at("key", 4) == made[0].id


def test_a_belt_pouch_the_narrator_hangs_on_the_carrier_is_the_carriers(monkeypatch) -> None:
    """Marc: a belt pouch is its carrier's. The Narrator says so; the engine does not decide it from the thing put in."""
    r, k, _refused, undeclared = _run(monkeypatch, {
        "new_entities": [{"id": "NEW:pouch", "name": "a belt pouch", "in": "wen", "scene": False}],
        "moves": [{"entity": "wenkey", "into": "NEW:pouch"}]}, extra_entities={"wenkey": Entity("wenkey", ["Wen's key"], 0, "genesis", kosha="wen")})
    pouch = _made(r, "a belt pouch")
    assert len(pouch) == 1 and not undeclared
    assert k.chain("wenkey", 4)[:2] == [pouch[0].id, "wen"]


def test_a_thing_with_no_id_is_made_by_the_move(monkeypatch) -> None:
    r, k, _refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "the brass leaf", "into": "mug"}]})
    leaf = _made(r, "the brass leaf")
    assert len(leaf) == 1 and [u["role"] for u in undeclared] == ["thing"]
    assert k.container_at(leaf[0].id, 4) == "mug"                   # the container exists and is reached: the new thing is in it


def test_two_things_put_into_the_same_undeclared_container_make_one(monkeypatch) -> None:
    r, _k, _refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": "a cracked bowl"}, {"entity": "mug", "into": "a cracked bowl"}]})
    assert len(_made(r, "a cracked bowl")) == 1 and len(undeclared) == 1


def test_a_new_things_declared_holder_that_is_not_an_entity_is_minted_too(monkeypatch) -> None:
    r, k, refused, undeclared = _run(monkeypatch, {"new_entities": [{"id": "NEW:copper", "name": "a copper coin", "in": "the cracked bowl", "scene": False}]})
    assert len(_made(r, "the cracked bowl")) == 1 and [u["role"] for u in undeclared] == ["container"]
    assert not [x for x in refused if x["reason"] == "unknown_container"]


def test_an_invented_id_is_cleaned_of_its_hash_tail(monkeypatch) -> None:
    """Training log 4093's cousin, the one real unknown_container in the witness logs: the Narrator wrote an id-shaped string for a door that did not exist."""
    r, _k, _refused, _undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": "barred_cellar_door_west_82c019"}]})
    assert len(_made(r, "barred cellar door west")) == 1


def test_words_that_mean_nothing_are_not_a_container(monkeypatch) -> None:
    r, k, refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": "nothing"}]})
    assert not [e for e in r.new_entities if e.labels == ["nothing"]] and not undeclared
    assert [(x["entity"], x["reason"]) for x in refused] == [("key", "unknown_container")]
    assert k.container_at("key", 4) == "inn"


def test_a_known_container_is_never_minted_again(monkeypatch) -> None:
    r, k, _refused, undeclared = _run(monkeypatch, {"moves": [{"entity": "key", "into": "mug"}]})
    assert not r.new_entities and not undeclared and k.container_at("key", 4) == "mug"


def test_the_narrator_is_told_that_a_move_declares_what_it_names(monkeypatch) -> None:
    """Rule 3 and rule 5 are the Narrator's half of the operation (Marc, 2026-10-15: "The narrator has to be the one to declare it"): a thing picked up or put
    into something is moved; a thing or a container with no id is declared in the same answer, with what holds it. Old words back would put the gap back."""
    from smritilila.smrti import narrator as nar
    s = str(nar.system_for(None, None, False))
    assert "whatever changes what holds it this pulse is moved" in s and "picked up, put down, handed over, put into something or taken out of it" in s
    assert "a part that comes out of its whole" in s and "everything is a container, and a container can hold things" in s
    assert "a container nobody has named -- is declared there, with what holds it, in the same answer" in s
    assert "what the memory only tells of in its prose (a leaf of a book, a coin in a purse) when it is moved or acted on alone" in s
    assert "ONLY for what genuinely does not appear anywhere" not in s
    assert "for every entity the pulse moved" not in s
    assert "NEW:<mnemonic> for what has no id yet" in str(nar.SCHEMA["properties"]["moves"])


def test_held_by_is_gone_from_the_contract_the_schema_and_the_pulse(monkeypatch) -> None:
    """Marc, 2026-10-15: "we are removing held_by entirely". Nothing reads SMRITILILA_HELD_BY, the prompt and the schema carry no held_by, and a chain a model writes
    anyway (from habit, or a stale prompt) is ignored: it makes nothing and moves no one."""
    import json

    from smritilila.smrti import narrator as nar
    from smritilila.smrti import pulse as pl
    monkeypatch.delenv("SMRITILILA_HELD_BY", raising=False)
    off = str(nar.system_for(None, None, False))
    monkeypatch.setenv("SMRITILILA_HELD_BY", "1")
    assert str(nar.system_for(None, None, False)) == off and "held_by" not in off and "HELD BY." not in off
    assert "held_by" not in json.dumps(nar.SCHEMA) and not hasattr(pl, "chains_to_placement") and not hasattr(nar, "held_by_on")
    monkeypatch.setattr("smritilila.smrti.pulse.SANDHI_ALL", False)
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda e, **kw: None)
    monkeypatch.setattr("smritilila.smrti.pulse.log_discarded", lambda **kw: None)
    ents, mems = _world()

    class C:
        def complete_with_tool(self, system, user, **kw):
            return {"narration": "x", "perceiver_tuples": [{"perceiver": "wen", "text": "I climbed into the boat at the landing and pushed off from the shore.", "salience": "ordinary", "held_by": ["NEW:rowboat", "inn"]}],
                    "scene_tuples": [], "suggested_next_action": ""}
    canon = Maya(world_time=3, entities=dict(ents))
    r = run_smrti_pulse(canon, "wen", "I climb in", 3, "p3", mems, C(), None, view={"darsana": None, "becomings": {}, "world_data": {"sandhi": True}})
    assert not r.new_entities and Kosha(canon, mems + r.memories_written, r.records_written).container_at("wen", 4) == "inn"
