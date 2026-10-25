# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""SMRITILILA_HOLDS (Marc, 2026-10-13, Stonemarch 1637-1658): everything is a container and a container can hold things.

Jakal's spellbook took five leaves and held nothing: the prompt listed the actor's own things as bare names and showed an empty one as nothing, so
it looked like any item and nothing marked it as a place a leaf could go. The prompt now says what each thing the actor holds holds, empty
included; rule 7 says a thing that holds things is declared with them. Neither names a kind of thing, a verb or a book. (The first version
named the things the action mentioned, and a text match could not do it: "and" counted as a rare word, and "my spell book" is not "Jakal's spellbook".)"""

from __future__ import annotations

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti import narrator
from smritilila.smrti.context import Context
from smritilila.smrti.kosha import Kosha
from smritilila.smrti.pulse import HOLDS_SHOWN, fill_holds


def _world():
    ents = {
        "room": Entity("room", ["the upper chamber"], 0, "genesis", avarana="scene"),
        "jak": Entity("jak", ["Jakal Hierophant"], 0, "genesis", kosha="room"),
        "spe": Entity("spe", ["Spugnoir of Dyvers"], 0, "genesis", kosha="room"),
        "book": Entity("book", ["Jakal's spellbook"], 0, "genesis", kosha="jak"),
        "pair": Entity("pair", ["a pair of throwing knives"], 0, "genesis", kosha="jak"),
        "dagger": Entity("dagger", ["Jakal's thrown dagger"], 0, "genesis", kosha="pair"),
        "knife": Entity("knife", ["Jakal's throwing knife"], 0, "genesis", kosha="pair"),
        "satchel": Entity("satchel", ["a canvas satchel"], 0, "genesis", kosha="spe"),
        "skin": Entity("skin", ["a waterskin"], 0, "genesis", kosha="satchel"),
        "mace": Entity("mace", ["a flanged iron mace"], 0, "genesis", kosha="jak"),
    }
    recs = [Record(f"r{i}", "kosha", e, ents[e].kosha, 0, "p0", "declared") for i, e in enumerate(ents) if ents[e].kosha]
    mems = [Memory(f"m{i}", e, frozenset(), f"{ents[e].labels[0]}", "PRIMARY", 4, "room", 1, "p1", None, None) for i, e in enumerate(ents)]
    return Kosha(Maya(world_time=5, entities=ents), mems, recs), ents


def _ctx(present, held_by_actor):
    holder = {x: "jak" for x in held_by_actor}
    holder.update({"satchel": "spe", "skin": "satchel"})
    return Context(actor="jak", place="room", tuples=[], present=set(present) | set(held_by_actor), holder=holder)


def test_each_thing_the_actor_holds_is_shown_with_what_it_holds_or_as_empty() -> None:
    k, _ = _world()
    ctx = _ctx({"spe", "satchel", "skin"}, {"book", "pair", "mace"})
    assert ctx.holds == {}
    fill_holds(k, ctx, 6)
    assert ctx.holds == {"book": [], "pair": ["Jakal's thrown dagger", "Jakal's throwing knife"], "mace": []}
    shown = narrator._present(ctx)
    assert "book (empty)" in shown and "mace (empty)" in shown
    assert "pair (holding: Jakal's thrown dagger, Jakal's throwing knife)" in shown


def test_without_it_the_actors_things_are_bare_names() -> None:
    k, _ = _world()
    ctx = _ctx({"spe", "satchel", "skin"}, {"book", "pair", "mace"})
    shown = narrator._present(ctx)                                  # SMRITILILA_HOLDS off: fill_holds not called
    assert "book" in shown and "(empty)" not in shown and "holding: Jakal" not in shown


def test_what_other_people_hold_is_left_to_present_as_before() -> None:
    k, _ = _world()
    ctx = _ctx({"spe", "satchel", "skin"}, {"book"})
    fill_holds(k, ctx, 6)
    assert set(ctx.holds) == {"book"}                               # not Spugnoir's satchel
    assert "satchel (holding: skin)" in narrator._present(ctx)


def test_one_thing_lists_at_most_the_cap() -> None:
    ents = {"room": Entity("room", ["the upper chamber"], 0, "genesis", avarana="scene"), "jak": Entity("jak", ["Jakal"], 0, "genesis", kosha="room"),
            "coffer": Entity("coffer", ["a cedar coffer"], 0, "genesis", kosha="jak")}
    for i in range(HOLDS_SHOWN + 3):
        ents[f"c{i}"] = Entity(f"c{i}", [f"coin number {i}"], 0, "genesis", kosha="coffer")
    recs = [Record(f"r{e}", "kosha", e, ents[e].kosha, 0, "p0", "declared") for e in ents if ents[e].kosha]
    k = Kosha(Maya(world_time=5, entities=ents), [Memory("m", "jak", frozenset(), "x", "PRIMARY", 4, "room", 1, "p1", None, None)], recs)
    ctx = Context(actor="jak", place="room", tuples=[], present={"coffer"}, holder={"coffer": "jak"})
    fill_holds(k, ctx, 6)
    assert len(ctx.holds["coffer"]) == HOLDS_SHOWN + 1 and ctx.holds["coffer"][-1] == "3 more"


def test_rule_7_says_what_holds_things_is_declared_with_them_and_names_no_verb(monkeypatch) -> None:
    rule = narrator.HOLDS_RULE
    assert "everything is a container" in rule and "declare each thing it holds as NEW" in rule
    for specific in ("write", "writing", "inscri", "stitch", "bind", "spellbook"):     # Marc: writing and binding were ruled too specific
        assert specific not in rule.lower().replace("a book with its spells", ""), specific
    system = narrator.SYSTEM
    assert narrator.HOLDS_ANCHOR in system and rule not in system
    once = narrator.holds_rules(system)
    assert once.count(rule) == 1 and narrator.holds_rules(once) == once                  # said once
    monkeypatch.delenv("SMRITILILA_HOLDS", raising=False)
    assert rule not in str(narrator.system_for())                                          # off by default
    monkeypatch.setenv("SMRITILILA_HOLDS", "1")
    assert str(narrator.system_for()).count(rule) == 1
