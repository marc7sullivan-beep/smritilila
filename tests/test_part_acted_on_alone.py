# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""A PART ACTED ON ALONE (Marc, 2026-10-14, Stonemarch 1641-1658, Jakal's vellum): a group is one thing until a member is picked out and
acted on by itself; then that member is its own thing, declared NEW in the whole, and the whole is simply smaller.

The old clause (rule 7, "a part LEAVING its whole") was written for separations: a dart thrown, an arrow loosed, a coin paid out. A sheet of
vellum written on moves nothing, so nothing in it read as that, and in five logged pulses no leaf was ever minted (the nine leaves and the
inkpot had been minted as one lump at the purchase, which is fine: the lump is the stack). The clause is now about being acted on alone.
It names no verb of any one kind of thing beyond a list of different ones, and no book, leaf or writing is named as a case.
Replayed once on the second model, one pulse on each of ten groups and actions: nine minted the one member, the kick did not (no control)."""

from __future__ import annotations

from smritilila.smrti import narrator

CLAUSE_START = "A PART ACTED ON ALONE:"


def test_rule_7_says_a_part_picked_out_and_acted_on_alone_is_declared_new_in_the_whole() -> None:
    s = narrator.SYSTEM
    assert s.count(CLAUSE_START) == 1
    clause = s[s.index(CLAUSE_START): s.index("8. ACTION is what the actor ATTEMPTS")]
    assert "picked out and acted on by itself" in clause
    assert "declare that one as NEW `in` the whole" in clause
    assert "the whole stays where it is, and is simply smaller" in clause
    assert "A part that comes back is moved INTO its whole" in clause and clause.rstrip().endswith("never beside it.")


def test_the_old_clause_that_needed_a_separation_is_gone() -> None:
    s = narrator.SYSTEM
    assert "A PART LEAVING ITS WHOLE" not in s
    assert "leaves what holds them together" not in s


def test_the_clause_names_no_single_kind_of_thing_as_its_case() -> None:
    """Marc: no specific rule (10-13: 'We never use specific rules like that'). Several different kinds are listed as examples; no book, leaf,
    page, spell or ink is."""
    s = narrator.SYSTEM
    clause = s[s.index(CLAUSE_START): s.index("8. ACTION is what the actor ATTEMPTS")].lower()
    for specific in ("vellum", "leaf", "leaves", "page", "spell", "ink", "book", "jakal"):
        assert specific not in clause, specific


def test_the_holds_rule_still_finds_its_anchor_exactly_once() -> None:
    """SMRITILILA_HOLDS puts its sentence after the part-and-whole clause, found by its last words; the new clause keeps them."""
    s = narrator.SYSTEM
    assert s.count(narrator.HOLDS_ANCHOR) == 1
    once = narrator.holds_rules(s)
    assert once.count(narrator.HOLDS_RULE) == 1 and once.index(narrator.HOLDS_RULE) > s.index(CLAUSE_START)
