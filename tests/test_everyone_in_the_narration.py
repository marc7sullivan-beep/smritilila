# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""EVERYONE PRESENT IS IN THE NARRATION (Marc, 2026-11-17: "when multiple named characters are in a scene they aren't all named. Let's put in a
narrator rule that everyone in a scene gets named in the narration in some way" -- "Named or described at least").

Rule 13 already says the narration knows only what its reader knows, so "named" cannot mean the engine's name for someone the player has
not met: the rule is that each person present appears at least once, by name where the reader knows it and otherwise as the reader would see
them. Read off 3,450 recorded the second model calls: in the Welcome Wench common room (Kobort, Zert, Turuko, Maldun, Ostler, Jakal) one telling named
Jakal and nobody else of six. Replayed on the second model, the recorded prompt and the prompt with this rule, two runs each on twelve pulses
(nine crowded rooms that left people out, three small scenes as controls): 40 of 96 present people mentioned (42%) before, 72 of 96 (75%) after,
by a keyword count that scores "the two mercenaries" as a miss, so a floor; mean narration 144 words before, 137 after; the three controls
unchanged; one run in 24 named someone the actor's memory does not carry (none before), which smrti.narration_named_unknown counts live."""

from __future__ import annotations

import pytest

from smritilila.smrti import narrator

CLAUSE_START = "EVERYONE PRESENT IS IN THE NARRATION."
KNOWS_ONLY = "THE NARRATION KNOWS ONLY WHAT ITS READER KNOWS."


def _clause() -> str:
    s = narrator.SYSTEM
    assert s.count(CLAUSE_START) == 1
    return s[s.index(CLAUSE_START): s.index("14. SUGGESTIONS AND SARA.")]


def test_rule_13_says_each_person_present_is_named_or_described_at_least_once() -> None:
    clause = _clause()
    assert "appears in the narration at least once" in clause
    assert "by name where the reader knows the name" in clause
    assert "otherwise as the reader would see them" in clause


def test_it_follows_what_the_reader_knows_so_a_name_is_never_given_to_a_reader_who_lacks_it() -> None:
    s = narrator.SYSTEM
    assert s.index(KNOWS_ONLY) < s.index(CLAUSE_START)
    assert "(the rule above)" in _clause()


def test_a_model_with_its_own_lengths_gets_the_clause_with_its_own_length_not_the_shared_one() -> None:
    """providers' swaps rewrite "80 to 150 words" across the contract; the clause names the length, so it follows (the replay's clause said 80 to
    150 beside the recorded 130 to 170 of the model it ran on, which is why its 137-word mean is probably a little low). The test names no
    provider in its identifiers: the release export rewrites vendor words everywhere, and a function name is not a string; a build with no
    provider that has its own lengths (the release has none) skips it."""
    variants = narrator.system_for().variants
    if "the second model" not in variants:
        pytest.skip("no provider with its own lengths in this build")
    variant = variants["the second model"]
    clause = variant[variant.index(CLAUSE_START): variant.index("14. SUGGESTIONS AND SARA.")]
    assert "the narration stays 130 to 170 words" in clause and "80 to 150" not in clause
    assert "the narration stays 80 to 150 words" in _clause()   # the shared text, for a model with no variant


def test_it_keeps_the_length_and_the_players_and_asks_nothing_of_things() -> None:
    clause = _clause()
    assert "the narration stays 80 to 150 words" in clause
    assert "Several may share a phrase" in clause
    assert "rule 10 still holds: present, never voiced" in clause
    assert "Things need no mention" in clause
