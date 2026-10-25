# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""llm/handles.py -- readable handles at the model boundary."""

from __future__ import annotations

from smritilila.llm.handles import HandleMap, ids_in, slug_for

DOC = "fcce8473ae0840b6bdd0f580c912f109"
COURT = "bd31df41228a43e68768a584d6c99dbc"
CLERK1 = "a4393ffa20804042b118aceec7b5c62b"
CLERK2 = "e9d58d7d00000000000000000000aaaa"
LABELS = {DOC: ["John Henry 'Doc' Holliday"], COURT: ["the garden court"], CLERK1: ["the clerk"], CLERK2: ["the clerk"]}


def test_slugs_read_like_the_label_without_articles() -> None:
    assert slug_for("the garden court") == "garden-court"
    assert slug_for("John Henry 'Doc' Holliday") == "john-henry-doc-holliday"
    assert slug_for("a lean, hollow-cheeked fellow in a good coat") == "lean-hollow-cheeked-fellow-in-a-good"  # cut at a word


def test_encode_replaces_every_id_and_collisions_get_a_number() -> None:
    m = HandleMap.build([DOC, COURT, CLERK1, CLERK2], LABELS)
    assert m.id_to_handle[CLERK1] == "clerk" and m.id_to_handle[CLERK2] == "clerk-2"
    text = f"- {DOC} (tier=PRIMARY)\n{COURT} holds {CLERK2}"
    assert m.encode_text(text) == "- john-henry-doc-holliday (tier=PRIMARY)\ngarden-court holds clerk-2"


def test_ids_in_finds_hex_ids_in_order() -> None:
    assert ids_in(f"a {DOC} b", f"{COURT}") == [DOC, COURT]


# 2026-10-02: one handle rule for every id, whatever its era (Marc: "32 hex IDs are known seams. Let's close them.")
GUARD, GOBLIN1, GOBLIN2, SEED = "gate_man_at_arms_1f7c3b", "goblin_scout_aaaaaa", "goblin_scout_bbbbbb", "the_crossing_v4seed"
ALL = {**LABELS, GUARD: ["Oswin Hale"], GOBLIN1: ["a goblin scout"], GOBLIN2: ["a goblin scout"], SEED: ["the crossing"],
       "west": ["the west landing"], "east": ["the east landing"]}


def test_a_minted_id_is_shown_by_its_own_words_never_by_a_declared_name() -> None:
    m = HandleMap.build([GUARD, SEED, DOC], ALL)
    assert m.id_to_handle[GUARD] == "gate-man-at-arms"  # not "oswin-hale": the actor may never have heard the name
    assert m.id_to_handle[SEED] == "crossing-v4seed"    # a hand-made id: its own words too
    assert m.id_to_handle[DOC] == "john-henry-doc-holliday"  # 32 hex has no words of its own: its first label


def test_minted_ids_that_share_words_get_a_number_and_decode_apart() -> None:
    m = HandleMap.build([GOBLIN1, GOBLIN2], ALL)
    assert sorted(m.id_to_handle.values()) == ["goblin-scout", "goblin-scout-2"]
    assert {m.decode_id(h) for h in m.handle_to_id} == {GOBLIN1, GOBLIN2}


def test_encode_meets_every_known_id_the_text_names_and_no_tail_reaches_the_model() -> None:
    m = HandleMap.build([GUARD], ALL)
    text = m.encode_text(f"[{GUARD}, 5] I stood by {GOBLIN1}.\nNOT HERE: {DOC}, at {SEED}")
    assert text == "[gate-man-at-arms, 5] I stood by goblin-scout.\nNOT HERE: john-henry-doc-holliday, at crossing-v4seed"
    assert m.decode_id("goblin_scout") == GOBLIN1  # written with underscores, as models do


def test_a_real_id_comes_back_as_itself_shown_or_not_and_short_words_are_never_fuzzed() -> None:
    from smritilila.smrti.narrator import decode_handle
    m = HandleMap.build(["east", GUARD], ALL)
    assert decode_handle("west", m) == "west"            # the world's own id, unshown: never read as "east" garbled
    assert decode_handle(GUARD, m) == GUARD              # a raw minted id copied back is accepted
    assert decode_handle("gate-man-at-arm", m) == GUARD  # a long handle garbled by a letter is still its id
    assert decode_handle("wast", m) == "wast"            # a short word is never fuzzed onto another


def test_a_handle_written_into_prose_becomes_plain_words() -> None:
    m = HandleMap.build([GUARD, GOBLIN1, GOBLIN2, "east"], ALL)
    text, n = m.plain("You nod to gate-man-at-arms; goblin-scout-2 watches from the east. A well-worn path.")
    assert text == "You nod to gate man at arms; goblin scout watches from the east. A well-worn path." and n == 2
