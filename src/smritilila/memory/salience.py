# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The salience scale, one place (a light module: no LLM client rides along).

The Narrator chooses a word; the store keeps the number (structure -- retrieval weights it, item 8 floors on it).
1 is the most salient, 7 the least: an indelible memory is 1, a habitual one 7. 2026-09-19: bare integers had
collapsed to the middle, so a perceiver is shown and gives the word. Pratyabhijna (2026-10-06) reads and writes the
same scale instead of keeping a second, reversed, of its own.
"""
from __future__ import annotations

SALIENCE_WORDS = {"indelible": 1, "marking": 2, "weighty": 3, "notable": 4, "ordinary": 5, "passing": 6, "habitual": 7}
SALIENCE_NAMES = {v: k for k, v in SALIENCE_WORDS.items()}


def salience_of(value: object, default: int = 4) -> int:
    """A word from the scale, or an integer 1-7 (older rows, tests), to the
    stored number; anything else is the default."""
    if isinstance(value, str):
        w = value.strip().lower()
        if w in SALIENCE_WORDS:
            return SALIENCE_WORDS[w]
        if w.isdigit():
            value = int(w)
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return min(7, max(1, value))
