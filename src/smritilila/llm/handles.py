# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Readable handles at the model boundary.
Its name: saṃjñā (संज्ञा), a conventional designation (v5 §16).

Marc, 2026-09-17: "it would be better not to use confusing numeric IDs.
LLMs have difficulty with those versus a meaningful text description."
The evidence agreed: the Narrator reused Doc Holliday's hex id for any
nameless figure, wrote "94e5a133... (Temple of Vesta)" on the Vestal
Virgins' line, put on the presiding magistrate as a toga. Thirty-two
hex characters are noise to a language model; it binds by position.

The engine keeps its ids. At the boundary, every id the prompt carries
is replaced by a HANDLE -- `gate-man-at-arms`, `west-landing`,
`goblin-scout-2` on collision -- and every handle the model returns is
replaced by its id before anything downstream sees it. A handle the
model invents is, by construction, not in the map: it stays as text,
and the engine's existing "an id it has no record of" path treats it
as NEW -- an invention made detectable instead of a misbinding. Ledger
handles (r1, r2, ...) are untouched; a raw id in the output is still
accepted. The map is per pulse, bounded by the ids the prompt names.

One rule for every id, whatever its era (Marc, 2026-10-02: "32 hex IDs
are known seams. Let's close them."). Until then only 32-hex ids had
handles; a minted id (`gate_man_at_arms_1f7c3b`) went to the model raw,
and its six-hex tail is the part models garble -- Stonemarch 09-26: the
Narrator meant moathouse_courtyard_b5e34e, wrote
NEW:moathouse_courtyard_4b9a11, and the world got a second courtyard.
A handle is the id's OWN words, its tail dropped; only an id with no
words of its own (32 hex) takes its first label. Never the label for
all (option b, refused): a label can be a declared name the actor has
never heard -- the gate guard minted as gate_man_at_arms is labelled
"Oswin Hale" -- and the prompt shows names only where they are known.
"""

from __future__ import annotations

import re

from smritilila.text import ARTICLES
from smritilila.text import slug as _slug

_HEX_ID_RE = re.compile(r"\b[0-9a-f]{32}\b")
_MINTED_TAIL_RE = re.compile(r"_[0-9a-f]{6}$")  # pulse.py's mnemonic_id: words plus a short hex tail
_LEDGER_HANDLE_RE = re.compile(r"^r\d+$")
_WORD_RUN_RE = re.compile(r"\w+")  # an id of any shape is one run of word characters
_COLLISION_RE = re.compile(r"-\d+$")
MAX_SLUG_LEN = 40


def slug_for(label: str) -> str:
    text = label.strip().lower()
    for article in ARTICLES:
        if text.startswith(article + " "):
            text = text[len(article) + 1:]
            break
    slug = _slug(text)  # v5 §7: letters of any script; was [a-z0-9]
    if len(slug) > MAX_SLUG_LEN:
        cut = slug[:MAX_SLUG_LEN]
        slug = cut[:cut.rfind("-")] if "-" in cut else cut  # cut at a word, never mid-word -- but for a single word longer than MAX_SLUG_LEN
    return slug or "thing"


def words_of(entity_id: str, labels: list[str]) -> str | None:
    """What a handle is made from: the id's own words (its minted tail
    dropped), or -- for an id with no words of its own -- its first label."""
    if _HEX_ID_RE.fullmatch(entity_id):
        return labels[0] if labels else None
    return _MINTED_TAIL_RE.sub("", entity_id).replace("_", " ")


class HandleMap:
    """id <-> handle for one pulse. Build from the ids a prompt names and
    the labels the engine knows; encode the prompt text (any known id the
    text names joins the map); decode the model's output."""

    def __init__(self, labels_by_id: dict[str, list[str]] | None = None) -> None:
        self.id_to_handle: dict[str, str] = {}
        self.handle_to_id: dict[str, str] = {}
        self._labels = labels_by_id or {}

    @classmethod
    def build(cls, ids: list[str], labels_by_id: dict[str, list[str]]) -> "HandleMap":
        m = cls(labels_by_id)
        for entity_id in sorted(set(ids)):
            m.add(entity_id)
        return m

    def add(self, entity_id: str) -> None:
        if entity_id in self.id_to_handle:
            return
        words = words_of(entity_id, self._labels.get(entity_id) or [])
        if words is None:
            return  # an id with no words and no label stays an id
        base = slug_for(words)
        handle, n = base, 1
        while handle in self.handle_to_id and self.handle_to_id[handle] != entity_id:
            n += 1
            handle = f"{base}-{n}"
        self.id_to_handle[entity_id] = handle
        self.handle_to_id[handle] = entity_id

    def encode_text(self, text: str) -> str:
        """Every known id in the text, as its handle. An id the text names
        that the map has not met joins it first (the rerun, moment and time
        lines name ids beyond the shown set) -- in sorted order, so a
        collision falls the same way for the same prompt."""
        for entity_id in sorted(ids_in(text, known=self._labels)):
            self.add(entity_id)
        if not self.id_to_handle:
            return text
        return _WORD_RUN_RE.sub(lambda mt: self.id_to_handle.get(mt.group(0), mt.group(0)), text)

    def decode_id(self, value: str | None) -> str | None:
        """A single id-shaped field from the output: a handle -> its id;
        a raw id or a ledger handle unchanged; anything else (an invented
        handle) unchanged too, for the engine's NEW path."""
        if value is None:
            return None
        v = value.strip()
        if v in self.id_to_handle or v in self._labels or _HEX_ID_RE.fullmatch(v) or _LEDGER_HANDLE_RE.match(v):
            return v  # any id the world has, shown or not -- never read as a garbled handle
        return self.handle_to_id.get(handle_key(v), value)

    def plain(self, text: str) -> tuple[str, int]:
        """A handle the model wrote into prose, as plain words
        (`gate-man-at-arms` -> `gate man at arms`); a one-word handle is a
        word already. (text, how many replaced)"""
        if not text or not self.handle_to_id:
            return text, 0
        count = 0

        def words(mt: re.Match) -> str:
            nonlocal count
            h = mt.group(0)
            if "-" not in h or h not in self.handle_to_id:
                return h
            count += 1
            return _COLLISION_RE.sub("", h).replace("-", " ")

        return re.sub(r"(?<![\w-])[\w-]+(?![\w-])", words, text), count


def handle_key(value: str) -> str:
    """A handle as the map keys it: lower case, hyphens for spaces and underscores."""
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def ids_in(*texts: str, known: dict | set | None = None) -> list[str]:
    """The ids the texts name, in order: with `known`, any id of it (every
    shape); without, the 32-hex ones."""
    found: list[str] = []
    for text in texts:
        if known is None:
            found.extend(_HEX_ID_RE.findall(text))
        else:
            found.extend(w for w in _WORD_RUN_RE.findall(text) if w in known)
    return found
