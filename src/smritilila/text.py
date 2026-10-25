# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Text, without a language (Design Principles v5 §7; 2026-09-23).

The ontology audit found English and Western orthography enforced in the
engine: names were capitalised words, a memory had 24 Latin letters,
quotes were ASCII or curly English, a sentence ended at `.!?`, a handle
was `[a-z0-9]`. Marc's call: strip what can be stripped, and make the
rest Unicode -- no per-world language profile. Everything here reads the
SCRIPT of a string, never its language: which characters are letters,
which scripts do not put spaces between words, which marks quote and
end. Nothing here knows a word of any language, but the articles ARTICLES names.
"""
from __future__ import annotations

import re
import unicodedata

# Scripts written without spaces between words: a name inside a sentence
# has no word boundary around it, and one character carries what several
# Latin letters do.
_UNSEGMENTED = (
    (0x3040, 0x30FF),    # Hiragana, Katakana
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F),  # CJK Extensions B-F and supplements
    (0x0E00, 0x0EFF),    # Thai, Lao
    (0x1000, 0x109F),    # Myanmar
    (0x1780, 0x17FF),    # Khmer
    (0x0F00, 0x0FFF),    # Tibetan
)

# How many Latin-letter equivalents one unsegmented character counts for,
# when a floor on a memory's length is measured.
DENSE_WEIGHT = 3

QUOTES = "\"'“”‘’«»„‚‹›「」『』＂＇"
OPEN_QUOTES = "\"'“‘«„‚‹「『"
CLOSE_QUOTES = "\"'”’»“‘›」』"
SPACED_ENDS = ".!?"
DENSE_ENDS = "。！？"  # 。！？


def is_dense(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _UNSEGMENTED)


def unsegmented(text: str) -> bool:
    """Whether the text is in a script that does not space its words."""
    if text.isascii():
        return False  # every unsegmented range lies far above ASCII: nothing to look at
    return any(is_dense(ch) for ch in text)


def weight(text: str) -> int:
    """A length that means about the same in every script: letters count
    one, an unsegmented character DENSE_WEIGHT."""
    t = text.strip()
    if t.isascii():
        return len(t)  # no unsegmented character in ASCII: each counts one
    return sum(DENSE_WEIGHT if is_dense(ch) else 1 for ch in t)


def sentences(text: str) -> list[str]:
    """Split at sentence ends in any script: `.!?` followed by a space,
    `。！？` followed by anything."""
    parts = re.split(rf"(?<=[{re.escape(SPACED_ENDS)}])\s+|(?<=[{DENSE_ENDS}])\s*", text.strip())
    return [p for p in parts if p]


def slug(text: str, sep: str = "-") -> str:
    """A handle from a label in any script: letters and digits kept as they
    are (lower-cased where the script has case), everything else a
    separator. Unicode-normalised so the same label gives the same slug."""
    t = unicodedata.normalize("NFC", text.strip().lower())
    return re.sub(r"[\W_]+", sep, t).strip(sep)


def tokens(text: str) -> set[str]:
    """The words of a text in any script: runs of letters and digits,
    case-folded; in an unsegmented script, every pair of adjacent
    characters (a word there is most often two)."""
    out: set[str] = set()
    for run in re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold()):
        if unsegmented(run):
            out |= {run[i:i + 2] for i in range(len(run) - 1)} or {run}
        else:
            out.add(run)
    return out


def rare_words(labels: list[str], share: float = 0.05, floor: int = 1) -> set[str]:
    """The words of a world's labels that are rare among them -- a word in
    more than `share` of the labels (and at least `floor` of them) is too
    common to identify anything ("the", "of", "de", "的"). Measured from
    the world itself, so no stopword list in any language is needed."""
    from collections import Counter
    counts = Counter(w for label in labels for w in tokens(label))
    limit = max(floor, int(share * len(labels)))
    # under three letters identifies no one in any script (weight: script-weighted length)
    return {w for w, n in counts.items() if n <= limit and weight(w) >= 3}


def units(text: str) -> list[str]:
    """Text as an ordered run of units: words where a script spaces them,
    single characters where it does not."""
    out: list[str] = []
    for run in re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold()):
        out += list(run) if unsegmented(run) else [run]
    return out


def runs(text: str, n: int = 4) -> set[tuple[str, ...]]:
    """Every run of `n` consecutive units: what the echo detectors compare."""
    u = units(text)
    return {tuple(u[i:i + n]) for i in range(len(u) - n + 1)}


# English's articles, where a leading one is set aside from a name or a
# label. The one word list here, kept in one place so it can be seen.
ARTICLES = frozenset({"the", "a", "an"})


def shared_run(a: str, b: str, words: int = 8, dense: int = 14) -> str | None:
    """The first verbatim run the two texts share -- `words` words where
    words are spaced, `dense` characters where they are not -- or None.
    Text compared, never meaning (v5 §1): the memory-leak detector's test."""
    ua, ub = units(a), units(b)
    n = dense if (ua and all(len(u) == 1 and is_dense(u) for u in ua[:3])) else words
    if len(ua) < n or len(ub) < n:
        return None
    grams = {tuple(ub[i:i + n]) for i in range(len(ub) - n + 1)}
    for i in range(len(ua) - n + 1):
        g = tuple(ua[i:i + n])
        if g in grams:
            return ("" if is_dense(g[0][0]) else " ").join(g)
    return None
