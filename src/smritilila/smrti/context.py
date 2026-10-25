# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""v5 §26 retrieval over tuples. Pure: no store, no graph.

Context for a pulse = the actor's last tuple; the place's first tuple
(what it is called) and last-k; the first and last tuple of every
perceiver in attention (the first binds the id to who they are); the last tuple of every perceiver or place whose NAME the
actor already carries (in the action or in their own retrieved tuples --
paricaya as retrieval, s7 item 5, added 2026-09-18 after the canon-less
replay minted seven duplicates that were all "name in the actor's prose,
no id supplied"); a similarity fill (MMR) of K from the actor's stream
and the place's stream; the first tuple of every place the actor has
lately been in (item 6). Items 1-3, 5 and 6 are structural -- "who is
here", "what did I just do", "who do I mean" and "where have I been" are
never a retrieval gamble.

A place has a stream of its own, its scene tuples; `by_place` also
gathers every perceiver's tuples filed at it. (Written in phase 1, when
v2 data gave places no streams and this was the only way to have one.)
"""
from __future__ import annotations


import bisect
import copy
from collections import defaultdict
from dataclasses import dataclass, field

import re

import numpy as np

from smritilila.smrti.kosha import NOWHERE, whole_word  # noqa: F401 -- NOWHERE re-exported: log, pulse and main take it from here
from smritilila.text import rare_words, tokens

RECENT_PLACES = 20  # item 6: how far back "lately" reaches, in the actor's own tuples
OWN_SALIENT = 6  # item 8: how many of the actor's own most salient tuples are always present
OCCUPANTS_PER_PLACE = 4  # item 6: how many of a recent place's present occupants ride along
WEIGHT_FOLLOW_UP = 2  # item 7 for a tuple retrieved for its weight: how far what came of it is followed (3 -> 2, 2026-09-21: +13% context on an s3-heavy stream, and the third beat is mostly the next question)
PRESENT_SALIENT = 2  # item 8 for those present: what each of them cannot forget, since the Narrator voices them this pulse

# (label_words_of, its English stopword list and its tokenizer were here until 2026-09-30: they served
# `same_label`'s word-set fold alone, which v5 §17 retired; it folded once in all logged play, a donkey
# into "the man with the laden donkey", "man" being a stopword. A declared name folds, v5 §17.)


@dataclass(frozen=True)
class Tuple:
    perceiver: str
    seq: int
    chronology: int
    memory: str
    salience: int = 4
    place: str | None = None  # the scene the experience happened in
    embedding: tuple[float, ...] | None = None
    use: str | None = None  # v5 §3: key | useful | none; None before v4 (read as none)
    clock: int | None = None  # v5 §19: the minute its pulse ended, in a world that keeps a clock; None before it began


@dataclass
class Smrti:
    """The log, indexed three ways. Append-only by construction. labels
    maps a perceiver/place to the head of its FIRST tuple -- what a thing
    is called, seeded at genesis or written at first appearance; a name
    the actor carries finds its id through the containment index
    (Kosha.names_in)."""
    tuples: list[Tuple] = field(default_factory=list)
    by_perceiver: dict[str, list[Tuple]] = field(default_factory=lambda: defaultdict(list))
    by_place: dict[str, list[Tuple]] = field(default_factory=lambda: defaultdict(list))
    labels: dict[str, str] = field(default_factory=dict)  # perceiver -> its first tuple's head (what it is called)
    # v4: the containment index (smrti/kosha.py), set by build_log; None
    # for a hand-built log, which then reads places from tuples as v3 did
    kosha: object | None = None
    acquainted: dict[str, set[str]] = field(default_factory=dict)  # perceiver -> who their own tuples have been about (v5 §17)

    # Per-stream indexes (2026-09-25), kept as tuples are appended, so retrieval
    # reads a stream's weightiest, its keys, a tuple's place in it and the last
    # visit to a scene without a pass over the stream. Each is used only for a
    # stream read whole (stream() returned it as it is); a stream cut at an
    # earlier `before` is read by the passes, as before.
    _latest: dict = field(default_factory=dict)      # stream key -> its latest chronology
    _pos: dict = field(default_factory=lambda: defaultdict(dict))   # perceiver -> {seq: index}
    _ranked: dict = field(default_factory=dict)      # (perceiver, ranking) -> ([sort keys], [tuples]), the few first
    _keys: dict = field(default_factory=lambda: defaultdict(list))  # perceiver -> its use:key tuples, in order
    _runs: dict = field(default_factory=lambda: defaultdict(list))  # perceiver -> [[place, first, last index]]
    _runs_at: dict = field(default_factory=lambda: defaultdict(list))  # (perceiver, place) -> indexes into _runs
    # VERSIONS (2026-09-25): the log is kept on the world and read on (smrti/log.extend_log), not
    # built from history each pulse. A version shares every stream with the one before it until
    # it appends to that stream (copy-on-write, _own); a pulse reading an older one in its thread
    # keeps it. `tuples` alone is shared and appended to: nothing reads it but tests.
    _owned: object = None                              # None: this version owns everything (built fresh)
    places: set = field(default_factory=set)            # the scenes (and v3 places) the log seeded
    _seq: int = 0                                       # the last seq given
    _n_canon: int = 0                                   # how many of the canon's entities are seeded
    _pulses: set = field(default_factory=set)           # every pulse read (shared by versions: only ever added to, only asked)
    # Each stream's embeddings as rows of one matrix, grown as tuples come (the similarity search
    # reads rows, never converts a tuple again): stream key -> [rows, has an embedding, perceiver].
    # Shared by versions and grown in place: a version reads only as many rows as its own stream
    # holds, and a grown buffer keeps every earlier row.
    _mat: dict = field(default_factory=dict)
    _ppos: dict = field(default_factory=lambda: defaultdict(dict))  # place -> {(perceiver, seq): index} (shared, only added to)
    # v5.33 (Marc, 2026-10-13): which experience closes which, as the Narrator wrote it ("closes" records)
    _closes: dict = field(default_factory=dict)      # (perceiver, closer tick) -> the head ticks of that perceiver it closes
    _closed_by: dict = field(default_factory=dict)   # (perceiver, head tick) -> the later ticks that close it
    _at: dict = field(default_factory=dict)          # (perceiver, tick) -> that perceiver's tuples at that tick

    def _own(self, attr: str, key, copier):
        d = getattr(self, attr)
        if self._owned is not None and (attr, key) not in self._owned:
            self._owned.add((attr, key))
            cur = d.get(key)
            d[key] = copier(cur) if cur is not None else copier(None)
        elif key not in d:
            d[key] = copier(None)
        return d[key]

    def clone(self) -> "Smrti":
        """Another version, sharing every stream until it appends to one.
        Cost: the tables' keys (the world's perceivers and places)."""
        c = copy.copy(self)
        for attr in ("by_perceiver", "by_place", "_keys", "_runs", "_runs_at", "_closes", "_closed_by", "_at"):
            d = getattr(self, attr)
            setattr(c, attr, defaultdict(d.default_factory, d) if isinstance(d, defaultdict) else dict(d))
        for attr in ("labels", "acquainted", "_latest", "_ranked", "_mat"):
            setattr(c, attr, dict(getattr(self, attr)))
        c.places = set(self.places)
        c._owned = set()
        return c

    def append(self, t: Tuple) -> None:
        self.tuples.append(t)
        first = not self.by_perceiver.get(t.perceiver)
        s = self._own("by_perceiver", t.perceiver, lambda x: list(x or ()))
        s.append(t)
        i = len(s) - 1
        self._own("_at", (t.perceiver, t.chronology), lambda x: list(x or ())).append(t)
        self._latest[("p", t.perceiver)] = max(t.chronology, self._latest.get(("p", t.perceiver), t.chronology))
        self._pos[t.perceiver][t.seq] = i  # shared by versions: keys are never reused, and only looked up
        self._row(("p", t.perceiver), i, t)
        for ranking, key, k in (("salient", (t.salience, -t.chronology), OWN_SALIENT + 1),
                                ("heavy", (heavier(t), -t.chronology), PRESENT_SALIENT + 1)):
            keys, items = self._own("_ranked", (t.perceiver, ranking), lambda x: (list(x[0]), list(x[1])) if x else ([], []))
            j = bisect.bisect_right(keys, key)  # after its equals: stream order, as a stable sort keeps
            if j < k:
                keys.insert(j, key)
                items.insert(j, t)
                del keys[k:], items[k:]
        if t.use == "key":
            self._own("_keys", t.perceiver, lambda x: list(x or ())).append(t)
        runs = self._own("_runs", t.perceiver, lambda x: list(x or ()))
        if runs and runs[-1][0] == t.place:
            runs[-1] = [t.place, runs[-1][1], i]  # a new list: an older version keeps its own
        else:
            runs.append([t.place, i, i])
            self._own("_runs_at", (t.perceiver, t.place), lambda x: list(x or ())).append(len(runs) - 1)
        if t.place:
            ps = self._own("by_place", t.place, lambda x: list(x or ()))
            ps.append(t)
            self._ppos[t.place][(t.perceiver, t.seq)] = len(ps) - 1
            self._row(("pl", t.place), len(ps) - 1, t)
            self._latest[("pl", t.place)] = max(t.chronology, self._latest.get(("pl", t.place), t.chronology))
        if first:  # the name is the label, not what it lies within
            self.labels[t.perceiver] = t.memory.split(" -- ")[0][:80]

    def close(self, perceiver: str, tick: int, head: int) -> None:
        """v5.33: `perceiver`'s experience at `tick` closes their own at `head` (< tick)."""
        if head >= tick:
            return
        a = self._own("_closes", (perceiver, tick), lambda x: list(x or ()))
        if head not in a:
            a.append(head)
        b = self._own("_closed_by", (perceiver, head), lambda x: list(x or ()))
        if tick not in b:
            b.append(tick)

    def stream(self, perceiver: str, before: int | None = None) -> list[Tuple]:
        """A perceiver's tuples before `before`. Nothing in it that late: the
        stream itself, not a copy -- read-only to every caller."""
        s = self.by_perceiver.get(perceiver, [])
        if before is None or not s or before > self._latest.get(("p", perceiver), before):
            return s
        return [t for t in s if t.chronology < before]

    def place_stream(self, place: str, before: int | None = None) -> list[Tuple]:
        s = self.by_place.get(place, [])
        if before is None or not s or before > self._latest.get(("pl", place), before):
            return s
        return [t for t in s if t.chronology < before]

    def _row(self, key: tuple, i: int, t: Tuple) -> None:
        """Row i of the stream's matrix: t's embedding (none: marked absent)."""
        h = self._mat.get(key)
        if h is None:
            if t.embedding is None:
                return
            cap = max(64, 2 * (i + 1))
            h = [np.zeros((cap, len(t.embedding))), np.zeros(cap, dtype=bool), np.empty(cap, dtype=object)]
            self._mat[key] = h
        buf, has, who = h
        if t.embedding is not None and len(t.embedding) != buf.shape[1]:
            self._mat[key] = None  # a stream of mixed sizes: no matrix, the search converts as before
            return
        if i >= buf.shape[0]:
            cap = max(2 * buf.shape[0], i + 1)
            nb, nh, nw = np.zeros((cap, buf.shape[1])), np.zeros(cap, dtype=bool), np.empty(cap, dtype=object)
            nb[:buf.shape[0]], nh[:has.shape[0]], nw[:who.shape[0]] = buf, has, who
            buf, has, who = nb, nh, nw
            h[0], h[1], h[2] = buf, has, who  # earlier rows kept: an older version reads the new buffer as well
        if t.embedding is not None:
            buf[i] = t.embedding
            has[i] = True
        who[i] = t.perceiver

    def matrix(self, key: tuple, n: int):
        """The first n rows of a stream's matrix -- (embeddings, has one,
        perceivers) -- or None: none kept."""
        h = self._mat.get(key)
        if h is None:
            return None
        buf, has, who = h
        return buf[:n], has[:n], who[:n]

    def whole(self, perceiver: str, s: list[Tuple]) -> bool:
        """Whether `s` is the perceiver's stream read whole, so its indexes hold."""
        return s is self.by_perceiver.get(perceiver)

    def ranked(self, perceiver: str, ranking: str, k: int) -> list[Tuple]:
        """sorted(stream[:-1], key=the ranking)[:k], for a stream read whole."""
        last = self.by_perceiver[perceiver][-1]
        return [t for t in self._ranked.get((perceiver, ranking), ([], []))[1] if t is not last][:k]

    def keys_of(self, perceiver: str) -> list[Tuple]:
        """The use:key tuples of stream[:-1], in order, for a stream read whole."""
        last = self.by_perceiver[perceiver][-1]
        return [t for t in self._keys.get(perceiver, []) if t is not last]

    def last_here(self, perceiver: str, place: str) -> Tuple | None:
        """The last tuple at `place` before this visit (and before whatever
        came between), for a stream read whole: LAST_HERE's walk, by the runs."""
        runs, at = self._runs.get(perceiver, []), self._runs_at.get((perceiver, place), [])
        if not at:
            return None
        current = len(runs) - 1 if runs and runs[-1][0] == place else None
        before = [r for r in at[-2:] if r != current]
        if not before:
            return None
        return self.by_perceiver[perceiver][runs[before[-1]][2]]


USE_RUNG = {"key": 2, "useful": 4}  # v5 §26: in item 8, key counts as marking and useful as notable
KEY_ALLOWANCE = 4  # v5 §26: item 8's keys, chosen by relevance to the action (settled by replay 2026-09-23)


def use_factor(t: Tuple) -> float:
    return (1.0 if t.use in ("key", "useful") else 0.8) * (0.6 if t.salience == 7 else 1.0)


def heavier(t: Tuple) -> int:
    """The heavier of a tuple's two words as one rung (v5 §26)."""
    return min(t.salience, USE_RUNG.get(t.use or "", 9))


def _relevance(t: Tuple, q: np.ndarray | None) -> float:
    if q is None or t.embedding is None:
        return 0.0
    e = np.asarray(t.embedding)
    return float(e @ q / ((np.linalg.norm(e) + 1e-12) * (np.linalg.norm(q) + 1e-12)))


def _mmr(query: np.ndarray, pool: list[Tuple], k: int, lam: float = 0.7, E: np.ndarray | None = None) -> list[Tuple]:
    """`E`: the pool's embeddings as rows, already (every tuple in the pool has one)."""
    if E is None:
        pool = [t for t in pool if t.embedding is not None]
    if not pool or k <= 0:
        return []
    if E is None:
        E = np.asarray([t.embedding for t in pool])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    q = query / (np.linalg.norm(query) + 1e-12)
    sim = E @ q
    order = np.argsort(-sim)[: max(3 * k, k)]
    # v5 §26: relevance multiplied by USE, and by weight only to lower it --
    # nothing is boosted for being dramatic; the lightest weight (habitual,
    # 7) is discounted, as what the perceiver would not recall (use_factor).
    # Weight had been a second multiplier on every score, counting drama
    # twice (item 8 already carries it).
    sal = np.asarray([use_factor(pool[i]) for i in order])
    rel = sim[order] * sal
    Es = E[order]
    sel: list[int] = []
    cand = list(range(len(order)))
    while cand and len(sel) < k:
        if sel:
            red = (Es[cand] @ Es[sel].T).max(axis=1)
            score = lam * rel[cand] - (1 - lam) * red
        else:
            score = rel[cand]
        b = cand[int(np.argmax(score))]
        sel.append(b)
        cand.remove(b)
    return [pool[order[i]] for i in sel]


@dataclass
class Context:
    actor: str
    place: str | None
    tuples: list[Tuple]  # ordered by chronology
    present: set[str]    # perceivers whose own containment puts them in the scene (v5 §11)
    # which retrieval rule first supplied each tuple (2026-09-21: cuts are
    # made on measurement -- pulse.py logs the block by source)
    sources: dict[tuple[str, int], str] = field(default_factory=dict)
    neighbours: list[str] = field(default_factory=list)  # v5 §12: scenes one step away in the tree
    chain: list[str] = field(default_factory=list)       # v5 §26: what the actor is in, up to its scene
    above: list[str] = field(default_factory=list)       # v5 §26: what holds the scene, nearest first, to the root
    held: list[tuple[str, str, str]] = field(default_factory=list)  # v5 §26: (id, name, container) the scene's holder holds
    tree: list[tuple[str, str, list[tuple[str, str]]]] = field(default_factory=list)  # v4.30: (holder, name, [(place, name)]), top first
    quiet: set[str] = field(default_factory=set)        # present, past the participation limit: named, not remembered
    # what holds each one present, where that is the actor or another present, and what the actor holds -- so PRESENT
    # is shown by what holds what (the sample world, 2026-09-30: flat, the Narrator never saw what anyone carried)
    holder: dict[str, str] = field(default_factory=dict)
    # SMRITILILA_HOLDS (Marc, 2026-10-13: everything is a container): what each thing the actor holds holds, by label -- [] is empty; set by pulse.fill_holds
    holds: dict[str, list[str]] = field(default_factory=dict)
    # the place re-run's regions (the sample world, 2026-09-30): {region: (name, [(scene, name, what it is in or None)])} --
    # places beyond the actor's branch a first telling made beside without being shown them
    beyond: dict[str, tuple[str, list[tuple[str, str, str | None]]]] = field(default_factory=dict)
    # the region the actor last stood in, when the tree above shows the actor none of its own (Marc, 2026-10-10; Stonemarch): (region, name, moments ago, its places as `beyond`'s rows).
    # See last_region_stood.
    last_region: tuple[str, str, int, list[tuple[str, str, str | None]]] | None = None
    known: list[tuple[str, str]] = field(default_factory=list)  # (id, name): silent, acquainted, placed nowhere
    namesakes: list[tuple[str, str, str]] = field(default_factory=list)  # (id, name, part): a person a name part in the action or memory matches
    players_named: list[tuple[str, str]] = field(default_factory=list)   # (id, name): a player named so, not present
    elsewhere: list[tuple[str, str, str]] = field(default_factory=list)  # (id, name, container): acquainted, placed where this pulse does not show
    lately: list[tuple[str, str, str | None]] = field(default_factory=list)  # (id, name, scene): with the actor lately, not here now
    across: list[tuple[str, str, list[str]]] = field(default_factory=list)  # (boundary thing, the scene across it, what is there): sandhi
    # v4.60: a chain of things held one inside the next, read in order -- (head, [links read], [links not yet read]) --
    # the words of every link not yet read are in the memory above
    reading: list[tuple[str, list[str], list[str]]] = field(default_factory=list)

    def shown(self) -> set[str]:
        """Every id this context shows the Narrator -- what a pulse may bind,
        move or name (the Binder), and what gets a readable handle (the
        narrator's handle map): said once here, not copied into each."""
        return ({t.perceiver for t in self.tuples} | set(self.present) | {self.actor} | ({self.place} if self.place else set())
                | set(self.neighbours) | set(self.chain) | set(self.above) | {h[0] for h in self.held} | {k[0] for k in self.known}
                | {x for e in self.elsewhere for x in (e[0], e[2])} | {n[0] for n in self.namesakes} | {p[0] for p in self.players_named}
                | {x for face, other, there in self.across for x in (face, other, *there)}
                | {x for head, read, unread in self.reading for x in (head, *read, *unread)}
                | {x for region, (_n, rows) in self.beyond.items() for x in (region, *(r[0] for r in rows))}
                | ({self.last_region[0], *(r[0] for r in self.last_region[3])} if self.last_region else set()))


class _Sourced(dict):
    """The context under assembly: a dict of (perceiver, seq) -> Tuple
    that remembers which rule (`phase`) FIRST put each tuple in. A later
    rule re-adding a tuple changes nothing -- the first reason it was
    here is the reason it is here."""

    def __init__(self) -> None:
        super().__init__()
        self.phase = "last"
        self.source: dict[tuple[str, int], str] = {}

    def __setitem__(self, key, value) -> None:
        if key not in self:
            self.source[key] = self.phase
        super().__setitem__(key, value)

    def add(self, t: "Tuple") -> None:
        """One tuple in, under the rule now adding (`phase`)."""
        self[(t.perceiver, t.seq)] = t


PAIR_CAP = 8   # v5.33: the most tuples the head-and-tail rule adds to a pulse's memory (four pairs), whatever the situation


def head_tail_company(log: Smrti, ctx: "_Sourced", actor: str, now: int, cap: int = PAIR_CAP) -> int:
    """v5.33 (Marc, 2026-10-13, from his "Head and tail": "Engine should always retrieve the head and tail pair"; replaces v5.30's
    causal chains): a matter is a HEAD, the memory that opened it, and the memories that CLOSE it, as the Narrator wrote
    (`closes` records); its TAIL is the latest closer. Whichever of the two a pulse is shown, the pair comes: a head with its tail, a
    closer with the head (or heads) it closes and the latest closer of each. Nothing between is ever added, and nothing is followed
    further. The engine decides nothing about resolution: the Narrator says when a tail closes a head. A head nobody has closed
    is open and brings nothing. Own tuples first, newest shown first; a pair is added whole or not at all, `cap` tuples in all,
    so the cost is a constant, not the stream. Returns how many were added."""
    if not log._closes:
        return 0
    added = 0
    done: set[tuple[str, int]] = set()
    roots = sorted(ctx.values(), key=lambda t: (t.perceiver != actor, -t.chronology, -t.seq))
    for root in roots:
        who, tick = root.perceiver, root.chronology
        heads = ([tick] if log._closed_by.get((who, tick)) else []) + sorted((h for h in log._closes.get((who, tick), ()) if h < now), reverse=True)
        for head in heads:
            if (who, head) in done:
                continue
            done.add((who, head))
            closers = [k for k in log._closed_by.get((who, head), ()) if k < now]
            if not closers:
                continue
            need = [u for k in (head, max(closers)) for u in log._at.get((who, k), ()) if (u.perceiver, u.seq) not in ctx]
            if added + len(need) > cap:
                continue
            for u in need:
                ctx.add(u)
            added += len(need)
    return added


THEIR_SCENES = 3      # v5 §26 item 3: the last few scenes a perceiver in attention has been in
NEIGHBOUR_OCCUPANTS = 3  # v5 §12: who is in each neighbouring scene, first tuples
HELD = 8        # v5 §26: silent things the scene's holder holds, shown by name
HELD_DEPTH = 2  # v5 §26: how far inside the holder they may sit
# Known, and nowhere (the crossing, 523 -> 543): silent things the actor is
# acquainted with that the world has placed in nothing -- the first half of
# the blanket, named at 523 as having gone downriver, found in Cass's chest
# at 543 and minted again as "the river blanket". By declared name, newest
# first, minted within KNOWN_WINDOW ticks. 0 = off. Replayed at 543 (the larger model and
# the second model, 3 draws each): brought out, the blanket was minted again 2/2
# without it and the first half reused 4/4 with it; never misused at 547.
KNOWN = 6
KNOWN_WINDOW = 60
# Known, and elsewhere (Ravenmoor 524-582): KNOWN's twin for things placed
# SOMEWHERE this pulse does not show. Wentworth, in the yard, looked through
# a window into the reading room -- not a neighbour in the tree -- and each
# pulse saw nothing of the man the last one minted: Elias Lowell became four
# entities (524, 530, 532, 552), and the duplicate left behind took a turn
# of its own at 567. So too "the lit room on the third floor" beside the
# third-floor reading room his own pulse minted at 514, a second green-door
# boarding house (580 -> 582), "Lowell's roadster" beside the dark roadster
# (494 -> 496). By declared name and where each is, newest first, minted
# within KNOWN_WINDOW; players never (where a player is, is theirs). 0 = off.
KNOWN_ELSEWHERE = 6
# Seen, and elsewhere (Ravenmoor 669-670, Marc 2026-09-25: "solve by what is
# shown"): a perceiver whose tuples the memory shows, standing in a scene this
# pulse does not reach. Professor Hartley stood, by his last memory, in the
# Thompson house (t417); Thompson drove to Hartley's own house, the Narrator
# voiced Hartley there without moving him, and both his tuples were dropped as
# elsewhere -- he remembered nothing of what he was told. Shown where he is, a
# Narrator that has him here moves him. Those named in the action or the
# memory -- by a name found there -- those in the action first (Hartley was),
# then by their latest tuple shown; never a root (standing in no scene at all is not a
# where), never a player. (Listing every perceiver the memory shows added
# six to most contexts, mostly things: replay, 2026-09-25.)
SEEN_ELSEWHERE = 6
# With you lately, not here now (Marc, 2026-09-29: "when important people or things leave a scene or move out of a
# scene, any changes that would be noticeable should be part of the narration"). Those who shared the actor's scene
# in the actor's own last LATELY_TELLINGS tuples and are not in the scene now -- only whoever has ever kept a memory
# of their own (useful or key: by what they did, never by kind, as the draw's fold), players included (their absence
# is noticed, never their deeds). Shown, the Narrator's first field `who_stays` names them; replayed on Stonemarch
# (t494, three left in the guardroom): named 4 of 4 with the field, 0 of 4 as a rule. 0 = off.
LATELY_TELLINGS = 5
LATELY = 6
LAST_REGION_SCAN = 80   # the actor's own most recent tuples looked through for the last region stood in
LAST_REGION_ROWS = 14   # the most places of that region shown (the place rerun's PLACES_BEYOND)


def last_region_stood(log, kosha, actor: str, place: str | None, now: int, tree: list) -> tuple[str, str, int, list[tuple[str, str, str | None]]] | None:
    """The region the actor last stood in that the tree shows them none of: (region, name, moments ago, its places two deep, as `beyond`'s rows), or None.
    Found on Stonemarch, 2026-10-10 (Filonius): the city of Verbobonc was one level down in the Viscounty, the hospice he reached stood in the world at large, and
    the tree listed only the world at large, so every place made after it was filed there too -- 9 places outside a city that held 6. The place rerun asks whether a
    telling made a place in a region it was not shown, but the earth is always shown, so it never fired. Structure only: the first holder, other than the world's top,
    of the places in the actor's last 80 tuples, that the tree does not already list. Cost: at most 80 tuples and the region's places, only when the tree is flat."""
    if kosha is None or not place or not tree:
        return None
    shown = {h for h, _name, _places in tree} | {place}
    top = tree[0][0]
    seen: set[str] = set()
    for t in reversed(log.stream(actor, before=now)[-LAST_REGION_SCAN:]):
        p = t.place
        if not p or p in seen or p == place:
            continue
        seen.add(p)
        holder = kosha.container_at(p, now)
        if holder is None or holder == top or holder in shown:
            continue
        rows: list[tuple[str, str, str | None]] = []
        for x in kosha.holds(holder, now):
            if kosha.is_scene(x):
                rows.append((x, kosha.label(x), None))
                rows += [(y, kosha.label(y), x) for y in kosha.holds(x, now) if kosha.is_scene(y)]
            else:  # a building the region holds: its scenes, named as in it
                rows += [(y, kosha.label(y), x) for y in kosha.holds(x, now) if kosha.is_scene(y)]
        return holder, kosha.label(holder), now - t.chronology, rows[:LAST_REGION_ROWS]
    return None
# Namesakes (Ravenmoor playtest, Marc 2026-09-24): names match only whole and
# only among the met (s34), so a surname alone found nobody. Wentworth read
# "Lowell, E." in a register Danforth's thread wrote, asked the janitor about
# Lowell, and the Narrator -- never shown that an Edmund Lowell existed --
# made a second: Elias Lowell (554), "confusing to players". And "Professor
# Thompson" matched nothing of Nathaniel ("Asst. Prof. N. Thompson"), so other
# players' NPCs gave a player, away, a crime (516, 556, 590). A NAME PART --
# a capitalised word of a person's own name (a recorded name or the first
# label, never a description: nothing that begins with an article), titles
# and abbreviations excepted (Kosha.name_parts) -- found
# whole in the action or the actor's own memory shown brings in by name a
# person the actor has NOT met (the met are theirs already -- surveyed on 93
# player pulses, most hits were acquaintances), and marks any player so
# named, met or not, who is not present. At most NAMESAKES; 0 = off.
NAMESAKES = 4
CONTENTS_WINDOW = 12  # v5 §27: a thing in the scene rides along if moved or minted within this many ticks
CONTENTS_CAP = 6
REACHING = 5  # v5 §8: an actor whose perception reaches is shown this many distant perceivers, by similarity and weight
REACH_LATELY = 3  # ... each scored by the best of their last this-many tuples
# The participation limit (Marc, 2026-09-23 -- computational, never a
# prohibition): in a scene with more present than PARTICIPANTS, only the
# PARTICIPANTS most active bring their memories; the rest are present and
# quiet -- their first tuple (who they are) and their name, nothing more.
# Activity: own tuples in the last PARTICIPATION_WINDOW ticks, each worth
# half as much every PARTICIPATION_HALF_LIFE ticks back. The actor, the
# players and anyone the action names always take part. 0 = no limit. A
# world's pacing data may set all three (participants,
# participation_window, participation_half_life). Replayed on the most
# crowded pulses before it was switched on: the larger model wrote 21% fewer tuples,
# the quiet were still written for when touched (the sinkhole the dynamite
# went into), and none was minted again.
PARTICIPANTS = 4
# The last time here (the crossing, 594-596): for the actor and each present
# participant, the last tuple they filed at this scene before this visit.
# Tam said "the first time I've sat in this room" -- his own tuples had him
# asleep against its wall 200 ticks before, unshown -- and, challenged, a
# third past was invented over the recorded one. One tuple each. 0 = off.
# Replayed at 596 (the larger model and the second model, 3 draws each): the invented past 6/6
# without it, 0/6 with it and weighted participation (Tam was quiet under
# raw counts, and the rule never reached him).
LAST_HERE = 1
PARTICIPATION_WINDOW = 20
PARTICIPATION_HALF_LIFE = 5
# Marc's measure, refined (2026-09-23): memories that MATTER per unit time --
# each tuple counted by the heavier of its two words. Raw counts put the man
# being questioned fifth behind the furniture (the crossing, 596: every one
# present gets a tuple most pulses, the cup and the bench included);
# weighted, each person present scored over three times any thing.


def world_rare(kosha) -> set[str]:
    """The words rare among the world's labels (v5 §26) -- measured from the
    world, no stopword list. Cost: the world's labels."""
    return rare_words([e.labels[0] for e in kosha.canon.entities.values() if e.labels])


def named_by_words(kosha, text: str, among, need: int, rare: set[str] | None = None) -> set[str]:
    """v5 §26's "about them", counted: those `among` with at least `need` words
    of their first label, rare among the world's labels, standing in `text`.
    A paraphrase reaches what a whole name would (Marc, 2026-09-26:
    "paraphrase should be allowed"). Cost: the candidates' labels, and the
    world's labels once when `rare` is not given."""
    if not text or not among:
        return set()
    rare = world_rare(kosha) if rare is None else rare
    tt = tokens(text)
    out = set()
    for p in among:
        e = kosha.canon.entities.get(p)
        if e is not None and e.labels and len(tokens(e.labels[0]) & rare & tt) >= need:
            out.add(p)
    return out


# Participation's "anyone the action names" reaches a paraphrase: two words
# of the label, rare among the world's labels, or the whole name. Clara's
# run, Ravenmoor 707: "the schedule card PINNED on the door frame" named
# nothing whole, the card stayed quiet, its own text ("Room 114") unshown;
# at 709 the Narrator made the room "Frye 12" and at 710 walked her into a
# second ground-floor space in Frye. Replayed as of 707, the larger model and the second model, 3
# draws: its text shown and "114" written 6/6, against 0/6 as it ran. Two,
# not one: at one word, every action with "from" woke both of the
# Crossing's cloths (12 pulses) and "back" the back room key -- a function
# word is rare among short labels -- and participation changed on 40 of
# Ravenmoor's 270 player pulses (+4 tuples) against 11 (+1). Mauna's
# exemption keeps ONE word, on purpose: a missed exemption hides a thing
# in use ("Check my watch" -- one of "the pocket watch"'s two), a missed
# wake here only leaves a thing quiet that its whole name still reaches.
NAMED_BY_WORDS = 2


def participation(log: Smrti, present: set[str], now: int, action: str, keep: set[str], limit: int,
                  window: int | None = None, half_life: float | None = None) -> set[str]:
    """The present who stay quiet this pulse: beyond the `limit` most
    active, never one in `keep` or named in the action -- whole, or by
    NAMED_BY_WORDS words of its label. Bounded: each present perceiver's
    tuples in the window, and no more."""
    if not limit or len(present) <= limit:
        return set()
    window = PARTICIPATION_WINDOW if window is None else window
    half_life = PARTICIPATION_HALF_LIFE if half_life is None else half_life
    kosha = log.kosha
    named = (set(kosha.names_in(action, present)) | named_by_words(kosha, action, present, NAMED_BY_WORDS)
             if kosha is not None and action else set())
    always = (present & set(keep)) | named

    def active(p: str) -> float:
        score = 0.0
        for t in reversed(log.stream(p, before=now)):
            if now - t.chronology > window:
                break  # the window, never the history
            score += 0.5 ** ((now - t.chronology) / half_life) * (8 - heavier(t)) / 7
        return score

    rest = sorted(present - always, key=lambda p: (-active(p), p))
    return set(rest[limit:])


# Mauna (Marc names it, 2026-09-26): what is HELD -- by the actor, or by
# someone present the action names -- is present and quiet, its first tuple
# only, unless a word of its name is in the action or its holder's latest
# tuple. Ravenmoor, from 423: a carried thing written in its carrier's last
# pulse was written again 76% of the time, 13% otherwise -- its tuple in
# the scene's window put it in attention, rule 1 asked a tuple of every
# perceiver present, and with nothing from the present to take it copied
# its own past (the cork, the bag, the tide table: a quarter to three
# quarters of each tuple already in their own last twenty). Replayed, the larger model
# and the second model, 3 draws: momentum tuples ~1 a pulse -> ~0; things in use
# still written (the watch at 697, "time?", named nowhere: used 6/6 and
# written 6/6 shown by name only; Bren's cloth when the story reached for
# it); none minted again in 165 pulses. False = off.
MAUNA = True


def mauna(log: Smrti, actor: str, now: int, place: str, present: set[str], action: str, keep: frozenset[str]) -> set[str]:
    """The held, present and quiet this pulse (MAUNA). Held: its chain
    reaches a holder -- the actor, or someone present the action names --
    before any scene. Never the actor, a player, or anyone the action
    names. Cost: the present, each chain up to its holder, and the latest
    tuple of the actor and each holder."""
    kosha = log.kosha
    named = set(kosha.names_in(action, present)) if action else set()
    holders = ({actor} | (named & present)) - {place}
    rare: set[str] | None = None
    latest: dict[str, str] = {}

    def last_words(who: str) -> str:
        if who not in latest:
            s = log.stream(who, before=now)
            latest[who] = s[-1].memory if s else ""
        return latest[who]

    out: set[str] = set()
    for x in present:
        if x == actor or x in keep or x in named:
            continue
        holder = None
        for c in kosha.chain(x, now):
            if c == place or kosha.is_scene(c):
                break
            if c in holders and c != x:
                holder = c
                break
        if holder is None:
            continue
        text = "\n".join((action, last_words(actor), last_words(holder)))
        if kosha.names_in(text, [x]):
            continue  # its name, whole
        if rare is None:
            rare = world_rare(kosha)
        if named_by_words(kosha, text, [x], 1, rare):
            continue  # one word of its label is enough to speak (NAMED_BY_WORDS says why)
        out.add(x)
    return out


def _similar_rows(log: Smrti, actor: str, own: list[Tuple], place: str | None, pstream: list[Tuple], ctx: dict,
                  whole_own: bool) -> tuple[list[Tuple], np.ndarray] | None:
    """The similarity pool -- own, then the scene's, minus what the context
    holds and the actor's own tuples the scene's stream repeats, those with
    an embedding -- and its rows, from the streams' kept matrices. None when
    a stream is not read whole or keeps no matrix: the pool is built as
    before. Cost: numpy over the two streams, and the context's keys."""
    if not whole_own:
        return None
    mo = log.matrix(("p", actor), len(own))
    if mo is None:
        return None
    mp = None
    if pstream:
        if not place or pstream is not log.by_place.get(place):
            return None
        mp = log.matrix(("pl", place), len(pstream))
        if mp is None:
            return None
    ok_o = mo[1].copy()
    pos_o = log._pos.get(actor, {})
    for key in ctx:
        if key[0] == actor:
            i = pos_o.get(key[1])
            if i is not None and i < len(own):
                ok_o[i] = False
    idx_o = np.flatnonzero(ok_o)
    parts, pool = [mo[0][idx_o]], [own[i] for i in idx_o]
    if mp is not None:
        ok_p = mp[1] & (mp[2] != actor)  # the actor's own are in the pool already, from their own stream
        pos_p = log._ppos.get(place, {})
        for key in ctx:
            j = pos_p.get(key)
            if j is not None and j < len(pstream):
                ok_p[j] = False
        idx_p = np.flatnonzero(ok_p)
        parts.append(mp[0][idx_p])
        pool += [pstream[j] for j in idx_p]
    return pool, np.vstack(parts)


def _minted_lately(kosha, eid: str, now: int) -> int | None:
    """When eid was minted, if within KNOWN_WINDOW before now -- KNOWN's and
    KNOWN_ELSEWHERE's shared bound; else None."""
    e = kosha.canon.entities.get(eid)
    minted = e.created_at_world_time if e is not None else None
    return None if minted is None or minted >= now or now - minted > KNOWN_WINDOW else minted


CHAIN_MIN = 2  # a chain of at least this many links is read in order (a pouch with one coin is not)


def chain_head(kosha, c: str, now: int, actor: str) -> str:
    """Up from c while what holds it holds only it -- never to the actor, nor past them."""
    head, up = c, kosha.container_at(c, now)
    while up is not None and up != actor and len(kosha.holds(up, now)) == 1:
        head, up = up, kosha.container_at(up, now)
    return head


def chain_links(kosha, head: str, now: int) -> list[str]:
    """What head holds one inside the next, each holding exactly one: [] unless CHAIN_MIN long."""
    links, cur, seen = [], head, {head}
    while True:
        k = kosha.holds(cur, now)
        if len(k) != 1 or k[0] in seen:
            break
        cur = k[0]
        seen.add(cur)
        links.append(cur)
    return links if len(links) >= CHAIN_MIN else []


def read_chains(log: Smrti, actor: str, now: int, action: str, named: set[str], own: list[Tuple]) -> list[tuple[str, list[str], list[str]]]:
    """v4.60 (Marc, 2026-10-02: "a page at a time ... each page nested within the container of the book"): things held
    one inside the next -- each holding exactly one -- are read in order. A chain is read when the action names its head
    or any link, or when the actor's own last pulse touched a link (reading on: "I turn the page"). A link is read by the
    actor once a pulse of theirs wrote it a tuple past what is written on it -- the two on one tick; the rest, in order,
    are not yet read, and their words are shown. Its links are otherwise quiet, whatever else would wake them
    (retrieve). Structure only: nothing is typed a book or a page. Cost: the chains of the named and of what the actor
    holds, link by link."""
    kosha = log.kosha
    if kosha is None:
        return []
    own_ticks = {t.chronology for t in own}
    last_tick = own[-1].chronology if own else None
    heads: dict[str, list[str]] = {}
    for c in set(named) | set(kosha.holds(actor, now)):
        head = chain_head(kosha, c, now, actor)
        if head not in heads:
            links = chain_links(kosha, head, now)
            if links:
                heads[head] = links
    out = []
    for head, links in heads.items():
        def touched(link: str) -> set[int]:
            return {t.chronology for t in log.stream(link, before=now)[2:]}  # past its label and what is written on it
        read = [x for x in links if touched(x) & own_ticks]
        # held by the actor, the action names it by its own words, met or not -- ONE rare word, as mauna's exemption
        # keeps one: a missed name here leaves a page unread and the Narrator inventing it
        chain = [head, *links]
        named_here = (bool(set(chain) & named) or bool(kosha.names_in(action, chain))
                      or bool(action and named_by_words(kosha, action, chain, 1)))
        reading_on = last_tick is not None and any(last_tick in touched(x) for x in links)
        if named_here or reading_on:
            out.append((head, read, [x for x in links if x not in read]))
    return out


def retrieve(
    log: Smrti,
    actor: str,
    now: int,
    query_embedding: np.ndarray | None,
    *,
    k_place: int = 12,
    k_sim: int = 16,
    action: str = "",
    own_only: bool = False,
    keep: frozenset[str] = frozenset(),
    pacing: dict | None = None,
    sandhi: bool = False,
) -> Context | None:
    """v5 §26 retrieval, restated on scenes (v5 §26).
    Its name: smaraṇa (स्मरण), the act of remembering (v5 §26).

    own_only (itihāsa, Hillands tick 41): the world at large reads the
    bhuvana and its own stream and nothing else."""
    own = log.stream(actor, before=now)
    if not own:
        return None
    last = own[-1]
    kosha = log.kosha
    if own_only:
        ctx0 = _Sourced()
        ctx0[(last.perceiver, last.seq)] = last
        ctx0.phase = "own_salient"
        for t in sorted(own[:-1], key=lambda t: (t.salience, -t.chronology))[:OWN_SALIENT]:
            ctx0[(t.perceiver, t.seq)] = t
        ctx0.phase = "own_recent"
        for t in own[-6:]:
            ctx0[(t.perceiver, t.seq)] = t
        return Context(actor=actor, place=None, tuples=sorted(ctx0.values(), key=lambda t: (t.chronology, t.seq)), present=set(), sources=dict(ctx0.source))

    # where the actor is: its scene by containment (v5 §11); a hand-built
    # log with no index reads it from the actor's own last tuple, as v3
    place = kosha.scene_of(actor, now) if kosha is not None else last.place

    def scene_now(p: str) -> str | None:
        if kosha is not None:
            return kosha.scene_of(p, now)
        s = log.stream(p, before=now)
        return s[-1].place if s else None

    def is_perceiver(p: str) -> bool:
        return kosha.is_perceiver(p) if kosha is not None else p not in log.by_place

    pstream = log.place_stream(place, before=now) if place else []
    ctx = _Sourced()
    ctx.add(last)
    supplied_places: list[str] = []
    across: list[tuple[str, str, list[str]]] = []  # sandhi: (boundary thing, the scene across it, what is there)
    # In transit together (Vadhu tick 341): whoever was written in no
    # scene in the SAME pulse as the actor came along with them.
    in_transit: set[str] = set()
    ctx.phase = "in_transit"
    if place is None:
        for p in log.by_perceiver:
            if p == actor or p in log.by_place:
                continue
            s = log.stream(p, before=now)
            if len(s) > 1 and s[-1].place is None and s[-1].chronology == last.chronology and scene_now(p) is None:
                in_transit.add(p)
                ctx.add(s[0])
                ctx.add(s[-1])
    # the chain: what the actor is in, up to its scene -- first tuples,
    # so "in the bed", "in the ward" bind rather than mint (v5 §26)
    chain = kosha.chain(actor, now) if kosha is not None else []
    ctx.phase = "chain"
    for c in chain:
        s = log.stream(c, before=now)
        if s:
            ctx.add(s[0])
    # what holds the scene, to the root -- first tuples, so a place named at
    # a distance is put inside the region it belongs to rather than in
    # nothing (the telepath playtest, replayed: in nothing 4/4 without it,
    # nested 4/4 with it)
    above = kosha.above(place, now) if kosha is not None and place else []
    # v4.30: the tree to reason with when a new place is placed -- the scene and what holds it, top first,
    # each with the places it directly holds (Marc, 2026-09-28)
    tree = ([(h, kosha.label(h), [(x, kosha.label(x)) for x in xs]) for h, xs in kosha.tree(place, now)]
            if kosha is not None and place else [])
    ctx.phase = "above"
    for c in above:
        s = log.stream(c, before=now)
        if s:
            ctx.add(s[0])
    neighbours: list[str] = []
    if place:  # item 2: the scene's first tuple -- what it is called -- always
        ctx.phase = "place_first"
        own_first = log.stream(place, before=now)
        if own_first:
            ctx.add(own_first[0])
        # item 9, v5 §12 (v4.29): the scenes within reach -- across a boundary,
        # then the container directly above if a scene, the scenes directly in
        # this one, and those directly in the same container -- each its first
        # and latest record and who is in it. Structural: v3's rule here was a
        # label-word search.
        if kosha is not None:
            ctx.phase = "neighbours"
            neighbours = kosha.neighbours(place, now)
            for nb in neighbours:
                s = log.stream(nb, before=now)
                if s:
                    ctx.add(s[0])
                    ctx.add(s[-1])
                for occ in sorted(kosha.occupants(nb, now, exclude={actor}))[:NEIGHBOUR_OCCUPANTS]:
                    so = log.stream(occ, before=now)
                    if so:
                        ctx.add(so[0])
                        ctx.add(so[-1])
                supplied_places.append(nb)
            # sandhi: what is across each boundary thing this scene shares with another -- who is
            # there and what was placed there lately, by the same bounds as the scene's own
            ctx.phase = "across"
            for face, other in kosha.across(place, now):
                there = sorted(kosha.occupants(other, now, exclude={actor}))[:NEIGHBOUR_OCCUPANTS]
                things = sorted(kosha.contents(other, now, since=now - CONTENTS_WINDOW))[:CONTENTS_CAP]
                # the boundary thing itself perceives both sides (live, collide 670-676: the door's own
                # tuples held the dark room behind it) -- its latest is shown to either side
                for x in [face] + there + things:
                    sx = log.stream(x, before=now)
                    if sx:
                        ctx.add(sx[-1])
                across.append((face, other, there + things))
            # v5 §27: things in the scene moved or minted lately -- the
            # granularity bound: never the scene's whole inventory
            ctx.phase = "contents"
            for th in sorted(kosha.contents(place, now, since=now - CONTENTS_WINDOW))[:CONTENTS_CAP]:
                s = log.stream(th, before=now)
                if s:
                    ctx.add(s[0])
                    if len(s) > 1:
                        ctx.add(s[-1])
    recent = pstream[-k_place:]
    ctx.phase = "place_window"
    for t in recent:
        ctx.add(t)
    present = set(in_transit)
    if kosha is not None and place:
        present |= kosha.occupants(place, now, exclude={actor})
    elif place:
        for p in {t.perceiver for t in recent if t.perceiver != place and t.perceiver not in log.by_place}:
            s = log.stream(p, before=now)
            if s and s[-1].place == place:
                present.add(p)
    holder: dict[str, str] = {}
    if kosha is not None:
        for x in present | set(kosha.holds(actor, now)):
            c = kosha.container_at(x, now)
            if c is not None and (c == actor or c in present):
                holder[x] = c
    pace = pacing or {}
    quiet = participation(log, present, now, action, keep | {actor}, int(pace.get("participants", PARTICIPANTS)),
                          pace.get("participation_window"), pace.get("participation_half_life"))
    held_quiet = mauna(log, actor, now, place, present, action, keep) if MAUNA and kosha is not None and place else set()
    quiet = quiet | held_quiet
    if kosha is not None:  # v4.60: a chain's links speak only as they are read (read_chains, below)
        quiet |= {x for x in present if x in chain_links(kosha, chain_head(kosha, x, now, actor), now)}
    attention = {t.perceiver for t in ctx.values()} | present
    if pstream:
        prev = pstream[-1].chronology
        attention |= {t.perceiver for t in pstream if t.chronology == prev}

    def witnessed_last(p: str, s: list[Tuple]) -> Tuple | None:
        """A scene witnesses a departure and nothing after (s4; Vadhu tick
        283). Another perceiver's last tuple enters this context only if
        it is here, or is the step out of here and no later than the
        scene's own last tuple. A scene's own stream is always its own."""
        if p in log.by_place or s[-1].place == place or p in present:
            return s[-1]
        if len(s) > 1 and s[-2].place == place and pstream and s[-1].chronology <= pstream[-1].chronology:
            return s[-1]
        return None

    ctx.phase = "attention"
    for p in attention:
        s = log.stream(p, before=now)
        if s:
            ctx.add(s[0])
            if p in quiet:
                continue  # present and quiet: who they are, and no more
            last_seen = witnessed_last(p, s)
            if last_seen is None:
                continue
            ctx.add(last_seen)
            # ... and the name of the scene they are in now (Vadhu 154)
            there = scene_now(p)
            if there and there != place and there in log.by_perceiver:
                first = log.stream(there, before=now)
                if first:
                    ctx.add(first[0])
    # v5 §26 item 3: where those in attention have LATELY been -- the last
    # few scenes of their own stream, named (Ravenmoor 360: driving
    # Hartley "to his house", the Narrator was shown Nathaniel's house and
    # not Hartley's, and filed him at Nathaniel's)
    ctx.phase = "their_scenes"
    for p in sorted(p for p in attention if p != actor and p not in quiet and is_perceiver(p)):
        seen_scenes: list[str] = []
        for t in reversed(log.stream(p, before=now)):
            if t.place and t.place not in seen_scenes and t.place != place:
                seen_scenes.append(t.place)
            if len(seen_scenes) >= THEIR_SCENES:
                break
        for sc in seen_scenes:
            first = log.stream(sc, before=now)
            if first:
                ctx.add(first[0])
    # v5 §8 (Marc, 2026-09-23): an actor whose perception REACHES -- by
    # its world's default or by becoming -- is shown the few perceivers
    # anywhere in the world whose latest tuples lie nearest the action and
    # the actor's own latest memory (embedding similarity; no text read):
    # their first tuple, the best of their last few, and their latest. The engine had permitted a
    # reaching perceiver's tuple since the capability existed, and never
    # SHOWN one -- the telepath playtest's supervisor reached for the miner
    # trapped below, and the Narrator, shown only the shift office, minted
    # a second miner. Only actors who reach; bounded by REACHING, one pass
    # over each perceiver's last REACH_LATELY tuples.
    ctx.phase = "reaching"
    if kosha is not None and kosha.capable(actor, "reaches", now):
        anchors = [np.asarray(a) for a in (query_embedding, own[-1].embedding if own else None) if a is not None]
        scored = []
        for p in kosha.perceivers if anchors else ():
            if p == actor or p in present or p in attention:
                continue
            ps = log.stream(p, before=now)
            lately = [t for t in ps[1:][-REACH_LATELY:] if t.embedding is not None]
            if not lately:
                continue
            # what they are sending now is in their last few, not only the last
            # (the playtest: the miner's latest was of his wife; his counting
            # and "let someone hear" was the one before)
            # ... similarity, lifted by the tuple's weight: a mind in extremity
            # is heard before a mind at rest (replay: the small embedding
            # model scored vocabulary -- "damp", "hum" -- over situation, and
            # left the trapped miner sixth; weight brought him fourth)
            def loud(t: Tuple) -> float:
                return max(_relevance(t, a) for a in anchors) * (1 + 0.15 * (4 - t.salience))
            best = max(lately, key=loud)
            scored.append((loud(best), p, ps, best))
        for _score, _p, ps, best in sorted(scored, key=lambda x: (-x[0], x[1]))[:REACHING]:
            for t in (ps[0], best, ps[-1]):
                ctx.add(t)
    ctx.phase = "similar"
    if k_sim > 0 and query_embedding is not None:
        rows = _similar_rows(log, actor, own, place, pstream, ctx, log.whole(actor, own))
        if rows is not None:  # the kept rows: the same pool, in the same order, never converted again
            pool, E = rows
            chosen = _mmr(query_embedding, pool, k_sim, E=E)
        else:
            pool = {(t.perceiver, t.seq): t for t in own + pstream if (t.perceiver, t.seq) not in ctx}
            chosen = _mmr(query_embedding, list(pool.values()), k_sim)
        for t in chosen:
            ctx.add(t)
    # item 8, two channels (v5 §26, settled by replay 2026-09-23): the six
    # weightiest by WEIGHT -- what haunts -- and up to KEY_ALLOWANCE keys
    # chosen by relevance to what the actor is doing now. Ranking one list
    # by the heavier of the two let the oldest key age out once a story
    # held many (Ravenmoor t=52, Dyer's transcript, 2 ticks in 7).
    ctx.phase = "own_salient"
    whole_own = log.whole(actor, own)  # read whole: its indexes answer (the same answers as the passes)
    drama = (log.ranked(actor, "salient", OWN_SALIENT) if whole_own
             else sorted(own[:-1], key=lambda t: (t.salience, -t.chronology))[:OWN_SALIENT])
    for t in drama:
        ctx.add(t)
    ctx.phase = "own_keys"
    keys = [t for t in (log.keys_of(actor) if whole_own else own[:-1]) if t.use == "key" and t not in drama]
    for t in sorted(keys, key=lambda t: -_relevance(t, query_embedding))[:KEY_ALLOWANCE]:
        ctx.add(t)
    # ... and what those PRESENT cannot forget (Vadhu tick 345) -- by the
    # heavier of their two words, so a read document's key lines come
    # whenever it is present (v5 §18)
    ctx.phase = "present_salient"
    for p in present:
        if p == actor or p in log.by_place or p in quiet:
            continue
        s = log.stream(p, before=now)
        if not s:
            continue
        for t in (log.ranked(p, "heavy", PRESENT_SALIENT) if log.whole(p, s)
                  else sorted(s[:-1], key=lambda t: (heavier(t), -t.chronology))[:PRESENT_SALIENT]):
            ctx.add(t)
    ctx.phase = "last_here"
    if LAST_HERE and place:
        for p in [actor] + sorted(present - quiet):
            if p in log.by_place:
                continue
            s = log.stream(p, before=now)
            if log.whole(p, s):  # by the runs of the stream, not a walk back through it
                t = log.last_here(p, place)
                if t is not None:
                    ctx.add(t)
                continue
            i = len(s) - 1
            while i >= 0 and s[i].place == place:
                i -= 1  # this visit
            while i >= 0 and s[i].place != place:
                i -= 1  # elsewhere since
            if i >= 0:
                ctx[(s[i].perceiver, s[i].seq)] = s[i]
    # item 7: what came of it (Vadhu 139), and for a tuple retrieved for
    # its weight, anyone's, the next WEIGHT_FOLLOW_UP of its stream.
    ctx.phase = "followed"
    streams: dict[str, list[Tuple]] = {actor: own}
    # where a tuple stands in its stream: the stream's own position index when read whole
    index: dict[str, dict[int, int]] = {actor: log._pos[actor] if whole_own else {t.seq: i for i, t in enumerate(own)}}
    for t in list(ctx.values()):
        by_weight = ctx.source.get((t.perceiver, t.seq)) in ("own_salient", "own_keys", "present_salient")
        if t.perceiver != actor and not by_weight:
            continue
        if t.perceiver not in streams:
            streams[t.perceiver] = log.stream(t.perceiver, before=now)
            index[t.perceiver] = (log._pos[t.perceiver] if log.whole(t.perceiver, streams[t.perceiver])
                                  else {u.seq: i for i, u in enumerate(streams[t.perceiver])})
        stream, i = streams[t.perceiver], index[t.perceiver].get(t.seq)
        if i is None:
            continue
        for nxt in stream[i + 1: i + 1 + (WEIGHT_FOLLOW_UP if by_weight else 1)]:
            ctx.add(nxt)
    # item 5, v5 §17: a DECLARED name found in the action or the actor's
    # own retrieved tuples brings that entity -- exact, and bounded by
    # who the actor's own stream has met (paricaya). v3's label-word
    # rules (half the words, the head noun) are retired.
    ctx.phase = "named"
    if kosha is not None:
        texts = action + "\n" + "\n".join(t.memory for t in ctx.values() if t.perceiver == actor)
        among = set(log.acquainted.get(actor, set())) | {t.perceiver for t in ctx.values()}
        among.discard(actor)
        # ... and what the ACTION paraphrases, by NAMED_BY_WORDS words of a label (Marc: "paraphrase should
        # be allowed"). The action only: over the actor's own tuples too, two words woke 16-22 beings on nearly
        # every pulse of both worlds -- old place labels on "road", "night", "ridge"; over the action, one
        # being on a pulse in seven (Ravenmoor) or three (the crossing): the tin peddler off the hill roads,
        # the boarding house with the green door, the rusted tin cup (measured 2026-09-26)
        paraphrased = named_by_words(kosha, action, among, NAMED_BY_WORDS)
        for pid in sorted(set(kosha.names_in(texts, among)) | paraphrased):
            s = log.stream(pid, before=now)
            if not s:
                continue
            if pid in log.by_place:
                supplied_places.append(pid)
            last_seen = witnessed_last(pid, s)
            if last_seen is not None:
                ctx.add(last_seen)
            if len(s) > 1 or last_seen is None:
                ctx.add(s[0])
    reading: list[tuple[str, list[str], list[str]]] = []
    if kosha is not None:
        ctx.phase = "reading"
        reading = read_chains(log, actor, now, action,
                              set(kosha.names_in(action, among)) | named_by_words(kosha, action, among, NAMED_BY_WORDS), own)
        for _head, _read, unread in reading:
            for link in unread:
                s = log.stream(link, before=now)
                for t in s[1:2] or s[:1]:  # what is written on it: its first memory, past the label
                    ctx.add(t)
    namesakes: list[tuple[str, str, str]] = []
    players_named: list[tuple[str, str]] = []
    if NAMESAKES and kosha is not None:
        ctx.phase = "namesakes"
        text = action + "\n" + "\n".join(t.memory for t in ctx.values() if t.perceiver == actor)
        # only people a name part of whom stands whole in the text can be hit: found through the
        # part index by the text's words -- cost the text, not the world's people (2026-09-25)
        idx = kosha.part_index()
        bearers: set[str] = set()
        for w in set(re.findall(r"\w+", text)):
            bearers |= idx.get(w, set())
        for p, ents in kosha.odd_parts().items():  # a part that is not one plain word is tried as it is
            if p in text:
                bearers |= ents
        # a namesake is one who perceives (v5 §17: no type tells them apart; svabhava "actor" stood in beside it
        # until 2026-09-30, and in the live worlds made persons of a wallet, a derringer and "my horse")
        perceivers = [e for e in kosha.canon.entities if e in bearers and e != actor and not kosha.is_scene(e)
                   and kosha.is_perceiver(e)]
        met = set(log.acquainted.get(actor, set()))
        found: list[tuple[int, str, str]] = []
        for eid in perceivers:
            parts = sorted(kosha.name_parts(eid))
            hit = next((p for p in parts if re.search(whole_word(p), text)), None)
            if hit is None or eid in present:
                continue
            if eid not in keep and eid in met:
                continue  # someone the actor has met is theirs already (s34's rules); the namesake is for strangers
            s = log.stream(eid, before=now)
            found.append((-(s[-1].chronology if s else -1), eid, hit))
        for _k, eid, hit in sorted(found)[:NAMESAKES]:
            # by name and id only: a stranger's or an absent player's own
            # tuples are theirs, never shown to another's turn (s34's bound)
            name = kosha._first_name.get(eid) or (kosha.canon.entities[eid].labels or [eid])[0]
            if eid in keep:
                players_named.append((eid, name))
            else:
                namesakes.append((eid, name, hit))
    # item 6: the actor's recent scenes, named, and who is in each now
    have = {k[0] for k in ctx}
    ctx.phase = "recent_places"
    recent_places = []
    for t in own[-RECENT_PLACES:]:
        if t.place and t.place not in recent_places:
            recent_places.append(t.place)
    for rp in recent_places + [p for p in supplied_places if p not in recent_places]:
        if rp not in have:
            first = log.stream(rp, before=now)
            if first:
                ctx.add(first[0])
                have.add(rp)
        standing = 0
        if kosha is not None:
            occupants = sorted(kosha.occupants(rp, now, exclude={actor}))
        else:
            occupants = []
            for p, s in log.by_perceiver.items():
                before_now = [u for u in s if u.chronology < now]
                if p not in log.by_place and p != actor and before_now and before_now[-1].place == rp:
                    occupants.append(p)
        for pid in occupants:
            if pid in have:
                continue
            s = log.stream(pid, before=now)
            if not s:
                continue
            ctx.add(s[-1])
            if len(s) > 1:
                ctx.add(s[0])
            have.add(pid)
            standing += 1
            if standing >= OCCUPANTS_PER_PLACE:
                break
    # item 10: the name of every scene the memory shown stands in
    ctx.phase = "stood_in"
    have = {k[0] for k in ctx}
    for t in list(ctx.values()):
        if t.place and t.place != place and t.place not in have and t.place in log.by_perceiver:
            first = log.stream(t.place, before=now)
            if first:
                ctx.add(first[0])
                have.add(t.place)
    # v5.33: a matter shown is shown by its head and its tail, together (the Narrator says when a tail closes a head)
    ctx.phase = "head_tail"
    head_tail_company(log, ctx, actor, now)
    # mauna: the held keep their first tuple -- who they are -- and nothing any
    # rule above brought of theirs (the scene's window, what followed, a name)
    # -- except what a chain being read shows of the links not yet read (v4.60: that is why they are shown)
    being_read = {x for _head, _read, unread in reading for x in unread}
    for key in [key for key in ctx if key[0] in held_quiet and key[0] not in being_read
                and key != (key[0], log.stream(key[0])[0].seq)]:
        del ctx[key]
        ctx.source.pop(key, None)
    ordered = sorted(ctx.values(), key=lambda t: (t.chronology, t.seq))
    # what the scene's holder holds that has no voice here, by declared name:
    # a thing with no tuple is never retrieved, and was minted again (the
    # telepath playtest's cage; replayed: reused 8/8 with this line, 0/16
    # without). Bounded: HELD entries, HELD_DEPTH steps inside the holder.
    held: list[tuple[str, str, str]] = []
    if kosha is not None and place:
        shown = {t.perceiver for t in ordered} | set(present) | set(neighbours) | set(chain) | set(above) | {actor}
        held = kosha.held(place, now, shown, depth=HELD_DEPTH, cap=HELD)
    known: list[tuple[str, str]] = []
    # placed nowhere: in nothing, or in the earth with no place of its own (a mention, Marc 2026-10-04)
    # every chain ends at the earth: the top of the actor's own (the chain's depth, never the world's size)
    top = kosha.root(actor, now) if kosha is not None else None
    earth = top if top is not None and getattr(kosha.canon.entities.get(top), "origin", None) == "itihasa" else None
    unplaced = (None, earth)
    if KNOWN and kosha is not None:
        seen = {t.perceiver for t in ordered} | set(present) | {h[0] for h in held} | set(above) | set(chain)
        found = []
        for eid in log.acquainted.get(actor, ()):
            if eid in seen or eid not in kosha._first_name or kosha.container_at(eid, now) not in unplaced or eid == earth:
                continue
            if kosha.is_perceiver(eid, now) and (not sandhi or eid in keep):  # sandhi: a person placed nowhere is known
                continue
            minted = _minted_lately(kosha, eid, now)  # a name alone has no containment record
            if minted is None:
                continue
            found.append((-minted, eid, kosha._first_name[eid]))
        known = [(eid, name) for _m, eid, name in sorted(found)[:KNOWN]]
    elsewhere: list[tuple[str, str, str]] = []
    if KNOWN_ELSEWHERE and kosha is not None:
        seen = ({t.perceiver for t in ordered} | set(present) | {h[0] for h in held} | {k[0] for k in known}
                | set(above) | set(chain) | set(neighbours) | {actor} | ({place} if place else set()))
        found = []
        for eid in log.acquainted.get(actor, ()):
            if eid in seen or eid in keep or eid not in kosha._first_name:
                continue
            where = kosha.container_at(eid, now)
            if where in unplaced:
                continue  # placed nowhere: KNOWN's
            minted = _minted_lately(kosha, eid, now)
            if minted is None:
                continue
            found.append((-minted, eid, kosha._first_name[eid], where))
        elsewhere = [(eid, name, where) for _m, eid, name, where in sorted(found)[:KNOWN_ELSEWHERE]]
    if SEEN_ELSEWHERE and kosha is not None:
        reach = ({place} | set(neighbours)) if place else set()
        listed = {e[0] for e in elsewhere}
        named = {p for (p, _seq), phase in ctx.source.items() if phase in ("named", "namesakes")}
        in_action = (set(kosha.names_in(action, named)) | named_by_words(kosha, action, named, NAMED_BY_WORDS)
                     if action else set())  # the one the actor is going to, first -- whole or paraphrased
        latest: dict[str, int] = {}
        for t in ordered:
            latest[t.perceiver] = max(t.chronology, latest.get(t.perceiver, t.chronology))
        found = []
        for eid, when in latest.items():
            if (eid == actor or eid in keep or eid in listed or eid in present or eid not in named
                    or eid not in kosha.canon.entities or kosha.is_scene(eid)):
                continue
            where = kosha.scene_of(eid, now)
            if where is None or where in reach or kosha.is_root_scene(where):
                continue
            name = kosha._first_name.get(eid) or (kosha.canon.entities[eid].labels or [eid])[0]
            found.append((eid not in in_action, -when, eid, name, where))
        elsewhere += [(eid, name, where) for _a, _w, eid, name, where in sorted(found)[:SEEN_ELSEWHERE]]
    lately: list[tuple[str, str, str | None]] = []
    if LATELY_TELLINGS and kosha is not None and place:
        moments = {(t.chronology, t.place) for t in own[-LATELY_TELLINGS:] if t.place}
        found: dict[str, int] = {}
        for when, where in moments:  # by_place holds every perceiver's tuples by the scene they happened in
            for t in reversed(log.by_place.get(where, ())):  # newest first; commits come nearly in tick order
                if t.chronology < when - 10:
                    break  # bounded: the scene's recent tail, never its history
                if t.chronology == when:
                    found[t.perceiver] = max(when, found.get(t.perceiver, when))
        for eid, when in sorted(found.items(), key=lambda x: -x[1]):
            if (eid == actor or eid in present or eid == place or kosha.is_scene(eid)
                    or eid not in kosha.canon.entities or kosha.scene_of(eid, now) == place):
                continue
            if not any(u.use in ("useful", "key") for u in log.stream(eid, before=now)):
                continue
            name = kosha._first_name.get(eid) or (kosha.canon.entities[eid].labels or [eid])[0]
            lately.append((eid, name, kosha.scene_of(eid, now)))
            if len(lately) >= LATELY:
                break
    return Context(actor=actor, place=place, tuples=ordered, present=present, sources=dict(ctx.source), holder=holder,
                   neighbours=neighbours, chain=chain, above=above, held=held, quiet=quiet, known=known, elsewhere=elsewhere,
                   namesakes=namesakes, players_named=players_named, across=across, tree=tree, lately=lately, reading=reading,
                   last_region=last_region_stood(log, kosha, actor, place, now, tree))
