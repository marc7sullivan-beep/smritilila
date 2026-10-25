# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Kośa: what is in what (Design Principles v5 §9-§12, §16).

One relation, *x is in y*, for any entity, nested without limit and
changing over time; one bit on a container, whether it is a *scene*
(āvaraṇa). Everything v3 did with places is read through these two:
where someone is, who is with them, what is next door.

Containment is recorded, not stored (v5 §10): an entity's container at tick
t is its latest kosha record before t. Worlds and entities that predate
v4 carry no records, so containment is DERIVED on read from what v3 kept
(v5 Appendix B, corrected 2026-09-23 while building -- no conversion writes):
  - a perceiver is in the scene of its own latest tuple (v3's "where
    someone is, is the place their own tuples last record them");
  - a v3 place is in the building v3.10's kośa named, if any;
  - a v3 place (any avarana: v3's building/open/road as v4's scene) is a scene.
A record always outranks the derivation, except where a later tuple
files the perceiver in a different scene than its record's chain
reaches -- a v3-style move with no record -- and then the tuple wins.

Declared names (v5 §16) are exact, case-folded strings. Entities that
predate v4 have no name records; their names are derived on read from
their labels: each label whole, and each proper name in it (v3's
label_names_of) -- the same derivation rule, so nothing is written.

Cost: building the index is one pass over memories and records; every
lookup after is a bisect or a walk up a chain. Nothing here reads a word
of a label to decide what is inside what (v5 §1); names are compared, never
interpreted.
"""
from __future__ import annotations

import bisect
import copy
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator

from smritilila.graph.canon import Maya
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.text import ARTICLES, unsegmented, weight

# Between places. A tuple's location when its perceiver stands in no
# enclosure -- walking open country, mid-river -- is this sentinel: a
# value the store can hold (location_id is NOT NULL), an id no entity
# owns, and NEVER a stream: everyone in transit is not thereby together.
# Tombstone tick 317-331: Wind walked fifteen ticks north from the Lakota
# camp and was filed at the camp the whole way, its people listing him
# present; nothing named the ground he crossed, so nothing could bind him
# to it. Nowhere is what the world honestly records until something does.
NOWHERE = "nowhere"


def whole_word(*terms: str) -> str:
    """A pattern for any of `terms` found whole in text -- not inside a
    longer word, and a hyphen is part of the word ("Marsh-Corwin" is not
    Corwin). Every place a name in a spaced script is looked for in text uses
    this one (an unsegmented script takes a plain `in`, names_in)."""
    return r"(?<![\w\-])(?:" + "|".join(re.escape(x) for x in terms) + r")(?![\w\-])"


# No depth cap on a walk up the tree (2026-10-03): CHAIN_DEPTH = 16 called a deeper chain a loop, but v4.60's
# chains nest one inside the next -- Stonemarch's book, page twelve, is 17 deep -- and the walk stopped short of the
# earth (its root read as a county) while `inside` (cap 32) could have let a long chain's cycle through. A walk
# never revisits what it passed, so it ends by itself. Cost: the chain's own length.
_RUN = re.compile(r"[\w\-]+")  # a word, as names_in bounds one (a hyphen joins)
_EVER = 10**9  # a tick after everything: "now", for the present index


class _Now:
    """The present (Kosha._now): each entity's container and scene now, what
    each container holds, who is in each scene, and the tree of scenes --
    every value computed by the same container_at / scene_of / parent_scene
    the scans use, so a lookup and a scan cannot disagree."""
    __slots__ = ("n", "cur", "kids", "sn", "members", "ps", "child_scenes", "owned")

    def __init__(self, n: int) -> None:
        self.owned: set | None = None                                # None: every set its own (built fresh)
        self.n = n                                                   # the canon's size when read
        self.cur: dict[str, str | None] = {}                         # entity -> its container now
        self.kids: dict[str, set[str]] = defaultdict(set)            # container -> what is directly in it now
        self.sn: dict[str, str | None] = {}                          # entity -> its scene now
        self.members: dict[str, set[str]] = defaultdict(set)         # scene -> every entity whose scene it is now
        self.ps: dict[str, str | None] = {}                          # scene -> the scene it sits in now
        self.child_scenes: dict[str, set[str]] = defaultdict(set)    # scene -> the scenes sitting in it now

    def clone(self) -> "_Now":
        """Another version: the tables copied, each set shared until changed."""
        c = _Now.__new__(_Now)
        c.n, c.cur, c.sn, c.ps = self.n, dict(self.cur), dict(self.sn), dict(self.ps)
        c.kids, c.members = defaultdict(set, self.kids), defaultdict(set, self.members)
        c.child_scenes, c.owned = defaultdict(set, self.child_scenes), set()
        return c

    def s(self, attr: str, key: str) -> set[str]:
        """The set at attr[key], this version's own (copied on first change)."""
        d = getattr(self, attr)
        if self.owned is not None and (attr, key) not in self.owned:
            self.owned.add((attr, key))
            d[key] = set(d.get(key, ()))
        return d[key]


class _Over(dict):
    """A pulse's own entries over the world's (Kosha.clone, fork=True): read through,
    written here -- a pulse held and let go leaves nothing in the world's."""
    __slots__ = ("under",)

    def __init__(self, under: dict) -> None:
        super().__init__()
        self.under = under

    def get(self, k, default=None):
        return dict.get(self, k) if dict.__contains__(self, k) else self.under.get(k, default)

    def __getitem__(self, k):
        return dict.__getitem__(self, k) if dict.__contains__(self, k) else self.under[k]

    def __contains__(self, k) -> bool:
        return dict.__contains__(self, k) or k in self.under


# Kosha's per-key containers, shared by versions until one changes them (_own)
_COW = {"_clauses": list, "_caps": list, "_rec": list, "_mem": list, "_active": list, "_names": set, "_named": set,
        "_recorded_names": set, "_by_run": set, "_clause_keys": set, "_actions_by": dict, "_part_index": set,
        "_odd_parts": set, "_sandhi": list, "_sandhi_at": set, "_prakriti": list}
_FLAT = ("_name_text", "_first_name", "_name_heavy")          # values never changed: the table copied, no more
_SETS = ("_perceivers", "_unkeyed", "_label_declared", "_repairs", "_shared")  # copied whole: the world's people and its few repairs, not its history

# A containment record a god's repair wrote, not something lived (Marc, 2026-10-01: "a repair isn't a lived change"):
# a seed's (seed, seed2, ...), the earth's gathering, an owner's or a gravestone's. Any other note is lived.
REPAIR_NOTES = frozenset({"earth", "owner", "gravestone"})


def no_scenes() -> bool:
    """No scenes (Marc, 2026-10-12: "Scenes really don't exist ... It's all just containers"; TODO 2026-10-12, simulated read-only on five
    worlds, option 1): SMRITILILA_NO_SCENES=1 makes the scene bit derived, not read. A container is a shared moment once a perceiver has been
    directly in it, and a perceiver never is (Kosha.is_scene). Off unless set (the launchers, scripts/dev_server.py and scripts/serve.py, set it to 1 by default since 2026-10-19; SMRITILILA_NO_SCENES=0 opts a server out); per server, and a server runs one world. Read at each ask,
    so a test or a replay can turn it on and off."""
    return os.environ.get("SMRITILILA_NO_SCENES") == "1"


def is_repair(note: str | None) -> bool:
    return bool(note) and (note in REPAIR_NOTES or note.startswith("seed"))


def _fold(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).casefold()


class Kosha:
    """The containment index of one world at one moment of reading."""

    def __init__(self, canon: Maya, memories: Iterable[Memory] = (), records: Iterable[Record] = (),
                 defaults: dict | None = None, absent: Iterable[str] = ()) -> None:
        self.canon = canon
        # Out of the world (Marc, 2026-09-24): players who logged off and do
        # not allow the world to act for them -- "entirely suspended from the
        # game, not present at all". Where they are is kept; they are in no
        # scene's present until they come back.
        self.absent: frozenset[str] = frozenset(absent)
        # v5 §7-§8: the world's capability defaults ({"perceives": False,
        # "reaches": False} unless the world data says otherwise)
        self.defaults = {"perceives": False, "reaches": False, **(defaults or {})}
        self._clauses: dict[tuple[str, str], list[tuple[int, int, str, str | None]]] = defaultdict(list)
        self._caps: dict[tuple[str, str], list[tuple[int, int, bool, bool]]] = defaultdict(list)  # (tick, order, on, within)
        # PRAKRITI (v5 §15): the world's nature, borne by the earth or a branch -- (entity, key) -> [(tick, order, JSON or None)]
        self._prakriti: dict[tuple[str, str], list[tuple[int, int, str | None]]] = defaultdict(list)
        self._caps_within = False  # any capability reaching within: whether a scene perceives then hangs on what holds it
        self._repairs: set[int] = set()  # the order of each containment record a repair wrote: kāla's locks pass over them
        self.said: dict[tuple[str, str], tuple[str, str | None]] = {}   # (scene, pulse_id) -> (line, speaker)
        self.marks: dict[tuple[str, str], set[str]] = defaultdict(set)  # (entity, pulse_id) -> marks
        self._rec: dict[str, list[tuple[int, int, str | None]]] = defaultdict(list)
        self._mem: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
        self._perceivers: set[str] = set()
        self._shared: set[str] = set()   # no-scenes: the containers a perceiver has been directly in, never a perceiver (is_scene)
        self._names: dict[str, set[str]] = defaultdict(set)      # entity -> declared names (folded)
        self._named: dict[str, set[str]] = defaultdict(set)      # folded name -> entities
        self._name_text: dict[str, str] = {}                     # folded -> as declared
        self._first_name: dict[str, str] = {}                    # entity -> its first name, as declared
        self._recorded_names: dict[str, set[str]] = defaultdict(set)  # entity -> names declared by a record, as written
        self._part_index: dict[str, set[str]] | None = None
        self._by_run: dict[str, set[str]] = defaultdict(set)     # a name's longest word -> the folded names keyed by it
        self._unkeyed: set[str] = set()                           # unspaced names, and names with no word: each is tried
        self._name_heavy: dict[str, bool] = {}                    # folded name -> weighs enough to be found (v5 §7)
        self.actions: dict[tuple[str, int], tuple[str, str | None]] = {}
        self._actions_by: dict[str, dict[int, tuple[str, str | None]]] = defaultdict(dict)  # actor -> {tick: their words}
        # VERSIONS (2026-09-25): a Kosha is kept on the world and read on (extend) rather than
        # built from history each pulse; each version and each pulse's fork shares what it has
        # not changed (_own copies a list or set on first change). None: this one owns all.
        self._owned: set | None = None
        self._label_declared: set[str] = set()   # entities whose labels were declared as names (none recorded then)
        self._first_named: int | None = None     # the tick the world first declared a name: born after it, a name is declared or none
        self._stale = False                      # read on past what a fresh build would equal: build again
        self._odd_parts: dict[str, set[str]] | None = None
        self._order = 0
        self._active: dict[str, list[int]] = defaultdict(list)  # entity -> ticks it had a tuple, or a tuple was set in it
        # THE PRESENT (2026-09-25): what everything is in NOW, kept, so a question about now is a
        # lookup and not a pass over the world -- built on the first such question, then updated
        # only under what moves (_Now). A question about the past, or anything added out of tick
        # order, is answered by the exact scans. use_now=False answers everything by the scans.
        self.use_now = True
        self._latest = -1                 # the latest tick anything that places things holds
        self._nowix: _Now | None = None
        self._clause_keys: dict[str, set[str]] = defaultdict(set)  # entity -> its clause keys (clauses_of)
        self._caps_version = 0
        self._perceivers_cache: tuple[tuple, frozenset[str]] | None = None
        self._order_cache: tuple[int, dict[str, int]] | None = None
        # SANDHI (a junction; collision run, question 4 -- Marc: "aren't opening and containment
        # very closely related?"): a boundary thing -- a window, a door, a gate -- is in BOTH scenes
        # it joins. Its first holder is its containment; a sandhi record gives it the other side.
        # thing -> [(tick, order, scene, open)]; scene -> things ever joined to it.
        self._sandhi: dict[str, list[tuple[int, int, str, bool]]] = defaultdict(list)
        self._sandhi_at: dict[str, set[str]] = defaultdict(set)
        for m in memories:
            self._latest = max(self._latest, m.world_time)
            self._active[m.entity_id].append(m.world_time)
            if m.location_id and m.location_id != m.entity_id:
                self._active[m.location_id].append(m.world_time)
            if m.tier in ("PRIMARY", "SECONDARY"):
                self._perceivers.add(m.entity_id)  # a scene's own perspective counts only if it perceives (v5 §8)
            if m.location_id == m.entity_id:
                continue  # a scene's own record says nothing of where the scene is (crossing 469: it made the overhang its own container)
            loc = None if m.location_id == NOWHERE else m.location_id
            self._mem[m.entity_id].append((m.world_time, loc))
        for lst in self._mem.values():
            lst.sort(key=lambda x: x[0])
        for r in records:
            self.add(r)
        for eid in canon.entities:
            self._declare_labels(eid)
        for p in list(self._perceivers):
            self._perceiver_history(p)   # where each has stood: the memories' place field and the seed's container too
        self._n_canon = len(canon.entities)
        self._last_canon_key = next(reversed(canon.entities), None)

    def _declare_labels(self, eid: str) -> None:
        """A name-less entity's names, derived from its labels (v5 §16) -- only
        while no record names it, as a fresh reading has it."""
        e = self.canon.entities.get(eid)
        if e is None or eid in self._names:
            return
        if self._first_named is not None and e.origin == "smrti" and e.created_at_world_time >= self._first_named:
            return  # minted in play once the world declares names: a name is asked at mint, or there is none (v5 §16)
                    # -- what a seed authored keeps its labels' names (Arkham, planted by Ravenmoor's containment seed)
        for label in e.labels[:1] + [l for l in e.labels[1:] if l]:
            self._declare(eid, label)
            for n in _proper_names(label):
                self._declare(eid, n)
        if eid in self._names:
            self._label_declared.add(eid)

    def _undeclare_labels(self, eid: str) -> None:
        """Its first recorded name has come: the names its labels gave it are
        taken back (a fresh build declares labels only for the unnamed), and
        its first name will be the recorded one. The name index may keep the
        label as a candidate key -- a candidate is found only if someone
        bears it (names_in reads _named)."""
        for f in self._names.get(eid, ()):
            self._own("_named", f).discard(eid)
        self._names[eid] = set()
        if self._owned is not None:
            self._owned.add(("_names", eid))
        self._first_name.pop(eid, None)
        self._label_declared.discard(eid)

    # -- versions ----------------------------------------------------------

    def _own(self, attr: str, key):
        """The container at attr[key], this version's own: copied from the
        shared one on first change (copy-on-write)."""
        d = getattr(self, attr)
        if self._owned is not None and (attr, key) not in self._owned:
            self._owned.add((attr, key))
            cur = d.get(key)
            d[key] = _COW[attr](cur) if cur is not None else _COW[attr]()
        elif key not in d:
            d[key] = _COW[attr]()
        return d[key]

    def clone(self, canon: Maya | None = None, absent: Iterable[str] | None = None, fork: bool = False) -> "Kosha":
        """Another version of this index, sharing everything until it changes
        it. `fork`: a pulse's own -- what it adds to the tables kept per pulse
        (lines said, marks, the player's words) is layered over the world's,
        not added to them. Cost: the tables' keys (the world's entities and
        names), never its history."""
        k = copy.copy(self)
        if canon is not None:
            k.canon = canon
        if absent is not None:
            k.absent = frozenset(absent)
        for attr in _COW:
            d = getattr(self, attr)
            if d is not None:
                setattr(k, attr, defaultdict(d.default_factory, d) if isinstance(d, defaultdict) else dict(d))
        for attr in _FLAT:
            setattr(k, attr, dict(getattr(self, attr)))
        for attr in _SETS:
            setattr(k, attr, set(getattr(self, attr)))
        k._owned = set()
        if fork:
            k.said, k.actions, k.marks = _Over(self.said), _Over(self.actions), _Over(self.marks)
        k._nowix = self._nowix.clone() if self._nowix is not None else None
        return k

    def extend(self, canon: Maya, memories: Iterable[Memory], records: Iterable[Record]) -> "Kosha | None":
        """A new version with a commit's memories and records read on -- this
        one unchanged. None when a fresh build would differ and must be made:
        the canon lost or reordered an entity, or `_stale` was set. A record
        that names an entity whose labels had been declared as its names needs
        no fresh build: add takes the labels back (_undeclare_labels).
        Cost: what was committed, and the tables' keys."""
        n = len(canon.entities)
        keys = list(canon.entities)
        if n < self._n_canon or (self._n_canon and keys[self._n_canon - 1] != self._last_canon_key):
            return None
        k = self.clone(canon=canon)
        for m in sorted(memories, key=lambda m: m.world_time):
            if m.world_time < k._latest:
                k._nowix = None
            k._latest = max(k._latest, m.world_time)
            k._own("_active", m.entity_id).append(m.world_time)
            if m.location_id and m.location_id != m.entity_id:
                k._own("_active", m.location_id).append(m.world_time)
            if m.tier in ("PRIMARY", "SECONDARY") and m.entity_id not in k._perceivers:
                k._perceivers.add(m.entity_id)
                k._perceiver_history(m.entity_id)
                if k._nowix is not None:
                    k._now_update(m.entity_id)
            if m.location_id == m.entity_id:
                continue
            loc = None if m.location_id == NOWHERE else m.location_id
            if loc and m.entity_id in k._perceivers:
                k._seen_in(loc)
            bisect.insort(k._own("_mem", m.entity_id), (m.world_time, loc), key=lambda x: x[0])
            if k._nowix is not None:
                k._now_update(m.entity_id)
        named = set()
        for r in records:
            k.add(r)
            if r.kind == "name" and r.value:
                named.add(r.entity_id)
        for eid in keys[self._n_canon:]:
            k._declare_labels(eid)
        if k._stale:
            return None
        if k._part_index is not None:
            for eid in named | set(keys[self._n_canon:]):
                k._part_add(eid)
        k._n_canon, k._last_canon_key = n, keys[-1] if keys else None
        return k

    def actions_of(self, eid: str) -> dict[int, tuple[str, str | None]]:
        """An actor's own words, by tick (read-only)."""
        return self._actions_by.get(eid, {})

    # -- records --------------------------------------------------------

    def add(self, r: Record) -> None:
        """A record appended in this pulse, or loaded; kept in tick order."""
        self._order += 1
        if r.kind == "kosha":
            bisect.insort(self._own("_rec", r.entity_id), (r.world_time, self._order, r.value or None))
            if r.value and r.entity_id in self._perceivers:
                self._seen_in(r.value)
            if is_repair(r.note):
                self._repairs.add(self._order)
            if self._nowix is not None:
                if r.world_time < self._latest:
                    self._nowix = None  # out of tick order: the present is read again when next asked
                else:
                    self._now_update(r.entity_id)
            self._latest = max(self._latest, r.world_time)
        elif r.kind == "name" and r.value:
            self._first_named = r.world_time if self._first_named is None else min(self._first_named, r.world_time)
            if r.entity_id in self._label_declared:
                self._undeclare_labels(r.entity_id)  # named now: its labels are not its names, as a fresh build has it
            self._declare(r.entity_id, r.value)
            self._own("_recorded_names", r.entity_id).add(r.value.strip())
        elif r.kind == "sandhi" and r.value:
            bisect.insort(self._own("_sandhi", r.entity_id), (r.world_time, self._order, r.value, r.note != "closed"))
            self._own("_sandhi_at", r.value).add(r.entity_id)
        elif r.kind == "action" and r.value is not None:
            self.actions[(r.entity_id, r.world_time)] = (r.value, r.note)
            self._own("_actions_by", r.entity_id)[r.world_time] = (r.value, r.note)
        elif r.kind == "clause" and r.note:
            reach, _, key = r.note.partition(":")
            bisect.insort(self._own("_clauses", (r.entity_id, key or reach)), (r.world_time, self._order, reach if key else "self", r.value))
            self._own("_clause_keys", r.entity_id).add(key or reach)
        elif r.kind == "capability" and r.value:
            reach, _, state = (r.note or "").rpartition(":")
            within = reach == "within"
            bisect.insort(self._own("_caps", (r.entity_id, r.value)), (r.world_time, self._order, state == "on", within))
            self._caps_within = self._caps_within or within
            self._caps_version += 1
            if self._nowix is not None:
                if r.world_time < self._latest:
                    self._nowix = None
                else:
                    self._now_update(r.entity_id)  # whether it perceives decides v5 §11's root rule beneath it
            self._latest = max(self._latest, r.world_time)
        elif r.kind == "prakriti" and r.note:
            bisect.insort(self._own("_prakriti", (r.entity_id, r.note)), (r.world_time, self._order, r.value))
        elif r.kind == "said" and r.value:
            self.said[(r.entity_id, r.pulse_id)] = (r.value, r.note)
        elif r.kind == "mark" and r.value:
            key = (r.entity_id, r.pulse_id)
            if isinstance(self.marks, _Over):  # a fork: its own set, over the world's
                if not dict.__contains__(self.marks, key):
                    dict.__setitem__(self.marks, key, set(self.marks.under.get(key, ())))
                dict.__getitem__(self.marks, key).add(r.value)
            else:
                self.marks[key].add(r.value)

    # -- views and capabilities (v5 §7-§8) ---------------------------------

    def clauses_of(self, eid: str, t: int) -> list[tuple[str, str, str]]:
        """The clauses eid bears before tick t: [(key, reach, text)], a
        withdrawn clause (latest record None) left out."""
        out = []
        for key in self._clause_keys.get(eid, ()):  # its own keys, not every clause in the world
            recs = self._clauses[(eid, key)]
            i = _before([(w, o, None) for w, o, _r, _v in recs], t)
            if i >= 0 and recs[i][3] is not None:
                out.append((key, recs[i][2], recs[i][3]))
        return sorted(out)

    def view_of(self, eid: str, t: int) -> list[tuple[str, str, str]]:
        """The clauses eid stands under, outermost first and eid's own last
        -- nearest wins (v5 §7): every container's clauses that reach WITHIN,
        up the whole chain, then eid's own (self and within alike).
        [(bearer, key, text)]."""
        stack: list[tuple[str, str, str]] = []
        for c in self._up(eid, self.container_at(eid, t), t):
            stack[:0] = [(c, k, text) for k, reach, text in self.clauses_of(c, t) if reach == "within"]
        stack += [(eid, k, text) for k, _reach, text in self.clauses_of(eid, t)]
        return stack

    def capable(self, eid: str, cap: str, t: int) -> bool:
        """eid's own capability record if it has one (either reach), else the
        nearest up its chain that reaches WITHIN (v5 §15: a seed's -- a
        becoming's reaches only its bearer, v5 §8), else the world's default."""
        recs = self._caps.get((eid, cap))
        if recs:
            i = _before(recs, t)
            if i >= 0:
                return recs[i][2]
        if self._caps_within:
            for c in self._up(eid, self.container_at(eid, t), t):
                recs = self._caps.get((c, cap))
                if recs:
                    i = _before([x for x in recs if x[3]], t)
                    if i >= 0:
                        return [x for x in recs if x[3]][i][2]
        return bool(self.defaults.get(cap, False))

    def prakriti_of(self, eid: str, t: int) -> dict[str, object]:
        """What eid itself bears of the world's nature before t (v5 §15):
        {key: value}, a withdrawn key left out."""
        out = {}
        for (e, key), recs in self._prakriti.items():
            if e == eid:
                i = _before(recs, t)
                if i >= 0 and recs[i][2] is not None:
                    out[key] = json.loads(recs[i][2])
        return out

    def rate(self, here: str | None, t: int) -> float:
        """v5 §19, time kept through the branches: world minutes per minute lived
        here. Rates COMPOSE (Marc, 2026-10-01): each branch's own `time` rate
        is relative to what holds it, multiplied up the chain to the earth --
        the one given that composes rather than the nearest winning. A rate
        not above zero says nothing (time may be slowed, never stopped: a
        stopped branch's turns would be behind for ever). 1.0: the world's own."""
        r = 1.0
        if here is None:
            return r
        for c in [here, *self._up(here, self.container_at(here, t), t)]:
            recs = self._prakriti.get((c, "time"))
            if recs:
                i = _before(recs, t)
                if i >= 0 and recs[i][2] is not None:
                    v = json.loads(recs[i][2])
                    x = v.get("rate") if isinstance(v, dict) else None
                    if isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0:
                        r *= float(x)
        return r

    def seed(self, key: str, here: str | None, t: int, field: str | None = None) -> object | None:
        """The world's nature where a question lives (v5 §15): from `here` up
        through what holds it to the earth, the first that says -- a value
        withdrawn says nothing -- else None (the engine's own). With `field`,
        the first whose value says that field (a closed valley's pacing says
        arrivals, not the world's other paces), and the field's value. Cost:
        the chain's depth."""
        if here is None:
            return None
        for c in [here, *self._up(here, self.container_at(here, t), t)]:
            recs = self._prakriti.get((c, key))
            if recs:
                i = _before(recs, t)
                if i >= 0 and recs[i][2] is not None:
                    v = json.loads(recs[i][2])
                    if field is None:
                        return v
                    if isinstance(v, dict) and field in v:
                        return v[field]
        return None

    def _declare(self, eid: str, name: str) -> None:
        f = _fold(name)
        if not f:
            return
        self._own("_names", eid).add(f)
        self._own("_named", f).add(eid)
        if f not in self._name_heavy:
            # the name index (names_in): each name keyed by its longest word, once
            self._name_heavy[f] = weight(f) >= 2
            runs = _RUN.findall(f)
            if unsegmented(f) or not runs:
                self._unkeyed.add(f)
            else:
                self._own("_by_run", max(runs, key=len)).add(f)
        self._name_text.setdefault(f, name.strip())
        self._first_name.setdefault(eid, name.strip())

    def name_parts(self, eid: str) -> set[str]:
        """The capitalised words of an entity's PERSONAL names -- declared
        names and first label that do not begin with an article, up to any
        comma -- titles and abbreviations excepted: "Lowell" of "Edmund
        Lowell", "Thompson" of "Asst. Prof. N. Thompson"; nothing of "the
        Lowell boy", "the Frye Hall janitor", "the door of Room 214" (a
        description, not a name) or of what follows "Professor Hartley,".
        For NAMESAKES: a part is compared whole, never interpreted."""
        # recorded names, never a legacy label's split words (Hartley's
        # "Anthropology", "Miskatonic" were names of his only that way)
        raw = set(self._recorded_names.get(eid, ()))
        e = self.canon.entities.get(eid)
        if e is not None and e.labels:
            raw.add(e.labels[0])
        out: set[str] = set()
        for n in raw:
            out |= personal_name_parts(n)
        return out

    def part_index(self) -> dict[str, set[str]]:
        """Name part -> every non-scene entity bearing it, built once per
        Kosha (one pass over the entities: O(world) per pulse, as the
        namesake line already is)."""
        if self._part_index is None:
            idx: dict[str, set[str]] = defaultdict(set)
            for eid in self.canon.entities:
                if not self.is_scene(eid):
                    for p in self.name_parts(eid):
                        idx[p].add(eid)
            self._part_index = idx
            self._odd_parts = defaultdict(set, {p: set(ents) for p, ents in idx.items() if not re.fullmatch(r"\w+", p)})
        return self._part_index

    def _part_add(self, eid: str) -> None:
        """A new entity's, or a newly named one's, parts into the part index
        (a version read on: names only grow, a label is fixed)."""
        if self.is_scene(eid):
            return
        for p in self.name_parts(eid):
            self._own("_part_index", p).add(eid)
            if not re.fullmatch(r"\w+", p):
                self._own("_odd_parts", p).add(eid)

    def odd_parts(self) -> dict[str, set[str]]:
        """The name parts that are not one plain word (an apostrophe, a
        hyphen): the namesake search tries these as they are."""
        self.part_index()
        return self._odd_parts

    def mark_perceiver(self, eid: str) -> None:
        new = eid not in self._perceivers
        self._perceivers.add(eid)
        if new:
            self._perceiver_history(eid)
        if new and self._nowix is not None:
            self._now_update(eid)  # now someone: present where they are, and no one's scene beneath them (v5 §11)

    def _seen_in(self, c: str) -> None:
        """A perceiver has been directly in container c: with no scenes, c is a shared moment from now on. The present index
        (who stands in which scene) is read again when next asked, since c may have become the scene of what lies beneath it.
        Cost: once per container, the first time anyone stands in it."""
        if c and c not in self._shared and c not in self._perceivers:
            self._shared.add(c)
            self._nowix = None

    def _perceiver_history(self, p: str) -> None:
        """p is a perceiver now (or was read as one): every container it has stood in, by the records, the memories' place field and
        the seed's own container, is a shared moment; and p, a perceiver, is none."""
        for _t, _o, c in self._rec.get(p, ()):
            if c:
                self._seen_in(c)
        for _t, loc in self._mem.get(p, ()):
            if loc:
                self._seen_in(loc)
        e = self.canon.entities.get(p)
        if e is not None and getattr(e, "kosha", None):
            self._seen_in(e.kosha)
        if p in self._shared:
            self._shared.discard(p)
            self._nowix = None

    # -- the present ---------------------------------------------------------

    def _now(self, t: int) -> "_Now | None":
        """The present index, if `t` asks about now (after everything held);
        None for the past, or with use_now off -- the scans answer those."""
        if not self.use_now or t <= self._latest:
            return None
        ix = self._nowix
        if ix is not None and len(self.canon.entities) < ix.n:
            ix = None  # something was taken out of the world: read the present again
        if ix is None:
            ix = self._nowix = self._now_build()
        return ix

    def _now_build(self) -> "_Now":
        ix = _Now(len(self.canon.entities))
        for e in set(self.canon.entities) | self._perceivers | set(self._rec) | set(self._mem):
            c = self.container_at(e, _EVER)
            ix.cur[e] = c
            if c is not None:
                ix.kids[c].add(e)
        for e in list(ix.cur):
            self._now_place(ix, e)
        return ix

    def _now_place(self, ix: "_Now", e: str) -> None:
        """Where e is now, and, for a scene, the scene it sits in -- by the
        same scene_of and parent_scene every scan uses."""
        old = ix.sn.get(e)
        new = self.scene_of(e, _EVER)
        if old != new:
            if old is not None:
                ix.s("members", old).discard(e)
            if new is not None:
                ix.s("members", new).add(e)
            ix.sn[e] = new
        if self.is_scene(e):
            po, pn = ix.ps.get(e), self.parent_scene(e, _EVER)
            if po != pn:
                if po is not None:
                    ix.s("child_scenes", po).discard(e)
                if pn is not None:
                    ix.s("child_scenes", pn).add(e)
                ix.ps[e] = pn

    def _now_update(self, x: str) -> None:
        """x's containment changed: its own container again, then everything
        beneath it -- whose chains pass through x -- placed again. Cost: what
        x holds, not the world."""
        ix = self._nowix
        old, new = ix.cur.get(x), self.container_at(x, _EVER)
        if old != new:
            if old is not None:
                ix.s("kids", old).discard(x)
            if new is not None:
                ix.s("kids", new).add(x)
        ix.cur[x] = new
        stack, seen = [x], set()
        while stack:
            y = stack.pop()
            if y in seen:
                continue
            seen.add(y)
            ix.cur.setdefault(y, self.container_at(y, _EVER))
            self._now_place(ix, y)
            stack.extend(ix.kids.get(y, ()))

    def _canon_order(self) -> dict[str, int]:
        """Each entity's place in the canon's order (neighbours keep it),
        read again only when the canon changed size."""
        n = len(self.canon.entities)
        if self._order_cache is None or self._order_cache[0] != n:
            self._order_cache = (n, {e: i for i, e in enumerate(self.canon.entities)})
        return self._order_cache[1]

    # -- the bit ----------------------------------------------------------

    def is_scene(self, eid: str | None) -> bool:
        if not eid:
            return False
        if no_scenes():
            return eid in self._shared  # derived, not read: a container a perceiver has been in (no_scenes)
        e = self.canon.entities.get(eid)
        # the scene bit (v5 §11): v4's "scene", or any v3 avarana -- every v3 place
        # is a scene by construction (v5 Appendix B). svabhava was read here too until
        # 2026-09-30, and decided nothing the bit did not in any live world
        return e is not None and e.avarana is not None

    def shared_scenes(self) -> frozenset[str]:
        """The containers that count as a shared moment under no_scenes (read-only)."""
        return frozenset(self._shared)

    def is_perceiver(self, eid: str, t: int = 10**9) -> bool:
        """Whoness for the draw and for presence: has an experience tuple
        (v5 §4). A containment record is not one; a scene's impersonal
        record is not one either -- a scene perceives only where it has
        the capability (v5 §8: the house that wakes)."""
        return eid in self._perceivers and (not self.is_scene(eid) or self.capable(eid, "perceives", t))

    @property
    def perceivers(self) -> frozenset[str]:
        """Kept until someone new perceives, a capability changes or the
        canon changes size; read-only (no caller changes it)."""
        # with a capability reaching within, whether a scene perceives hangs on where it is: any record may change it
        key = (len(self._perceivers), self._caps_version, len(self.canon.entities), self._order if self._caps_within else 0)
        if self._perceivers_cache is None or self._perceivers_cache[0] != key:
            self._perceivers_cache = (key, frozenset(p for p in self._perceivers if self.is_perceiver(p)))
        return self._perceivers_cache[1]

    # -- where -------------------------------------------------------------

    def container_at(self, eid: str, t: int) -> str | None:
        """What eid is directly in before tick t, or None for in nothing."""
        recs = self._rec.get(eid)
        rec = None
        if recs:
            i = _before(recs, t)
            rec = recs[i] if i >= 0 else None
        mems = self._mem.get(eid)
        mem = None
        if mems:
            j = bisect.bisect_left(mems, (t,)) - 1
            mem = mems[j] if j >= 0 else None
        if rec is not None:
            # a v3-style move, told by a tuple, with no record: the tuple wins --
            # judged against where the record's chain reached AT THE TUPLE'S
            # TICK -- as the tick began, since a pulse files its tuples at the
            # scene it started in, before its moves -- not now. Judged now, a
            # carried thing that perceives was stranded the moment its carrier
            # moved on: the crossing, 580 -- Sela walked out holding the comb,
            # and the comb's own tuple of that pulse, filed at the inn it was
            # carried out of, kept it at the inn.
            #
            # ... or AS IT ENDED (Ravenmoor 666, Marc 2026-09-25): a v4 pulse files a
            # tuple where it ended, and a carried thing is not moved -- it goes with
            # its carrier -- so its tuple of the drive lands where the drive ended.
            # Judged only as the tick began, Thompson's watch and revolver were set
            # down on the campus he drove to, and stayed when he drove on (the
            # watch's tuple dropped at 697 as elsewhere). And a tuple filed where no
            # scene is (a root standing in: the thing had none) moves nothing.
            if (mem is not None and mem[0] > rec[0] and not self.is_root_scene(mem[1])
                    and mem[1] != self._scene_from(rec[2], mem[0], eid) and mem[1] != self._scene_from(rec[2], mem[0] + 1, eid)):
                return mem[1]
            return rec[2]
        if mem is not None:
            return mem[1]
        e = self.canon.entities.get(eid)
        return e.kosha if e is not None and e.kosha in self.canon.entities else None

    def _up(self, start: str | None, first: str | None, t: int) -> Iterator[str]:
        """Up a chain from `first` -- each container, then what it is in --
        never back to `start` or to anything already passed (a loop the
        records should never hold), however deep. Every walk up the tree is
        this one; each caller stops where it needs to."""
        seen = {start} if start is not None else set()
        c = first
        while c is not None and c not in seen:
            yield c
            seen.add(c)
            c = self.container_at(c, t)

    def _scene_from(self, container: str | None, t: int, start: str) -> str | None:
        return next((c for c in self._up(start, container, t) if self.is_scene(c)), None)

    def scene_of(self, eid: str, t: int) -> str | None:
        """The nearest container up eid's chain that is a scene -- eid
        itself, if it is one (v5 §11). None: in no scene."""
        if self.is_scene(eid):
            return eid
        c = self.container_at(eid, t)
        found = self._scene_from(c, t, eid)
        if found is not None or c is None:
            return found
        # v5 §11 (Marc, 2026-09-23): a chain that reaches no scene has one --
        # its topmost container, unless that is a perceiver (the cup in
        # Teo's coat, Teo in nothing: Teo is no one's scene). The crossing,
        # 513: four travellers on a floating path were each alone.
        root = self.root(eid, t)
        return root if root is not None and root != eid and not self.is_perceiver(root) else None

    def root(self, eid: str, t: int) -> str | None:
        """The topmost container up eid's chain, or None if eid is in nothing."""
        top = None
        for top in self._up(eid, self.container_at(eid, t), t):
            pass
        return top

    def is_root_scene(self, sid: str | None) -> bool:
        """A container standing in as a scene by v5 §11's root rule."""
        return bool(sid) and not self.is_scene(sid)

    def chain(self, eid: str, t: int) -> list[str]:
        """What eid is in, and what that is in, up to and including its
        scene -- the actor's own chain (v5 §26)."""
        out: list[str] = []
        for c in self._up(eid, self.container_at(eid, t), t):
            out.append(c)
            if self.is_scene(c):
                break
        return out

    def above(self, scene: str | None, t: int) -> list[str]:
        """What holds a scene, nearest first, up to the root -- the regions
        the scene sits in (the telepath playtest: the shaft named from the
        headframe yard was minted in nothing, the mine above the yard never
        having been shown)."""
        return list(self._up(scene, self.container_at(scene, t), t)) if scene else []

    # -- the tree, shown (v4.30) ----------------------------------------------

    def holds(self, container: str, t: int) -> list[str]:
        """What is directly in `container` before t, in the canon's order."""
        order = self._canon_order()
        ix = self._now(t)
        if ix is not None:
            return sorted((x for x in ix.kids.get(container, ()) if x in order), key=order.__getitem__)
        return [x for x in self.canon.entities if self.container_at(x, t) == container]

    def is_place(self, eid: str, t: int, depth: int = 3) -> bool:
        """Structure, never a quality (v5 §1): a scene; what holds a place, a few
        levels down (a village, a county, a realm); or what no one perceives
        from and has no scene anywhere above it -- a region just named, not yet
        holding anything. A cup on a table or a purse in a pack is not: a scene
        holds it. A region planted with a first tuple perceives by that tuple,
        so it is a place by what it holds."""
        if self.is_scene(eid):
            return True
        if depth > 0 and any(self.is_place(x, t, depth - 1) for x in self.holds(eid, t)):
            return True
        return (not self.is_perceiver(eid, t)
                and not any(self.is_scene(c) for c in self._up(eid, self.container_at(eid, t), t)))

    def label(self, eid: str) -> str:
        e = self.canon.entities.get(eid)
        return self._first_name.get(eid) or (e.labels[0] if e is not None and e.labels else eid)

    def tree(self, scene: str | None, t: int) -> list[tuple[str, list[str]]]:
        """What the Narrator reasons with when it places a new place (v4.30;
        Marc, 2026-09-28: "making sure the narrator has the tree to reason
        with"): the scene and what holds it, from the world's top down, each
        with the places it directly holds. Bounded by the chain's depth times
        what each level holds -- never the world."""
        if not scene:
            return []
        chain = [scene] + self.above(scene, t)
        on = set(chain)
        return [(h, [x for x in self.holds(h, t) if x not in on and self.is_place(x, t)]) for h in reversed(chain)]

    def held(self, scene: str | None, t: int, shown: Iterable[str] = (), depth: int = 2, cap: int = 8) -> list[tuple[str, str, str]]:
        """What the scene holds, and what holds it holds: things, regions and
        scenes no more than `depth` steps inside the scene itself or each
        holder up its chain, that perceive nothing and are not already shown
        -- (id, first name, container). The scene first, then the nearest
        holder, and so up; within each, latest active first (a tuple of its
        own or set in it; for a silent thing, a record moving it), nearer
        breaking a tie; bounded by `cap`. A thing with no tuple of its own is
        never retrieved; this is how it stays seen -- the telepath playtest's
        cage at Eleven, and the crossing's Wenna's daughter (536), declared
        in the yard eleven ticks before Teo knocked, cut by the cap behind
        quiet places one step nearer, and minted again at the door."""
        if not scene:
            return []
        holders = [scene] + self.above(scene, t)
        ex = set(shown) | set(holders)
        level = {h: n for n, h in enumerate(holders)}
        found: list[tuple[int, int, int, str, str, str]] = []
        ix = self._now(t)
        if ix is not None:
            # now: down from each holder through what each thing directly holds, `depth` steps,
            # never through another holder -- the walk up stops at the nearest one (the same set
            # as the pass over the world, without the world)
            cand: set[str] = set()
            for h in holders:
                frontier, seen_down = [h], {h}
                for _ in range(depth):
                    nxt = []
                    for c in frontier:
                        for y in ix.kids.get(c, ()):
                            if y not in seen_down:
                                seen_down.add(y)
                                cand.add(y)
                                if y not in level:
                                    nxt.append(y)
                    frontier = nxt
            order = self._canon_order()
            universe = sorted((e for e in cand if e in order), key=order.__getitem__)
        else:
            universe = list(self.canon.entities)
        for eid in universe:
            if eid in ex or eid not in self._first_name or self.is_perceiver(eid, t):
                continue
            c, steps, seen = self.container_at(eid, t), 1, {eid}
            first = c
            while c is not None and c not in level and c not in seen and steps < depth:
                seen.add(c)
                c = self.container_at(c, t)
                steps += 1
            if c not in level:
                continue
            # its tuples, or tuples set in it; a silent thing's own records
            # (a seed restating where a place is, is not something happening)
            ticks = [w for w in self._active.get(eid, ()) if w <= t] or [r[0] for r in self._rec.get(eid, ()) if r[0] <= t]
            latest = max(ticks, default=-1)
            found.append((level[c], -latest, steps, eid, self._first_name[eid], first))
        return [(eid, name, first) for _n, _l, _s, eid, name, first in sorted(found)[:cap]]

    def inside(self, eid: str, container: str, t: int) -> bool:
        """Whether `container` is eid itself or anywhere up eid's chain --
        a move of `container` into eid would make a cycle (v5 §10)."""
        return any(c == container for c in self._up(None, eid, t))

    # -- the tree of scenes ------------------------------------------------

    def parent_scene(self, scene: str, t: int) -> str | None:
        return self._scene_from(self.container_at(scene, t), t, scene)

    # -- sandhi: what joins two scenes ---------------------------------------

    def joins(self, thing: str, t: int) -> set[str]:
        """The scenes a boundary thing is in before tick t: the scene its
        containment puts it in, and each other side a sandhi gives it. Empty
        when it has no sandhi or every side is closed -- not even the scene
        it is in."""
        entries = self._sandhi.get(thing)
        if not entries:
            return set()
        latest: dict[str, bool] = {}
        for tick, _o, scene, is_open in entries:
            if tick < t:
                latest[scene] = is_open
        sides = {x for x, is_open in latest.items() if is_open}
        if not sides:
            return set()
        own = self.scene_of(thing, t)
        return sides | ({own} if own else set())

    def sandhis(self, scene: str | None, t: int) -> list[str]:
        """The boundary things this scene shares with another, before t. The
        things that have a sandhi are few: bounded by the world's boundaries,
        never by its history."""
        if not scene:
            return []
        return sorted(x for x in self._sandhi if scene in self.joins(x, t))

    def across(self, scene: str | None, t: int) -> list[tuple[str, str]]:
        """(boundary thing, the scene on its other side) for each sandhi of
        this scene -- what is across it is perceived as the view allows."""
        return [(x, s) for x in self.sandhis(scene, t) for s in sorted(self.joins(x, t)) if s != scene and self.is_scene(s)]

    def neighbours(self, scene: str | None, t: int) -> list[str]:
        """Scenes within reach (v5 §12, v4.29): across a shared boundary thing
        (sandhi) first, then the tree's -- what this scene is directly in,
        if that is a scene; the scenes directly in this one; and the scenes
        directly in the same container, whatever that container is (Marc,
        2026-09-28: "a scene is just the lowest child ... the container
        directly above"). Whole, never capped: the cost is one container's
        children, never the world."""
        if not scene:
            return []
        joined = [s for _x, s in self.across(scene, t)]
        joined = [s for i, s in enumerate(joined) if s not in joined[:i]]
        return joined + [s for s in self._tree_neighbours(scene, t) if s not in joined]

    def _tree_neighbours(self, scene: str, t: int) -> list[str]:
        up = self.container_at(scene, t)
        out: list[str] = [up] if up is not None and self.is_scene(up) else []
        order = self._canon_order()
        ix = self._now(t)
        if ix is not None:  # now: what each container directly holds, looked up
            def held_by(c: str) -> list[str]:
                return sorted((s for s in ix.kids.get(c, ()) if s != scene and s in order and self.is_scene(s)),
                              key=order.__getitem__)
        else:  # the past: asked of every scene, as the scans answer
            def held_by(c: str) -> list[str]:
                return [s for s in self.canon.entities if s != scene and self.is_scene(s) and self.container_at(s, t) == c]
        out += [s for s in held_by(scene) if s not in out]
        if up is not None:
            out += [s for s in held_by(up) if s not in out]
        return out

    def occupants(self, scene: str | None, t: int, exclude: Iterable[str] = ()) -> set[str]:
        """Present in a scene: every perceiver whose own containment puts
        them there -- the spec's word (v5 §11), not a window."""
        if not scene:
            return set()
        ex = set(exclude) | self.absent
        ix = self._now(t)
        if ix is not None:  # now: who is in it, looked up -- the scene's own, not the world
            return {p for p in ix.members.get(scene, ()) if p in self._perceivers and self.is_perceiver(p)
                    and p not in ex and p != scene}
        return {p for p in self.perceivers if p not in ex and p != scene and self.scene_of(p, t) == scene}  # a perceiving scene is not present inside itself

    def absent_in(self, scene: str | None, t: int) -> set[str]:
        """The players out of the world whose place is kept in this scene."""
        if not scene:
            return set()
        return {p for p in self.absent if self.scene_of(p, t) == scene}

    def origin(self, eid: str) -> str | None:
        """What eid was first in, when that is no scene: the whole it came
        from. Decomposition is containment (Marc, 2026-09-28: the world
        builder decomposes by putting places in places) -- a part leaving a
        whole is declared in it, then moved out, and its first container
        says so for good. Cost: one lookup."""
        e = self.canon.entities.get(eid)
        first = e.kosha if e is not None else None  # set at birth by a seed, read as a first record would be (v5 §39)
        if first is None:
            recs = self._rec.get(eid)
            first = recs[0][2] if recs else None
        return first if first and first != eid and first in self.canon.entities and not self.is_scene(first) else None

    def departures(self, since: int, t: int) -> list[tuple[int, str, str, str | None]]:
        """What left a scene after tick `since` and before t: (tick, what, the
        scene it left, the scene it went to -- None for none). A containment
        record that took it out of the scene it stood in; what is carried
        leaves with its carrier and is counted once, by the carrier's record.
        Cost: the latest records of each entity that has any -- the world's
        entities, never its history."""
        out: list[tuple[int, str, str, str | None]] = []
        for eid, recs in self._rec.items():
            for w, _, _ in reversed(recs):
                if w <= since:
                    break
                if w >= t:
                    continue
                was, now = self.scene_of(eid, w), self.scene_of(eid, w + 1)
                if was and was != now:
                    out.append((w, eid, was, now))
        return out

    def contents(self, scene: str, t: int, since: int) -> set[str]:
        """Things in a scene whose containment was recorded lately (v5 §27's
        granularity bound): moved or minted since `since`."""
        ix = self._now(t)
        if ix is not None:  # now: what is in the scene, looked up
            return {e for e in ix.members.get(scene, ()) if e not in self._perceivers and self._rec.get(e)
                    and self._rec[e][-1][0] >= since}
        out = set()
        for eid, recs in self._rec.items():
            if eid in self._perceivers or not recs or recs[-1][0] < since:
                continue
            if self.scene_of(eid, t) == scene:
                out.add(eid)
        return out

    # -- names -------------------------------------------------------------

    def names_of(self, eid: str) -> set[str]:
        return set(self._names.get(eid, ()))

    def named(self, name: str) -> set[str]:
        return set(self._named.get(_fold(name), ()))

    def names_in(self, text: str, among: Iterable[str]) -> dict[str, set[str]]:
        """Declared names of the entities `among` found whole in `text`
        (word-bounded, case-folded): {entity: names found}. Exact -- a
        name is compared, never interpreted (v5 §16).

        Cost: the text's words and the names keyed by them -- not the
        world's names, nor how many the actor has met (2026-09-25: one
        regex per acquaintance's name was three quarters of retrieval).
        A spaced name that matches has every word of it standing whole in
        the text -- the lookarounds bound its ends and its own punctuation
        bounds the rest -- so only names keyed by one of the text's words
        can match; unspaced ones are few, and each is tried."""
        low = " " + re.sub(r"\s+", " ", text.casefold()) + " "
        among = set(among)
        candidates = set(self._unkeyed)
        for run in set(_RUN.findall(low)):
            candidates |= self._by_run.get(run, set())
        out: dict[str, set[str]] = {}
        for n in candidates:
            ents = self._named.get(n)
            if not ents or ents.isdisjoint(among) or not self._name_heavy.get(n) or n not in low:
                continue
            # v5 §16: the script decides -- a word-bounded match where words are
            # spaced (a hyphen joins a name: "Marsh-Corwin" is not "Corwin"),
            # a plain one where they are not (李白 inside 我见到李白了)
            if not unsegmented(n) and re.search(whole_word(n), low) is None:
                continue
            for eid in ents & among:
                out.setdefault(eid, set()).add(n)
        return out


def _before(recs: list[tuple[int, int, str | None]], t: int) -> int:
    """Index of the last record with world_time < t, or -1."""
    lo, hi = 0, len(recs)
    while lo < hi:
        mid = (lo + hi) // 2
        if recs[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def whole_name(name: str, titles: Iterable[str] | None = None) -> bool:
    """A name made whole: two words at least once titles and abbreviations
    are set aside -- a given name and a family name, or a byname ("Ruth
    Pell", "Grim of Stakkar", "Li Wei"); never "Ruth", "Mr. Brinsmade" or
    "Brother Magnús". A description ("the smith's widow") is no name and
    is not held to it (Marc, 2026-09-25: names are made full). `titles`:
    the world's own (world data "name_titles", v5 §16: an English list becomes
    world data); none given, today's English list."""
    n = name.split(",")[0].split()
    if not n or n[0].lower() in ARTICLES:
        return True
    set_aside = (_NOT_NAMES | _TITLES) if titles is None else {t.lower() for t in titles}
    return len([w for w in n if not w.endswith(".") and w.lower().strip("'") not in set_aside]) >= 2


def begins_with_name(name: str, taken: Iterable[str]) -> list[str]:
    """The names in `taken` (casefolded) that `name` begins with, whole --
    followed by nothing, a space or a comma: "Grim of Stakkar" begins with
    Grim (v5 §17: he is Grim); "Unnr of Kross" does not begin with Kross, nor
    "Walter Pike's apprentice" with Walter Pike (someone else, his)."""
    folded = name.casefold()
    return sorted(t for t in taken if t != folded and re.match(re.escape(t) + r"(?=$|[\s,])", folded))


def personal_name_parts(name: str) -> set[str]:
    """The capitalised words of a PERSONAL name, up to any comma: nothing
    of a description (a name that begins with an article), titles and
    abbreviations excepted. "Edward Lowell" -> {Edward, Lowell};
    "Asst. Prof. N. Thompson" -> {Thompson}; "the Lowell boy" -> {}."""
    n = name.split(",")[0].strip()
    if not n or n.split()[0].lower() in ARTICLES:
        return set()
    out: set[str] = set()
    for tok in n.split():
        if tok.endswith("."):
            continue  # an abbreviation: "Asst.", "Prof.", "N."
        out |= {p for p in _proper_names(tok) if len(p) >= 3 and p.lower() not in _NOT_NAMES}
    return out
# words that stand before a name and are not one (NAMESAKES)
_NOT_NAMES = frozenset({"sergeant", "captain", "officer", "mister", "miss", "mrs", "mr", "dr", "prof", "asst", "rev", "reverend",
                        "sheriff", "judge", "mayor", "deacon", "nurse", "governor", "general", "colonel", "major", "lieutenant",
                        "sir", "dame", "madam", "madame", "monsieur", "herr", "frau", "don", "dona", "senor", "senora"})
_TITLES = frozenset({"brother", "father", "sister", "mother", "bishop", "saint", "lord", "lady", "king", "queen",
                     "master", "mistress", "professor", "doctor", "old", "young", "aunt", "uncle"})


def _proper_names(label: str) -> set[str]:
    """A legacy label's proper names, for entities with no declared name:
    its capitalised words, a leading article and titles excepted -- the
    derivation v3 used (context.label_names_of, since removed), applied once on read."""
    out = set()
    for i, tok in enumerate(label.split()):
        w = tok.strip(",.;:!?()\"")
        if re.search(r"['’]s$", w):
            continue  # a possessive names its owner, not this: "Corwin's heart" is not Corwin
        parts = w.split("-")
        # hyphenated names stay whole (v5 §16: "Marsh-Corwin" is not "Corwin")
        capitalised = all(len(x) >= 2 and x[0].isupper() and x[1:].isalpha() and x[1:].islower() for x in parts)
        if capitalised and not (i == 0 and w.lower() in ARTICLES) and w.lower() not in _TITLES:
            out.add(w)
    return out
