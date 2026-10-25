# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Saṃskāra (संस्कार): the impressions a world's history has left, kept and
added to -- so that no pulse reads the whole history again (2026-09-25).

A pulse used to rebuild its reading of the world from every memory and
record ever written: the containment index (Kosha), the log retrieval
reads (Smrti), the clock (Kala), and the scheduler's passes over memory.
Each grew with history. The saṃskāra is those readings, kept on the
world and READ ON: each commit's memories and records are added to a new
version (Kosha.extend, extend_log, Kala.extend, MemIndex.extend), sharing
everything it does not change with the one before (copy-on-write). A
pulse takes the version current when it starts and forks the containment
index for its own writes; a pulse that is held and let go leaves nothing
behind, and a commit landing meanwhile does not change what it reads.

Where reading on could differ from a fresh reading, the version is not
made and the saṃskāra is built afresh: memories rewritten (an absorb), the
players, the clock or the capability defaults changed, the canon lost or
reordered an entity, a record that changes a tuple already read (a repair
after the fact), a clock span that belongs mid-list, an entity that lived
as a person turning out to be a scene. Each is proved against a fresh
reading by replay, read-only, on every world (the stage-2 replays of
2026-09-25): the only difference that remains is the order of two
commits' memories within one tick, and the log's seq numbers -- replay's
cost, accepted by Marc.

Cost of a commit: what it wrote, and the tables' keys (the world's
entities and names) -- never its history.
"""
from __future__ import annotations

import bisect
import copy
from collections.abc import Iterable
from dataclasses import dataclass, field

from smritilila.graph.canon import Maya
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import Smrti
from smritilila.smrti.kala import Kala, age_spans, chronicle_ids
from smritilila.smrti.kosha import NOWHERE, Kosha
from smritilila.smrti.log import build_log, extend_log


@dataclass
class MemIndex:
    """What the scheduler asks of memory, kept (the routes, itihasa and
    WorldSession.left_at read it too): each entity's latest memory,
    each scene's own rows in order, each entity's memories in commit order,
    the memories of each tick, and the players' own pulses (not the copy of
    a row a player standing there is given)."""
    players: frozenset = frozenset()
    last_by: dict = field(default_factory=dict)       # entity -> its latest memory, by (tick, id)
    rows: dict = field(default_factory=dict)          # scene -> its own rows, by (tick, id)
    by_entity: dict = field(default_factory=dict)     # entity -> its memories, in commit order
    by_tick: dict = field(default_factory=dict)       # tick -> its memories, in commit order
    player_pulses: list = field(default_factory=list)  # players' own memories, by (tick, id)
    _player_keys: list = field(default_factory=list)  # their (tick, id), for bisection
    ambient: dict = field(default_factory=dict)       # pulse -> the texts of its scene rows (only added to)
    max_tick: int = -1

    def _own(self, attr: str, key) -> list:
        return getattr(self, attr).setdefault(key, [])

    def extend(self, memories: list[Memory]) -> "MemIndex":
        """These memories read on, IN PLACE: only code on the event loop reads
        this index (the scheduler, the routes), where commits are read on too --
        never a pulse in its thread -- so no version needs keeping."""
        self._read(memories)
        return self

    def _read(self, memories: list[Memory]) -> None:
        for m in memories:
            if m.tier == "AMBIENT":
                self.ambient.setdefault(m.pulse_id, set()).add(m.experience)
        for m in memories:
            key = (m.world_time, m.id)
            self.max_tick = max(self.max_tick, m.world_time)
            last = self.last_by.get(m.entity_id)
            if last is None or key > (last.world_time, last.id):
                self.last_by[m.entity_id] = m
            if m.tier == "AMBIENT" and m.entity_id == m.location_id:
                rows = self._own("rows", m.entity_id)
                bisect.insort(rows, m, key=lambda x: (x.world_time, x.id))
            self._own("by_entity", m.entity_id).append(m)
            self._own("by_tick", m.world_time).append(m)
            if m.entity_id in self.players and m.location_id != NOWHERE \
                    and m.experience not in self.ambient.get(m.pulse_id, ()):
                i = bisect.bisect_right(self._player_keys, key)
                self._player_keys.insert(i, key)
                self.player_pulses.insert(i, m)

    def player_pulses_after(self, tick: int) -> list[Memory]:
        """The players' own memories later than `tick` (the cadence's count)."""
        return self.player_pulses[bisect.bisect_right(self._player_keys, (tick, "\U0010ffff")):]


@dataclass
class Samskara:
    """One version of the world's kept reading (see the module)."""
    log: Smrti                     # the retrieval log; log.kosha is the containment index, log.places the scenes
    kala: Kala | None              # the clock, for a world that keeps one
    mem: MemIndex
    players: frozenset
    clock: dict | None
    defaults: dict | None
    n_mem: int = 0                 # how much of the world's memory list is read, and the last read
    last_mem: Memory | None = None
    n_rec: int = 0
    last_rec: Record | None = None
    age: dict = field(default_factory=dict)  # chronicle entry's pulse id -> (began, ended): the age's clock

    @classmethod
    def build(cls, canon: Maya, memories: list[Memory], records: list[Record], players: Iterable[str],
              clock: dict | None, defaults: dict | None) -> "Samskara":
        """The whole history, read afresh."""
        players = frozenset(players)
        log, _ = build_log(canon, memories, records, defaults=defaults)
        log.kosha.part_index()  # built once here, then read on
        mem = MemIndex(players=players)
        mem._read(memories)
        aside = chronicle_ids(canon)
        return cls(log, Kala.of(clock, records, memories, players, aside), mem, players, clock, defaults,
                   len(memories), memories[-1] if memories else None, len(records), records[-1] if records else None,
                   age_spans(records, aside))

    def extend(self, canon: Maya, memories: list[Memory], records: list[Record]) -> "Samskara | None":
        """A new version with a commit read on, or None: build afresh."""
        kosha = self.log.kosha.extend(canon, memories, records)
        if kosha is None:
            return None
        log = extend_log(self.log, canon, kosha, memories, records)
        if log is None:
            return None
        timed = [r for r in records if r.kind == "time"]
        aside = chronicle_ids(canon) if timed else frozenset()
        kala = None
        if self.kala is not None:
            if any(r.entity_id not in aside for r in timed):
                return None  # the first build's time records: Kala reads them only in a fresh reading (none written since)
            kala = self.kala.extend(memories)
            if kala is None:
                return None
        new = copy.copy(self)
        new.log, new.kala, new.mem = log, kala, self.mem.extend(memories)
        if timed:
            new.age = {**self.age, **age_spans(timed, aside)}  # the chronicle's own: the age moved
        return new

    def kosha_for(self, canon: Maya, absent: Iterable[str] = ()) -> Kosha:
        """A fork of the containment index for one reader: its own writes
        stay its own, the world's version is untouched."""
        return self.log.kosha.clone(canon=canon, absent=absent, fork=True)

    def log_for(self, canon: Maya, absent: Iterable[str] = ()) -> Smrti:
        """The log for one pulse: the world's streams (read-only), and its
        own fork of the containment index."""
        log = copy.copy(self.log)
        log.kosha = self.kosha_for(canon, absent)
        return log


def keep(sk: "Samskara | None", canon: Maya, memories: list[Memory], records: list[Record], players: Iterable[str],
         clock: dict | None, defaults: dict | None) -> tuple["Samskara", str | None]:
    """The saṃskāra for the world as it now stands: `sk` read on with what
    was committed since it was made (the lists grown at their ends), or built
    afresh -- with why, or None when it was read on (or unchanged)."""
    players = frozenset(players)
    why = None
    if sk is None:
        why = "first"
    elif sk.players != players or sk.clock != clock or sk.defaults != defaults:
        why = "players, clock or defaults changed"
    elif len(memories) < sk.n_mem or (sk.n_mem and memories[sk.n_mem - 1] is not sk.last_mem) \
            or len(records) < sk.n_rec or (sk.n_rec and records[sk.n_rec - 1] is not sk.last_rec):
        why = "history rewritten"
    if why is None:
        tail_m, tail_r = memories[sk.n_mem:], records[sk.n_rec:]
        if tail_m or tail_r or sk.log.kosha._n_canon != len(canon.entities):
            new = sk.extend(canon, tail_m, tail_r)
            if new is None:
                why = "a commit a fresh reading would read differently"
            else:
                new.n_mem, new.last_mem = len(memories), memories[-1] if memories else None
                new.n_rec, new.last_rec = len(records), records[-1] if records else None
                return new, None
        else:
            return sk, None
    return Samskara.build(canon, memories, records, players, clock, defaults), why
