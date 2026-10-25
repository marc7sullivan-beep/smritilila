# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The world's memories as the log (v5 §3, §4; v5 §10-§11): the bridge
from the store's rows -- entities, memories, records -- to the tuples
retrieval reads. Each entity is seeded with a first tuple that names it
(v5 §39); every memory becomes a tuple filed at the SCENE it happened in.
The containment index (smrti/kosha.py) rides on the log, so retrieval
reads where anyone is from the same object it reads what they remember.
Nothing here reads a quality (v5 §1).
"""
from __future__ import annotations

from collections.abc import Iterable

from smritilila.graph.canon import Maya
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import NOWHERE, Smrti, Tuple
from smritilila.smrti.kosha import Kosha


def v3_enclosure(canon: Maya, place_id: str | None, kosha: Kosha | None = None) -> str | None:
    """The scene a stored location stands for (v5 §11). A memory is
    filed at a scene; a location that is not a scene (a v4 write never
    makes one, but a hand-made row might) is read at the scene its
    containment reaches. Kept under its v3 name for its callers. Pass the
    index at hand: without one, a whole Kosha is built for the one answer
    (thread_from_memory did that once per memory, every player pulse)."""
    if not place_id or place_id == NOWHERE:
        return place_id
    k = kosha if kosha is not None else Kosha(canon)
    return k.scene_of(place_id, 10**9) or place_id


# Minting is meeting (the crossing, 523 and 602): whatever a pulse declared --
# named or placed -- is known to everyone who wrote their own tuple in that
# pulse, as if their tuple had been about it. Tam named his father's iron at
# 600; asked for "the soldering iron" at 602, the Narrator could not see it
# and minted it again, handle and all. Bounded by the records of one's own
# pulses. Replayed at 602 (the larger model and the second model, 3 draws each): minted again 6/6
# without it, the existing iron reused 6/6 with it.


def build_log(canon: Maya, memories: Iterable[Memory], records: Iterable[Record] = (),
              defaults: dict | None = None, absent: Iterable[str] = ()) -> tuple[Smrti, set[str]]:
    """The world's memories as tuples, each entity seeded with a first
    tuple that names it. Returns the log and the set of scene ids (what
    v3 called places: the streams that are impersonal records). The log
    carries the containment index (`log.kosha`) and, per perceiver, who
    their own tuples have been about (`log.acquainted`, v5 §17's bound).
    The whole history, read afresh: tests and replays; the world keeps
    its log and reads each commit on (extend_log)."""
    memories = list(memories)
    records = list(records)
    kosha = Kosha(canon, memories, records, defaults=defaults, absent=absent)
    log = Smrti()
    log.kosha = kosha
    _seed(log, canon, list(canon.entities))
    _read(log, memories, records)
    return log, log.places


def extend_log(log: Smrti, canon: Maya, kosha: Kosha, memories: Iterable[Memory], records: Iterable[Record]) -> Smrti | None:
    """A new version of the log with a commit read on: the entities it
    brought seeded, its memories appended -- the same reading as
    build_log's, for what is new. `kosha`: the containment index with the
    commit already read on (Kosha.extend). This version is unchanged.
    Cost: the commit, and the log's tables' keys; never its history.
    Its seq numbers go on from this version's: a fresh build numbers
    differently (same-tick order may differ; nothing else can -- replay's
    cost, accepted by Marc, 2026-09-25).

    None when the commit would change what is already read: a record that
    decorates a pulse read in an earlier commit (a line said, a mark, a
    name, a time -- a repair appended after the fact: the crossing's 623,
    Sela's line corrected later). The log is then built afresh."""
    memories, records = list(memories), list(records)
    if any((r.kind in _DECORATES or (r.kind == "kosha" and r.note == "declared")) and r.pulse_id in log._pulses
           for r in records):
        return None  # it would change tuples already read (a pulse with no memories has none to change)
    new = log.clone()
    new.kosha = kosha
    keys = list(canon.entities)
    _seed(new, canon, keys[log._n_canon:])
    _read(new, list(memories), list(records))
    return new


_DECORATES = frozenset({"said", "mark", "name", "time"})  # records a tuple is read with (_read), and "kosha" declared
_SEED_BASE = 10**9  # first tuples take seqs below every memory's (_seed)


def _seed(log: Smrti, canon: Maya, eids: list[str]) -> None:
    """Each entity's first tuple: its label -- a scene's (or v3 place's)
    filed at itself."""
    kosha = log.kosha
    for eid in eids:
        e = canon.entities[eid]
        # a first tuple's seq is its entity's place in the canon, below every memory's: at one tick
        # it comes before the memories, in the canon's order -- as a fresh reading has it, and the
        # same whether the log is built afresh or read on (2026-09-25)
        seq = log._n_canon - _SEED_BASE
        log._n_canon += 1
        if not e.labels:
            continue
        if kosha.is_scene(e.id):
            log.places.add(e.id)
            log.append(Tuple(e.id, seq, e.created_at_world_time - 1, e.labels[0], 6, e.id, None))
        else:
            log.append(Tuple(e.id, seq, e.created_at_world_time - 1, e.labels[0], 6, None, None))


def _read(log: Smrti, memories: list[Memory], records: list[Record]) -> None:
    """Memories as tuples, in tick order, with what the records say of
    them: the line said, "as narrated", the minute their pulse ended, and
    whom each perceiver has met."""
    kosha = log.kosha
    declared_in: dict[str, set[str]] = {}
    for r in records:
        if r.kind == "name" or (r.kind == "kosha" and r.note == "declared"):
            declared_in.setdefault(r.pulse_id, set()).add(r.entity_id)
    for r in records:  # v5.33: what an experience closes, as the Narrator wrote it
        if r.kind == "closes" and r.value and r.value.lstrip("-").isdigit():
            log.close(r.entity_id, r.world_time, int(r.value))
    # v5 §2: the line spoken in a scene, and an actor tuple stored as
    # narrated, are records -- shown to the Narrator here, never written
    # into the memory itself. A witnessed copy of the scene's record (same
    # pulse, same text) shows the line too.
    scene_text = {(m.entity_id, m.pulse_id): m.experience for m in memories if m.tier == "AMBIENT"}
    # v5 §19: the minute each pulse ended (a world that keeps a clock) -- on its
    # memories, or, for the first build's pulses, in its time records
    minute_of: dict[str, int] = {}
    for r in records:
        if r.kind == "time" and r.value and r.value.lstrip("-").isdigit():
            minute_of[r.pulse_id] = max(int(r.value), minute_of.get(r.pulse_id, int(r.value)))
    for m in memories:
        if m.ended is not None:
            minute_of[m.pulse_id] = max(m.ended, minute_of.get(m.pulse_id, m.ended))
    copied: set[str] = set()
    for m in sorted(memories, key=lambda m: (m.world_time, m.id)):
        log._pulses.add(m.pulse_id)
        log._seq += 1
        place = None if m.location_id == NOWHERE else m.location_id
        if place and not (kosha.is_scene(place)):
            place = kosha.scene_of(place, m.world_time + 1) or place
        text = m.experience
        said = kosha.said.get((m.location_id, m.pulse_id))
        if said and (m.tier == "AMBIENT" or scene_text.get((m.location_id, m.pulse_id)) == m.experience):
            line, speaker = said
            text = f'{text} -- said{" by " + speaker if speaker else ""}: "{line}"'
        if "as_narrated" in kosha.marks.get((m.entity_id, m.pulse_id), ()):
            text = f"[as narrated to them, not in their own words] {text}"
        log.append(Tuple(m.entity_id, log._seq, m.world_time, text, m.salience, place,
                         tuple(m.embedding) if m.embedding else None, m.use, minute_of.get(m.pulse_id)))
        if m.tier in ("PRIMARY", "SECONDARY"):
            # met (v5 §17): whoever their tuples were with, and the scenes they stood in
            if log._owned is not None and m.entity_id not in copied:  # a version: its own set, once
                log.acquainted[m.entity_id] = set(log.acquainted.get(m.entity_id, ()))
                copied.add(m.entity_id)
            met = log.acquainted.setdefault(m.entity_id, set())
            met.update(m.subject_ids)
            met.update(declared_in.get(m.pulse_id, ()))
            if place:
                met.add(place)


