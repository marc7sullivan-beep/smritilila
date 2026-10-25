# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Prakṛti (प्रकृति, nature -- Sāṅkhya's mūlaprakṛti, root-nature, the
unmanifest from which everything manifest unfolds): the seed lies at the
root and grows through the branches (spec v4.53, v5 §15; Marc and Anu,
2026-10-01).

A world's givens are written onto its earth as records and read from the
tree -- the nearest that says, else the earth, else the engine's own:

  the world's own view   -- `clause`, note "within:darsana", on the earth;
                            shown where the world's view is shown today
                            (the system prompt), parity first
  a capability default   -- `capability`, note "within:on" / "within:off"
  everything else        -- `prakriti`, note the key, value its JSON
                            (pacing, daiva, clock, phala, source, ...; the
                            library of becomings under "becomings")

World data stays as the record of what the gods gave at birth; once a
world's earth bears its seed, nothing reads world data again. A god
changing the seed appends what changed; the latest stands. Nothing here
reads the text of anything (v5 §1): keys and values are carried, not judged.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

from smritilila.ids import new_id
from smritilila.memory.record import Record

DARSANA = "darsana"      # the world's own view: a clause on the earth, reaching within
BECOMINGS = "becomings"  # the library of becomings: prakriti
CAPABILITIES = "capabilities"  # world data's capability defaults: capability records, reaching within

View = tuple[str | None, dict, dict]  # (darsana, becomings, world_data), as WorldSession holds it
EMPTY: View = (None, {}, {})


def _same(a: object, b: object) -> bool:
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)


def sow(earth: str, old: View, new: View, t: int, pulse_id: str) -> list[Record]:
    """The records that carry the seed from `old` to `new` on the earth:
    only what changed; a key taken away is withdrawn (value None), a
    capability taken away is off, as its default is. From EMPTY, the whole
    seed. Insert-only: nothing written is ever changed."""
    (d0, b0, w0), (d1, b1, w1) = old, new
    out: list[Record] = []
    if (d0 or None) != (d1 or None):
        out.append(Record(new_id(), "clause", earth, d1 or None, t, pulse_id, note=f"within:{DARSANA}"))
    if not _same(b0 or {}, b1 or {}):
        out.append(Record(new_id(), "prakriti", earth, json.dumps(b1, ensure_ascii=False) if b1 else None, t, pulse_id,
                          note=BECOMINGS))
    c0, c1 = dict(w0.get(CAPABILITIES) or {}), dict(w1.get(CAPABILITIES) or {})
    for cap in sorted(set(c0) | set(c1)):
        if cap in c1 and (cap not in c0 or bool(c0[cap]) != bool(c1[cap])):
            out.append(Record(new_id(), "capability", earth, cap, t, pulse_id, note=f"within:{'on' if c1[cap] else 'off'}"))
        elif cap not in c1 and c0.get(cap):
            out.append(Record(new_id(), "capability", earth, cap, t, pulse_id, note="within:off"))
    for key in sorted((set(w0) | set(w1)) - {CAPABILITIES}):
        if key in w1 and (key not in w0 or not _same(w0[key], w1[key])):
            out.append(Record(new_id(), "prakriti", earth, json.dumps(w1[key], ensure_ascii=False), t, pulse_id, note=key))
        elif key not in w1:
            out.append(Record(new_id(), "prakriti", earth, None, t, pulse_id, note=key))
    return out


def sown(records: Iterable[Record], earth: str | None) -> bool:
    """Whether the earth bears the seed: any record of it there."""
    return earth is not None and any(r.entity_id == earth and _of_seed(r) for r in records)


def _of_seed(r: Record) -> bool:
    return (r.kind == "prakriti"
            or (r.kind == "clause" and r.note == f"within:{DARSANA}")
            or (r.kind == "capability" and (r.note or "").startswith("within:")))


def reap(records: Iterable[Record], earth: str) -> View:
    """The world's view as its earth bears it -- the latest of each, in
    the order the records stand (world time, then as written)."""
    darsana: str | None = None
    becomings: dict = {}
    data: dict = {}
    caps: dict = {}
    mine = [r for r in records if r.entity_id == earth and _of_seed(r)]
    for r in sorted(mine, key=lambda r: r.world_time):  # stable: records of one tick keep the order they were written
        if r.kind == "clause":
            darsana = r.value or None
        elif r.kind == "capability":
            caps[r.value] = (r.note or "").endswith(":on")
        elif r.note == BECOMINGS:
            becomings = json.loads(r.value) if r.value is not None else {}
        elif r.value is None:
            data.pop(r.note, None)
        else:
            data[r.note] = json.loads(r.value)
    if caps:
        data[CAPABILITIES] = caps
    return darsana, becomings, data
