# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Janaśruti (जनश्रुति), "what people hear" -- hearsay. Marc, 2026-10-02: "how rumours and information spread ... the
goblin raid that occurred in Stonemarch. News of that should be able to spread through Millhaven."

The trace (2026-10-02): the world had one path for news, the age's chronicle reaching a quiet scene; a local event had
none. Goblins killed Farmer Hobb at an outlying farmstead at ticks 849-851 and not one memory in the world mentioned
goblins after -- it reached only whoever walked there. Now a moment the world weighed heavily becomes talk that travels.

By structure, the engine reading no kind (v5 §1):
  what   -- a pulse in which someone's memory was weighed indelible or marking -- told by its scene's own impersonal
            record of that pulse, never a person's private memory (minds stay private). A planted being's first pulse, its
            birth, is identity, not news.
  where  -- outward along the tree: the scene it happened in, up through what holds it, and down to other scenes, one
            step at a time. Indelible travels farther and faster than marking.
  when   -- on the world's clock (the pulse's minutes; ticks in a world that keeps none): a scene has heard once enough
            time has passed for each step; after a horizon it is old news and no longer talked of.
  shown  -- TALK HERE in the prompt of a pulse in a scene it has reached: as heard, not witnessed, and the Narrator may let
            it come out partial or wrong in the telling (no ground truth). When the telling writes it into someone's
            memory, they carry it on; the engine seeds only the first hearing.
A branch may close itself to hearsay: pacing {"hearsay": false} on it or above (v5 §15).
"""
from __future__ import annotations

# by weight (1 indelible, 2 marking): how long each step of the tree takes to cross, and how many steps it carries
STEP_MINUTES = {1: 45, 2: 120}
MAX_STEPS = {1: 6, 2: 3}
STEP_TICKS = {1: 1, 2: 3}       # a world that keeps no clock: steps in ticks
HORIZON_MINUTES = 3 * 24 * 60   # old news after three days
HORIZON_TICKS = 150
SCAN_TICKS = 450                # the scan's own bound in ticks, clock or no clock: never the world's whole history
TALK_SHOWN = 3                  # what a pulse is shown of what has reached its scene, freshest first


def _ancestry(kosha, x: str, t: int) -> list[str]:
    out, seen = [], set()
    while x is not None and x not in seen:
        out.append(x)
        seen.add(x)
        x = kosha.container_at(x, t)
    return out


def steps_between(kosha, a: str, b: str, t: int) -> int | None:
    """Tree steps from scene a to scene b: up to what holds both, and down. None if nothing holds both."""
    up_a, up_b = _ancestry(kosha, a, t), _ancestry(kosha, b, t)
    pos_b = {x: i for i, x in enumerate(up_b)}
    for i, x in enumerate(up_a):
        if x in pos_b:
            return i + pos_b[x]
    return None


def heard(kosha, entities, memories, here: str, t: int, now_minute: int | None) -> list[tuple[str, str, int, int]]:
    """What has reached `here` by now and is still talked of: [(origin scene, its record of the moment, age, weight)],
    freshest first, at most TALK_SHOWN. `memories` in the world's order. Age is in minutes, or ticks without a clock.
    Cost: the memories of the last SCAN_TICKS ticks (read from the newest back), and a tree path for each heavy moment."""
    if here is None or kosha.seed("pacing", here, t, field="hearsay") is False:
        return []
    clocked = now_minute is not None
    heavy: dict[str, int] = {}      # pulse -> its heaviest personal weight
    records: dict[str, tuple] = {}  # pulse -> (scene, record, ended, tick)
    for m in reversed(memories):
        if t - m.world_time > SCAN_TICKS:
            break
        if clocked:
            when = m.ended if m.ended is not None else m.began
            if when is None:
                continue
            if now_minute - when > HORIZON_MINUTES:
                continue  # minutes need not follow ticks across people's own times: skipped, not stopped
        elif t - m.world_time > HORIZON_TICKS:
            break
        e = entities.get(m.entity_id)
        if e is not None and e.origin == "genesis" and m.world_time <= e.created_at_world_time:
            continue  # a planted being's birth -- who it is, never news. One the telling minted lived its first pulse:
            # Farmer Hobb was minted in the pulse he died, and that pulse is the news
        if m.tier == "AMBIENT" and m.entity_id == m.location_id:
            records.setdefault(m.pulse_id, (m.location_id, m.experience, m.ended if m.ended is not None else m.began, m.world_time))
        elif m.salience in STEP_MINUTES:
            heavy[m.pulse_id] = min(heavy.get(m.pulse_id, 9), m.salience)
    out = []
    for pid, w in heavy.items():
        rec = records.get(pid)
        if rec is None:
            continue  # no public account of it: nobody could have told it
        origin, text, ended, tick = rec
        if origin == here:
            continue  # it happened here: this scene's own record already holds it
        n = steps_between(kosha, origin, here, t)
        if n is None or n > MAX_STEPS[w]:
            continue
        age = (now_minute - ended) if clocked and ended is not None else (t - tick)
        if age < 0 or age < n * (STEP_MINUTES[w] if clocked else STEP_TICKS[w]):
            continue  # not yet: the news is still on its way
        out.append((origin, text, age, w))
    out.sort(key=lambda x: x[2])
    seen, one_each = set(), []
    for x in out:  # one item from each place it came from, the freshest
        if x[0] not in seen:
            seen.add(x[0])
            one_each.append(x)
    return one_each[:TALK_SHOWN]


def talk_line(kosha, items: list[tuple[str, str, int, int]], clocked: bool) -> str:
    def ago(n: int) -> str:
        if not clocked:
            return f"{n} turns of the world ago"
        if n < 90:
            return "within the hour"
        if n < 24 * 60:
            return f"about {round(n / 60)} hours ago"
        return f"about {round(n / 1440)} days ago"
    parts = [f'from {kosha.label(o)}, {ago(a)}: "{text.strip()[:320]}"' for o, text, a, _w in items]
    return ("TALK HERE (what has reached this place by now, as people here would have HEARD it, never as witnessed -- "
            "nobody here saw it unless their own memory says so; it may be told partly, wrongly or not at all, as talk "
            "is; never a revelation. A name in it is the place's record's, not the teller's: a teller names someone only "
            "if their own memory knows the name): " + "; ".join(parts))
