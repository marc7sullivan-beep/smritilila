# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Kāla (काल, time): a world's clock across players -- spec v5 §19-§23
(Marc, 2026-09-24).

The tick orders pulses and says nothing of duration. A world that keeps a
clock (world_data["clock"] = {"epoch", "start", "confirm_minutes"}) has
every pulse report how long it took -- or, for a world with no calendar of
ours (an invented one), {"calendar": "days", "start", ...}: its days are
counted from the first ("the second day, 5:00 in the morning") -- and:

- PEOPLE CARRY TIME. A clocked pulse's minutes (began, ended) are carried
  by each memory written in it (Marc: "aren't time records just another
  field associated with a memory?"). A person's timeline is read from them:
  a span for each pulse they lived -- voiced in it, or present in its scene
  -- from their own previous moment to its end, from where they last were
  to where their memory of it is. Threads apart run at their own paces.
  (The first build wrote a `time` record per person per pulse; those are
  still read, for their pulses.)
- PLACES SHOW THEMSELVES AS OF THE VIEWER'S TIME. A memory carries its
  pulse's minute; a viewer at T is shown nothing later, and nobody who
  arrives after T is present at T. Memory is ordered by the clock.
- CHANGES AND LIVED MOMENTS LOCK A SCENE; LOOKING DOES NOT. S, seen from T,
  is locked until the latest of: each structural change in S after T that
  still shows in S (a thing taken that was there, a thing come to be there
  since) and, for each person who has lived past T and was in S at T, the
  moment they left S (or their own time, if they are still there).
- THE MIRROR. A span already lived in a scene by others is shown to a pulse
  that passes through it, which must hold it.
- EVERY JUMP IS CONSENTED TO (the prompts are smrti/pulse.py's, `locked_prompt`
  and a long jump's `pending`; the consent is routes/spanda.py's, `_confirmed`).
- NPCs FOLLOW AND NEVER LEAD (pravrtti.py's kāla gate, from `clock`).

The clock logic here is numbers of minutes, never read for meaning (s0);
watch_phrases and names_any read labels and names as text, and the *_line
functions are prompt prose.
Cost: one pass over the memories to index them when the clock is first read
(Kala.of) -- as build_log already makes -- then each commit's memories read
on (Kala.extend, kept by samskara.py), and lookups after that. Nothing is
stored beyond two integers on each memory.
"""
from __future__ import annotations

import bisect
import copy
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.kosha import NOWHERE
from smritilila.text import ARTICLES

MAX_ELAPSED = 2_000_000_000  # a bound for the integer column (about 3,800 years), not a rule: the Narrator declares how long a pulse takes (Marc, 2026-10-11; it was a day)
FIRST_WINDOW = 60      # a place or person with no time of its own is first asked about the hour before its reference
CONFIRM_MINUTES = 30   # a player's pulse longer than this asks first (the world's may differ); 30 from 2026-10-16 (Marc: every world, to match Stonemarch's), was 10: a new world is born with it
ITIHASA = "itihasa"    # Entity.origin of the world at large (genesis.py plants it)
AGE_LEAD = 131_400     # a season, in minutes: the chronicle waits while further ahead of the latest player


def chronicle_ids(canon) -> frozenset[str]:
    """The world at large: the one actor whose time is the age's, not a
    scene's -- its entries span weeks and months, and share ticks with
    everyone else's pulses (the collide copy's tick 666 held three)."""
    return frozenset(e.id for e in canon.entities.values() if e.origin == ITIHASA)


def age_spans(records: Iterable[Record], chronicle: frozenset[str]) -> dict[str, tuple[int, int]]:
    """Each chronicle entry's span, by the pulse that wrote it: its `time`
    record (note "began||"). An entry without one is untimed -- the birth
    year, or an entry written before the age kept time."""
    out: dict[str, tuple[int, int]] = {}
    for r in records:
        if r.kind == "time" and r.entity_id in chronicle and r.value is not None:
            try:
                end = int(r.value)
                began = int((r.note or "").split("|")[0] or end)
            except ValueError:
                continue
            out[r.pulse_id] = (began, end)
    return out


def age_minutes(elapsed: object) -> int:
    """The span a chronicle entry says it covers: its own grain, uncapped
    (an age moves by weeks and months; a pulse's own span is the Narrator's as well, v5.31)."""
    try:
        return max(0, int(float(elapsed)))
    except (TypeError, ValueError):
        return 0


def age_line(kala: "Kala", began: int) -> str:
    """What the chronicle is shown of its own time."""
    return (f"THE TIME HERE: the age has reached {kala.say(began)}; your entry begins there. Give `elapsed_minutes`: the span it "
            "covers (a week is 10080, a month about 43800, a season about 131400).")


@dataclass
class Span:
    start: int
    end: int
    origin: str | None
    dest: str | None
    tick: int


@dataclass
class Kala:
    epoch: datetime
    start: int
    confirm_minutes: int
    days: bool = False  # a world without our calendar: its days are counted, never dated
    spans: dict[str, list[Span]] = field(default_factory=dict)      # person -> spans, by (end, tick)
    scene_clock: dict[str, int] = field(default_factory=dict)       # scene -> the latest minute anything happened there
    pulse_minute: dict[str, int] = field(default_factory=dict)      # pulse id -> the minute it ended (append-only; shared by versions)
    tick_minute: dict[int, int] = field(default_factory=dict)       # tick -> the minute its pulse ended (append-only; shared by versions)
    # The reading's own state, kept so extend() reads new memories on from it
    # (2026-09-25: the clock is not rebuilt from history each pulse).
    _players: frozenset = frozenset()
    _recorded: set = field(default_factory=set)       # pulses the first build's time records tell
    _scenes: set = field(default_factory=set)         # who has a scene's own row: not a person
    _where: dict = field(default_factory=dict)        # entity -> where its last memory is
    _voiced_ever: set = field(default_factory=set)
    _reached: dict = field(default_factory=dict)      # person -> the latest minute they have lived to
    _last_tick: int = -1                              # the latest tick read
    # Lookup indexes, per person, and per person and scene
    _ends: dict = field(default_factory=dict)         # person -> [span end], parallel to spans
    _starts: dict = field(default_factory=dict)       # person -> [span start]
    _mono: dict = field(default_factory=dict)         # person -> their starts never go back (the lookups may bisect)
    _touch: dict = field(default_factory=dict)        # (person, scene) -> [index of each span touching the scene]
    _leaves: dict = field(default_factory=dict)       # (person, scene) -> [(start, index) of each span leaving it]
    _in_scene: dict = field(default_factory=dict)     # scene -> everyone with a span touching it
    _arr: dict = field(default_factory=dict)          # person -> (the scene of their trailing stay, when they came in or None)

    @classmethod
    def of(cls, clock: dict | None, records: Iterable[Record], memories: Iterable[Memory] = (),
           players: Iterable[str] = (), aside: frozenset[str] = frozenset()) -> "Kala | None":
        """The world's clock, or None if the world has not opted in: read
        from the memories' minutes, and from the first build's `time`
        records for the pulses that have them. `players`: whose clocks move
        only on pulses of their own (an idle player keeps their own clock).
        `aside`: whose time records are not the scenes' time (the world at
        large: the age's clock -- age_spans reads them)."""
        days = bool(clock) and clock.get("calendar") == "days"
        if not clock or not (clock.get("epoch") or days):
            return None
        k = cls(datetime(2000, 1, 1) if days else datetime.fromisoformat(clock["epoch"]), int(clock.get("start", 0)),
                int(clock.get("confirm_minutes", CONFIRM_MINUTES)), days)
        k._players = frozenset(players)
        spans: dict[str, list[Span]] = defaultdict(list)
        for r in records:
            if r.kind != "time" or r.value is None or r.entity_id in aside:
                continue
            try:
                end = int(r.value)
            except ValueError:
                continue
            k.pulse_minute[r.pulse_id] = max(end, k.pulse_minute.get(r.pulse_id, end))
            k.tick_minute[r.world_time] = max(end, k.tick_minute.get(r.world_time, end))
            parts = (r.note or "").split("|")
            if len(parts) == 3:
                start = int(parts[0]) if parts[0] else end
                spans[r.entity_id].append(Span(start, end, parts[1] or None, parts[2] or None, r.world_time))
            else:  # a scene's own record: when something last happened there
                k.scene_clock[r.entity_id] = max(end, k.scene_clock.get(r.entity_id, end))
            k._last_tick = max(k._last_tick, r.world_time)
        k._recorded = set(k.pulse_minute)
        memories = sorted(memories, key=lambda m: m.world_time)
        k._scenes = {m.entity_id for m in memories if m.entity_id == m.location_id}
        k._reached = {p: ss and max(s.end for s in ss) for p, ss in spans.items()}
        k.spans = dict(spans)
        k._read(k._group(memories))
        for pid, ss in k.spans.items():
            ss.sort(key=lambda s: (s.end, s.tick))
        for pid in k.spans:
            k._index(pid)
        return k

    def extend(self, memories: Iterable[Memory]) -> "Kala | None":
        """A new version with these memories read on -- this one unchanged
        (a pulse reading it in its thread keeps it). None when they cannot
        be read on and the clock must be read again from history: a memory
        older than the latest read, an entity that has been living as a
        person turning out to be a scene (the full reading excludes it from
        the start), or a span that would belong mid-list (the full reading
        sorts them). Cost: the persons and scenes (their tables are copied),
        and what the memories touch -- not history."""
        memories = sorted(memories, key=lambda m: m.world_time)
        if not memories:
            return self
        if memories[0].world_time < self._last_tick:
            return None
        found = {m.entity_id for m in memories if m.entity_id == m.location_id} - self._scenes
        if any(x in self.spans or x in self._voiced_ever for x in found):
            return None
        k = copy.copy(self)
        for name in ("spans", "scene_clock", "_where", "_reached", "_ends", "_starts", "_mono", "_touch", "_leaves",
                     "_in_scene", "_arr"):
            setattr(k, name, dict(getattr(self, name)))
        k._scenes = self._scenes | found
        k._voiced_ever = set(self._voiced_ever)
        touched = k._read(k._group(memories), fresh=set())
        for pid in touched:
            ss, had = k.spans[pid], len(self.spans.get(pid, []))
            if any((ss[i - 1].end, ss[i - 1].tick) > (ss[i].end, ss[i].tick) for i in range(max(had, 1), len(ss))):
                return None  # a span that belongs mid-list: read again (the full reading sorts)
            k._index(pid, from_=had)
        return k

    @staticmethod
    def _group(memories: list[Memory]) -> list[list[Memory]]:
        groups: dict[str, list[Memory]] = {}
        for m in memories:
            groups.setdefault(m.pulse_id, []).append(m)
        return list(groups.values())

    def _read(self, groups: list[list[Memory]], fresh: set | None = None) -> set[str]:
        """Spans from the memories' minutes, pulse by pulse in tick order.
        Who lived a pulse: whoever is voiced in it, and whoever its scene
        rows show present who perceives (v5 §4 -- has been voiced by then; a
        scene is not a person). Each from their own previous moment (or
        the pulse's start, if they have none) to the minute their memory
        (or their scene's row) says they lived to -- the whole pulse where
        it ends, its first moment elsewhere; from where they last were to
        where their memory of it is.

        AN IDLE PLAYER KEEPS THEIR OWN CLOCK (Marc, 2026-09-25): a player
        lives only the pulses they have a memory of their own in -- their
        own actions, and the world's turns taken for them. Being present in
        a scene's row does not move them, nor does the copy of that row a
        player standing there is given (v5 §31's witnessed tuple: the row's text,
        word for word -- which is how it is known, old memories included).
        Ravenmoor: the floorboard, the notebook and the key in Danforth's
        back room took their turns and carried him 1:00 to 2:00.

        `fresh`: in extend(), the persons whose span list this version has
        already copied (copy-on-write: a list another version holds is never
        appended to). Returns the persons given spans."""
        scenes, players, where, reached = self._scenes, self._players, self._where, self._reached
        touched: set[str] = set()
        for ms in groups:
            pid = ms[0].pulse_id
            self._last_tick = max(self._last_tick, ms[-1].world_time)
            ended = max((m.ended for m in ms if m.ended is not None), default=None)
            rows = {m.experience for m in ms if m.entity_id == m.location_id}
            voiced = {m.entity_id: m.location_id for m in ms if m.tier != "AMBIENT" and m.entity_id not in scenes
                      and not (m.entity_id in players and m.experience in rows)}  # a player's copy of the row is not theirs
            self._voiced_ever |= set(voiced)
            if ended is not None and pid not in self._recorded:
                began = min((m.began for m in ms if m.began is not None), default=ended)
                tick = ms[0].world_time
                self.pulse_minute[pid] = ended
                self.tick_minute[tick] = max(ended, self.tick_minute.get(tick, ended))
                lived: dict[str, tuple[str, int]] = {}  # who -> (where, the minute they lived to)

                def live(x: str, loc: str, to: int) -> None:
                    if x not in lived or to > lived[x][1]:
                        lived[x] = (loc, to)
                for m in ms:
                    to = m.ended if m.ended is not None else ended
                    if m.entity_id == m.location_id:  # a scene's row: who was there
                        self.scene_clock[m.entity_id] = max(to, self.scene_clock.get(m.entity_id, to))
                        for x in m.subject_ids:
                            if x in self._voiced_ever and x not in scenes and x not in players:
                                live(x, m.location_id, to)
                    elif m.entity_id in voiced and not (m.entity_id in players and m.experience in rows):
                        live(m.entity_id, m.location_id, to)
                for x, (loc, to) in lived.items():
                    dest = None if loc == NOWHERE else loc
                    origin = where.get(x, loc)
                    origin = None if origin == NOWHERE else origin
                    start = max(reached[x] if reached.get(x) is not None else began, self.start)
                    if fresh is not None and x not in fresh:
                        self.spans[x] = list(self.spans.get(x, []))
                        fresh.add(x)
                    self.spans.setdefault(x, []).append(Span(min(start, to), max(to, start), origin, dest, tick))
                    touched.add(x)
                    reached[x] = max(to, start)
                    for sc in {origin, dest} - {None}:
                        self.scene_clock[sc] = max(to, self.scene_clock.get(sc, to))
            for m in ms:
                if m.entity_id != m.location_id:
                    where[m.entity_id] = m.location_id
        return touched

    def _index(self, pid: str, from_: int = 0) -> None:
        """The person's lookup indexes, from span `from_` on (0: afresh).
        Copy-on-write: every list is replaced, never appended to, so another
        version holding the old one keeps it."""
        ss = self.spans[pid]
        if from_ == 0:
            ends, starts, mono, arr = [], [], True, (None, None)
            touch_new: dict[str, list[int]] = {}
            leave_new: dict[str, list[tuple[int, int]]] = {}
        else:
            ends, starts = list(self._ends[pid]), list(self._starts[pid])
            mono, arr = self._mono[pid], self._arr[pid]
            touch_new, leave_new = {}, {}
        for i in range(from_, len(ss)):
            s = ss[i]
            if starts and s.start < starts[-1]:
                mono = False
            ends.append(s.end)
            starts.append(s.start)
            for sc in {s.origin, s.dest} - {None}:
                touch_new.setdefault(sc, list(self._touch.get((pid, sc), [])) if from_ else []).append(i)
            if s.origin is not None and s.origin != s.dest:
                leave_new.setdefault(s.origin, list(self._leaves.get((pid, s.origin), [])) if from_ else []).append((s.start, i))
            # arrived(): the trailing stay's scene, and the latest coming-in within it
            if s.dest == arr[0] and i > 0:
                if s.origin != s.dest:
                    arr = (arr[0], s.start)
            else:
                arr = (s.dest, s.start if s.origin != s.dest else None)
        self._ends[pid], self._starts[pid], self._mono[pid], self._arr[pid] = ends, starts, mono, arr
        for sc, idx in touch_new.items():
            self._touch[(pid, sc)] = idx
            if pid not in self._in_scene.get(sc, ()):
                self._in_scene[sc] = self._in_scene.get(sc, frozenset()) | {pid}
        for sc, lv in leave_new.items():
            self._leaves[(pid, sc)] = lv

    # -- where people are in time ----------------------------------------

    def clock(self, eid: str | None) -> int | None:
        """A person's time: where their timeline has got to."""
        ss = self.spans.get(eid or "")
        return ss[-1].end if ss else None

    def now(self, actor: str, wait_until: int | None = None) -> int:
        """The actor's time: their own, never before the clock began, or
        later if they agreed to wait (a locked scene's consent)."""
        own = self.clock(actor)
        if own is None:
            own = self.scene_clock.get(actor)  # a scene asked whether anyone came: when it last saw anything
        return max(x for x in (own, self.start, wait_until) if x is not None)

    def where_at(self, person: str, t: int) -> str | None:
        """The scene a person was in at minute t, from their own spans; None
        if t is before their timeline or after it."""
        ss = self.spans.get(person)
        if not ss or t > ss[-1].end:
            return None
        if self._mono.get(person):  # starts only go forward: the last span begun by t, found by bisection
            i = bisect.bisect_right(self._starts[person], t) - 1
            if i < 0:
                return None
            s = ss[i]
            return s.origin if t < s.end and s.origin != s.dest else s.dest
        at = None
        for s in ss:
            if s.start > t:
                break
            # a pulse that moves someone: in the place left until it ends, in the new one from then
            at = s.origin if t < s.end and s.origin != s.dest else s.dest
        return at

    def arrived(self, person: str, scene: str) -> int | None:
        """When a person last came into a scene, if their latest stay is in it;
        None if their spans never show them coming in (they were there before
        the clock), or their latest stay is in another scene."""
        last_scene, came = self._arr.get(person, (None, None))
        return came if last_scene == scene and person in self.spans else None

    def present_at(self, person: str, scene: str, t: int) -> bool:
        """Is someone whose containment puts them in the scene there at t?
        Not if they came in after t -- they are not there yet."""
        came = self.arrived(person, scene)
        return came is None or came <= t

    # -- locks -------------------------------------------------------------

    def locked_until(self, scene: str | None, t: int, viewer: str, changes: list[tuple[int, str]] = ()) -> int | None:
        """The minute the scene opens to someone at t, or None if it is open.
        `changes`: (minute, what) of the structural changes in it that still
        show (kala.changes_in, computed by the caller from containment).
        Only those whose spans ever touched the scene are asked."""
        return self.locked_why(scene, t, viewer, changes)[0]

    def locked_why(self, scene: str | None, t: int, viewer: str, changes: list[tuple[int, str]] = ()
                   ) -> tuple[int | None, list[tuple[str, str, int]]]:
        """locked_until, and why (Marc, 2026-09-27: "add that to the advance
        time Y/N message so the player knows why"): (until, reasons), each
        reason ("present", who, until they leave) -- a perceiver already there,
        further along -- or ("change", what, at what minute)."""
        reasons: list[tuple[str, str, int]] = []
        if not scene:
            return None, reasons
        until = None
        for person in self._in_scene.get(scene, ()):
            ss = self.spans[person]
            if person == viewer or not ss or ss[-1].end <= t:
                continue
            if self.where_at(person, t) != scene:
                continue
            if self._mono.get(person):
                lv = self._leaves.get((person, scene), [])
                j = bisect.bisect_left(lv, (t, -1))
                left = lv[j][0] if j < len(lv) else ss[-1].end
            else:
                left = next((s.start for s in ss if s.start >= t and s.origin == scene and s.dest != scene), ss[-1].end)
            until = left if until is None else max(until, left)
            if left > t:
                reasons.append(("present", person, left))
        for minute, what in changes:
            if minute > t:
                until = minute if until is None else max(until, minute)
                reasons.append(("change", what, minute))
        return (until, reasons) if until is not None and until > t else (None, [])

    def lived_in(self, scene: str | None, a: int, b: int, viewer: str) -> list[tuple[int, str]]:
        """The mirror: (minute, person) for others already in the scene
        during (a, b] -- a span someone has lived there. Only those whose
        spans ever touched the scene are asked."""
        if not scene or b <= a:
            return []
        out = []
        for person in self._in_scene.get(scene, ()):
            if person == viewer:
                continue
            ss = self.spans[person]
            idx = self._touch.get((person, scene), [])
            if self._mono.get(person):
                # ends and starts only go forward: the first span touching the scene that ends after a
                ends = self._ends[person]
                lo, hi = 0, len(idx)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if ends[idx[mid]] > a:
                        hi = mid
                    else:
                        lo = mid + 1
                if lo < len(idx) and ss[idx[lo]].start < b:
                    out.append((max(ss[idx[lo]].start, a), person))
                continue
            for i in idx:
                s = ss[i]
                if a < s.end and s.start < b:
                    out.append((max(s.start, a), person))
                    break
        return sorted(out)

    def say(self, minutes: int) -> str:
        """Friday, 27 April 1928, 11:40 at night -- or, counted: the first
        day, 11:40 at night. Before 5 is night; from 5 it is morning (Marc)."""
        t = self.epoch + timedelta(minutes=minutes)
        h = t.hour
        part = ("at night" if h < 5 or h >= 21 else "in the morning" if h < 12 else "in the afternoon" if h < 17 else "in the evening")
        clock = "midnight" if (h, t.minute) == (0, 0) else "noon" if (h, t.minute) == (12, 0) else f"{(h % 12) or 12}:{t.minute:02d} {part}"
        if self.days:
            n = (t - self.epoch).days
            return f"{f'the {_ORDINALS[n]} day' if n < len(_ORDINALS) else f'day {n + 1}'}, {clock}"
        return f"{t.strftime('%A')}, {t.day} {t.strftime('%B')} {t.year}, {clock}"


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def world_turn(kala: Kala, kosha, eid: str, scene: str | None, ref: int, now_tick: int) -> tuple[int, int] | None:
    """v5 §23, the world's own turn (an NPC, or a scene asked whether anyone
    came), deterministic: (the minute it begins, the most it may take), or
    None -- not drawn.

    - One with no time of its own begins the hour before its reference, not
      at the clock's first minute: nothing in it has been seen since, and
      the empty hours are not narrated one by one (Marc, 2026-09-25).
    - Its own scene, where someone has already lived past its time: it
      begins when they are gone (it waited), and ends before the next
      moment anyone lived there. A turn never narrates over a span lived
      in its own scene. (Next door is shown to it, not capped: see
      near_lived.)
    - At or past its reference: not drawn."""
    own = kala.now(eid)
    if kala.clock(eid) is None and eid not in kala.scene_clock:
        own = max(kala.start, ref - FIRST_WINDOW)
    if scene:
        until = kala.locked_until(scene, own, eid, changes_in(kosha, kala, scene, own, now_tick, set(kosha.perceivers)))
        if until:
            own = until
    if own >= ref:
        return None
    nxt = min((m for m, _ in kala.lived_in(scene, own, ref, eid) if m > own), default=ref)
    cap = min(ref, nxt) - own
    return (own, cap) if cap > 0 else None


def mirror(kala: Kala, scenes: Iterable[str | None], a: int, b: int, viewer: str, here_now: set[str],
           out_of_reach: frozenset[str] = frozenset(), holders=lambda who: ()) -> list[tuple[int, str]]:
    """The mirror (v5 §21): what was lived in these scenes after `a`, the
    viewer's now, during (a, b] -- as (minute, who). A span begun after `a`:
    someone came in later, shown whether or not they are here now. A span
    already running at `a`: shown only for someone not here now -- the racer
    asleep in the room the laggard is in, gone since. Someone here now whose
    own time ran ahead is company, already shown as present, not the future
    (Marc, 2026-09-30; 34 of 42 mirror reruns had told a correct telling
    again for it). kala.lived_in itself is unchanged: world_turn caps by it.

    `out_of_reach` (v4.31: a player whose time is their own, their past in
    your place) is never shown -- nor what they hold, by `holders` (what
    holds a thing now, nearest first): Teo's loaf end fired the Crossing's
    mirror three times after Teo himself was dropped from it."""
    return sorted({(m, who) for sc in scenes if sc for m, who in kala.lived_in(sc, a, b, viewer)
                   if (m > a or who not in here_now)
                   and who not in out_of_reach and not out_of_reach.intersection(holders(who))})


def apart(kala: Kala, who: Iterable[str], now: int, players: Iterable[str], carried: Iterable[str] = ()) -> frozenset[str]:
    """Whoever `now` is not the present of (Marc, 2026-09-30: each perceiver's
    time is their own). Anyone who has lived past it -- it is their past, out
    of reach (v4.27) -- and a player behind it, not there yet (v4.31), unless
    the world carries them (`carried`: away, allowing it, v5 §37). An NPC behind
    is not apart: NPCs follow, and a turn takes them along. Whoever has no time
    of their own yet is not apart. A turn neither moves, voices, hears, makes
    become, lists present nor gives a witnessed copy to anyone apart: Stonemarch,
    2026-09-29, Dunstan's turn 58 minutes behind moved Jakal and Spugnoir out
    of Corl's cabin; 33 NPCs voiced before their own time; Jakal given a memory
    of a moment 17 hours before he came. Cost: the persons asked, one clock
    each."""
    players, carried = set(players), set(carried)
    out = set()
    for x in who:
        c = kala.clock(x)
        if c is not None and (c > now or (x in players and x not in carried and c < now)):
            out.add(x)
    return frozenset(out)


def near_lived(kala: Kala, places: Iterable[str], a: int, b: int, viewer: str, players: Iterable[str]) -> list[tuple[int, str, str]]:
    """Players' spans already lived here and next door during (a, b], as
    (minute, who, where): shown to a world's turn, which holds them."""
    who = set(players)
    return sorted({(m, p, sc) for sc in places for m, p in kala.lived_in(sc, a, b, viewer) if p in who})


def near_line(kala: Kala, lived: list[tuple[int, str, str]]) -> str:
    """What players already lived here and nearby in this turn's time --
    it happened, and the narration holds it."""
    return ("ALREADY LIVED HERE AND NEARBY, IN THIS TURN'S TIME (these happened; the narration holds them): "
            + "; ".join(f"from {kala.say(m)}, {who} in {where}" for m, who, where in lived))


def window_line(kala: Kala, events: list[tuple[int, str, str]]) -> str:
    """What the world did in the time a long action takes (routes/spanda.py's
    window turns, run before the actor lives it): the action is lived up to
    the first of these that reaches the actor, and stops there."""
    return ("IN THE TIME THIS ACTION WOULD TAKE, THESE HAPPEN (they happened, in this order; the action goes on only until "
            "the first that reaches the actor -- seen, heard, come to them -- and stops there: the narration ends at that "
            "moment, and `elapsed_minutes` ends there too; what does not reach them passes unnoticed): "
            + " | ".join(f"at {kala.say(m)}, in {where}: {text}" for m, where, text in events))


def setting_line(rows: list[tuple[str, str]]) -> str:
    """v4.32: the window's quiet turns, shown to the actor's telling -- the neighbouring scenes' own accounts of
    the same hours, so the actor lives under the same sky. Replayed before building (the Crossing, Teo's
    13h50 night, the larger model): shown the crossing's and the well yard's frost, 3 of 3 told frost and rime, none rain;
    blind, the held telling had rain and a flooding river. Never a stop: nothing here reaches the actor."""
    return ("THE WORLD AROUND, IN THIS ACTION'S TIME (the neighbouring scenes' own accounts of the same hours -- they "
            "happened; their weather, light, cold and water are the actor's too, and the narration and every tuple agree "
            "with them; nothing here reaches the actor or stops the action): "
            + " | ".join(f"in {where}: {text}" for where, text in rows))


def presence_line(kala: Kala, who: str, a: int, b: int, where: str | None, theirs: Iterable[str]) -> str:
    """A world's turn in a player's window is told only that they are there
    -- never what they do: their time is theirs to tell, and may yet be cut
    short. It does not describe them, where they are, or what is with them."""
    with_them = sorted(set(theirs) - {who, where})
    return (f"IN THIS TURN'S TIME A PLAYER IS AT {where or 'a place nearby'}, from {kala.say(a)} to {kala.say(b)}: {who}"
            + (f", with {', '.join(with_them)}" if with_them else "")
            + ". That place, the player and what is with them are not this turn's to tell: do not describe them at all -- "
              "not seen, not heard, not their tracks. Narrate only this scene, and what comes to it or goes from it.")


def watch_phrases(kosha, ids: Iterable[str]) -> tuple[set[str], set[str]]:
    """What a row must not name: (name parts, compared whole and as written;
    label phrases -- each label's pieces between commas, a leading article
    dropped -- compared whole, any case). Only the engine's own strings: a
    description in other words ("the boy") is not caught (s0)."""
    parts: set[str] = set()
    phrases: set[str] = set()
    for eid in ids:
        parts |= kosha.name_parts(eid)
        e = kosha.canon.entities.get(eid)
        for label in (e.labels if e is not None else []):
            for piece in label.split(","):
                words = piece.strip().split()
                if words and words[0].lower() in ARTICLES:
                    words = words[1:]
                phrase = " ".join(words)
                if len(phrase) >= 4:
                    phrases.add(phrase.lower())
    return parts, phrases


def names_any(text: str, watched: tuple[set[str], set[str]]) -> list[str]:
    """The watched names and phrases a text uses, whole-word."""
    import re

    from smritilila.smrti.kosha import whole_word
    parts, phrases = watched
    hit = [p for p in parts if re.search(whole_word(p), text)]
    low = text.lower()
    return sorted(hit + [p for p in phrases if re.search(whole_word(p), low)])


def changes_in(kosha, kala: Kala, scene: str, t: int, now_tick: int, perceivers: set[str]) -> list[tuple[int, str]]:
    """Structural changes in a scene after minute t that still show there
    now: (minute, thing). A thing that has come to be in the scene since (left,
    dropped, made and left), or one that was there and has been taken.
    People's own comings and goings are their spans, not changes; what was
    made and carried away does not show, and does not lock (Marc). Nor does a
    repair: a god's correction of the tree (a seed, a gravestone) is not
    something that happened in the scene (Marc, 2026-10-01: "a repair isn't
    a lived change")."""
    out = []
    for eid, recs in kosha._rec.items():
        if eid in perceivers:
            continue
        prev = kosha.canon.entities[eid].kosha if eid in kosha.canon.entities else None
        for rec in recs:
            tick, into = rec[0], rec[-1]
            if rec[1] in kosha._repairs:
                prev = into
                continue
            minute = kala.tick_minute.get(tick)
            before = kosha.scene_of(prev, now_tick) if prev else None
            after = kosha.scene_of(into, now_tick) if into else None
            if minute is not None and minute > t and (before == scene) != (after == scene):
                here_now = kosha.scene_of(eid, now_tick) == scene
                if (after == scene and here_now) or (before == scene and not here_now):
                    out.append((minute, eid))
            prev = into
    return out


def _minutes(elapsed: object) -> int:
    try:
        e = int(float(elapsed))
    except (TypeError, ValueError):
        return 1  # said nothing: a moment
    return max(0, min(MAX_ELAPSED, e))


def span_said(minutes: int) -> str:
    """A span as a person says it: 20 minutes; 3 hours, 5 minutes; 2 days, 1 hour."""
    d, r = divmod(max(0, int(minutes)), 1440)
    h, m = divmod(r, 60)
    parts = [f"{n} {unit}{'s' if n != 1 else ''}" for n, unit in ((d, "day"), (h, "hour"), (m, "minute")) if n]
    return ", ".join(parts) or "0 minutes"


def lived_span(world_minutes: int, rate: float) -> int:
    """A world span as lived where time runs at `rate` (v5 §19): world minutes
    per lived minute."""
    return max(0, min(MAX_ELAPSED, round(world_minutes / rate))) if rate != 1.0 else max(0, world_minutes)


def world_span(lived_minutes: int, rate: float) -> int:
    """Minutes lived where time runs at `rate`, as the world's (v5 §19)."""
    return max(0, round(lived_minutes * rate)) if rate != 1.0 else max(0, lived_minutes)


def time_line(kala: Kala, now: int, own: int, max_elapsed: int | None = None, rate: float = 1.0) -> str:
    """What the Narrator is shown: the time here -- and, if the actor's own
    last moment was earlier, that the time between passed without them
    doing anything (they agreed to wait).

    A world's turn (`max_elapsed`, its span from kala.world_turn) is told
    the span it covers and to fill it, with more done the longer it is
    (Marc, 2026-09-30). It was told the longest it MAY take, and took a
    median 23% of it in Stonemarch: its NPC stayed most of a day behind and
    was owed again at once. Replayed on 15 of Stonemarch's turns, told this,
    every telling covered its span in steps -- a gate watch kept two days,
    a badger's night -- and held the one player lived in it at his time.

    Where time runs at its own pace (v5 §19, `rate` world minutes per minute
    lived here), the clock shown is the world's, and every span the
    Narrator gives or is given is the time lived here."""
    line = f"THE TIME HERE: {kala.say(now)}. The narration and every tuple agree with it."
    if rate != 1.0:
        line += (f" TIME RUNS AT ITS OWN PACE HERE: an hour lived here is {span_said(world_span(60, rate))} in the world beyond. "
                 "The time above is the world's; every span below, and `elapsed_minutes`, is time as lived here.")
    if max_elapsed is None:
        line += (" Give `elapsed_minutes`: how long this pulse takes in the world from that moment -- a glance or a word 0-1, a "
                 "conversation a few minutes, a walk across town 20-40, a wait or a sleep as long as it is.")
    if own < now:
        line += (f" The actor's own last moment was {kala.say(own)}: the time between has passed with them waiting, doing nothing "
                 "anyone would remember. Narrate that it is later now.")
    if max_elapsed is not None:
        n = lived_span(max_elapsed, rate)
        line += (f" THIS TURN COVERS THE NEXT {span_said(n)} -- give `elapsed_minutes` {n}. Tell what is done here across all of "
                 "that time, not one moment of it. The longer the span, the more is done: a few minutes is one action; an hour "
                 "or more is several actions in order, each taking its time and leading to the next -- work begun and finished, "
                 "a watch kept and relieved, a journey with its stops; a day holds a day's doings, and its night. Nothing after "
                 "its end: the world does not run ahead of its players.")
    return line


def presents_line(kala: Kala, now: int, presents: list[tuple[str, int]], meet_at: int | None = None) -> str:
    """v5 §20 v4.31: the presents within reach, shown before the telling -- (where, their present), soonest
    first. Where and when, never who (v4.27: whoever is lived further along is not shown; a meeting shows
    them). Replayed before building (Stonemarch t154, the east track, the larger model): shown Jakal's present, the
    telling stopped at it 3 of 3 and ended as they met; without it, 20-25 minutes and "nobody here yet".
    `meet_at`: set only by the meeting rerun -- the telling ran past this present, and is told it plainly."""
    line = ("PRESENTS WITHIN REACH (someone whose time is their own stands at their present in each): "
            + "; ".join(f"at {where}, as of {kala.say(c)}" for where, c in presents)
            + ". Before that moment nobody is there for the actor. If this action brings the actor there by then, the "
              "pulse ends at that moment, as they come upon whoever is there: elapsed_minutes is exactly the time until "
              "then (" + ", ".join(f"{c - now} for {where}" for where, c in presents) + "), the narration ends as they "
              "meet -- nothing after it, nothing said of who it is beyond what the actor sees at that moment -- and "
              "nothing is written for them.")
    if meet_at is not None:
        line += f" THIS PULSE REACHES THEM: it ends at {kala.say(meet_at)}, elapsed_minutes {meet_at - now}, as they meet."
    return line


def lived_line(kala: Kala, lived: list[tuple[int, str]]) -> str:
    """The mirror: moments others have already lived in the scene the pulse
    passes through -- they happened, and the narration holds them."""
    return ("ALREADY LIVED HERE, AFTER YOUR NOW (these happened; the narration holds them -- the time passes through them "
            "or ends before them): " + "; ".join(f"at {kala.say(m)}, {who} was here" for m, who in lived))
