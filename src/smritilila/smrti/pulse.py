# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The pulse: retrieve -> narrate -> append (v5 §31), on v4's containment
(Design Principles v4 draft, 2026-09-23).

One function builds the v5 §26 context from the world's memories and
records, asks the Narrator, and turns what it returns into what is
appended: one tuple per perceiver present, one impersonal tuple per
scene touched, and v4's structural records -- containment moves (v5 §10),
declared names (v5 §16), and a player's own words (v5 §35).

Where anyone is, is what they are in (smrti/kosha.py). The Narrator says
what moved; the engine checks only structure -- the container exists, no
cycle, one container at a time, only what was shown moves -- and never
whether a move is possible, which is the Narrator's (Marc, 2026-09-23).
"""
from __future__ import annotations

import dataclasses
import hashlib

import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.ids import new_id
from smritilila.llm.client import LLMClient, bind_pulse_id
from smritilila.llm.handles import words_of
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import NOWHERE, PARTICIPATION_WINDOW, Context, Smrti, Tuple, retrieve
from smritilila.smrti.kosha import no_scenes
from smritilila.smrti.log import build_log, v3_enclosure
from smritilila.smrti.narrator import (
    AUTONOMOUS_ACTION, CHRONICLE_ACTION, DAIVA_WORDS, SCENE_ARRIVAL_ACTION, PulseOut, daiva_line, handle_map, holds_on, moment_lines, narrate,
    round_daiva_line, round_lines, salience_of, use_of,
)
from smritilila.sakshi import log_event
from smritilila.training_data import log_discarded
from smritilila.smrti.kala import (Kala, age_line, age_minutes, apart, changes_in, chronicle_ids, lived_line, lived_span, mirror, near_line, near_lived, presents_line,
                              setting_line, time_line, window_line, world_span, _minutes)
from smritilila.smrti.prakriti import DARSANA
from smritilila.text import runs, sentences, shared_run, units, weight
from smritilila.text import slug as _slug


# a trailing _01 or a copied id tail (_f9a213: the Narrator imitating our own minted ids) is not part of the name
_TAIL = re.compile(r"_(\d+|[0-9a-f]{6})$")
OWN_TURN = "(their own turn)"   # what THIS MOMENT says of one whose turn is the world's own, with no action of theirs


def _humanize(mnemonic: str) -> str:
    return _TAIL.sub("", mnemonic).replace("_", " ").strip() or "something"  # no words: not "someone" -- the engine does not say it is a person (v5 §1)


_MARKUP = re.compile(r"</?(narration|parameter|invoke|function_calls?)\b|<parameter ", re.I)


def clean_narration(text: str, canon: Maya, pulse_id: str) -> str:
    """What the player reads must be prose: cut at any tool markup the
    model leaked into the narration string (Tombstone tick 230), and
    replace any id that still appears with its plain words -- the id's own
    words as v4.55's handles have them, never a label that may be a name
    the reader has not heard; a hex id has only its label."""
    m = _MARKUP.search(text)
    if m:
        log_event("smrti.narration_markup", pulse_id=pulse_id, at=m.start())
        text = text[: m.start()].rstrip()
    leaked = 0
    for eid, e in canon.entities.items():
        if len(eid) >= 12 and eid in text:
            text = text.replace(eid, words_of(eid, e.labels or []) or "someone")
            leaked += 1
    if leaked:
        log_event("smrti.narration_id_leak", pulse_id=pulse_id, count=leaked)
    return text


@dataclass
class SmrtiPulseResult:
    narration: str
    memories_written: list[Memory]
    new_entities: list[Entity]
    suggested_next_action: str
    context_tuples: int
    bound_ids: list[str]
    resolved_new: dict[str, str] = field(default_factory=dict)  # NEW:x -> id (minted or folded)
    dropped: list[str] = field(default_factory=list)  # ids the Narrator used that exist nowhere
    elsewhere: list[str] = field(default_factory=list)  # ids the Narrator put in a scene they are not in
    sara: str = ""
    said: str = ""
    turn: str = ""
    suggested_next_action_from_memory: str = ""
    records_written: list[Record] = field(default_factory=list)  # v4: containment, names, the player's own words
    route: list[str] = field(default_factory=list)  # sandhi: where it began, the scenes its boundaries joined, where it ended
    scene: str | None = None  # the actor's scene at the pulse's end
    # v5 §22: a jump the player must consent to before this pulse is committed --
    # {"kind": "long" | "locked", "prompt": str, "minutes"/"until": int}; None: commit. "locked" arises only for a
    # world's own turn (a player goes into a place lived further along, v5 §21), and that turn is let go (npc_locked)
    pending: dict | None = None
    # saṅgama: each other player of the round -- {player: {"narration", "suggested_next_action"}}
    round_narrations: dict[str, dict] = field(default_factory=dict)
    elapsed: int | None = None  # the pulse's minutes, in a world that keeps a clock (phala reckons after long ones)


def _fallback_context(canon: Maya, actor: str, log: Smrti) -> Context | None:
    """An actor with no stream but a label (its first tuple): in no scene,
    with only that."""
    tuples = log.stream(actor)[:1]
    if not tuples:
        return None
    return Context(actor=actor, place=None, tuples=tuples, present=set())


def nearest_id(handle: str, shown: set[str], max_distance: int = 2) -> str | None:
    """An id the Narrator garbled by a character or two (Vadhu tick 34) is
    the one shown id within that distance -- exactly one, or none."""
    if len(handle) < 8:
        return None
    close = []
    for sid in shown:
        if len(sid) != len(handle):
            continue
        d = sum(1 for a, b in zip(sid, handle) if a != b)
        if d <= max_distance:
            close.append(sid)
    return close[0] if len(close) == 1 else None


def mnemonic_id(mnemonic: str, fresh: str) -> str:
    """An opaque id that is still copyable: the mnemonic's words plus a
    short unique tail. 'drover_mule_man' -> 'drover_mule_man_5157a0'."""
    slug = _TAIL.sub("", _slug(mnemonic, "_"))[:40] or "x"  # v5 §7: any script
    return f"{slug}_{fresh[:6]}"


_MNEMONIC = re.compile(r"^[^\W\d_]+(?:_[^\W\d_]+)+$")  # snake_case words of any script, no digits: a mnemonic, never a hex id


def new_handle(handle: str) -> str:
    """The NEW marker wherever the Narrator put it -- `NEW:x`, `x_NEW`,
    `NEW_x`, `new:x` -- as `NEW:x` (Vadhu tick 211)."""
    h = handle.strip()
    m = re.match(r"^(?:new[:_\-\s]+)(.+)$", h, re.I) or re.match(r"^(.+?)(?:[:_\-\s]+new)$", h, re.I)
    return "NEW:" + m.group(1).strip("_: ") if m else h


def unplaced_mention(kosha, canon: Maya, eid: str, t: int, earth: str | None) -> bool:
    """Minted in play and never placed: in the earth (or, before it was planted, in nothing) and nowhere else.
    A mention is filed in the earth (Marc, 2026-10-04: "otherwise it's unreachable"), so the earth, not nothing,
    is where an unplaced thing stands; nothing still reads as unplaced for what came before."""
    e = canon.entities.get(eid)
    return e is not None and e.origin == "smrti" and eid != earth and kosha.container_at(eid, t) in (None, earth)


@dataclass
class Binder:
    """The handle-to-id state of one pulse: what the Narrator was shown
    (and so may name by id), what it minted, and what it garbled.

    v5 §17: a NEW whose DECLARED name equals a name or alias of someone
    the actor already knows is folded onto them -- exact, never a label
    guess; two such is ambiguity, and mints. Label words fold nothing
    (the label-word fold was retired, v4.44). v3's partial guards -- qualifiers, head nouns,
    world-wide word sets -- are retired with places (v5 §17)."""
    canon: Maya
    ctx: Context
    places: set[str]
    world_time: int
    pulse_id: str
    id_factory: Callable[[], str]
    new_handles: frozenset[str]
    log: Smrti | None = None
    declared: dict[str, dict] = field(default_factory=dict)  # NEW:x -> its new_entities row
    shown: set[str] = field(default_factory=set)
    earth: str | None = None  # the world at large, where whatever a telling places nowhere is (Marc, 2026-09-30)
    new_entities: list[Entity] = field(default_factory=list)
    resolved: dict[str, str] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    mention_hits: dict[str, tuple[str, str]] = field(default_factory=dict)  # NEW:x -> (a mention by its name, the name)

    def __post_init__(self) -> None:
        self.shown = self.ctx.shown()
        self.earth = next(iter(chronicle_ids(self.canon)), None)

    @property
    def minted(self) -> set[str]:
        return {e.id for e in self.new_entities}

    @property
    def kosha(self):
        return self.log.kosha

    def acquaintance(self) -> set[str]:
        # a namesake is shown so the Narrator may name them by id -- never so
        # a NEW of the same name folds onto a stranger by itself (s34)
        acq = set(self.shown) - {n[0] for n in self.ctx.namesakes}
        acq |= self.log.acquainted.get(self.ctx.actor, set())
        return acq

    def resolve(self, handle: str, *, scene: bool = False) -> str | None:
        """The id a handle names, minting a NEW. `scene`: a NEW minted
        here with no declaration is a scene (a scene tuple's own id)."""
        canon = self.canon
        handle = str(handle or "").strip().strip(",;.'\"`").strip()
        if not handle:
            return None
        if handle not in canon.entities:
            handle = new_handle(handle)
        if handle in self.resolved:
            return self.resolved[handle]
        if handle.startswith("NEW:"):
            decl = self.declared.get(handle) or {}
            name = str(decl.get("name") or "").strip()
            hit = self._fold_by_name(handle, name) if name else None  # a declared name folds (v5 §17); label words never (v5 §17)
            if hit is None:
                eid = mnemonic_id(handle[4:], self.id_factory())
                label = name or _humanize(handle[4:])
                is_scene = bool(decl.get("scene")) if decl else scene
                ent = Entity(eid, [label], self.world_time, "smrti",
                             avarana="scene" if is_scene else None)  # the scene bit (v5 §11); v4 mints no svabhava
                canon.entities[eid] = ent
                self.new_entities.append(ent)
                if is_scene:
                    self.places.add(eid)
                hit = eid
                log_event("smrti.minted", pulse_id=self.pulse_id, handle=handle, id=eid, name=name or None, scene=is_scene,
                          declared=bool(decl))
            self.resolved[handle] = hit
            return hit
        if handle in canon.entities:
            return handle
        if "NEW:" + handle in self.new_handles:
            return self.resolve("NEW:" + handle, scene=scene)
        near = nearest_id(handle, self.shown)
        if near is not None:
            self.resolved[handle] = near
            return near
        if _MNEMONIC.match(handle):
            log_event("smrti.bare_mnemonic", pulse_id=self.pulse_id, handle=handle)
            return self.resolve("NEW:" + handle, scene=scene)
        self.dropped.append(handle)
        return None

    def _fold_by_name(self, handle: str, name: str) -> str | None:
        kosha = self.kosha
        known = kosha.named(name) & self.acquaintance()
        if len(known) == 1:
            hit = next(iter(known))
            log_event("smrti.name_fold", pulse_id=self.pulse_id, handle=handle, name=name, onto=hit)
            return hit
        if len(known) > 1:
            log_event("smrti.name_ambiguous", pulse_id=self.pulse_id, handle=handle, name=name, candidates=sorted(known))
            return None
        holders = kosha.named(name)
        if not holders:
            return None
        # a thing or a place -- no stream to tell two apart -- declared by
        # the same name in the same container is the one already there
        # (spec correction while building, v5 §17: the crossing 497, "the mill
        # path" minted twice in the same road by two perceivers)
        said_in = str((self.declared.get(handle) or {}).get("in") or "").strip()
        container = said_in if said_in in self.canon.entities else self.resolved.get(new_handle(said_in)) if said_in else None
        same = {h for h in holders if container is not None and not kosha.is_perceiver(h)
                and kosha.container_at(h, self.world_time + 1) == container}
        if len(same) == 1:
            hit = next(iter(same))
            log_event("smrti.name_container_fold", pulse_id=self.pulse_id, handle=handle, name=name, onto=hit, container=container)
            return hit
        # a thing named in passing, never reached, bearing this name: the telling is asked again, shown it, and the
        # Narrator says whether this is that one (the mention rerun). The engine never folds on a name alone: a
        # cottage door in Nulb is not Gunder's (replayed 2026-10-04: bound 8/8 where meant, 0 false folds of 10)
        ment = sorted(h for h in holders if unplaced_mention(kosha, self.canon, h, self.world_time + 1, self.earth))
        if ment:
            self.mention_hits[handle] = (ment[0], name)
        log_event("smrti.name_taken_elsewhere", pulse_id=self.pulse_id, handle=handle, name=name, holders=sorted(holders),
                  mention=ment[0] if ment else None)
        return None

    def label_of(self, eid: str) -> str:
        e = self.canon.entities.get(eid)
        return e.labels[0] if e and e.labels else eid


_HEARD = re.compile(r'^(?P<text>.*) -- (?:(?P<by>[^"]+?) said|Said): "(?P<said>[^"]*)"$', re.S)


_SEAM = re.compile(r'\s*--\s*(?:[^"\n]{1,80}?\s+said|Said|said(?:\s+by\s+[^:"\n]{1,80})?):\s*"[^"]*"\s*')  # v3's seam and v4's rendered line


def strip_seams(text: str, pulse_id: str = "") -> str:
    """The record's seam, wherever the Narrator wrote one into its own
    text, removed -- counted as smrti.seam_imitated."""
    cleaned, n = _SEAM.subn(" ", text)
    if n:
        log_event("smrti.seam_imitated", pulse_id=pulse_id, count=n)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def split_heard(text: str) -> tuple[str, str, str]:
    """The record, the line and the speaker, apart again."""
    m = _HEARD.match(text)
    if not m:
        return text, "", ""
    return m.group("text"), m.group("said"), m.group("by") or ""


def watched_tuples() -> frozenset[tuple[str, int]]:
    """Tuples under watch, from SMRITILILA_WATCH="<perceiver>:<tick>,..."."""
    raw = os.environ.get("SMRITILILA_WATCH", "").strip()
    out = set()
    for item in raw.split(","):
        who, _, tick = item.strip().rpartition(":")
        if who and tick.lstrip("-").isdigit():
            out.add((who, int(tick)))
    return frozenset(out)


THREAD_RECENT = 15  # the actor's last own tuples: whoever is in them has been "with the actor lately"


def thread_from_memory(canon: Maya, memories: Iterable[Memory], actor_id: str, present: set[str], here: str | None,
                       kosha=None) -> str:
    """The second suggestion's ground, chosen by the engine (2026-09-21):
    the PERSON the actor's weightier own tuples are most about, who is not
    here and not in the actor's last THREAD_RECENT tuples; else such a
    SCENE stood in and not lately; else nothing."""
    from smritilila.smrti.narrator import SALIENCE_NAMES
    memories = list(memories)
    has_stream = {m.entity_id for m in memories}
    own = sorted((m for m in memories if m.entity_id == actor_id), key=lambda m: (m.world_time, m.id))
    if len(own) <= THREAD_RECENT:
        return ""
    recent, older = own[-THREAD_RECENT:], own[:-THREAD_RECENT]
    lately = set(present) | {actor_id} | {p for m in recent for p in m.subject_ids}
    from smritilila.smrti.kosha import Kosha
    from smritilila.text import rare_words, tokens
    names = kosha if kosha is not None else Kosha(canon)
    enclosure = Kosha(canon)  # built once for the call (it was built once per memory); the canon's own reading, as before
    lately_places = {v3_enclosure(canon, m.location_id, enclosure) for m in recent} | ({here} if here else set())
    score: dict[str, float] = {}
    anchor: dict[str, tuple[int, int, int, Memory]] = {}

    # v5 §34: "is this tuple about them" -- a declared name, or a word of their
    # label that is RARE among the world's labels (measured from the world,
    # no stopword list in any language). Exact names alone lost the tin-man
    # of "the tin peddler off the hill roads" (replay, 28 of 60 threads moved)
    rare = rare_words([e.labels[0] for e in canon.entities.values() if e.labels])

    def note(key: str, m: Memory, need_words: bool = False) -> None:
        words = bool(names.names_in(m.experience, [key])) or bool(tokens(canon.entities[key].labels[0]) & rare & tokens(m.experience))
        if need_words and not words:
            return
        score[key] = score.get(key, 0.0) + (4 - m.salience) * (2 if words else 1)
        rank = (0 if words else 1, m.salience, -m.world_time, m)
        if key not in anchor or rank[:3] < anchor[key][:3]:
            anchor[key] = rank

    for m in older:
        if m.salience > 3:
            continue
        for p in m.subject_ids:
            e = canon.entities.get(p)
            if p == actor_id or p in lately or p not in has_stream or e is None or names.is_scene(p) or not e.labels:
                continue
            note(p, m, need_words=True)
    if not score:
        for m in older:
            where = v3_enclosure(canon, m.location_id, enclosure)
            e = canon.entities.get(where) if where else None
            if m.salience > 3 or where in lately_places or e is None or not e.labels or (m.subject_ids & set(present)):
                continue
            note(where, m)
    if not score:
        return ""
    # the id last: an exact tie was decided by hash order, and the second suggestion differed from one server
    # start to the next (2026-09-25, found by the context replay: Wentworth 582, 584)
    pick = max(score, key=lambda k: (score[k], anchor[k][3].world_time, k))
    a = anchor[pick][3]
    return f"{pick} ({canon.entities[pick].labels[0]}) -- [{a.world_time}, {SALIENCE_NAMES.get(a.salience, a.salience)}] {a.experience}"


def anusmrti(samskara, canon: Maya, actor_id: str, now: int, embed: Callable[[str], list[float]] | None,
             players: Iterable[str] = (), absent: Iterable[str] = (), view: dict | None = None) -> list[tuple[Tuple, str]]:
    """Anusmṛti (अनुस्मृति), recollection -- the calling to mind of what one
    has lived (Marc, 2026-10-02: "the agent isn't using the same retrieval
    that a player character does"). A being's own tuples as the world's
    retrieval selects them for its own turn -- the same call, the same
    rules, with no action given -- and nothing of anyone else's: no scene's
    record, no one else's stream. An agent player reads this, so it knows
    its character as the world does, no more and no less. Read-only: the
    log is the pulse's own fork. [(tuple, the rule that brought it)], in
    order. Cost: one retrieval, as a turn's."""
    absent = frozenset(absent) - {actor_id}
    log = samskara.log_for(canon, absent)
    if not log.stream(actor_id, before=now):
        return []
    world_data = (view or {}).get("world_data") or {}
    ctx = _context_for(canon, log, actor_id, None, now, embed, keep=frozenset(players),
                       pacing=world_data.get("pacing"), sandhi=bool(world_data.get("sandhi")) or SANDHI_ALL)
    return [(t, ctx.sources.get((t.perceiver, t.seq), "")) for t in ctx.tuples if t.perceiver == actor_id]


def context_sections(ctx: Context) -> dict[str, dict[str, int]]:
    """The context by the rule that supplied each tuple -- {rule: {tuples,
    tokens}} with tokens ~ chars/4."""
    out: dict[str, dict[str, int]] = {}
    for t in ctx.tuples:
        name = ctx.sources.get((t.perceiver, t.seq), "unsourced")
        d = out.setdefault(name, {"tuples": 0, "tokens": 0})
        d["tuples"] += 1
        d["tokens"] += (len(t.memory) + len(t.perceiver) + 24) // 4
    return out


def _context_for(
    canon: Maya, log: Smrti, actor_id: str, action_text: str | None, world_time: int,
    embed: Callable[[str], list[float]] | None, keep: frozenset[str] = frozenset(),
    pacing: dict | None = None, sandhi: bool = False,
) -> Context:
    q = np.asarray(embed(action_text or log.stream(actor_id)[-1].memory)) if (embed and (action_text or log.stream(actor_id))) else None
    ctx = retrieve(log, actor_id, world_time, q, action=action_text or "", keep=keep, pacing=pacing,
                   sandhi=sandhi)
    if ctx is not None and ctx.place is None and len(log.stream(actor_id, before=world_time)) > 1:
        return ctx  # in no scene: their own stream, nobody present
    if ctx is None or ctx.place is None:
        fb = _fallback_context(canon, actor_id, log)
        if fb is None:
            raise ValueError(f"{actor_id} has no memory")
        seen = {(t.perceiver, t.seq) for t in (ctx.tuples if ctx else [])}
        merged = sorted((ctx.tuples if ctx else []) + [t for t in fb.tuples if (t.perceiver, t.seq) not in seen],
                        key=lambda t: (t.chronology, t.seq))
        # the context kept whole, its tuples widened (a field-by-field copy dropped across, elsewhere,
        # namesakes and players_named -- 2026-09-25 review)
        ctx = dataclasses.replace(ctx, tuples=merged) if ctx else Context(actor=actor_id, place=None, tuples=merged, present=set())
    return ctx


def _clean(out: PulseOut, canon: Maya, actor_id: str, pulse_id: str) -> None:
    out.narration = clean_narration(out.narration, canon, pulse_id)
    for r in out.round_narrations:
        r["narration"] = clean_narration(str(r["narration"]), canon, pulse_id)
    out.sara = clean_narration(out.sara, canon, pulse_id) if out.sara else ""
    out.said = clean_narration(out.said, canon, pulse_id) if out.said else ""
    out.turn = clean_narration(out.turn, canon, pulse_id) if out.turn else ""


# -- placement (v5 §10-§12) --------------------------------------------------


@dataclass
class Placement:
    """Where everything the pulse touched is at its end."""
    scene_rows: list[tuple[str, dict]]           # (scene id, the Narrator's scene tuple), in its order
    records: list[Record]                        # records this pulse appended: containment, sandhi, becomings and capabilities
    elsewhere: list[str]                         # put in a scene they are not in, and not moved there
    moved: set[str]
    origin: str | None                           # the actor's scene when the pulse began
    t_after: int = 0
    route: list[str] = field(default_factory=list)  # origin, the scenes its boundaries joined, where it ended
    unshown: list[str] = field(default_factory=list)  # someone with a memory of their own, moved though never shown

    def scene_after(self, b: Binder, who: str) -> str | None:
        return b.kosha.scene_of(who, self.t_after)


def _record(b: Binder, entity: str, into: str | None, why: str) -> Record:
    if into is None and b.earth is not None and entity != b.earth:
        into = b.earth  # in nothing is in the earth: still in no scene, the earth being no one's (it perceives)
    r = Record(new_id(), "kosha", entity, into, b.world_time, b.pulse_id, note=why)
    b.kosha.add(r)
    return r


def _become(out: PulseOut, b: Binder, actor_id: str, players: frozenset[str], movable: frozenset[str],
            library: dict[str, str], away: frozenset[str] = frozenset()) -> list[Record]:
    """v5 §8: what a being became, as records -- a clause it bears (authored
    in play, so reaching the bearer alone) and any capability the structure
    must now allow. Structure only: the entity exists and was shown; a
    player becomes in another's turn only if they allow the world to act
    for them (v5 §37). Every becoming is logged for review."""
    kosha, out_records = b.kosha, []
    for r in out.becomings:
        ent = b.resolve(str(r.get("entity", "")))
        key = re.sub(r"[^\w]+", "_", str(r.get("key") or "").strip().lower()).strip("_")[:60]
        reason = None
        if ent is None or not key:
            reason = "unknown"
        elif ent not in b.shown and ent not in b.minted:
            reason = "not_shown"
        elif ent in players and ent != actor_id and ent not in movable:
            reason = "player_not_actor"
        elif ent in away:
            reason = "apart_in_time"  # kala.apart: this moment is not their present
        text = None if r.get("ends") else (str(r.get("clause") or "").strip() or library.get(key, "").strip())
        if reason is None and not r.get("ends") and not text:
            reason = "no_text"
        if reason is not None:
            log_event("smrti.becoming_refused", pulse_id=b.pulse_id, entity=ent, key=key, reason=reason)
            continue
        rec = Record(new_id(), "clause", ent, text, b.world_time, b.pulse_id, note=f"self:{key}")
        out_records.append(rec)
        kosha.add(rec)
        log_event("smrti.becoming", pulse_id=b.pulse_id, entity=ent, label=b.label_of(ent), key=key, ends=bool(r.get("ends")),
                  authored=bool(str(r.get("clause") or "").strip()), text=(text or "")[:300])
    for r in out.capabilities:
        ent = b.resolve(str(r.get("entity", "")))
        cap = r.get("capability")
        if ent is None or cap not in ("perceives", "reaches") or (ent not in b.shown and ent not in b.minted) \
                or (ent in players and ent != actor_id and ent not in movable):
            log_event("smrti.capability_refused", pulse_id=b.pulse_id, entity=ent, capability=cap)
            continue
        rec = Record(new_id(), "capability", ent, cap, b.world_time, b.pulse_id, note="on" if r.get("on") else "off")
        out_records.append(rec)
        kosha.add(rec)
        log_event("smrti.capability", pulse_id=b.pulse_id, entity=ent, capability=cap, on=bool(r.get("on")))
    return out_records


# v5 §37, shown (Ravenmoor, 423): the Narrator was told which players it MAY
# move and never which ones it may not -- a suspended player present was
# narrated as gone ("the hall is empty") while the record kept him there.

# Walking in (Ravenmoor, 478): Danforth, on the library steps, went to the
# lecture hall "to check whether the man who sat at the front is still
# there". The pulse showed the library steps' present, not the hall's; the
# Narrator had only Nathaniel's last line as memory ("stayed seated", 463)
# and wrote the hall empty -- into Danforth's, the hall's and Nathaniel's
# own streams -- while the record kept him there; Wentworth found him
# standing at 500. Where the actor ends up is known only once the Narrator
# has answered, and a destination named in the action matches no declared
# name (typed "the prehistory lecture hall"; declared "lecture hall
# prehistory"). So the check is made after: a pulse that takes its actor
# into a scene holding a player it was not shown is narrated once more,
# shown who is there. A second call only on those pulses (Marc: B).

# A name already borne (Ravenmoor playtest, Marc 2026-09-24: "the Narrator is
# reusing almost identical names, confusing to players"). Wentworth's thread
# named the man in the reading room Elias Lowell (554) while Danforth's had
# an Edmund Lowell; replayed, the Narrator said "Edward Lowell" 2/2 even when
# shown that Edmund existed -- the register and the player's own "Are you
# Lowell?" outweigh a line. So the check is made after, on what was written:
# a pulse that gives someone a name sharing a name part with a DIFFERENT
# person already in the world is narrated once more, told whose name it is.
# Kin by the name stay possible. A second call only on a collision.


PLACES_BEYOND = 14  # the most places a place rerun shows of one region

# A holder that lists this many places is full (Marc, 2026-10-12: neighbours are one container's children, so a container's width is
# what a pulse pays for, and it is bounded at creation, not by a cap on what is shown). A new place made directly in a full holder is
# narrated once more, told the holder is full and to make one new place for the group inside it. Replayed on 23 flat placements
# (the second model): with an escape ("or put it in one of its places if it is truly part of it") about 14 of 23 reruns were
# sensible; with none, about 20 of 23; the prompt alone (a count on the tree line and a clause in rule 7) did not bind (13 of 18
# flat against 18 of 26 today). Built and watched in play (Marc); tests/test_fanout.py has tested holders_full since 2026-10-13.
FANOUT_CAP = 8


def fill_holds(kosha, ctx: Context, t: int) -> None:
    """SMRITILILA_HOLDS (Marc, 2026-10-13: everything is a container and a container can hold things): what each thing the ACTOR holds holds, or
    that it holds nothing, so the prompt shows `jakal-spellbook (empty)` and `the pair (holding: a dagger, a knife)` where it showed bare
    names. PRESENT shows what the people there hold, but a held item's own children were left out of it altogether and an empty one looked
    like any other item, so a book that held nothing never looked like a place a leaf could go (Stonemarch 1637-1658). The first version named
    the things the action mentions; that needed a text match, which counted "and" as a rare word (four unrelated pouches) and could not
    tell "my spell book" from "Jakal's spellbook". This names nothing: every thing the actor holds, the same way. No kind of thing is read.
    Cost: the actor's own things and one lookup each, a few labels apiece."""
    for x, c in ctx.holder.items():
        if c == ctx.actor:
            kids = [kosha.label(k) for k in sorted(kosha.holds(x, t))]
            ctx.holds[x] = kids[:HOLDS_SHOWN] + ([f"{len(kids) - HOLDS_SHOWN} more"] if len(kids) > HOLDS_SHOWN else [])


HOLDS_SHOWN = 8   # the most things one named thing's line lists


def holders_full(kosha, b, records: list[Record], t_after: int, cap: int = FANOUT_CAP) -> dict[str, tuple[str, int, list[str]]]:
    """The fanout rerun's finding: {holder: (its name, how many places it already listed, the new places put directly in it)} for
    each holder this pulse put a new place straight into when it already listed `cap` or more. The world's top is exempt (a
    realm made around a place is declared in it and takes the place in by a move, rule 7). Counted as THE TREE lists a holder's
    places: what is directly in it and is a place. Cost: this pulse's own records and one holder's children.

    A new container is counted by what it holds, not by a bit on it (Marc, 2026-10-13: scenes do not exist, it is all containers): one
    this pulse's records put anything into -- a person who walked in, a thing set down, another new place -- is a container. The scene bit
    still counts too, only because a new empty place cannot be told from a thing any other way until something is in it; it goes when the
    bit does. Stonemarch 1553: a leatherworker's cottage, a holder and not a scene, was put straight into Millhaven (27 places) and nothing
    looked at it.

    THE GROUP (Marc, 2026-10-13: "Group exemption. That needs fixing"): a new container is the group the rerun asks for, and not a flat
    addition, only if it GATHERS two or more places: new containers put in it this pulse, and existing places of the holder moved into it.
    Then it stands for those places in the holder's width, one where there would have been two. It used to be exempt if it held ANY entity
    made this pulse, a person included, so a wrapper made for one place (Stonemarch 1553's cottage round its porch, 1543's Cellars round the
    vaults) passed as a group while changing nothing: the holder gained one child, as it would have without it."""
    placed: dict[str, list[str]] = {}
    inside: dict[str, list[str]] = {}      # what this pulse put into each container
    for r in records:
        if r.kind == "kosha" and r.value and r.value != r.entity_id:
            inside.setdefault(r.value, []).append(r.entity_id)
        if r.kind == "kosha" and r.note == "declared" and r.value and r.entity_id in b.minted:
            placed.setdefault(r.value, []).append(r.entity_id)

    def is_container(x: str) -> bool:
        return x in inside or kosha.is_scene(x)

    def gathers(w: str) -> int:
        """The places a new container w stands for: the new containers put in it, and the holder's existing places moved into it."""
        return sum(1 for x in inside.get(w, []) if (is_container(x) if x in b.minted else kosha.is_place(x, t_after)))

    out: dict[str, tuple[str, int, list[str]]] = {}
    for holder, new in placed.items():
        if holder == b.earth or holder in b.minted or not kosha.is_place(holder, t_after):
            continue
        flat = [e for e in new if is_container(e) and gathers(e) < 2]
        if not flat:
            continue
        n = sum(1 for x in kosha.holds(holder, t_after) if x not in b.minted and kosha.is_place(x, t_after))
        if n >= cap:
            out[holder] = (kosha.label(holder), n, [kosha.label(e) for e in flat])
    return out


def places_beyond(kosha, b, ctx: Context, t: int, t_after: int) -> dict[str, tuple[str, list[tuple[str, str, str | None]]]]:
    """The place rerun (the sample world, 2026-09-30): arriving on another
    island, the Narrator knew the island and none of its places -- the tree
    shows the actor's own branch -- and minted a landing beside the one
    there, three arrivals of three. A scene this pulse made in a region off
    the actor's branch that already holds scenes: that region, with its
    scenes two deep -- past anything the pulse made around the new scene
    (live, the same day: a new "Ansgar House" around a new counting parlour,
    on the island where the parlour already stood). Replayed on the kept
    prompts: 8 of 9 found the place
    that was there; places that were not there were still made. Cost: the
    region's places, and a second call on about one player pulse in a
    hundred."""
    shown = set(ctx.chain) | set(ctx.above) | ({ctx.place} if ctx.place else set())
    name = lambda x: kosha._first_name.get(x) or (kosha.canon.entities[x].labels or [x])[0]
    out: dict[str, tuple[str, list[tuple[str, str, str | None]]]] = {}
    for eid in sorted(b.minted):
        if not kosha.is_scene(eid):
            continue
        region = kosha.container_at(eid, t_after)
        while region is not None and region in b.minted:  # a new house around the new room: the region that was there
            region = kosha.container_at(region, t_after)
        if region is None or region in shown or region in out:
            continue
        rows: list[tuple[str, str, str | None]] = []
        for x in kosha.holds(region, t):
            if x in b.minted or not kosha.is_scene(x):
                continue
            rows.append((x, name(x), None))
            rows += [(y, name(y), x) for y in kosha.holds(x, t) if y not in b.minted and kosha.is_scene(y)]
        if rows:
            out[region] = (name(region), rows[:PLACES_BEYOND])
    return out


SANKALPA_CAP = 8  # v5 §28: at most this many who may act besides the players, the most lately active first


def who_may_act(log: Smrti, ctx: Context, actor: str, world_time: int, action: str | None, players: frozenset[str],
                out_of_reach: frozenset[str] = frozenset(), pacing: dict | None = None) -> list[str]:
    """v5 §28 (Marc, 2026-10-01): who, besides the players, may form an intention
    this moment -- each one present, or across an open boundary from the scene
    (sandhi), who has a tuple of their own in the last participation window
    (the world's `pacing` "participation_window", else PARTICIPATION_WINDOW:
    the same window that decides who stays quiet in the background, Marc
    2026-10-10) ticks or is named in the action; never the actor, a player, someone out of
    reach (lived later), or anything in the hand of the actor, a player or
    someone the action names (mauna: a carried thing is quiet). By structure only: what they are is never asked. Replayed on
    the moathouse fights: surprised guards across a doorway took no part in
    the first moment and fought from the next. Cost: those present and across,
    each one's latest tuple."""
    kosha = log.kosha
    window = (pacing or {}).get("participation_window")
    window = PARTICIPATION_WINDOW if window is None else window
    t = world_time + 1
    here = set(ctx.present)
    if ctx.place:
        for _x, side in kosha.across(ctx.place, t):
            here |= set(kosha.occupants(side, t))
    named = set(kosha.names_in(action, here)) if action else set()
    holders = {actor} | set(players) | named  # mauna's held: in the hand of the actor, a player, or someone the action names
    lately: dict[str, int] = {}
    for p in sorted(here - set(players) - {actor} - set(out_of_reach)):
        if kosha.container_at(p, t) in holders:
            continue  # a carried thing is quiet -- a man riding in a cart is not carried
        own = log.stream(p, before=world_time)
        last = own[-1].chronology if own else None
        if (last is not None and world_time - last <= window) or p in named:
            lately[p] = last if last is not None else world_time
    return sorted(lately, key=lambda p: (-lately[p], p))[:SANKALPA_CAP]


def named_in(log, canon: Maya, eid: str) -> str:
    """Where and in whose words a mention was named: the first tuple, at the tick it was minted, of someone whose own
    tuples are about it (minting is meeting). Asked only by the mention rerun, which is rare: one pass over who is
    acquainted, then over such streams until one has it -- never a pass over the log."""
    e = canon.entities.get(eid)
    if e is None or log is None:
        return ""
    for who in sorted(p for p, about in log.acquainted.items() if eid in about):
        for tu in log.by_perceiver.get(who, ()):
            if tu.chronology == e.created_at_world_time:
                said = " ".join(tu.memory.split())
                said = said if len(said) <= 300 else said[:297] + "..."
                def called(x: str) -> str:  # by words, never an id the prompt has no handle for
                    return log.labels.get(x) or next(iter(getattr(canon.entities.get(x), "labels", None) or ["someone"]))
                at = f" at {called(tu.place)}" if tu.place else ""
                return f'named at tick {tu.chronology} by {called(who)}{at}: "{said}"'
    return f"named at tick {e.created_at_world_time}"


def view_block(kosha, ctx: Context, t: int, library: dict[str, str], movable: frozenset[str], shown: Iterable[str],
               players: frozenset[str] = frozenset(), arriving: dict[str, list[str]] | None = None,
               returning: int | None = None, name_taken: dict[str, tuple[str, str]] | None = None,
               mention_named: dict[str, tuple[str, str]] | None = None, mention_words: dict[str, str] | None = None,
               out_of_time: frozenset[str] = frozenset(), lived_at: dict[str, str] | None = None) -> str:
    """v5 §7-§8: what the Narrator is told of the views that hold here -- the
    scene's chain, outermost first (nearest wins), each present being's
    own clauses, what the structure allows beyond the default, which
    players may be moved, the world's library of becomings. Text the engine
    assembles by structure and never reads."""
    lines: list[str] = []
    scene = ctx.place
    scene_view = kosha.view_of(scene, t) if scene else []
    # v5 §15: the earth's view is the world's own, shown in the system prompt as it always was -- not again here
    def sown(bearer: str, key: str) -> bool:
        return key == DARSANA and kosha.container_at(bearer, t) is None
    shown_view = [(bearer, k, text) for bearer, k, text in scene_view if not sown(bearer, k)]
    if shown_view:
        lines.append("VIEWS THAT HOLD HERE (the scene and what it is in, outermost first; the nearer wins where they differ):")
        lines += [f"- [{bearer}] {text}" for bearer, _k, text in shown_view]
    above = {bearer for bearer, _k, _t in scene_view} | ({scene} if scene else set())
    own = []
    for who in [ctx.actor] + sorted(ctx.present - {ctx.actor}):
        mine = [(bearer, text) for bearer, k, text in kosha.view_of(who, t) if bearer not in above and not sown(bearer, k)]
        own += [f"- [{who}] (held by {bearer}) {text}" if bearer != who else f"- [{who}] {text}" for bearer, text in mine]
    if own:
        lines.append("WHAT HOLDS FOR EACH ONE PRESENT (their own, nearest of all):")
        lines += own
    shown = set(shown)
    perceiving = sorted(x for x in shown if kosha.is_scene(x) and kosha.capable(x, "perceives", t))
    if kosha.defaults.get("perceives"):
        lines.append("PERCEIVING: in this world, scenes have a perspective of their own.")
    elif perceiving:
        lines.append("PERCEIVING (scenes with a perspective of their own): " + ", ".join(perceiving))
    reaching = sorted(x for x in shown if not kosha.is_scene(x) and kosha.capable(x, "reaches", t))
    if kosha.defaults.get("reaches"):
        lines.append("REACHING: in this world, perception is not bounded by the scene and its neighbours.")
    elif reaching:
        lines.append("REACHING (may perceive beyond the scene and its neighbours): " + ", ".join(reaching))
    # Marc, 2026-10-13 (Stonemarch 1533-1573): a player whose time has run past this moment is apart (kala.apart) and every move of them is
    # refused, `movable` or not; the line said they could be moved, and the Narrator moved Dunstan onto the church steps five times. They are
    # not listed as movable, and are told where they were at this moment -- the place only, never a deed of theirs (rule 10).
    movable_now = sorted(set(movable) - set(out_of_time))
    if movable_now:
        lines.append("MOVABLE (players another's turn may move): " + ", ".join(movable_now))
    if lived_at:
        lines.append("LIVED PAST THIS MOMENT (players whose own time has already run further on than this one: out of reach, never seen, "
                     "heard, moved or written for here; where each was at this moment, and nothing of what they did): "
                     + "; ".join(f'{eid} "{name}", at {where}' for eid, (name, where) in sorted(lived_at.items())))
    here = sorted(p for p in ctx.present if p in players and p not in movable and p != ctx.actor)
    if here:
        lines.append("PLAYERS HERE (rule 10): " + ", ".join(here))
    away = sorted(kosha.absent_in(ctx.place, t) - {ctx.actor}) if ctx.place else []
    if away:
        # memory still holds them here; without this the Narrator writes them in
        lines.append("NOT IN THE WORLD NOW (players away from it: neither here nor gone -- never seen, heard, moved or "
                     "written for, and nothing is said of their going): " + ", ".join(away))
    if returning is not None:
        lines.append(f"RETURNING: the actor has been out of the world since tick {returning} and is in it again now, where they "
                     "were. Narrate their being here again first -- as those present would notice someone come back in, "
                     "never as having been here all along -- then the action.")
    if name_taken:
        lines.append("NAMES ALREADY BORNE (this turn gave someone a name another person in the world already bears): "
                     + "; ".join(f'{part} -- {eid} "{name}"' for part, (eid, name) in sorted(name_taken.items()))
                     + ". If the one meant is that person, they are that id; if it is someone else, give them a name of their "
                     "own, a whole one (two words at least -- a given name and a family name, or a byname -- even when only one "
                     "of them is spoken) -- unless they are kin by that name and the memory says so.")
    for eid, name in sorted({v for v in (mention_named or {}).values()}):
        # said as a fact of the world, not of the first telling the Narrator never saw: worded so, it bound 8/8 where
        # meant (the second model, the mid-size model); "this turn made something new under its name" was ignored in the dragon's lair, 0/8
        where = (mention_words or {}).get(eid) or ""
        lines.append(f'NAMED IN PASSING, NOT YET FOUND (something the world already holds by this name, spoken of but never '
                     f'reached or placed): {eid} "{name}"' + (f" -- {where}" if where else "") + f'. If this telling comes to the '
                     f'same thing, it is {eid}: use that id (moving it into where it is found), never a NEW one. If the telling '
                     f'comes to something else that happens to share the name, make it NEW with a name of its own.')
    for scene_id, there in (arriving or {}).items():
        lines.append(f"PLAYERS IN {scene_id}, WHERE THIS ACTION LEADS (if the actor goes into {scene_id}, these are there when "
                     "they arrive -- rule 10): " + ", ".join(there))
    if library:
        lines.append("BECOMINGS THIS WORLD KNOWS:")
        lines += [f"- {k}: {v}" for k, v in list(library.items())[:12]]
    return "\n".join(lines)



# Mentions (collision run, question 4; Marc: "if shadows become entities,
# that's where the real troubles start"). A pulse may mint what nobody in it
# reaches -- a room the aunt speaks of, a man said to sit upstairs -- and it
# was placed there, and things were put into it, on hearsay: the third-floor
# reading room (Ravenmoor 514) was a place nobody had been when the grey-faced
# young man was put in it at 524 and again at 530, and each later visit minted
# him again (four Elias fragments). Measured: 25% of Ravenmoor's mints, 16% of
# the Crossing's; 21 things and people put into them before anyone came. Now
# a mint the pulse does not reach is a MENTION: a name, placed nowhere, and
# nothing is put into it; the pulse that reaches it places it, shown it as
# KNOWN (placed nowhere). Reached: in a scene where an entity that existed
# before this pulse, and has a tuple in it, stood before or after, on the
# route, or across a sandhi from one of them -- or holding such a scene (a
# region above), or directly in what holds one (its neighbours; v5 §10).
# Structure only.
SANDHI_ALL = os.environ.get("SMRITILILA_SANDHI", "") == "1"  # sandhi for every world this server runs (else: world data "sandhi")


def _mentions(out: PulseOut, b: Binder, actor_id: str, records: list[Record], moved: set[str], prior: dict[str, str | None],
              scene_rows: list[tuple[str, dict]], t_after: int, route: list[str] = ()) -> tuple[list[Record], set[str]]:
    """Its name: nāmamātra (नाममात्र), name only (v5 §10)."""
    kosha = b.kosha
    t = b.world_time

    def scene(x: str, when: int) -> str | None:
        return x if kosha.is_scene(x) else kosha.scene_of(x, when)

    witnesses = {actor_id} | {sid for sid, _ in scene_rows}
    for p in out.perceiver_tuples:
        x = b.resolve(str(p.get("perceiver", "")))
        if x is not None:
            witnesses.add(x)
    anchor = {s for w in witnesses - set(b.minted) for s in (scene(w, t), scene(w, t_after)) if s}
    # the pulse's reach: where its witnesses stood, its route, and what is across a sandhi from any of them
    anchor |= set(route)
    anchor |= {s for a in list(anchor) for _x, s in kosha.across(a, t_after)}

    def unplaced(x: str) -> bool:
        """Minted now, or a mention from before: minted in play and placed
        nowhere (a seed's root region is placed nowhere too, and is none)."""
        if x in b.minted:
            return True
        return prior.get(x, kosha.container_at(x, t)) in (None, b.earth) and unplaced_mention(kosha, b.canon, x, t, b.earth)

    # the neighbourhood (v4 adjacency is the tree): what directly holds a witness's scene holds its neighbours
    holders = {kosha.container_at(a, t_after) for a in anchor} - {None}
    touched = set(b.minted) | {x for r in records if r.kind == "kosha" for x in (r.entity_id, r.value) if x}
    mentions = set()
    for eid in touched:
        if not unplaced(eid):
            continue
        s = scene(eid, t_after)
        if s in anchor or kosha.container_at(eid, t_after) in anchor | holders or any(kosha.inside(a, eid, t_after) for a in anchor):
            continue  # reached: placed now (a mention from before is placed on arrival)
        mentions.add(eid)
    if not mentions:
        return records, moved
    kept, held = [], []
    for r in records:
        if r.kind == "kosha" and (r.entity_id in mentions or r.value in mentions):
            # the fork follows what is kept: anything put into a mention back where it was
            if r.entity_id not in mentions:
                kosha.add(Record(new_id(), "kosha", r.entity_id, prior.get(r.entity_id), t, b.pulse_id, note="mention"))
                held.append(r.entity_id)
            continue
        kept.append(r)
    for m in sorted(mentions):
        # a mention is in the earth, like anything unplaced (Marc, 2026-10-04): in the tree, so a later telling that
        # names it again finds it; still in no scene, the earth being no one's. Recorded once: one already there
        # (re-mentioned, unreached) gets the fork's record only
        if b.earth is not None and prior.get(m, kosha.container_at(m, t)) == b.earth and m not in b.minted:
            kosha.add(Record(new_id(), "kosha", m, b.earth, t, b.pulse_id, note="mention"))
        else:
            kept.append(_record(b, m, None, "mention"))
    still = {r.entity_id for r in kept if r.kind == "kosha"}
    log_event("smrti.mention", pulse_id=b.pulse_id, ids=sorted(mentions), labels=[b.label_of(x) for x in sorted(mentions)],
              held_back=sorted(set(held)), anchor=sorted(anchor))
    return kept, {x for x in moved if x in still or x not in set(held)}


def _place(out: PulseOut, b: Binder, actor_id: str, players: frozenset[str], as_origin: bool,
           movable: frozenset[str] = frozenset(), library: dict[str, str] | None = None, mentions: bool = False,
           frozen: frozenset[str] = frozenset(), away: frozenset[str] = frozenset()) -> Placement:
    """The Narrator's moves, new entities and scene rosters turned into
    containment records, checked for structure only (v5 §10)."""
    kosha = b.kosha
    t_after = b.world_time + 1
    origin = b.ctx.place
    library = library or {}
    records: list[Record] = []
    moved: set[str] = set()
    refused: list[dict] = []
    prior: dict[str, str | None] = {}  # what each moved thing was in before this pulse

    def move(entity: str | None, into: str | None, why: str, said_into: str = "") -> bool:
        if entity is None:
            return False
        if entity not in prior:
            prior[entity] = kosha.container_at(entity, t_after)
        if ((into is not None or not said_into) and kosha.container_at(entity, t_after) == into
                and (entity not in b.minted or entity in moved)):
            return True  # already there (a mint's own first record included): nothing to record, nothing to refuse
        reason = None
        if into is None and said_into:
            reason = "unknown_container"
        elif entity == into:
            reason = "self"
        elif entity == b.earth:
            # the earth is never moved (Marc, 2026-10-03: "make the earth unmovable"): it holds all that can come from
            # it and nothing holds it, so the root and the earth never part -- the tree grows upward beneath it
            reason = "earth"
        elif into is not None and kosha.inside(into, entity, t_after):
            reason = "cycle"
        elif entity not in b.shown and entity not in b.minted and entity not in b.ctx.present:
            reason = "not_shown"
        elif entity in frozen:
            reason = "lived_later"  # v5 §21 v4.27: someone or something already lived further along here
        elif entity in away and entity != actor_id:
            reason = "apart_in_time"  # kala.apart: this moment is not their present
        elif entity in players and entity != actor_id and entity not in movable:
            # v5 §37: a player whose body is theirs moves only in their own
            # pulse; one who allows the world to act for them is movable
            reason = "player_not_actor"
        if reason is not None:
            refused.append({"entity": entity, "into": into or said_into or None, "reason": reason, "why": why})
            return False
        records.append(_record(b, entity, into, why))
        moved.add(entity)
        log_event("smrti.move", pulse_id=b.pulse_id, entity=entity, label=b.label_of(entity), into=into,
                  into_label=b.label_of(into) if into else None, why=why)
        return True

    NOWHERE = {"nothing", "none", "null", "nowhere", "empty", "same", "n/a"}   # words that say "in nothing", not a container

    def not_words(words: str) -> str | None:
        """Why these cannot be a name, or None (Tier 0, 2026-10-15: a sentence, 400 letters, "!!!" and a truncated handle each became an entity).
        Structure only, as `_MNEMONIC` is: a letter in it, at most 60 characters and 8 words, no sentence punctuation or line break."""
        if not any(c.isalpha() for c in words):
            return "no_letter"
        if len(words) > 60 or len(words.split()) > 8:
            return "too_long"
        if re.search(r"[,;:!?()\[\]{}<>\"|\\=\n]", words):
            return "sentence"
        return None

    def mint_from_words(said: str, role: str) -> str | None:
        """Marc, 2026-10-15: "If something is attempted to be moved into an unknown container, a container MUST be minted first", and the same for an
        unknown thing: a move names what it moves and what it moves it into, and whichever has no id yet is made by the move. A thing put into
        something that is not an entity cannot be put anywhere, and used to be refused (`unknown_container`). NO HOLDER IS GUESSED (Marc: the Narrator
        declares it, in new_entities, in the same answer; rule 3): one the Narrator left undeclared is a mention, in the earth, and anything put into it is
        put back where it was (`_mentions`, as for any unreached mint), so the move does not stand until the Narrator says what holds the container.
        `smrti.move_undeclared` counts them.
        Order (Marc, 2026-10-15: "do A"): (1) words that say "in nothing" keep their old reading; (2) the Narrator's own row, exactly: the same words with a
        copied hash tail or a hyphen for an underscore are its declaration, and the move is onto that row (training log 1866); (3) only WORDS are made
        into an entity: not a sentence, not 60 characters or 8 words, not without a letter, no sentence punctuation, and not an id-shaped handle with no row (a hyphened or underscored
        token is a copy of an id, so a truncated one, `marren-innkeeper`, is a slip and not a second Marren, log 3033) -- unless it ends in the hash tail the
        Narrator imitates when it invents an id for a new thing (log 4093). What is refused keeps its old reading, is dropped or `unknown_container`, and is
        counted `smrti.move_not_made`; (4) said twice in a pulse, it is one."""
        words = said.strip().strip(",;.'\"`").strip()
        if not words or words.lower() in NOWHERE:
            return None
        key = new_handle(words)
        if key in b.resolved:
            return b.resolved[key]
        name = _humanize(words) if "_" in words else words
        slug = _slug(name, "_")
        for h in b.declared:
            if h.startswith("NEW:") and _slug(h[4:], "_") in (slug, _slug(words, "_")):
                cid = b.resolve(h)
                if cid is not None:
                    b.resolved[key] = cid
                return cid
        why = not_words(name)
        if why is None and not re.search(r"\s", words) and re.search(r"[-_\d]", words) and not re.search(r"_[0-9a-f]{6}$", words):
            why = "handle_like"
        if why is not None:
            log_event("smrti.move_not_made", pulse_id=b.pulse_id, said=words[:80], role=role, why=why)
            return None
        handle = "NEW:" + (slug or "x")
        b.declared.setdefault(handle, {"id": handle, "name": name, "scene": False})
        cid = b.resolve(handle)
        if cid is None:
            return None
        b.resolved[key] = cid
        if cid in b.minted:
            log_event("smrti.move_undeclared", pulse_id=b.pulse_id, said=words, id=cid, label=b.label_of(cid), role=role)
        return cid

    # 1. new entities: minted, named, and put where the memory says they are
    for d in out.new_entities:
        h = new_handle(str(d.get("id", "")).strip())
        if h.startswith("NEW:"):
            b.declared[h] = d
    for d in out.new_entities:
        eid = b.resolve(str(d.get("id", "")))
        if eid is None:
            continue
        said_in = str(d.get("in", "") or "").strip()
        into = b.resolve(said_in, scene=False) if said_in else None
        if into is None and said_in:
            into = mint_from_words(said_in, "container")
        if eid in b.minted and (said_in or into is not None):
            move(eid, into, "declared", said_in)
        elif into is not None and unplaced_mention(kosha, b.canon, eid, b.world_time, b.earth):
            # a mention declared again by its own id, as the place it now is (the mention replay: the Narrator
            # chose it, and listed it among the new): its first placement contradicts no record
            move(eid, into, "mention_placed", said_in)
    # 2. the scene rows
    # No scenes (kosha.no_scenes): a container is a shared moment once a perceiver is in it, and the telling that takes someone into a
    # place writes that place's row in the same breath -- before the moves below are placed. Read in that order, the first arrival
    # at every new place was refused (the Austin copy, 2026-10-12: 5 of 5 arrivals, switch on; none with it off). A container a
    # perceiver is moved into by this telling counts as one for its row.
    arriving_in: set[str] = set()
    if no_scenes():
        for m in out.moves:
            ent_ = b.resolve(str(m.get("entity", "")))
            into_ = b.resolve(str(m.get("into", "") or "")) if str(m.get("into", "") or "").strip() else None
            if ent_ is not None and into_ is not None and (ent_ == actor_id or kosha.is_perceiver(ent_)):
                arriving_in.add(into_)
    scene_rows: list[tuple[str, dict]] = []
    for p in out.place_tuples:
        sid = b.resolve(str(p.get("place", "") or p.get("scene", "")), scene=True)
        if sid is None:
            continue
        if not kosha.is_scene(sid) and sid not in arriving_in:
            folded = kosha.scene_of(sid, t_after)
            if folded is None and kosha.container_at(sid, t_after) is None and not kosha.is_perceiver(sid):
                folded = sid  # v5 §11: a floating container is its occupants' scene, and keeps their record
            log_event("smrti.scene_row_not_a_scene", pulse_id=b.pulse_id, said=sid, into=folded)
            if folded is None:
                continue
            sid = folded
        scene_rows.append((sid, p))
    # 3. the moves, as the Narrator reported them
    said_moves = []
    for m in out.moves:
        said_ent = str(m.get("entity", "") or "").strip()
        ent = b.resolve(said_ent)
        if ent is None and said_ent:
            ent = mint_from_words(said_ent, "thing")            # a thing with no id is made by the move
        said_into = str(m.get("into", "") or "").strip()
        into = b.resolve(said_into) if said_into else None
        if into is None and said_into and ent is not None:
            into = mint_from_words(said_into, "container")      # and so is the container, first
        # No guess from who holds what (Marc, 2026-10-13: "Everything is a container. A container can hold things. If the narrative says
        # something merged with or goes into something that should happen."): a thing goes where the narrative puts it -- into its whole,
        # into a book, onto a person -- and not where the engine thinks it belongs. Until 2026-10-13 a part moved into whoever HELD the whole
        # it first came from was sent into the whole (smrti.rejoined, v4.34); that was a patch for the Narrator's slip in one replay (the
        # recovered dart put on Jakal, 2 of 3), it took the first container for the whole, and it did nothing for a part going into some
        # other thing. Rule 7 still asks the Narrator to move a part that comes back INTO its whole.
        said_moves.append((ent, into, said_into,
                           str(m.get("from", "") or "").strip()))
    goes_to = {ent: into for ent, into, _, _ in said_moves if ent is not None and into is not None}

    def carrier_of(ent: str | None, into: str | None, said_from: str) -> str | None:
        """Who carries ent into the same place this pulse -- the move is then
        already true by carrying, not a letting go (which says `from`)."""
        if ent is None or into is None or said_from:
            return None
        return next((c for c, x in goes_to.items() if x == into and c != ent and kosha.inside(ent, c, t_after)), None)

    carried = []
    for ent, into, said_into, said_from in said_moves:
        if carrier_of(ent, into, said_from) is not None:
            carried.append((ent, into, said_into, said_from))
        else:
            move(ent, into, "moved", said_into)
    for ent, into, said_into, said_from in carried:
        carrier = carrier_of(ent, into, said_from)
        if carrier is not None and carrier in moved:  # the carrier's own move stood
            log_event("smrti.carried_stays", pulse_id=b.pulse_id, entity=ent, carrier=carrier, into=into)
            continue
        move(ent, into, "moved", said_into)
    # 3b. boundaries (sandhi, narrator rule 15): a thing that is the boundary between two scenes, perceived
    # or passed across this pulse, is given its other side -- a sandhi. Checked for structure only.
    crossed: list[str] = []
    for bd in out.boundaries:
        thing = b.resolve(str(bd.get("thing", "") or ""))
        side = b.resolve(str(bd.get("also_in", "") or ""), scene=True)
        reason = None
        if thing is None or side is None:
            reason = "unknown"
        elif thing == side or kosha.inside(side, thing, t_after):
            reason = "cycle"
        elif not kosha.is_scene(side) and not (no_scenes() and side in b.canon.entities and not kosha.is_perceiver(side)):
            reason = "not_a_scene"  # (no scenes: a boundary may join any container, one nobody has entered yet included)
        elif thing in players:
            reason = "a_player"  # where a player is, is theirs (anything interacted with perceives: a door knocked on has
            # its own tuples -- live, collide 671 -- so perceiving says nothing of what is a boundary)
        elif thing not in b.shown and thing not in b.minted and thing not in b.ctx.present:
            reason = "not_shown"
        if reason is not None:
            refused.append({"entity": thing, "into": side, "reason": reason, "why": "sandhi"})
            continue
        if side not in kosha.joins(thing, t_after) and side != kosha.scene_of(thing, t_after):
            r = Record(new_id(), "sandhi", thing, side, b.world_time, b.pulse_id, note="open")
            kosha.add(r)
            records.append(r)
            log_event("smrti.sandhi", pulse_id=b.pulse_id, thing=thing, label=b.label_of(thing), side=side,
                      side_label=b.label_of(side), held_by=kosha.container_at(thing, t_after))
        sides = kosha.joins(thing, t_after) or {x for x in (kosha.scene_of(thing, t_after), side) if x}
        crossed += [s for s in sorted(sides) if s not in crossed]
    # 4. the roster corroborates; it never removes, and never overrides a
    # move the Narrator reported (v5 §10). It adds a move of someone already placed only from a
    # neighbouring scene (the actor excepted): from the scene the actor left, it is refused --
    # the crossing's 323 (a watcher on the bank listed present at the far
    # landing) was v3's reason to ASK who came along, and `moves` is that
    # ask (spec correction, 2026-09-23, while building)
    elsewhere: list[str] = []
    for sid, p in scene_rows:
        for h in p.get("present", []) or []:
            x = b.resolve(str(h))
            if x is None or x == sid or kosha.is_scene(x):
                continue
            now_scene = kosha.scene_of(x, t_after)
            if now_scene == sid or sid in kosha.joins(x, t_after):
                continue  # there already -- or there through a sandhi: a door is in both rooms (live, collide 678)
            if x in moved:
                log_event("smrti.roster_disagrees", pulse_id=b.pulse_id, entity=x, listed=sid, moved_to=now_scene)
                continue
            if kosha.container_at(x, t_after) is None or unplaced_mention(kosha, b.canon, x, t_after, b.earth):
                # in nothing, a mention in the earth, or new: a first placement contradicts no record
                move(x, sid, "roster_new" if x in b.minted else "roster_from_nothing")
                continue
            if x == actor_id:
                move(x, sid, "roster_actor")
                continue
            before = kosha.scene_of(x, b.world_time)
            neighbours = set(kosha.neighbours(sid, b.world_time))
            if (x not in players or x in movable) and before is not None and before in neighbours:
                if move(x, sid, "roster"):
                    log_event("smrti.roster_added", pulse_id=b.pulse_id, entity=x, into=sid, was=before)
                continue
            if before == origin and before is not None:
                log_event("smrti.roster_from_origin_refused", pulse_id=b.pulse_id, entity=x, listed=sid, stays=before)
            elsewhere.append(x)
    # 5. anyone new to the world with nowhere said: where the pulse happened
    here = kosha.scene_of(actor_id, t_after) or (scene_rows[0][0] if scene_rows else origin)
    for t in out.perceiver_tuples:
        x = b.resolve(str(t.get("perceiver", "")))
        if x is not None and x in b.minted and kosha.container_at(x, t_after) is None and here:
            move(x, here, "new_here")
    # 6. in nothing is in the earth (Marc, 2026-09-30): whatever this telling minted and left nowhere, last of
    # every rule above -- still in no scene (the earth perceives, and is no one's); a loose root cannot arise
    if b.earth is not None:
        for x in sorted(b.minted):
            if x != b.earth and kosha.container_at(x, t_after) is None:
                records.append(_record(b, x, None, "earth"))
    # v5 §12: a departure is recorded where it happens -- by the scene's own
    # row. A scene someone left with no row this pulse is logged, never
    # written for the Narrator (the fallback is measurement only)
    rowed = {s for s, _ in scene_rows}
    for x in moved:
        left = kosha.scene_of(x, b.world_time)
        if left and left != kosha.scene_of(x, t_after) and left not in rowed:
            log_event("smrti.departure_unrecorded", pulse_id=b.pulse_id, entity=x, left=left)
    roots = {x: kosha.scene_of(x, t_after) for x in moved}
    roots = {x: s for x, s in roots.items() if kosha.is_root_scene(s)}
    if roots:
        # v5 §11: how often the Narrator leaves a container floating, counted
        log_event("smrti.root_as_scene", pulse_id=b.pulse_id, entities=sorted(roots), roots=sorted(set(roots.values())))
    if refused:
        log_event("smrti.move_refused", pulse_id=b.pulse_id, refused=refused)
    # v4.30: a place standing in nothing beside the world's top is a loose root -- with the tree shown and
    # aligned it should never arise (Marc, 2026-09-28: "If that case occurs we're missing something"):
    # logged to dig into, never repaired. A place that took the top into it is the tree growing upward.
    top = kosha.root(origin, b.world_time) if origin else None
    for x in sorted(set(b.minted) | moved):
        if x == top or kosha.container_at(x, t_after) is not None or not kosha.is_place(x, t_after):
            continue
        if not (kosha.is_scene(x) or kosha.holds(x, t_after)):
            continue  # a thing left in nothing is v5 §11's root_as_scene count, not a root of places
        if top is not None and kosha.inside(top, x, t_after):
            log_event("smrti.root_grew", pulse_id=b.pulse_id, top=x, label=b.label_of(x), was=top)
        else:
            log_event("smrti.loose_root", pulse_id=b.pulse_id, entity=x, label=b.label_of(x), top=top,
                      holds=kosha.holds(x, t_after)[:8])
    # the pulse's route: where it began, the scenes joined by the boundaries it crossed or looked
    # across, where it ended (the engine cannot tell a door gone through from a window looked through)
    dest = kosha.scene_of(actor_id, t_after)
    route = [x for x in [origin, *crossed, dest] if x]
    route = [x for i, x in enumerate(route) if x not in route[:i]]
    if crossed:
        log_event("smrti.route", pulse_id=b.pulse_id, route=route)
    if mentions:
        records, moved = _mentions(out, b, actor_id, records, moved, prior, scene_rows, t_after, route)
    records += _become(out, b, actor_id, players, movable, library, away)
    if elsewhere:
        log_event("smrti.elsewhere", pulse_id=b.pulse_id, ids=sorted(set(elsewhere)), scenes=[s for s, _ in scene_rows])
    unshown = sorted({r["entity"] for r in refused if r.get("reason") == "not_shown" and r.get("entity") in b.canon.entities
                      and b.log is not None and b.log.stream(r["entity"], before=b.world_time)})
    return Placement(scene_rows, records, elsewhere, moved, origin, t_after, route, unshown)


_USE_ORDER = {"key": 0, "useful": 1, "none": 2}


CLOSES_MAX = 3  # the most earlier memories one moment is written to close (a short list: one moment rarely settles more than a matter or two)


def closes_records(cited, who: str, now: int, pulse_id: str, log: "Smrti | None") -> list[Record]:
    """v5.33: the Narrator's `closes` for one perceiver's tuple -- ticks of that perceiver's own earlier memories, the heads of the
    matters this moment closes -- as `closes` records. Only ticks the perceiver has a tuple at, before now; the newest CLOSES_MAX;
    anything else is dropped (counted, not blocked), as a mistaken id is."""
    if not cited or log is None or not isinstance(cited, (list, tuple)):
        return []
    ticks = {int(k) for k in cited if isinstance(k, (int, float, str)) and str(k).lstrip("-").isdigit()}
    real = sorted((k for k in ticks if k < now and log._at.get((who, k))), reverse=True)[:CLOSES_MAX]
    if len(real) < len(ticks):
        log_event("smrti.closes_dropped", pulse_id=pulse_id, perceiver=who, cited=sorted(ticks), kept=real)
    return [Record(new_id(), "closes", who, str(k), now, pulse_id) for k in real]


def over_one_moment(perceiver_tuples: list[dict]) -> dict[str, int]:
    """v5 §3, "a moment that matters stands alone": a pulse gives a perceiver one tuple, and at most one more for a moment that stands alone, so no more than two. The Narrator is told so
    (narrator.py, rule 1); the engine does not judge or drop an experience for it, it counts (Marc, 2026-10-10: measured, never blocked). Returns each perceiver, as written, that has more than two tuples
    with text in this one answer. A birth writes several by design and never comes through a pulse."""
    seen: dict[str, int] = {}
    for t in perceiver_tuples:
        who = str(t.get("perceiver", "")).strip()
        if who and t.get("text"):
            seen[who] = seen.get(who, 0) + 1
    return {who: n for who, n in seen.items() if n > 2}


def _append(
    out: PulseOut, b: Binder, pl: Placement, actor_id: str, as_origin: bool, as_chronicle: bool,
    players: frozenset[str], embed: Callable[[str], list[float]] | None,
    round_told: dict[str, str] | None = None,
    hidden: frozenset[str] = frozenset(),
    away: frozenset[str] = frozenset(),
    log: "Smrti | None" = None,
) -> tuple[list[Memory], list[Record]]:
    """One tuple per perceiver (a player's never in another's pulse), one
    impersonal tuple per scene touched; names declared; what a player
    witnessed filed as theirs, carrying the others' USE (v5 §31).
    round_told: saṅgama -- each other player of the round, and the narration
    told to them. Every one of them is written for, as the actor is (one who
    did not act, with what they perceived); none is filed a witnessed copy."""
    round_told = round_told or {}
    pulse_id, id_factory = b.pulse_id, b.id_factory
    kosha = b.kosha
    records: list[Record] = []
    written: list[Memory] = []

    def scene_of(who: str) -> str:
        s = pl.scene_after(b, who)
        return s or NOWHERE

    # names: declared at mint, and every alias (v5 §16)
    for d in out.new_entities:
        name = str(d.get("name") or "").strip()
        eid = b.resolved.get(new_handle(str(d.get("id", "")).strip()))
        if name and eid in b.minted:
            r = Record(new_id(), "name", eid, name, b.world_time, pulse_id)
            records.append(r)
            kosha.add(r)
    for a in out.aliases:
        eid = b.resolve(str(a.get("entity", "")))
        name = str(a.get("name") or "").strip()
        if eid and name and name.casefold() not in kosha.names_of(eid):
            r = Record(new_id(), "name", eid, name, b.world_time, pulse_id, note="alias")
            records.append(r)
            kosha.add(r)
            log_event("smrti.alias", pulse_id=pulse_id, entity=eid, name=name)

    if not as_origin and out.narration and not any(b.resolve(str(t.get("perceiver", ""))) == actor_id for t in out.perceiver_tuples):
        # the actor's own tuple is missing (Vadhu tick 81): the narration
        # stands in, AS NARRATED -- not rewritten (v5 §32: the you-to-I table
        # was English, and rewrote others' quoted speech too), and marked
        # so the prompt shows it for what it is
        out.perceiver_tuples.insert(0, {"perceiver": actor_id, "text": out.narration, "weight": "ordinary", "use": "none"})
        if not as_chronicle:
            records.append(Record(new_id(), "mark", actor_id, "as_narrated", b.world_time, pulse_id))
        log_event("smrti.actor_tuple_from_narration", pulse_id=pulse_id, actor=actor_id)
    for pid, told in round_told.items():
        if told and not any(b.resolve(str(t.get("perceiver", ""))) == pid for t in out.perceiver_tuples):
            # a round player's own tuple missing: their narration stands in, as the actor's does
            out.perceiver_tuples.append({"perceiver": pid, "text": told, "weight": "ordinary", "use": "none"})
            records.append(Record(new_id(), "mark", pid, "as_narrated", b.world_time, pulse_id))
            log_event("smrti.actor_tuple_from_narration", pulse_id=pulse_id, actor=pid, round=True)

    for who, n in over_one_moment(out.perceiver_tuples).items():
        log_event("smrti.moment_over", pulse_id=pulse_id, perceiver=who, tuples=n, actor=actor_id)   # counted, not blocked: the watch for "at most one stand-alone moment"
    allowed_scenes = {pl.origin, scene_of(actor_id)} | {s for s, _ in pl.scene_rows}
    for s in list(allowed_scenes):
        if s and s != NOWHERE:
            allowed_scenes |= set(kosha.neighbours(s, b.world_time))
    uses_by_scene: dict[str, list[str]] = {}
    # everyone voiced this pulse perceives from now (v5 §4): marked before
    # any present set is read, so a newcomer is present to the others
    for t in out.perceiver_tuples:
        who = b.resolved.get(new_handle(str(t.get("perceiver", "")).strip())) or (
            str(t.get("perceiver", "")).strip() if str(t.get("perceiver", "")).strip() in b.canon.entities else None)
        if who is not None and t.get("text") and (who == actor_id or who not in players or who in round_told):
            kosha.mark_perceiver(who)
    for t in out.perceiver_tuples:
        who = b.resolve(str(t.get("perceiver", "")))
        if who is None or not t.get("text"):
            continue
        if who != actor_id and who in players and who not in round_told:
            log_event("smrti.player_not_voiced", pulse_id=pulse_id, player=who, actor=actor_id)
            continue
        if who in hidden:
            log_event("smrti.lived_later_voiced", pulse_id=pulse_id, perceiver=who, actor=actor_id)
            continue
        if who in away and who != actor_id:
            log_event("smrti.apart_voiced", pulse_id=pulse_id, perceiver=who, actor=actor_id)
            continue
        if kosha.is_scene(who) and not kosha.capable(who, "perceives", pl.t_after):
            # a scene keeps the impersonal record (v5 §31); its row is that record --
            # unless it perceives (v5 §8: the house that wakes)
            log_event("smrti.scene_voiced_dropped", pulse_id=pulse_id, scene=who)
            continue
        loc = scene_of(who)
        if loc == NOWHERE and who != actor_id and who not in b.shown and who not in b.minted and not as_chronicle:
            log_event("smrti.elsewhere", pulse_id=pulse_id, ids=[who], scenes=[s for s, _ in pl.scene_rows], reason="in_nothing_not_shown")
            continue
        reaches = who in b.shown and kosha.capable(who, "reaches", pl.t_after)
        if loc not in allowed_scenes and who not in pl.moved and who not in b.minted and not as_chronicle and loc != NOWHERE and not reaches:
            log_event("smrti.elsewhere", pulse_id=pulse_id, ids=[who], scenes=[s for s, _ in pl.scene_rows], reason="not_in_scene_or_neighbour")
            continue
        use = use_of(t.get("use"))
        present = frozenset(kosha.occupants(loc, pl.t_after) if loc != NOWHERE else ())
        written.append(Memory(
            id_factory(), who, present, t["text"], "PRIMARY", salience_of(t.get("weight", t.get("salience"))),
            loc, b.world_time, pulse_id, embed(t["text"]) if embed else None, use,
        ))
        records.extend(closes_records(t.get("closes"), who, b.world_time, pulse_id, log))
        kosha.mark_perceiver(who)
        uses_by_scene.setdefault(loc, []).append(use)
    for sid, p in pl.scene_rows:
        if not p.get("text"):
            continue
        said_by = str(p.get("said_by") or "").strip()
        speaker = (said_by if said_by in b.canon.entities else (nearest_id(said_by, b.shown) or b.resolved.get(new_handle(said_by), ""))) if said_by else ""
        said = p.get("said")
        if speaker and speaker in players and speaker != actor_id and said:
            # v5 §33 C: no one else's turn speaks for a player
            log_event("smrti.player_said_dropped", pulse_id=pulse_id, player=speaker, actor=actor_id, said=str(said)[:80])
            said, speaker = "", ""
        if speaker and speaker in away and speaker != actor_id and said:
            log_event("smrti.apart_said_dropped", pulse_id=pulse_id, speaker=speaker, actor=actor_id, said=str(said)[:80])
            said, speaker = "", ""
        text = strip_seams(p["text"], pulse_id)
        if said and str(said).strip():
            # v5 §31: the line and its speaker are a record, not English in the memory
            r = Record(new_id(), "said", sid, str(said).strip(), b.world_time, pulse_id, note=speaker or None)
            records.append(r)
            kosha.add(r)
        emb = embed(text) if embed else None
        present = frozenset(kosha.occupants(sid, pl.t_after)) - away  # who this moment is the present of
        scene_uses = uses_by_scene.get(sid, [])
        scene_use = min(scene_uses, key=lambda u: _USE_ORDER.get(u, 2)) if scene_uses else "none"
        written.append(Memory(id_factory(), sid, present, text, "AMBIENT", 5, sid, b.world_time, pulse_id, emb, scene_use))
        # What a player WITNESSED is their own tuple too (v5 §31): the scene's
        # record of a pulse they were in and were not voiced in. v5 §31: it
        # carries the most useful USE the others present gave this pulse --
        # a fact is as useful to whoever overheard it -- and an ordinary
        # weight: the others' grief is theirs.
        for x in sorted(present):
            if x in players and x != actor_id and x not in round_told:
                written.append(Memory(id_factory(), x, present, text, "PRIMARY", 5, sid, b.world_time, pulse_id, emb, scene_use))
                log_event("smrti.player_witnessed", pulse_id=pulse_id, player=x, place=sid, use=scene_use)
    return written, records




def names_taken(kosha, name_records: list[Record], minted: set[str], t: int | None = None) -> dict[str, tuple[str, str]]:
    """The names this pulse gave that a DIFFERENT, already existing person
    bears in part: {part: (their id, their name)}. Exact: a name part is
    compared whole (Kosha.personal_name_parts), never interpreted.

    A part borne by what holds the named thing -- its container, and theirs,
    short of the scene -- is skipped: "Walter Pike's matchbox" declared into
    Walter Pike names its holder (collision run, Ravenmoor 662, the one false
    rerun in history), while a new "Mrs. Pickering" beside an existing
    landlady Pickering is still a namesake (667). Compared by name, not read."""
    from smritilila.smrti.kosha import personal_name_parts
    idx = kosha.part_index()
    out: dict[str, tuple[str, str]] = {}
    for r in name_records:
        if r.kind != "name" or not r.value:
            continue
        held_by: set[str] = set()  # the holders' name parts
        if t is not None:
            for c in kosha._up(r.entity_id, kosha.container_at(r.entity_id, t), t):
                if kosha.is_scene(c):
                    break
                held_by |= kosha.name_parts(c)
        for part in sorted(personal_name_parts(r.value)):
            if part in held_by:
                continue
            others = sorted(idx.get(part, set()) - {r.entity_id} - set(minted))
            if others and part not in out:
                y = others[0]
                out[part] = (y, kosha._first_name.get(y) or (kosha.canon.entities[y].labels or [y])[0])
    return out


def locked_prompt(kala, canon: Maya, scene: str, until: int, now: int, why: list[tuple[str, str, int]], playable: frozenset[str]) -> str:
    """The closed scene's question, with why (Marc, 2026-09-27): the place is
    named (the player chose it); a fellow player there is named (they will
    meet anyway); anyone else is "someone" and a change is "something there
    changes" -- the reason, never a spoiler of who or what is inside."""
    def label(x: str) -> str:
        e = canon.entities.get(x)
        return e.labels[0] if e is not None and e.labels else "that place"
    place = label(scene)
    parts = []
    for kind, who, minute in why:
        if kind == "present":
            name = label(who) if who in playable else "someone"
            parts.append(f"{name} is there until {kala.say(minute)}")
        else:
            parts.append(f"something there changes at {kala.say(minute)}")
    seen: set[str] = set()
    parts = [x for x in parts if not (x in seen or seen.add(x))]
    reason = f"{place[0].upper() + place[1:]} is already lived later than your present: {'; '.join(parts)}. " if parts else ""
    return f"{reason}You can go in at {kala.say(until)} ({_duration(until - now)} from now). Go in then?"


def _duration(minutes: int) -> str:
    """45 min; 1 h 25 min; 7 h; and from a day on, as a person says it: 2 days, 3 hours."""
    if minutes >= 1440:
        from smritilila.smrti.kala import span_said
        return span_said(minutes)
    h, m = divmod(max(0, int(minutes)), 60)
    return f"{m} min" if not h else (f"{h} h" if not m else f"{h} h {m} min")


def player_named(kosha, written: list[Memory], actor_id: str, players: frozenset[str], pulse_id: str) -> int:
    """A player spoken of in another's turn (Ravenmoor playtest, Marc
    2026-09-24: "another player was effectively writing Nathan's
    backstory"). The engine cannot read what is said of them (s0), so this
    does not judge or block -- it COUNTS: one line per tuple written this
    pulse, by anyone, that names a player other than the actor by a part
    of their name. The rate says how often the world talks about players
    behind their backs; NAMESAKES' player line is what the Narrator is
    shown about it."""
    from smritilila.smrti.kosha import whole_word
    n = 0
    for pid in sorted(players - {actor_id}):
        parts = kosha.name_parts(pid)
        if not parts:
            continue
        pat = re.compile("(" + whole_word(*sorted(parts)) + ")")
        for m in written:
            hit = pat.search(m.experience)
            if hit and m.entity_id != pid:
                log_event("smrti.player_named", pulse_id=pulse_id, actor=actor_id, player=pid, part=hit.group(1),
                          perceiver=m.entity_id, tier=m.tier)
                n += 1
    return n


def narration_named(kosha, told: dict[str, str], log: Smrti, written: list[Memory], known_ids: Iterable[str],
                    world_time: int, pulse_id: str) -> int:
    """A name in a player's narration that the player has never been given
    (Marc, 2026-09-27; Stonemarch: Dunstan's narration named Turuko, Kobort and
    Zert, whom he knew only as "the Baklunish", "the big one", "the Wild Coast
    man" -- his own tuple kept to the descriptions). An instrument, never a
    rule: it COUNTS, one line per narration. A part of a person's or thing's
    declared name, found whole in the narration, is known to the player if it
    stands in their own tuples (this pulse's included) or it names someone
    the engine lists as known to them. Cost: the narration's capitalised
    words against the name index, and the player's own stream."""
    from smritilila.smrti.kosha import whole_word
    idx = kosha.part_index()
    n = 0
    for pid, text in told.items():
        if not text:
            continue
        cands = {w for w in re.findall(r"[^\W\d_][\w\-']*", text) if w in idx and pid not in idx[w]}
        if not cands:
            continue
        own = " ".join([t.memory for t in log.stream(pid, before=world_time)] + [m.experience for m in written if m.entity_id == pid])
        known = set(known_ids)
        unknown = sorted(w for w in cands if not re.search(whole_word(w), own) and not (idx[w] & known))
        if unknown:
            log_event("smrti.narration_named_unknown", pulse_id=pulse_id, player=pid, parts=unknown,
                      ids=sorted({x for w in unknown for x in idx[w]})[:10])
            n += 1
    return n


# Pratidhvani (प्रतिध्वनि, "echo" -- a sound returning; the name approved
# by Marc, 2026-09-26): the alive/echo health check. An
# instrument, never a rule -- it changes nothing the Narrator is shown or
# writes. A thread is alive while its appearances take something from the
# present, an echo once they take only from its own past (Marc, the same
# day). Ravenmoor's Frye haunt, measured: "ticking" and "counting" ran
# through seven and six minds varying every time, half tied to the room;
# "the count ran on under my thoughts" copied itself (a third of each use
# already said) and took from the present one time in six. Per sentence
# written for a perceiver: an ECHO when ECHO_SHARE of its 4-word runs stand
# in one of their own sentences of their last ECHO_WINDOW tuples; ALIVE when
# the words it adds beyond that sentence are in what the scene itself
# recorded this pulse. ECHO_TIC dead echoes of one phrase in a row of their
# own tuples is where a thread has crossed into a tic. Cost: each written
# tuple against its perceiver's last ECHO_WINDOW tuples -- never the history.
ECHO_WINDOW = 20
ECHO_SHARE = 0.5
ECHO_TIC = 3


def pratidhvani(written: list[Memory], log: Smrti, pulse_id: str) -> list[dict]:
    """The echoes in what this pulse wrote, alive or dead, and any tic."""
    present = {w for m in written if m.tier == "AMBIENT" for w in units(m.experience) if weight(w) >= 4}
    found: list[dict] = []
    for m in written:
        if m.tier not in ("PRIMARY", "SECONDARY"):
            continue
        past = [t for t in log.stream(m.entity_id) if t.seq >= 0][-ECHO_WINDOW:]  # their own tuples, not the name seeded first
        past_sents = [[(s, runs(s)) for s in sentences(t.memory)] for t in past]
        for s in sentences(m.experience):
            mine = runs(s)
            if not mine:
                continue
            share, like = 0.0, ""
            for sents in past_sents:
                for ps, pr in sents:
                    x = len(mine & pr) / len(mine)
                    if x > share:
                        share, like = x, ps
            if share < ECHO_SHARE:
                continue
            added = {w for w in units(s) if weight(w) >= 4} - set(units(like))
            alive = bool(added & present)
            streak = 1
            for sents in reversed(past_sents):  # the same phrase in their tuples just before, unbroken
                if any(pr and len(mine & pr) / len(mine) >= ECHO_SHARE for _, pr in sents):
                    streak += 1
                else:
                    break
            f = {"perceiver": m.entity_id, "share": round(share, 2), "alive": alive, "streak": streak, "sentence": s[:160]}
            found.append(f)
            log_event("smrti.echo", pulse_id=pulse_id, **f)
            if not alive and streak >= ECHO_TIC:
                log_event("smrti.echo_tic", pulse_id=pulse_id, perceiver=m.entity_id, streak=streak, sentence=s[:160])
    return found


# Daiva (Marc's name; narrator.DAIVA_WORDS holds the words and why): each
# pulse's fortune, drawn by the engine -- which judges nothing, only supplies
# chance. Seeded by the pulse, so a held pulse's second pass, or a replay of
# it, draws the same word. DAIVA_SPREAD: the share of each word, ruinous to
# providential -- wide and even-handed (Marc). A world's data may reshape it
# ("daiva": {"spread": [7 weights]}) or switch it off ({"off": true}).
DAIVA_SPREAD = (1, 9, 20, 38, 20, 9, 3)  # Marc, 2026-09-30: centred -- fewer extremes at either end (was 6, 12, 20, 24, 20, 12, 6); 2026-10-17: ruinous 3 -> 1 ("Actually 1%"), the two points to even
# ... and by the model that narrates (Marc, 2026-09-28: "we need at least a different distribution of luck for each
# individual llm ... beneficial effects should be roughly equivalent to negative ones"). Measured blind over 233 of
# the day's player moments (scripts/judge): the Narrator's baseline is success, so the three bad words bite and the
# three good ones barely help -- per draw, ~22% of outcomes worse against ~2% better on the even-handed spread, and
# setbacks invented where nothing was in doubt only on bad draws (ruinous 56%, adverse 24%, grim 20%; the second model's
# ruinous 71%). Leaned toward the good end until the words themselves are rebalanced: about half the harm. A world's
# own spread (world data "daiva": {"spread": [...]}) still wins. Matched by a word in the model's name.
# the second model's own lean ((2, 5, 10, 28, 25, 20, 10)) went 2026-09-30 (Marc): writing four pulses in five, it gave Stonemarch
# 17% bad draws and a run of good ones through Jakal's whole moathouse fight -- and replayed on those prompts with the
# draw forced bad, the second model bit hard (Lareth: "My answer is blood"), so the lean was not needed to keep the bad words sane.
# the first model's ((4, 8, 16, 28, 22, 14, 8)) went the same evening (Marc: "Let's have the first model use the same distribution"):
# every pulse draws the same luck. The table stays for a model that one day needs its own.
DAIVA_SPREAD_BY_MODEL: dict[str, tuple[int, ...]] = {}


def daiva_spread(model: str | None) -> tuple[int, ...]:
    """The spread for the model that will narrate, or the even-handed one."""
    m = (model or "").lower()
    return next((s for key, s in DAIVA_SPREAD_BY_MODEL.items() if key in m), DAIVA_SPREAD)


def daiva(seed: str, world_data: dict | None = None, model: str | None = None) -> str | None:
    """A fortune: one of DAIVA_WORDS, or None when the world has it off.
    Deterministic in the seed -- who acts, and at what tick (fortune_seed)."""
    cfg = (world_data or {}).get("daiva") or {}
    if cfg.get("off"):
        return None
    spread = cfg.get("spread")
    if not (isinstance(spread, (list, tuple)) and len(spread) == len(DAIVA_WORDS) and all(
            isinstance(w, (int, float)) and w >= 0 for w in spread) and sum(spread) > 0):
        spread = daiva_spread(model)
    u = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:13], 16) / 16 ** 13 * sum(spread)
    for word, w in zip(DAIVA_WORDS, spread):
        if u < w:
            return word
        u -= w
    return list(DAIVA_WORDS)[-1]


def fortune_seed(actor: str, world_time: int) -> str:
    """Whose fortune, and when -- never the attempt (Marc, 2026-09-27, Stonemarch
    t205: Jakal declined a closed scene's question three times and each retry
    drew anew -- adverse, providential, adverse -- so declining and trying again
    would re-roll fortune). An attempt taken back and tried again at the same
    moment meets the same fortune; the moment moves on when anything is written."""
    return f"{actor}@{world_time}"


# Energy (Marc, 2026-09-26: "Players inject energy into the world" -- and a
# world's turns far from them run on its own past). Where a pulse stands
# relative to the players, measured so the echo rate can be read against it
# over time (smrti.energy). An instrument, like pratidhvani. SINCE_LOOKBACK
# bounds the walk back through the scene's own stream.
SINCE_LOOKBACK = 300


def player_distance(log: Smrti, place: str | None, now: int, players: Iterable[str]) -> tuple[int | None, str | None]:
    """Containment-tree steps from `place` to the nearest player's scene
    (0 = the same scene; up to where the two chains meet and down again),
    and that player. None when no player stands anywhere the tree joins.
    Cost: each player's chain and the scene's own, to the root."""
    kosha = log.kosha
    if kosha is None or not place:
        return None, None
    mine = [place] + kosha.above(place, now)
    at = {s: i for i, s in enumerate(mine)}
    best: tuple[int | None, str | None] = (None, None)
    for p in sorted(players):
        ps = kosha.scene_of(p, now)
        if not ps:
            continue
        for j, s in enumerate([ps] + kosha.above(ps, now)):
            if s in at:
                d = at[s] + j
                if best[0] is None or d < best[0]:
                    best = (d, p)
                break
    return best


def ticks_since_player(log: Smrti, place: str | None, now: int, players: Iterable[str]) -> int | None:
    """Ticks since a player last wrote a tuple in this scene; None when
    none in its last SINCE_LOOKBACK tuples. Cost: that bound."""
    if not place:
        return None
    players = set(players)
    for t in reversed(log.place_stream(place, before=now)[-SINCE_LOOKBACK:]):
        if t.perceiver in players:
            return now - t.chronology
    return None


def memory_leaks(ctx: Context, written: list[Memory], log: Smrti, places: set[str], pulse_id: str) -> list[dict]:
    """v5 §5 / the contract: a tuple is that perceiver's own experience. The
    engine cannot read meaning (v5 §1), so this does not judge -- it MEASURES:
    a new tuple for B that repeats a long verbatim run of A's tuple shown in
    this pulse's context, where B had no tuple in that scene at that tick
    (not a moment they shared). Scene records and a player's witnessed copy
    of one are shared by design, and are skipped. In a world whose view
    opens minds some of these are the view at work; the rate says how much."""
    scene_rows = {(m.location_id, m.pulse_id, m.experience) for m in written if m.tier == "AMBIENT"}
    found = []
    for w in written:
        if w.tier != "PRIMARY" or (w.location_id, w.pulse_id, w.experience) in scene_rows:
            continue
        mine = {(t.chronology, t.place) for t in log.stream(w.entity_id)}
        for t in ctx.tuples:
            if t.perceiver == w.entity_id or t.perceiver in places or (t.chronology, t.place) in mine:
                continue
            span = shared_run(w.experience, t.memory)
            if span:
                found.append({"perceiver": w.entity_id, "source": t.perceiver, "source_tick": t.chronology, "span": span[:160]})
                break
    for f in found:
        log_event("smrti.memory_leak_suspected", pulse_id=pulse_id, **f)
    return found


def run_smrti_pulse(
    canon: Maya,
    actor_id: str,
    action_text: str | None,
    world_time: int,
    pulse_id: str,
    memories: Iterable[Memory],
    client: LLMClient,
    embed: Callable[[str], list[float]] | None,
    *,
    id_factory: Callable[[], str] = new_id,
    as_origin: bool = False,
    players: frozenset[str] = frozenset(),
    as_chronicle: str | None = None,
    news: str | None = None,
    pull: str | None = None,
    age_from: int | None = None,
    records: Iterable[Record] = (),
    action_source: str | None = None,
    movable: frozenset[str] = frozenset(),
    playable: frozenset[str] = frozenset(),
    view: dict | None = None,
    arriving: dict[str, list[str]] | None = None,
    beyond: dict[str, tuple[str, list[tuple[str, str, str | None]]]] | None = None,
    absent: frozenset[str] = frozenset(),
    returning: int | None = None,
    name_taken: dict[str, tuple[str, str]] | None = None,
    mention_named: dict[str, tuple[str, str]] | None = None,
    max_elapsed: int | None = None,
    confirmed: bool = False,
    wait_until: int | None = None,
    lived: list[tuple[int, str]] | None = None,
    window: list[tuple[int, str, str]] | None = None,
    shown: list[str] | None = None,
    samskara=None,
    unshown: list[str] | None = None,
    hidden: frozenset[str] | None = None,
    sangama: dict[str, str] | None = None,
    sangama_sources: dict[str, str] | None = None,
    meet_at: int | None = None,
    setting: list[tuple[str, str]] | None = None,
    not_here: dict[str, str] | None = None,
    no_scene: dict[str, str | None] | None = None,
    not_to_intend: frozenset[str] | None = None,
    holder_full: dict[str, tuple[str, int, list[str]]] | None = None,
) -> SmrtiPulseResult:
    """retrieve -> narrate -> append, in named steps: the context (v5 §26),
    the Narrator's answer cleaned, everything placed (v5 §10), the tuples and
    records written.
    Its name: spanda (स्पन्द), the pulse or throb of consciousness (v5 §2, §31).
    Its energy log (smrti.energy): ojas (ओजस्), vital force (v5 §44).

    as_origin: the actor is a quiet SCENE asked whether anyone arrived
    (v5 §38; v3's road turn). players: the playable ids the Narrator never
    voices in another's turn (WorldSession.unvoiced; `playable` is all of
    them) -- a co-present player is perceived, never voiced (v5 §37). as_chronicle: the actor is
    the world at large and this is the bhuvana it chronicles against.
    records: v4's containment, name and action records so far.
    action_source: "typed" or "suggested" (v5 §35), for a player's pulse.
    view: the world's own view (v5 §7-§8) -- {"darsana": text or None,
    "becomings": {key: text}, "world_data": {"capabilities": {...}}}.
    arriving: {scene: [players]} -- set only by this function's own second
    pass (the arrival rerun): the players in the scene the first pass took the
    actor into, shown this time.
    absent: players out of the world (logged off, frozen) -- in no scene's
    present, never written for. returning: the tick the actor left the
    world at, on their first pulse back; their return is narrated.
    name_taken: {part: (id, name)} -- set only by this function's own
    second pass (the name rerun): the names the first pass gave that another
    person already bears.
    mention_named: {NEW:x: (id, name)} -- set only by the mention rerun: a
    thing the first pass made new under the name of one named in passing
    and never reached, shown so the Narrator may say it is that one.
    v5 §21-§23, a world that keeps a clock: max_elapsed caps an autonomous turn
    (NPCs follow and never lead); confirmed -- the player already agreed to
    a long jump; wait_until -- the player agreed to wait for a locked scene
    to open; lived -- the mirror's second pass, the span others have
    already lived where this pulse went.
    sangama: saṅgama (Marc, 2026-09-27) -- the other players of a round in the
    actor's scene, {player: their action, or "" for one who did not act}.
    One narration tells the round; each of them is told it too
    (round_narrations). One who acts is an actor of this pulse -- voiced,
    moved, speaking -- and draws a fortune of their own; sangama_sources:
    "typed" or "suggested", per player, as action_source.
    unshown: set only by this function's own second pass (the unshown
    rerun) -- people it moved without having been shown them; now shown,
    where they are.
    hidden: set only by this function's own second pass (the lived-later
    rerun) -- who and what, in the place a player went into, is already lived
    further along than the player's present: out of their reach (v5 §21 v4.27).
    holder_full: set only by this function's own second pass (the fanout rerun)
    -- {holder: (its name, the places it lists, the new places the first telling
    put directly in it)}: told the holder is full, to make one new place for the
    group inside it (holders_full)."""
    call = dict(locals())  # the call itself, for a second narration (_again): every argument, never a copied list of them
    sangama = dict(sangama or {})
    acting = {p: a for p, a in sangama.items() if a and a.strip()}
    still = [p for p in sangama if p not in acting]
    players = frozenset(players) - frozenset(acting)  # saṅgama: the round's actors act in it
    bind_pulse_id(pulse_id)
    start = time.monotonic()
    memories = list(memories)
    records = list(records)  # read again if the pulse is narrated a second time
    view = view or {}
    library = dict(view.get("becomings") or {})
    if samskara is not None:
        # the world's kept reading (smrti/samskara.py): its streams, read-only, and this pulse's own
        # fork of the containment index -- not the whole history read again
        log, places = samskara.log_for(canon, absent - {actor_id}), set(samskara.log.places)
    else:
        log, places = build_log(canon, memories, records, defaults=(view.get("world_data") or {}).get("capabilities"),
                                absent=absent - {actor_id})
    # sandhi, per world (world data "sandhi": true; SMRITILILA_SANDHI=1 turns it on for a whole server): the Narrator
    # reports boundaries, mentions are placed nowhere, a person placed nowhere is known
    if no_scenes():
        places = set(places) | set(log.kosha.shared_scenes())  # derived scenes: a place first stood in after the log was seeded is one too
    sandhi = bool((view.get("world_data") or {}).get("sandhi")) or SANDHI_ALL
    age_kala = None
    if as_chronicle is not None:
        ctx = retrieve(log, actor_id, world_time, None, k_place=0, k_sim=0, own_only=True)
        if ctx is None:
            raise ValueError(f"{actor_id} has no stream to chronicle from")
        action = CHRONICLE_ACTION + as_chronicle
        age_kala = Kala.of((view.get("world_data") or {}).get("clock"), ()) if age_from is not None else None
        if age_kala is not None:
            action = age_line(age_kala, age_from) + "\n" + action
    else:
        action = action_text or (SCENE_ARRIVAL_ACTION if as_origin else AUTONOMOUS_ACTION)
        if as_origin and news:
            action += f"\nNEWS OF THE WIDE WORLD, as it would have reached this scene by now (the age's latest chronicle): {news}"
        if as_origin and pull:
            action += ("\nTHE AGE'S PULL -- what is coming in the wide world, NOT YET HAPPENED here: it may colour rumour, worry and "
                       f"the talk of travellers, and draw threads toward this place; never tell any of it as having happened: {pull}")
        ctx = _context_for(canon, log, actor_id, action_text, world_time, embed,
                           keep=frozenset(players or ()) | frozenset(playable or ()),
                           pacing=(view.get("world_data") or {}).get("pacing"), sandhi=sandhi)
        if sangama:
            # saṅgama: what each other player of the round brings -- their own retrieval, merged in
            have = {(t.perceiver, t.seq) for t in ctx.tuples}
            for pid in sangama:
                try:
                    theirs = _context_for(canon, log, pid, sangama[pid] or None, world_time, embed,
                                          keep=frozenset(players or ()) | frozenset(playable or ()),
                                          pacing=(view.get("world_data") or {}).get("pacing"), sandhi=sandhi)
                except ValueError:
                    continue
                for t in theirs.tuples:
                    if (t.perceiver, t.seq) not in have:
                        ctx.tuples.append(t)
                        ctx.sources[(t.perceiver, t.seq)] = "sangama"
                        have.add((t.perceiver, t.seq))
            ctx.tuples.sort(key=lambda t: (t.chronology, t.seq))
    log_event("smrti.context_sections", pulse_id=pulse_id, world_time=world_time, actor=actor_id, **context_sections(ctx))
    log_event("smrti.scene", pulse_id=pulse_id, world_time=world_time, actor=actor_id, scene=ctx.place, chain=ctx.chain,
              present=sorted(ctx.present), neighbours=ctx.neighbours, above=ctx.above, held=[h[0] for h in ctx.held],
              elsewhere=[e[0] for e in ctx.elsewhere], namesakes=[n[0] for n in ctx.namesakes],
              players_named=[p[0] for p in ctx.players_named])
    for t in ctx.tuples:
        if (t.perceiver, t.chronology) in watched_tuples():
            log_event("smrti.watched", pulse_id=pulse_id, world_time=world_time, actor=actor_id, perceiver=t.perceiver, tick=t.chronology,
                      source=ctx.sources.get((t.perceiver, t.seq), "unsourced"))
    # v5 §20 kāla: a world that keeps a clock shows everything as of the actor's time
    _kala_t = time.perf_counter()
    clock_cfg = (view.get("world_data") or {}).get("clock")
    if as_chronicle is not None:
        kala = None
    elif samskara is not None and samskara.players == frozenset(playable) and samskara.clock == clock_cfg:
        kala = samskara.kala  # kept, read on at each commit
    else:
        kala = Kala.of(clock_cfg, records, memories, playable, chronicle_ids(canon))
    kala_ms = round((time.perf_counter() - _kala_t) * 1000, 1)  # monitored: the full rebuild from all memories (v5 §19 cost) happens only when the kept one cannot be read (else ~0)
    time_now: int | None = None
    kala_lines: list[str] = []
    rate = 1.0  # v5 §19: world minutes per minute lived where this pulse happens -- composed up the scene's chain
    if kala is not None:
        rate = log.kosha.rate(ctx.place, world_time + 1) if ctx.place else 1.0
        time_now = kala.now(actor_id, wait_until)
        own = kala.clock(actor_id)
        # nothing later than now; nobody who arrives after now; memory by the clock -- except the actor's own last
        # moment and whoever is in transit with them: an actor always receives its last turn (Austin t33, 2026-10-05:
        # a carried phone, its clock a pulse behind its carrier's, was shown nothing of leaving and narrated the desk)
        ctx.tuples = [t for t in ctx.tuples if t.clock is None or t.clock <= time_now
                      or ctx.sources.get((t.perceiver, t.seq)) in ("last", "in_transit")]
        if ctx.place:
            ctx.present = {x for x in ctx.present if kala.present_at(x, ctx.place, time_now)}
        kala_lines.append(time_line(kala, time_now, own if own is not None else time_now, max_elapsed, rate))
        # a closed scene is NOT shown: told, the Narrator turns the lock into a fact of the world (a
        # building "locked fast until morning", with a player inside -- live, 2026-09-24). The pulse
        # goes where the action goes, and a closed destination asks its player after (below).
        if window:
            kala_lines.append(window_line(kala, window))
        if setting:  # v4.32: the window's quiet turns -- the same hours around the actor, the same sky
            kala_lines.append(setting_line([(log.kosha.label(s), text) for s, text in setting]))
        kala_lines += shown or []  # routes/spanda.py's: a window turn is shown the player's own time in it
    # janaśruti (v4.62): what has reached this scene by now as talk -- heavy moments elsewhere, on the world's clock
    if ctx.place and not as_chronicle:
        from smritilila.smrti.janasruti import heard, talk_line
        talk = heard(log.kosha, canon.entities, memories, ctx.place, world_time + 1, time_now)
        if talk:
            kala_lines.append(talk_line(log.kosha, talk, time_now is not None))
            log_event("smrti.talk", pulse_id=pulse_id, actor=actor_id, scene=ctx.place, heard=[(o, a, w) for o, _t, a, w in talk])
    if kala is not None:
        if lived:
            kala_lines.append(lived_line(kala, lived))
        elif action_text is None and max_elapsed is not None:
            # a world's turn: what players already lived here and next door in its window (held, not capped)
            nearby = near_lived(kala, [x for x in [ctx.place, *ctx.neighbours] if x], time_now,
                                time_now + max(0, max_elapsed), actor_id, playable)
            if nearby:
                kala_lines.append(near_line(kala, nearby))
                log_event("smrti.near_lived", pulse_id=pulse_id, actor=actor_id, lived=nearby)
    # v5 §21 v4.27 (Marc, 2026-09-27: "I couldn't change anything in the scene that's been future locked ... the
    # player who locked shouldn't be seen either"): a place lived further along by others is open to a player,
    # but whoever and whatever is already lived later there is out of their reach -- not perceived, not moved,
    # not written for. What they lived stays as they lived it; if it is now a little untrue, that is a scar.
    ahead: set[str] = set(hidden or ())
    if kala is not None and ctx.place and actor_id in playable and as_chronicle is None and not as_origin:
        _u, why_here = kala.locked_why(ctx.place, time_now, actor_id,
                                       changes_in(log.kosha, kala, ctx.place, time_now, world_time, set(log.kosha.perceivers)))
        ahead |= {w for _kind, w, _m in why_here}
    if ahead:
        ctx.present = {x for x in ctx.present if x not in ahead}
        ctx.tuples = [t for t in ctx.tuples if t.perceiver not in ahead]
        log_event("smrti.lived_later_hidden", pulse_id=pulse_id, actor=actor_id, scene=ctx.place, ids=sorted(ahead))
    # v5 §20 v4.31: the presents within reach. A player whose time is their own -- in the world, not one who lets
    # the world act for them (v5 §37, `movable`: those are carried, having waited) -- is met only at their present.
    # Ahead of now: shown as a bound, where the span would meet them. Behind: not there yet -- not shown, never
    # written for; they meet the actor when their own span reaches the actor's present.
    presents: dict[str, int] = {}
    behind: set[str] = set()
    if kala is not None and ctx.place and actor_id in playable and as_chronicle is None and not as_origin:
        reach = {ctx.place, *ctx.neighbours}
        # Marc, 2026-10-13 (Stonemarch 1577-1590: Jakal "caught up with" Dunstan at the smithy with no jump offered): a player who waits
        # (`movable`: offline, letting the world act for them) is carried only while level with the actor or behind -- the world takes them
        # along. One who waited AHEAD (his player raced on, then went quiet: Dunstan, eight days on) is carried by nothing, and is a present
        # like any other: met only at it, the pulse ends there, the long jump asks consent. Without this he was only hidden as lived past,
        # and the Narrator wrote the meeting in the actor's memory alone.
        waiting_ahead = {w for w in (set(movable) & set(playable)) - set(players) if (kala.clock(w) or 0) > time_now}
        for p in sorted((set(players) | waiting_ahead) - {actor_id} - set(sangama) - set(absent)):  # players: whose time is their own (adhisthana.py's WorldSession.unvoiced), and the waiting who are ahead
            c = kala.clock(p)
            if c is None or log.kosha.scene_of(p, world_time + 1) not in reach:
                continue
            if c > time_now:
                presents[p] = c
            elif c < time_now:
                behind.add(p)
        if behind:
            ctx.present = {x for x in ctx.present if x not in behind}
            log_event("smrti.not_yet_here", pulse_id=pulse_id, actor=actor_id, ids=sorted(behind))
        if presents:
            at = sorted({(c, log.kosha.scene_of(p, world_time + 1)) for p, c in presents.items()})
            kala_lines.append(presents_line(kala, time_now, [(log.kosha.label(w), c) for c, w in at], meet_at))
    # Apart (kala.apart, Marc 2026-09-30): whoever this moment is not the present of -- and what they hold, which
    # goes with them. Every write below consults it: moves, tuples, lines, becomings, witnessed copies, presence.
    away: frozenset[str] = frozenset()
    if kala is not None and time_now is not None:
        _away = set(apart(kala, kala.spans, time_now, playable, movable)) - {actor_id} - set(sangama)
        _held = list(_away)
        while _held:
            for y in log.kosha.holds(_held.pop(), world_time + 1):
                if y not in _away and y != actor_id:
                    _away.add(y)
                    _held.append(y)
        away = frozenset(_away | set(ahead))
        if away:
            log_event("smrti.apart", pulse_id=pulse_id, actor=actor_id, now=time_now, count=len(away), ids=sorted(away)[:20])
    if beyond:  # the place rerun: the regions its first telling made beside without seeing
        ctx.beyond = dict(beyond)
    if arriving:
        # the second pass: the scene (by its first tuple, its label) and each player there, by their
        # latest tuple, so they are shown -- and have handles -- like anyone here
        have = {(t.perceiver, t.seq) for t in ctx.tuples}
        for scene_id, there in arriving.items():
            for eid, pick in [(scene_id, 0)] + [(pid, -1) for pid in there]:
                s_ = log.stream(eid, before=world_time)
                if s_ and (s_[pick].perceiver, s_[pick].seq) not in have:
                    ctx.tuples.append(s_[pick])
                    ctx.sources[(s_[pick].perceiver, s_[pick].seq)] = "arriving"
                    have.add((s_[pick].perceiver, s_[pick].seq))
        ctx.tuples.sort(key=lambda t: (t.chronology, t.seq))
    if unshown:
        have = {(t.perceiver, t.seq) for t in ctx.tuples}
        for eid in unshown:
            s_ = log.stream(eid, before=world_time)
            if s_ and (s_[-1].perceiver, s_[-1].seq) not in have:
                ctx.tuples.append(s_[-1])
                ctx.sources[(s_[-1].perceiver, s_[-1].seq)] = "unshown"
        ctx.tuples.sort(key=lambda t: (t.chronology, t.seq))
    if kala is not None:
        ctx.tuples.sort(key=lambda t: (t.clock if t.clock is not None else -1, t.chronology, t.seq))
    for part, (eid, name) in (name_taken or {}).items():
        if eid not in {n[0] for n in ctx.namesakes}:
            ctx.namesakes.append((eid, name, part))  # by name and id only; never folded onto by itself (s34)
    mention_words: dict[str, str] = {}
    for _h, (eid, name) in (mention_named or {}).items():
        if eid not in {k[0] for k in ctx.known}:
            ctx.known.append((eid, name))  # shown, so it can be bound, moved and handled
        mention_words[eid] = named_in(log, canon, eid)
    handles = handle_map(ctx, {i: e.labels for i, e in canon.entities.items()})
    if handles.id_to_handle:
        log_event("smrti.handles", pulse_id=pulse_id, count=len(handles.id_to_handle))
    thread = thread_from_memory(canon, memories, actor_id, ctx.present, ctx.place, log.kosha) if action_text is not None else ""
    if thread:
        log_event("smrti.thread", pulse_id=pulse_id, actor=actor_id, thread=thread[:120])
    own_words = {}
    if actor_id in players or actor_id in playable:
        own_words = dict(log.kosha.actions_of(actor_id))  # their own words, by tick: indexed, not a pass over every action
    lived_at: dict[str, tuple[str, str]] = {}   # a named player lived past this moment: where their own record has them at it
    if kala is not None and time_now is not None:
        for _pid, _pname in ctx.players_named:
            if _pid in away and (kala.clock(_pid) or 0) > time_now:
                _sc = kala.where_at(_pid, time_now)
                if _sc:
                    lived_at[_pid] = (_pname, log.kosha.label(_sc))
        # Marc, 2026-10-13 (Stonemarch 1577-1579, "Let's try A"): the actor stands in the very scene such a player lived in at this moment, so
        # they are present, hidden as lived later, and never reach players_named (it skips whoever is present): the Narrator was told
        # nothing and a freshly minted guard told Jakal what Dunstan was doing and where he would be. Named the same way (a part of their
        # name found whole in the action or the actor's own memory shown), they get the same line, with the place the actor is in.
        from smritilila.smrti.kosha import whole_word
        _said = (action_text or "") + "\n" + "\n".join(t.memory for t in ctx.tuples if t.perceiver == actor_id)
        for _pid in sorted((set(playable) & set(away)) - {actor_id} - set(lived_at)):
            if (kala.clock(_pid) or 0) > time_now:
                _parts = log.kosha.name_parts(_pid)
                _sc = kala.where_at(_pid, time_now) if _parts and re.search(whole_word(*sorted(_parts)), _said) else None
                if _sc:
                    lived_at[_pid] = (log.kosha.label(_pid), log.kosha.label(_sc) + (", where the actor is now" if _sc == ctx.place else ""))
    view_text = view_block(log.kosha, ctx, world_time, library, movable,
                           {t.perceiver for t in ctx.tuples} | set(ctx.present) | set(ctx.neighbours) | ({ctx.place} if ctx.place else set()),
                           players=frozenset(players), arriving=arriving, returning=returning, name_taken=name_taken,
                           mention_named=mention_named, mention_words=mention_words, out_of_time=away, lived_at=lived_at)
    if kala_lines:
        view_text = "\n".join([view_text] + kala_lines if view_text else kala_lines)
    if holds_on():  # what the things the actor holds hold, nothing included (SMRITILILA_HOLDS, Marc 2026-10-13)
        fill_holds(log.kosha, ctx, world_time + 1)
    if not_here:  # the shadow rerun's line: who the first telling wrote for, and where they really stand
        line = ("NOT HERE (named in the memory or the action, but standing elsewhere now -- they are not in this telling: not "
                "seen, heard, met or written for; they may arrive only by their own time): "
                + "; ".join(f'{x} "{log.kosha.label(x)}", at {log.kosha.label(w)}' for x, w in sorted(not_here.items())))
        view_text = "\n".join([view_text, line]) if view_text else line
    if no_scene:  # the no-scene rerun's line: who the first telling left standing in no scene
        line = ("IN NO SCENE (the first telling left each of these standing in nothing that is a scene -- in a place that "
                "is not one, or in nothing at all; put each, or what holds it, in the smallest scene of THE TREE where it "
                "is): " + "; ".join(f'{x} "{log.kosha.label(x)}", in {w or "nothing"}'
                                    for x, w in sorted(no_scene.items())))
        view_text = "\n".join([view_text, line]) if view_text else line
    if holder_full:  # the fanout rerun's line: a holder the first telling put new places straight into, already full
        line = ("A HOLDER IS FULL (the first telling put " + "; ".join(
                    f'{", ".join(new)} directly in {h} "{name}", which already holds {n} places, listed in THE TREE'
                    for h, (name, n, new) in sorted(holder_full.items()))
                + "): do not add to it directly, and do not file the new place inside a place it is not part of. Make one new place "
                  "for the group inside the holder -- a lane, a wing, a floor, a quarter, whatever this world calls such a part, "
                  "named for what the new place belongs to -- and put the new place in it.")
        view_text = "\n".join([view_text, line]) if view_text else line
    if unshown:
        # Stonemarch t148: Elmo written "waiting at the east track" from the plan in memory, while the record had him at
        # the trading post -- his move refused, the narration kept. Shown where they are, the telling can agree
        line = ("NOT HERE, SHOWN WHERE THEY ARE (this pulse, told without them, had them here): "
                + "; ".join(f"{eid}, in {log.kosha.scene_of(eid, world_time) or 'nothing'}" for eid in unshown)
                + ". One of them is here this pulse only if `moves` brings them -- and only if they would come.")
        view_text = "\n".join([view_text, line] if view_text else [line])
    # daiva: an attempt's fortune -- players' and the world's own turns alike; not a quiet
    # scene's "did anyone come" (it would bend arrivals), not the chronicle (the age is no attempt)
    # the model that will narrate, known before the prompt is built: a rotating client plans its pick
    narrator_model = client.plan() if hasattr(client, "plan") else getattr(client, "model", None)
    # v5 §15: the dice are the attempt's, so they are read where it happens -- the scene's branch, else the earth
    here_daiva = log.kosha.seed("daiva", ctx.place, world_time + 1) if ctx.place else None
    fortune_data = view.get("world_data") if here_daiva is None else {**(view.get("world_data") or {}), "daiva": here_daiva}
    fortune = (daiva(fortune_seed(actor_id, world_time), fortune_data, narrator_model)
               if not as_origin and as_chronicle is None else None)
    if fortune:
        view_text = "\n".join([view_text, daiva_line(fortune)] if view_text else [daiva_line(fortune)])
        log_event("smrti.daiva", pulse_id=pulse_id, actor=actor_id, word=fortune, autonomous=action_text is None,
                  model=narrator_model)
    moment: list[tuple[str, str | None, str | None]] = []
    if sangama:
        told = round_lines({}, still)  # v5 §28: the actions are in THIS MOMENT, in no order -- not in the order pressed
        for pid in acting:
            w = daiva(fortune_seed(pid, world_time), fortune_data, narrator_model)  # each one's own, by who and when
            moment.append((pid, acting[pid], w))
            if w:
                told.append(round_daiva_line(pid, w))
                log_event("smrti.daiva", pulse_id=pulse_id, actor=pid, word=w, autonomous=False, sangama=True,
                          model=narrator_model)
        view_text = "\n".join([view_text] + told if view_text else told)
    # v5 §28 saṅkalpa: each one else who may act, with a fortune of their own (v4.35's "fortune is the players' alone"
    # withdrawn, Marc 2026-10-01)
    may_act = (who_may_act(log, ctx, actor_id, world_time, action_text, frozenset(players) | frozenset(playable),
                           frozenset(ahead) | frozenset(hidden or ()), pacing=(view.get("world_data") or {}).get("pacing"))
               if not as_origin and as_chronicle is None else [])
    for p in may_act:
        w = daiva(fortune_seed(p, world_time), fortune_data, narrator_model)
        moment.append((p, None, w))
        if w:
            log_event("smrti.daiva", pulse_id=pulse_id, actor=p, word=w, autonomous=True, sankalpa=True, model=narrator_model)
    if moment:
        moment.append((actor_id, action_text if action_text is not None else OWN_TURN, fortune))
        lines = moment_lines(moment, f"{ctx.place}|{world_time}")
        # v5.25 (Marc, 2026-10-08): said on the first pass, not only after the slip. Replayed on the 71 Stonemarch, Crossing and Ravenmoor cases
        # that had been told again: the plain first prompt gave a player an intention in 124 of 266 answers (57 of 71 cases); with this line,
        # 0 of 269, and the intentions for those who may act unchanged (3.78 and 3.82 a call)
        acting_players = sorted(pid for pid, act, _w in moment if act is not None and act != OWN_TURN and (pid in players or pid in playable))
        if acting_players:
            lines.append("NOT TO INTEND: " + "; ".join(f"{x} (a player with an ACTION above: their deeds are their own)" for x in acting_players)
                         + ". Only those listed as able to act (may act) form an intention this moment.")
        if not_to_intend:
            lines.append("TOLD AGAIN: these may not intend anything this moment -- " + "; ".join(
                f"{x} ({'a player: their deeds are their own' if x in players or x in playable else 'not listed as able to act'})"
                for x in sorted(not_to_intend)) + ".")
        view_text = "\n".join([view_text] + lines if view_text else lines)
    out: PulseOut = narrate(client, ctx, action, places, handles, thread, own_words=own_words, action_source=action_source,
                            darsana=view.get("darsana"), view_text=view_text,
                            source=(view.get("world_data") or {}).get("source"), faces=sandhi, in_round=bool(sangama))
    _clean(out, canon, actor_id, pulse_id)

    new_handles = frozenset(
        new_handle(h) for h in
        [t.get("perceiver", "") for t in out.perceiver_tuples] + [p.get("place", "") for p in out.place_tuples]
        + [x for p in out.place_tuples for x in (p.get("present") or [])]
        + [m.get("entity", "") for m in out.moves] + [m.get("into", "") for m in out.moves]
        + [d.get("id", "") for d in out.new_entities] + [r.get("entity", "") for r in out.becomings]
        if isinstance(h, str) and new_handle(h).startswith("NEW:"))
    b = Binder(canon, ctx, places, world_time, pulse_id, id_factory, new_handles, log=log)
    pl = _place(out, b, actor_id, players, as_origin, movable, library, mentions=sandhi, frozen=frozenset(ahead), away=away)
    second_pass = (arriving is not None or beyond is not None or name_taken is not None or mention_named is not None
                   or lived is not None or window is not None
                   or unshown is not None or hidden is not None or meet_at is not None or setting is not None
                   or not_here is not None or no_scene is not None or not_to_intend is not None
                   or holder_full is not None)  # one rerun per pulse, whichever
    # the fanout rerun's finding: new places put straight into a holder that already lists FANOUT_CAP or more (carried by any rerun)
    full_holders = holders_full(b.kosha, b, pl.records, pl.t_after) if not second_pass and as_chronicle is None else {}
    lived_rerun: frozenset[str] = frozenset()
    # the telling's own writes show who it thought was here (Stonemarch 2026-09-28, the east track: shadow-Dunstan and
    # shadow-Elmo crouched in the brush while both stood at the inn's kitchen postern; the engine dropped the writes,
    # the narration kept them). Anyone written for who stands outside the reach of where the pulse began and ended
    shadows: dict[str, str] = {}
    if action_text is not None and as_chronicle is None:
        t1 = pl.t_after
        reach_: set[str] = set()
        for s_ in (ctx.place, pl.scene_after(b, actor_id)):
            if s_ and s_ != NOWHERE:
                reach_ |= {s_, *log.kosha.neighbours(s_, t1)}
        for t in out.perceiver_tuples:
            raw = str(t.get("perceiver", "")).strip()
            who = b.resolved.get(new_handle(raw)) or raw
            if (who not in canon.entities or who == actor_id or who in b.minted or who in sangama
                    or log.kosha.is_scene(who)):
                continue
            where = log.kosha.scene_of(who, t1)
            # and stood out of reach before the pulse too: what the telling itself carried off is not a shadow
            # (Stonemarch 2026-09-28, the fen: the purse Jakal held, dropped into a bog hole left in the region above)
            was = log.kosha.scene_of(who, b.world_time)
            if where and where != NOWHERE and where not in reach_ and (not was or was not in reach_):
                shadows[who] = where
    # IN NO SCENE (Marc, 2026-09-28: "dig into how Kobort ended up nowhere"): Stonemarch 2026-09-27 19:19, the ditch --
    # the Narrator minted "the lane to the Welcome Wench" as a thing beside the scene of that name and moved Kobort
    # and Turuko into it: in the village, which is no scene, they stood in none, and no one could meet them after
    # (a night later the Narrator minted a look-alike ruffian for Kobort). v5 §11's root_as_scene counted it and did no
    # more. Anyone this telling wrote for or moved, left in no scene, has it told again. Not a place declared far
    # off (a city, a road, the moathouse): only what the telling touched. Stonemarch's history: 7 of 713 pulses left
    # something in no scene -- among them Kobort and Turuko, and the purse in the bog hole; most were places declared
    lost: dict[str, str | None] = {}
    if action_text is not None and as_chronicle is None:
        t1 = pl.t_after
        touched = {b.resolved.get(new_handle(str(t.get("perceiver", "")).strip())) or str(t.get("perceiver", "")).strip()
                   for t in out.perceiver_tuples} | (set(pl.moved) - set(b.minted))
        for x in sorted(e for e in touched if e):
            if x not in canon.entities or log.kosha.is_scene(x) or log.kosha.is_place(x, t1):
                continue
            s_ = log.kosha.scene_of(x, t1)
            if s_ is None or log.kosha.is_root_scene(s_):
                c_ = log.kosha.container_at(x, t1)  # by name: what the first telling minted leaves with it
                lost[x] = b.label_of(c_) if c_ else None
    overran: int | None = None  # v5 §20 v4.31: the present a first telling ran past
    met_event: dict | None = None  # logged only by the telling that is kept (a rerun after it discards it)
    pending = None
    elapsed: int | None = None  # the pulse's minutes, in a world that keeps a clock (smrti.energy reads it)
    if kala is not None:
        dest = pl.scene_after(b, actor_id)
        dest = None if dest in (None, NOWHERE) else dest
        # v5 §19: the Narrator gives the time LIVED here, as long as it says (v5.31); the world moves as far as the rate says
        lived = _minutes(out.raw.get("elapsed_minutes"))
        elapsed = world_span(lived, rate)
        if max_elapsed is not None:
            # a world's turn covers the span it was given (kala.time_line), whatever it said -- as lived here
            lived = lived_span(max_elapsed, rate)
            elapsed = max(0, max_elapsed) if rate == 1.0 else min(max(0, max_elapsed), world_span(lived, rate))
        end = time_now + elapsed
        perceivers = set(log.kosha.perceivers)
        # v5 §20 v4.31 (Marc, 2026-09-28: "It feels like it should stop where they meet"): every span stops at the
        # first present it touches. The bound was shown before the telling (presents_line); a telling that ran
        # past a present within reach of where it ends is told once more with the bound restated, and a second
        # pass that runs past it again ends where they meet. The ten-minute wait (v4.28) is withdrawn: it was a
        # local guess at this.
        if presents:
            where_end = dest or ctx.place
            reach_end = {where_end, *log.kosha.neighbours(where_end, world_time + 1)} if where_end else set()
            reached = sorted(c for p, c in presents.items() if c <= end and log.kosha.scene_of(p, world_time + 1) in reach_end)
            if reached:
                met = reached[0]
                if met < end and not second_pass and meet_at is None:
                    overran = met
                else:
                    if met < end:
                        elapsed, end = met - time_now, met
                    met_event = dict(pulse_id=pulse_id, actor=actor_id, at=met, scene=where_end,
                                     players=sorted(p for p, c in presents.items() if c == met), clamped=meet_at is not None)
        # v5 §21-§22: a scene closed to the actor at the time they would arrive -- consent first, before
        # any rerun narrates again a pulse the player may not want
        if dest and dest != ctx.place:
            until, why = kala.locked_why(dest, end, actor_id, changes_in(log.kosha, kala, dest, end, world_time, perceivers))
            if until and not (wait_until is not None and wait_until >= until):
                if actor_id in playable and not as_origin:
                    # v5 §21 v4.27: a player goes in; who and what is lived further along there is out of their
                    # reach. If this telling touched a PLAYER'S CHARACTER who lived later, it is told once more with them hidden.
                    # v5.50 (Marc, 2026-10-19: "lived-later people who are not a player's character stay seen but are not written
                    # for"): the world's own people and things are not hidden by a second telling. It had to invent them again, and
                    # three of the 36 reruns since 09-22 made a twin (Jaroo Ashstaff, Miller Mytch, Tad: the first telling had the
                    # real one, its only offence a tuple for him). The first telling stands; what it wrote for or moved of them
                    # is dropped as it already is on a second pass (`ahead`, `away`), and counted
                    there = {w for _kind, w, _m in why} - ahead
                    touched = set(pl.moved) | {b.resolve(str(t.get("perceiver", ""))) for t in out.perceiver_tuples}
                    touched |= {b.resolve(str(x)) for _s, row in pl.scene_rows for x in (row.get("present") or [])}
                    if there & touched & playable and not second_pass:
                        lived_rerun = frozenset(ahead | there)
                    elif there & touched and not second_pass:
                        log_event("smrti.lived_later_kept", pulse_id=pulse_id, actor=actor_id, ids=sorted(there & touched))
                    ahead |= there
                else:
                    pending = {"kind": "locked", "until": until, "scene": dest,
                               "prompt": locked_prompt(kala, canon, dest, until, time_now, why, frozenset(playable))}
                    second_pass = True  # held: no rerun until the player answers
    def _again(reason: str, detail: str, fields: dict, **changes) -> SmrtiPulseResult:
        """This pulse narrated again: the first pass is discarded (what it minted leaves the canon; its
        answer, already captured as training data, is tagged, not imitated), and the same call runs with
        only `changes` -- e.g. who was unshown, whose name was taken, what others had lived.
        Its name: punarāvṛtti (पुनरावृत्ति), a turning back again (v5 §33)."""
        # one rerun per pulse, carrying everything the first telling got wrong (Marc, 2026-09-28): the names it
        # declared that another person already bears go with any rerun, not only the name rerun. Stonemarch t133:
        # the arrival rerun spent the pulse's one rerun, the second telling minted Ostler Gundigoot again beside
        # the one already in the taproom, and two Ostlers stood in one room. Shown the borne name, the Narrator
        # decides whether it means that person (s34: the engine never folds a person by name alone)
        if "not_here" not in changes and not_here is None and shadows:  # and who it wrote for that is not here
            changes["not_here"] = shadows
            fields = {**fields, "not_here": sorted(shadows)}
        if "no_scene" not in changes and no_scene is None and lost:  # and whom it left in no scene
            changes["no_scene"] = lost
            fields = {**fields, "no_scene": sorted(lost)}
        if "holder_full" not in changes and holder_full is None and full_holders:  # and the full holder it put a new place straight into
            changes["holder_full"] = full_holders
            fields = {**fields, "full": sorted(full_holders)}
        if "name_taken" not in changes and name_taken is None:
            declared = [Record(new_id(), "name", eid, str(d.get("name")).strip(), world_time, pulse_id)
                        for h, d in b.declared.items() if str(d.get("name") or "").strip()
                        and (eid := b.resolved.get(h)) in b.minted]
            borne = names_taken(log.kosha, declared, b.minted, world_time + 1) if declared else {}
            if borne:
                changes["name_taken"] = borne
                fields = {**fields, "taken": {p: v[0] for p, v in borne.items()}}
        if "mention_named" not in changes and mention_named is None and b.mention_hits:  # and what it made new of a mention
            changes["mention_named"] = dict(b.mention_hits)
            fields = {**fields, "mentions": {h: v[0] for h, v in b.mention_hits.items()}}
        if "beyond" not in changes and beyond is None and as_chronicle is None:
            # and the places it made beside places it was not shown -- before its mints leave the canon. Stonemarch t29
            # replayed (2026-10-04): a saṅkalpa rerun spent the pulse's one rerun, and the second telling minted
            # "Ruined Moathouse Approach" beside the seed's own moathouse scenes, never shown them (1 of 6)
            missed = places_beyond(log.kosha, b, ctx, world_time, pl.t_after)
            if missed:
                changes["beyond"] = missed
                fields = {**fields, "beyond": sorted(missed)}
        for e in b.new_entities:
            canon.entities.pop(e.id, None)
        log_event(f"smrti.{reason}", pulse_id=pulse_id, world_time=world_time, actor=actor_id,
                  first_narration=(out.narration or "")[:300], **fields)
        log_discarded(pulse_id=pulse_id, reason=reason, detail=detail)
        return run_smrti_pulse(**{**call, **changes})

    allowed = set(may_act)
    intents = [(str(r.get("who")), str(r.get("intends") or "")) for r in out.sankalpa]
    off = sorted({w for w, _i in intents if w not in allowed})
    off_players = [w for w in off if w in players or w in playable]
    # v5.25 (Marc, 2026-10-08): a player who HAS given an action this moment and is written an "intention" has had their own action restated.
    # All 76 recorded reruns were that (70 the actor, 13 another player acting in the same moment; never a player present and not acting).
    # Replayed, those answers' narrations gave the player something substantial in 3% of slipped answers against 5% of clean ones: the
    # second call mended nothing and cost 13 s. The entry is dropped and counted, as for anyone else off the list.
    has_acted = {pid for pid, act, _w in moment if act is not None and act != OWN_TURN}
    if action_text is not None:
        # the pulse with no moment block (Marc, 2026-10-13, Jakal at Stonemarch 1522): when nobody else may act there is no THIS MOMENT, so the
        # actor was not among those listed as having acted, an intention the Narrator wrote for the actor counted as a player not acting, and
        # the pulse was told again with its first telling thrown away (the oak it had made went with it). 8 of 9 saṅkalpa reruns since 10-08
        has_acted.add(actor_id)
    told_off = [w for w in off_players if w not in has_acted]
    if told_off and not second_pass:  # v5 §28: the saṅkalpa rerun -- for a player NOT acting: their deeds are their own (rule 10)
        return _again("sankalpa_rerun", "gave an intention to a player: " + ", ".join(told_off), {"who": told_off},
                      not_to_intend=frozenset(told_off))
    if off:  # anyone else off the list -- the lived-later the telling still saw (the watch item) -- dropped and counted;
        # a rerun did not mend the telling (replayed, Filonius at the Welcome Wench: 5 reruns, 0 mended; Marc 2026-10-01)
        log_event("smrti.sankalpa_refused", pulse_id=pulse_id, actor=actor_id, who=off,
                  lived_later=sorted(set(off) & set(ahead)), acting=sorted(set(off) & has_acted))
    if intents:
        log_event("smrti.sankalpa", pulse_id=pulse_id, actor=actor_id,
                  intents=[(w, i[:200]) for w, i in intents if w in allowed], may_act=may_act)
    if overran is not None:  # the meeting rerun (v5 §20 v4.31): it ran past a present it reached
        return _again("meeting_rerun", f"ran past a present at {overran}", {"meet_at": overran}, meet_at=overran)
    if lived_rerun:  # the lived-later rerun
        return _again("lived_later_rerun", "went where others have lived further along: " + ", ".join(sorted(lived_rerun)),
                      {"hidden": sorted(lived_rerun)}, hidden=lived_rerun)
    if not second_pass and as_chronicle is None:  # the place rerun
        missed = places_beyond(log.kosha, b, ctx, world_time, pl.t_after)
        if missed:
            return _again("place_rerun", "made a place beside places it was not shown, in " + ", ".join(sorted(missed)),
                          {"regions": sorted(missed)}, beyond=missed)
    if full_holders:  # the fanout rerun: a new place put straight into a holder that already lists FANOUT_CAP or more
        return _again("fanout_rerun", "put a new place straight into a full holder: " + ", ".join(
                          f"{h} ({n} places)" for h, (_name, n, _new) in sorted(full_holders.items())),
                      {"holders": {h: v[1] for h, v in full_holders.items()}}, holder_full=full_holders)
    if not second_pass and action_text is not None:  # the arrival rerun
        into = pl.scene_after(b, actor_id)
        if into and into != NOWHERE and into != ctx.place:
            # a player behind is not there yet, and a present shown as a bound was already shown
            unshown = sorted(p for p in log.kosha.occupants(into, pl.t_after, exclude={actor_id})
                             if p in players and p not in movable and p not in ctx.present and p not in ahead
                             and p not in behind and p not in presents)
            if unshown:
                return _again("arrival_rerun", f"took the actor into {into}, where {', '.join(unshown)} stood unshown",
                              {"into": into, "players": unshown}, arriving={into: unshown})
    told_to: dict[str, dict] = {}
    for r in out.round_narrations:
        pid = r["player"] if r["player"] in sangama else b.resolve(str(r["player"]))
        if pid in sangama and pid not in told_to:
            told_to[pid] = {"narration": str(r["narration"]), "suggested_next_action": str(r.get("suggested_next_action") or "")}
    if not second_pass and shadows:  # the shadow rerun: it wrote for someone who stands elsewhere
        return _again("shadow_rerun", "wrote for someone not here: " + ", ".join(sorted(shadows)), {})
    if not second_pass and as_chronicle is None and b.mention_hits:  # the mention rerun: it made new what was named before
        return _again("mention_rerun", "made new what was named in passing: "
                      + ", ".join(f"{h} ({v[0]})" for h, v in sorted(b.mention_hits.items())),
                      {"mentions": {h: v[0] for h, v in b.mention_hits.items()}}, mention_named=dict(b.mention_hits))
    if not second_pass and lost:  # the no-scene rerun: it left someone standing in no scene
        return _again("no_scene_rerun", "left in no scene: " + ", ".join(sorted(lost)), {})
    if not second_pass and as_chronicle is None and pl.unshown:  # the unshown rerun
        return _again("unshown_rerun", "moved someone it was not shown: " + ", ".join(pl.unshown), {"ids": pl.unshown},
                      unshown=pl.unshown)
    written, name_records = _append(out, b, pl, actor_id, as_origin, as_chronicle is not None, players, embed, hidden=frozenset(ahead), away=away, log=log,
                                    round_told={pid: told_to.get(pid, {}).get("narration", "") for pid in sangama})
    if not second_pass and as_chronicle is None:  # the name rerun
        taken = names_taken(log.kosha, name_records, b.minted, world_time + 1)
        if taken:
            return _again("name_rerun", "gave a name another person bears: " + ", ".join(sorted(taken)),
                          {"taken": {p: v[0] for p, v in taken.items()}}, name_taken=taken)
    records_written = list(pl.records) + name_records
    if as_chronicle is not None and age_kala is not None:
        # the age's clock: this entry covers from where the last one ended (kala.age_spans)
        span = age_minutes(out.raw.get("elapsed_minutes"))
        records_written.append(Record(new_id(), "time", actor_id, str(age_from + span), world_time, pulse_id, note=f"{age_from}||"))
        log_event("smrti.age_moved", pulse_id=pulse_id, began=age_from, ended=age_from + span, minutes=span)
    if kala is not None:
        # the mirror: a span others have already lived where this pulse went is shown, and held
        scenes_ = [dest] + ([ctx.place] if ctx.place != dest else [])
        here_now = {x for s_ in scenes_ if s_ for x in log.kosha.occupants(s_, world_time + 1)}
        # v4.27: out of reach is out of sight too; v4.31: so are the minutes before a present the span meets -- their past
        # and a player whose time is their own is never shown by the mirror at all: their past, in your place,
        # is out of reach (v4.31 region 1), and neither is what they hold. Stonemarch 271-272 (2026-09-28): shown
        # "at 7:57, Dunstan was here" at the church he had left for the green, Jakal's world later had Elmo say
        # Dunstan sat on the knoll steps
        span_lived = mirror(kala, scenes_, time_now, end, actor_id, here_now,
                            out_of_reach=frozenset(ahead) | frozenset(presents) | frozenset(players),
                            holders=lambda who: log.kosha.above(who, world_time + 1))
        if span_lived and pending is None and not second_pass and action_text is not None:
            return _again("mirror_rerun", "passed through a span others had lived",
                          {"lived": [(m, who) for m, who in span_lived]}, lived=span_lived)
        # a long jump: the player's consent first
        if pending is None and action_text is not None and actor_id in playable and not confirmed and elapsed > kala.confirm_minutes:
            jump = (f"You are about to jump ahead {_duration(elapsed)}" if rate == 1.0 else
                    f"You are about to spend {_duration(lived)} here -- {_duration(elapsed)} in the world outside")
            pending = {"kind": "long", "minutes": elapsed, "from": time_now, "to": end, "origin": ctx.place, "dest": dest,
                       "prompt": f"{jump} (to {kala.say(end)}). Proceed?"}
        # the minutes each perceiver lived, on their memory of it -- who lived it and where is read from
        # them (kala.py): where the pulse ends it is lived whole; elsewhere (the room left, a neighbour
        # who heard) only its first moment is -- a room left behind does not live the journey. `dest` alone
        # (2026-10-05, the pass's P1): it was `dest or ctx.place`, so an actor walking INTO NO SCENE left a room
        # that lived his whole walk -- Austin t32, the apartment and the notebook spanning 7:17-7:27 while Franklin
        # went down to the street. Replayed on every world: Austin alone, 2 pulses of 47, no clock anyone read moved.
        whole = dest
        # ... and whatever went WITH the actor lives it whole too: moved this pulse and written where the actor was
        # written (Austin t32, 2026-10-05: a phone carried out in a pocket, into no scene, was cut to its first minute
        # as if left behind; a pulse behind its carrier, its own turn showed it nothing of leaving and it lit on the desk)
        actor_written_at = next((m.location_id for m in written if m.entity_id == actor_id), whole)
        written = [dataclasses.replace(m, began=time_now, ended=end if (m.location_id == whole or m.entity_id == actor_id
                                                                        or m.entity_id in sangama
                                                                        or (m.entity_id in pl.moved and m.location_id == actor_written_at))
                                       else time_now)
                   for m in written]
        log_event("smrti.time", pulse_id=pulse_id, actor=actor_id, pending=(pending or {}).get("kind"), now=time_now,
                  elapsed=elapsed, end=end, origin=ctx.place, dest=dest, kala_ms=kala_ms,
                  **({"lived": lived, "rate": rate} if rate != 1.0 else {}))
        # v5 §44 counters (Marc, 2026-09-25: gather data before building the meeting prompt)
        if whole:
            # a round's members live this pulse's span with the actor: behind only if the round's start pulled them forward
            for p in sorted((log.kosha.occupants(whole, pl.t_after) & set(playable) | set(sangama)) - {actor_id}):
                theirs = kala.clock(p) if kala.clock(p) is not None else kala.start
                meets = time_now if p in sangama else end
                if theirs < meets - 10:  # a meeting across time: they keep their own clock, and are behind it
                    log_event("smrti.meeting_behind", pulse_id=pulse_id, actor=actor_id, player=p, scene=whole,
                              their_clock=theirs, end=meets, gap=meets - theirs, in_round=p in sangama,
                              pending=(pending or {}).get("kind"))
        for m in written:
            c = kala.clock(m.entity_id)
            if m.tier != "AMBIENT" and m.entity_id != actor_id and m.entity_id not in sangama and c is not None and c > end:
                # voiced before their own time: they have already lived past this moment
                log_event("smrti.voiced_ahead", pulse_id=pulse_id, actor=actor_id, voiced=m.entity_id, their_clock=c, end=end,
                          gap=c - end, pending=(pending or {}).get("kind"))
    leaks = memory_leaks(ctx, written, log, places, pulse_id)
    echoes = pratidhvani(written, log, pulse_id)
    player_named(log.kosha, written, actor_id, frozenset(players) | frozenset(playable), pulse_id)
    named_unknown = 0
    if action_text is not None and actor_id in playable:
        named_unknown = narration_named(log.kosha, {actor_id: out.narration or "", **{p: t["narration"] for p, t in told_to.items()}},
                                        log, written, [e for e, _n in ctx.known], world_time, pulse_id)
    if action_text is None and actor_id in playable and actor_id not in players:
        # v5 §37: the Narrator acted for a player who allows it, while they
        # were away -- recorded, so their own-words line shows it was not them
        records_written.append(Record(new_id(), "action", actor_id, "", world_time, pulse_id, note="autonomous"))
        log_event("smrti.player_autonomous", pulse_id=pulse_id, player=actor_id)
    if action_text is not None and actor_id in players:
        records_written.append(Record(new_id(), "action", actor_id, action_text, world_time, pulse_id, note=action_source or "typed"))
    for pid, a in acting.items():  # saṅgama: each actor's own words, as the actor's
        records_written.append(Record(new_id(), "action", pid, a, world_time, pulse_id, note=(sangama_sources or {}).get(pid) or "typed"))
    for pid in sangama:
        if pid not in told_to:
            # never left without a telling: their own tuple of the moment (first person) stands in
            own = next((m.experience for m in written if m.entity_id == pid), "")
            told_to[pid] = {"narration": own, "suggested_next_action": ""}
            log_event("smrti.round_narration_missing", pulse_id=pulse_id, player=pid, fallback="tuple" if own else "none")

    actor_at = pl.scene_after(b, actor_id) or NOWHERE
    bound = sorted({m.entity_id for m in written} | {r.entity_id for r in pl.records} | set(pl.moved))
    uses = {}
    for m in written:
        if m.tier == "PRIMARY":
            uses[m.use or "none"] = uses.get(m.use or "none", 0) + 1
    log_event(
        "smrti.pulse", pulse_id=pulse_id, world_time=world_time, actor=actor_id, autonomous=action_text is None, origin=as_origin,
        tuples=len(ctx.tuples), place=ctx.place, scene_after=None if actor_at == NOWHERE else actor_at, written=len(written),
        records=len(records_written), moves=sum(1 for r in pl.records if r.kind == "kosha"), new=[e.labels[0] for e in b.new_entities],
        guarded={h: i for h, i in b.resolved.items() if i not in b.minted}, dropped=b.dropped, uses=uses,
        action_source=action_source, leaks=len(leaks), duration_ms=int((time.monotonic() - start) * 1000),
        **({"sangama": sorted(sangama), "round_acting": sorted(acting)} if sangama else {}),
    )
    in_world = frozenset(playable) - frozenset(absent)
    steps, nearest = player_distance(log, ctx.place, world_time, in_world)
    personal = [m for m in written if m.tier in ("PRIMARY", "SECONDARY")]
    log_event(
        "smrti.energy", pulse_id=pulse_id, world_time=world_time, actor=actor_id,
        # what brought the present in
        player_action=action_text is not None and actor_id in playable, autonomous=action_text is None,
        arrival=as_origin, chronicle=as_chronicle is not None, news=bool(news), pull=bool(pull), elapsed=elapsed,
        # where the pulse stood relative to the players
        steps_to_player=steps, nearest_player=nearest, ticks_since_player=ticks_since_player(log, ctx.place, world_time, in_world),
        # what came back from its own past (pratidhvani), zeros included -- the rate's denominator
        sentences=sum(len(sentences(m.experience)) for m in personal), tuples=len(personal),
        echoes=len(echoes), dead=sum(1 for f in echoes if not f["alive"]),
        tics=sum(1 for f in echoes if not f["alive"] and f["streak"] >= ECHO_TIC),
        daiva=fortune,  # the fortune drawn, beside what came of it
        named_unknown=named_unknown,  # player narrations naming someone the player was never given
    )
    if met_event is not None:  # only the telling that is kept: past every rerun (2026-09-28, logged twice before)
        log_event("smrti.met", **met_event)
    return SmrtiPulseResult(
        narration=out.narration, memories_written=written, new_entities=b.new_entities,
        suggested_next_action=out.suggested_next_action, context_tuples=len(ctx.tuples), bound_ids=bound,
        resolved_new=b.resolved, dropped=b.dropped, sara=out.sara, said=out.said, turn=out.turn,
        suggested_next_action_from_memory=out.suggested_next_action_from_memory, elsewhere=sorted(set(pl.elsewhere)),
        records_written=records_written, scene=None if actor_at == NOWHERE else actor_at, pending=pending, route=pl.route,
        round_narrations=told_to, elapsed=elapsed,
    )


def neti_neti() -> None:
    """Deliberately empty and deliberately never called."""
