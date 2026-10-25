# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Phala (फल, "fruit" -- the fruit of action, what a deed yields its doer
afterwards; beside daiva, the part no doer controls). Marc, 2026-09-27:
experience never paid out in play -- Stonemarch's measure was in every prompt
and on every sheet, and across a fifteen-turn fight no single turn "finished"
anything in the models' eyes (the second model awarded none; the larger model, which wrote one turn
in five, would have). So the fruit is reckoned at REST, as the source itself
does it: after a player's pulse long enough to need consent (v5 §22 -- a march,
a night, a day's wait), one focused call looks at what the company lived since
their last reckoning and writes what it earned, as their own becomings (v5 §8).

The engine judges nothing (s0): it decides only WHEN to ask -- by structure,
a long pulse -- and WHOM -- the players of the pulse and whoever has a tuple of
their own in the span. What was finished and what it is worth are the view's,
and the reckoner's. A world opts in with world data "phala": {"ask": the
world's own question, "keys": the becoming keys it may write}; a world without
it is never reckoned. Nothing is narrated; nothing reaches a player's page.

Cost: one call per long player pulse, over a bounded span (SPAN_CAP tuples,
never the history)."""
from __future__ import annotations

from dataclasses import dataclass

from smritilila.memory.memory import Memory

SPAN_CAP = 120   # tuples shown of what the company lived since their last reckoning
TUPLE_CHARS = 400

SYSTEM = """You are the RECKONER of Smritilila, a persistent world. A span of the world has just come to rest -- a march, a night, a wait -- and you reckon what the company's deeds in it have yielded them, by THIS WORLD'S OWN VIEW below, which says what the measure is, what earns it and what it is worth. You write no story: only what is to be recorded.

You are given: the company (each by id, with the measure they bear now), and what they lived since their last reckoning -- their own memories and the record of the scenes they were in, oldest first.

Answer two things:
- finished: what the company FINISHED in this span that they, or those against them, had at stake -- each in one clause; [] if nothing was finished.
- becomings: the measure to record now, as the view says it is kept -- for EACH one of the company who took part in what was finished, never anyone else, and only ids from the company list. Each new record under a key replaces that one's last, so carry forward what they had and add what they earned. [] if nothing was finished.
Judge as the view and the world's source would; do not invent deeds the memories do not show."""

SCHEMA = {
    "type": "object",
    "properties": {
        "finished": {"type": "array", "items": {"type": "string"}, "description": "what the span finished, one clause each; [] if nothing"},
        "becomings": {"type": "array", "description": "the measure to record, per company member who took part; [] if nothing",
                      "items": {"type": "object", "properties": {
                          "entity": {"type": "string", "description": "an id from the company list"},
                          "key": {"type": "string", "description": "the becoming's key, as the view names it"},
                          "clause": {"type": "string", "description": "what now holds for them under that key"}},
                          "required": ["entity", "key", "clause"]}},
    },
    "required": ["finished", "becomings"],
}


@dataclass
class Company:
    ids: list[str]                 # players first, then everyone else with a tuple of their own in the span
    labels: dict[str, str]
    sheets: dict[str, list[tuple[str, str]]]   # id -> [(key, clause)] they bear now
    span: list[Memory]
    since: int = 0                 # the earliest mark among the players: where the span shown begins


def company_of(players: list[str], scene: str | None, kosha, mem_by_entity: dict, by_tick: dict, marks: dict[str, int],
               now: int, labels: dict[str, str]) -> Company:
    """Who and what the reckoner is shown: the pulse's players, the scenes
    they lived in since their marks, and every perceiver with a tuple of their
    own in those scenes since THEIR OWN mark -- wherever they are now. The
    fruit follows the doer (Marc, 2026-09-30): it took whoever stood in those
    scenes at the rest, and Zert, who fought at the gate and had gone on,
    was reckoned nothing while five others were. Each doer's mark is their
    own (`marks`: their latest `reckoned` record), so what is reckoned is reckoned
    once, whoever's rest calls it; one never reckoned counts from their first
    tuple, not the world's birth. Bounded to SPAN_CAP tuples, latest kept.
    Cost: the tuples of the ticks since the earliest player's mark."""
    def mark(x: str) -> int:
        if marks.get(x) is not None:
            return int(marks[x])
        own_ = mem_by_entity.get(x, ())
        return min((m.world_time for m in own_), default=now) - 1
    since = min((mark(p) for p in players), default=now)
    own = [m for p in players for m in mem_by_entity.get(p, ()) if mark(p) < m.world_time <= now]
    scenes = {m.location_id for m in own if m.location_id} | ({scene} if scene else set())
    window = [m for t in range(since + 1, now + 1) for m in by_tick.get(t, ())]
    rows = [m for m in window if m.entity_id == m.location_id and m.location_id in scenes]
    theirs = [m for m in window if m.entity_id != m.location_id and m.entity_id not in players and m.location_id in scenes
              and m.world_time > mark(m.entity_id)]
    took_part = sorted({m.entity_id for m in theirs})
    seen: set[str] = set()
    span = []
    for m in sorted(own + rows + theirs, key=lambda m: (m.world_time, m.id)):
        if m.id not in seen:
            seen.add(m.id)
            span.append(m)
    span = span[-SPAN_CAP:]
    ids = list(players) + [x for x in took_part if x not in players]
    return Company(ids=ids, labels={i: labels.get(i, i) for i in ids},
                   sheets={i: [(k, text) for _b, k, text in kosha.view_of(i, now + 1) if _b == i] for i in ids}, span=span,
                   since=since)


def render(company: Company, ask: str, labels: dict[str, str]) -> str:
    lines = ["THE COMPANY (id -- name -- the measure they bear now):"]
    for i in company.ids:
        sheet = "; ".join(f"{k}: {text}" for k, text in company.sheets.get(i, [])) or "(nothing recorded yet)"
        lines.append(f"- {i} -- {company.labels[i]} -- {sheet}")
    lines.append("")
    lines.append("WHAT THEY LIVED SINCE THEIR LAST RECKONING (oldest first; [whose memory, where, tick]):")
    for m in company.span:
        lines.append(f"[{labels.get(m.entity_id, m.entity_id)}, {labels.get(m.location_id, m.location_id)}, {m.world_time}] "
                     f"{m.experience[:TUPLE_CHARS]}")
    lines.append("")
    lines.append("THIS WORLD ASKS: " + ask.strip())
    return "\n".join(lines)


def accepted(out: dict, company: Company, keys: list[str] | None) -> tuple[list[dict], list[dict]]:
    """The reckoner's becomings the structure allows: an id of the company, a
    key the world permits (any, if it names none), a clause. (kept, refused)"""
    kept, refused = [], []
    allowed = {k.strip().lower() for k in keys} if keys else None
    for b in out.get("becomings") or []:
        if not isinstance(b, dict):
            continue
        ent, key, clause = str(b.get("entity") or "").strip(), str(b.get("key") or "").strip().lower(), str(b.get("clause") or "").strip()
        reason = None
        if ent not in company.ids:
            reason = "not_in_company"
        elif not key or (allowed is not None and key not in allowed):
            reason = "key_not_allowed"
        elif not clause:
            reason = "no_clause"
        (refused if reason else kept).append({"entity": ent, "key": key, "clause": clause, **({"reason": reason} if reason else {})})
    return kept, refused
