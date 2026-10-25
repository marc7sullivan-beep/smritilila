# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Pratiṣṭhā (प्रतिष्ठा): directed growth -- the world builder decides what a world is missing, by importance, and
proposes it; a god reviews; only then is it planted, and given breath. Marc, 2026-10-02: "If a world has a strong
backstory (historical or a book) then we should be able to let the world builder decide what to build based on
importance" -- and "for now we should view each new build as a proposal first so we can check it over."

The name: in a temple, prāṇa-pratiṣṭhā is the rite that installs breath into what has been made -- the step from a
seeded thing to a living one. Bīja births a world; vistāra grows it outward, undirected; pratiṣṭhā grows it where it
matters, and lets what it planted live.

Four steps, each its own function, the engine's part nirguṇa (v5 §1):
  survey  -- one model call: what the world is missing, from its source (what the story it comes from holds) and its own
             talk (what its memories speak of that it does not hold), ranked by importance and by nearness to play.
  write   -- one model call: the chosen thing as a seed: places in the existing tree, people with first memories,
             things (written words as a chain of pages, v4.60), and breath -- a branch's own pace (v4.61).
  check   -- deterministic: everything stands in the tree, no name already borne, nothing written for a player,
             bounded size. A proposal with an error is never planted.
  plant   -- the seed as entities, memories and records: inserts only, nothing rewritten.
The world's lived record outranks the source: the builder is shown what was lived and told never to contradict it.
"""
from __future__ import annotations

import json

from smritilila.graph.entity import Entity
from smritilila.ids import new_id
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.genesis import plant_person
from smritilila.smrti.narrator import salience_of, use_of
from smritilila.smrti.pulse import mnemonic_id

MAX_PLACES, MAX_PEOPLE, MAX_THINGS, MAX_PAGES = 12, 12, 16, 40
MAX_MAP_PLACES = 400  # a seed read from a map (smrti/manacitra.py, spec["map"]) is places only: one sheet's names, not one thing the world is missing
MAX_BREATH = 8  # far_every: 1 is every tick; the world's own far pace is 8

SURVEY_SYSTEM = """You are the world builder for Smritilila, a world kept as memories. You are shown a world as it stands: its \
source (the story, history or setting it comes from), the world at large, the places it holds (by id), the people and \
beings it holds, what has been lived lately, and what the players are doing now. Find what the world is MISSING that \
matters: a power, a place, a person, a thing or an unfolding event the source holds and the world does not -- or that \
the world's own memories speak of (a name, a rumour, a 'quiet man', 'strangers in the marsh') with no being behind it. \
Never propose what the world already holds under another name. Rank each by importance (how much it matters to the \
story the world is living: 5 is a power that shapes the region) and nearness (how soon the players' own threads will \
reach it: 5 is the next few scenes). Then choose the one to build first: what both matters and is near; a great power \
far off may come first, so it can be living before anyone arrives. Never explain."""

SURVEY_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {"type": "array", "items": {"type": "object", "properties": {
            "what": {"type": "string", "description": "what is missing, in a sentence"},
            "why": {"type": "string", "description": "why it matters: what in the source, or in the world's own talk, says so"},
            "source": {"type": "string", "enum": ["source", "talk", "both"]},
            "importance": {"type": "integer"}, "nearness": {"type": "integer"},
            "anchor": {"type": "string", "description": "the id of the existing place it belongs in or nearest to"},
        }, "required": ["what", "why", "source", "importance", "nearness", "anchor"]}},
        "choice": {"type": "integer", "description": "the index of the candidate to build first"},
    },
    "required": ["candidates", "choice"],
}

WRITE_SYSTEM = """You are the world builder for Smritilila, a world kept as memories. Write ONE thing the world is missing as a \
seed, to be planted into the world as it stands. Everything you write is the world's from the moment it is planted -- \
written in the world's idiom, accurate to its source where the source speaks, and NEVER contradicting what the world has \
lived (shown below): where the source and the lived world disagree, the lived world is true, and you write the source's \
thing as it would stand now, after what was lived.

The seed:
- places: each a key, a label, a declared name if it bears one, `in` -- the id of an existing place shown, or the key of \
one of your own -- whether it is a scene (a room, a hall, a clearing: somewhere people stand together) or a region that \
holds scenes, and its first tuple: an impersonal account of what is there.
- people: each a key, a label (what they are, as anyone would see them -- never their name), their declared name, `in` -- \
the key or id of the SCENE they stand in (or of a thing in it) -- and their first tuples: two to four, first person, past \
tense, what they are, what they want, what they know and whom they serve or deal with; each with a weight (indelible, \
marking, weighty, notable, ordinary) and a use (key, useful, none). Bind to beings the world already holds by naming them \
as they are named below; never write someone who already exists.
- things: what matters enough to be held -- each a key, a label, `in` (a person's or place's key, or an existing id), a \
first tuple of what it is, and, if it bears writing, `pages`: its words, one page each, in order.
- breath: if the seed is a power that acts whether or not anyone is near (a temple, a court, a band on the move), the key \
of its outermost place and how often its people act when no player is near: far_every 1 (every turn of the world) to 8 \
(rarely). Leave it out for anything that only waits.

Keep it to what matters: a few places, the people who move things, the few things that count. Never explain."""

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "why": {"type": "string"},
        "places": {"type": "array", "items": {"type": "object", "properties": {
            "key": {"type": "string"}, "label": {"type": "string"}, "name": {"type": "string"},
            "in": {"type": "string"}, "scene": {"type": "boolean"}, "tuple": {"type": "string"},
        }, "required": ["key", "label", "in", "scene", "tuple"]}},
        "people": {"type": "array", "items": {"type": "object", "properties": {
            "key": {"type": "string"}, "label": {"type": "string"}, "name": {"type": "string"}, "in": {"type": "string"},
            "tuples": {"type": "array", "items": {"type": "object", "properties": {
                "text": {"type": "string"}, "weight": {"type": "string"}, "use": {"type": "string"}},
                "required": ["text", "weight", "use"]}},
        }, "required": ["key", "label", "name", "in", "tuples"]}},
        "things": {"type": "array", "items": {"type": "object", "properties": {
            "key": {"type": "string"}, "label": {"type": "string"}, "in": {"type": "string"}, "tuple": {"type": "string"},
            "pages": {"type": "array", "items": {"type": "string"}},
        }, "required": ["key", "label", "in", "tuple"]}},
        "breath": {"type": "object", "properties": {"in": {"type": "string"}, "far_every": {"type": "integer"}}},
    },
    "required": ["title", "why", "places", "people", "things"],
}


def world_view(source: str | None, bhuvana: str | None, places: list[tuple[str, str, str, str | None]],
               people: list[tuple[str, str, str | None, str | None]], lived: list[str], threads: list[str]) -> str:
    """What the builder is shown, bounded by its callers: places (id, label, scene|region, in), people (id, label, name,
    where), a window of what was lived, the players' latest."""
    lines = [f"SOURCE: {source.strip() if source else '(none given -- the world itself is the source)'}"]
    if bhuvana:
        lines.append(f"THE WORLD AT LARGE: {bhuvana.strip()[:1500]}")
    lines.append("PLACES (id: label -- kind, in):")
    lines += [f"  {i}: {label} -- {kind}, in {up or 'nothing'}" for i, label, kind, up in places]
    lines.append("PEOPLE AND BEINGS (id: label, declared name -- where):")
    lines += [f"  {i}: {label}{', ' + name if name else ''} -- {where or 'in no scene'}" for i, label, name, where in people]
    lines.append("LIVED LATELY (the world's memories, oldest first):")
    lines += [f"  {x}" for x in lived]
    lines.append("THE PLAYERS NOW (their latest):")
    lines += [f"  {x}" for x in threads]
    return "\n".join(lines)


def survey(client, view: str) -> dict:
    out = client.complete_with_tool(SURVEY_SYSTEM, view, tool_name="survey", tool_description="What the world is missing, ranked.",
                                    input_schema=SURVEY_SCHEMA, max_tokens=8000)
    cands = [c for c in (out.get("candidates") or []) if isinstance(c, dict) and str(c.get("what") or "").strip()]
    choice = out.get("choice") if isinstance(out.get("choice"), int) and 0 <= out.get("choice", -1) < len(cands) else 0
    return {"candidates": cands, "choice": choice}


def write(client, view: str, what: str, why: str = "", anchor: str = "") -> dict:
    user = "\n".join([view, "", f"WRITE THIS: {what}", *([f"WHY IT MATTERS: {why}"] if why else []),
                      *([f"IT BELONGS IN OR NEAR: {anchor}"] if anchor else [])])
    out = client.complete_with_tool(WRITE_SYSTEM, user, tool_name="pratistha", tool_description="One seed, to be planted.",
                                    input_schema=WRITE_SCHEMA, max_tokens=16000)
    return out if isinstance(out, dict) else {}


def check(spec: dict, kosha, players: set[str], t: int) -> tuple[list[str], list[str]]:
    """Deterministic: (errors, warnings). An error means the seed is never planted as it stands."""
    errors, warnings = [], []
    places, people, things = spec.get("places") or [], spec.get("people") or [], spec.get("things") or []
    if not places and not people and not things:
        errors.append("the seed is empty")
    if spec.get("map"):  # a map's seed is places and nothing else, and as many as the sheet names
        if people or things:
            errors.append("a seed read from a map holds places only")
        too_many = len(places) > MAX_MAP_PLACES
    else:
        too_many = len(places) > MAX_PLACES or len(people) > MAX_PEOPLE or len(things) > MAX_THINGS
    if too_many:
        errors.append(f"too large: {len(places)} places, {len(people)} people, {len(things)} things")
    keys: dict[str, str] = {}
    for kind, rows in (("place", places), ("person", people), ("thing", things)):
        for r in rows:
            k = str(r.get("key") or "").strip()
            if not k or k in keys or k in kosha.canon.entities:
                errors.append(f"{kind} {k or '(no key)'}: a key must be new and unique")
            keys[k] = kind
    ents = kosha.canon.entities
    labels_taken = {e.labels[0].casefold() for e in ents.values() if e.labels}
    # A near-duplicate is a warning, never an error (the builder may mean a second one): the engine's own paraphrase test (v5 §26), two rare words of an
    # existing scene's label standing in the new place's label and name. Found 2026-10-10 on Stonemarch: a focused build of the river quarter made "Lower Market"
    # beside the existing "approach to the lower market", and the check was silent. Cost: the world's labels once, and its scene labels for each new place.
    from smritilila.smrti.context import NAMED_BY_WORDS, named_by_words, world_rare
    scenes_now = [i for i in ents if kosha.is_scene(i)]
    rare = world_rare(kosha) if scenes_now else set()

    def resolves(x: str) -> bool:
        return x in keys or x in ents

    for r in places:
        if not resolves(str(r.get("in") or "")):
            errors.append(f"place {r.get('key')}: in {r.get('in')!r}, which neither the world nor the seed holds")
        if str(r.get("in")) in players:
            errors.append(f"place {r.get('key')}: in a player")
        if str(r.get("label") or "").casefold() in labels_taken:
            warnings.append(f"place {r.get('key')}: the label {r.get('label')!r} is already borne")
        else:
            for x in sorted(named_by_words(kosha, f"{r.get('label') or ''} {r.get('name') or ''}", scenes_now, NAMED_BY_WORDS, rare)):
                warnings.append(f"place {r.get('key')}: reads as {x} {ents[x].labels[0]!r}, which the world already holds; planted, it would be a second one beside it")
    seed_scenes = {str(r["key"]) for r in places if r.get("scene")}

    def in_a_scene(x: str) -> bool:
        """A scene found walking up from x -- not the root a scene-less thing would share (v5 §11)."""
        seen: set[str] = set()
        while x and x not in seen:
            if kosha.is_scene(x):
                return True
            seen.add(x)
            x = kosha.container_at(x, t)
        return False
    for r in people:
        where = str(r.get("in") or "")
        if not resolves(where) or where in players:
            errors.append(f"person {r.get('key')}: stands in {where!r}, not a place the world or the seed holds")
        elif where in ents and not in_a_scene(where):
            errors.append(f"person {r.get('key')}: {where} is in no scene")
        elif where in keys and keys[where] == "place" and where not in seed_scenes:
            errors.append(f"person {r.get('key')}: {where} is a region of the seed, not a scene to stand in")
        name = str(r.get("name") or "").strip()
        if name and kosha.named(name):
            errors.append(f"person {r.get('key')}: someone in the world is already called {name}")
        if not [x for x in r.get("tuples") or [] if isinstance(x, dict) and str(x.get("text") or "").strip()]:
            errors.append(f"person {r.get('key')}: no first tuples")
    for r in things:
        if not resolves(str(r.get("in") or "")) or str(r.get("in")) in players:
            errors.append(f"thing {r.get('key')}: in {r.get('in')!r}, not the world's or the seed's (and never a player's)")
        if len(r.get("pages") or []) > MAX_PAGES:
            errors.append(f"thing {r.get('key')}: more than {MAX_PAGES} pages")
    b = spec.get("breath") or {}
    if b:
        if not resolves(str(b.get("in") or "")):
            errors.append(f"breath: in {b.get('in')!r}, not a place of the world or the seed")
        if not isinstance(b.get("far_every"), int) or not 1 <= b["far_every"] <= MAX_BREATH:
            errors.append(f"breath: far_every must be 1-{MAX_BREATH}")
    return errors, warnings


def plant(spec: dict, now: int, minute: int | None, pulse_id: str, embed) -> tuple[list[Entity], list[Memory], list[Record], dict[str, str]]:
    """The seed as inserts: (entities, memories, records, key -> id). What a thing bears written is filed in no place
    at the world's start, a page inside the one before it (v4.60); a person is planted as plant_person plants."""
    ents, mems, recs, ids = [], [], [], {}
    for r in spec.get("places") or []:
        ids[r["key"]] = mnemonic_id(r["key"], new_id())
    for r in spec.get("people") or []:
        ids[r["key"]] = mnemonic_id(r["key"], new_id())
    for r in spec.get("things") or []:
        ids[r["key"]] = mnemonic_id(r["key"], new_id())

    def idof(x: str) -> str:
        return ids.get(x, x)

    for r in spec.get("places") or []:
        pid, up = ids[r["key"]], idof(str(r["in"]))
        ents.append(Entity(pid, [r["label"]], now, "genesis", avarana="scene" if r.get("scene") else None, kosha=up))
        recs += [Record(new_id(), "kosha", pid, up, now, pulse_id, note="declared"),
                 Record(new_id(), "name", pid, str(r.get("name") or r["label"]).strip(), now, pulse_id)]
        text = str(r.get("tuple") or "").strip()
        if text:
            mems.append(Memory(new_id(), pid, frozenset(), text, "AMBIENT", 5, pid, now, pulse_id, embed(text),
                               began=minute, ended=minute))
    for r in spec.get("people") or []:
        tuples = [{"text": str(x["text"]).strip(), "weight": x.get("weight"), "use": x.get("use")}
                  for x in r.get("tuples") or [] if isinstance(x, dict) and str(x.get("text") or "").strip()]
        person, pm, placed = plant_person(r["label"], tuples, idof(str(r["in"])), now, pulse_id, embed, eid=ids[r["key"]], minute=minute)
        ents.append(person)
        mems += pm
        recs += [placed, Record(new_id(), "name", person.id, str(r.get("name") or r["label"]).strip(), now, pulse_id)]
    for r in spec.get("things") or []:
        tid, up = ids[r["key"]], idof(str(r["in"]))
        ents.append(Entity(tid, [r["label"]], 0, "genesis"))
        recs.append(Record(new_id(), "kosha", tid, up, now, pulse_id, note="declared"))
        text = str(r.get("tuple") or "").strip()
        if text:
            mems.append(Memory(new_id(), tid, frozenset(), text, "PRIMARY", 5, "nowhere", 0, pulse_id, embed(text), use="none"))
        pages = r.get("pages") or []
        if len(pages) == 1:  # one sheet is no chain (v4.60 reads two or more): its words are its own memory
            mems.append(Memory(new_id(), tid, frozenset(), pages[0], "PRIMARY", 3, "nowhere", 0, pulse_id, embed(pages[0]), use="none"))
            pages = []
        holder = tid
        for i, words in enumerate(pages):
            page = mnemonic_id(f"{r['key']}_page_{i + 1}", new_id())
            ents.append(Entity(page, [f"page {i + 1}"], 0, "genesis"))
            recs.append(Record(new_id(), "kosha", page, holder, now, pulse_id, note="declared"))
            mems.append(Memory(new_id(), page, frozenset(), words, "PRIMARY", 3, "nowhere", 0, pulse_id, embed(words), use="none"))
            holder = page
    b = spec.get("breath") or {}
    if b.get("in") and isinstance(b.get("far_every"), int):
        recs.append(Record(new_id(), "prakriti", idof(str(b["in"])), json.dumps({"far_every": b["far_every"]}), now, pulse_id, note="pacing"))
    return ents, mems, recs, ids


def summary(spec: dict) -> list[str]:
    """The proposal as a reviewing god reads it."""
    out = [f"{spec.get('title', '(untitled)')} -- {spec.get('why', '')}"]
    for r in spec.get("places") or []:
        out.append(f"  place  {r.get('label')}{' (' + r['name'] + ')' if r.get('name') else ''} -- {'scene' if r.get('scene') else 'region'} in {r.get('in')}")
    for r in spec.get("people") or []:
        out.append(f"  person {r.get('name')}, {r.get('label')} -- in {r.get('in')}; {len(r.get('tuples') or [])} first memories")
    for r in spec.get("things") or []:
        pages = len(r.get("pages") or [])
        out.append(f"  thing  {r.get('label')} -- in {r.get('in')}{f'; {pages} pages' if pages else ''}")
    b = spec.get("breath") or {}
    if b:
        out.append(f"  breath {b.get('in')}: its people act every {b.get('far_every')} turn(s) of the world with no player near")
    return out


def texts_of(memories: list[Memory]) -> list[str]:
    """The planted words a probe looks for in a prompt: anything long enough to be its own."""
    return [m.experience for m in memories if len(m.experience) >= 40]



def probe(canon, memories: list[Memory], records: list[Record], ents: list[Entity], mems: list[Memory], recs: list[Record],
          players: list[str], ids: dict[str, str], spec: dict) -> dict:
    """The proof, read-only, before anything is planted (the book, 2026-10-02: a copy caught four failures by name and
    window that would have reached play). On a fork of the world with the seed in it, by the engine's own retrieval:
    leaks -- planted words in a player's prompt on an idle action, by player; stands -- every person in a scene, every
    place under a place the world already holds; reads -- each written thing read whole by name by its holder, and
    silent when not. Cost: one retrieval a player, two a written thing."""
    from smritilila.smrti import narrator as N
    from smritilila.smrti.pulse import _context_for
    from smritilila.smrti.samskara import Samskara
    work = canon.fork()
    for e in ents:
        work.entities[e.id] = e
    now = max([canon.world_time] + [m.world_time for m in mems]) + 1
    sk = Samskara.build(work, memories + mems, records + recs, players=players, clock=None, defaults=None)
    planted = texts_of(mems)

    def prompt(actor: str, action: str) -> str:
        log = sk.log_for(work)
        ctx = _context_for(work, log, actor, action, now, None, keep=frozenset({actor}))
        return N.render(ctx, action, set(log.places), N.handle_map(ctx, {i: e.labels for i, e in work.entities.items()}))

    out: dict = {"leaks": {}, "stands": [], "reads": {}}
    for p in players:
        if p in work.entities and sk.log.stream(p, before=now):
            user = prompt(p, "I look around.")
            leaked = [t[:80] for t in planted if t in user]
            out["leaks"][p] = leaked
    k = sk.log_for(work).kosha
    for r in spec.get("people") or []:
        pid = ids[r["key"]]
        if k.scene_of(pid, now) is None:
            out["stands"].append(f"{r.get('name') or r['key']} stands in no scene")
    new = set(ids.values())
    for r in spec.get("places") or []:
        c, seen = ids[r["key"]], set()
        while c in new and c not in seen:
            seen.add(c)
            c = k.container_at(c, now)
        if c is None or c in new:
            out["stands"].append(f"{r['label']} is under nothing the world holds")
    for r in spec.get("things") or []:
        pages = r.get("pages") or []
        if len(pages) < 2:
            continue  # no chain: a single sheet's words are its own memory, shown as anything held is
        holder = k.container_at(ids[r["key"]], now)
        if holder not in new or holder not in [ids[x["key"]] for x in spec.get("people") or []]:
            out["reads"][r["label"]] = "not held by a person of the seed -- not probed"
            continue
        whole = prompt(holder, f"I read the {r['label']} from beginning to end.")
        idle = prompt(holder, "I wait and keep watch.")
        out["reads"][r["label"]] = {"pages": len(pages), "shown when read": sum(1 for w in pages if w in whole),
                                    "shown when idle": sum(1 for w in pages if w in idle)}
    return out
