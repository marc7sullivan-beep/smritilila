# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Bīja (बीज, seed): the Narrator writes the seed of a world; the engine
plants it (smrti/genesis.py). v5 §39, built 2026-09-20.

A world begins as tuples. What a world IS -- its sky, its roads, its
coin, its people -- is saguṇa, with qualities, and lives here: in the
text the Narrator writes from a player's self-description and a
setting. The engine is nirguṇa (v5 §1): it plants places, people and their
first tuples and reads none of it. Authored seeds (a world file written
by hand, worlds/vadhu_crossing.json) and generated seeds (this module)
have one shape and one planter.

Breadth is how far the seed reaches beyond what the description names:
  here    -- only the scene the description puts the player in
  around  -- what a reader of that setting would expect within a day's
             walk: the neighbouring places, the figures who belong there
  wide    -- the region: its towns, its powers, the named people of the
             age -- each asleep until someone walks in
A bigger seed costs one main call at birth (a seed may be retried once,
extended or regrouped, and the world at large is a call of its own) and
nothing at runtime: retrieval is bounded (v5 §26), dormancy sleeps the
untouched (v5 §38), near/far keeps a large cast from starving the player's
scene.

v4 (Design Principles v5 §9-§14; rewritten 2026-09-23): there are
no places, only what is in what. A seed states its tree -- each place says
what it is `in` and whether it is a scene (v5 §11) -- so no world is born
flat, as every v3 world was (v5 Appendix B: the crossing's 22 scenes had no
neighbours). Regions are places that are not scenes (v5 §14). Roads are no
longer required: arrivals are asked of quiet scenes (v5 §38), and a road is a
place like any other.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from smritilila.llm.client import LLMClient
from smritilila.text import tokens, weight
from smritilila.smrti.kosha import begins_with_name, whole_name
from smritilila.smrti.narrator import DEFAULT_VIEW, own_view, use_of
from smritilila.sakshi import log_event

Breadth = Literal["here", "around", "wide"]

SYSTEM = """You write the SEED of a world for Smritilila, a persistent generative world whose engine keeps memory and knows nothing else. Everything a world is -- its weather, its roads, its money, its gods, its manners -- exists only as text you write now and text written later by play. Write the seed as if the world had always been there.

A seed is: a NAME for the world; a SETTING (a paragraph: where and when, what the days and seasons are like, what this world takes for granted); PLACES; PEOPLE; and which person the player is.

PLACES are what people and things can be IN, and they nest: a room in a house, the house in a farmstead, the farmstead in a valley, the valley in a county. Each place has a label, what it is IN (the key of another place in the seed, or empty for nothing above it), and one bit, SCENE:
  scene: true -- everyone inside it, and nobody outside it, would share what happens there: a room, a hall, a kitchen, a church, a yard, a square, a landing, a clearing, a stretch of road where travellers pass one another.
  scene: false -- it is too large to share, and holds scenes: a farm, a village, a town, a valley, a district, a county. These are the REGIONS; put each scene in the region that holds it.
A house can be a scene with rooms that are scenes inside it; the rooms are then next to one another, and someone in one may catch what happens in the next. So put places that are in sight or earshot of each other INSIDE ONE COMMON SCENE -- the forge and the well inside the square they face, the rooms inside the house, both ends of a bridge inside the bridge -- and places that are not (two farms a mile apart) inside a region. The buildings of one holding open onto the same yard, field or court: put them INSIDE that yard as scenes of their own, so the hall, the byre and the yard are next to one another -- a farm, a village or a town is then the region around it. What is in what is structure; the engine reads nothing else about a place. `in` is what a place is physically inside: a village is never in the road that leads to it; a road is in the region it crosses.
Each place has one impersonal first tuple: what an onlooker with no interior would see there, and -- in prose, by name -- what it is near and where its ways lead, so that anyone reading it could walk from it to the places around it. A region's tuple is a line or two: what the land is and who holds it.

PEOPLE are the living who stand somewhere now. Each has a label (a name if they have one, else what they are: "the smith's widow"; a name is whole, two words at least -- a given name and a family name, or a byname: "Anna Tolland", "Hal of the Weir" -- never one word alone), the key of the SCENE they stand in, and two or three first tuples in their OWN voice, first person, past tense: who they are, what they carry, what is unfinished in their life. These tuples are their memory; nothing else about them exists. Give each a WEIGHT: indelible, marking, weighty, notable, ordinary; and a USE -- what it gives them to act on: key (it locates something, names something, opens a way, or changes how they will read what follows), useful (a fact they may need again), or none (mood, routine). The dead, the absent and the far-off are not people in the seed: they live in the tuples of those who remember them, with no place and no stream of their own -- unless this world is one where the dead walk, and the setting says so.

The PLAYER is one of the people, written from the self-description given: keep every fact the description states, invent nothing that contradicts it, and fill the rest as the world would. The OPENING shows the player where they stand: only people standing in that same scene appear in it -- a person the seed puts elsewhere is elsewhere, and the opening may not show them at the player's elbow (the engine files people where they stand; a man the opening shows who is not there will not remember the scene).

BREADTH tells you how far to reach, and SIZE gives the floors -- a seed under its floors is short, and the world will be poorer for it (floors count scenes, not regions):
  here   -- only what the self-description puts in the scene: the player, the scene they stand in, whoever is with them, and what that scene is in.
  around -- the player's own holding AND the OTHER holdings and settlements within a day's walk -- the neighbours, the nearest market, the seat of whoever holds this land -- each with its scenes and the people who belong in them, all inside the region that holds them. Not more corners of the player's own holding: other places.
  wide   -- the region beyond a day's walk: its towns, its seats of power and worship, the roads and waters between, and the people of the age, each in its own region. Where the setting is historical or a known fiction, the REAL named people and seats belong here -- and it is the NEAREST ones that belong, not the most famous: the see this district answers to, the chieftain whose thingmen these are, the market these farms actually walk to, the lord this valley owes, the feud of that year in this quarter. A northern fjord answers to the northern bishop; the Shire's nearest seat is Bree, not Minas Tirith. Accurate to the record where it speaks -- who held which seat in which year, which road led where -- and invented only where it is silent. Each in their own place, with first tuples in their own voice.

THE WORLD'S OWN VIEW. The engine keeps memory and knows nothing of what a world is; what a world is taken to be is text. The DEFAULT VIEW (given below the self-description) holds unless you say otherwise. Write `view` ONLY where this world differs from it -- plainly, as fact of this world: what perceives, what the dead are, whether minds are private, whether places remember, how often strangers come, how rare it is for a being to become something else. Most worlds differ in little or nothing: leave `view` empty rather than invent. BELIEF IS NOT PHYSICS: what the people of this world believe -- that the dead walk, that the saints hear, that the hidden folk live in the rocks -- belongs in their own tuples, and becomes `view` only where the setting makes it so. A historical or realistic setting keeps the default: its people believe much, and their world is still the default one.
Where one PLACE's physics differs from the rest -- the marsh where the drowned walk, the grove where the dead can speak -- give that place a `clause`: it holds for everything inside it and nowhere else. Write a law where it HOLDS: a law of a whole island, valley or district goes on that region, once -- everything in the region, and everything later found or built there, stands under it -- never copied onto each of its scenes; a scene gets a clause of its own only where it differs from the region around it. Never put a whole world's difference on one place, or one place's on the world.
A place may differ in its NATURE as well as its physics. Give these only where the setting makes it so, on the region they hold for -- once, never copied onto its scenes: `time_rate` where time itself runs differently inside it than in what holds it, as world hours per hour lived inside (2: an hour inside is two outside; 0.25: four hours inside pass while one passes outside) -- the hill where a night costs a year (a night of 8 hours, a year of 8,760: `time_rate` 1095), the hall where a feast lasts an age; work the number out from what the setting says, however large; `no_strangers` where no stranger comes in from beyond it -- a hidden valley, a warded house; `rooms_remember` where the places inside it themselves perceive and remember; `sight_unbounded` where perception inside it is not bounded by where one stands. Most places have none of these: leave `time_rate` 0 and the rest false. A whole world's nature is the three switches below, never a place's. A `clause` only tells the Narrator; these fields are what the world itself keeps -- where a place's clause says time runs differently there, that no stranger comes, that its rooms remember, or that perception there is unbounded, set the field on that same place as well.
BECOMINGS: what beings in this world may become that the setting anticipates -- the sight, the bite, a curse, an initiation -- each a short key and one or two sentences of what then holds for the one who becomes it. Empty if the setting anticipates none; becoming is always possible, this is only what the world expects.
Three switches, false unless this world says otherwise: `places_perceive` (places themselves perceive and remember), `perception_reaches` (perception is not bounded by where one stands -- telepathy, a god who sees everything), `closed` (nothing and no one comes into this world from beyond it).
WHEN IT OPENS: `hour`, the hour of the day the world opens (0-23), as its first moment has it; and `date`, that day as YYYY-MM-DD ONLY where the world is dated in our own calendar -- a historical setting, or a fiction set in a year of ours -- else empty. The engine keeps the world's time from them.

Every `in` and every person's `place` is the key of a place you have defined under `places` -- define it first. Keys are short snake_case mnemonics used only inside the seed. Never explain; never address the player; write the world."""

BHUVANA_SYSTEM = """You write the WORLD AT LARGE for Smritilila -- bhuvana: the whole of a world as a literate person living in it would know it, before any place in it is built. It is read by whoever later writes the world's places and people, so that what they add lies where it should and answers to whom it should. It is never read by the engine, which keeps memory and knows nothing else.

Write it as prose in sections, plainly, as fact:
  THE YEAR -- when this is, and the state of things: what has lately happened, what is feared or hoped, who holds power and who contests it.
  THE REGIONS -- every region of the world that matters, each with: its name; where it lies relative to the others (north of, across the river from, a fortnight's ride east); its character (land, people, manners, faith, trade); its seats -- the towns, halls, courts and holy places that hold it; and who rules or leads it.
  THE CENTRAL FIGURES -- the people of the age a person in this world would have heard of: name, station, where they are this year, what they want. They are heard of here, not met; they become people in the world only when a place is built where they stand.
  THE WAYS -- the great roads, rivers and sea-routes between the regions, and the distances between seats in days of travel.
  THE STARTING REGION -- which region the player's self-description places them in, and what a person there would know of the rest.

Write it in the SOURCE's own register. A world is known as its own people know it: what they speak of openly, and what they lower their voices for, avoid, or will not say. For a history, the state of the age; for a fiction, the fiction's own preoccupations and mood -- in a tale of dread, the rumours, the shunned places and the unease under the ordinary are as much the world at large as who holds office. The public events of the age belong only as the source's own people would weigh them.

Where the setting is historical or a known fiction, be accurate to the record where it speaks: which seat was held by whom in this year, which road led where. Invent only where the record is silent, and in its idiom. Never explain; never address the player; write the world. 600 to 1200 words."""


# A world born from a brief (Marc, 2026-09-24): the new world screen has no
# character creation -- a world is made by its creator from history, a book
# or a film, and everyone who plays in it arrives later as a newcomer. Its
# prompts are the player seed's, with the player swapped for where the
# world BEGINS; _swap fails loudly if SYSTEM is edited under it.
def _swap(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        assert text.count(old) == 1, f"bija: prompt text moved under a brief swap: {old[:60]!r}"
        text = text.replace(old, new)
    return text


BRIEF_SYSTEM = _swap(SYSTEM, [
    ("PEOPLE; and which person the player is.", "PEOPLE; and the scene where it BEGINS."),
    (SYSTEM[SYSTEM.index("The PLAYER is one of the people"):SYSTEM.index("\n\nBREADTH tells you")],
     "No one plays in this world yet: it is made by its creator from a BRIEF, and everyone who plays in it will arrive later as a newcomer. It BEGINS in one scene -- the one the brief names, or, where the brief leaves it, the scene where a story in this world most likely opens: where the book or film opens its own story; for history, the place this year turns on. Newcomers arrive there and in the busiest scenes near it, so build the beginning in the most detail, with people standing in it. Every person in the seed belongs to the world."),
    ("only what the self-description puts in the scene: the player, the scene they stand in, whoever is with them, and what that scene is in.",
     "only the scene where the world begins, whoever is in it, and what that scene is in."),
    ("the player's own holding AND the OTHER", "the holding where the world begins AND the OTHER"),
    ("Not more corners of the player's own holding: other places.", "Not more corners of that holding: other places."),
    ("(given below the self-description)", "(given below the brief)"),
])

BHUVANA_BRIEF_SYSTEM = _swap(BHUVANA_SYSTEM, [
    ("THE STARTING REGION -- which region the player's self-description places them in,",
     "THE STARTING REGION -- the region where the world begins (the brief's WHERE IT BEGINS, or, where it is not given, where a story in this world most likely opens),"),
])

# The new world screen's boxes, in order: (field, heading). Only the source
# and where and when are required (and the world's name); a blank box is the
# builder's to fill.
BRIEF_FIELDS: tuple[tuple[str, str], ...] = (
    ("source", "SOURCE"),
    ("where_when", "WHERE AND WHEN"),
    ("begins", "WHERE IT BEGINS"),
    ("different", "WHAT IS DIFFERENT HERE"),
    ("feel", "WHAT PLAY SHOULD FEEL LIKE"),
)


def brief_text(name: str, brief: dict[str, str]) -> str:
    """The creator's brief as the seed and the world at large read it."""
    lines = ["THE CREATOR'S BRIEF. This world is from history, a book or a film: keep every fact the creator states; "
             "be accurate to the record or the source where it speaks, and invent only where it is silent, in its idiom.",
             f"WORLD NAME: {name.strip()}"]
    for field, heading in BRIEF_FIELDS:
        given = (brief.get(field) or "").strip()
        if given:
            lines.append(f"{heading}: {given}")
        elif field == "source":
            continue  # required by the screen; a brief without one says nothing rather than guess
        elif field == "begins":
            lines.append(f"{heading}: not given -- choose the scene where a story in this world most likely opens")
        else:
            lines.append(f"{heading}: not given -- as the record or the source has it")
    return "\n".join(lines)


# Its sections, all of which a finished world at large has (Ravenmoor's
# first, 2026-09-24, stopped mid-sentence in THE STARTING REGION at a
# 4096-token cap that thinking shares, and nothing noticed).
BHUVANA_SECTIONS = ("THE YEAR", "THE REGIONS", "THE CENTRAL FIGURES", "THE WAYS", "THE STARTING REGION")


def plain(text: str) -> str:
    """Markdown emphasis off (**THE YEAR**, *Kingsport.*): the world at large
    is prose shown as text on the page, where the marks show as marks."""
    return re.sub(r"(\*{1,2}|_{2})(?=\S)(.+?)(?<=\S)\1", r"\2", text)


def seeded_block(spec: dict) -> str:
    """A seed already written, as the world at large written AFTER it reads it (Marc, 2026-10-05: every world has
    its wider account; below wide breadth it follows the seed instead of preceding it, because a seed built from
    an account at those breadths came out flatter and smaller). Its setting, where it begins, its places by their
    labels and what holds them, its people by name: names and facts the account must agree with."""
    places = spec.get("places") or {}
    label = lambda k: (places.get(k) or {}).get("label") or k  # noqa: E731
    lines = ["THE WORLD AS ALREADY SEEDED (built before this account, and never to be contradicted: every name and fact below "
             "stands; the account is the wider world around it -- where these places lie within it, who holds its regions, "
             "what lies beyond. The people below are in the world already, not figures heard of)."]
    if (spec.get("setting") or "").strip():
        lines.append(f"SETTING AS SEEDED: {spec['setting'].strip()}")
    begins = [label(k) for k in spec.get("entry_points") or []]
    if begins:
        lines.append("IT BEGINS AT: " + "; ".join(begins))
    if places:
        lines.append("PLACES SEEDED: " + "; ".join(
            label(k) + (f" (in {label(v['in'])})" if v.get("in") in places else "") for k, v in places.items()))
    people = [v.get("label") or k for k, v in (spec.get("people") or {}).items()]
    if people:
        lines.append("PEOPLE SEEDED: " + ", ".join(people))
    return "\n".join(lines)


def write_bhuvana(client: LLMClient, self_description: str, setting: str | None, brief: str | None = None,
                  source: str | None = None, darsana: str | None = None, seeded: str | None = None) -> str:
    """One call at birth: the world at large, fixed thereafter (Marc,
    2026-09-20). At wide breadth it comes first and the seed grows within
    it; below wide it follows the seed (`seeded`: seeded_block of it), so
    the seed keeps its own shape (Marc, 2026-10-05). Returned as text; stored on the
    session row; handed to every later seed, expansion and newcomer.
    `brief`: a world born from its creator's brief (brief_text, which
    carries its source), with no player -- the self-description and
    setting are then unused. `source`: the world's source, first (Marc,
    2026-09-24: always up front). Refused unless it finished: every
    section, and a last sentence that ends."""
    if brief:
        system, user = BHUVANA_BRIEF_SYSTEM, brief + (f"\n\n{seeded}" if seeded else "")
    else:
        system, user = BHUVANA_SYSTEM, "\n".join([
            *([f"THIS WORLD'S SOURCE: {source.strip()}"] if source and source.strip() else []),
            f"SETTING (the player's own words, keep every fact): {setting.strip()}" if setting and setting.strip() else "SETTING: not given -- take it from the self-description",
            f"SELF-DESCRIPTION OF THE PLAYER: {self_description.strip()}",
            *_views(darsana, default=False),
            *([seeded] if seeded else []),
        ])
    text = client.complete(system, user, max_tokens=16000)
    text = plain(text or "").strip()
    if len(text) < 200:
        raise ValueError("the world at large came back empty")
    missing = [s for s in BHUVANA_SECTIONS if s not in text]
    if missing or not text.rstrip("*_ ").endswith((".", "!", "?", '"', "'", ")", "”")):
        raise ValueError(f"the world at large came back unfinished (missing {missing or 'its last sentence'})")
    log_event("bija.bhuvana", words=len(text.split()))
    return text


DESCRIBE_SYSTEM = (
    "You are shown the root of a world: the world at large, from which everything in it grows. "
    "Write ONE line, at most 20 words, that tells someone about to enter this world what they are entering -- "
    "where and when, and what it feels like. Plain words, present tense. No title, no quotation marks, nothing else. "
    "If the root also gives the age's newest entries, say where and when the world has reached NOW, not when it began, and name no one.")


def describe_root(client: LLMClient, name: str, root: str) -> str:
    """A world's one line, drawn from its root (Marc, 2026-09-30: "each world
    should have a one line description drawn from its root ... so people have
    a good idea of what world they are entering").
    Its name: sūtra (सूत्र), a thread, and an aphorism (v5 §42).
    One call when the root is
    laid, never per visit. Refused unless it came back as one line."""
    # room for a model's thinking as well as the line (the sample world's came back cut at 200 tokens)
    text = plain(client.complete(DESCRIBE_SYSTEM, f"The world: {name}\n\n{root.strip()}", max_tokens=4000) or "")
    line = " ".join(text.split()).strip(' "\'“”')
    if not line or len(line) > 240:
        raise ValueError("the world's line came back empty or too long")
    if not line.endswith((".", "!", "?")):
        raise ValueError(f"the world's line came back unfinished: {line!r}")
    return line


def _at_large(bhuvana: str | None) -> list[str]:
    return [f"THE WORLD AT LARGE (fixed; place what you add within it, and answer to its seats and roads):\n{bhuvana.strip()}"] if bhuvana and bhuvana.strip() else []


# floors: (places, places beyond the player's own holding, people)
FLOORS: dict[str, tuple[int, int, int]] = {"here": (1, 0, 1), "around": (10, 6, 8), "wide": (20, 14, 16)}

_EXTENDING = """

You are EXTENDING a seed that came in under its floors. You are shown what it already holds. Add ONLY new places and new people -- none already listed, no repeats under other names -- reaching outward from what is there: further holdings, the seats and roads of the region, the people of the age. The same rules for places, people and keys apply; a new place may be IN a listed one (use its key), and people may stand in the scenes already listed or in the ones you add."""
EXTEND_SYSTEM = SYSTEM + _EXTENDING
BRIEF_EXTEND_SYSTEM = BRIEF_SYSTEM + _EXTENDING

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "the world's name"},
        "setting": {"type": "string", "description": "a paragraph: where and when; the days, the seasons, what this world takes for granted"},
        "opening": {"type": "string", "description": "80-150 words for the player to read first: second person, present tense, where they stand and what is around them, from their own first tuples. Only people whose `place` is the player's place may be seen or heard in it; everyone else may be thought of, never present"},
        "places": {"type": "array", "items": {"type": "object", "properties": {
            "key": {"type": "string"},
            "label": {"type": "string"},
            "name": {"type": "string", "description": "what it is called here: a proper name, or the phrase people use for it; empty if it is only glimpsed"},
            "in": {"type": "string", "description": "the key of the place this one is physically inside, or empty for nothing above it"},
            "clause": {"type": "string", "description": "only where this place's physics differs from the world's: what holds inside it; else empty"},
            "time_rate": {"type": "number", "description": "only where time runs differently inside this place than in what holds it: world hours per hour lived inside; else 0"},
            "no_strangers": {"type": "boolean", "description": "true only where no stranger comes into this place from beyond it"},
            "rooms_remember": {"type": "boolean", "description": "true only where the places inside this one perceive and remember"},
            "sight_unbounded": {"type": "boolean", "description": "true only where perception inside this place is not bounded by where one stands"},
            "scene": {"type": "boolean", "description": "true if everyone inside it and nobody outside it would share what happens there; false for a region that holds scenes"},
            "tuple": {"type": "string", "description": "the impersonal first tuple: what is there, what it is near, where its ways lead"},
        }, "required": ["key", "label", "in", "scene", "tuple"]}},
        "people": {"type": "array", "items": {"type": "object", "properties": {
            "key": {"type": "string"},
            "label": {"type": "string"},
            "name": {"type": "string", "description": "what they are called: a whole proper name, two words at least (given and family name, or a byname), or the phrase people use for them"},
            "place": {"type": "string", "description": "the key of the scene they stand in"},
            "tuples": {"type": "array", "items": {"type": "object", "properties": {
                "text": {"type": "string"}, "weight": {"type": "string", "enum": ["indelible", "marking", "weighty", "notable", "ordinary"]},
                "use": {"type": "string", "enum": ["key", "useful", "none"], "description": "what it gives them to act on"},
            }, "required": ["text", "weight", "use"]}},
        }, "required": ["key", "label", "place", "tuples"]}},
        "player": {"type": "string", "description": "the key of the person the player is"},
        "view": {"type": "string", "description": "where this world differs from the DEFAULT VIEW, as fact of this world; empty if it does not"},
        "becomings": {"type": "array", "items": {"type": "object", "properties": {"key": {"type": "string"}, "text": {"type": "string"}},
                                                  "required": ["key", "text"]}},
        "places_perceive": {"type": "boolean"}, "perception_reaches": {"type": "boolean"}, "closed": {"type": "boolean"},
        "date": {"type": "string", "description": "the day the world opens, YYYY-MM-DD, ONLY if this world is dated in our own calendar (a historical setting, or a fiction set in a year of ours); else empty"},
        "hour": {"type": "integer", "description": "the hour of that day the world opens, 0-23, as the opening has it"},
    },
    "required": ["name", "setting", "opening", "places", "people", "player", "date", "hour"],
}

# A brief's seed: no player and no opening -- where the world begins instead.
BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: v for k, v in SCHEMA["properties"].items() if k not in ("opening", "player")},
        "begins": {"type": "string", "description": "the key of the SCENE where the world begins"},
    },
    "required": ["name", "setting", "places", "people", "begins", "date", "hour"],
}

ARRIVAL_SYSTEM = """You write one NEWCOMER into an existing world for Smritilila, from a self-description. You are shown the place they arrive at -- its first tuple (what it is) and its latest (what is happening there now) -- and nothing else of the world. Write the newcomer as one who remembers, whatever they are: a label (a name if the description gives one, else what they are), two or three first tuples in their OWN voice, first person, past tense -- who they are, what they carry, what is unfinished -- keeping every fact the description states; each with a weight (indelible, marking, weighty, notable, ordinary) and a USE -- what it gives them to act on: key (it locates something, names something, opens a way, or changes how they will read what follows), useful (a fact they may need again), or none (mood, routine). The last first tuple is their arrival at this place. Then one impersonal tuple for the PLACE: the newcomer arriving, as an onlooker would see it. And an opening: 80-150 words for the newcomer's player to read, second person, present tense. Never explain; never address the player except in the opening."""

ARRIVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "name": {"type": "string", "description": "what they are called: a proper name, or the phrase people use for them"},
        "tuples": SCHEMA["properties"]["people"]["items"]["properties"]["tuples"],
        "place_tuple": {"type": "string", "description": "the place's impersonal tuple of the arrival"},
        "opening": {"type": "string"},
    },
    "required": ["label", "tuples", "place_tuple", "opening"],
}

_KEY = re.compile(r"[\W]+")  # v5 §7: letters of any script; was [a-z0-9], which made every non-Latin key "x"


def _key(s: str) -> str:
    return _KEY.sub("_", str(s).strip().lower()).strip("_") or "x"


def _rows(items, required: tuple[str, ...]) -> list[dict]:
    good = []
    for r in items or []:
        if isinstance(r, dict) and all(isinstance(r.get(k), (str, list)) and r.get(k) for k in required):
            good.append(r)
    return good


def _tuples(items) -> list[dict]:
    out = []
    for t in items or []:
        if isinstance(t, str) and weight(t) >= 24:
            out.append({"text": t.strip(), "weight": "notable"})
        elif isinstance(t, dict) and isinstance(t.get("text"), str) and weight(t["text"]) >= 24:
            out.append({"text": t["text"].strip(), "weight": str(t.get("weight") or "notable"), "use": use_of(t.get("use"))})
    return out


def _place(p: dict) -> dict:
    """One place row as the planter takes it: v4's scene bit and `in`. A
    row with no scene bit (v3's `kind`, an authored world file) is a scene,
    as every v3 place was (v5 Appendix B) -- its kind is not kept (the nirguna audit,
    2026-10-03: the engine takes no word for what kind of place it is)."""
    row = {"label": p["label"].strip(), "tuple": p["tuple"].strip()}
    if str(p.get("name") or "").strip():
        row["name"] = str(p["name"]).strip()
    row["scene"] = bool(p["scene"]) if "scene" in p else True
    if str(p.get("in") or "").strip():
        row["in"] = _key(str(p["in"]))
    if str(p.get("clause") or "").strip():
        row["clause"] = str(p["clause"]).strip()  # v5 §7: this place's physics, holding within it
    nature = {}  # v5 §15: this place's nature, holding within it -- only what the seed says differs
    rate = p.get("time_rate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0 and rate != 1:
        nature["rate"] = float(rate)
    for field, flag in (("no_strangers", "closed"), ("rooms_remember", "perceives"), ("sight_unbounded", "reaches")):
        if p.get(field) is True:
            nature[flag] = True
    if nature:
        row["prakriti"] = nature
    return row


def _view(out: dict) -> dict:
    """The world's own view as the seed wrote it (v5 §7-§8): the darśana text
    (None: the default holds), the library of becomings, and world data."""
    becomings = {}
    for b in out.get("becomings") or []:
        if isinstance(b, dict) and str(b.get("key") or "").strip() and str(b.get("text") or "").strip():
            becomings[_key(b["key"])] = str(b["text"]).strip()
    view = str(out.get("view") or "").strip()
    return {"darsana": view or None, "becomings": becomings,
            "world_data": {"capabilities": {"perceives": bool(out.get("places_perceive")), "reaches": bool(out.get("perception_reaches"))},
                           "pacing": {"arrivals": not bool(out.get("closed"))}, "clock": _clock(out),
                           "sandhi": True}}  # born seeing across what joins two scenes (Marc, 2026-10-01: sandhi on in all worlds)


def _clock(out: dict) -> dict:
    """The world's clock from when the seed says it opens (2026-09-30: a world
    was born keeping no time -- the live ones had theirs set by hand). A day
    of our calendar dates it; anything else, or nothing, counts its days.
    The hour it opens, else the morning."""
    from datetime import date
    from smritilila.smrti.kala import CONFIRM_MINUTES
    hour = out.get("hour")
    start = (hour if isinstance(hour, int) and not isinstance(hour, bool) and 0 <= hour <= 23 else 8) * 60
    try:
        day = date.fromisoformat(str(out.get("date") or "").strip()).isoformat()
    except ValueError:
        return {"calendar": "days", "start": start, "confirm_minutes": CONFIRM_MINUTES}
    return {"epoch": f"{day}T00:00", "start": start, "confirm_minutes": CONFIRM_MINUTES}


def _prune_tree(places: dict[str, dict], known: set[str] = frozenset()) -> None:
    """`in` must name a place the seed or the world holds, and no chain
    may loop; a bad `in` is dropped (the place stands in nothing) and
    logged -- structure only, the Narrator's judgement is not second-guessed."""
    for k, row in places.items():
        target = row.get("in")
        if target and target not in places and target not in known:
            log_event("bija.in_unknown", key=k, into=target)
            row.pop("in")
    for k in list(places):
        seen, c = {k}, places[k].get("in")
        while c and c in places:
            if c in seen:
                log_event("bija.in_cycle", key=k)
                places[k].pop("in", None)
                break
            seen.add(c)
            c = places[c].get("in")


def _person(q: dict) -> tuple[str, dict]:
    """A person row as a seed, its extension and an expansion all keep it:
    (key, {label, place, tuples, name?}). Whether it is kept is each
    caller's own rule."""
    row = {"label": q["label"].strip(), "place": _key(q["place"]), "tuples": _tuples(q["tuples"])}
    if str(q.get("name") or "").strip():
        row["name"] = str(q["name"]).strip()
    return _key(q["key"]), row


def as_spec(out: dict, begins: bool = False) -> dict:
    """The Narrator's seed as the world-file shape genesis.build plants:
    keys normalised, every `in` naming a place the seed holds and no loop,
    every person standing in a place that exists, the player among the
    people. Raises ValueError when the seed cannot be planted.

    begins=True: a brief's seed (BRIEF_SCHEMA) -- no player; where the
    world begins is its one entry point. A `begins` that is not a scene
    the seed holds falls back, by structure, to the scene with the most
    people standing in it."""
    places: dict[str, dict] = {}
    for p in _rows(out.get("places"), ("key", "label", "tuple")):
        places[_key(p["key"])] = _place(p)
    if not places:
        raise ValueError("the seed names no place")
    _prune_tree(places)
    people: dict[str, dict] = {}
    for q in _rows(out.get("people"), ("key", "label", "place", "tuples")):
        k, row = _person(q)
        if row["place"] not in places or not row["tuples"] or k in people:  # a second under one key: the first stands
            log_event("bija.person_dropped", key=k, place=row["place"], tuples=len(row["tuples"]), duplicate=k in people)
            continue
        if not whole_name(row.get("name") or row["label"]):  # counted, kept: a seed's cast is its world
            log_event("bija.name_not_whole", key=k, label=row["label"], name=row.get("name"), seed=True)
        people[k] = row
    if begins:
        start = _key(out.get("begins", ""))
        if start not in places or not places[start].get("scene", True):
            scenes = [k for k, p in places.items() if p.get("scene", True)] or list(places)
            crowd = Counter(q["place"] for q in people.values())
            chose = max(scenes, key=lambda k: crowd.get(k, 0))  # first of the most crowded
            log_event("bija.begins_fallback", asked=start, chose=chose)
            start = chose
        return {
            "name": str(out.get("name") or "a world").strip(),
            "setting": str(out.get("setting") or "").strip(),
            "opening": "",
            "scene_owner": "",
            "playable": [],
            "entry_points": [start],
            "places": places,
            "people": people,
            "view": _view(out),
        }
    player = _key(out.get("player", ""))
    if player not in people:
        raise ValueError(f"the seed's player {player!r} is not among its people")
    return {
        "name": str(out.get("name") or "a world").strip(),
        "setting": str(out.get("setting") or "").strip(),
        "opening": str(out.get("opening") or "").strip(),
        "scene_owner": player,
        "playable": [player],
        "entry_points": [people[player]["place"]],
        "places": places,
        "people": people,
        "view": _view(out),
    }


def _kind_of(p: dict) -> str:
    return "scene" if p.get("scene", True) else "region"  # no scene bit: a v3 row, a scene (v5 Appendix B)


def _size_line(breadth: Breadth, holding: str = "the player's own holding") -> str:
    places, beyond, people = FLOORS[breadth]
    if breadth == "here":
        return "SIZE: the scene only -- at most five places, the regions that hold them included."
    return (f"SIZE: at least {places} scenes, of which at least {beyond} are beyond {holding}, "
            f"and at least {people} people, each with two or three first tuples. These are floors, not targets.")


def _short(spec: dict, breadth: Breadth) -> tuple[int, int]:
    """How far under its floors a seed is: (scenes short, people short).
    Regions do not count toward the floor: they hold places, they are not ones."""
    places, _, people = FLOORS[breadth]
    scenes = sum(1 for p in spec["places"].values() if p.get("scene", True))
    return max(0, places - scenes), max(0, people - len(spec["people"]))


def write_seed(client: LLMClient, self_description: str, setting: str | None, breadth: Breadth = "here", bhuvana: str | None = None) -> dict:
    """The main call: the seed, as a spec genesis.build can plant (asked once
    more if it cannot be) -- and, when it comes in under its floors, one
    extension call that reaches outward from it (a full retry would redraw
    the world; an extension keeps what was written and adds beyond it); a
    holder left too wide is regrouped in one more. The world at large is a
    call of its own (write_bhuvana)."""
    user = _seed_user(breadth, _size_line(breadth), [
        f"SETTING (the player's own words, keep every fact): {setting.strip()}" if setting and setting.strip() else "SETTING: not given -- take it from the self-description",
        f"SELF-DESCRIPTION OF THE PLAYER: {self_description.strip()}"], bhuvana)
    def ask() -> dict:
        out = client.complete_with_tool(
            SYSTEM, user, tool_name="write_seed", tool_description="The seed of the world: places, people, the player.",
            input_schema=SCHEMA, max_tokens=16384 if breadth == "wide" else 8192,
        )
        return as_spec(out if isinstance(out, dict) else {})
    return _grow(client, _once_more(ask), user, breadth)


def write_world_seed(client: LLMClient, name: str, brief: dict[str, str], breadth: Breadth = "wide", bhuvana: str | None = None) -> dict:
    """write_seed for a world born from its creator's brief, with no player
    (the new world screen, Marc 2026-09-24). The world takes the creator's
    name; the Narrator's is kept only if none was given."""
    user = _seed_user(breadth, _size_line(breadth, "the holding where the world begins"), [brief_text(name, brief)], bhuvana)
    def ask() -> dict:
        out = client.complete_with_tool(
            BRIEF_SYSTEM, user, tool_name="write_seed", tool_description="The seed of the world: places, people, where it begins.",
            input_schema=BRIEF_SCHEMA, max_tokens=16384 if breadth == "wide" else 8192,
        )
        return as_spec(out if isinstance(out, dict) else {}, begins=True)
    spec = _once_more(ask)
    spec["name"] = name.strip() or spec["name"]
    return _grow(client, spec, user, breadth, brief=True, system=BRIEF_EXTEND_SYSTEM, schema=BRIEF_SCHEMA,
                 keep=f"Set `begins` to {spec['entry_points'][0]}; do not rewrite it.")


def _once_more(ask):
    """A seed that came back but cannot be planted is asked for once more
    (Marc, 2026-10-01): a broken tool call -- `places` as text or markup --
    is the model's slip, not the brief's; a second failure is the brief's,
    and the birth fails as it always did."""
    try:
        return ask()
    except ValueError as exc:
        log_event("bija.retry", reason=str(exc)[:200])
        return ask()


def _views(darsana: str | None, default: bool = True) -> list[str]:
    """What a world is taken to be, for whoever writes into a world that
    exists -- the default view and the world's own, which wins -- as the
    Narrator is shown them (2026-09-25 review: an arrival, an expansion and
    the world at large were written without the world's own physics)."""
    return ([DEFAULT_VIEW] if default else []) + ([own_view(darsana)] if darsana and darsana.strip() else [])


def _seed_user(breadth: Breadth, size_line: str, own: list[str], bhuvana: str | None) -> str:
    """The seed's ask: its breadth, the writer's own lines, the world at
    large if there is one, the default view."""
    return "\n".join([f"BREADTH: {breadth}", size_line, *own, *_at_large(bhuvana), DEFAULT_VIEW,
                       "The seed is the STARTING REGION of the world at large: build it in detail, inside a place for that region; the other regions are named in tuples and reached by the ways that lead there, not built now." if bhuvana else "",
                      ])


def _grow(client: LLMClient, spec: dict, user: str, breadth: Breadth, brief: bool = False, **extend) -> dict:
    """A seed under its floors reaches outward once (an extension keeps what
    was written); either way, logged."""
    places_short, people_short = _short(spec, breadth)
    if places_short or people_short:
        log_event("bija.short", breadth=breadth, places=len(spec["places"]), people=len(spec["people"]), places_short=places_short, people_short=people_short)
        spec = _extend(client, spec, user, places_short, people_short, breadth, **extend)
    spec = _regroup(client, spec)  # a holder the seed left too wide, grouped once (Marc, 2026-10-12)
    log_event("bija.seed", name=spec["name"], breadth=breadth, places=len(spec["places"]), people=len(spec["people"]),
              scenes=sum(1 for p in spec["places"].values() if p.get("scene", True)),
              contained=sum(1 for p in spec["places"].values() if p.get("in")),
              **({"brief": True, "begins": spec["entry_points"][0]} if brief else {}))
    return spec


def _extend(client: LLMClient, spec: dict, user: str, places_short: int, people_short: int, breadth: Breadth,
            system: str = EXTEND_SYSTEM, schema: dict = SCHEMA, keep: str | None = None) -> dict:
    have_places = "\n".join(f"  {k}: {p['label']} ({_kind_of(p)}{', in ' + p['in'] if p.get('in') else ''})" for k, p in spec["places"].items())
    have_people = "\n".join(f"  {k}: {q['label']} @ {q['place']}" for k, q in spec["people"].items())
    ask = "\n".join([
        user,
        f"ALREADY IN THE SEED -- places:\n{have_places}",
        f"ALREADY IN THE SEED -- people:\n{have_people}",
        f"ADD at least {max(places_short, 1)} new places and at least {max(people_short, 1)} new people, none of the above. "
        + (keep or f"Set `player` to {spec['scene_owner']}; do not rewrite them."),
    ])
    out = client.complete_with_tool(
        system, ask, tool_name="write_seed", tool_description="Places and people to add to the seed.",
        input_schema=schema, max_tokens=16384,
    )
    out = out if isinstance(out, dict) else {}
    added_places = added_people = 0
    for p in _rows(out.get("places"), ("key", "label", "tuple")):
        k = _key(p["key"])
        if k in spec["places"] or any(p["label"].strip().lower() == q["label"].lower() for q in spec["places"].values()):
            continue
        spec["places"][k] = _place(p)
        added_places += 1
    _prune_tree(spec["places"])
    taken = {str(p.get("name") or p["label"]).strip().casefold() for p in spec["people"].values()}
    for q in _rows(out.get("people"), ("key", "label", "place", "tuples")):
        k, row = _person(q)
        name = str(row.get("name") or row["label"]).strip().casefold()
        if k in spec["people"] or row["place"] not in spec["places"] or not row["tuples"] or name in taken:
            log_event("bija.person_dropped", key=k, place=row["place"], tuples=len(row["tuples"]), extending=True, name_taken=name in taken)
            continue
        spec["people"][k] = row
        taken.add(name)
        added_people += 1
    log_event("bija.extended", breadth=breadth, added_places=added_places, added_people=added_people)
    return spec


# A holder's width is bounded where places are made (Marc, 2026-10-12; pulse.FANOUT_CAP, spec v5.32): the Narrator writes a seed's
# whole tree at once and puts most of a village's places directly in the village, so a seed is asked once to group a holder that
# holds more than the cap. Mint-time only and structure only: the new places are the Narrator's, the engine re-points the listed
# children and checks the rest. Built and not tested (Marc: monitor actual play).
REGROUP_SYSTEM = """You tidy the SEED of a world for Smritilila. Some places in it hold too many places directly, and a world is easier to move through when no place holds more than eight. For each holder you are shown, make new places INSIDE it -- a lane, a wing, a floor, a quarter, a bank, a close, whatever this world calls such a part -- and say which of the holder's places each one holds, so that the holder ends with at most eight places directly in it. Group places that lie near one another or on the same ground. A group holds at least two of the holder's places and only places that are really part of it; a place that fits no group stays in the holder. Each new place has a key (new, short snake_case), a label, a name if it bears one, `in` (the holder's key), a one-line impersonal first tuple (what it is and which places it holds, by name), and `holds` (the keys of the holder's places now inside it). Do not rewrite or move any other place. Never explain."""

REGROUP_SCHEMA = {
    "type": "object",
    "properties": {"groups": {"type": "array", "items": {"type": "object", "properties": {
        "key": {"type": "string"}, "label": {"type": "string"},
        "name": {"type": "string", "description": "what it is called here; empty if it has no name of its own"},
        "in": {"type": "string", "description": "the key of the holder this new place goes inside"},
        "tuple": {"type": "string", "description": "the impersonal first tuple: what it is and which places it holds, by name"},
        "holds": {"type": "array", "items": {"type": "string"}, "description": "the keys of the holder's places that are now inside this one"},
    }, "required": ["key", "label", "in", "tuple", "holds"]}}},
    "required": ["groups"],
}


def crowded_holders(places: dict[str, dict], cap: int | None = None) -> dict[str, list[str]]:
    """{holder key: the keys of the places directly in it} for each holder with more than `cap` of them."""
    from smritilila.smrti.pulse import FANOUT_CAP
    cap = FANOUT_CAP if cap is None else cap
    kids: dict[str, list[str]] = {}
    for k, row in places.items():
        if row.get("in") in places:
            kids.setdefault(row["in"], []).append(k)
    return {h: ks for h, ks in kids.items() if len(ks) > cap}


def _regroup(client: LLMClient, spec: dict) -> dict:
    """One call that groups the holders a seed left too wide (see REGROUP_SYSTEM). Whatever it returns is applied by structure only:
    a group stands inside a holder that was shown, holds only that holder's own places (two at least), and takes a key not already
    used; a place named by two groups goes to the first. Counted, never raised: a seed that stays wide is planted as it is."""
    crowded = crowded_holders(spec["places"])
    if not crowded:
        return spec
    places = spec["places"]
    shown = "\n".join(
        f"HOLDER {h}: {places[h]['label']} -- holds {len(ks)}:\n" + "\n".join(f"  {k}: {places[k]['label']}" for k in ks)
        for h, ks in crowded.items())
    ask = f"THE WORLD: {spec.get('name', '')}. {spec.get('setting', '')}\n\nThese holders hold too many places directly:\n{shown}"
    before = max(len(ks) for ks in crowded.values())
    try:
        out = client.complete_with_tool(REGROUP_SYSTEM, ask, tool_name="regroup", tool_description="New places that group a holder's places.",
                                        input_schema=REGROUP_SCHEMA, max_tokens=8192)
    except Exception as exc:  # noqa: BLE001 -- a slip here leaves a wide seed, never a failed birth
        log_event("bija.regroup_failed", error=str(exc)[:200])
        return spec
    made = moved = 0
    for g in _rows((out or {}).get("groups"), ("key", "label", "in", "tuple", "holds")):
        holder, key = _key(g["in"]), _key(g["key"])
        if holder not in crowded or key in places or not isinstance(g.get("holds"), list):
            continue
        held = [c for c in dict.fromkeys(_key(str(x)) for x in g["holds"]) if c in crowded[holder] and places[c].get("in") == holder]
        if len(held) < 2:
            continue
        row = _place({"label": g["label"], "name": g.get("name"), "tuple": g["tuple"], "in": holder, "scene": False})
        places[key] = row
        for c in held:
            places[c]["in"] = key
        made += 1
        moved += len(held)
    after = max((len(ks) for ks in crowded_holders(places).values()), default=0)
    log_event("bija.regrouped", crowded=len(crowded), widest_before=before, widest_after=after, groups=made, moved=moved)
    return spec


# A join sees who is there (Ravenmoor, 422): the arrival was shown the entry
# place's first and latest tuples only, and Walter Pike's opening called the
# lecture hall empty while Nathaniel (a player, away) and the students stood
# in it -- written into the newcomer's own first tuples, so every later pulse
# sided with it. Replayed at 422-423 (the larger model, chained): the opening saw the
# still man by the lectern 2/3 (0/3 without); with PLAYERS HERE the next
# pulse kept him present 2/3 and never called the hall empty (6/6 before).


def write_arrival(client: LLMClient, self_description: str, place_label: str, place_first: str, place_last: str, bhuvana: str | None = None,
                  present: list[tuple[str, str, bool]] | None = None, source: str | None = None, darsana: str | None = None) -> dict:
    """One newcomer into an existing world, at a place that already has a
    stream: their first tuples, the place's tuple of their arrival, and
    the opening their player reads. `present`: who is there now, by
    structure -- (what they are, what they last did, still) -- still meaning
    a player who is away: present, doing and saying nothing."""
    who = []
    if present:
        who = ["WHO IS THERE NOW (they ARE there when the newcomer arrives; the newcomer sees them and knows none of them by name -- "
               "describe them only as seen; one marked STILL does nothing and says nothing):"]
        who += [f"  - {what}{' (STILL: away, present, silent)' if still else ''} -- lately: {last}" for what, last, still in present]
    user = "\n".join([
        *([f"THIS WORLD'S SOURCE: {source.strip()} -- write the newcomer in its idiom, accurate to it where it speaks"] if source and source.strip() else []),
        f"THE PLACE: {place_label}",
        f"  what it is: {place_first}",
        f"  what is happening there now: {place_last}",
        *who,
        f"SELF-DESCRIPTION OF THE NEWCOMER: {self_description.strip()}",
        *_at_large(bhuvana),
        *_views(darsana),
    ])
    out = client.complete_with_tool(
        ARRIVAL_SYSTEM, user, tool_name="write_arrival", tool_description="One newcomer's first tuples and the place's account of their arrival.",
        input_schema=ARRIVAL_SCHEMA,
    )
    out = out if isinstance(out, dict) else {}
    tuples = _tuples(out.get("tuples"))
    if not tuples or not isinstance(out.get("label"), str) or not out["label"].strip():
        raise ValueError("the arrival names no one")
    return {
        "label": out["label"].strip(), "tuples": tuples, "name": str(out.get("name") or "").strip(),
        "place_tuple": str(out.get("place_tuple") or "").strip(),
        "opening": str(out.get("opening") or "").strip(),
    }


START_SYSTEM = """You place one NEWCOMER into an existing world for Smritilila, from where their player says they start. You are shown the world's places, each as key: label (scene or region, and the place that holds it). If one listed SCENE is the place described -- the same place, not merely one like it or near it -- give its key as `existing`, and nothing else. Otherwise leave `existing` empty and add ONE new scene where the description puts them: its `label` (never one already listed), its `name` if it bears one, `in` -- the key of the listed place that holds it, the nearest that fits (the region it lies in, or a scene it lies within) -- and its `tuple`: the place's impersonal first tuple, what is there, what it is near and where its ways lead, in this world's idiom. Keep every fact the description states about the place. Never explain."""

START_SCHEMA = {
    "type": "object",
    "properties": {
        "existing": {"type": "string", "description": "the key of the listed scene that IS the described place; else empty"},
        "label": {"type": "string", "description": "a new scene's label; empty when existing is given"},
        "name": {"type": "string", "description": "a new scene's proper name, if it bears one"},
        "in": {"type": "string", "description": "the key of the listed place that holds the new scene"},
        "tuple": {"type": "string", "description": "the new scene's impersonal first tuple"},
    },
    "required": ["existing"],
}


def write_start(client: LLMClient, start: str, self_description: str, places: dict[str, tuple[str, str]],
                source: str | None = None, darsana: str | None = None) -> dict:
    """Where one newcomer starts, in their player's words (Marc, 2026-10-01):
    {"existing": key} for a listed scene that is the place, else
    {"new": {label, name, in, tuple}} -- one new scene inside a listed place.
    `places` maps a key to (label, "scene|region, in <key>"). Structure is
    the engine's to check (routes/bija.py's _described_start), never this call's."""
    listing = "\n".join(f"  {k}: {label} ({kind})" for k, (label, kind) in places.items())
    user = "\n".join([
        *([f"THIS WORLD'S SOURCE: {source.strip()} -- a new place belongs to it: accurate where it speaks, in its idiom where it is silent"]
          if source and source.strip() else []),
        f"THE WORLD'S PLACES:\n{listing}",
        f"WHERE THE NEWCOMER STARTS, IN THEIR PLAYER'S WORDS: {start.strip()}",
        f"WHO THE NEWCOMER IS: {self_description.strip()}",
        *_views(darsana),
    ])
    out = client.complete_with_tool(START_SYSTEM, user, tool_name="place_start",
                                    tool_description="Where one newcomer starts: a listed scene, or one new scene.", input_schema=START_SCHEMA)
    out = out if isinstance(out, dict) else {}
    existing = str(out.get("existing") or "").strip()
    if existing:
        return {"existing": existing}
    label, into, tup = (str(out.get(k) or "").strip() for k in ("label", "in", "tuple"))
    if not label or not into or not tup:
        raise ValueError("the start names no place")
    return {"new": {"label": label, "name": str(out.get("name") or "").strip(), "in": into, "tuple": tup}}


# -- Vistāra (विस्तार, expansion): the world grows outward after birth ------

EXPAND_SYSTEM = SYSTEM + """

You are EXPANDING a world that already exists and has been played in. You are shown the places at its outer edge (label; scene or region; what it is in) and the people there. Add ONLY new places and new people -- none already listed, no repeats under other names -- reaching OUTWARD from the world's edges: what lies along the ways that lead out, the next holdings and settlements beyond the farthest ones listed, the NEAREST seats and people of the age not yet present (the next ring out, not the far capital). A new place may be IN a listed place or region (use its key) or in one you add; new regions hold the new scenes that lie together. Every new place's tuple names, in prose, which listed or new places it is near. New people stand in the scenes you add or in listed ones. The same rules for places, people, weights and keys apply. The living only."""

EXPAND_FLOORS = (10, 8)  # places, people per expansion


def expand(client: LLMClient, setting: str | None, places: dict[str, tuple[str, str]], standing: list[tuple[str, str]],
           names_taken: frozenset[str] = frozenset(), labels_taken: frozenset[str] = frozenset(), bhuvana: str | None = None,
           known_keys: frozenset[str] = frozenset(), source: str | None = None, darsana: str | None = None,
           titles: frozenset[str] | None = None) -> dict:
    """One call: new places and people reaching outward from a world that
    exists. `places` maps a key (for the Narrator to stand people in) to
    (label, kind); `standing` is (label, place key): whoever and whatever
    stands at the edge, unsorted -- the engine does not say which are people
    (v5 §1; 2026-10-03: an island was listed under THE PEOPLE). Returns
    {"places": {key: {label, tuple, scene, name?, in?, ...}}, "people": {key: {label, place, tuples, name?}}}
    with only what is new; a person standing nowhere known is dropped.
    `names_taken`: every proper name the world's memory already carries
    (living or dead -- Eyri: the expansion minted "Grim, a traveling
    tinker" beside Halla's drowned husband Grim, and every tuple of hers
    that says his name would have handed the Narrator a living tinker);
    a new person wearing one is refused. `labels_taken`: every label the
    world holds -- shown as a refusal list, not as the ring (Eyri, ring
    seven: shown only ring six, the Narrator minted the farm of Kross
    again, three rings in)."""
    have_places = "\n".join(f"  {k}: {label} ({kind})" for k, (label, kind) in places.items())  # kind: "scene", "region", or "... in <key>"
    have_standing = "\n".join(f"  {label} @ {pk}" for label, pk in standing)
    user = "\n".join([
        *([f"THIS WORLD'S SOURCE: {source.strip()} -- what is added belongs to it: accurate where it speaks, in its idiom where it is silent"]
          if source and source.strip() else []),
        f"SETTING: {setting.strip()}" if setting and setting.strip() else "SETTING: as the places and people show it",
        *_at_large(bhuvana),
        f"THE WORLD'S OUTER EDGE -- the places most lately added (the rest of the world lies inward of these; reach OUTWARD from them):\n{have_places}",
        f"WHO AND WHAT STANDS AT THAT EDGE:\n{have_standing}",
        "NAMES THIS WORLD'S MEMORY ALREADY CARRIES (living or dead) -- give no new person any of these: " + ", ".join(sorted(names_taken)) if names_taken else "",
        "PLACES THE WORLD ALREADY HOLDS, inward of the edge -- do not add these again, though a new place may lie near them: " + "; ".join(sorted(labels_taken)) if labels_taken else "",
        f"ADD at least {EXPAND_FLOORS[0]} new places and at least {EXPAND_FLOORS[1]} new people, none of the above. "
        "Set `name`, `setting`, `opening` and `player` to the single word 'unchanged'.",
        *_views(darsana),
    ])
    out = client.complete_with_tool(
        EXPAND_SYSTEM, user, tool_name="write_seed", tool_description="Places and people to add to the world.",
        input_schema=SCHEMA, max_tokens=16384,
    )
    out = out if isinstance(out, dict) else {}
    known_labels = {label.lower() for label, _ in places.values()} | {label.lower() for label, _ in standing} | {l.lower() for l in labels_taken}
    new_places: dict[str, dict] = {}
    for p in _rows(out.get("places"), ("key", "label", "tuple")):
        k = _key(p["key"])
        if k in places or k in new_places or p["label"].strip().lower() in known_labels:
            continue
        new_places[k] = _place(p)
    _prune_tree(new_places, known=set(places) | set(known_keys))
    new_people: dict[str, dict] = {}
    taken = {n.casefold() for n in names_taken}  # and each name this ring takes, as it takes it
    refused: set[str] = set()  # names refused as taken; whoever is defined by them goes too
    for q in _rows(out.get("people"), ("key", "label", "place", "tuples")):
        k, row = _person(q)
        place, tuples = row["place"], row["tuples"]
        if row["label"].lower() in known_labels or k in new_people or not tuples or (place not in places and place not in new_places):
            log_event("bija.person_dropped", key=k, place=place, tuples=len(tuples), expanding=True)
            continue
        # v5 §16/§17: the person's DECLARED name, matched exactly (casefolded)
        # against the names the world carries -- not the first capitalised
        # word of the label, which was English, and in a surname-first
        # world refused every new Li or Wang
        name = str(q.get("name") or q["label"]).strip()
        if not whole_name(name, titles):  # a name is made whole (Marc, 2026-09-25), by the world's titles (world data)
            log_event("bija.name_not_whole", key=k, label=q["label"].strip(), name=name)
            continue
        # and one that BEGINS with a name the world carries is that person
        # (Marc, 2026-09-25: "Grim of Stakkar" beside Grim; "Unnr of Kross"
        # beside Kross is still Unnr)
        begins = begins_with_name(name, taken)
        if name.casefold() in taken or begins:
            log_event("bija.name_taken", key=k, label=q["label"].strip(), name=[name], begins_with=begins or None)
            refused.update(begins or [name.casefold()])
            continue
        taken.add(name.casefold())
        new_people[k] = row
    if refused:
        # (Eyri, fourth ring: "Grim of Stakkar" refused, "Ranveig, Grim's
        # wife" let through -- a wife of a man the world does not have,
        # whose every tuple would hand the Narrator the drowned Grim)
        for k in [k for k, q in new_people.items() if tokens(q["label"]) & {t for r in refused for t in tokens(r)}]:
            log_event("bija.name_taken", key=k, label=new_people[k]["label"], name=sorted(refused), defined_by_refused=True)
            del new_people[k]
    log_event("bija.expanded", places=len(new_places), people=len(new_people),
              contained=sum(1 for p in new_places.values() if p.get("in")))
    return {"places": new_places, "people": new_people}

