# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Ābhāsa (आभास, "shining forth", an appearance -- the name Marc's to sign off, 2026-11-11): the pictures of a world.

Marc, 2026-11-11: "I want to try adding image generation and presentation. Use the second model flash-lite. Images should be created when a new character or
environment is introduced or a critical moment. Not for every turn." Then: a critical moment is a player's own weight of weighty or heavier (1 in 17
of Stonemarch's tellings); every player gets a portrait; any picture that includes a character is given that portrait; a non-player gets one only
after being referenced three times; a scene with several characters is given their portraits appended into one image as its seed.

A player's portrait is made with the player (Marc, 2026-11-11: "create a portrait based on the physical description and then let the player edit the
portrait through conversation until they accept it as final. Then save it."): a draft from how they wrote that they look, changed by whatever they ask
for, the model given the draft and the request; only the draft they accept is kept, and it is final. Drafts are held in this process (a restart
asks them to begin again), each change is a picture's cost against the day's cap, and a player may ask for MAX_EDITS of them.

What is made, and when, after a player's telling is committed and never inside it (a picture takes 3 to 5 s; the telling is already told):
  * the telling is of a moment if a player's own weight in it is weighty or heavier and they have an accepted portrait (the picture has their
    face in it, and an invented face would be unlike the one they accept): ONE picture of it however many players it holds (a round is one telling;
    Marc: "Yes, one per round"), of the heaviest such tuple, seeded with the portraits of every player told in it and then of whoever that tuple is about;
  * else, if the place it happened in has no picture, an establishing view of the place with no one in it (shared by every visitor, so it carries
    nobody's face);
  * one non-player the telling is about who has now been in a player's tuples in three tellings: a small model decides whether they have a face
    (the engine knows no quality, and three references alone pass 253 of Stonemarch's 326 non-players, keys and darts among them), and if so
    writes their look from those three accounts; a portrait follows. A being judged to have none is kept as a row with no picture, so it is asked once.
    The accounts seldom say how a being looks, and a vague look is painted with the model's habitual face (Marc, 2026-11-11: the Austin bartender looked
    like Charisse), so a few points of the look -- build, complexion, hair, a mark on the face -- are fixed from a hash of the being's id and handed to
    the judge with the world's description (fixed_points); the judge keeps the sex and the age;
A world is off until its entry in worlds.json says "pictures" (true, or {"daily": 40, "style": "...", "moment_weight": 3}); a day's pictures are
capped (40 by default). A painter that is not configured, or one that fails, leaves the world exactly as it was: no telling waits on a picture.

A picture is kept in the world (memory/chitra.py) and is never one of its memories: the Narrator is never shown one, and its prompt is never read
back as if it were the world. Who sees one is structural, and narrow (Marc, 2026-11-12: "Only the self portrait and pics from the current session should be
shown. Portraits of players/npcs present in the scene should be shown", and "the current place pic if there is one"): a player sees their own portrait; the
portraits of those in the scene they stand in, and that place's picture; and what was made this session from what they lived this session -- never a being
they met in an earlier session, and never someone else's private moment. (The first rule, 2026-11-11, was every picture of what their own tuples ever named:
it showed Filonius's portrait to Jakal, who had met him once, a long way from where Jakal stood.)

Cost: constant per telling (the telling's own tuples and a lookup); a call or two on one telling in five of Stonemarch's, the daily cap above
them. The count of references is kept as tellings commit and rebuilt once, from the players' own tuples, when the first telling after a start finds none.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila.adhisthana import WorldSession, store_of, stream_of
from smritilila.ids import new_id
from smritilila.llm import providers
from smritilila.llm.client import Picture
from smritilila.memory.chitra import Chitra
from smritilila.memory.memory import Memory
from smritilila.sakshi import log_event
from smritilila.smrti.genesis import ITIHASA

DEFAULT_DAILY = 40        # pictures a world makes in a day, unless its entry says another number
MOMENT_WEIGHT = 3         # weighty (3) or heavier: Marc, 2026-11-11, "heavier so we get 1 in 17"
REFERENCES = 3            # tellings a non-player is in a player's tuples before they are asked about
ACCOUNT_CHARS = 450       # of each account of a being, and of a moment, a place's tuple, a player's early memory
SEEDS_MAX = 4             # portraits in one seed (the model's own limit on characters it keeps alike is 4)
MAX_WAITING = 6           # tellings' picture work waiting on the one in hand; more are let go
MAX_SIDE = 1280           # a picture is kept no larger than this on its long side (the model's own 1K comes back at 1376 wide, 850 KB)
PLACE_ROWS = 12           # rows from the start of a place's own record that are looked at for a row naming no one, for its picture
TRIES = 3                 # a failed picture is tried again at the next telling, this many times in a process
MAX_EDITS = 12            # changes a player may ask of one draft of their portrait
MIN_LOOK = 8              # characters of physical description a draft needs

STYLE = ("Painterly illustrated art, muted palette, grounded and serious, like the plates of a good book. "
         "No text, lettering, labels, captions or letter-like marks anywhere in the image.")

JUDGE_SYSTEM = (
    "You decide whether a portrait should be made of someone met in a story, and if so you describe how they look. You are given a name, up "
    "to three short accounts that mention it, and sometimes the world's own description.\n"
    "being: true only if it is a person or creature with a face a portrait could be made of -- a named or unnamed individual, a creature, "
    "even a monster. False for an object, a place, a door, a coin, a weapon, a vial, a heap or a group of things.\n"
    "look: if being, two or three plain sentences of how they look. Settle EVERY one of these: sex (as the name or accounts say, else choose), age "
    "(a decade or a number), build, complexion, hair (colour and style, or none), the face (its shape, and one thing a viewer would know it by -- a "
    "scar, freckles, a crooked nose, deep lines, a gap in the teeth), and clothing with anything they carry that marks them. Take whatever the "
    "accounts say and contradict nothing; where they are silent, COMMIT to a definite specific that could be in the world they are in -- never leave "
    "a point open, never write around the sex with 'they', and do not settle on a generic look (dark hair, lean, serious): when nothing decides, "
    "choose the less usual. If FIXED POINTS are given, write each into the look as given. If not a being, an empty string.")
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"being": {"type": "boolean"}, "look": {"type": "string"}},
    "required": ["being", "look"],
}

# Marc, 2026-11-11, on Austin: "The bartender looks like clarrise." The bartender's three accounts said nothing of how they look, so the judge wrote an
# age and an apron and the painter drew its habitual face for that (three repaints of the exact prompt: one face, hers). Independent judge calls fall
# into one archetype too (stout, ruddy, square-jawed, grey in a knot), so asking each to "be specific" is not enough: a few points of every being's
# look are fixed from a hash of its id -- deterministic, the same every time that being is asked about, and the same on every machine. The judge keeps
# the sex and the age (the name and the accounts say them: "newsboy", "she"); the lists are the painter's, not the engine's, and no kind of thing is
# named in them. Replayed on six real non-players of three worlds before it was built (TODO, 2026-11-11).
LOOK_BUILD = ("slight", "lean and wiry", "of middling build", "stocky", "broad-shouldered", "heavyset", "tall and rangy", "short and sturdy",
              "round-bellied", "narrow-framed")
LOOK_COMPLEXION = ("fair", "ruddy", "olive", "tan", "brown", "dark brown", "pale and freckled", "sallow", "weathered bronze")
LOOK_HAIR = ("black, cropped short", "dark brown, tied back", "auburn, to the shoulder", "chestnut, wavy", "fair, thinning", "grey, close-cut",
             "salt-and-pepper, swept back", "light brown, in a braid", "red, curly", "black, long and loose", "sandy and unkempt",
             "white, fine and combed flat", "dark blond, cut straight")
LOOK_FACE = ("a long narrow face", "a round face", "a square jaw", "high cheekbones", "a broad nose", "deep-set eyes", "a heavy brow", "a small chin",
             "freckles", "a crooked nose", "weathered lines", "full cheeks", "a scar through one eyebrow", "a gap between the front teeth", "large ears",
             "a long upper lip")


# -- the world's own state ---------------------------------------------------------------------------------------------------------------

@dataclass
class Draft:
    """A player's portrait, not yet theirs: how they wrote that they look, the versions painted (each a JPEG; the last is the one shown), and what they
    asked for between them (asks[i] made versions[i+1]). Not kept anywhere but here until accepted."""
    look: str
    background: str = ""
    versions: list[bytes] = field(default_factory=list)
    asks: list[str] = field(default_factory=list)
    busy: bool = False
    error: str | None = None
    edits_used: int = 0   # changes made so far, counted even when one is gone back from
    rev: int = 0     # changes whenever what is shown does, so a browser never shows a version from its cache for another


class Refused(Exception):
    """A portrait request that cannot be done, in words for the player."""


class PicturesOff(Exception):
    """This world makes no pictures, or this machine has no painter."""


@dataclass
class Session:
    """A player's stretch of being here (Marc, 2026-11-12: "pics from the current session"): when it began, when it was last asked about, and how many
    tuples of their own they held at its start (the stream is in commit order, so what they have lived since is everything after that)."""
    began: datetime
    asked: datetime
    held: int


@dataclass
class State:
    drafts: dict[str, Draft] = field(default_factory=dict)       # a player -> their portrait, until they accept it
    draft_stamps: list[datetime] = field(default_factory=list)   # when each picture of a draft was made: a day's cap counts them
    index: list[Chitra] = field(default_factory=list)            # every picture the world keeps, without bytes, in the order kept
    of: dict[str, Chitra] = field(default_factory=dict)          # an entity's standing picture: its latest with a face (else its first declined row)
    moments: set[str] = field(default_factory=set)               # the tellings that have their moment's picture: one a telling, however many players it has
    loaded: bool = False
    load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)     # one picture at a time in a world
    waiting: int = 0
    built: bool = False                                          # the reference counts have been read from the history once
    count: dict[str, int] = field(default_factory=dict)          # a being -> tellings that named it in a player's tuple
    last: dict[str, str] = field(default_factory=dict)           # a being -> the last telling counted (a round writes several tuples in one)
    accounts: dict[str, list[str]] = field(default_factory=dict) # a being -> the first of those tellings' tuples
    tried: dict[str, int] = field(default_factory=dict)          # a picture's key -> failed attempts in this process
    portraits: dict[str, bytes] = field(default_factory=dict)    # a portrait's bytes once read, by entity
    sessions: dict[str, Session] = field(default_factory=dict)   # a player -> their current session (runtime: the engine keeps only whether they are online)
    sees: dict[str, tuple] = field(default_factory=dict)         # a player -> ((session began, stream length), entities named or stood in this session, tellings)


def state_of(world: WorldSession) -> State:
    st = getattr(world, "abhasa", None)
    if st is None:
        st = State()
        world.abhasa = st
    return st


async def ensure(world: WorldSession) -> State:
    """The world's pictures, read once from its store."""
    st = state_of(world)
    if st.loaded:
        return st
    async with st.load_lock:
        if not st.loaded:
            st.index = list(await store_of(world).load_chitra_index())
            st.of = {}
            st.moments = {c.pulse_id for c in st.index if c.pulse_id and c.has_picture()}
            for c in st.index:
                _remember_entity(st, c)
            st.loaded = True
    return st


def _remember_entity(st: State, c: Chitra) -> None:
    if c.entity_id and c.pulse_id is None:
        prev = st.of.get(c.entity_id)
        if prev is None or c.has_picture():   # the latest with a face stands; a refusal never takes a face's place (Marc, 2026-11-11: "all of the above")
            st.of[c.entity_id] = c


def settings(world: WorldSession) -> dict | None:
    """This world's picture settings from its entry in worlds.json ("pictures": true, or {"daily", "style", "moment_weight"}), or None: pictures off. A
    registry that cannot be read is no setting: the world makes no pictures, and its telling and its joining go on (met on the first real join, 2026-11-11:
    a file the server could not parse failed the arrival of someone already committed)."""
    try:
        entry = None
        if world.name:
            entry = next((w for w in adhisthana.get_worlds_registry() if w.name == world.name), None)
        if entry is None:
            entry = adhisthana.active_world_registry_entry()
        cfg = getattr(entry, "pictures", None)
        return dict(cfg) if cfg is not None else None
    except Exception as exc:  # noqa: BLE001
        log_event("abhasa.settings_failed", error=repr(exc)[:200])
        return None


_painter = None


def painter_of():
    """The picture maker this machine is configured with, or None (a provider that has one is found once; none found is asked again)."""
    global _painter
    if _painter is None:
        _painter = providers.painter()
    return _painter


# -- the words and the pictures, as pure functions -------------------------------------------------------------------------------------------

def style_of(world: WorldSession, cfg: dict) -> str:
    source = str((world.world_data or {}).get("source") or "").strip()
    return " ".join(x for x in (str(cfg.get("style") or "").strip() or STYLE, f"The world: {source}." if source else "") if x)


def _clip(text: str, n: int = ACCOUNT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def portrait_prompt(style: str, name: str, look: str | None, accounts: list[str], background: str | None = None) -> str:
    asked = f"{style} A character reference portrait, head and shoulders, facing the viewer, on a plain neutral background: {name}."
    if look and look.strip():
        asked += f" How they look, and what they wear: {_clip(look, 700)}"
        if background and background.strip():
            asked += f" Who they are, for context: {_clip(background, 400)}"
    elif accounts:
        asked += " What is known of them, in the words of those who met them: " + " ".join(_clip(a) for a in accounts)
    return asked


PLACE_ASK = ("An establishing view of the place itself, empty of the people the accounts mention: no one in the foreground, no named or described person "
             "and no group doing anything, only the place -- its ground or rooms, its objects, its light and weather -- and, where people would gather "
             "there, at most a few anonymous figures far in the background.")


def place_prompt(style: str, label: str, accounts: list[str]) -> str:
    """A place's view carries nobody's face (it is shared by every visitor, and the painter is given no one's portrait), so it is asked to be the place
    and not the people in its accounts. Replayed on the painter (2026-11-12): told of "no named person in it", it painted the people its accounts
    named, as people of its own invention (4 of 4); told this, 8 of 8 on the cases that went wrong, the original stored accounts among them."""
    asked = f"{style} {PLACE_ASK} {label}."
    return asked + (" What is known of the place (the people in these words are not to be shown): " + " ".join(_clip(a) for a in accounts) if accounts else "")


def sheet_line(names: list[str]) -> str:
    return f"The image above is one reference sheet of portraits, left to right: {', '.join(names)}. "


def moment_prompt(style: str, names: list[str], text: str) -> str:
    keep = (f"Keep each person's face, hair and clothing exactly as in their reference portrait. The people in it: {', '.join(names)}. "
            if names else "")
    return f"{style} Paint this single moment as a scene seen from outside. {keep}The moment, as it was told: {_clip(text, 900)}"


def shrink(data: bytes, mime: str, max_side: int = MAX_SIDE, quality: int = 82) -> tuple[bytes, str]:
    """A picture as it is kept: a JPEG no larger than `max_side` on its long side. One that cannot be read is kept as it came."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=quality, optimize=True)
        return out.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001 -- the model's bytes, whatever they are, are better kept than lost
        return data, mime


def sheet(portraits: list[bytes], height: int = 768) -> Picture:
    """The portraits appended side by side into one image: the seed of a picture with several people in it (Marc, 2026-11-11)."""
    from PIL import Image
    tiles = [Image.open(io.BytesIO(p)).convert("RGB") for p in portraits]
    tiles = [t.resize((max(1, round(t.width * height / t.height)), height)) for t in tiles]
    out = Image.new("RGB", (sum(t.width for t in tiles), height))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=88)
    return Picture(data=buf.getvalue(), mime="image/jpeg", name="reference sheet")


def _pick(eid: str, salt: str, options: tuple[str, ...]) -> str:
    return options[int.from_bytes(hashlib.sha256(f"{salt}:{eid}".encode()).digest()[:4], "big") % len(options)]


def fixed_points(eid: str) -> str:
    """The points of a being's look that are fixed by its id, in the words the judge is handed (see LOOK_BUILD): the same for the same id, always."""
    return ("FIXED POINTS of their look -- write each into the look as given, in your own words, whatever you would otherwise have chosen: "
            f"build {_pick(eid, 'build', LOOK_BUILD)}; complexion {_pick(eid, 'complexion', LOOK_COMPLEXION)}; hair {_pick(eid, 'hair', LOOK_HAIR)}; "
            f"face: {_pick(eid, 'face', LOOK_FACE)}. Only a plain statement in the accounts that says otherwise (they say the person is bald, or stooped) "
            "overrides one, and so does what the world's description rules out (then take the nearest that could be there). You still settle the sex and "
            "the age yourself, from the name and the accounts.")


def judge(client, name: str, accounts: list[str], *, eid: str | None = None, setting: str = "") -> dict:
    """Whether a portrait should be made of `name`, and how they look (see JUDGE_SYSTEM): {"being": bool, "look": str}. One model call. `setting` is the
    world's own description, and `eid` fixes the points of the look that would otherwise fall to the model's habitual face (see LOOK_BUILD)."""
    user = f"NAME: {name}\n\nACCOUNTS:\n" + "\n".join(f"{i}. {_clip(a)}" for i, a in enumerate(accounts, 1))
    if setting and setting.strip():
        user += f"\n\nTHE WORLD: {_clip(setting, 300)}"
    if eid:
        user += "\n\n" + fixed_points(eid)
    out = client.complete_with_tool(JUDGE_SYSTEM, user, tool_name="judge_being", tool_description="Decide whether this has a face, and describe how it looks.",
                                    input_schema=JUDGE_SCHEMA, max_tokens=500)
    return {"being": bool(out.get("being")), "look": str(out.get("look") or "").strip()}


# -- who sees what -----------------------------------------------------------------------------------------------------------------------

def _session(world: WorldSession, st: State, pid: str, back: int = 0) -> Session | None:
    """The player's current session, or None when they are not online (Marc, 2026-11-12: "pics from the current session"). The engine knows only whether a
    player is online (seen within PLAYER_IDLE_SECONDS), not when that began, so it is kept here as it is asked: a player who is not online or has logged off
    has none, and one asked after a gap as long as the idle time begins a new one. `back`: tuples of the telling that is begun from, already in the stream,
    that are part of it (the first moment of a session is made from tuples that were committed before it is asked about)."""
    now = datetime.now(timezone.utc)
    if not world.online(pid) or pid in world.logged_off:
        st.sessions.pop(pid, None)
        return None
    s = st.sessions.get(pid)
    if s is None or (now - s.asked).total_seconds() >= adhisthana.PLAYER_IDLE_SECONDS:
        s = st.sessions[pid] = Session(now, now, max(0, len(stream_of(world, pid)) - back))
    s.asked = now
    return s


def _lived(world: WorldSession, st: State, pid: str, s: Session) -> tuple[frozenset, frozenset]:
    """What the player has lived this session: the entities their own tuples since it began name or stand in, and the tellings they are in. Kept by the
    length of their stream. Cost: the session's tuples, never the history."""
    stream = stream_of(world, pid)
    key = (s.began, len(stream))
    cached = st.sees.get(pid)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    entities, tellings = set(), set()
    for m in stream[s.held:]:
        if m.tier != "PRIMARY":
            continue
        entities.add(m.location_id)
        entities.update(m.subject_ids)
        tellings.add(m.pulse_id)
    st.sees[pid] = (key, frozenset(entities), frozenset(tellings))
    return st.sees[pid][1], st.sees[pid][2]


def _here(world: WorldSession, st: State, pid: str) -> tuple[str | None, frozenset]:
    """The scene the player stands in now, and the pictured beings in it, players and non-players alike (not one out of the world: `absent` is those logged
    off who do not allow the world to act for them, and those held back). Cost: the beings that have a picture, never the world."""
    k = world.kosha()
    t = _tick(world)
    scene = k.scene_of(pid, t)
    if scene is None:
        return None, frozenset()
    return scene, frozenset(e for e in st.of if e != pid and e != scene and e not in k.absent and not _is_scene(world, e) and k.scene_of(e, t) == scene)


def _tick(world: WorldSession) -> int:
    """The tick the world is read at: the one after its latest."""
    sk = world.samskara()
    return (sk.mem.max_tick if sk.mem.max_tick >= 0 else world.canon.world_time) + 1


def _through_portals(world: WorldSession, scene: str | None) -> frozenset[str]:
    """The scenes on the far side of a boundary thing that is open now (v5 §13) from `scene`: what a player standing there can see besides their own scene (Marc, 2026-11-21:
    "pictures from adjacent containers ... shouldn't be shown unless there is a portal (window/door) between them"; 2026-11-24, of the scenes not beside theirs: "Hide
    them"). The engine cannot tell a door from a window and does not try: whatever is open is a portal. Cost: the boundary things of one scene, never the world's."""
    if scene is None:
        return frozenset()
    return frozenset(far for _thing, far in world.kosha().across(scene, _tick(world)))


def _stands_in(world: WorldSession, eid: str) -> str | None:
    """The scene a picture's subject is in now: a place, itself; a being, the scene its container chain reaches (None: in no scene)."""
    k = world.kosha()
    return eid if (_is_scene(world, eid) or k.is_scene(eid)) else k.scene_of(eid, _tick(world))


def _made_since(c: Chitra, s: Session) -> bool:
    return c.made_at is None or c.made_at >= s.began


def visible_to(world: WorldSession, st: State, pid: str) -> list[Chitra]:
    """The pictures `pid` may see, in the order kept (Marc, 2026-11-12: "Only the self portrait and pics from the current session should be shown.
    Portraits of players/npcs present in the scene should be shown", and "also add in the current place pic if there is one"):
      * their own portrait;
      * the portrait of every player and non-player in the scene they stand in now, and the picture of that place;
      * what was made this session from what they lived this session: a picture of an entity their own tuples since it began name or stand in, or of a
        telling they are in.
    So a being met in an earlier session, or in a part of this one a picture was not made of, is not shown unless it is here. And what was lived is shown only while it
    is in their own scene or in one an open portal joins to it: a scene beside theirs with nothing open between the two is not (Marc, 2026-11-21), and nor is any other
    scene they are not in (Marc, 2026-11-24, "Hide them"): a place is seen from where one stands, or through a window or door, or not at all. The pictures of the tellings
    they lived are theirs wherever they stood."""
    scene, beings = _here(world, st, pid)
    s = _session(world, st, pid)
    entities, tellings = _lived(world, st, pid, s) if s is not None else (frozenset(), frozenset())
    reach: frozenset[str] | None = None   # the scenes an open portal joins to theirs, looked up once, and only if a lived picture asks
    out = []
    for c in st.index:
        if not c.has_picture() or superseded(st, c):
            continue
        if c.entity_id is not None:
            shown = c.entity_id == pid or c.entity_id == scene or c.entity_id in beings
            if not shown and s is not None and c.entity_id in entities and _made_since(c, s):
                if reach is None:
                    reach = _through_portals(world, scene)
                there = _stands_in(world, c.entity_id)
                shown = there is not None and (there == scene or there in reach)
        else:
            shown = s is not None and c.pulse_id in tellings   # a telling they lived this session was told after it began, so its picture was made after
        if shown:
            out.append(c)
    return out


def superseded(st: State, c: Chitra) -> bool:
    """An entity's picture that a later one of theirs has replaced (kept, as every picture is, and no longer shown or used as a seed)."""
    standing = st.of.get(c.entity_id) if c.entity_id and c.pulse_id is None else None
    return standing is not None and standing.id != c.id


def caption_of(world: WorldSession, c: Chitra) -> str:
    e = world.canon.entities.get(c.entity_id) if c.entity_id else None
    return e.labels[0] if e is not None and e.labels else ("this moment" if c.pulse_id else "")


def listing(world: WorldSession, st: State, pid: str, after: int = 0) -> list[dict]:
    return [{"id": c.id, "seq": c.seq, "entity_id": c.entity_id, "caption": caption_of(world, c), "moment": c.entity_id is None,
             "place": c.entity_id is not None and _is_scene(world, c.entity_id), "world_time": c.world_time}
            for c in visible_to(world, st, pid) if c.seq > after]


# -- the work ----------------------------------------------------------------------------------------------------------------------------

def _label(world: WorldSession, eid: str) -> str:
    e = world.canon.entities.get(eid)
    return e.labels[0] if e is not None and e.labels else eid


def _is_scene(world: WorldSession, eid: str) -> bool:
    e = world.canon.entities.get(eid)
    return e is not None and e.avarana == "scene"


def _is_being_candidate(world: WorldSession, eid: str) -> bool:
    """Whatever the engine can say before a model is asked: it exists, it is not a player, a place, or the world at large."""
    e = world.canon.entities.get(eid)
    return e is not None and e.avarana != "scene" and e.origin != ITIHASA and eid not in world.playable_entity_ids


def count_refs(world: WorldSession, st: State, perceiver: str, subjects, pulse_id: str, text: str) -> None:
    """A player's tuple names these: each being is counted once for a telling, and the first accounts are kept (they are all a portrait's look is made from)."""
    for s in subjects:
        if s == perceiver or not _is_being_candidate(world, s) or st.last.get(s) == pulse_id:
            continue
        st.last[s] = pulse_id
        st.count[s] = st.count.get(s, 0) + 1
        if len(st.accounts.setdefault(s, [])) < REFERENCES:
            st.accounts[s].append(_clip(text))


def _build_counts(world: WorldSession, st: State) -> None:
    rows: list[Memory] = []
    for pid in world.playable_entity_ids:
        rows += [m for m in stream_of(world, pid) if m.tier == "PRIMARY"]
    for m in sorted(rows, key=lambda m: m.world_time):
        count_refs(world, st, m.entity_id, m.subject_ids, m.pulse_id, m.experience)
    st.built = True


def _spent_today(st: State) -> int:
    """Pictures made in the last day: those kept, and those of drafts (a player's changes are as dear as any picture, kept or not)."""
    since = datetime.now(timezone.utc) - timedelta(days=1)
    st.draft_stamps = [t for t in st.draft_stamps if t >= since]
    return sum(1 for c in st.index if c.has_picture() and (c.made_at is None or c.made_at >= since)) + len(st.draft_stamps)


class Capped(Exception):
    pass


async def _keep(world: WorldSession, st: State, c: Chitra) -> Chitra:
    kept = await store_of(world).save_chitra(c)
    st.index.append(kept)
    _remember_entity(st, kept)
    if kept.pulse_id and kept.has_picture():
        st.moments.add(kept.pulse_id)
    if c.data and c.entity_id and not _is_scene(world, c.entity_id):
        st.portraits[c.entity_id] = c.data  # a portrait is read again for every picture its being is in; a place's view is not
    return kept


async def _paint(world: WorldSession, st: State, cfg: dict, painter, *, entity_id: str | None, pulse_id: str | None, prompt: str, aspect: str,
                 seeds: list[str] = (), seed: Picture | None = None, kind: str) -> Chitra | None:
    """One picture, asked for and kept. None: the model made none (an entity's refusal is kept as a row with no picture, so it is not asked again),
    or the day's cap is spent."""
    if _spent_today(st) >= int(cfg.get("daily") or DEFAULT_DAILY):
        log_event("abhasa.capped", kind=kind, entity_id=entity_id, pulse_id=pulse_id)
        raise Capped()
    made = await asyncio.to_thread(painter.paint, prompt, seed=seed, aspect=aspect)
    if made is None:
        if entity_id is not None:
            await _keep(world, st, Chitra(new_id(), entity_id, None, world.canon.world_time, prompt, painter.model, note="declined by the model"))
        return None
    data, mime = shrink(made.data, made.mime)
    kept = await _keep(world, st, Chitra(new_id(), entity_id, pulse_id, world.canon.world_time, prompt, painter.model, mime, data, None, tuple(seeds)))
    log_event("abhasa.made", kind=kind, entity_id=entity_id, pulse_id=pulse_id, id=kept.id, seq=kept.seq, bytes=kept.size, seeds=len(seeds))
    return kept


async def _portrait_bytes(world: WorldSession, st: State, eid: str) -> bytes | None:
    c = st.of.get(eid)
    if c is None or not c.has_picture():
        return None
    if eid not in st.portraits:
        got = await store_of(world).load_chitra_data(c.id)
        if got is None:
            return None
        st.portraits[eid] = got[1]
    return st.portraits[eid]


async def _seed_of(world: WorldSession, st: State, actor: str, subjects, together=()) -> tuple[Picture | None, list[str]]:
    """The actor's portrait, then those of the players told with them (`together`), then those of whoever the tuple is about, who have one (never a
    place's), appended into one image: the picture's seed and the entities in it, in order."""
    ids = [actor]
    for e in list(sorted(together)) + sorted(subjects):
        if e not in ids and not _is_scene(world, e) and e in world.canon.entities:
            ids.append(e)
    got: list[tuple[str, bytes]] = []
    for eid in ids:
        data = await _portrait_bytes(world, st, eid)
        if data is not None:
            got.append((eid, data))
        if len(got) == SEEDS_MAX:
            break
    if not got:
        return None, []
    return (sheet([d for _, d in got]) if len(got) > 1 else Picture(got[0][1], "image/jpeg", "portrait")), [e for e, _ in got]


# -- a player's portrait: a draft, changed by conversation, kept when accepted -------------------------------------------------------------------

def edit_prompt(style: str, name: str, look: str, asked: str) -> str:
    return (f"{style} The image above is a character reference portrait of {name}, head and shoulders on a plain background. Change it as asked and "
            f"nothing else: the same person, face, framing, pose, lighting and background, except where the change needs them to differ. "
            f"Who they are: {_clip(look, 500)} What is asked: {_clip(asked, 400)}")


def has_portrait(st: State, pid: str) -> bool:
    c = st.of.get(pid)
    return c is not None and c.has_picture()


def _ready(world: WorldSession) -> tuple[dict, object]:
    cfg, painter = settings(world), painter_of()
    if cfg is None or painter is None:
        raise PicturesOff("this world makes no pictures")
    return cfg, painter


def _player_of(world: WorldSession, pid: str) -> None:
    if pid not in world.playable_entity_ids:
        raise Refused("that is not a character a portrait is made for")


async def _paint_draft(world: WorldSession, st: State, cfg: dict, painter, d: Draft, prompt: str, seed: Picture | None) -> None:
    """One picture of a draft: shown if made. A refusal or a failure is said in `d.error` and leaves the version already shown as it was."""
    d.busy, d.error = True, None
    try:
        if _spent_today(st) >= int(cfg.get("daily") or DEFAULT_DAILY):
            log_event("abhasa.capped", kind="draft")
            raise Capped()
        made = await asyncio.to_thread(painter.paint, prompt, seed=seed, aspect="3:4")
        st.draft_stamps.append(datetime.now(timezone.utc))
        if made is None:
            d.error = "The painter would not make that. Try asking for it another way."
            return
        d.versions.append(shrink(made.data, made.mime)[0])
        d.rev += 1
    except Capped:
        d.error = "Today's pictures for this world are used up. Try again tomorrow."
    except Exception as exc:  # noqa: BLE001
        log_event("abhasa.failed", kind="draft", error=repr(exc)[:200])
        d.error = "The painter failed. Try again."
    finally:
        d.busy = False


async def draft_start(world: WorldSession, pid: str, look: str, background: str = "") -> None:
    """A first draft of a player's portrait from how they wrote that they look (a new draft replaces any other). Raises Refused or PicturesOff; a failure
    of the painter is in the draft's `error`."""
    cfg, painter = _ready(world)
    _player_of(world, pid)
    st = await ensure(world)
    if has_portrait(st, pid):
        raise Refused("you have a portrait already: it is final")
    look = " ".join((look or "").split())
    if len(look) < MIN_LOOK:
        raise Refused("describe how you look first: your face, hair, build, what you wear")
    old = st.drafts.get(pid)
    if old is not None and old.busy:
        raise Refused("your portrait is being painted: wait for it")
    d = st.drafts[pid] = Draft(look=look, background=(background or "").strip())
    await _paint_draft(world, st, cfg, painter, d, portrait_prompt(style_of(world, cfg), _label(world, pid), look, [], d.background), None)


async def draft_edit(world: WorldSession, pid: str, asked: str) -> None:
    """The player's request, made of the portrait shown: the model is given that picture and the request (Marc, 2026-11-11: "edit the portrait through
    conversation"). At most MAX_EDITS."""
    cfg, painter = _ready(world)
    _player_of(world, pid)
    st = await ensure(world)
    d = st.drafts.get(pid)
    if d is None or not d.versions:
        raise Refused("there is no portrait yet: describe how you look and have one painted")
    asked = " ".join((asked or "").split())
    if not asked:
        raise Refused("say what to change")
    if d.busy:
        raise Refused("your portrait is being painted: wait for it")
    if d.edits_used >= MAX_EDITS:
        raise Refused(f"that is as many changes as one portrait is given ({MAX_EDITS}): accept this one, or go back a step and ask differently")
    before = len(d.versions)
    await _paint_draft(world, st, cfg, painter, d, edit_prompt(style_of(world, cfg), _label(world, pid), d.look, asked), Picture(d.versions[-1], "image/jpeg", "portrait"))
    if len(d.versions) > before:
        d.asks.append(asked)
        d.edits_used += 1


def draft_back(world: WorldSession, pid: str) -> None:
    """Go back one version: the last change is undone, and may be asked for differently (it still counts toward the changes allowed)."""
    _ready(world)
    st = state_of(world)
    d = st.drafts.get(pid)
    if d is None or len(d.versions) < 2 or d.busy:
        raise Refused("there is nothing to go back to")
    d.versions.pop()
    if d.asks:
        d.asks.pop()   # the conversation shown is the one that made the picture shown
    d.error = None
    d.rev += 1


async def draft_accept(world: WorldSession, pid: str) -> Chitra:
    """The portrait shown is theirs, and final: kept in the world (Marc, 2026-11-11: "until they accept it as final. Then save it.")."""
    cfg, painter = _ready(world)
    _player_of(world, pid)
    st = await ensure(world)
    d = st.drafts.get(pid)
    if d is None or not d.versions:
        raise Refused("there is no portrait to keep")
    if d.busy:
        raise Refused("your portrait is being painted: wait for it")
    if has_portrait(st, pid):
        raise Refused("you have a portrait already: it is final")
    prompt = d.look + ((" | changes asked: " + "; ".join(d.asks)) if d.asks else "")
    kept = await _keep(world, st, Chitra(new_id(), pid, None, world.canon.world_time, prompt, painter.model, "image/jpeg", d.versions[-1], f"accepted by the player after {len(d.asks)} changes"))
    st.drafts.pop(pid, None)
    log_event("abhasa.accepted", entity_id=pid, id=kept.id, changes=len(d.asks), bytes=kept.size)
    return kept


def draft_state(world: WorldSession, pid: str) -> dict:
    """What the page needs to show a player's portrait: whether the world makes them, whether theirs is final, and the draft if there is one."""
    cfg, painter = settings(world), painter_of()
    if cfg is None or painter is None or pid not in world.playable_entity_ids:
        return {"enabled": False}
    st = state_of(world)
    if has_portrait(st, pid):
        return {"enabled": True, "accepted": True}
    d = st.drafts.get(pid)
    return {"enabled": True, "accepted": False, "busy": bool(d and d.busy), "version": len(d.versions) if d else 0, "rev": d.rev if d else 0,
            "asks": list(d.asks) if d else [], "look": d.look if d else "", "error": d.error if d else None,
            "edits_left": MAX_EDITS - d.edits_used if d else MAX_EDITS}


def draft_image(world: WorldSession, pid: str) -> bytes | None:
    d = state_of(world).drafts.get(pid)
    return d.versions[-1] if d is not None and d.versions else None


async def _being_portrait(world, st, cfg, painter, writer, eid: str) -> None:
    key = "portrait:" + eid
    if st.tried.get(key, 0) >= TRIES or eid in st.of:
        return
    name, accounts = _label(world, eid), list(st.accounts.get(eid) or [])
    try:
        verdict = await asyncio.to_thread(judge, writer, name, accounts, eid=eid, setting=str((world.world_data or {}).get("source") or ""))
        if not verdict["being"] or not verdict["look"]:
            await _keep(world, st, Chitra(new_id(), eid, None, world.canon.world_time, "", "", note="judged to have no face"))
            log_event("abhasa.no_face", entity_id=eid, name=name)
            return
        await _paint(world, st, cfg, painter, entity_id=eid, pulse_id=None, kind="portrait",
                     prompt=portrait_prompt(style_of(world, cfg), name, verdict["look"], accounts), aspect="3:4")
    except Capped:
        raise
    except Exception as exc:  # noqa: BLE001
        st.tried[key] = st.tried.get(key, 0) + 1
        log_event("abhasa.failed", kind="being", entity_id=eid, error=repr(exc)[:200], tries=st.tried[key])


async def _moment_picture(world, st, cfg, painter, told: dict[str, list[Memory]], actor: str | None, pulse_id: str) -> set[str]:
    """The ONE picture of a telling's moment (Marc, 2026-11-11, of a round: "Yes, one per round"), however many players it holds. A telling has a
    moment if any player's own weight in it is heavy enough and that player has an accepted portrait: the picture is of the heaviest such tuple
    (the actor's, on a tie), seeded with the portraits of every player told in it who has one, then of whoever that tuple is about. A heavy player with no
    portrait has no moment of their own made (the face would be invented): it is counted, and their place may still be pictured. Returns the players the
    moment is of, whose places are not pictured besides."""
    limit = int(cfg.get("moment_weight") or MOMENT_WEIGHT)
    heavy = {pid: min(ts, key=lambda m: m.salience) for pid, ts in told.items() if min(m.salience for m in ts) <= limit}
    for pid in sorted(heavy):
        if not has_portrait(st, pid):
            log_event("abhasa.moment_waits_for_a_face", pulse_id=pulse_id, entity_id=pid)
    faced = {pid: m for pid, m in heavy.items() if has_portrait(st, pid)}
    if not faced:
        return set()
    key = "moment:" + pulse_id
    if pulse_id in st.moments or st.tried.get(key, 0) >= TRIES:
        return set(faced)
    owner = min(faced, key=lambda p: (faced[p].salience, p != actor, p))
    heaviest = faced[owner]
    try:
        seed, seeded = await _seed_of(world, st, owner, heaviest.subject_ids, together=told)
        names = [_label(world, e) for e in seeded]
        prompt = (sheet_line(names) if len(seeded) > 1 else (f"The image above is the reference portrait of {names[0]}. " if seeded else "")) \
            + moment_prompt(style_of(world, cfg), names, heaviest.experience)
        await _paint(world, st, cfg, painter, entity_id=None, pulse_id=pulse_id, kind="moment", prompt=prompt, aspect="16:9", seeds=seeded, seed=seed)
    except Capped:
        raise
    except Exception as exc:  # noqa: BLE001
        st.tried[key] = st.tried.get(key, 0) + 1
        log_event("abhasa.failed", kind="moment", pulse_id=pulse_id, error=repr(exc)[:200], tries=st.tried[key])
    return set(faced)


def _place_accounts(world: WorldSession, st: State, place: str) -> list[str]:
    """What a place's picture is told of it (Marc, 2026-11-12: "Check the latest felonious scene pic. It got filonious quite wrong"): the first row of the
    place's own record, among the first PLACE_ROWS, that names no one whose face is held (a player, or a being with a portrait), and nothing if none does
    (the painter then has the place's name alone). The view carries nobody's face and is shared by every visitor, but a row that names a player
    ("Filonius Applegate asks Wat Barris...") was painted as that player, with a face the painter had to invent: the one face the world already has is
    the one that came out wrong (Stonemarch's hostelry, Austin's apartment: 4 of 4 draws on the stored prompts). It was the first and the latest row, and
    the telling's own text when the place had none; the latest row is the telling's own account of the people in it. The names are those whose faces are
    held, not every perceiver: a carried bike perceives, and its name is "electric bike" (Austin's apartment's first row, which then was skipped as naming
    someone). A name is compared exactly, never interpreted (kosha.names_in). Cost: at most PLACE_ROWS rows, once for a place."""
    k = world.kosha()
    who = set(world.playable_entity_ids) | {e for e, c in st.of.items() if c.has_picture()}
    for m in stream_of(world, place)[:PLACE_ROWS]:
        if not k.names_in(m.experience, who):
            return [_clip(m.experience)]
    return []


async def _place_picture(world, st, cfg, painter, tuples: list[Memory], scene: str | None, pulse_id: str) -> None:
    """The place a player's telling happened in, pictured if it has none yet (shared by every visitor, so with no one in it)."""
    heaviest = min(tuples, key=lambda m: m.salience)
    place = scene or heaviest.location_id
    if not place or not _is_scene(world, place) or place in st.of:
        return
    key = "place:" + place
    if st.tried.get(key, 0) >= TRIES:
        return
    try:
        await _paint(world, st, cfg, painter, entity_id=place, pulse_id=None, kind="place",
                     prompt=place_prompt(style_of(world, cfg), _label(world, place), _place_accounts(world, st, place)), aspect="16:9")
    except Capped:
        raise
    except Exception as exc:  # noqa: BLE001
        st.tried[key] = st.tried.get(key, 0) + 1
        log_event("abhasa.failed", kind="place", entity_id=place, error=repr(exc)[:200], tries=st.tried[key])


async def _illustrate(world: WorldSession, roles, cfg: dict, painter, told: dict[str, list[Memory]], scenes: dict[str, str | None], pulse_id: str) -> None:
    st = state_of(world)
    for pid, tuples in told.items():
        _session(world, st, pid, back=len(tuples))   # whoever is told this and is online is in a session, begun before this telling's own tuples
    if st.waiting >= MAX_WAITING:
        log_event("abhasa.dropped", pulse_id=pulse_id, waiting=st.waiting)
        return
    st.waiting += 1
    try:
        await ensure(world)
        async with st.lock:
            if not st.built:
                _build_counts(world, st)  # reads this telling too: it is already in the history
            else:
                for pid, tuples in told.items():
                    for m in tuples:
                        count_refs(world, st, pid, m.subject_ids, m.pulse_id, m.experience)
            try:
                covered = await _moment_picture(world, st, cfg, painter, told, next(iter(scenes), None), pulse_id)
                for pid, tuples in told.items():
                    if pid not in covered:
                        await _place_picture(world, st, cfg, painter, tuples, scenes.get(pid), pulse_id)
                due = sorted((e for e, n in st.count.items() if n >= REFERENCES and e not in st.of and e in world.canon.entities
                              and any(e in m.subject_ids for ms in told.values() for m in ms)), key=lambda e: (-st.count[e], e))
                if due and roles is not None and getattr(roles, "smrti_client", None) is not None:
                    from smritilila.udbhava import seeder  # a the second model-first writer, as the seeds are (lazily: udbhava calls this module)
                    await _being_portrait(world, st, cfg, painter, seeder(roles.smrti_client), due[0])
            except Capped:
                pass
    except Exception as exc:  # noqa: BLE001 -- nothing here may reach the telling it follows
        log_event("abhasa.failed", kind="telling", pulse_id=pulse_id, error=repr(exc)[:200])
    finally:
        st.waiting -= 1


_tasks: set = set()


def _spawn(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _tasks.add(task)  # held until done: a bare task may be collected mid-flight
    task.add_done_callback(_tasks.discard)


def after_telling(world: WorldSession, roles, memories: list[Memory], scene: str | None, actor: str, pulse_id: str) -> None:
    """A player's telling, committed: its pictures, made in the background (the first thing this does is to find out whether the world asks for any)."""
    try:
        cfg = settings(world)
        painter = painter_of() if cfg is not None else None
        if cfg is None or painter is None:
            return
        told: dict[str, list[Memory]] = {}
        for m in memories:
            if m.tier == "PRIMARY" and m.entity_id in world.playable_entity_ids:
                told.setdefault(m.entity_id, []).append(m)
        if told:
            _spawn(_illustrate(world, roles, cfg, painter, told, {actor: scene}, pulse_id))
    except Exception as exc:  # noqa: BLE001 -- nothing here may reach the telling it follows
        log_event("abhasa.failed", kind="after_telling", pulse_id=pulse_id, error=repr(exc)[:200])


def after_join(world: WorldSession, newcomer: str, look: str, background: str) -> None:
    """A newcomer, committed: a first draft of their portrait from how they wrote that they look (Marc, 2026-11-11: two boxes, appearance and clothing,
    background), painted in the background for them to change and accept (draft_edit, draft_accept). With nothing written of how they look, none is
    painted: the page asks them to describe themselves."""
    try:
        if settings(world) is None or painter_of() is None or len((look or "").strip()) < MIN_LOOK:
            return
    except Exception as exc:  # noqa: BLE001 -- nothing here may reach the arrival it follows
        log_event("abhasa.failed", kind="after_join", entity_id=newcomer, error=repr(exc)[:200])
        return

    async def run() -> None:
        try:
            await draft_start(world, newcomer, look, background)
        except (Refused, PicturesOff):
            pass
        except Exception as exc:  # noqa: BLE001 -- nothing here may reach the arrival it follows
            log_event("abhasa.failed", kind="join", entity_id=newcomer, error=repr(exc)[:200])
    _spawn(run())
