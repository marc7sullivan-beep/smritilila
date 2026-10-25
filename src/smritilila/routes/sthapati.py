# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Sthapati (स्थपति), the master builder (Marc, 2026-10-06): the conversation in which a world's brief is made.

In the building tradition the sthapati is the architect who works out the plan with the one who commissions the building,
before a stone is laid. Here the owner talks with an AI builder in the new-world screen; the builder asks, proposes, and
fills in the brief beside the conversation, which the owner may edit at any time; a preview shows what the world will open
with; and only then does the page ask for it to be built (POST /builder/build: the world's own database, port and server, seeded
there by /genesis, as the Building section below says). The conversation lives OUTSIDE the
world, as a god does: it produces the brief and nothing else, and the engine is not changed by it.

Owner only, or a registered user the owner has enabled for it (2026-10-06); a guest builds no world otherwise (2026-10-05). Stateless: the page holds the conversation and sends it each turn, bounded
(the last forty messages, twenty-four thousand characters), so a turn costs what a short one does however long the talk has run.
The finished conversation is kept with the world (builder_sessions/, never committed, like the witness's record)."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from smritilila import adhisthana, dvara, users_registry
from smritilila.llm.client import ImagesNotSupported
from smritilila.paths import data_dir
from smritilila.adhisthana import app
from smritilila.sakshi import log_event

router = APIRouter()

FIELDS = ("name", "source", "where_when", "begins", "different", "feel", "breadth")   # 2026-10-23: `different` and `feel` replaced days, granted and who
BREADTHS = ("here", "around", "small", "medium", "large")   # 2026-10-23: the engine's "wide" is labelled small here, and medium and large are added
SEED_BREADTH = {"here": "here", "around": "around", "small": "wide", "medium": "wide", "large": "wide"}   # the engine's own breadth for the seed (v5 §39)
RINGS = {"medium": 3, "large": 7}   # outward rings (POST /world/expand) after the seed. One ring is about 12 to 17 places and 7 to 11 people (the expansions on record),
                                    # and a small seed 25 to 30 places and 17 to 20 people: so about 70 places, and about 120, the size of the map that built Stonemarch out (116 to 132)
EXPAND_SECONDS = 900                # one ring is one seed call
MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 4000
MAX_TRANSCRIPT_CHARS = 24000
MAX_FIELD_CHARS = 2000
SESSIONS_DIR = data_dir() / "builder_sessions"
MAX_MAPS = 3                      # maps shown to the builder in one message, and carried to a build
MAX_SAVED_MAPS = 200
UPLOAD_FOLDER = "uploaded"        # where an image browsed for on the owner's own machine is kept: maps/uploaded/
MAX_UPLOADED = 50                 # files in it, so a page cannot fill the disk
MAX_MAP_BYTES = 25 * 1024 * 1024
MAP_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


class BuilderMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class BuilderDraft(BaseModel):
    name: str = ""
    source: str = ""
    where_when: str = ""
    begins: str = ""
    different: str = ""
    feel: str = ""
    breadth: str = ""


class BuilderTurnRequest(BaseModel):
    messages: list[BuilderMessage]
    draft: BuilderDraft = BuilderDraft()
    look: list[str] = Field(default=[], max_length=MAX_MAPS)   # saved maps (their ids) shown to the builder with this message


class SavedMap(BaseModel):
    id: str
    folder: str
    name: str
    bytes: int


class BuilderTurnResponse(BaseModel):
    message: str
    patch: dict[str, str] = {}
    ready: bool = False


class BuilderPreviewRequest(BaseModel):
    draft: BuilderDraft


class BuilderPreviewResponse(BaseModel):
    line: str = ""
    opening: str = ""
    places: list[str] = []
    people: list[str] = []
    cautions: list[str] = []


TURN_SYSTEM = """You are the builder of worlds for Smritilila, a persistent world whose engine keeps memory and knows nothing else: everything a world is -- its weather, its roads, its money, its gods, its manners -- exists only as text written once at its birth and then by play. You are talking with the person who is making a new world (its owner). Your work is the BRIEF: the few fields a seed writer reads to build the world. You do not write the world. You help the owner say what they mean, and you write it into the brief as you go. The brief is shown beside this conversation and the owner can edit it at any time.

THE FIELDS (put each into the patch as soon as you are sure of it, even partly; the owner sees it appear):
  name -- the world's name.
  source -- History, a book, a film, the present day, or an original invention, and which: "History: the Arizona Territory, 1881"; "Book: Treasure Island, R. L. Stevenson"; "Present day: Austin, Texas, as it is now"; "Original: ...". It is the first thing the Narrator reads on every turn of the world, so make it exact and short.
  where_when -- a real place and year, or a book's or film's world and the moment in it.
  begins -- the scene the first arrivals find: a house, a ship, a road, a square. Blank is allowed: the seed writer picks where a story here most likely opens.
  different -- what is different here, from the source or from ordinary life: an alternative history, magic, a technology that does not exist, what the dead are, whether anyone can read another's mind or perceive beyond where they stand, whether anything can come from outside the world, how time runs. A faithful world has nothing different: say "nothing: as the source has it". The engine knows no quality: whatever the owner wants true, say it plainly as a fact of this world.
  feel -- what play should feel like: its tone (grim, comic, quiet, strange), its danger and pace, and how hard it is: forgiving, fair or deadly (whether people die, how often fortune turns against them). Where the owner is unsure, offer two or three concrete choices and say what each would mean for play.
  breadth -- how much to build now: "here" (only where it begins), "around" (a day's walk of places and people), "small" (writes the world at large first, then the region where it begins: about 25 places and 17 people, several minutes), "medium" (small, and three rings outward: about 70 places), "large" (small, and seven rings outward: about 120 places, the size of the map that built Stonemarch out; many minutes and many model calls). A larger world costs more and is written from less of the owner's own words, so suggest large only where the owner wants a wide country to explore, or has a map. Suggest one and say why in a few words; the owner decides.

HOW TO TALK:
  Ask AT MOST TWO questions in a message, the two that would most change the world, and never ask for what is optional (begins, different, feel) once you have enough to set something sensible: set it, say what you chose, and let the owner correct it. Where the owner is unsure, offer two or three concrete choices and say what each would mean for the world. Do not ask about what they have already said. When they say "you decide", decide and say what you chose. Use their own words in the brief where they speak well.
  Be brief: under eighty words a message, plain, no headings, no lecturing, no praise for its own sake. The brief is shown beside you, so do not restate it; say only what you changed and what you need. Say a plain thing when you see a problem: a living person's name that would be seeded as a character, a fact about a real source that you are recalling and are not sure of, two fields that contradict, a breadth that does not fit.
  Never invent facts about a real place, period or source and present them as sure; say when you are recalling.
  Set ready=true only when name, source and where_when are set, at least one of begins, different or feel says something, and you have no open question that would change the world. Then tell the owner they can preview or build.

A person you name in the brief has a whole name (a given name and a family name, or a byname), never a single word.

A MAP: the owner may show you a map, a picture sent with their message. Read its title, legend and period, and set the name, source, where and when, or where it begins only where the map settles them; say in a sentence or two what it shows and what it leaves open. Do not list its places: the map is read in full, place by place, when the world is built.

AN EXAMPLE of a good brief, for a present-day place (Austin, built 2026-10-05): name "Austin"; source "Present day: Austin, Texas, as it is now"; where_when "Austin, Texas, present day: Monday, 5 October 2026"; begins "Zilker Park and Barton Springs Pool, on a warm early-October afternoon"; different "Nothing: ordinary present-day reality, the physics and technology of 2026 -- phones, cars, the internet, the city as it is. Nothing supernatural."; feel "A tired lull between festival weekends: quiet, low danger, forgiving."; breadth "around".

Reply through the tool: message (what you say to the owner), patch (only the fields you are setting or changing), ready."""

PREVIEW_SYSTEM = """You show the owner of a new world what it will open with, from its brief, before anything is built -- so they can judge it and change the brief if it is wrong. Be accurate to the record where the source speaks (history, a book, a film, the present day); invent only where it is silent, in its idiom.

Write: line -- one line, at most twenty words, present tense, telling someone about to enter what they are entering (where and when, and what it feels like); opening -- one paragraph of about a hundred words: the state of things at the moment the world begins; places -- about six named places that would be in it, each a few words, nested where they naturally nest; people -- about six named people, each with a station and what they want (a name is whole; real people only where the record places them there); cautions -- anything the owner should decide or check before building: the name of a living person who would be seeded as a character, a fact about the source that you are not sure of, a contradiction between fields, a breadth that does not fit, something the brief leaves open that would change the world. Cautions concern the BRIEF and its source, never your own preview (do not count your own places or people). Empty if there is none. Never explain or address the owner; write the preview."""

TURN_TOOL = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "what you say to the owner"},
        "patch": {"type": "object", "description": "only the brief fields you are setting or changing",
                  "properties": {**{f: {"type": "string"} for f in FIELDS if f != "breadth"}, "breadth": {"type": "string", "enum": list(BREADTHS)}}},
        "ready": {"type": "boolean", "description": "true when the brief can be previewed and built"},
    },
    "required": ["message"],
}

PREVIEW_TOOL = {
    "type": "object",
    "properties": {
        "line": {"type": "string"},
        "opening": {"type": "string"},
        "places": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["line", "opening"],
}


def _builder_client():
    """The model that builds: the best one, not the rotation (low volume, owner only). the first provider's primary, or SMRITILILA_BUILDER_MODEL;
    with no the first provider key, a local model if one is configured, else the second model (smritilila.llm.providers.best). Tests replace this."""
    from smritilila.llm.client import best_available_client
    try:
        return best_available_client(os.environ.get("SMRITILILA_BUILDER_MODEL") or None)
    except (KeyError, ValueError):
        from smritilila.llm import providers
        raise HTTPException(503, "no model is configured for the builder: " + providers.nothing_configured().split(": ", 1)[1])


def _owner_only(request: Request, what: str) -> str | None:
    """Who may use the builder: the owner, or a registered user the owner has enabled for it (2026-10-06). The gate already refuses
    anyone else; this is the endpoint's own word, so it holds if the gate's list is ever edited. Returns the user's email, or None for
    an owner."""
    guest = dvara._guest(request)
    if guest is not None and not users_registry.may_build(guest):
        log_event("guest.refused", guest=guest, method="POST", path=request.url.path, reason="builder")
        raise HTTPException(403, f"only the owner, or a user the owner has enabled, {what}")
    return guest


def _clip(text: object, n: int) -> str:
    return text.strip()[:n] if isinstance(text, str) else ""


def clean_draft(draft: BuilderDraft | dict) -> dict[str, str]:
    d = draft.model_dump() if isinstance(draft, BuilderDraft) else dict(draft)
    out = {f: _clip(d.get(f), MAX_FIELD_CHARS) for f in FIELDS}
    if out["breadth"] == "wide":   # the engine's word, and an older page's: now small
        out["breadth"] = "small"
    if out["breadth"] not in BREADTHS:
        out["breadth"] = ""
    return out


def clean_patch(patch: object, draft: dict[str, str]) -> dict[str, str]:
    """What the builder may set: the brief's fields only, as text, clipped; a breadth that is one of the three; and only what differs
    from what the owner already has."""
    if not isinstance(patch, dict):
        return {}
    out = {}
    for f in FIELDS:
        v = _clip(patch.get(f), MAX_FIELD_CHARS)
        if f == "breadth" and v == "wide":
            v = "small"
        if not v or v == draft.get(f):
            continue
        if f == "breadth" and v not in BREADTHS:
            continue
        out[f] = v
    return out


def transcript(messages: list[BuilderMessage]) -> list[tuple[str, str]]:
    """The conversation as it is sent: the last forty messages, each clipped, the oldest dropped until it fits the character budget."""
    rows = [("Creator" if m.role == "user" else "Builder", _clip(m.content, MAX_MESSAGE_CHARS)) for m in messages[-MAX_MESSAGES:]]
    rows = [r for r in rows if r[1]]
    while len(rows) > 1 and sum(len(t) for _, t in rows) > MAX_TRANSCRIPT_CHARS:
        rows.pop(0)
    return rows


def _brief_lines(draft: dict[str, str]) -> str:
    return "\n".join(f"{f}: {draft[f] or '(not yet)'}" for f in FIELDS)


def _today() -> str:
    """The owner's day, for a present-day world: the builder is a model, and a model's own idea of 'now' is a year or more behind."""
    return f"TODAY (the owner's date, for any present-day world): {time.strftime('%A %d %B %Y')}.\n\n"


# -- Maps (Marc, 2026-10-23: "a button in the world builder that allows the user to select saved map images"). The images kept on this
# machine, in maps/<folder>/<file> (a map is someone's drawing, outside git; the folder is the world a map was kept for): the page lists
# them, the owner picks up to three, the builder looks at a picked map once, with the message after it is picked, and a build copies
# each into the new world's own maps folder and has it read there, which makes a proposal and plants nothing. Looking at a map takes a
# model that takes images; a model that does not is refused plainly, and the page says so beside the button.
NEEDS_IMAGES = "A map needs a model that takes images; none of the configured ones did."


def _maps_dir() -> Path:
    from smritilila.routes import pratistha  # its MAPS_DIR: one place for maps, which a test replaces there
    return pratistha.MAPS_DIR


def saved_maps() -> list[dict]:
    """The map images on this machine, one folder deep, as {id: "folder/file", folder, name, bytes}: png, jpeg, webp or gif, not hidden, not
    empty, not past 25 MB, at most 200."""
    root = _maps_dir()
    out: list[dict] = []
    if not root.is_dir():
        return out
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for f in sorted(folder.iterdir()):
            if not f.is_file() or f.name.startswith(".") or f.suffix.lower() not in MAP_MEDIA:
                continue
            size = f.stat().st_size
            if 0 < size <= MAX_MAP_BYTES:
                out.append({"id": f"{folder.name}/{f.name}", "folder": folder.name, "name": f.name, "bytes": size})
            if len(out) >= MAX_SAVED_MAPS:
                return out
    return out


def map_path(map_id: str) -> Path:
    """A saved map by its id: only an id the listing gives, so no request reaches any other file on the disk."""
    for m in saved_maps():
        if m["id"] == map_id:
            return _maps_dir() / m["folder"] / m["name"]
    raise HTTPException(404, "no such saved map")


def _look(ids: list[str]) -> list:
    """The maps named, as pictures a model takes."""
    from smritilila.smrti.manacitra import load_picture
    pictures = []
    for map_id in ids[:MAX_MAPS]:
        try:
            pictures.append(load_picture(map_path(map_id)))
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 -- a file that is not an image, or a format that needs a library that is not there
            raise HTTPException(422, f"that map could not be read as a picture: {str(exc)[:160]}")
    return pictures


def _vision_client():
    """The model that looks at a map: the seed writer's own choice, the one that takes pictures first and the other only if it fails
    (the same as the map reading a build makes). Tests replace this."""
    from smritilila.routes.bija import _seeder
    try:
        roles = adhisthana._resolve_roles()
    except (KeyError, ValueError):
        roles = None
    if roles is None or roles.smrti_client is None:
        raise HTTPException(503, "no model is configured for the builder. " + NEEDS_IMAGES)
    return _seeder(roles.smrti_client)


def _image_extension(data: bytes) -> str | None:
    """The extension of the image these bytes are, by what they begin with (png, jpeg, gif, webp), or None: a file is kept only for what it is, never for what it is called."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def _safe_stem(name: str) -> str:
    """A file name's stem made safe to keep: no folder, no dot, no odd character, not long."""
    stem = Path(name.replace("\\", "/").rsplit("/", 1)[-1]).stem
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._-")[:80] or "map"


@router.get("/builder/maps", response_model=list[SavedMap])
async def builder_maps(request: Request) -> list[dict]:
    """The saved map images the owner may pick from."""
    _owner_only(request, "builds worlds")
    return saved_maps()


@router.post("/builder/maps", response_model=SavedMap, status_code=201)
async def builder_map_upload(name: str, request: Request) -> dict:
    """An image from the owner's own machine (the page's file browser), kept with the saved maps in maps/uploaded/. The body is the image itself,
    sent as it is (no multipart: nothing else is needed to read it); at most 25 MB; kept only if its bytes are a png, jpeg, gif or webp, under
    the extension they are and a safe form of the name given; one already kept is not kept twice, and another of the same name is kept beside it."""
    _owner_only(request, "builds worlds")
    if int(request.headers.get("content-length") or 0) > MAX_MAP_BYTES:
        raise HTTPException(413, f"a map is at most {MAX_MAP_BYTES // (1024 * 1024)} MB")
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_MAP_BYTES:
            raise HTTPException(413, f"a map is at most {MAX_MAP_BYTES // (1024 * 1024)} MB")
    data = bytes(data)
    extension = _image_extension(data)
    if extension is None:
        raise HTTPException(422, "that is not a png, jpeg, gif or webp image")
    folder = _maps_dir() / UPLOAD_FOLDER
    stem = _safe_stem(name)
    folder.mkdir(parents=True, exist_ok=True)
    target, n = folder / f"{stem}{extension}", 1
    while target.exists() and target.read_bytes() != data:
        n += 1
        target = folder / f"{stem}-{n}{extension}"
    if not target.exists():
        if len([f for f in folder.iterdir() if f.is_file()]) >= MAX_UPLOADED:
            raise HTTPException(409, f"the folder of uploaded maps holds {MAX_UPLOADED} already: remove some from maps/{UPLOAD_FOLDER}/")
        target.write_bytes(data)
        log_event("builder.map_uploaded", bytes=len(data), name=target.name)
    return {"id": f"{UPLOAD_FOLDER}/{target.name}", "folder": UPLOAD_FOLDER, "name": target.name, "bytes": len(data)}


@router.get("/builder/maps/file")
async def builder_map_file(id: str, request: Request):
    """One saved map's image, for the picker's thumbnail."""
    _owner_only(request, "builds worlds")
    path = map_path(id)
    return FileResponse(path, media_type=MAP_MEDIA[path.suffix.lower()], headers={"Cache-Control": "private, max-age=300"})


@router.post("/builder/turn", response_model=BuilderTurnResponse)
async def builder_turn(req: BuilderTurnRequest, request: Request) -> BuilderTurnResponse:
    """One turn of the conversation: the builder's reply and what it sets in the brief. `look`: saved maps shown with this message."""
    _owner_only(request, "builds worlds")
    rows = transcript(req.messages)
    if not rows or rows[-1][0] != "Creator":
        raise HTTPException(422, "the conversation must end with the owner's message")
    draft = clean_draft(req.draft)
    pictures = _look(req.look)
    user = (_today() + "THE BRIEF SO FAR (shown to the owner beside this conversation):\n" + _brief_lines(draft)
            + "\n\nTHE CONVERSATION:\n" + "\n".join(f"{who}: {text}" for who, text in rows)
            + ("\n\nThe owner has shown you " + ("a map" if len(pictures) == 1 else f"{len(pictures)} maps") + " with this message ("
               + ", ".join(p.name for p in pictures) + "): look at it first." if pictures else "")
            + "\n\nReply to the owner's latest message.")
    try:
        client = _vision_client() if pictures else _builder_client()
        out = await asyncio.to_thread(client.complete_with_tool, TURN_SYSTEM, user, tool_name="builder_turn",
                                      tool_description="Say what you say to the owner, and set what the brief now says.",
                                      input_schema=TURN_TOOL, max_tokens=3000, **({"images": pictures} if pictures else {}))
    except HTTPException:
        raise
    except ImagesNotSupported:
        raise HTTPException(422, NEEDS_IMAGES)
    except Exception as exc:  # noqa: BLE001 -- a model's failure is the owner's retry, not a crash
        log_event("builder.failed", stage="turn", looked=len(pictures), error=str(exc)[:200])
        raise HTTPException(502, f"the builder could not answer: {str(exc)[:160]}" + (" " + NEEDS_IMAGES if pictures else ""))
    message = _clip((out or {}).get("message"), 6000)
    if not message:
        raise HTTPException(502, "the builder came back empty")
    patch = clean_patch((out or {}).get("patch"), draft)
    ready = bool((out or {}).get("ready")) and all({**draft, **patch}[f] for f in ("name", "source", "where_when"))
    log_event("builder.turn", messages=len(rows), patched=sorted(patch), ready=ready, looked=len(pictures))
    return BuilderTurnResponse(message=message, patch=patch, ready=ready)


@router.post("/builder/preview", response_model=BuilderPreviewResponse)
async def builder_preview(req: BuilderPreviewRequest, request: Request) -> BuilderPreviewResponse:
    """What the world will open with, from the brief as it stands, before anything is built."""
    _owner_only(request, "builds worlds")
    draft = clean_draft(req.draft)
    if not (draft["name"] and draft["source"] and draft["where_when"]):
        raise HTTPException(422, "a preview needs a name, a source, and where and when")
    user = _today() + "THE BRIEF:\n" + _brief_lines(draft)
    try:
        out = await asyncio.to_thread(_builder_client().complete_with_tool, PREVIEW_SYSTEM, user, tool_name="builder_preview",
                                      tool_description="Show what the world will open with.", input_schema=PREVIEW_TOOL, max_tokens=3000)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log_event("builder.failed", stage="preview", error=str(exc)[:200])
        raise HTTPException(502, f"the preview could not be written: {str(exc)[:160]}")
    out = out or {}
    lists = lambda k: [_clip(x, 300) for x in (out.get(k) or []) if _clip(x, 300)][:8]  # noqa: E731
    line, opening = _clip(out.get("line"), 400), _clip(out.get("opening"), 2000)
    if not line or not opening:
        raise HTTPException(502, "the preview came back incomplete")
    log_event("builder.preview", cautions=len(lists("cautions")))
    return BuilderPreviewResponse(line=line, opening=opening, places=lists("places"), people=lists("people"), cautions=lists("cautions"))


def _slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")[:40] or "world"


def _write_session(name: str, messages: list[BuilderMessage], draft: BuilderDraft | dict) -> str:
    """builder_sessions/<time>-<name>.json, written once, never changed; its file name."""
    SESSIONS_DIR.mkdir(exist_ok=True)
    path = SESSIONS_DIR / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{_slug(name)}.json"
    rows = [{"role": m.role, "content": _clip(m.content, MAX_MESSAGE_CHARS)} for m in messages[-200:]]
    path.write_text(json.dumps({"saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "name": _clip(name, 200),
                                "draft": clean_draft(draft), "messages": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    log_event("builder.saved", messages=len(rows))
    return path.name


# -- Building (2026-10-06). Every world has a server of its own, and a server whose port a world holds never makes or shows another
# (v5 section 42), so Build cannot make a world on the server the page is open on: it was a 409 on every one of them. A built world
# is made the way an invited author's was: its own database, its own port and server, then seeded THERE. It runs as a job the page
# polls, because seeding takes minutes at wide breadth and one request through the tunnel is cut off at about a hundred seconds.
HEALTH_SECONDS = 90
GENESIS_SECONDS = 900
_BUILDS: dict[str, dict] = {}
_BUILD_LOCK = threading.Lock()


class BuilderBuildRequest(BaseModel):
    name: str
    draft: BuilderDraft
    messages: list[BuilderMessage] = []
    maps: list[str] = Field(default=[], max_length=MAX_MAPS)   # saved maps (their ids) to be read for the new world, after it is seeded


class BuilderBuildStarted(BaseModel):
    job: str


class BuilderBuildStatus(BaseModel):
    job: str
    name: str
    step: str
    done: bool = False
    error: str | None = None
    result: dict | None = None


def _http(method: str, url: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, object]:
    """One call to a world's own server: (status, decoded body); 0 and a reason where it could not be reached."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8") if body is not None else None, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(text)
        except ValueError:
            return e.code, text[:300]
    except OSError as e:
        return 0, str(e)[:200]


def _wait_health(base: str) -> bool:
    deadline = time.monotonic() + HEALTH_SECONDS
    while time.monotonic() < deadline:
        code, _ = _http("GET", base + "/health", timeout=3.0)
        if code == 200:
            return True
        time.sleep(1.0)
    return False


def _detail(out: object) -> str:
    return str(out.get("detail") if isinstance(out, dict) and out.get("detail") else out)[:240]


def _set(job: str, **fields) -> None:
    with _BUILD_LOCK:
        _BUILDS[job].update(fields)


MAP_SECONDS = 1800   # a map is read in several passes of several model calls each


def _read_maps(job: str, base: str, name: str, paths: list[Path]) -> list[dict]:
    """After the seed: each map copied into the new world's own maps folder and read there (POST /world/manacitra on its server), which
    makes a proposal and plants nothing -- a god reviews it. A map that cannot be read leaves the world built and says why."""
    from smritilila.backup import _safe_dirname
    dest = _maps_dir() / _safe_dirname(name)
    done = []
    for p in paths:
        _set(job, step=f"Reading the map {p.name} (several minutes)")
        got_back = {"image": p.name, "proposal": None, "error": None}
        try:
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / p.name
            if target.exists() and target.stat().st_size != p.stat().st_size:   # another map of the same file name
                target = dest / f"{p.parent.name}-{p.name}"
            if not target.exists():
                shutil.copyfile(p, target)
            got_back["image"] = target.name
            code, out = _http("POST", base + "/world/manacitra", {"image": target.name}, timeout=MAP_SECONDS)
            if code == 200 and isinstance(out, dict) and out.get("id"):
                got_back["proposal"] = out["id"]
            else:
                got_back["error"] = f"({code}) {_detail(out)}"
        except OSError as exc:
            got_back["error"] = str(exc)[:200]
        done.append(got_back)
    return done


def _grow_rings(job: str, base: str, rings: int) -> dict:
    """After the seed, a medium or large world grows outward: one ring after another (POST /world/expand, a ring being one seed call), each from the edge
    the one before left. A ring that adds nothing ends it; a ring that fails leaves the world as grown so far and says why."""
    places = people = done = 0
    for n in range(1, rings + 1):
        _set(job, step=f"Growing the world outward: ring {n} of {rings} (a minute or two each)")
        code, out = _http("POST", base + "/world/expand", {}, timeout=EXPAND_SECONDS)
        if code != 200 or not isinstance(out, dict):
            return {"rings": done, "places": places, "people": people, "error": f"ring {n}: ({code}) {_detail(out)}"}
        added_places, added_people = len(out.get("places") or []), len(out.get("people") or [])
        done, places, people = done + 1, places + added_places, people + added_people
        if not added_places and not added_people:
            return {"rings": done, "places": places, "people": people, "stopped": "a ring added nothing"}
    return {"rings": done, "places": places, "people": people}


def _run_build(job: str, brief: dict[str, str], messages: list[BuilderMessage], god: str | None = None, maps: list[Path] | None = None) -> None:
    """The job: the world's own database, port and server, then its seed, there, then each map read (a proposal, planted by no one). A
    failed step leaves what was made (a retry of the same name seeds the empty world it finds, never makes a second). `god`: the
    registered user who asked, who is a god of the world they make and of nothing else; None when the owner did (the owner is a god of
    every world)."""
    from smritilila.routes import loka
    from smritilila.worlds_registry import add_world
    name = brief["name"]
    try:
        entry = next((w for w in adhisthana.get_worlds_registry() if w.name.lower() == name.lower()), None)
        if entry is None:
            _set(job, step="Making the world's own database")
            _dbname, url = asyncio.run(loka._make_world_database(name))
            entry = add_world(name, url, gods=(god,) if god else (), port=loka._free_world_port())
            app.state.worlds_registry = None  # read the registry again
            _set(job, step="Starting its server")
            loka._spawn_world_server(entry)
        elif not entry.port or (god and god not in entry.gods):
            raise RuntimeError(f"a world named {name!r} already exists")
        base = f"http://127.0.0.1:{entry.port}"
        if _http("GET", base + "/health", timeout=3.0)[0] != 200:
            _set(job, step="Starting its server")
            loka._spawn_world_server(entry)
        if not _wait_health(base):
            raise RuntimeError(f"its server (port {entry.port}) did not start in {HEALTH_SECONDS} s")
        code, there = _http("GET", base + "/world/exists")
        if code == 200 and isinstance(there, dict) and there.get("exists"):
            raise RuntimeError(f"a world named {name!r} already exists and has been seeded")
        _set(job, step="Seeding the world (a minute or two; several at wide breadth)")
        body = {"name": name, **{f: brief[f] for f in ("source", "where_when", "begins", "different", "feel")}, "breadth": SEED_BREADTH[brief["breadth"]]}
        code, out = _http("POST", base + "/genesis", body, timeout=GENESIS_SECONDS)
        if code != 200:
            raise RuntimeError(f"seeding failed ({code}): {_detail(out)}")
        try:
            _write_session(name, messages, brief)
        except OSError as exc:  # the world is built; the talk is a record
            log_event("builder.failed", stage="save", error=str(exc)[:200])
        result = {"name": name, "port": entry.port, "address": entry.url or base, "public": bool(entry.url)}
        if RINGS.get(brief["breadth"]):
            result["grown"] = _grow_rings(job, base, RINGS[brief["breadth"]])
        if maps:
            result["maps"] = _read_maps(job, base, name, maps)
        log_event("builder.built", name=name, port=entry.port)
        _set(job, done=True, step="Built", result=result)
    except Exception as exc:  # noqa: BLE001 -- the owner reads it on the page
        log_event("builder.failed", stage="build", error=str(exc)[:200])
        _set(job, done=True, error=str(exc)[:300])


@router.post("/builder/build", response_model=BuilderBuildStarted, status_code=202)
async def builder_build(req: BuilderBuildRequest, request: Request, background: BackgroundTasks) -> BuilderBuildStarted:
    """Build the world from the brief, as a job: this answers at once with its id; GET /builder/build/{job} says where it is."""
    who = _owner_only(request, "builds worlds")
    brief = clean_draft(req.draft)
    brief["name"] = _clip(req.name, MAX_FIELD_CHARS) or brief["name"]
    if not (brief["name"] and brief["source"] and brief["where_when"]):
        raise HTTPException(422, "a world needs a name, a source, and where and when")
    if brief["breadth"] not in BREADTHS:
        raise HTTPException(422, "how much to build must be here, around, small, medium or large")
    maps = [map_path(i) for i in req.maps]   # a map that is not saved is refused now, not after the world is made
    with _BUILD_LOCK:
        if any(b["name"].lower() == brief["name"].lower() and not b["done"] for b in _BUILDS.values()):
            raise HTTPException(409, f"{brief['name']!r} is already being built")
        job = uuid.uuid4().hex[:12]
        _BUILDS[job] = {"job": job, "name": brief["name"], "step": "Starting", "done": False, "error": None, "result": None, "by": who}
    background.add_task(_run_build, job, brief, req.messages, who, maps)
    return BuilderBuildStarted(job=job)


@router.get("/builder/build/{job}", response_model=BuilderBuildStatus)
async def builder_build_status(job: str, request: Request) -> BuilderBuildStatus:
    who = _owner_only(request, "builds worlds")
    with _BUILD_LOCK:
        state = dict(_BUILDS.get(job) or {})
    if not state or (who is not None and state.get("by") != who):  # a user sees only the build they asked for
        raise HTTPException(404, "no such build")
    return BuilderBuildStatus(**state)
