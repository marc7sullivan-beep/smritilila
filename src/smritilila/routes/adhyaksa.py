# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Adhyakṣa (अध्यक्ष), the overseer: the one who looks over every part of a household and answers for it. The owner's admin screen
(Marc, 2026-10-06): the worlds side by side -- each one's port, whether its server answers, who is on it, who may enter -- and the
guest setting of each, changed here and kept in worlds.json, which every server rereads when it changes (so no restart). Also the
registered users (users.json: who may use the world builder) and the bots (bots.json: which characters an agent plays when a person
is active in their world; scripts/bots_supervisor.py carries them out).

Owner-only, wholly: the gate refuses a guest every /admin route (dvara._OWNER_ONLY_PREFIX). It reads and writes the registry, and asks
a world's own server what it holds; it touches a world only to make an NPC playable when it is switched on as a bot (PUT /admin/bots),
and nothing here writes a world's memory. Starting, stopping and restarting servers stay by hand."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from smritilila import adhisthana  # the registry, read through it so a test's patch there reaches it
from smritilila import dvara
from smritilila.sakshi import log_event
from smritilila import bots_registry, users_registry
from smritilila.worlds_registry import set_world_guests

router = APIRouter()

_PAGE = Path(__file__).resolve().parent.parent / "web" / "admin.html"
_EMAIL = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")


class NewUser(BaseModel):
    email: str
    builder: bool = False


class BuilderSetting(BaseModel):
    builder: bool


class BotSetting(BaseModel):
    world: str
    entity_id: str
    active: bool
    existing: bool = False   # a character people play, made a bot by the owner's own word (Marc, 2026-10-06: Dunstan)


class GuestSetting(BaseModel):
    mode: str            # "all", "none" or "list"
    emails: list[str] = []


def _ask(port: int, path: str) -> object | None:
    """What a world's own server answers on this machine -- from here, so as an owner -- or None if it does not."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _look(port: int | None) -> dict:
    if port is None:
        return {"up": None, "online": None}
    if _ask(port, "/health") is None:
        return {"up": False, "online": None}
    players = _ask(port, "/players")
    online = [p.get("label") or p.get("entity_id") for p in players if p.get("online")] if isinstance(players, list) else None
    return {"up": True, "online": online}


def _owner_only(request: Request) -> None:
    """The gate already refuses a guest; this is the endpoint's own word, so it holds if the gate's list is ever edited."""
    guest = dvara._guest(request)
    if guest is not None:
        log_event("guest.refused", guest=guest, method=request.method, path=request.url.path, reason="admin")
        raise HTTPException(403, "only the owner oversees the worlds")


def _setting(guests) -> dict:
    if isinstance(guests, tuple):
        return {"mode": "list", "emails": list(guests)}
    return {"mode": "all" if guests else "none", "emails": []}


@router.get("/admin")
def admin_page(request: Request) -> FileResponse:
    _owner_only(request)
    return FileResponse(_PAGE, headers={"Cache-Control": "no-store"})


@router.get("/admin/worlds")
async def admin_worlds(request: Request) -> dict:
    _owner_only(request)
    worlds = adhisthana.get_worlds_registry()
    looks = await asyncio.gather(*(asyncio.to_thread(_look, w.port) for w in worlds))
    return {
        "always_open": sorted(dvara.GUEST_WORLDS),
        "worlds": [{"name": w.name, "port": w.port, "url": w.url, "description": w.description, "gods": list(w.gods),
                    "guests": _setting(w.guests), "open_to_all_by_server": w.name in dvara.GUEST_WORLDS, **look}
                   for w, look in zip(worlds, looks)],
    }


@router.put("/admin/worlds/{name}/guests")
def admin_set_guests(name: str, req: GuestSetting, request: Request) -> dict:
    _owner_only(request)
    if req.mode not in ("all", "none", "list"):
        raise HTTPException(422, "mode is all, none or list")
    emails = sorted({e.strip().lower() for e in req.emails if e.strip()})
    bad = [e for e in emails if not _EMAIL.match(e)]
    if bad:
        raise HTTPException(422, f"not an email: {', '.join(bad)}")
    if req.mode == "list" and not emails:
        raise HTTPException(422, "a list of guests needs at least one email; choose none to close the world to guests")
    value = True if req.mode == "all" else (emails if req.mode == "list" else False)
    if not set_world_guests(name, value):
        raise HTTPException(404, f"no world named {name!r}")
    log_event("admin.guests", world=name, mode=req.mode, emails=emails)
    return {"name": name, **_setting(True if req.mode == "all" else (tuple(emails) if req.mode == "list" else False))}


# -- registered users: who may use the world builder -------------------------------------------------------------------------------

@router.get("/admin/users")
def admin_users(request: Request) -> dict:
    _owner_only(request)
    return {"users": [{"email": u.email, "builder": u.builder, "added": u.added} for u in users_registry.load_users()]}


@router.post("/admin/users", status_code=201)
def admin_add_user(req: NewUser, request: Request) -> dict:
    _owner_only(request)
    email = req.email.strip().lower()
    if not _EMAIL.match(email):
        raise HTTPException(422, "not an email")
    if email in dvara.OWNERS:
        raise HTTPException(409, "an owner may do everything already")
    try:
        user = users_registry.add_user(email, builder=req.builder)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    log_event("admin.user_added", email=email, builder=req.builder)
    return {"email": user.email, "builder": user.builder, "added": user.added}


@router.put("/admin/users/{email}/builder")
def admin_user_builder(email: str, req: BuilderSetting, request: Request) -> dict:
    _owner_only(request)
    if not users_registry.set_builder(email, req.builder):
        raise HTTPException(404, "no such user")
    log_event("admin.user_builder", email=email.strip().lower(), builder=req.builder)
    return {"email": email.strip().lower(), "builder": req.builder}


@router.delete("/admin/users/{email}")
def admin_remove_user(email: str, request: Request) -> dict:
    _owner_only(request)
    if not users_registry.remove_user(email):
        raise HTTPException(404, "no such user")
    log_event("admin.user_removed", email=email.strip().lower())
    return {"removed": email.strip().lower()}


# -- bots: the beings an agent plays whenever a person is active in their world ---------------------------------------------------------

LIST_MAX = 20        # beings offered for each world in all (Marc, 2026-10-06): those that matter first, then the richest in memory, then the most spoken of
PER_RANKING = 10     # of the living, at most this many from each of the two rankings
ALIVE_POOL = 40      # how many of each ranking are read, to find those that live

ALIVE_SYSTEM = """You read what a world says about some beings, and judge which of them are ALIVE: a living creature that perceives and acts for itself -- a person, an animal, a spirit, a god, a ghost that speaks and wills. NOT alive: an object, a device, a document, a place, a vehicle, a substance, an abstraction -- even when the world has given it memories, and even when people carry, use or talk to it. Judge by what its memories say it is and does, not by its name alone. When in doubt, a thing people use is not alive and a being that chooses is. Reply through the tool: alive is the list of ids of the alive ones, and only those."""
ALIVE_SCHEMA = {"type": "object", "properties": {"alive": {"type": "array", "items": {"type": "string"}, "description": "ids of the beings that are alive"}},
                "required": ["alive"]}


IMPORTANT_MAX = 8    # at most this many picked in one reading

IMPORTANT_SYSTEM = """You read a world's setting and the names of its beings, and pick the IMPORTANT ones: living beings that matter to the world's story and its powers -- rulers, faction heads, villains and their lieutenants, high priests, dragons and other great monsters, legendary figures. NOT important: ordinary townsfolk, travellers, guards, shopkeepers, servants; and never an object, a place or an abstraction. A name may carry its description ("a pale priest draped in silk"); the number is how many memories it holds, and a being that matters may hold few. Use what you know of the setting's source where it has one. Pick only those you are fairly sure of: at most 8, and none if none is. Reply through the tool: important is the list of ids."""
IMPORTANT_SCHEMA = {"type": "object", "properties": {"important": {"type": "array", "items": {"type": "string"}, "description": "ids of the important beings"}},
                    "required": ["important"]}


def _judge_important(world: str, setting: str, beings: list[dict]) -> dict[str, bool]:
    """A model reads the setting and the names of the beings not yet judged, and picks the ones that matter to the world's story."""
    lines = [f"{b['id']} | {b.get('name') or b['id']} | {b.get('memories', 0)}" for b in beings]
    out = _alive_client().complete_with_tool(
        IMPORTANT_SYSTEM, f"THE WORLD: {world}\nTHE SETTING: {setting or '(none written)'}\n\nBEINGS (id | name | memories):\n" + "\n".join(lines),
        tool_name="judge_important", tool_description="Say which of the beings are important.", input_schema=IMPORTANT_SCHEMA, max_tokens=1024)
    said = [str(i) for i in (out.get("important") if isinstance(out, dict) else None) or []]
    candidates = {b["id"] for b in beings}
    picked = set([i for i in dict.fromkeys(said) if i in candidates][:IMPORTANT_MAX])      # the first eight it names, and only beings it was shown
    return {b["id"]: b["id"] in picked for b in beings}


def _alive_client():
    """The model that reads: the builder's (the best, low volume, owner only). Tests replace this."""
    from smritilila.routes.sthapati import _builder_client
    return _builder_client()


def _judge_alive(world: str, npcs: list[dict]) -> dict[str, bool]:
    """A model reads each being's name and what it holds, and says which are alive. One call for all of them."""
    lines = []
    for n in npcs:
        lines.append(f"{n['id']} | {n.get('name') or n['id']}")
        lines += [f"    - {m}" for m in n.get("sample") or []]
    out = _alive_client().complete_with_tool(
        ALIVE_SYSTEM, f"THE WORLD: {world}\n\nBEINGS (id | name, then some of what each holds):\n" + "\n".join(lines),
        tool_name="judge_alive", tool_description="Say which of the beings are alive.", input_schema=ALIVE_SCHEMA, max_tokens=1024)
    said = {str(i) for i in (out.get("alive") if isinstance(out, dict) else None) or []}
    return {n["id"]: n["id"] in said for n in npcs}


def _post(port: int, path: str) -> bool:
    """A POST to a world's own server from this machine, so as its owner; whether it was accepted."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception:
        return False


def _characters(port: int | None) -> dict[str, str] | None:
    """{id: name} of a world's playable beings, asked of its own server; None if it does not answer."""
    there = _ask(port, "/world/exists") if port else None
    if not isinstance(there, dict) or not isinstance(there.get("playable_entities"), dict):
        return None
    return {eid: (labels[0] if labels else eid) for eid, labels in there["playable_entities"].items()}


def _npcs(port: int | None) -> list[dict]:
    got = _ask(port, f"/world/npcs?limit={ALIVE_POOL}&sample=true") if port else None
    return [n for n in got if isinstance(n, dict) and n.get("id")] if isinstance(got, list) else []


def _alive_npcs(world: str, port: int | None) -> tuple[list[dict], str | None]:
    """The richest NPCs that are alive, at most PER_RANKING from each ranking, and a warning if a model could not be asked. A being is judged once
    and the judgement kept; a being a model could not judge is not offered until it can."""
    pool = _npcs(port)
    judged = bots_registry.load_alive().get(world, {})
    unknown = [n for n in pool if n["id"] not in judged]
    warning = None
    if unknown:
        try:
            fresh = _judge_alive(world, unknown)
            bots_registry.set_alive(world, fresh)
            judged = {**judged, **fresh}
        except Exception as exc:  # noqa: BLE001 -- no model: the list is what is known, and says so
            log_event("admin.alive_failed", world=world, error=str(exc)[:200])
            warning = "could not ask a model which of its beings are alive, so none new are offered"
    living = [n for n in pool if judged.get(n["id"])]
    present = sorted(living, key=lambda n: -(n.get("memories") or 0))[:PER_RANKING]            # who is most here...
    spoken = [n for n in sorted(living, key=lambda n: -(n.get("known_by") or 0)) if n["id"] not in {p["id"] for p in present}]
    return present + [n for n in spoken if n.get("known_by")][:PER_RANKING], warning           # ...and who is most spoken of, though met seldom


def _important_npcs(world: str, port: int | None) -> tuple[list[dict], str | None]:
    """The beings a model judged important to this world's story, with a warning if it could not be asked. The whole cast is read once;
    afterwards only the beings new since. A being is judged once and the judgement kept."""
    cast = _ask(port, "/world/cast") if port else None
    if not isinstance(cast, dict) or not isinstance(cast.get("beings"), list):
        return [], None
    beings = [b for b in cast["beings"] if isinstance(b, dict) and b.get("id")]
    judged = bots_registry.load_important().get(world, {})
    unknown = [b for b in beings if b["id"] not in judged]
    warning = None
    if unknown:
        try:
            fresh = _judge_important(world, str(cast.get("setting") or ""), unknown)
            bots_registry.set_important(world, fresh)
            judged = {**judged, **fresh}
        except Exception as exc:  # noqa: BLE001 -- no model: nothing new is offered, and it says so
            log_event("admin.important_failed", world=world, error=str(exc)[:200])
            warning = "could not ask a model who matters to this world's story, so none new are offered"
    return [b for b in beings if judged.get(b["id"])], warning


def _offered(world: str, port: int | None) -> tuple[bool, list[dict], str | None, list[dict]]:
    """(does its server answer, the beings offered as bots, a warning, the characters people play): those ever switched on, then those
    judged to matter to the world's story, then the living NPCs richest in memory, then those most spoken of. A person's character (playable, never switched on here) is not offered;
    it is listed apart, for the owner to make a bot by name."""
    chars = _characters(port)
    if chars is None:
        return False, [], None, []
    npcs, warning = _alive_npcs(world, port)
    important, warn_important = _important_npcs(world, port)
    warning = warning or warn_important
    names = {**{n["id"]: n.get("name") or n["id"] for n in npcs + important}, **chars}
    known = bots_registry.load_known().get(world, [])
    fresh = [n["id"] for n in important + npcs if n["id"] not in known]            # those that matter first, then the richest, then the most spoken of
    ids = known + list(dict.fromkeys(fresh))[:LIST_MAX]                              # what was switched on stays, whatever the cap
    mattering = {n["id"] for n in important}
    active = bots_registry.load_bots().get(world, [])
    memories = {n["id"]: n.get("memories") for n in important + npcs}
    known_by = {n["id"]: n.get("known_by") for n in npcs}
    players = [{"id": i, "name": n} for i, n in chars.items() if i not in known]
    return True, [{"id": i, "name": names.get(i, i), "bot": i in active, "memories": memories.get(i), "known_by": known_by.get(i), "important": i in mattering} for i in ids], warning, players


@router.get("/admin/bots")
async def admin_bots(request: Request) -> dict:
    _owner_only(request)
    worlds = [w for w in adhisthana.get_worlds_registry() if w.port]
    offered = await asyncio.gather(*(asyncio.to_thread(_offered, w.name, w.port) for w in worlds))
    return {"worlds": [{"name": w.name, "port": w.port, "up": up, "characters": beings, "warning": warning, "players": players}
                       for w, (up, beings, warning, players) in zip(worlds, offered)]}


@router.put("/admin/bots")
async def admin_set_bot(req: BotSetting, request: Request) -> dict:
    _owner_only(request)
    world = next((w for w in adhisthana.get_worlds_registry() if w.name == req.world), None)
    if world is None or not world.port:
        raise HTTPException(404, f"no world named {req.world!r} with a server of its own")
    up, beings, _warning, players = await asyncio.to_thread(_offered, world.name, world.port)
    if not up:
        raise HTTPException(409, f"{world.name}'s server is not answering, so its beings cannot be listed")
    being = next((b for b in beings if b["id"] == req.entity_id), None)
    if being is None and req.existing:
        being = next((p for p in players if p["id"] == req.entity_id), None)        # a character people play, made a bot by name
    if being is None:
        raise HTTPException(404, "that is not offered as a bot in this world (a person's character is made one only by name, with existing=true)")
    if req.active and req.entity_id not in (await asyncio.to_thread(_characters, world.port) or {}):
        # an NPC becomes playable to be played: logged off and waiting until its agent arrives (v5 section 37)
        if not await asyncio.to_thread(_post, world.port, f"/world/playable/{req.entity_id}"):
            raise HTTPException(502, f"{being['name']} could not be made playable in {world.name}")
    ids = bots_registry.set_bot(world.name, req.entity_id, req.active)
    log_event("admin.bot", world=world.name, entity_id=req.entity_id, name=being["name"], active=req.active)
    return {"world": world.name, "bots": ids}
