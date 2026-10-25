# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Upasthiti (2026-10-05, Marc's sign-off): being present. Who the
players are (/players, /me), making an entity playable, whether the world
may act for a player while they are away, their presence and heartbeat,
and retiring a character.

Moved whole from main.py, nothing inside changed."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Query, Request

from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila import dvara  # the gate, called through it for the same reason
from smritilila import users_registry
from smritilila.adhisthana import WorldSession, _store
from smritilila.sakshi import log_event
from smritilila.schemas import AutonomyRequest, MeOut, MelanaOut, PlayerOut, PresenceRequest, RetireRequest, _labelled
from smritilila.dvara import _require_own, _require_player


router = APIRouter()


def _player_out(world: WorldSession, pid: str) -> PlayerOut:
    e = world.canon.entities.get(pid)
    return PlayerOut(entity_id=pid, label=(e.labels[0] if e and e.labels else pid),
                     autonomous=pid in world.autonomous, online=world.online(pid), waiting=pid in world.waiting)


@router.get("/players", response_model=list[PlayerOut])
async def list_players(request: Request = None) -> list[PlayerOut]:
    world = await adhisthana.get_world()
    guest = dvara._guest(request) if request is not None else None
    return [_player_out(world, pid) for pid in world.playable_entity_ids if guest is None or dvara._plays(world, guest, pid)]


@router.get("/world/sync", response_model=list[MelanaOut])
async def melana(as_: str | None = Query(None, alias="as"), own: bool = Query(False), request: Request = None) -> list[MelanaOut]:
    """Melana (मेलन), coming together (Marc, 2026-10-16: "Show all online players to everyone"): who is online, and the scene and
    the minute each stands at, so a player's page can show who is level with them (the same scene and minute) and who is remote, and
    offer to go to them. The same for an owner and a guest: it says where and when, nothing of what anyone remembers. A logged-off
    player is not listed (Marc: "remove logged off from the list"); `as` names the character the page plays, listed whether or not
    it is online, so the page always finds itself. `own` (an owner's, never a guest's: 2026-10-22, for an agent's time guard): also every player whose time is still their own although
    their page is closed -- offline, not yet logged off, not waiting (v5 §37: the world stops at their present) -- with `online` false. Cost: one pass over
    the players."""
    world = await adhisthana.get_world()
    guest = dvara._guest(request) if request is not None else None
    mine = as_ if as_ in world.playable_entity_ids and (guest is None or dvara._plays(world, guest, as_)) else None
    sk = world.samskara()
    kosha = sk.kosha_for(world.canon, ())
    t = (sk.mem.max_tick if sk.mem.max_tick >= 0 else world.canon.world_time) + 1
    out = []
    for pid in world.playable_entity_ids:
        away = own and guest is None and not world.online(pid) and pid not in world.logged_off and pid not in world.waiting   # time still their own
        if not (world.online(pid) or pid == mine or away):
            continue
        scene = kosha.scene_of(pid, t)
        at = sk.kala.clock(pid) if sk.kala is not None else None
        e, s = world.canon.entities.get(pid), world.canon.entities.get(scene) if scene else None
        out.append(MelanaOut(entity_id=pid, label=(e.labels[0] if e and e.labels else pid), scene_id=scene,
                             scene=(s.labels[0] if s and s.labels else scene), at=at, at_text=sk.kala.say(at) if at is not None else None,
                             online=world.online(pid)))
    return out


@router.get("/world/npcs")
async def list_npcs(limit: int = 4, sample: bool = False) -> list[dict]:
    """The beings of this world no one plays, ranked two ways (Marc, 2026-10-06: the overseer's bot list offers the NPCs that matter,
    "the dragon, the head evil cleric"): by the memories each holds (who is most present), and by `known_by`, how many other beings
    hold a memory ABOUT it (who is most spoken of, though it may be met seldom: a dragon in its mountain). Both are structure the
    world already keeps (a memory's perceiver and what it is about); the engine knows no kind, so it cannot say who is alive or who
    matters, and the overseer asks a model that of what is here. Up to `limit` from each ranking, no being twice. A being is an NPC
    when it is not playable and not a place. Linear in the memories, once: this is the overseer's call, never a player's. `sample`:
    three of what each holds (its most salient, then newest), for whoever must judge what it is. Owner only."""
    world = await adhisthana.get_world()
    k = world.kosha()
    by = world.samskara().mem.by_entity
    playable = set(world.playable_entity_ids)
    eligible = {eid for eid in world.canon.entities if eid not in playable and not k.is_scene(eid)}
    knowers: dict[str, set[str]] = {}
    for perceiver, mems in by.items():
        if k.is_scene(perceiver):
            continue                                              # a place's rows list who is present, which is no one's thought of them
        for m in mems:
            if m.tier == "AMBIENT":
                continue
            for sid in m.subject_ids:
                if sid != perceiver and sid in eligible:
                    knowers.setdefault(sid, set()).add(perceiver)
    n = max(1, min(limit, 50))
    held = lambda eid: len(by.get(eid, ()))
    known = lambda eid: len(knowers.get(eid, ()))
    chosen: list[str] = []
    for key in (held, known):
        for eid in sorted((e for e in eligible if key(e)), key=lambda e: (-key(e), -held(e), e))[:n]:
            if eid not in chosen:
                chosen.append(eid)
    out = []
    for eid in chosen:
        row = {"id": eid, "name": (world.canon.entities[eid].labels or [eid])[0], "memories": held(eid), "known_by": known(eid)}
        if sample:
            own = sorted(by.get(eid, ()), key=lambda m: (m.salience, -m.world_time))[:3]     # salience 1 is the most salient
            row["sample"] = [" ".join(m.experience.split())[:200] for m in own]
        out.append(row)
    return out


@router.get("/world/cast")
async def world_cast() -> dict:
    """The world's setting and every being no one plays, by name (Marc, 2026-10-06: the overseer picks out "the important" ones, the
    dragon, the head evil cleric, which neither memory nor being spoken of finds: they may be met seldom and named seldom). The cast is
    what a model must read to say who matters, so it is all of them, a name and a memory count each: bounded by the world's own size,
    read when the overseer asks, never by a player's page. A being is an NPC when it is not playable and not a place. Owner only."""
    world = await adhisthana.get_world()
    k = world.kosha()
    by = world.samskara().mem.by_entity
    playable = set(world.playable_entity_ids)
    beings = [{"id": eid, "name": (e.labels or [eid])[0], "memories": len(by.get(eid, ()))}
              for eid, e in world.canon.entities.items() if eid not in playable and not k.is_scene(eid)]
    return {"setting": (world.setting_description or "")[:900], "beings": sorted(beings, key=lambda b: (-b["memories"], b["id"]))}


@router.post("/world/playable/{eid}", response_model=PlayerOut)
async def make_playable(eid: str) -> PlayerOut:
    """A being the world already holds becomes playable (Marc, 2026-09-28: Elmo, played by an agent from his own
    page, as Dunstan is). Until someone plays them they are logged off and waiting (v5 §37, v4.58): in the world,
    carried, voiced, answering when touched, no turn of their own -- and no round waits on them. Their first
    presence brings them in; thirty minutes without acting leaves them waiting again. Owner only."""
    world = await adhisthana.get_world()
    k = world.kosha()
    if eid not in world.canon.entities or k.is_scene(eid):
        raise HTTPException(404, f"no being {eid} to play")
    if eid not in world.playable_entity_ids:
        world.playable_entity_ids = list(world.playable_entity_ids) + [eid]
        world.autonomous.add(eid)
        world.waiting.add(eid)
        world.canon.entities[eid].last_simulated_tick = world.canon.world_time  # quiet from now: a touch wakes them
        world.logged_off.add(eid)
        store = _store(world)
        await store.commit(world.canon, [], [], touched_ids=[eid])
        await store.save_waiting(sorted(world.waiting))
        await store.save_session_config(world.scene_owner_entity_id, world.entry_point_location_ids,
                                        world.playable_entity_ids, world.setting_description)
        await store.save_autonomy(sorted(world.autonomous))
        await store.save_logged_off(sorted(world.logged_off))
        world._samskara = None  # the players are part of its reading
        log_event("player.made_playable", entity_id=eid, label=world.canon.entities[eid].labels[0])
    return _player_out(world, eid)


@router.post("/players/{pid}/autonomy", response_model=PlayerOut)
async def set_autonomy(pid: str, req: AutonomyRequest) -> PlayerOut:
    """v5 §37: a player away is frozen (out of the world) or waiting (Marc,
    2026-10-01): in the world, seen, answering in others' turns, and taking
    no turn of their own until someone touches them. Allowed is waiting --
    never the world acting for a player on its own (v4.58, Marc, 2026-10-02:
    "remove autonomous for human players and only allow frozen or waiting");
    `wait` is kept for old pages and means nothing more. Persisted."""
    world = await adhisthana.get_world()
    _require_player(world, pid)
    was_absent = pid in world.absent()
    (world.autonomous.add if req.allow else world.autonomous.discard)(pid)
    store = _store(world)
    if req.allow:
        if pid not in world.waiting:
            world.waiting.add(pid)
            e = world.canon.entities.get(pid)
            if e is not None:  # quiet from now: only a touch after this wakes them
                e.last_simulated_tick = world.canon.world_time
                await store.commit(world.canon, [], [], touched_ids=[pid])
    else:
        world.waiting.discard(pid)
    if was_absent and pid not in world.absent():
        world.returning.add(pid)  # logged off, and now the world acts for them: they come back into it
    await store.save_autonomy(sorted(world.autonomous))
    await store.save_waiting(sorted(world.waiting))
    log_event("player.autonomy", entity_id=pid, allow=req.allow, wait=pid in world.waiting)
    return _player_out(world, pid)


@router.post("/players/{pid}/presence", response_model=PlayerOut)
async def set_presence(pid: str, req: PresenceRequest) -> PlayerOut:
    """v5 §37: a player leaving or returning, said explicitly. Logged off and
    not allowing the world to act for them, they are out of the world --
    in no scene, perceiving nothing (Marc, 2026-09-24; it was "suspended
    but present"); allowing it, they stay in the world (waiting: no turn of
    their own until touched, v4.58). Back online, their next pulse narrates
    their return."""
    world = await adhisthana.get_world()
    _require_player(world, pid)
    world.presence[pid] = "online" if req.online else "offline"
    if req.online:
        world.last_seen[pid] = time.monotonic()
        back = world.come_back(pid)
    else:
        world.logged_off.add(pid)  # Marc, 2026-09-24: logged off, and frozen, is out of the world
        back = True
    if back:
        await _store(world).save_logged_off(sorted(world.logged_off))
    log_event("player.presence", entity_id=pid, online=req.online, absent=pid in world.absent())
    return _player_out(world, pid)


@router.post("/players/{pid}/heartbeat", status_code=204)
async def heartbeat(pid: str, request: Request = None) -> None:
    """v5 §37: someone is at the page -- it is on screen and was touched (a key, a tap, a scroll, a click) since the
    last heartbeat, which the page sends every few minutes.
    Its name: sānnidhya (सान्निध्य), presence, nearness (v5 §37).
    It keeps an online player online without acting, so one
    may wait for another to catch up (Filonius in the moathouse courtyard, waiting on Jarun, 2026-10-03). An open tab
    is still not a player: a page nobody touches sends nothing, and thirty minutes after the last touch they are away
    as before. It only keeps a player online; it never brings back one logged off -- a deliberate log-off holds until
    they act or come back online."""
    world = await adhisthana.get_world()
    _require_player(world, pid)
    if request is not None:
        _require_own(request, world, pid)
    if pid in world.logged_off:
        log_event("player.heartbeat_ignored", entity_id=pid, why="logged_off")
        return
    world.last_seen[pid] = time.monotonic()
    world.presence[pid] = "online"
    log_event("player.heartbeat", entity_id=pid)


@router.post("/players/{pid}/retire", status_code=204)
async def retire_player(pid: str, req: RetireRequest, request: Request = None) -> None:
    """Retiring a character (Marc, 2026-10-01): the player cuts the tether, and
    cannot take it up again.
    Its name: saṃnyāsa (संन्यास), renunciation (v5 §42).
    The being stays in the world as anyone does -- off
    the players, drawn only when the world draws it; quiet, it sleeps (dormancy)
    until something handles it. Death is a special case: the engine knows no
    "dead" (s0), only tethered or not -- a body is a retired being whose own
    memories say it died. Only the player (or an owner) retires; the name typed
    back is the confirmation. Refused while the character is in an open round
    or a telling held for consent. The world's own record is untouched: no
    memory, tuple or containment is written. Cost: the player sets, once."""
    world = await adhisthana.get_world()
    _require_player(world, pid)
    name = _labelled(world.canon, pid)
    if req.confirm.strip().lower() != name.strip().lower():
        raise HTTPException(422, f"type the character's name, {name!r}, to retire them")
    if pid in world.held or any(pid in (r.get("actions") or {}) for r in world.rounds.values()):
        raise HTTPException(409, f"{name} is in a round still open; retire them once it is told")
    # out of every player set first: a stale logged-off entry would hold the body out of the world (absent())
    world.logged_off.discard(pid)
    world.autonomous.discard(pid)
    world.waiting.discard(pid)
    world.returning.discard(pid)
    for d in (world.presence, world.last_seen, world.round_told, world.typed, world.first_suggestion):
        d.pop(pid, None)
    world.played_by.pop(pid, None)
    world.playable_entity_ids = [p for p in world.playable_entity_ids if p != pid]
    if world.scene_owner_entity_id == pid:  # "" is a world nobody plays yet, never re-crowned on load
        world.scene_owner_entity_id = world.playable_entity_ids[0] if world.playable_entity_ids else ""
    store = _store(world)
    await store.save_logged_off(sorted(world.logged_off))
    await store.save_autonomy(sorted(world.autonomous))
    await store.save_waiting(sorted(world.waiting))
    await store.save_played_by(world.played_by)
    await store.save_session_config(world.scene_owner_entity_id, world.entry_point_location_ids,
                                    world.playable_entity_ids, world.setting_description)
    guest = dvara._guest(request) if request is not None else None
    if guest is not None and not dvara._god_of_active(guest):
        # a guest's retirement is held back until a god reviews it (Marc, 2026-10-02): out of the world till then
        world.held_back.add(pid)
        await store.save_held_back(sorted(world.held_back))
        # Marc, 2026-10-02: "make sure to alert me if a guest retires a character!" -- one push, as a guest online does
        entry = adhisthana._active_world_registry_entry()
        wname = entry.name if entry is not None else "Smritilila"
        dvara._schedule_alert(f"{guest} retired {name} on {wname}", f"{name} is held back until you review it (GET /world/held).",
                        getattr(entry, "url", None))
    world._samskara = None  # the players are part of its reading
    log_event("player.retired", entity_id=pid, label=name, world_time=world.canon.world_time, held_back=pid in world.held_back)


@router.get("/me", response_model=MeOut)
async def who_am_i(request: Request) -> MeOut:
    """Who the page is for: an owner, or a guest -- the admin link and the new-world button are the owner's (and a builder's), and a guest's intro is shown once."""
    guest = dvara._guest(request)
    return MeOut(owner=guest is None, builder=guest is None or users_registry.may_build(guest))
