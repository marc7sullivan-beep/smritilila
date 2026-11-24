# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Spanda's routes (2026-10-05, Marc's sign-off; his name for the pulse): the requests a pulse arrives by. POST /pulses with
its retry joined to the action it repeats and the request's own checks (an action that says something, a playable character,
the player's own), the round an action joins when another player stands in the scene; consent (/pulses/confirm, the held jumps
and their release); and the rounds' routes (/rounds/{player}: its state, edit, typing, Go).

What a pulse does once it is accepted -- the submit, the long jump's consent and the die that may interrupt it, the commit that
adopts into the live world what the pulse bound or minted, the births, the witnessing queued for each player, the world's draw
and activations after a commit, phala's reckoning and the writing of a round -- is smritilila/spanda.py's (moved there 2026-11-10,
nothing inside changed). It is called through that module (spanda.X), never imported bare, so a test's patch there reaches it;
this module binds none of it, so a patch left here fails loudly instead of faking nothing.
"""

from __future__ import annotations

import asyncio
import re
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila.adhisthana import WorldSession, store_of, app
from smritilila.sangama import (
    decline_round, go_in, round_acting, round_held, round_key, round_members, round_state_of,
)
from smritilila.training_data import log_discarded
from smritilila.sakshi import log_event
from smritilila.schemas import ConfirmRequest, HeldOut, PulseRequest, PulseResponse, RoundOut, labelled, told_nothing
from smritilila.dvara import require_own, require_player
from smritilila import spanda  # the pulse's services, called through it so a test's patch there reaches them


router = APIRouter()


@router.get("/world/held", response_model=list[HeldOut])
async def list_held() -> list[HeldOut]:
    """A guest's retired characters waiting for review (Marc, 2026-10-02): who they are and what holds for them.
    Owner and the world's gods."""
    world = await adhisthana.get_world()
    k = world.kosha()
    out = []
    for eid in sorted(world.held_back):
        own = world.samskara().mem.by_entity.get(eid, [])
        out.append(HeldOut(entity_id=eid, label=labelled(world.canon, eid),
                           clauses=[text for _b, _k, text in k.view_of(eid, world.canon.world_time + 1)],
                           first=[m.experience for m in own[:4]]))
    return out


@router.post("/world/held/{eid}/release", status_code=204)
async def release_held(eid: str) -> None:
    """A god releases a held-back character into the world: from now it is the world's, as any retired being is.
    Its name: pratīkṣā (प्रतीक्षा), waiting, expectation (v5 §42)."""
    world = await adhisthana.get_world()
    if eid not in world.held_back:
        raise HTTPException(404, f"{eid} is not held back")
    world.held_back.discard(eid)
    await store_of(world).save_held_back(sorted(world.held_back))
    world._samskara = None
    log_event("player.released", entity_id=eid, world_time=world.canon.world_time)


@router.post("/pulses/confirm", response_model=PulseResponse)
async def confirm_pulse(req: ConfirmRequest, background_tasks: BackgroundTasks, request: Request = None) -> PulseResponse:
    """v5 §22: the player's answer to a held pulse.
    Its name: anumati (अनुमति), assent (v5 §22).
    No: nothing happens, and the held answer's training example is tagged discarded. Yes to a long
    jump: it is committed if nothing in the world was written meanwhile, else
    run again (consented, so it does not ask twice). A player is only ever
    held for a long jump: since v4.27 a player is never asked to wait for a
    closed scene (v5 §21), and the lock binds only the world's own turns,
    which are let go."""
    world = await adhisthana.get_world()
    require_own(request, world, req.entity_id)
    held = world.held.get(req.entity_id)
    if held is None:
        raise HTTPException(409, "nothing is waiting on this character")
    if "round" in held:
        # saṅgama: the jump is everyone's who acted. One no ends the round (nothing is written, and each
        # acts again); the pulse goes on once all have said yes -- answered by whoever says it last
        rnd = held["round"]
        everyone = round_acting(rnd, held["actor"])
        if not req.proceed:
            decline_round(world, held, req.entity_id)
            return told_nothing()
        held["yes"].add(req.entity_id)
        log_event("sangama.agreed", pulse_id=held["pulse_id"], entity_id=req.entity_id, waiting=sorted(everyone - held["yes"]))
        if held["yes"] != everyone:
            return told_nothing(round=round_state_of(world, rnd["scene"], req.entity_id).model_dump())
        for p in everyone:
            world.held.pop(p, None)
        return await spanda.confirmed(world, held, held["actor"], background_tasks, runner=req.entity_id)
    world.held.pop(req.entity_id, None)
    if not req.proceed:
        log_discarded(pulse_id=held["pulse_id"], reason="declined", detail=held["kind"])
        log_event("smrti.declined", pulse_id=held["pulse_id"], entity_id=req.entity_id, kind=held["kind"])
        return told_nothing()
    return await spanda.confirmed(world, held, req.entity_id, background_tasks)


@router.get("/rounds/{pid}", response_model=RoundOut)
async def round_state(pid: str) -> RoundOut:
    """The round where this player stands, as their page sees it -- and,
    once written, their own telling (handed over once)."""
    world = await adhisthana.get_world()
    require_player(world, pid)
    told = world.round_told.pop(pid, None)
    out = round_state_of(world, round_key(world, pid), pid)
    if told is not None and told.round and told.round.get("declined_by"):
        out.declined_by = told.round["declined_by"]
        told = None
    out.told = told
    return out


@router.post("/rounds/{pid}/edit", response_model=RoundOut)
async def round_edit(pid: str) -> RoundOut:
    """They are changing what they do: their action is out of the round until
    they act again (Marc: editing puts you back in the not-acted state)."""
    world = await adhisthana.get_world()
    require_player(world, pid)
    scene = round_key(world, pid)
    r = world.rounds.get(scene or "")
    if r is not None and not r["running"] and not round_held(world, scene) and pid in r["actions"]:
        del r["actions"][pid]
        r["order"].remove(pid)
        if not r["actions"]:
            world.rounds.pop(scene, None)
        log_event("sangama.edit", scene=scene, entity_id=pid)
    return round_state_of(world, scene, pid)


@router.post("/rounds/{pid}/typing", status_code=204)
async def round_typing(pid: str) -> None:
    """Their page: they are writing an action (sent at most every few
    seconds while they type). Go waits GO_QUIET seconds after the last."""
    world = await adhisthana.get_world()
    require_player(world, pid)
    world.typed[pid] = time.monotonic()


@router.post("/rounds/{pid}/go", response_model=PulseResponse)
async def round_go(pid: str, background_tasks: BackgroundTasks) -> PulseResponse:
    """Go: one who has acted closes the round, and it is written now."""
    start = time.monotonic()
    world = await adhisthana.get_world()
    require_player(world, pid)
    scene = round_key(world, pid)
    r = world.rounds.get(scene or "")
    # Go a step behind the round is not an error (Marc saw 409s, 2026-09-29): the page polls, so its Go can land after
    # the last action already wrote the round, while it is written, while it waits on a yes, or before the others have
    # stopped typing. It is answered with the round as it stands -- nothing run -- and the page's poll collects the telling.
    late = r is None or pid not in r["actions"] or r.get("running") or round_held(world, scene)
    wait = 0.0 if late else go_in(world, scene, round_members(world, scene, pid), r["actions"])
    if late or wait > 0:
        log_event("sangama.go_early" if wait > 0 else "sangama.go_late", scene=scene, entity_id=pid, go_in=wait)
        return told_nothing(round=round_state_of(world, scene, pid).model_dump())
    world.last_seen[pid] = time.monotonic()
    return await spanda.run_round(world, adhisthana.resolve_roles(), background_tasks, scene, pid, start)


CLIENT_PULSES_KEPT = 64  # recent action ids remembered (bounded: the world's players' last few actions, never history)


@router.post("/pulses", response_model=PulseResponse)
async def submit_pulse(req: PulseRequest, background_tasks: BackgroundTasks, request: Request = None) -> PulseResponse:
    """An action, once: a retry (the same page id) of one already received
    waits for it and gets its answer -- it is never run twice."""
    if not req.client_id:
        return await _submit_pulse_once(req, background_tasks, request)
    key = f"{req.entity_id}:{req.client_id}"
    seen = world_now().client_pulses.get(key) if world_now() is not None else None
    if seen is not None:
        log_event("pulse.retry_joined", entity_id=req.entity_id, client_id=req.client_id, done=seen.done())
        return await asyncio.shield(seen)
    fut = asyncio.get_running_loop().create_future()
    world = world_now()
    if world is not None:
        world.client_pulses[key] = fut
        while len(world.client_pulses) > CLIENT_PULSES_KEPT:
            world.client_pulses.pop(next(iter(world.client_pulses)))
    try:
        resp = await _submit_pulse_once(req, background_tasks, request)
    except BaseException as exc:
        if world is not None:
            world.client_pulses.pop(key, None)  # it did not happen: a retry may try again
        if not fut.done():
            fut.set_exception(exc)
            fut.exception()  # retrieved: never "exception was never retrieved"
        raise
    fut.set_result(resp)
    return resp


def world_now() -> "WorldSession | None":
    """The live world if one is loaded (never loads one)."""
    return getattr(app.state, "world", None)


async def _submit_pulse_once(req: PulseRequest, background_tasks: BackgroundTasks, request: Request = None) -> PulseResponse:
    # End-to-end "action to narration" wall-clock time -- live testing,
    # 2026-09-07: no telemetry existed anywhere to say whether a slow
    # response was the player's own pulse, the scheduler's own activation
    # pulse tacked on after it, or something else. Now that the
    # activation pulse runs as a background task (_after_smrti_commit), this covers
    # only the player's own pulse -- exactly what "action to narration"
    # should mean, no longer conflated with unrelated background work.
    request_start = time.monotonic()
    # Ravenmoor 508: an action of "{" ran a whole pulse, and the Narrator
    # made up what was done. An action says something: a letter or digit,
    # in any script. Structure, not meaning -- checked before any model call.
    if not re.search(r"\w", req.action_text):
        raise HTTPException(422, "say what you do")
    world = await adhisthana.get_world()  # may raise if the store's database is unreachable
    acting_entity_id = req.entity_id or world.scene_owner_entity_id
    if acting_entity_id not in world.playable_entity_ids:
        raise HTTPException(422, f"entity_id must be one of {world.playable_entity_ids}")
    require_own(request, world, acting_entity_id)
    roles = adhisthana.resolve_roles()  # may raise 503 -- deliberately after validation
    # saṅgama: where another player stands online in the scene, or a round is open there, the action joins it
    held = world.held.get(acting_entity_id)
    if held is not None and "round" in held:
        decline_round(world, held, acting_entity_id)  # v5 §36: a new action answers the waiting question with no
    if world.come_back(acting_entity_id):  # acting is being here (v5 §37): back in the world, and so in its round
        await store_of(world).save_logged_off(sorted(world.logged_off))
    # v5 §36 v4.31: within reach (the container directly above) and at the same moment -- a player behind or
    # ahead of an open round acts alone until their span meets its present
    scene = round_key(world, acting_entity_id)
    level = set(round_members(world, scene, acting_entity_id))
    if scene is not None and acting_entity_id not in world.held and (
            level - {acting_entity_id} or (scene in world.rounds and level & set(world.rounds[scene]["actions"]))):
        return await spanda.round_submit(world, roles, background_tasks, scene, acting_entity_id, req, request_start)
    # v3: the pulse is retrieve -> narrate -> append and nothing else.
    return await spanda.submit_smrti_pulse(req, world, acting_entity_id, roles, background_tasks, request_start)
