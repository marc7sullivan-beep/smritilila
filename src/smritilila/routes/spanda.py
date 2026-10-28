# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Spanda (2026-10-05, Marc's sign-off; his name for the pulse): the
routes a pulse runs through and what follows it. POST /pulses and its
one-at-a-time submit, the commit that adopts what the live world moved
on to meanwhile, the witnessing queued for each player, consent
(/pulses/confirm, the held jumps and their release) with the die that may
interrupt a long jump, the rounds' routes, the world's activation after a commit
and phala's reckoning.

Moved whole from main.py, nothing inside changed. The model call is
run_smrti_pulse, imported here and called only from here: a test fakes
it on this module (spanda.run_smrti_pulse), as it did on main. Names a
test patches that other modules call (_players_latest) are called there
as spanda.X.

From main.py's docstring, moved here with the code it explains (2026-10-05):

Appendix A's world-tick scheduler runs alongside every /pulses call: it
picks one entity (the oldest-unsimulated from the exploration
pool) synchronously, then grants it ACTIVATION_PULSE_BUDGET
(1, scheduler/episode.py) autonomous pulses of its own. Execution is
_run_smrti_activation_in_background -- the same v3 pulse with no action
supplied -- and it runs as a FastAPI BackgroundTask, AFTER the response
that triggered its selection has already been sent (a real, measured
15.2s it used to add to every single response otherwise). What a player
sees of it is what they witnessed, decided as it is written (_queue_witnessed;
until 2026-09-30 its narration also waited on the world for whoever pulsed
next, which item 6 removed). Unlike v2, an activation is never dropped when the player's
own pulse takes its tick: two pulses in one tick are two things that
happened in the same tick, so the result is folded in instead (see
_run_smrti_activation_in_background's own docstring, and the Vadhu
tick-110 incident it records).
"""

from __future__ import annotations

import asyncio
import dataclasses
import random
import re
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from smritilila.graph.canon import Maya
from smritilila.ids import new_id
from smritilila.scheduler.episode import advance_world_tick
from smritilila.llm.client import LLMClient, best_available_client, bind_pulse_id
from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila.adhisthana import GenerativeRoles, WorldSession, _extend_history, _fold_in, _store, app
from smritilila.pravrtti import (
    _fold_absorbed, _hold_for_away, _hold_for_rounds, _kala_gate, _log_pool, _paused,
    _smrti_exploration_pool,
)
from smritilila.sangama import (
    _decline_round, _go_in, _round_acting, _round_held, _round_key, _round_members,
    _round_state, confirm_of,
)
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.context import NOWHERE
from smritilila.smrti.pulse import SmrtiPulseResult, fortune_seed, run_smrti_pulse, split_heard
from smritilila.smrti.genesis import ITIHASA
from smritilila.smrti.log import v3_enclosure
from smritilila.smrti.kala import apart, interruption_chance, names_any, watch_phrases
from smritilila.training_data import log_discarded
from smritilila.sakshi import log_event
from smritilila.itihasa import _age_at, _age_clock
from smritilila.schemas import ConfirmRequest, HeldOut, PulseRequest, PulseResponse, RoundOut, _labelled, _told_nothing
from smritilila.dvara import _require_own, _require_player


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
        out.append(HeldOut(entity_id=eid, label=_labelled(world.canon, eid),
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
    await _store(world).save_held_back(sorted(world.held_back))
    world._samskara = None
    log_event("player.released", entity_id=eid, world_time=world.canon.world_time)


async def _submit_smrti_pulse(
    req: PulseRequest, world: WorldSession, acting_entity_id: str, roles: GenerativeRoles,
    background_tasks: BackgroundTasks, request_start: float, confirmed: bool = False, wait_until: int | None = None,
    window: list[tuple[int, str, str]] | None = None, sangama: dict | None = None, runner: str | None = None,
    setting: list[tuple[str, str]] | None = None, interrupt: tuple[int, str] | None = None,
) -> PulseResponse:
    if roles.smrti_client is None:
        raise HTTPException(503, "no LLM client for the v3 Narrator")
    # v5 §35: the page's own words, sent back unchanged, are an accepted
    # suggestion; anything else the player typed
    action_source = "suggested" if req.action_text.strip() in world.offered.get(acting_entity_id, set()) else "typed"
    world.presence[acting_entity_id] = "online"  # v5 §37: acting is being here
    world.last_seen[acting_entity_id] = time.monotonic()
    if world.come_back(acting_entity_id):
        await _store(world).save_logged_off(sorted(world.logged_off))
    returning = world.left_at(acting_entity_id) if acting_entity_id in world.returning else None
    stale = world.held.pop(acting_entity_id, None)
    if stale is not None:  # v5 §22: a new action answers a waiting question with no
        log_discarded(pulse_id=stale["pulse_id"], reason="declined", detail=stale["kind"])
    # v5 §33: a turn in flight in the player's scene commits first, and
    # this pulse reads what it wrote
    async with world.in_scene(world.kosha().scene_of(acting_entity_id, world.canon.world_time + 1), "player", acting_entity_id):
        working_canon = world.canon.fork()
        working_canon.world_time += 1
        pulse_id = new_id()
        log_event("pulse.player_action", kind="pulse", pulse_id=pulse_id, entity_id=acting_entity_id,
                  world_time=working_canon.world_time, action_text=req.action_text, smrti=True, action_source=action_source)
        bind_pulse_id(pulse_id)
        result = await asyncio.to_thread(
            run_smrti_pulse, working_canon, acting_entity_id, req.action_text, working_canon.world_time, pulse_id,
            list(world.recent_memories), roles.smrti_client, adhisthana.get_embedder().embed, players=world.unvoiced(),
            records=list(world.records), action_source=action_source, movable=world.movable(),
            playable=frozenset(world.playable_entity_ids), view=world.view(),
            absent=world.absent(), returning=returning, confirmed=confirmed, wait_until=wait_until, window=window, setting=setting, interrupt=interrupt,
            samskara=world.samskara(),
            sangama=dict(sangama["others"]) if sangama else None,
            sangama_sources=sangama["sources"] if sangama else None,
        )
        if result.pending:
            # v5 §22: a jump the player must consent to -- nothing is written yet
            held = {"req": req, "result": result, "working_canon": working_canon, "pulse_id": pulse_id,
                    "scene": world.kosha().scene_of(acting_entity_id, world.canon.world_time + 1),
                    "tick": world.canon.world_time, **result.pending}
            world.held[acting_entity_id] = held
            if sangama:
                # saṅgama: the jump is everyone's who acted -- each must agree (one held, under each of them)
                held.update(round=sangama, actor=acting_entity_id, yes=set())
                for p in _round_acting(sangama, acting_entity_id):
                    world.held[p] = held
            log_event("smrti.held", pulse_id=pulse_id, entity_id=acting_entity_id, kind=result.pending["kind"],
                      prompt=result.pending["prompt"], round=sorted(_round_acting(sangama, acting_entity_id)) if sangama else None)
            if sangama:
                return _told_nothing(confirm=confirm_of(result.pending), round=_round_state(world, sangama["scene"], runner).model_dump())
            return _told_nothing(confirm=confirm_of(result.pending))
        world.returning.discard(acting_entity_id)
        await _commit_smrti_result(world, working_canon, result, acting_entity_id, pulse_id,
                                   **({"round_scene": sangama["scene"]} if sangama else {}))
    return _after_smrti_commit(world, roles, background_tasks, result, acting_entity_id, pulse_id, request_start, runner=runner,
                               defaulted=(sangama or {}).get("defaulted"))


def _adopt(world: WorldSession, working: Maya, result: SmrtiPulseResult, actor: str) -> None:
    """A committed pulse's entities, into the live world: what it bound or
    minted (exactly what the store wrote) replaces the live copy; everything
    else stays as the live world has it. The pulse's copy was taken when it
    began, and meanwhile turns in other scenes commit and the scheduler moves
    its bookkeeping on in place (last_simulated_tick): swapping the copy in
    rolled those back in memory while the store kept them, and a drawn NPC
    was "oldest" again. Witnessed: which live entities the old swap would
    have rolled back (one pass over the canon, as the pulse's own copy is)."""
    mine = {*result.bound_ids, actor, *(e.id for e in result.new_entities)}
    kept = [eid for eid, ent in world.canon.entities.items()
            if eid not in mine and eid in working.entities and working.entities[eid] != ent]
    if kept:
        log_event("smrti.adopt_kept_live", actor=actor, count=len(kept), ids=kept[:10],
                  overtaken=world.canon.world_time + 1 != working.world_time)
    _fold_in(world, [working.entities[eid] for eid in {*result.bound_ids, actor} if eid in working.entities] + list(result.new_entities),
             working.world_time)


async def _commit_smrti_result(world: WorldSession, working_canon: Maya, result: SmrtiPulseResult, acting_entity_id: str,
                               pulse_id: str, round_scene: str | None = None) -> None:
    """A player's pulse, committed: to the store, then the live world. A
    round's (saṅgama, `round_scene`): the round is over once it is written."""
    _commit_start = time.monotonic()
    try:
        await _store(world).commit(working_canon, result.memories_written, result.new_entities,
                                       touched_ids=list(result.bound_ids) + [acting_entity_id], records=result.records_written)
    except Exception as exc:
        log_event("commit_pulse.failed", source="smrti_pulse", pulse_id=pulse_id, error=repr(exc))
        raise
    _adopt(world, working_canon, result, acting_entity_id)
    _extend_history(world, result.memories_written, result.records_written)
    # another player in the scene sees this turn as it happens (Stonemarch, Marc and the first model playing together:
    # Dunstan's offer reached Jakal's memory and never his page)
    # (a round's players have each their own telling)
    _queue_witnessed(world, result.memories_written, exclude=frozenset({acting_entity_id}) | frozenset(result.round_narrations))
    world.offered[acting_entity_id] = {x.strip() for x in (result.suggested_next_action, result.suggested_next_action_from_memory) if x and x.strip()}
    world.first_suggestion[acting_entity_id] = (result.suggested_next_action or "").strip()
    for pid, told in result.round_narrations.items():
        world.offered[pid] = {told["suggested_next_action"].strip()} if (told.get("suggested_next_action") or "").strip() else set()
        world.first_suggestion[pid] = (told.get("suggested_next_action") or "").strip()
    if round_scene is not None:
        world.rounds.pop(round_scene, None)
        world.returning.difference_update(result.round_narrations)  # the round told them here
        log_event("sangama.written", pulse_id=pulse_id, scene=round_scene, actor=acting_entity_id,
                  players=sorted(result.round_narrations))
    log_event("commit_pulse.timing", source="smrti_pulse", pulse_id=pulse_id,
              duration_ms=int((time.monotonic() - _commit_start) * 1000), committed_delta_count=0,
              records=len(result.records_written))


def _after_smrti_commit(world: WorldSession, roles: GenerativeRoles, background_tasks: BackgroundTasks, result: SmrtiPulseResult,
                        acting_entity_id: str, pulse_id: str, request_start: float, runner: str | None = None,
                        defaulted: dict[str, str] | None = None) -> PulseResponse:
    """After a player's pulse is committed: the scheduler's turn, and the
    answer. A round's (saṅgama): each of its players is told it from where
    they stand; `runner`, whose request ran it, gets theirs as the answer,
    and the rest collect theirs at GET /rounds/{player}."""
    # the actor's own telling; the world's turns reach a player only as what they witnessed, decided when each
    # was written (item 6, 2026-09-30 -- they had waited on the world for whoever next pulsed near the doer)
    witnessed = world.pending_witnessed.pop(acting_entity_id, [])
    all_narrations = {acting_entity_id: result.narration}
    all_promoted = [e.id for e in result.new_entities]

    pool, turns = _kala_gate(world, _hold_for_rounds(world, _hold_for_away(world, _fold_absorbed(world, _smrti_exploration_pool(world)))))
    _log_pool(world, pool)
    tick_result = advance_world_tick(world.canon, world.canon.world_time, pool)
    if tick_result.pulses_run and tick_result.activated_entity_id is not None:
        begin, cap = turns.get(tick_result.activated_entity_id, (None, None))
        background_tasks.add_task(_run_smrti_activation_in_background, world, roles, tick_result.activated_entity_id,
                                  tick_result.pulses_run, begin=begin, cap=cap)
    log_event("request.timing", pulse_id=pulse_id, duration_ms=int((time.monotonic() - request_start) * 1000),
              activation_pulses_run=tick_result.pulses_run, smrti=True)
    answer = PulseResponse(
        narrations=all_narrations, promoted_entity_ids=all_promoted,
        suggested_next_action=result.suggested_next_action,
        suggested_next_action_from_memory=result.suggested_next_action_from_memory,
        witnessed={acting_entity_id: witnessed} if witnessed else {},
        sara={acting_entity_id: result.sara} if result.sara else {},
        sara_detail={acting_entity_id: {"said": result.said, "turn": result.turn,
                                        # the actor's own weight this pulse: the UI opened a marking or
                                        # indelible moment whole and folded anything lighter (Marc, 2026-09-19); since 2026-10-19 the page shows every narration whole and reads no weight
                                        "weight": next((m.salience for m in result.memories_written if m.entity_id == acting_entity_id), None)}},
    )
    if _phala_due(world, result):
        background_tasks.add_task(_reckon, world, [acting_entity_id] + list(result.round_narrations), result.scene)
    if not result.round_narrations:
        return answer
    told = {acting_entity_id: answer}
    for pid, t in result.round_narrations.items():
        w = world.pending_witnessed.pop(pid, [])
        told[pid] = PulseResponse(
            narrations={pid: t["narration"]}, promoted_entity_ids=[e.id for e in result.new_entities],
            suggested_next_action=t.get("suggested_next_action") or "", suggested_next_action_from_memory="",
            witnessed={pid: w} if w else {},
            round={"defaulted": defaulted[pid]} if (defaulted or {}).get(pid) else None,
            sara_detail={pid: {"said": "", "turn": "",
                               "weight": next((m.salience for m in result.memories_written if m.entity_id == pid), None)}})
    runner = runner if runner in told else acting_entity_id
    for pid, resp in told.items():
        if pid != runner:
            world.round_told[pid] = resp
    return told[runner]


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
    _require_own(request, world, req.entity_id)
    held = world.held.get(req.entity_id)
    if held is None:
        raise HTTPException(409, "nothing is waiting on this character")
    if "round" in held:
        # saṅgama: the jump is everyone's who acted. One no ends the round (nothing is written, and each
        # acts again); the pulse goes on once all have said yes -- answered by whoever says it last
        rnd = held["round"]
        everyone = _round_acting(rnd, held["actor"])
        if not req.proceed:
            _decline_round(world, held, req.entity_id)
            return _told_nothing()
        held["yes"].add(req.entity_id)
        log_event("sangama.agreed", pulse_id=held["pulse_id"], entity_id=req.entity_id, waiting=sorted(everyone - held["yes"]))
        if held["yes"] != everyone:
            return _told_nothing(round=_round_state(world, rnd["scene"], req.entity_id).model_dump())
        for p in everyone:
            world.held.pop(p, None)
        return await _confirmed(world, held, held["actor"], background_tasks, runner=req.entity_id)
    world.held.pop(req.entity_id, None)
    if not req.proceed:
        log_discarded(pulse_id=held["pulse_id"], reason="declined", detail=held["kind"])
        log_event("smrti.declined", pulse_id=held["pulse_id"], entity_id=req.entity_id, kind=held["kind"])
        return _told_nothing()
    return await _confirmed(world, held, req.entity_id, background_tasks)


async def _confirmed(world: WorldSession, held: dict, actor: str, background_tasks: BackgroundTasks,
                     runner: str | None = None) -> PulseResponse:
    """v5 §22: a held pulse its player (a round's: all who acted) agreed to."""
    rnd = held.get("round")
    rkw = {"sangama": rnd, "runner": runner} if rnd else {}
    roles = adhisthana._resolve_roles()
    start = time.monotonic()
    if held["kind"] == "long" and held.get("to") is not None:
        # v5 §22 (Marc, 2026-10-27: "It's the wrong flow"): no scene takes a turn before the player lives a long jump. What next door already
        # RECORDED in its time stops it (no model call); else one die is rolled, its chance rising with the jump's length, and if it falls the
        # jump is told again, once, stopping at the minute the die drew, for the Narrator to say what interrupts. Before, up to three scenes were
        # asked one after another and the telling thrown away and told again: a median 30 s after Yes over 173 jumps, and in 89% of those
        # turns nothing found.
        events, setting = _recorded_events(world, actor, held)
        interrupt = None if events else _interruption(world, held, actor)
        if events or interrupt:
            # v4.32: the held telling was written blind to what is recorded in its time; it is told again, shown it (it stops at the first
            # that reaches the actor) and the world around it (the same sky, never a stop)
            log_discarded(pulse_id=held["pulse_id"], reason="window_rerun" if events else "interrupted",
                          detail=f"{len(events)} events, {len(setting)} rows" if events else f"minute {interrupt[0]}")
            if events:
                log_event("smrti.window_rerun", pulse_id=held["pulse_id"], entity_id=actor, events=len(events), setting=len(setting))
            return await _submit_smrti_pulse(held["req"], world, actor, roles, background_tasks, start, confirmed=True,
                                             window=events or None, setting=setting or None, interrupt=interrupt,
                                             wait_until=(rnd or {}).get("from"), **rkw)
        # nothing in the way: the pulse stands as lived, on the live world (its commit adopts only what it bound or minted)
        working = held["working_canon"]
        working.world_time = world.canon.world_time + 1
        # not yet written: it is committed after the window's turns, and carries the tick it lands at
        r = held["result"]
        r.memories_written = [dataclasses.replace(m, world_time=working.world_time) for m in r.memories_written]
        r.records_written = [dataclasses.replace(x, world_time=working.world_time) for x in r.records_written]
        held["tick"] = world.canon.world_time
    if held["kind"] == "long" and world.canon.world_time == held["tick"]:
        async with world.in_scene(held["scene"], "confirm", actor):
            await _commit_smrti_result(world, held["working_canon"], held["result"], actor, held["pulse_id"],
                                       **({"round_scene": rnd["scene"]} if rnd else {}))
        log_event("smrti.consented", pulse_id=held["pulse_id"], entity_id=actor, kind="long", rerun=False)
        return _after_smrti_commit(world, roles, background_tasks, held["result"], actor, held["pulse_id"], start, runner=runner,
                                   defaulted=(rnd or {}).get("defaulted"))
    log_discarded(pulse_id=held["pulse_id"], reason="consented_rerun", detail=held["kind"])
    log_event("smrti.consented", pulse_id=held["pulse_id"], entity_id=actor, kind=held["kind"], rerun=True)
    return await _submit_smrti_pulse(held["req"], world, actor, roles, background_tasks, start, confirmed=True,
                                     wait_until=(rnd or {}).get("from"), **rkw)


def _recorded_events(world: WorldSession, actor: str, held: dict) -> tuple[list[tuple[int, str, str]], list[tuple[str, str]]]:
    """v5 §22: what the scenes next to where a long jump ends already RECORDED in its time -- no model call. Returns (events, the world
    around): what HAPPENED that is not the actor's doing (a move, mark or line by someone the held pulse did not bind; a row naming what is
    the actor's, by the engine's own names and labels only: kala.watch_phrases), as (minute, place, its row), and the quiet rows, as
    (place, its row), whose weather, light and water the actor's telling shares. The scenes the actor attends to (the held pulse bound them)
    first; not the actor's own scenes (the actor's pulse narrates those). It was `_window_turns` before 2026-10-27, which ran up to three of
    those scenes' turns for the span on top of this; they are gone, and only the recorded is read.
    Cost: the neighbours' own rows and one pass over the records, per long jump (`_recorded_around`)."""
    kala = world.samskara().kala  # kept, read on with what was committed
    if kala is None:
        return [], []
    k = world.kosha()
    t = world.canon.world_time + 1
    dest = held.get("dest") or held.get("origin")
    theirs = {dest, held.get("origin")} | set(held["result"].route)  # the actor's route: every scene its boundaries joined
    mine = set(held["result"].bound_ids) | {e.id for e in held["result"].new_entities} | {actor}
    near = [x for x in k.neighbours(dest, t) if k.is_scene(x) and x not in theirs] if dest else []
    scenes = sorted(near, key=lambda x: x not in mine)  # what the actor attends to first (the far bank watched), then the rest
    with_them = {x for x in mine if not k.is_scene(x)}
    watched = watch_phrases(k, with_them | ({dest} if dest else set()))
    return _recorded_around(world, scenes, held["from"], held["to"], mine, watched)


def _interruption(world: WorldSession, held: dict, actor: str) -> tuple[int, str] | None:
    """v5 §22 (Marc, 2026-10-27: "a probability of an interrupting event as a function of the length of time of the jump. The event should
    occur randomly over the length of the jump. If the jump is interrupted then a single model call to narrate the interruption."): the die
    for a long jump its player agreed to. The chance is `kala.interruption_chance` of the jump's length at the world's mean time between
    interruptions (pacing `interruption_hours`, 96 by default); if it falls, the minute is drawn uniformly over the jump and the scene is
    the one the actor ends in (or stands in, if it goes nowhere). A scene closed to arrivals (pacing `arrivals` false, v5 §15) is never
    interrupted: nothing comes in. The die is the pulse's own, as the fortune is (`fortune_seed`: whose and when, never the attempt):
    declining and trying again meets the same one, and a longer jump is interrupted whenever a shorter one was. The engine says only that
    something does, and when: what it is, is the Narrator's (s0). Witnessed whatever the die gave: smrti.interruption_roll.
    Cost: one scene's pacing and a random draw -- no model call."""
    a, b = held["from"], held["to"]
    span = b - a
    scene = held.get("dest") or held.get("origin")
    mean = world.pace()["interruption_hours"]
    chance = interruption_chance(span, mean)
    closed = bool(scene) and world.kosha().seed("pacing", scene, world.canon.world_time + 1, field="arrivals") is False
    drawn = minute = None
    if chance > 0 and scene and not closed:
        rng = random.Random(fortune_seed(actor, held["working_canon"].world_time) + ":interruption")
        drawn = rng.random()
        if drawn < chance:
            minute = a + 1 + int(rng.random() * max(1, span - 1))
    log_event("smrti.interruption_roll", pulse_id=held["pulse_id"], entity_id=actor, minutes=span, mean_hours=mean, chance=round(chance, 4),
              drawn=None if drawn is None else round(drawn, 4), minute=minute, scene=scene, closed=closed)
    return (minute, scene) if minute is not None else None


def _recorded_around(world: WorldSession, scenes: list[str], a: int, b: int, mine: set[str],
                     watched) -> tuple[list[tuple[int, str, str]], list[tuple[str, str]]]:
    """What the scenes next door already RECORDED during (a, b], a long
    action's time -- told before the action, so no fresh window turn runs over
    it (a turn never narrates a span already lived), and until now shown by
    nothing: the mirror looks only in the actor's own scene. Stonemarch,
    2026-10-01: Aramus, a green dragon, circled Millhaven and roared on the green
    at 11:40-11:46, alarm horns sounding; Filonius waited under the Welcome
    Wench's eaves next door from 11:41 for three hours and saw only carters.
    Each of the scene's own rows is sorted by the fresh turns' rule (by
    structure, s0): its pulse moved, marked or voiced someone not the actor's,
    or it names what is theirs -> an event at the first minute of it in the
    action's time; else the world around, never a stop. Replayed on
    Filonius's call before building: shown so, both models stopped at the
    horns (the mid-size model 5/5, the second model 3/3; as told, 11/11 waited the three hours);
    every recorded row listed as an event instead, quiet ones cut 4 of 8
    waits short. Every neighbour, no model call. Cost: the neighbours' own
    rows and one pass over the records, per long action -- linear in their
    history (2026-10-01: at most 287 rows to a scene, 2,431 records)."""
    by_entity = world.samskara().mem.by_entity
    rows = [m for sid in scenes for m in by_entity.get(sid, ())
            if m.location_id == sid and m.experience and m.ended is not None and m.ended > a
            and (m.began if m.began is not None else m.ended) < b]
    if not rows:
        return [], []
    pulses = {m.pulse_id for m in rows}
    moved = {r.pulse_id for r in world.records if r.pulse_id in pulses
             and ((r.kind in ("kosha", "mark") and r.entity_id not in mine) or (r.kind == "said" and (r.note or "") not in mine))}
    events, around = [], []
    for m in rows:
        if m.pulse_id in moved or names_any(m.experience, watched):
            events.append((max(m.began if m.began is not None else m.ended, a), m.location_id, m.experience))
        else:
            around.append((m.location_id, m.experience))
    log_event("smrti.recorded_around", scenes=len(scenes), events=len(events), around=len(around))
    return sorted(events), around


def _phala_client() -> LLMClient:
    """The reckoner's model: always the primary (the first provider's PRIMARY_MODEL) -- one call per rest, judged whole (smrti/phala.py); a machine with no the first provider
    key uses the best model it has (a local one, else the second model)."""
    return best_available_client()


def _phala_due(world: WorldSession, result: SmrtiPulseResult) -> bool:
    """Phala: a world that asks (world data "phala") is reckoned after a player's pulse long enough to need consent."""
    if not (world.world_data.get("phala") or {}).get("ask") or result.elapsed is None:
        return False
    kala = world.samskara().kala
    return kala is not None and result.elapsed > kala.confirm_minutes


async def _reckon(world: WorldSession, players: list[str], scene: str | None) -> None:
    """Phala (smrti/phala.py): what the company lived since their last reckoning, judged at rest by one call, and
    what it earned written as their own becomings. Never narrated. Bounded: SPAN_CAP tuples, never the history."""
    from smritilila.llm.handles import HandleMap
    from smritilila.smrti import phala
    from smritilila.smrti.narrator import decode_handle
    from smritilila.smrti.narrator import own_view, source_line
    cfg = world.world_data.get("phala") or {}
    # each doer's own mark (phala.company_of): their latest `reckoned` record -- the world's own history
    # (2026-09-30; world data "phala_last" until then). Cost: the world's records, at a rest.
    marks = {r.entity_id: int(r.value) for r in world.records if r.kind == "reckoned" and r.value}
    now = world.canon.world_time
    sk = world.samskara()
    labels = {i: (e.labels[0] if e.labels else i) for i, e in world.canon.entities.items()}
    company = phala.company_of(players, scene, world.kosha(), sk.mem.by_entity, sk.mem.by_tick, marks, now, labels)
    since = company.since
    pulse_id = new_id()
    bind_pulse_id(pulse_id)
    if not company.span:
        return
    system = "\n\n".join(x for x in (source_line(world.world_data.get("source")), phala.SYSTEM,
                                       own_view(world.darsana) if world.darsana else "") if x)
    try:
        # one handle rule at every model boundary (llm/handles.py): the company's ids go out as handles and come back as ids
        handles = HandleMap.build(company.ids, {i: e.labels for i, e in world.canon.entities.items()})
        out = await asyncio.to_thread(_phala_client().complete_with_tool, system,
                                      handles.encode_text(phala.render(company, cfg["ask"], labels)),
                                      tool_name="reckon", tool_description="What the span yielded, as records.",
                                      input_schema=phala.SCHEMA, max_tokens=8000)
    except Exception as exc:  # noqa: BLE001 -- a reckoning missed is retried at the next rest; play goes on
        log_event("phala.failed", pulse_id=pulse_id, players=players, error=repr(exc)[:300])
        return
    for b in out.get("becomings") or []:
        if isinstance(b, dict) and isinstance(b.get("entity"), str) and b["entity"].strip():
            b["entity"] = decode_handle(b["entity"], handles)
    kept, refused = phala.accepted(out, company, cfg.get("keys"))
    t = world.canon.world_time
    recs = [Record(new_id(), "clause", b["entity"], b["clause"], t, pulse_id, note=f"self:{b['key']}") for b in kept]
    # every doer reckoned is marked: their deeds to now are reckoned, whoever's rest comes next
    recs += [Record(new_id(), "reckoned", x, str(now), t, pulse_id) for x in company.ids]
    await _store(world).commit(world.canon, [], [], records=recs)
    _extend_history(world, [], recs)
    log_event("phala.reckoned", pulse_id=pulse_id, players=players, since=since, now=now, company=company.ids,
              span=len(company.span), finished=[str(x)[:160] for x in out.get("finished") or []],
              written=[(b["entity"], b["key"]) for b in kept], refused=refused)


async def _run_smrti_activation_in_background(
    world: WorldSession, roles: GenerativeRoles, activated_entity_id: str, pulses_run: int, *,
    begin: int | None = None, cap: int | None = None,
) -> None:
    """v3 autonomy, provisional (phase 5): `activated NPC = last tuple +
    generated action` -- the same v3 call with no action supplied.
    Snapshot-compute, commit, then ADOPT -- never drop. v2 drops an
    activation whose tick the player's own pulse took meanwhile because
    two pulses at one tick would carry conflicting deltas; a smrti pulse
    appends tuples and at most mints an entity, and two pulses in one
    tick are two things that happened in the same tick (seq = world_time,
    id). Vadhu tick 110: Marren's turn was committed to the store and
    then dropped from memory, so the running server and the store
    disagreed until a restart. Now the result is always adopted into
    the live world (_adopt) -- its memories appended, what it bound or
    minted merged -- and the live canon's world_time left alone.

    What the gate gave the turn -- `begin` and `cap` (v5 §23, kala.world_turn) --
    goes to its first pulse only, as it always did."""
    for _ in range(pulses_run):
        # v5 §33: one pulse per scene at a time -- the snapshot is taken
        # once the scene is ours, so this turn reads the last one's result
        async with world.in_scene(world.kosha().scene_of(activated_entity_id, world.canon.world_time + 1), "activation",
                                  activated_entity_id):
            await _run_one_activation(world, roles, activated_entity_id, begin=begin, cap=cap)
        begin = cap = None


async def _run_one_activation(world: WorldSession, roles: GenerativeRoles, activated_entity_id: str, *,
                              begin: int | None = None, cap: int | None = None) -> None:
    started_at = world.canon.world_time
    snapshot_canon = world.canon.fork()
    snapshot_memories = list(world.recent_memories)
    snapshot_records = list(world.records)
    snapshot_canon.world_time += 1
    pulse_id = new_id()
    bind_pulse_id(pulse_id)
    try:
        _entity = snapshot_canon.entities.get(activated_entity_id)
        is_chronicle = bool(_entity is not None and _entity.origin == ITIHASA)
        # v5 §38: a scene drawn is asked whether anyone came (v3's road turn)
        is_scene = bool(_entity is not None and not is_chronicle and world.kosha().is_scene(activated_entity_id))
        news = pull = None
        if is_scene:  # the age as this scene's own time has it: news it has reached, the pull of what has not
            kala = world.samskara().kala
            reached, pull = _age_at(world, kala.now(activated_entity_id, begin) if kala else None)
            news = reached[-1] if reached else None
        result = await asyncio.to_thread(
            run_smrti_pulse, snapshot_canon, activated_entity_id, None, snapshot_canon.world_time, pulse_id,
            snapshot_memories, roles.smrti_client, adhisthana.get_embedder().embed,
            as_origin=is_scene,
            players=world.unvoiced(), movable=world.movable(),
            playable=frozenset(world.playable_entity_ids), view=world.view(),
            absent=world.absent(), max_elapsed=cap, wait_until=begin,
            # an away player the world now acts for, back in the world: this turn narrates it
            returning=world.left_at(activated_entity_id) if activated_entity_id in world.returning else None,
            as_chronicle=(world.bhuvana or "") if is_chronicle else None,
            news=news, pull=pull, age_from=_age_clock(world) if is_chronicle else None,
            records=snapshot_records, samskara=world.samskara(),
        )
    except Exception as exc:
        log_event("activation_pulse.run_failed", pulse_id=pulse_id, entity_id=activated_entity_id, error=repr(exc), smrti=True)
        return
    if result.pending:
        # v5 §21: the world's own turn went into a scene closed to it -- someone has already
        # lived it later. Nobody can be asked, and it may not be written: the turn is let go
        log_discarded(pulse_id=pulse_id, reason="npc_locked", detail=result.pending.get("scene"))
        log_event("smrti.npc_locked", pulse_id=pulse_id, entity_id=activated_entity_id, until=result.pending.get("until"))
        return
    world.returning.discard(activated_entity_id)
    if activated_entity_id in world.waiting and activated_entity_id in snapshot_canon.entities:
        # waiting: their own turn is not a touch -- quiet again from here, until someone else's pulse writes them one
        snapshot_canon.entities[activated_entity_id].last_simulated_tick = snapshot_canon.world_time
    try:
        await _store(world).commit(
            snapshot_canon, result.memories_written, result.new_entities, touched_ids=list(result.bound_ids) + [activated_entity_id],
            records=result.records_written,
        )
    except Exception as exc:
        log_event("commit_pulse.failed", source="smrti_activation", pulse_id=pulse_id, error=repr(exc))
        return
    overtaken = world.canon.world_time != started_at
    _adopt(world, snapshot_canon, result, activated_entity_id)
    _extend_history(world, result.memories_written, result.records_written)  # never the snapshot's lists: others appended meanwhile
    if overtaken:
        # the player's pulse took this tick meanwhile: same tick, both true
        log_event("activation_pulse.adopted_beside", entity_id=activated_entity_id, pulse_id=pulse_id,
                  world_time=snapshot_canon.world_time, live_world_time=world.canon.world_time)
    if activated_entity_id in world.playable_entity_ids:
        # the world acted for them: what was offered them was for the moment before (item 6) -- a round
        # going on without their action must not take it
        world.offered.pop(activated_entity_id, None)
        world.first_suggestion.pop(activated_entity_id, None)
    _queue_witnessed(world, result.memories_written)
    log_event("activation_pulse.timing", pulse_id=pulse_id, entity_id=activated_entity_id, smrti=True,
              written=len(result.memories_written))


def _players_latest(world: WorldSession) -> int | None:
    """The latest clock among the world's players (None: no clock)."""
    kala = world.samskara().kala
    if kala is None or not world.playable_entity_ids:
        return None
    return max(kala.now(p) for p in world.playable_entity_ids)


def _nearest_players_clock(world: WorldSession, start: str, new_place: bool = False) -> int | None:
    """The clock a newcomer joins at: that of the nearest online player in the tree (v5.24; Marc, 2026-10-07). The furthest-along player
    in the whole world put Kypazec in Stonemarch's common room 25 minutes ahead of the two players in it, because a third had made a long
    move far away. `start` is where the newcomer stands (or, with `new_place`, the container their new place is planted in, one level
    further). Distance is up the containment chain to the first container the two share. Several at the nearest distance: the latest of
    their clocks. A player who is away is not counted (their clock is stale), unless no one is online, and then all are. None: no
    player has a clock. The caller keeps the place's own clock as a floor."""
    kala = world.samskara().kala
    if kala is None or not world.playable_entity_ids:
        return None
    kosha = world.kosha()
    t = world.canon.world_time + 1

    def chain(scene: str | None) -> list[str]:
        return ([scene] + kosha.above(scene, t)) if scene else []
    mine = ([None] if new_place else []) + chain(start)
    at = {n: i for i, n in enumerate(mine) if n}
    pool = [p for p in sorted(world.playable_entity_ids) if world.online(p)] or sorted(world.playable_entity_ids)

    def distance(p: str) -> int:
        theirs = chain(kosha.scene_of(p, t))
        reach = [i + at[n] for i, n in enumerate(theirs) if n in at]
        return min(reach) if reach else 10 ** 6   # in no shared tree: farthest
    nearest = min(distance(p) for p in pool)
    return max(kala.now(p) for p in pool if distance(p) == nearest)


def _queue_witnessed(world: WorldSession, written: list[Memory], exclude: frozenset[str] = frozenset()) -> None:
    """A place tuple written where a player stands, while they were not
    acting, is what they witnessed (s19, s5) -- if its moment is their
    present (kala.apart, at the row's last minute: Marc, 2026-09-30). Where a
    player stands is their containment (v5 §11), the place of their own last
    tuple where that finds none; a player whose own last tuple is in no place
    sees nothing. By containment alone it showed Jakal, on the knoll,
    a moment 17 hours before he came. `exclude`: the one whose turn it was --
    they have their own narration. Called after the world's own turns, after
    a player's (Marc, 2026-09-26: two players together see each other's turns
    as they see an NPC's), and after an arrival (meeting: who walked in)."""
    written_ids = {m.id for m in written}
    said_kosha = world.kosha()
    mem = world.samskara().mem
    kala = world.samskara().kala
    for pid in world.playable_entity_ids:
        if pid in exclude:
            continue
        # their latest memory not written by this turn (from the end of their own, not a pass over all)
        last = next((m for m in reversed(mem.by_entity.get(pid, ())) if m.id not in written_ids), None)
        if last is None:
            continue
        if not last.location_id or last.location_id == NOWHERE:
            continue
        # v5 §11: where a player is, is their containment
        stands = said_kosha.scene_of(pid, world.canon.world_time + 1) or v3_enclosure(world.canon, last.location_id) or last.location_id
        for m in written:
            minute = m.ended if m.ended is not None else m.began
            if (kala is not None and minute is not None
                    and pid in apart(kala, [pid], minute, world.playable_entity_ids, world.movable())):
                continue  # not their present: they lived past it, or have not come to it
            if m.entity_id == stands and m.location_id == stands and m.experience:
                place = world.canon.entities.get(stands)
                text, said, said_by = split_heard(m.experience)  # a legacy seam, if any
                line = said_kosha.said.get((stands, m.pulse_id))
                if line:
                    said = line[0]
                    sp = world.canon.entities.get(line[1] or "")
                    said_by = sp.labels[0] if sp and sp.labels else (line[1] or "")
                world.pending_witnessed.setdefault(pid, []).append({
                    "place": place.labels[0] if place and place.labels else stands,
                    "text": text, "said": said, "said_by": said_by, "world_time": m.world_time,
                })


# -- saṅgama: players together act as one round -----------------------------
#
# Saṅgama (संगम, "confluence", Marc's name, 2026-09-27): where two or more
# players stand in one scene, in the world, their actions are a ROUND. Each acts;
# the round is written when every one of them has, or when one who has
# presses Go. One Narrator call resolves the actions together -- they may
# help, hinder or cross -- the clock moves once, and each player is told the
# moment from where they stand, the others' words as spoken: no summary box.
# A player who has not acted when the round runs does their first suggested
# action, as a default; with none offered, they are written for, and move
# or say nothing (v5 §36). Editing an action takes it back out until it is
# acted again. The round is the scene's: the world's own turns there wait.
# Cost: one Narrator call per round, its answer growing with the players in
# the scene; members are the players, one scene lookup each.


async def _run_round(world: WorldSession, roles: GenerativeRoles, background_tasks: BackgroundTasks, scene: str, runner: str,
                     request_start: float) -> PulseResponse:
    """The round written: the first to act is its ACTOR, every other player
    of the scene is in it -- with their action, or none."""
    r = world.rounds[scene]
    if r.get("running") or _round_held(world, scene):
        raise HTTPException(409, "the round is already being written")
    actor = next(p for p in r["order"] if p in r["actions"])
    members = sorted(set(_round_members(world, scene, actor)) | set(r["actions"]))
    # Marc, 2026-09-27: a player who has not acted when Go closes the round does their first suggested
    # action -- recorded as a default, not as their words; with none offered (a restart forgets them), none
    defaulted = {p: world.first_suggestion[p] for p in members if p not in r["actions"] and world.first_suggestion.get(p)}
    rnd = {"scene": scene, "others": {p: r["actions"].get(p) or defaulted.get(p, "") for p in members if p != actor},
           "sources": {**{p: r["sources"].get(p, "typed") for p in r["actions"] if p != actor}, **{p: "default" for p in defaulted}},
           "defaulted": defaulted}
    # a round's players are level (v5 §36 v4.31), so it begins at their one clock; the latest among them is kept as
    # the guard it was (v4.23, Stonemarch t155), and `behind` below should now always be empty
    kala = world.samskara().kala
    rnd["from"] = max(kala.now(p) for p in members) if kala is not None and members else None
    r["running"] = True
    log_event("sangama.run", scene=scene, actor=actor, runner=runner, acted=sorted(r["actions"]), defaulted=sorted(defaulted),
              begins=rnd["from"], behind={p: rnd["from"] - kala.now(p) for p in members if kala.now(p) < rnd["from"]} if rnd["from"] is not None else {},
              still=sorted(p for p in members if p not in r["actions"] and p not in defaulted))
    try:
        resp = await _submit_smrti_pulse(PulseRequest(action_text=r["actions"][actor], entity_id=actor), world, actor, roles,
                                         background_tasks, request_start, sangama=rnd, runner=runner, wait_until=rnd["from"])
    except BaseException:
        r["running"] = False  # nothing was written: the round stays open, and may be run again
        raise
    if scene in world.rounds and world.held.get(actor) is not None:
        world.rounds[scene]["running"] = False  # held for consent: open, and waiting on its players
    return resp


async def _round_submit(world: WorldSession, roles: GenerativeRoles, background_tasks: BackgroundTasks, scene: str, pid: str,
                        req: PulseRequest, request_start: float) -> PulseResponse:
    """An action into the scene's round (acted again, it replaces the last).
    The round is written as soon as every player there has acted."""
    world.presence[pid] = "online"  # v5 §37: acting is being here
    world.last_seen[pid] = time.monotonic()
    r = world.rounds.setdefault(scene, {"actions": {}, "sources": {}, "order": [], "running": False})
    if r["running"]:
        raise HTTPException(409, "the round is being written: act again once it is told")
    if _round_held(world, scene):
        raise HTTPException(409, "the round is waiting on everyone's yes to go on")
    r["actions"][pid] = req.action_text
    r["sources"][pid] = "suggested" if req.action_text.strip() in world.offered.get(pid, set()) else "typed"
    if pid not in r["order"]:
        r["order"].append(pid)
    members = _round_members(world, scene, pid)
    log_event("sangama.acted", scene=scene, entity_id=pid, members=members, acted=sorted(r["actions"]))
    if all(m in r["actions"] for m in members):
        return await _run_round(world, roles, background_tasks, scene, pid, request_start)
    return _told_nothing(round=_round_state(world, scene, pid).model_dump())


@router.get("/rounds/{pid}", response_model=RoundOut)
async def round_state(pid: str) -> RoundOut:
    """The round where this player stands, as their page sees it -- and,
    once written, their own telling (handed over once)."""
    world = await adhisthana.get_world()
    _require_player(world, pid)
    told = world.round_told.pop(pid, None)
    out = _round_state(world, _round_key(world, pid), pid)
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
    _require_player(world, pid)
    scene = _round_key(world, pid)
    r = world.rounds.get(scene or "")
    if r is not None and not r["running"] and not _round_held(world, scene) and pid in r["actions"]:
        del r["actions"][pid]
        r["order"].remove(pid)
        if not r["actions"]:
            world.rounds.pop(scene, None)
        log_event("sangama.edit", scene=scene, entity_id=pid)
    return _round_state(world, scene, pid)


@router.post("/rounds/{pid}/typing", status_code=204)
async def round_typing(pid: str) -> None:
    """Their page: they are writing an action (sent at most every few
    seconds while they type). Go waits GO_QUIET seconds after the last."""
    world = await adhisthana.get_world()
    _require_player(world, pid)
    world.typed[pid] = time.monotonic()


@router.post("/rounds/{pid}/go", response_model=PulseResponse)
async def round_go(pid: str, background_tasks: BackgroundTasks) -> PulseResponse:
    """Go: one who has acted closes the round, and it is written now."""
    start = time.monotonic()
    world = await adhisthana.get_world()
    _require_player(world, pid)
    scene = _round_key(world, pid)
    r = world.rounds.get(scene or "")
    # Go a step behind the round is not an error (Marc saw 409s, 2026-09-29): the page polls, so its Go can land after
    # the last action already wrote the round, while it is written, while it waits on a yes, or before the others have
    # stopped typing. It is answered with the round as it stands -- nothing run -- and the page's poll collects the telling.
    late = r is None or pid not in r["actions"] or r.get("running") or _round_held(world, scene)
    wait = 0.0 if late else _go_in(world, scene, _round_members(world, scene, pid), r["actions"])
    if late or wait > 0:
        log_event("sangama.go_early" if wait > 0 else "sangama.go_late", scene=scene, entity_id=pid, go_in=wait)
        return _told_nothing(round=_round_state(world, scene, pid).model_dump())
    world.last_seen[pid] = time.monotonic()
    return await _run_round(world, adhisthana._resolve_roles(), background_tasks, scene, pid, start)


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
    _require_own(request, world, acting_entity_id)
    roles = adhisthana._resolve_roles()  # may raise 503 -- deliberately after validation
    # saṅgama: where another player stands online in the scene, or a round is open there, the action joins it
    held = world.held.get(acting_entity_id)
    if held is not None and "round" in held:
        _decline_round(world, held, acting_entity_id)  # v5 §36: a new action answers the waiting question with no
    if world.come_back(acting_entity_id):  # acting is being here (v5 §37): back in the world, and so in its round
        await _store(world).save_logged_off(sorted(world.logged_off))
    # v5 §36 v4.31: within reach (the container directly above) and at the same moment -- a player behind or
    # ahead of an open round acts alone until their span meets its present
    scene = _round_key(world, acting_entity_id)
    level = set(_round_members(world, scene, acting_entity_id))
    if scene is not None and acting_entity_id not in world.held and (
            level - {acting_entity_id} or (scene in world.rounds and level & set(world.rounds[scene]["actions"]))):
        return await _round_submit(world, roles, background_tasks, scene, acting_entity_id, req, request_start)
    # v3: the pulse is retrieve -> narrate -> append and nothing else.
    return await _submit_smrti_pulse(req, world, acting_entity_id, roles, background_tasks, request_start)
