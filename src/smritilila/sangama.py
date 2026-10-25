# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Saṅgama (संगम, "confluence" -- Marc's name, 2026-09-27): players together act as one round (Design Principles
v5 §36). This module is the round's bookkeeping: which round a player is in (their scene, with the scenes a door or
a window joins to it, as one), who its members are (in the world, within reach, at the same moment), what each
player's page is shown of it, when Go may close it, and how a declined jump ends it.

The routes that act on a round, and the writing of it, stay with the pulse (routes/spanda.py): they run one. Moved out of main.py
unchanged (2026-10-05, Marc: "let's go get that main.py in shape"); main re-exported every name until c3e89a4; it no longer does.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from smritilila.sakshi import log_event
from smritilila.schemas import RoundMember, RoundOut, _labelled, _told_nothing
from smritilila.smrti.context import NOWHERE
from smritilila.training_data import log_discarded

if TYPE_CHECKING:
    from smritilila.adhisthana import WorldSession


def _reach_key(k, scene: str | None, t: int) -> str | None:
    """A round's key: the scene, with the scenes a boundary thing joins to it -- a door, a window, a gate
    (sandhi) -- as one. Marc, 2026-09-28: "Dunstan and I are still locked in rounds. Shouldn't we be going
    separately now?" -- keyed by the container directly above (v4.31), anyone anywhere in Millhaven (church,
    smithy, grove: all directly in the village) shared one round. Within reach still governs what is
    perceived and where spans meet (v5 §12, §20); a round is for actions that meet. Cost: the scene's joins,
    at most a dozen scenes."""
    if not scene or scene == NOWHERE:
        return None
    group, frontier = {scene}, [scene]
    while frontier and len(group) < 12:
        for _thing, other in k.across(frontier.pop(), t):
            if other and other not in group:
                group.add(other)
                frontier.append(other)
    return min(group)


def _round_key(world: WorldSession, pid: str) -> str | None:
    """The round a player would act in: their scene's reach key."""
    return _reach_key(world.kosha(), world.kosha().scene_of(pid, world.canon.world_time + 1), world.canon.world_time + 1)


def _round_members(world: WorldSession, key: str | None, level_with: str | None = None) -> list[str]:
    """The players of a round: in the world, within reach of one another, and
    at the same moment (v5 §36 v4.31; Marc, 2026-09-28: "Within reach is exactly
    correct"). In the world is not logged off -- by the button, or thirty
    minutes without acting (v5 §37) -- not "acted lately": after a restart nobody
    has, and the first to act would have gone alone. Level with `level_with`:
    a player behind or ahead acts alone until their span meets the others'
    present -- nobody is pulled forward (the round no longer begins at the
    latest clock among its players; that pull was v4.23's)."""
    if not key:
        return []
    k = world.kosha()
    t = world.canon.world_time + 1
    kala = world.samskara().kala
    at = kala.now(level_with) if kala is not None and level_with else None
    return sorted(p for p in world.playable_entity_ids if p not in world.logged_off
                  and _reach_key(k, k.scene_of(p, t), t) == key and (at is None or kala.now(p) == at))


def _decline_round(world: WorldSession, held: dict, by: str) -> None:
    """A round's jump refused by one who acted in it: nothing is written, the
    round is over, and each of its other players is told who said no."""
    rnd = held["round"]
    everyone = _round_acting(rnd, held["actor"])
    for p in everyone:
        world.held.pop(p, None)
    world.rounds.pop(rnd["scene"], None)
    name = _labelled(world.canon, by)
    for p in everyone | set(rnd["others"]):
        if p != by:
            world.round_told[p] = _told_nothing(round={"declined_by": name, "scene": rnd["scene"]})
    log_discarded(pulse_id=held["pulse_id"], reason="declined", detail=held["kind"])
    log_event("smrti.declined", pulse_id=held["pulse_id"], entity_id=by, kind=held["kind"], round=sorted(everyone))


def _round_held(world: WorldSession, scene: str) -> bool:
    """A scene's round written and waiting on its players' yes (v5 §36)."""
    return any("round" in h and h["round"]["scene"] == scene for h in world.held.values())


def _round_acting(rnd: dict, actor: str) -> set[str]:
    """Everyone who acted in a round being written: its actor and the others who acted."""
    return {actor} | {p for p, a in rnd["others"].items() if a}


def confirm_of(held: dict) -> dict:
    """The jump a player is asked to agree to, as a page or an agent's tool reads it: its kind and its words, and for a long action the minutes it spends
    and the world minute it ends at, as numbers (2026-10-22: the bot tool's time guard compares where a jump ends with where the people are; the words
    already say it, in a calendar a program should not have to parse)."""
    out = {"kind": held["kind"], "prompt": held["prompt"]}
    for k in ("minutes", "to"):
        if isinstance(held.get(k), int):
            out[k] = held[k]
    return out


def _round_state(world: WorldSession, scene: str | None, pid: str) -> RoundOut:
    r = world.rounds.get(scene or "") or {"actions": {}}
    members = sorted(set(_round_members(world, scene, pid)) | set(r["actions"]) | ({pid} if scene else set()))
    held = world.held.get(pid)
    confirm = waiting = None
    if held is not None and "round" in held:
        everyone = _round_acting(held["round"], held["actor"])
        waiting = [_labelled(world.canon, p) for p in sorted(everyone - held["yes"])]
        if pid not in held["yes"]:
            confirm = confirm_of(held)
    return RoundOut(scene=scene, members=[RoundMember(id=m, name=_labelled(world.canon, m), acted=m in r["actions"]) for m in members],
                    open=bool(r["actions"]), acted=pid in r["actions"], running=bool(r.get("running")),
                    confirm=confirm, waiting_on=waiting or [], go_in=_go_in(world, scene, members, r["actions"]))


GO_QUIET = 20  # seconds (Marc, 2026-09-27): Go waits this long after the last keystroke of anyone in the round yet to act


def _go_in(world: WorldSession, scene: str | None, members: Iterable[str], actions: dict) -> float:
    """Seconds until Go may close the round: nobody yet to act has typed for
    GO_QUIET seconds -- at once, if none of them is typing."""
    last = max((world.typed.get(p, 0.0) for p in members if p not in actions), default=0.0)
    return round(max(0.0, GO_QUIET - (time.monotonic() - last)), 1) if last else 0.0
