# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Pravṛtti (प्रवृत्ति), activity -- the world's own unfolding: the draw (Design Principles v5 §23, §38).

After a player's pulse the world takes a turn of its own. Who may take it is decided here, by structure only:
everyone who perceives and is not a player whose time is their own; a quiet scene SLEEPS (nidrā) and a being whose
last turn gave nothing is dormant until touched; a scene near a player that has been quiet is asked whether anyone
came, only if someone could have; a waiting player takes no turn until touched; what never kept a memory is folded
into what holds it (laya); a round open in a scene, or a player away in it, holds the scene's turns; and in a world
that keeps a clock, NPCs follow and never lead (the kāla gate). scheduler/episode.py then picks the oldest of what
is left, and routes/spanda.py runs its turn.

Moved out of main.py unchanged (2026-10-05, Marc: "let's go get that main.py in shape"); main re-exported every
name until c3e89a4; it no longer does, so a caller names this module. Each function takes the live WorldSession.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from smritilila.itihasa import _age_clock, _chronicle_id
from smritilila.memory.memory import Memory
from smritilila.sakshi import log_event
from smritilila.sangama import _reach_key
from smritilila.smrti.context import NOWHERE
from smritilila.smrti.kala import chronicle_ids, world_turn

if TYPE_CHECKING:
    from smritilila.adhisthana import WorldSession


def _smrti_exploration_pool(world: WorldSession) -> list[str]:
    """v4 (v5 §10-§12, §38): every perceiver who is not a player whose time is their own, drawn by scene.
    Its name: pravṛtti (प्रवृत्ति), activity, the world's own unfolding (v5 §38).
    Where anyone is, is their containment (smrti/kosha.py); near is a
    scene a player is in, has lately been in, or neighbours one (v5 §12).

    A quiet scene SLEEPS (v5 §38): its last QUIET_TURNS shared-record tuples
    all `use: none`, and no player in it or a neighbouring scene -- its
    occupants are not drawn. v3's "alone on open ground or a road" is
    retired with the kind it keyed on.

    A quiet scene near a player is ASKED WHETHER ANYONE CAME (v5 §38), as a
    road was: at most once in SCENE_QUIET_TICKS, far ones only on far
    ticks, and a near one may interrupt an engaged scene. Quiet, for
    arrivals, is also a scene nothing has happened in for
    SCENE_QUIET_TICKS -- worlds from before v4 carry no `use`, and a
    scene no turn has touched gave nobody anything (spec correction,
    2026-09-23, while building). A scene never asked counts as last asked
    at its latest record, so a converted world is not flooded with every
    scene's first ask (v5 Part 9's "measured before cutover", stood in)."""
    canon = world.canon
    t = canon.world_time + 1
    pace = world.pace()
    kosha = world.kosha()
    players = set(world.playable_entity_ids)
    drawn_players = players - set(world.unvoiced())  # v5 §37: offline, and allowing the world to act for them
    chronicle = _chronicle_id(world)

    def where(loc: str | None) -> str:
        if not loc or loc == NOWHERE:
            return NOWHERE
        return kosha.scene_of(loc, t) or NOWHERE

    def scene(eid: str) -> str:
        return kosha.scene_of(eid, t) or NOWHERE

    # each entity's latest memory, each scene's own rows in order: kept (smrti/samskara.MemIndex),
    # not a pass over every memory each draw
    mem = world.samskara().mem
    last_by: dict[str, Memory] = mem.last_by
    scene_rows: dict[str, list[Memory]] = mem.rows
    # the chronicle is among them: in no scene, it competes as a far actor
    actors = {eid for eid in kosha.perceivers if eid in canon.entities and (eid not in players or eid in drawn_players)
              and (eid != chronicle or world.bhuvana) and eid not in world.held_back}  # held back: until a god reviews  # the earth of a world that keeps no age perceives, and is quiet

    player_scenes = {scene(p) for p in players - world.absent()} - {NOWHERE}
    by_players = set(player_scenes)
    for s_ in player_scenes:
        by_players |= set(kosha.neighbours(s_, t))
    near_scenes: set[str] = set(by_players)
    for pid in world.playable_entity_ids:
        own = mem.by_entity.get(pid, [])[-pace["near_window"]:]
        near_scenes |= {where(m.location_id) for m in own} - {NOWHERE}
    far_tick = canon.world_time % pace["far_every"] in (0, 1)
    # The age moves (itihasa, 2026-09-20): on a far tick, if the chronicle
    # has not moved in CHRONICLE_EVERY ticks, it takes the draw ahead of
    # the far actors it would otherwise queue behind.
    if far_tick and chronicle is not None and chronicle in last_by and world.bhuvana:
        e = canon.entities[chronicle]
        if e.last_simulated_tick is None or canon.world_time - e.last_simulated_tick >= pace["chronicle_every"]:
            log_event("smrti.age_moves", tick=canon.world_time, last=e.last_simulated_tick)
            return [chronicle]

    def gave_nothing(sid: str) -> bool:
        """Its name: nidrā (निद्रा), sleep (v5 §38)."""
        rows = scene_rows.get(sid, [])[-pace["quiet_turns"]:]
        return len(rows) == pace["quiet_turns"] and all(m.use == "none" for m in rows)

    def untouched(sid: str) -> bool:
        rows = scene_rows.get(sid, [])
        return not rows or canon.world_time - rows[-1].world_time >= pace["scene_quiet_ticks"]

    # v4.61, breath (pratiṣṭhā): a branch may keep its own pace (v5 §38) -- pacing {"far_every": n} on it or above. Its
    # beings far from every player are drawn in a two-tick window every n ticks (1: every tick) instead of the world's far ticks, and its
    # scenes never sleep for quiet. Read once per scene a draw asks about. Cost: each such scene's chain, once.
    _breath: dict[str, int | None] = {}

    def breath(sid: str) -> int | None:
        if sid not in _breath:
            v = kosha.seed("pacing", sid, t, field="far_every") if sid and sid != NOWHERE else None
            try:
                _breath[sid] = max(1, int(v)) if v is not None and int(v) != pace["far_every"] else None
            except (TypeError, ValueError):
                _breath[sid] = None
        return _breath[sid]

    def far_here(sid: str) -> bool:
        n = breath(sid)
        return far_tick if n is None else (n == 1 or canon.world_time % n in (0, 1))

    sleeping = {sid for sid in scene_rows if gave_nothing(sid) and sid not in by_players and breath(sid) is None}
    quiet: list[tuple[str, int]] = []
    for sid, e in canon.entities.items():
        if not kosha.is_scene(sid) or sid in player_scenes:
            continue
        if not (gave_nothing(sid) or untouched(sid)):
            continue
        rows = scene_rows.get(sid, [])
        asked = e.last_simulated_tick if e.last_simulated_tick is not None else (rows[-1].world_time if rows else e.created_at_world_time)
        if canon.world_time - asked < pace["scene_quiet_ticks"]:
            continue
        open_here = kosha.seed("pacing", sid, t, field="arrivals")  # v5 §15: a closed valley is closed where it is
        if (pace["arrivals"] if open_here is None else bool(open_here)) and (sid in near_scenes or far_tick):  # a closed world is never asked (v5 §38)
            quiet.append((sid, asked))
    # v5 §38, asked only if someone could have come (Marc, 2026-09-28: "c sounds right"): something not a player
    # left a neighbouring scene since the scene was last asked, for this scene, for no scene recorded, or for a root scene. Stonemarch,
    # replayed: 51 asks, 2 of which moved or minted anything; this keeps 6, both of those among them -- the rest
    # were empty rooms telling that they were empty, each a model call and the world's one turn that tick
    arrivals = [sid for sid, _ in quiet] if not pace["arrivals_on_traffic"] else []  # a world may ask on the clock (v5 §38)
    if quiet and pace["arrivals_on_traffic"]:
        left = kosha.departures(min(a for _, a in quiet), t)
        unasked = []
        for sid, asked in quiet:
            near = set(kosha.neighbours(sid, t))
            if any(w > asked and eid not in players and was in near and (now == sid or not now or kosha.is_root_scene(now))
                   for w, eid, was, now in left):
                arrivals.append(sid)
            else:
                unasked.append(sid)
        if unasked:
            log_event("smrti.nobody_near", tick=canon.world_time, count=len(unasked), scenes=sorted(unasked)[:20])
    # the draw is oldest-first with never-run first: a never-asked scene
    # takes its latest record, or its minting, as its last ask (see the
    # docstring; crossing 468: an overhang minted at 467 was asked at once)
    for sid in arrivals:
        e = canon.entities[sid]
        if e.last_simulated_tick is None:
            e.last_simulated_tick = scene_rows[sid][-1].world_time if scene_rows.get(sid) else e.created_at_world_time

    def gave_nothing_itself(eid: str) -> bool:
        """DORMANT (Marc, 2026-09-27; Stonemarch: a bent dart, a barred door,
        a dead spider each drawn to act "on their own"): a perceiver whose own
        last turn moved nothing, said nothing and gave nothing (every tuple of
        it `use: none`) is not drawn again until another's pulse writes it a
        tuple -- handled again, it may act again.
        Its name: nidrā (निद्रा), sleep (v5 §38).
        As a quiet scene sleeps (v5 §38), by structure: the engine cannot tell a dart from a gnoll, and
        does not try. Cost: the candidate's own memories since its last turn."""
        since = canon.entities[eid].last_simulated_tick
        if since is None:
            return False
        own = []
        for m in reversed(mem.by_entity.get(eid, ())):
            if m.world_time <= since:
                break
            own.append(m)
        if not own or len({m.pulse_id for m in own}) > 1:
            return False  # its turn wrote nothing, or someone has touched it since
        if any((m.use or "none") != "none" for m in own):
            return False
        turn = own[0].world_time
        if kosha.container_at(eid, turn) != kosha.container_at(eid, turn + 1):
            return False  # it moved
        return not any((kosha.said.get((m.location_id, m.pulse_id)) or ("", None))[1] == eid for m in own)

    def touched_while_waiting(eid: str) -> bool:
        """WAITING (Marc, 2026-10-01): someone else's pulse wrote this player a
        tuple since their quiet mark (set when they began to wait, and after
        each turn of their own) -- not a pulse their own player acted in.
        Cost: the player's memories since the mark, and one pass over the
        records for their own actions."""
        since = canon.entities[eid].last_simulated_tick
        if since is None:
            return False
        acted = {r.pulse_id for r in world.records if r.kind == "action" and r.entity_id == eid}
        for m in reversed(mem.by_entity.get(eid, ())):
            if m.world_time <= since:
                break
            if m.pulse_id not in acted:
                return True
        return False

    pool = []
    slept = []
    dormant = []
    waiting = []
    for eid in actors:
        if eid not in last_by:
            continue
        at = scene(eid)
        if at not in near_scenes and not far_here(at):
            continue
        if at in sleeping:
            slept.append(eid)
            continue
        if eid in world.waiting and not touched_while_waiting(eid):
            waiting.append(eid)
            continue
        if eid != chronicle and eid not in players and gave_nothing_itself(eid):  # the age, and a player, are nobody's to touch
            dormant.append(eid)
            continue
        pool.append(eid)
    if slept:
        log_event("smrti.scene_sleeps", tick=canon.world_time, scenes=sorted(sleeping), actors=len(slept))
    if dormant:
        log_event("smrti.dormant", tick=canon.world_time, count=len(dormant), ids=sorted(dormant)[:20])
    if waiting:
        log_event("smrti.waiting", tick=canon.world_time, ids=sorted(waiting))
    # ENGAGED (2026-09-21): whoever wrote a tuple in the player's scene in
    # the player's most recent own pulse; the draw is among them alone.
    # A witnessed tuple is the scene's text under the player's id, so the
    # player's own is the one whose text is no scene's.
    player_pulses = mem.player_pulses  # the players' own, by (tick, id): kept
    if player_pulses:
        last_player = player_pulses[-1]
        here = where(last_player.location_id)
        at_tick = mem.by_tick.get(last_player.world_time, ())
        engaged = [eid for eid in pool if any(m.entity_id == eid and where(m.location_id) == here for m in at_tick)]
        # CADENCE (Marc, 2026-09-21): the scene's NPCs take the draw on
        # every second player pulse in a scene, not every one.
        last_draw = world.scene_draws.get(here)
        since = sum(1 for m in mem.player_pulses_after(last_draw) if where(m.location_id) == here) if last_draw is not None else None
        if engaged and since is not None and since < pace["scene_cadence"] and last_player.world_time != last_draw:
            log_event("smrti.scene_waits", place=here, tick=last_player.world_time, engaged=engaged, last_draw=last_draw, pulses_since=since)
            in_scene = {eid for eid in pool if scene(eid) == here}
            return [e for e in pool + arrivals if e not in in_scene and e != here]
        if engaged:
            world.scene_draws[here] = last_player.world_time
            # an arrival interrupts a scene (Hillands 19-31): a NEAR
            # quiet scene past its window takes the draw ahead of the engaged
            near_arrivals = [a for a in arrivals if a in near_scenes]
            due = world.last_arrival_interrupt is None or canon.world_time - world.last_arrival_interrupt >= pace["scene_quiet_ticks"]
            if near_arrivals and due:
                world.last_arrival_interrupt = canon.world_time
                log_event("smrti.arrival_interrupts", place=here, tick=last_player.world_time, scenes=near_arrivals)
                return near_arrivals
            log_event("smrti.engaged", place=here, tick=last_player.world_time, engaged=engaged)
            return engaged
    if arrivals:
        log_event("smrti.arrivals_offered", tick=canon.world_time, scenes=arrivals, near=[a for a in arrivals if a in near_scenes])
    return pool + arrivals


def _log_pool(world: WorldSession, pool: list[str]) -> None:
    """The draw's candidates as the world saw them, witnessed (Marc, 2026-09-28): each with its last turn's tick and
    the uses of its own tuples at that tick -- what its last turn gave. The pool hangs on state never written (the
    scene cadence, presence, rounds, the clock's caps), so history could not be replayed faithfully (22 of 86 picks
    rebuilt); from this line a draw rule is scored exactly on the same pools. Cost: the pool, each candidate's own
    memories back to its last turn."""
    by_entity = world.samskara().mem.by_entity
    rows = []
    for e in pool:
        ent = world.canon.entities.get(e)
        last = ent.last_simulated_tick if ent is not None else None
        uses = []
        if last is not None:
            for m in reversed(by_entity.get(e, ())):
                if m.world_time < last:
                    break
                if m.world_time == last:
                    uses.append(m.use or "none")
        rows.append([e, last, sorted(set(uses))])
    log_event("tick.pool", tick=world.canon.world_time, pool=rows)


def _paused(world: WorldSession) -> frozenset[str]:
    """The scenes that hold for a player away (Marc, 2026-09-26: "the room
    pauses around them"): where a player stands who is offline or out of
    the world and does not allow the world to act for them. Such a scene
    takes no autonomous turns -- its NPCs, its things, its own "did anyone
    come" -- until they are back. Before, the Green Door parlour drew four
    turns while Wentworth was offline, each written for him and blocked
    (smrti.player_not_voiced), each restating that the room waited on him.
    Cost: the players, one scene lookup each."""
    away = [p for p in world.playable_entity_ids if p not in world.autonomous and not world.online(p)]
    return frozenset(s for s in (world.here(p, absent=False)[0] for p in away) if s)


def _hold_for_rounds(world: WorldSession, pool: list[str]) -> list[str]:
    """Saṅgama: a scene with a round open takes no turn of the world's own
    until the round is written -- the clock does not move under its players.
    Cost: the open rounds, one scene lookup per candidate."""
    if not world.rounds or not pool:
        return pool
    k = world.kosha()
    t = world.canon.world_time + 1
    held = [e for e in pool if _reach_key(k, e if k.is_scene(e) else k.scene_of(e, t), t) in world.rounds]  # v4.31: rounds are keyed by reach
    if held:
        log_event("sangama.world_waits", tick=world.canon.world_time, scenes=sorted(world.rounds), held=held)
    return [e for e in pool if e not in held]


ABSORBED_QUIET_TICKS = 40  # absorption: quiet is nothing of one's own -- tuple, move, mention -- in this many of the world's ticks


def _fold_absorbed(world: WorldSession, pool: list[str]) -> list[str]:
    """Absorption (b) in the draw (Marc, 2026-09-29): a candidate is folded into its container -- not drawn on its own;
    its container's turn carries it -- when it is QUIET (no tuple of its own, no move, no mention in the last
    ABSORBED_QUIET_TICKS ticks) and has NEVER KEPT anything (no memory of its own, from anyone's pulse, ever kept
    useful or key).
    Its name: laya (लय), dissolution back into one's source (v5 §38).
    By what it has done, never by kind (s0): whoever has kept a memory -- Jaroo, Rannulf -- is drawn
    as before, and whatever is folded wakes as itself the moment a player or a telling touches it. Replayed on
    Stonemarch's 30 witnessed pools: 8 picks change, each had kept nothing; 5 draws become no world turn; nobody who
    ever kept a memory is folded. ("The last 5 tellings of its own scene", as first settled, never ages for a room
    nobody revisits: the ogre's lair still named its chest.) Cost per draw: passes over ALL the world's memories (two) and
    records (one), each filtered by tick -- linear in the world's age, about 0.7 ms a draw at Stonemarch's 11,932 memories and
    7,067 records (measured 2026-10-22) -- plus each candidate's own memories."""
    if not pool:
        return pool
    since = world.canon.world_time - ABSORBED_QUIET_TICKS
    active = {m.entity_id for m in world.recent_memories if m.world_time >= since}
    active |= {r.entity_id for r in world.records if r.kind == "kosha" and r.world_time >= since}
    said = " | ".join(m.experience.lower() for m in world.recent_memories if m.world_time >= since and m.experience)
    by_entity = world.samskara().mem.by_entity
    # the world at large is the age's own chronicler (itihasa): its entries are impersonal and span months, never kept
    # useful or key, so by the test it looked like a thing that never acted -- and the age would stop (Stonemarch t521)
    age = chronicle_ids(world.canon)
    folded = []
    for e in pool:
        if e in age or e in active or any(m.use in ("useful", "key") for m in by_entity.get(e, ())):
            continue
        ent = world.canon.entities.get(e)
        if ent is not None and any(len(label) > 3 and label.lower() in said for label in ent.labels):
            continue  # mentioned lately: not quiet
        folded.append(e)
    if folded:
        log_event("smrti.absorbed", tick=world.canon.world_time, folded=folded)
    return [e for e in pool if e not in folded]


def _hold_for_away(world: WorldSession, pool: list[str]) -> list[str]:
    """The draw without what stands in a paused scene (_paused)."""
    paused = _paused(world)
    if not paused or not pool:
        return pool
    k = world.kosha()
    t = world.canon.world_time + 1
    held = [e for e in pool if (e if k.is_scene(e) else k.scene_of(e, t)) in paused]
    if held:
        log_event("smrti.room_paused", tick=world.canon.world_time, scenes=sorted(paused), held=held)
    return [e for e in pool if e not in held]


def _kala_gate(world: WorldSession, pool: list[str]) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """v5 §23: NPCs follow and never lead. In a world that keeps a clock, a
    candidate at or past its reference -- the EARLIEST clock among the
    players whose time is their own (online, or offline and not allowing
    the world to act for them: v4.31) in or next to its scene, else the
    world's present, the earliest clock among online players (among all
    players in the world, if none is online) -- is not drawn; of the rest,
    those within 30 minutes of the oldest clock are, the oldest clocks
    first, each capped to end by its reference -- and, in its
    own scene, to begin after and end before what someone already lived
    there (kala.world_turn). Returns the drawn, and each gated turn's
    (begin, cap) for the caller to hand to the turn it schedules."""
    _t = time.perf_counter()
    kala = world.samskara().kala  # kept, read on with what was committed
    kala_ms = round((time.perf_counter() - _t) * 1000, 1)
    turns: dict[str, tuple[int, int]] = {}
    if kala is None or not pool:
        return pool, turns
    k = world.kosha()
    t = world.canon.world_time + 1
    here = [p for p in world.playable_entity_ids if p not in world.absent()]
    online = [p for p in here if world.online(p)]
    if not here:
        return pool, turns
    clocks = {p: kala.now(p) for p in here}
    present = min(clocks[p] for p in (online or here))  # nobody online: the world still waits for its players
    # v5 §23 v4.31: every span stops at the first present it touches -- a world turn too. A player whose time is
    # their own bounds the turns near them whether online or not; one who lets the world act for them while
    # away (v5 §37) is carried. Ravenmoor's parlour ran 1507 -> 1534 past an offline player standing in it at 1500.
    their_own = [p for p in here if p in online or p not in world.autonomous]
    where = {p: k.scene_of(p, t) for p in their_own}
    gated = []
    chronicle, age_draw = _chronicle_id(world), False
    for eid in pool:
        if eid == chronicle:
            # the age runs ahead of the players, a season at most (Marc, 2026-09-25); its own grain, no cap
            ahead = (_age_clock(world) or kala.start) - max(clocks.values())
            if ahead > int(world.pace()["age_lead"]):
                log_event("smrti.age_waits", tick=world.canon.world_time, ahead=ahead)
            else:
                age_draw = True
            continue
        sc = eid if k.is_scene(eid) else k.scene_of(eid, t)
        near = ({sc} | set(k.neighbours(sc, t))) if sc else set()
        near_clocks = [clocks[p] for p in their_own if where[p] in near]
        ref = min(near_clocks) if near_clocks else present
        turn = world_turn(kala, k, eid, sc, ref, t)
        if turn is None:
            continue
        own, cap = turns[eid] = turn
        gated.append((own, eid))
    if not gated:
        if age_draw:
            return [chronicle], turns
        log_event("smrti.world_waits", tick=world.canon.world_time, present=present, kala_ms=kala_ms)
        return [], turns
    gated.sort()
    oldest = gated[0][0]
    drawn = [eid for own, eid in gated if own <= oldest + 30]  # the oldest clocks wake first
    log_event("smrti.gate", tick=world.canon.world_time, pool=len(pool), drawn=len(drawn), kala_ms=kala_ms)
    return drawn + ([chronicle] if age_draw else []), turns
