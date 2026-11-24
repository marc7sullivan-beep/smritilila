# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Itihāsa (इतिहास, "thus it was"): the age, as the server reads it (Design Principles v5 §24).

The earth -- the world at large -- keeps the chronicle of the age: its own tuples, the first of them the bhuvana's
THE YEAR. What the server asks of it lives here: which entity is the earth, the chronicle's entries since its
birth, how far the age has reached on the clock, which entries a moment has passed and what pulls ahead of it, and
the world at large as a writer of the world is shown it. And the world's one line, drawn from the age once a real-time
day and kept with the list of worlds (moved from routes/bija.py on 2026-11-10, stage 2 of moving the orchestration out
of the routes, nothing inside changed).

Moved out of main.py unchanged (2026-10-05, Marc: "let's go get that main.py in shape"); main re-exported every
name until c3e89a4; it no longer does, so a caller names this module. Each function takes the live WorldSession; the one
line's retry clock (_describe_attempted) is the only state held here.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila.adhisthana import stream_of, app
from smritilila.memory.memory import Memory
from smritilila.sakshi import log_event
from smritilila.smrti.genesis import ITIHASA

if TYPE_CHECKING:
    from smritilila.adhisthana import GenerativeRoles, WorldSession


def chronicle_id(world: WorldSession) -> str | None:
    return next((e.id for e in world.canon.entities.values() if e.origin == ITIHASA), None)


def _age_tuples(world: WorldSession) -> list[Memory]:
    """The age's own tuples, birth first: the world at large's, from the one
    that carries the bhuvana on (Marc, 2026-09-30: earth is the world at
    large -- a world born quiet perceives by its setting before it is given
    an age, and that memory is the earth's, not the age's)."""
    cid = chronicle_id(world)
    if cid is None:
        return []
    own = sorted(world.samskara().mem.by_entity.get(cid, ()), key=lambda m: (m.world_time, m.id))
    head = (world.bhuvana or "").split("THE REGIONS", 1)[0].strip()
    born = next((i for i, m in enumerate(own) if head and m.experience.strip() == head), 0)
    return own[born:]


def chronicle(world: WorldSession, last: int = 3) -> list[str]:
    """The age's chronicle, latest last: the world at large's own tuples."""
    return [m.experience for m in _age_tuples(world)[-last:]]


def age_at(world: WorldSession, minute: int | None) -> tuple[list[str], str | None]:
    """The chronicle as a moment has it (Marc, 2026-09-25: it runs ahead,
    and is shown by time): (the entries reached, latest last; the first
    not reached -- the age's pull). An entry is reached when `minute` has
    passed its end; one with no span (the birth year, or one written
    before the age kept time) always is, and once one is not, none after
    it is. `minute` None -- a world keeping no clock -- reaches them all."""
    samskara = world.samskara()
    reached: list[str] = []
    for m in _age_tuples(world):
        span = samskara.age.get(m.pulse_id)
        if minute is not None and span is not None and span[1] > minute:
            return reached, m.experience
        reached.append(m.experience)
    return reached, None


def age_clock(world: WorldSession) -> int | None:
    """Where the age has reached: the end of its latest timed entry, or
    the clock's start. None in a world keeping no clock."""
    kala = world.samskara().kala
    if kala is None:
        return None
    return max((end for _b, end in world.samskara().age.values()), default=kala.start)


_PULL = ("THE AGE'S PULL -- what is coming in the wide world, NOT YET HAPPENED where this is written: it may colour rumour and "
         "worry and draw threads toward here; never write any of it as having happened:\n")


def at_large(world: WorldSession, minute: int | None = None) -> str | None:
    """Bhuvana and the chronicle since, for whoever writes the world, as of
    `minute` (an arrival: its place's time; an expansion: the latest
    player's) -- and what is ahead of it as the age's pull."""
    if not world.bhuvana:
        return None
    reached, pull = age_at(world, minute)
    since = reached[1:][-3:]  # the birth year is the bhuvana itself
    text = world.bhuvana
    if since:
        text += "\n\nTHE CHRONICLE SINCE (what has moved in the age, latest last):\n" + "\n".join(f"- {t}" for t in since)
    if pull:
        text += "\n\n" + _PULL + pull
    return text


# -- The world's one line (moved from routes/bija.py, 2026-11-10, nothing inside changed): drawn from the age by the server that
# serves the world, once a real-time day, and kept with the list of worlds. adhisthana's lifespan starts describe_daily; a
# genesis and the gods' POST /world/describe call describe. Called as itihasa.X, so a test's patch here reaches it.


def _root_of(world: WorldSession) -> str:
    """What the world grows from, as its earth carries it: the age, where the
    world keeps one; else the earth's first memory (its setting); else the setting.
    With the earth's newest entries after it (2026-10-20): the age moves, and a line drawn from the
    birth year alone said "April 27, 1928" of a world whose earth had reached July."""
    if world.bhuvana:
        base = world.bhuvana.split("THE REGIONS", 1)[0].strip() or world.bhuvana[:1200]
    else:
        earth = chronicle_id(world)
        first = min(stream_of(world, earth), key=lambda m: (m.world_time, m.id), default=None)
        base = first.experience if first is not None else (world.setting_description or "")
    newer = [m.experience.strip() for m in _age_tuples(world)[1:][-DESCRIBE_AGE_ENTRIES:] if m.experience.strip() and m.experience.strip() not in base]   # after the birth entry
    if not newer:
        return base
    return base + "\n\nTHE AGE SINCE (the world at large's newest entries: where and when the world has reached now):\n" + "\n\n".join(t[:1200] for t in newer)


async def describe(world: WorldSession, roles: GenerativeRoles) -> str | None:
    """Draw the world's one line from its root and keep it with the list of
    worlds. A world outside the registry has no line to keep; a draw that
    fails leaves the line it had."""
    from smritilila.smrti.bija import describe_root
    from smritilila.worlds_registry import set_world_description
    entry = next((w for w in adhisthana.get_worlds_registry() if w.name == world.name), None) if world.name else None
    root = _root_of(world)
    if entry is None or not root.strip() or roles.smrti_client is None:
        return None
    try:
        line = await asyncio.to_thread(describe_root, roles.smrti_client, entry.name, root)
    except Exception as exc:
        log_event("world.describe_failed", name=entry.name, error=repr(exc))
        return None
    set_world_description(entry.name, line)
    app.state.worlds_registry = None  # reload from the file
    log_event("world.described", name=entry.name, description=line)
    return line


DESCRIBE_AGE_ENTRIES = 2        # how many of the earth's newest entries the line is drawn with, after the root


DESCRIBE_EVERY = 24 * 3600      # a world's line is drawn again once per real-time day (Marc, 2026-10-20)


DESCRIBE_FIRST_AFTER = 600      # seconds after a server starts before it first looks (a restart is not a reason to draw)


DESCRIBE_CHECK_EVERY = 1800     # seconds between looks


DESCRIBE_RETRY_AFTER = 3600     # a draw that failed is tried again after this long, not at the next look


_describe_attempted: dict[str, float] = {}


def _describe_due(entry, now: datetime | None = None) -> bool:
    """Its line has never been drawn with a time, or was drawn a real-time day ago or more (an unreadable time counts as never)."""
    try:
        at = datetime.strptime(entry.described_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return True
    return (now or datetime.now(timezone.utc)) - at >= timedelta(seconds=DESCRIBE_EVERY)


async def describe_if_due(now: datetime | None = None, clock=time.monotonic) -> str | None:
    """One look: the line of the world this server serves, drawn if a day has passed. The world is not loaded, nor a model asked, unless
    it is due and no draw has failed within the hour. Returns the new line, or None."""
    entry = adhisthana.active_world_registry_entry()
    if entry is None or not _describe_due(entry, now):
        return None
    if clock() - _describe_attempted.get(entry.name, -DESCRIBE_RETRY_AFTER) < DESCRIBE_RETRY_AFTER:
        return None
    _describe_attempted[entry.name] = clock()
    world = await adhisthana.get_world()
    line = await describe(world, adhisthana.resolve_roles())
    if line is not None:
        _describe_attempted.pop(entry.name, None)
    return line


async def describe_daily() -> None:
    """Each world's one line, drawn again once per real-time day by the server that serves it (Marc, 2026-10-20: "the one sentence world
    description should be updated once per real time day"). Started with the server, stopped with it. A world with no entry of its own in the
    registry, or no root yet, keeps the line it has; a draw that fails leaves the line it had and is tried again within the hour."""
    port = os.environ.get("SMRITILILA_DEV_PORT", "8198")
    await asyncio.sleep(DESCRIBE_FIRST_AFTER + (int(port) % 10 if port.isdigit() else 0) * 20)   # six servers do not look in the same minute
    while True:
        try:
            await describe_if_due()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a failed look is tried again at the next one, never a reason to stop
            log_event("world.describe_failed", error=repr(exc))
        await asyncio.sleep(DESCRIBE_CHECK_EVERY)
