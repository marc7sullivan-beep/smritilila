# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Itihāsa (इतिहास, "thus it was"): the age, as the server reads it (Design Principles v5 §24).

The earth -- the world at large -- keeps the chronicle of the age: its own tuples, the first of them the bhuvana's
THE YEAR. What the server asks of it lives here: which entity is the earth, the chronicle's entries since its
birth, how far the age has reached on the clock, which entries a moment has passed and what pulls ahead of it, and
the world at large as a writer of the world is shown it.

Moved out of main.py unchanged (2026-10-05, Marc: "let's go get that main.py in shape"); main re-exported every
name until c3e89a4; it no longer does, so a caller names this module. Each function takes the live WorldSession; nothing here
holds state of its own.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from smritilila.memory.memory import Memory
from smritilila.smrti.genesis import ITIHASA

if TYPE_CHECKING:
    from smritilila.adhisthana import WorldSession


def _chronicle_id(world: WorldSession) -> str | None:
    return next((e.id for e in world.canon.entities.values() if e.origin == ITIHASA), None)


def _age_tuples(world: WorldSession) -> list[Memory]:
    """The age's own tuples, birth first: the world at large's, from the one
    that carries the bhuvana on (Marc, 2026-09-30: earth is the world at
    large -- a world born quiet perceives by its setting before it is given
    an age, and that memory is the earth's, not the age's)."""
    cid = _chronicle_id(world)
    if cid is None:
        return []
    own = sorted(world.samskara().mem.by_entity.get(cid, ()), key=lambda m: (m.world_time, m.id))
    head = (world.bhuvana or "").split("THE REGIONS", 1)[0].strip()
    born = next((i for i, m in enumerate(own) if head and m.experience.strip() == head), 0)
    return own[born:]


def _chronicle(world: WorldSession, last: int = 3) -> list[str]:
    """The age's chronicle, latest last: the world at large's own tuples."""
    return [m.experience for m in _age_tuples(world)[-last:]]


def _age_at(world: WorldSession, minute: int | None) -> tuple[list[str], str | None]:
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


def _age_clock(world: WorldSession) -> int | None:
    """Where the age has reached: the end of its latest timed entry, or
    the clock's start. None in a world keeping no clock."""
    kala = world.samskara().kala
    if kala is None:
        return None
    return max((end for _b, end in world.samskara().age.values()), default=kala.start)


_PULL = ("THE AGE'S PULL -- what is coming in the wide world, NOT YET HAPPENED where this is written: it may colour rumour and "
         "worry and draw threads toward here; never write any of it as having happened:\n")


def _at_large(world: WorldSession, minute: int | None = None) -> str | None:
    """Bhuvana and the chronicle since, for whoever writes the world, as of
    `minute` (an arrival: its place's time; an expansion: the latest
    player's) -- and what is ahead of it as the age's pull."""
    if not world.bhuvana:
        return None
    reached, pull = _age_at(world, minute)
    since = reached[1:][-3:]  # the birth year is the bhuvana itself
    text = world.bhuvana
    if since:
        text += "\n\nTHE CHRONICLE SINCE (what has moved in the age, latest last):\n" + "\n".join(f"- {t}" for t in since)
    if pull:
        text += "\n\n" + _PULL + pull
    return text
