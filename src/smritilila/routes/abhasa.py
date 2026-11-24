# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Ābhāsa (2026-11-11, Marc: "image generation and presentation"): a player's pictures -- which there are for them, and each one's image.

The pictures themselves are made by smritilila/abhasa.py after a telling is committed; this is only what the page asks. A player sees their own portrait,
the portraits of those in the scene they stand in now and that place's picture, and what was made this session from what they lived this session
(abhasa.visible_to), and nothing else: what the picture shows of someone else's private moment stays theirs. Both paths are under /players/{pid}, so the guest gate already holds a guest to their own character.
"""

from __future__ import annotations

import contextlib

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from smritilila import abhasa  # the services, called through it so a test's patch there reaches them
from smritilila import adhisthana
from smritilila.adhisthana import store_of
from smritilila.dvara import require_own, require_player
from smritilila.schemas import PortraitDraftRequest, PortraitEditRequest

router = APIRouter()


@router.get("/players/{pid}/pictures")
async def player_pictures(pid: str, after: int = 0, request: Request = None) -> list[dict]:
    """The pictures this player may see, in the order they were kept, those after `after` (the last `seq` the page saw): id, seq, whose or what,
    a caption, whether it is of a moment. [] in a world that makes none."""
    world = await adhisthana.get_world()
    require_player(world, pid)
    require_own(request, world, pid)
    if abhasa.settings(world) is None:
        return []   # a world that makes none touches nothing of its own for them
    st = await abhasa.ensure(world)
    return abhasa.listing(world, st, pid, after)


@router.get("/players/{pid}/pictures/{picture_id}")
async def player_picture(pid: str, picture_id: str, request: Request = None) -> Response:
    """One picture's image, if this player may see it."""
    world = await adhisthana.get_world()
    require_player(world, pid)
    require_own(request, world, pid)
    if abhasa.settings(world) is None:
        raise HTTPException(404, "no such picture")
    st = await abhasa.ensure(world)
    if not any(c.id == picture_id for c in abhasa.visible_to(world, st, pid)):
        raise HTTPException(404, "no such picture")
    got = await store_of(world).load_chitra_data(picture_id)
    if got is None:
        raise HTTPException(404, "no such picture")
    mime, data = got
    return Response(content=data, media_type=mime or "image/jpeg", headers={"Cache-Control": "private, max-age=86400, immutable"})


# -- a player's portrait (Marc, 2026-11-11): a draft from how they look, changed by what they ask for, kept when they accept it ------------------

@contextlib.contextmanager
def _said():
    """A request the engine will not do, as the page is told it: 409 and the words for the player."""
    try:
        yield
    except (abhasa.Refused, abhasa.PicturesOff) as exc:
        raise HTTPException(409, str(exc))


async def _mine(pid: str, request: Request):
    world = await adhisthana.get_world()
    require_player(world, pid)
    require_own(request, world, pid)
    if abhasa.settings(world) is not None:
        await abhasa.ensure(world)   # the pictures the world already keeps, so a portrait that is final is known to be
    return world


@router.get("/players/{pid}/portrait")
async def portrait_state(pid: str, request: Request = None) -> dict:
    """Whether this world makes portraits, whether this player's is final, and the draft they are changing: version, what they asked for to make it,
    changes left, any trouble in words for them, and `rev` (the picture at /portrait/image changes when it does)."""
    return abhasa.draft_state(await _mine(pid, request), pid)


@router.post("/players/{pid}/portrait/draft")
async def portrait_draft(pid: str, req: PortraitDraftRequest, request: Request = None) -> dict:
    """A first draft from how the player looks (a new one replaces the last). Waits for the painter: a few seconds."""
    world = await _mine(pid, request)
    with _said():
        await abhasa.draft_start(world, pid, req.look)
    return abhasa.draft_state(world, pid)


@router.post("/players/{pid}/portrait/edit")
async def portrait_edit(pid: str, req: PortraitEditRequest, request: Request = None) -> dict:
    """What the player asks to change in the portrait shown: the model is given that picture and the request."""
    world = await _mine(pid, request)
    with _said():
        await abhasa.draft_edit(world, pid, req.ask)
    return abhasa.draft_state(world, pid)


@router.post("/players/{pid}/portrait/back")
async def portrait_back(pid: str, request: Request = None) -> dict:
    """Undo the last change."""
    world = await _mine(pid, request)
    with _said():
        abhasa.draft_back(world, pid)
    return abhasa.draft_state(world, pid)


@router.post("/players/{pid}/portrait/accept")
async def portrait_accept(pid: str, request: Request = None) -> dict:
    """The portrait shown is theirs, and final: kept in the world."""
    world = await _mine(pid, request)
    with _said():
        await abhasa.draft_accept(world, pid)
    return abhasa.draft_state(world, pid)


@router.get("/players/{pid}/portrait/image")
async def portrait_image(pid: str, rev: int = 0, request: Request = None) -> Response:
    """The draft as it stands (never cached: a change is a new picture)."""
    world = await _mine(pid, request)
    data = abhasa.draft_image(world, pid)
    if data is None:
        raise HTTPException(404, "no portrait draft")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
