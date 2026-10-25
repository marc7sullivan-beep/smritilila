# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The HTTP entrypoint for Smritilila: the app, the gate at its door, and the
routes stitched together. Run it as `smritilila.main:app` (scripts/dev_server.py):
importing this module is what includes every router -- adhisthana's app on
its own serves none of them.

Main.py held all of it until 2026-10-05, when it was split, at Marc's
word, into modules named for what they are (each moved verbatim):
- smritilila.adhisthana -- the seat the routes stand on: the app object and its
  lifespan, which world this port serves, the WorldSession, its store, the
  embedder and the generative roles.
- smritilila.dvara -- the gate's tables and checks; guest_gate, below, stands
  at the door and reads them.
- smritilila.routes.spanda -- the pulse: /pulses, consent and the window's
  turns, the rounds' routes, the commit, activation, phala's reckoning.
- smritilila.routes.bija -- a world's birth and growth: genesis, joining,
  expand, describe, the bhuvana and the world's view.
- smritilila.routes.pratistha -- directed growth, proposed and reviewed before
  it is planted.
- smritilila.routes.loka -- the worlds: list, select, create, invite, back up,
  exist, clear.
- smritilila.routes.upasthiti -- the players and their presence.
- smritilila.routes.darshana -- what one entity is shown.
- smritilila.routes.sthapati -- the master builder: the conversation in which a world's brief is
  made (2026-10-06), before /worlds/create and /genesis.
- smritilila.routes.adhyaksa -- the overseer: the owner's admin screen (2026-10-06), the worlds and their guests.
And, before them: smritilila.itihasa (the age), smritilila.schemas (the API's
shapes), smritilila.sangama (the rounds' bookkeeping), smritilila.pravrtti (the
world's own draw). The paragraphs this docstring held about each part
went with it, into that module's docstring.

The services a test patches (adhisthana's, dvara's) are called through
their module (`adhisthana.get_world()`), never imported bare, so one patch
reaches every caller; tests fake the model call on spanda, where it is
made.

Its own history, kept ("below" was the v3 account now spread over those modules):
This docstring described v1.4 until 2026-09-22: the Entity/Relation
graph, the delta log, the spanda pulse pipeline, /world/reset's starter
world, and the activation-pulse machinery. All of it was deleted with
v2 on 2026-09-21 and the description was simply left behind -- the
first thing a reader met was an account of an engine that was no
longer here. Corrected below; what the v2 paragraphs recorded about
WHY a decision was made is kept where it still explains the code.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse

from smritilila import adhisthana, dvara  # called through, so a test's patch there reaches every caller
from smritilila.adhisthana import app
from smritilila.dvara import (
    _DECIDED_INSIDE, _OPEN_TO_ANY, _OWNER_ONLY, _OWNER_ONLY_PREFIX, _PLAYER_PATH, _WORLD_GODS_MAY,
    _WORLD_GODS_MAY_PREFIX, _guest_online, _open_to_guests, _refuse,
)
from smritilila import users_registry
from smritilila.routes import adhyaksa, bija, darshana, loka, pratistha, spanda, sthapati, upasthiti
from smritilila.sakshi import code_version, log_event

# The build stamp, resolved as the server's modules load rather than at
# the first logged line (2026-09-23): resolved lazily, a commit landing
# between process start and the first call would stamp this process
# with code it is not running.
code_version()


_WEB_INDEX = Path(__file__).parent / "web" / "index.html"


@app.middleware("http")
async def guest_gate(request: Request, call_next):
    refused = dvara._host_refusal(request)  # first, for tunnel and local alike: the headers below are only as good as the Host they came in on
    if refused is not None:
        return refused
    guest = dvara._guest(request)
    if guest is None:
        return await call_next(request)
    if guest and _guest_online(guest):  # named: an unnamed visitor ("") owns nothing, and is nobody to announce
        entry = adhisthana._active_world_registry_entry()
        name = entry.name if entry is not None else "Smritilila"
        log_event("guest.online", guest=guest, world=name)
        dvara._schedule_alert(f"{guest} is on {name}", f"{guest} came online in {name}.", getattr(entry, "url", None))
    path, method = request.url.path, request.method
    if (method, path) in _DECIDED_INSIDE:
        return await call_next(request)
    if path.startswith("/builder/") and users_registry.may_build(guest):
        return await call_next(request)  # a registered user the owner has enabled for the builder (2026-10-06); the endpoints say whose job is whose
    god = dvara._god_of_active(guest)
    if god and ((method, path) in _WORLD_GODS_MAY or path.startswith(_WORLD_GODS_MAY_PREFIX)
                or ("/absorb/" in path and path.startswith("/entities/"))):
        return await call_next(request)  # the world's own god, shaping their world
    if (method, path) in _OWNER_ONLY or path.startswith(_OWNER_ONLY_PREFIX) or ("/absorb/" in path and path.startswith("/entities/")):
        log_event("guest.refused", guest=guest, method=method, path=path, reason="owner_only")
        return _refuse("only the world's owner may do that")
    if god:
        return await call_next(request)  # in their own world, as its owner is
    if path not in _OPEN_TO_ANY:
        entry = adhisthana._active_world_registry_entry()
        if entry is None or not _open_to_guests(entry, guest):
            log_event("guest.refused", guest=guest, method=method, path=path, reason="world_closed")
            return _refuse("this world is not open to guests right now")
        m = _PLAYER_PATH.match(path)
        if m:
            world = await adhisthana.get_world()
            pid = m.group(1)
            if pid in world.playable_entity_ids and not dvara._plays(world, guest, pid):
                log_event("guest.refused", guest=guest, method=method, path=path, reason="not_their_character")
                return _refuse("that character is not yours")
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def web_ui() -> FileResponse:
    """The lightweight web UI -- a single self-contained HTML/CSS/JS file
    with no build step, no framework and no external requests. Talks to
    this app's own JSON API via fetch(); nothing here is server-rendered."""
    return FileResponse(_WEB_INDEX, headers={"Cache-Control": "no-store"})


# The routes, by group: each module's router, included here. No path is answered by two modules, so the
# order does not decide which route answers.
app.include_router(spanda.router)
app.include_router(bija.router)
app.include_router(pratistha.router)
app.include_router(loka.router)
app.include_router(upasthiti.router)
app.include_router(darshana.router)
app.include_router(sthapati.router)
app.include_router(adhyaksa.router)
