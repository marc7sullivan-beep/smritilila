# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Dvāra (2026-10-05, Marc's sign-off): the gate -- who may pass. The
owners, a guest's identity and the worlds open to guests, which paths
each may take, whether a guest plays the character a request names, the
alert sent when a guest comes back, and the two checks a route makes
inside itself (_require_own, _require_player). The middleware that
stands at the door, guest_gate, stays with the app in main.py and reads
its tables here.

Moved whole from main.py, nothing inside changed. Names a test patches
(_guest, _plays, _god_of_active, _guest_seen, _schedule_alert) are
called from other modules as dvara.X, never imported bare: one binding,
so a test's patch here reaches every caller."""

from __future__ import annotations

import asyncio
import re
import os
import time

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila.adhisthana import PLAYER_IDLE_SECONDS, WorldSession
from smritilila.sakshi import log_event


def _open_to_guests(entry, guest: str | None = None) -> bool:
    """Open to every guest, or -- a world naming its guests -- to this one."""
    guests = getattr(entry, "guests", False)
    if entry.name in GUEST_WORLDS or guests is True:
        return True
    return isinstance(guests, tuple) and bool(guest) and guest in guests


# -- The Host allow-list (2026-10-18, from the security audit of the machine this runs on) ----------------------
#
# Everything below trusts the request's headers: no Cloudflare headers means the owner, and a Cloudflare email
# header names the visitor. That is sound only while the requests come through the tunnel. A web page in Marc's own
# browser can reach 127.0.0.1 too -- DNS rebinding points an attacker's name at it -- and then sends any header it
# likes, the owner's email among them. What it cannot do is change the Host it is reached by: that is the attacker's
# own name. So the first thing the door asks of every request, tunnel or not, is whether the Host is a name this
# server answers to: this machine, or example.com and its subdomains (the tunnel keeps the public Host unless the
# tunnel's own setting rewrites it, and a rewrite to localhost is allowed too).
#
# SMRITILILA_ALLOWED_HOSTS: the domains this server is reached by through a tunnel, comma-separated; each allows itself and
# every subdomain ("*." or a leading dot is accepted and ignored). This machine's own names are always allowed.
# SMRITILILA_HOST_GUARD: "enforce" (default) refuses with 400; "report" logs what it would refuse and lets it through, for
# the first run on a real tunnel; "off" is the old behaviour.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _domains(raw: str) -> tuple[str, ...]:
    return tuple(d.strip().lower().lstrip("*.").rstrip(".") for d in raw.split(",") if d.strip())


ALLOWED_DOMAINS = _domains(os.environ.get("SMRITILILA_ALLOWED_HOSTS", "example.com"))
HOST_GUARD = os.environ.get("SMRITILILA_HOST_GUARD", "enforce").strip().lower()
_odd_hosts: dict[str, int] = {}  # Hosts refused (or, in report mode, noted), by count: logged once each, up to _ODD_HOSTS_MAX


_ODD_HOSTS_MAX = 64


def _host_of(request: Request) -> str:
    """The Host header as a bare lower-case name: no port, no trailing dot, an IPv6 literal without its brackets.
    "" when the header is missing or malformed."""
    raw = (request.headers.get("host") or "").strip().lower()
    if raw.startswith("["):  # [::1]:8198
        end = raw.find("]")
        return raw[1:end] if end > 0 else ""
    host = raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw
    return host.rstrip(".")


def _host_allowed(host: str) -> bool:
    """Exact names only for this machine; each allowed domain and its subdomains ("evil" + the domain is not one)."""
    if not host:
        return False
    if host in LOCAL_HOSTS:
        return True
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def _host_refusal(request: Request) -> JSONResponse | None:
    """None to let the request on; a 400 to stop it at the door (guest_gate calls this before anything else)."""
    if HOST_GUARD == "off":
        return None
    host = _host_of(request)
    if _host_allowed(host):
        return None
    enforcing = HOST_GUARD != "report"
    if host in _odd_hosts or len(_odd_hosts) < _ODD_HOSTS_MAX:
        first = host not in _odd_hosts
        _odd_hosts[host] = _odd_hosts.get(host, 0) + 1
        if first:
            log_event("host.refused" if enforcing else "host.would_refuse", host=host[:200], method=request.method,
                      path=request.url.path, via_cloudflare=bool(request.headers.get("cf-ray")))
    return JSONResponse(status_code=400, content={"detail": "host not allowed"}) if enforcing else None


# -- Guests (Marc, 2026-09-27: "allow only stonemarch access to anyone besides us")
#
# example.com reaches this server through a Cloudflare Tunnel behind
# Cloudflare Access, which names each visitor by the email they logged in
# with (Cf-Access-Authenticated-User-Email). An OWNER -- Marc's email, or a
# request from this machine, which carries no Cloudflare headers at all --
# may do anything. Anyone else is a GUEST: only while a guest world is the
# active one (Stonemarch), no world administration, and only their own
# characters -- the ones they joined as (the tether, `played_by`). A request
# that came through Cloudflare with no email is a guest who owns nothing:
# the gate fails closed.
OWNERS = frozenset(e.strip().lower() for e in os.environ.get("SMRITILILA_OWNERS", "owner@example.com").split(",") if e.strip())


# Gods (Marc, 2026-09-30): Marc and Anu (this machine) are gods of every world -- OWNERS above; a world's
# registry entry names its other gods, the authors Marc invited. Only Marc invites, for now; only Marc and
# Anu switch what a shared server shows (/worlds/select).
INVITER = os.environ.get("SMRITILILA_INVITER", "owner@example.com").strip().lower()


# what a world's own gods may do in it, beside Marc and Anu: most of what an owner does to that world (not /worlds/*,
# /admin*, /builder/*, or GET /world/npcs and /world/cast: owner only)
_WORLD_GODS_MAY = {("POST", "/world/clear"), ("GET", "/world/view"), ("PUT", "/world/view"), ("POST", "/world/bhuvana"), ("POST", "/world/describe"),
                   ("POST", "/world/expand"), ("POST", "/genesis")}


_WORLD_GODS_MAY_PREFIX = ("/world/playable/", "/world/becoming/", "/world/containment/", "/world/pratistha", "/world/manacitra", "/world/held")


_DECIDED_INSIDE = {("POST", "/worlds/create"), ("POST", "/worlds/invite")}  # the endpoint itself asks who is asking

# Marc, 2026-10-05: "Block guests from world creation". An invited guest could make a world of their own, on a server of its own
# (2026-09-30); no guest may now, invited or not. The invited-author path in /worlds/create stays, behind this one switch,
# for the day it is reopened.
GUESTS_MAY_CREATE_WORLDS = False


GUEST_WORLDS = frozenset(w.strip() for w in os.environ.get("SMRITILILA_GUEST_WORLDS", "Stonemarch").split(",") if w.strip())


# what only an owner may do: (method, path); and by path prefix, _OWNER_ONLY_PREFIX below
_OWNER_ONLY = {("POST", "/world/clear"), ("GET", "/world/view"), ("PUT", "/world/view"), ("POST", "/world/bhuvana"), ("POST", "/world/describe"),
               ("POST", "/world/expand"), ("POST", "/genesis"), ("POST", "/worlds/create"), ("POST", "/worlds/select"), ("GET", "/admin"), ("GET", "/world/npcs"), ("GET", "/world/cast")}


_OWNER_ONLY_PREFIX = ("/worlds/", "/world/playable/", "/world/becoming/", "/world/containment/", "/world/pratistha", "/world/manacitra", "/world/held",
                      "/builder/", "/admin/")  # the admin screen's routes (2026-10-06); /worlds/{name}/backup; a being made playable; the builder (2026-10-06): an owner makes worlds, every route of it


_OPEN_TO_ANY = {"/", "/health", "/worlds", "/favicon.ico", "/me"}  # never world-scoped; /worlds lists only guest worlds to a guest


_PLAYER_PATH = re.compile(r"^/(?:players|rounds|entities)/([^/]+)")


def _guest(request: Request) -> str | None:
    """The visitor's email if they are a guest; None for an owner. "" for a
    visitor who came through Cloudflare unnamed (a guest who owns nothing)."""
    email = request.headers.get("cf-access-authenticated-user-email")
    if email is None:
        return "" if request.headers.get("cf-ray") else None
    return None if email.strip().lower() in OWNERS else email.strip().lower()


def _plays(world: "WorldSession", guest: str, pid: str) -> bool:
    return bool(guest) and world.played_by.get(pid) == guest


def _god_of_active(guest: str | None) -> bool:
    """A named visitor who is one of the active world's own gods (its authors)."""
    entry = adhisthana._active_world_registry_entry() if guest else None
    return entry is not None and guest in entry.gods


def _refuse(detail: str) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": detail})


# Alerts (Marc, 2026-10-01: "setup a text message when a guest comes online"; ntfy chosen): a guest's first request
# after PLAYER_IDLE_SECONDS away -- the page polls every few seconds while open, so this is opening Smritilila after being
# away -- sends one push to the ntfy topic SMRITILILA_NTFY_TOPIC. The topic is private by obscurity: anyone who knows it
# reads the alerts, so it lives in the environment, never the repo. Unset: nothing is sent. Seen per process: a
# restart counts every guest as away once.
NTFY_URL = os.environ.get("SMRITILILA_NTFY_URL", "https://ntfy.sh").rstrip("/")


_guest_seen: dict[str, float] = {}


_alerts_in_flight: set = set()


def _guest_online(guest: str) -> bool:
    """True on a guest's first request, or their first after being away."""
    now = time.monotonic()
    last = _guest_seen.get(guest)
    _guest_seen[guest] = now
    return last is None or now - last >= PLAYER_IDLE_SECONDS


async def _alert(title: str, message: str, click: str | None = None) -> None:
    """One ntfy push; never raises -- an alert that fails is logged, and the request it rode on is untouched."""
    topic = os.environ.get("SMRITILILA_NTFY_TOPIC", "").strip()
    if not topic:
        return
    import httpx
    params = {"title": title, "tags": "door"} | ({"click": click} if click else {})  # query params: a world's name may not be latin-1
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{NTFY_URL}/{topic}", params=params, content=message.encode("utf-8"))
        log_event("alert.sent", status=r.status_code, title=title)
    except Exception as exc:
        log_event("alert.failed", title=title, error=str(exc)[:200])


def _schedule_alert(title: str, message: str, click: str | None = None) -> None:
    task = asyncio.get_running_loop().create_task(_alert(title, message, click))
    _alerts_in_flight.add(task)  # held until done: a bare task may be collected mid-flight
    task.add_done_callback(_alerts_in_flight.discard)


def _require_own(request: Request, world: "WorldSession", pid: str | None) -> None:
    """A guest acts only as their own character (the bodies of /pulses and /pulses/confirm)."""
    guest = _guest(request) if request is not None else None  # called in-process: this machine
    if guest is not None and not _plays(world, guest, pid or "") and not _god_of_active(guest):
        log_event("guest.refused", guest=guest, path=request.url.path, reason="not_their_character", entity_id=pid)
        raise HTTPException(403, "that character is not yours")


def _require_player(world: WorldSession, pid: str) -> None:
    if pid not in world.playable_entity_ids:
        raise HTTPException(404, f"{pid} is not a player")
