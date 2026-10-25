# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Pratiṣṭhā's routes (2026-10-05, Marc's sign-off; the engine is
smritilila.smrti.pratistha, v4.61): directed growth, proposed and reviewed
before it is planted. A proposal is written as a JSON file under
PROPOSALS_DIR, proved against the world, read, released, watched,
rejected, or planted -- the plant backed up first and proved against the
database. A secret proposal is shown only to its holder.

Moved whole from main.py, nothing inside changed. In tests this module is
imported as pratistha_routes: pratistha there is the engine."""

from __future__ import annotations

import asyncio
import re
import os

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from smritilila.graph.canon import Maya
from smritilila.ids import new_id
from smritilila import adhisthana  # the services, called through it so a test's patch there reaches them
from smritilila.adhisthana import WorldSession, _store, _stream_of, app
from smritilila.memory.record import Record
from smritilila.paths import data_dir
from smritilila.sakshi import current_world, log_event
from smritilila.world_store import PostgresWorldStore
from smritilila.schemas import ManacitraRequest, PratisthaRequest, ProposalOut
from smritilila.routes import spanda  # its routes, and the names a test patches there, called through it
from smritilila.routes.bija import _seeder
import json as _json  # the proposals are JSON files


router = APIRouter()


# -- pratiṣṭhā: directed growth, proposed and reviewed before it is planted (v4.61) ----------------------------


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


PROPOSALS_DIR = data_dir() / "proposals"  # outside git: a proposal may hold a secret (the data folder: SMRITILILA_DATA_DIR, else the repo root)


LIVED_WINDOW = 120   # what the builder is shown of what was lived: the world's latest memories, never its history


PEOPLE_SHOWN = 150   # the beings shown it, the most recently active first


SECRET_HOLDER = "anu"  # who holds a secret today: the god at this machine (a request with no cf-ray)


def _secret_holder(request: Request | None) -> bool:
    """The god a secret is kept by: a request straight to this machine, not through Cloudflare."""
    return request is None or not request.headers.get("cf-ray")


def _secret_terms(world: "WorldSession") -> set[str]:
    """Every way a secret entity is named -- id, labels, declared names, handle words -- casefolded. Cost: the records."""
    secret = {r.entity_id for r in world.records if r.kind == "secret"}
    terms: set[str] = set()
    for r in world.records:
        if r.kind == "name" and r.entity_id in secret and r.value:
            terms.add(r.value.casefold())
    for eid in secret:
        e = world.canon.entities.get(eid)
        terms.add(eid.casefold())
        terms.add(re.sub(r"_[0-9a-f]{6}$", "", eid).replace("_", " ").casefold())
        if e is not None:
            terms |= {lab.casefold() for lab in e.labels if lab}
    return {t for t in terms if len(t) >= 4}


def _touches_secret(world: "WorldSession", *parts: object) -> bool:
    text = _json.dumps(parts, ensure_ascii=False).casefold()
    return any(re.search(r"(?<![\w])" + re.escape(t) + r"(?![\w])", text) for t in _secret_terms(world))


def _proposals_dir(world: "WorldSession") -> Path:
    from smritilila.backup import _safe_dirname
    d = PROPOSALS_DIR / _safe_dirname(world.name or current_world() or "world")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _proposal_out(p: dict, request: Request | None = None) -> ProposalOut:
    from smritilila.smrti.pratistha import summary
    if p.get("secret") and not _secret_holder(request):
        # option 1 (Marc, 2026-10-02): a proposal touching a secret is withheld from every god outside it
        return ProposalOut(id=p["id"], status=p["status"], title="(withheld: it touches a secret)", summary=[], errors=[],
                           warnings=[], proof={}, survey=[])
    return ProposalOut(id=p["id"], status=p["status"], title=p["spec"].get("title", ""), summary=summary(p["spec"]),
                       errors=p["errors"], warnings=p["warnings"], proof=p["proof"], survey=p.get("survey") or [], report=p.get("report") or {})


def _builder_view(world: "WorldSession") -> str:
    """What the builder is shown of the world, bounded: its places, its most active beings, a window of what was lived,
    and the players' latest. Cost: the world's places and beings once, and the window."""
    from smritilila.smrti.pratistha import world_view
    k = world.kosha()
    t = world.canon.world_time + 1
    names: dict[str, str] = {}
    for r in world.records:
        if r.kind == "name" and r.note != "alias":
            names[r.entity_id] = r.value
    places, people = [], []
    for eid, e in world.canon.entities.items():
        if not e.labels:
            continue
        if k.is_scene(eid) or k.is_place(eid, t):
            places.append((eid, e.labels[0], "scene" if k.is_scene(eid) else "region", k.container_at(eid, t)))
    last_by = world.samskara().mem.last_by
    beings = sorted((eid for eid in k.perceivers if eid in world.canon.entities and not k.is_scene(eid) and eid in last_by),
                    key=lambda eid: -last_by[eid].world_time)[:PEOPLE_SHOWN]
    for eid in beings:
        e = world.canon.entities[eid]
        people.append((eid, e.labels[0] if e.labels else eid, names.get(eid), k.scene_of(eid, t)))
    lived = [f"[{m.world_time}, {m.entity_id}] {m.experience[:220]}" for m in world.recent_memories[-LIVED_WINDOW:]]
    threads = []
    for pid in world.playable_entity_ids:
        own = _stream_of(world, pid)[-4:]
        threads += [f"[{pid}] {m.experience[:220]}" for m in own]
    return world_view(world.world_data.get("source"), world.bhuvana, places, people, lived, threads)


async def _check_and_prove(world: "WorldSession", spec: dict) -> tuple[list[str], list[str], dict]:
    """A proposal checked against the world as it now stands and, if it passes, proved read-only on a fork of it
    (pratistha.probe): (errors, warnings, proof). Shared by proposing and proving again (the pass of 2026-10-05)."""
    from smritilila.smrti.pratistha import check, plant, probe
    errors, warnings = check(spec, world.kosha(), set(world.playable_entity_ids), world.canon.world_time + 1)
    proof: dict = {}
    if errors:
        return errors, warnings, proof
    minute = spanda._players_latest(world) if world.samskara().kala is not None else None
    ents, mems, recs, ids = plant(spec, world.canon.world_time + 1, minute, "proof", lambda text: None)
    proof = await asyncio.to_thread(probe, world.canon, list(world.recent_memories), list(world.records), ents, mems, recs,
                                    list(world.playable_entity_ids), ids, spec)
    warnings += [f"in {p}'s view at once (planted beside them): {len(x)} of its memories reach their idle prompt"
                 for p, x in proof.get("leaks", {}).items() if x]
    errors += proof.get("stands", [])
    for label, r in (proof.get("reads") or {}).items():
        if isinstance(r, dict) and (r["shown when read"] < r["pages"] or r["shown when idle"]):
            errors.append(f"{label}: read whole showed {r['shown when read']} of {r['pages']} pages; idle showed {r['shown when idle']}")
    return errors, warnings, proof


@router.post("/world/pratistha", response_model=ProposalOut)
async def propose_pratistha(req: PratisthaRequest, request: Request) -> ProposalOut:
    """Pratiṣṭhā, on demand (Marc, 2026-10-02: "on demand first ... view each new build as a proposal first so we can
    check it over"): the builder surveys what the world is missing -- by its source and by its own talk -- chooses by
    importance and nearness (or builds a god's `focus`), writes it as a seed, checks it, and proves it read-only on a
    fork of the world. Nothing is planted: the proposal waits for a god's review (POST .../{id}/plant). Owner and the
    world's gods. Cost: two model calls, a retrieval per player."""
    from smritilila.smrti.pratistha import survey, write
    world = await adhisthana.get_world()
    roles = adhisthana._resolve_roles()
    if roles.smrti_client is None:
        raise HTTPException(503, "no model to build with")
    view = _builder_view(world)
    found: list[dict] = []
    if req.focus and req.focus.strip():
        what, why, anchor = req.focus.strip(), "", ""
    else:
        s = await asyncio.to_thread(survey, _seeder(roles.smrti_client), view)
        found = s["candidates"]
        if not found:
            raise HTTPException(502, "the builder found nothing missing")
        c = found[s["choice"]]
        what, why, anchor = c["what"], c["why"], c.get("anchor", "")
    spec = await asyncio.to_thread(write, _seeder(roles.smrti_client), view, what, why, anchor)
    errors, warnings, proof = await _check_and_prove(world, spec)
    pid = new_id()[:12]
    # holder first (Marc, 2026-10-02): word-matching missed a paraphrased secret (the Temple's first proposal), so in a
    # world that holds any secret every proposal goes to the holder, released to other gods only by the holder's word
    secret = req.secret or any(r.kind == "secret" for r in world.records) or _touches_secret(world, what, why, spec)
    if not _secret_holder(request):  # a candidate that names a secret is not shown in the survey either
        found = [c for c in found if not _touches_secret(world, c)]
    doc = {"id": pid, "status": "proposed", "created": _now_iso(), "focus": req.focus, "what": what, "why": why,
           "survey": found, "spec": spec, "errors": errors, "warnings": warnings, "proof": proof, "secret": secret}
    (_proposals_dir(world) / f"{pid}.json").write_text(_json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log_event("pratistha.proposed", id=pid, title=None if secret else spec.get("title"), errors=len(errors),
              warnings=len(warnings), focus=bool(req.focus), secret=secret)
    return _proposal_out(doc, request)


MAPS_DIR = data_dir() / "maps"  # outside git: a map is someone's drawing, kept here per world and named in a request by its file name


def _map_file(world: "WorldSession", name: str) -> Path:
    """A file in this world's maps folder, by name -- never a path: a request cannot reach anywhere else on the disk."""
    from smritilila.backup import _safe_dirname
    if not name or name != Path(name).name or name.startswith("."):
        raise HTTPException(422, "name a file in this world's maps folder, not a path")
    f = MAPS_DIR / _safe_dirname(world.name or current_world() or "world") / name
    if not f.is_file():
        raise HTTPException(404, f"no map {name!r} in this world's maps folder")
    return f


@router.post("/world/manacitra", response_model=ProposalOut)
async def propose_manacitra(req: ManacitraRequest, request: Request) -> ProposalOut:
    """Mānacitra (Marc, 2026-10-21: "expand stonemarch based on the map including every named place on the map, roads, mountains, rivers, forests,
    swamps"): the builder reads a map -- the great areas from the whole sheet, then every name from each part of it -- and writes what it read
    as one seed of places, checked and proved read-only as any proposal is. Nothing is planted: the proposal waits for a god's review
    (POST .../{id}/plant). Owner and the world's gods. Cost: for each of `passes` readings (three by default), one model call for the areas
    and one for each part (cols x rows), each carrying the picture, then a retrieval per player to prove it. The seed is secret only if it
    names a secret or is asked to be: a map's geography is not one."""
    from smritilila.smrti import manacitra
    world = await adhisthana.get_world()
    roles = adhisthana._resolve_roles()
    if roles.smrti_client is None:
        raise HTTPException(503, "no model to build with")
    earth = adhisthana._earth(world.canon)
    if earth is None:
        raise HTTPException(409, "this world has no earth to file a map under")
    picture = await asyncio.to_thread(manacitra.load_picture, _map_file(world, req.image))
    k, t = world.kosha(), world.canon.world_time + 1
    known = [(eid, e.labels[0], "scene" if k.is_scene(eid) else "region") for eid, e in world.canon.entities.items()
             if e.labels and (k.is_scene(eid) or k.is_place(eid, t))]
    try:
        spec, report, readings = await asyncio.to_thread(manacitra.read_map, _seeder(roles.smrti_client), world.world_data.get("source"), picture,
                                                         known, set(world.canon.entities), earth, cols=req.cols, rows=req.rows, passes=req.passes)
    except manacitra.MapReadFailed as exc:
        raise HTTPException(502, f"the map could not be read whole: {exc}")
    errors, warnings, proof = await _check_and_prove(world, spec)
    pid = new_id()[:12]
    secret = req.secret or _touches_secret(world, spec)
    doc = {"id": pid, "status": "proposed", "created": _now_iso(), "focus": req.image, "what": f"the map {req.image}", "why": spec["why"],
           "survey": [], "spec": spec, "errors": errors, "warnings": warnings, "proof": proof, "secret": secret, "report": report,
           "readings": readings}
    (_proposals_dir(world) / f"{pid}.json").write_text(_json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log_event("pratistha.proposed", id=pid, title=None if secret else spec.get("title"), errors=len(errors), warnings=len(warnings), focus=True,
              secret=secret, map=req.image, places=len(spec["places"]))
    return _proposal_out(doc, request)


@router.get("/world/pratistha", response_model=list[ProposalOut])
async def list_pratistha(request: Request) -> list[ProposalOut]:
    world = await adhisthana.get_world()
    docs = [_json.loads(p.read_text(encoding="utf-8")) for p in sorted(_proposals_dir(world).glob("*.json"))]
    return [_proposal_out(d, request) for d in sorted(docs, key=lambda d: d.get("created", ""))]


@router.get("/world/pratistha/{pid}")
async def read_pratistha(pid: str, request: Request = None) -> dict:
    world = await adhisthana.get_world()
    f = _proposals_dir(world) / f"{pid}.json"
    if not re.fullmatch(r"[0-9a-f]{12}", pid) or not f.exists():
        raise HTTPException(404, "no such proposal")
    doc = _json.loads(f.read_text(encoding="utf-8"))
    if doc.get("secret") and not _secret_holder(request):
        raise HTTPException(403, "this proposal touches a secret")
    return doc


@router.post("/world/pratistha/{pid}/prove", response_model=ProposalOut)
async def prove_pratistha(pid: str, request: Request) -> ProposalOut:
    """A proposal checked and proved again against the world as it now stands -- worlds move between proposing and
    planting, and the engine's own rules may have grown. Nothing is planted."""
    world = await adhisthana.get_world()
    doc = await read_pratistha(pid, request)
    if doc["status"] != "proposed":
        raise HTTPException(409, f"this proposal is {doc['status']}")
    spec = doc["spec"]
    errors, warnings, proof = await _check_and_prove(world, spec)
    doc["errors"], doc["warnings"], doc["proof"] = errors, warnings, proof
    (_proposals_dir(world) / f"{pid}.json").write_text(_json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log_event("pratistha.proved", id=pid, errors=len(errors), warnings=len(warnings))
    return _proposal_out(doc, request)


@router.post("/world/pratistha/{pid}/release", response_model=ProposalOut)
async def release_pratistha(pid: str, request: Request) -> ProposalOut:
    """The secret's holder, having read a withheld proposal, releases it to every god: it touches no secret. Only the
    holder may (a request from anywhere else is refused before it reads the proposal)."""
    world = await adhisthana.get_world()
    if not _secret_holder(request):
        raise HTTPException(403, "only the secret's holder releases a proposal")
    doc = await read_pratistha(pid, request)
    doc["secret"] = False
    (_proposals_dir(world) / f"{pid}.json").write_text(_json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log_event("pratistha.released", id=pid)
    return _proposal_out(doc, request)


@router.get("/world/pratistha/{pid}/watch")
async def watch_pratistha(pid: str, request: Request) -> dict:
    """A planted seed, watched (Marc, 2026-10-02: "You'll have to track both"): for each being it planted, the turns of
    its own since planting, the latest, and the players whose memories have met it; its breath, if any. Read-only.
    Withheld as its proposal is. Cost: the planted beings' memories."""
    world = await adhisthana.get_world()
    doc = await read_pratistha(pid, request)
    if doc["status"] != "planted":
        raise HTTPException(409, f"this proposal is {doc['status']}")
    ids = doc.get("ids") or {}
    mem = world.samskara().mem
    # its places' and people's first memories are at the tick it was planted; a written thing's words, at the world's start
    planted_at = max((m.world_time for m in world.recent_memories if m.pulse_id == f"pratistha:{pid}"), default=None)
    players = set(world.playable_entity_ids)
    beings = {}
    for key, eid in ids.items():
        own = [m for m in mem.by_entity.get(eid, ()) if m.pulse_id != f"pratistha:{pid}"]
        met = sorted({p for p in players for m in mem.by_entity.get(p, ()) if eid in m.subject_ids})
        e = world.canon.entities.get(eid)
        beings[key] = {"label": e.labels[0] if e and e.labels else eid, "turns_since": len({m.pulse_id for m in own}),
                       "latest": own[-1].world_time if own else None, "met_by_players": met}
    breath = (doc["spec"].get("breath") or {})
    return {"id": pid, "planted_at_tick": planted_at, "world_tick": world.canon.world_time, "breath": breath, "beings": beings}


@router.post("/world/pratistha/{pid}/reject", response_model=ProposalOut)
async def reject_pratistha(pid: str, request: Request) -> ProposalOut:
    world = await adhisthana.get_world()
    doc = await read_pratistha(pid, request)
    doc["status"] = "rejected"
    (_proposals_dir(world) / f"{pid}.json").write_text(_json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log_event("pratistha.rejected", id=pid)
    return _proposal_out(doc, request)


def _db_proof(database_url: str, canon: Maya, ents: list, mems: list, recs: list) -> None:
    """The seed written into a copy of the world's own database, then the copy dropped: what only the store enforces
    (the book, 2026-10-02: a memory filed in no place broke a NOT NULL the fork never saw) fails here, not live."""
    import subprocess
    import tempfile
    from smritilila.backup import _pg_dump_path, _pg_dump_url
    bin_dir = Path(_pg_dump_path()).parent
    plain = _pg_dump_url(database_url)
    base, _, db = plain.rpartition("/")
    copy = f"{db}_pratistha_proof"
    env = {**os.environ}
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "w.dump"
        subprocess.run([str(bin_dir / Path(_pg_dump_path()).name), "-Fc", "-f", str(dump), plain], check=True, env=env, capture_output=True)
        admin = f"{base}/postgres"
        psql = str(bin_dir / ("psql.exe" if os.name == "nt" else "psql"))
        subprocess.run([psql, admin, "-q", "-c", f"drop database if exists {copy}", "-c", f"create database {copy}"], check=True, env=env, capture_output=True)
        try:
            restore = str(bin_dir / ("pg_restore.exe" if os.name == "nt" else "pg_restore"))
            subprocess.run([restore, "--no-owner", "-d", f"{base}/{copy}", str(dump)], check=True, env=env, capture_output=True)
            from smritilila.persistence.engine import make_engine, make_session_factory

            async def commit() -> None:
                engine = make_engine(database_url.rsplit("/", 1)[0] + f"/{copy}")
                try:
                    await PostgresWorldStore(make_session_factory(engine)).commit(canon, mems, ents, records=recs)
                finally:
                    await engine.dispose()
            asyncio.run(commit())
        finally:
            subprocess.run([psql, admin, "-q", "-c", f"drop database if exists {copy}"], env=env, capture_output=True)


@router.post("/world/pratistha/{pid}/plant", response_model=ProposalOut)
async def plant_pratistha(pid: str, request: Request) -> ProposalOut:
    """A reviewed proposal planted: checked again against the world as it now stands, a backup taken, the seed written
    first into a copy of the world's database (what only the store enforces), then live -- inserts only -- and the
    world reloaded. Owner and the world's gods."""
    from smritilila.backup import backup_world
    from smritilila.smrti.pratistha import check, plant
    world = await adhisthana.get_world()
    doc = await read_pratistha(pid, request)
    if doc["status"] != "proposed":
        raise HTTPException(409, f"this proposal is {doc['status']}")
    spec = doc["spec"]
    t = world.canon.world_time + 1
    errors, _warnings = check(spec, world.kosha(), set(world.playable_entity_ids), t)
    if errors or doc["errors"]:
        raise HTTPException(409, "the proposal does not stand: " + "; ".join(errors or doc["errors"]))
    entry = adhisthana._active_world_registry_entry()
    if entry is None:
        raise HTTPException(409, "no registered world to back up first")
    await asyncio.to_thread(backup_world, entry, reason="pratistha")
    now = world.canon.world_time + 1
    minute = spanda._players_latest(world) if world.samskara().kala is not None else None
    embed = adhisthana.get_embedder().embed
    ents, mems, recs, ids = plant(spec, now, minute, f"pratistha:{pid}", embed)
    if doc.get("secret"):  # grown from a secret, or asked to be one: born secret
        recs += [Record(new_id(), "secret", e.id, SECRET_HOLDER, now, f"pratistha:{pid}") for e in ents]
    try:
        await asyncio.to_thread(_db_proof, entry.database_url, world.canon, ents, mems, recs)
    except Exception as exc:  # noqa: BLE001 -- the copy refused it: the live world is never touched
        raise HTTPException(409, f"the seed failed against a copy of the world's database: {exc!r}"[:600])
    await _store(world).commit(world.canon, mems, ents, records=recs)
    doc["status"], doc["planted"], doc["ids"] = "planted", _now_iso(), ids
    (_proposals_dir(world) / f"{pid}.json").write_text(_json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log_event("pratistha.planted", id=pid, title=None if doc.get("secret") else spec.get("title"), entities=len(ents),
              memories=len(mems), records=len(recs), secret=bool(doc.get("secret")))
    if hasattr(app.state, "world"):
        del app.state.world  # the store is the truth now (as absorb reloads)
    return _proposal_out(doc, request)
