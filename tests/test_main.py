# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""API-level tests using FastAPI's TestClient, against the v3 API surface
(the v1.4 Entity/Relation engine behind it was removed with v2, 2026-09-21). Most tests monkeypatch
smritilila.adhisthana._resolve_roles with fakes -- no THE_FIRST_API_KEY/the second model_API_KEY
is available in this environment (or needed: these test routing,
request/response shaping, and world-state mutation across requests, not
model quality).
`_resolve_roles` is deliberately NOT wired as a FastAPI Depends() (see its
own docstring), so tests patch the module attribute directly rather than
using app.dependency_overrides.
"""

from __future__ import annotations
from smritilila import schemas
from smritilila import sangama
from smritilila import itihasa
from smritilila import pravrtti
from smritilila import adhisthana
from smritilila import dvara
from smritilila.routes import spanda
from smritilila.routes import bija as bija_routes
from smritilila.routes import loka

# The isolation (marker, registry, invitations) wherever this module is imported from: a scratch script that
# imported it outside pytest ran every test here against the real marker and Stonemarch's database (2026-09-30).
import conftest  # noqa: F401,E402

import json
import logging
import os
import zlib

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import smritilila.main as main
import smritilila.sangama as sangama
from smritilila.graph.canon import Maya
from smritilila.memory.record import Record
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.adhisthana import GenerativeRoles, WorldSession
from smritilila.main import app

class _SpyNarrator:
    """A v2 relic kept as a stub: the v3 tests assert it is never called."""

    def __init__(self):
        self.resolved_events: list[str] = []


class _FakeEmbedder:
    """Deterministic, near-instant -- these tests must never load the
    real local sentence-transformers model (a real, multi-second model
    load adhisthana.get_embedder's real SentenceTransformerEmbedder would
    trigger on first use).

    Word-hashed into a wider, sign-varying vector -- NOT a crude
    (len % 7, hash % 11) scalar pair. That original version was fine as
    long as nothing ever compared two embeddings for similarity, but
    provenance.find_similar_known_referent (v2, since removed) did exactly that: two
    small, always-non-negative components put every embedding in the
    same quadrant, so cosine similarity came out spuriously high (>0.85)
    between texts sharing no actual meaning ("the newcomer" vs "a quiet
    crossroads"), incorrectly merging unrelated genesis/join referents in
    several tests. Shared words between two texts now reinforce (higher
    similarity); unrelated words contribute effectively random +-1 noise
    across dimensions that mostly cancels out (low similarity) -- close
    enough to how a real sentence embedding actually discriminates.

    zlib.crc32, deliberately NOT Python's builtin hash(): str hashing is
    randomized per-process (PYTHONHASHSEED) unless disabled, which would
    make any test asserting a specific match/no-match outcome flaky
    across separate runs."""

    _DIM = 32

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._DIM
        for word in text.lower().split():
            h = zlib.crc32(word.encode())
            for i in range(self._DIM):
                vec[i] += 1.0 if (h >> i) & 1 else -1.0
        return vec


def _fake_roles(narrator=None) -> GenerativeRoles:
    roles = GenerativeRoles()
    roles.narrator = narrator or _SpyNarrator()  # a v2 relic the v3 tests assert is never called
    return roles


class _Commit:
    """One recorded commit: what a real store would have written."""

    def __init__(self, memories, entities, touched_ids):
        self.memories_written = list(memories)
        self.promoted_entities = list(entities)
        self.touched_ids = list(touched_ids)


class _FakeWorldStore:
    """Records commit() calls instead of touching a database."""

    def __init__(self, initial=None, session_config=None):
        self._loaded = initial
        self._session_config = session_config
        self.commit_calls: list[tuple] = []
        self.extra_entity_ids_calls: list = []
        self.clear_calls = 0
        self._memories: list = []  # every memory ever committed, in commit order
        self._records: list = []  # v4: every record ever committed

    async def load(self):
        return self._loaded

    async def commit(self, canon, memories, entities, touched_ids=(), records=()):
        result = _Commit(memories, entities, touched_ids)
        self.commit_calls.append((canon, result, None))
        self._loaded = canon  # a real store's load() would reflect this too
        self._memories.extend(memories)
        self._records.extend(records)

    async def clear(self):
        self.clear_calls += 1
        self._loaded = None
        self._session_config = None
        self._memories = []

    async def last_memory_for(self, entity_id):
        matches = [m for m in self._memories if m.entity_id == entity_id]
        return max(matches, key=lambda m: m.world_time) if matches else None

    async def load_memories(self):
        return list(self._memories)

    async def load_records(self):
        return list(self._records)

    async def save_autonomy(self, player_ids):
        self._autonomy = sorted(set(player_ids))

    async def load_autonomy(self):
        return list(getattr(self, "_autonomy", []))

    async def save_logged_off(self, player_ids):
        self._logged_off = sorted(set(player_ids))

    async def load_logged_off(self):
        return list(getattr(self, "_logged_off", []))

    async def save_waiting(self, player_ids):
        self._waiting = sorted(set(player_ids))

    async def load_waiting(self):
        return list(getattr(self, "_waiting", []))

    async def save_held_back(self, ids):
        self._held_back = sorted(set(ids))

    async def load_held_back(self):
        return list(getattr(self, "_held_back", []))

    async def save_view(self, darsana, becomings, world_data):
        self._view = (darsana, dict(becomings), dict(world_data))

    async def load_view(self):
        return getattr(self, "_view", (None, {}, {}))

    async def save_session_config(
        self, scene_owner_entity_id, entry_point_location_ids, playable_entity_ids, setting_description=None
    ):
        self._session_config = (
            scene_owner_entity_id, list(entry_point_location_ids), list(playable_entity_ids), setting_description,
        )

    async def load_session_config(self):
        return self._session_config

    async def save_bhuvana(self, text):
        self.bhuvana = text

    async def absorb(self, keep, gone):
        self.absorbed = (keep, gone)
        return {"tuples": 2, "filed": 1, "presence": 3, "dropped": 1}

    async def load_bhuvana(self):
        return getattr(self, "bhuvana", None)

    async def save_played_by(self, played_by):
        self.played_by = dict(played_by)

    async def load_played_by(self):
        return dict(getattr(self, "played_by", {}))

@pytest.fixture(autouse=True)
def _clean_world_state():
    yield
    for attr in ("world", "world_store", "worlds_registry", "active_database_url"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: _FakeEmbedder())


def _seed_square_world(client: TestClient) -> WorldSession:
    """A minimal v3 world put straight onto app.state: one player standing
    in one square, each with a first tuple -- what the v2 /world/reset
    starter world used to give these tests."""
    canon = Maya(world_time=0, entities={
        "player": Entity("player", ["the player"], 0, "genesis"),
        "square": Entity("square", ["Town Square"], 0, "genesis", avarana="open"),
    })
    world = WorldSession(canon=canon, scene_owner_entity_id="player", entry_point_location_ids=["square"], playable_entity_ids=["player"])
    world.recent_memories = [
        Memory("g-square", "square", frozenset(), "A town square, the lane running off it.", "AMBIENT", 5, "square", 0, "genesis", None),
        Memory("g-player", "player", frozenset(), "I stand in the town square in the morning.", "PRIMARY", 3, "square", 0, "genesis", None),
    ]
    client.app.state.world = world
    return world


def _install_fake_store(monkeypatch: pytest.MonkeyPatch, initial=None, session_config=None) -> _FakeWorldStore:
    store = _FakeWorldStore(initial, session_config)
    monkeypatch.setattr(adhisthana, "get_world_store", lambda: store)
    return store


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> _FakeWorldStore:
    return _install_fake_store(monkeypatch)


@pytest.fixture
def client():
    return TestClient(app)


def test_health_does_not_require_an_api_key(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_lifespan_warms_the_embedder_before_serving_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # Live investigation, 2026-09-08: get_embedder() already caches its
    # SentenceTransformerEmbedder on app.state (only ever constructed
    # once per process), but "once" used to mean "whenever the first
    # live /pulses request needs it" -- live-measured at several seconds
    # even after stripping an unwanted network round-trip out of
    # construction itself (llm/embedder.py). The lifespan handler moves
    # that one-time cost to process startup instead.
    calls = []
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: calls.append(1))
    with TestClient(app):
        pass  # entering the context runs the lifespan startup phase
    assert calls == [1]


def test_the_page_retries_a_failed_send_for_ninety_seconds_and_a_pressed_again_action_reuses_its_id(client: TestClient) -> None:
    """Marc, 2026-10-06: an iPhone on mobile data lost its connection for longer than the page's 25 s retry, three times; each press
    after "Load failed" was a new id, so one action ran three times. The page's own functions were exercised in a browser with the
    network stubbed (21 dropped calls then success resolves; endless drops give up near 90 s; the id is reused only for the same text
    and character after a failure); here, that the page is wired so."""
    page = client.get("/").text
    assert "const SEND_RETRY_MS = 90000;" in page and "Date.now() + SEND_RETRY_MS" in page and "Date.now() + 25000" not in page
    assert "let failedSend = null;" in page and "failedSend.text === text && failedSend.entity === entity ? failedSend.id : newClientId()" in page
    assert "sending = { text, entity: acting, id: sendIdFor(text, acting) }" in page and "client_id: sending.id" in page   # the id sent is the reused one
    assert "if (sending) failedSend = sending;" in page and "failedSend = null;   // the server answered" in page            # kept on a failure, cleared on an answer
    assert "client_id: newClientId() }" not in page                                                                          # no send mints a fresh id past it


def test_the_picker_goes_to_a_world_with_a_port_and_no_address_instead_of_asking_this_server_to_switch(client: TestClient) -> None:
    """Marc, 2026-10-06: choosing The Breakthrough (a world the builder made: a port, no public address) answered "Stonemarch's server", because the
    page asked the pinned server to /worlds/select it. The page now goes to the world's port on the host it came from when that is this machine
    or a numbered address, and says where the world is when it came by a public hostname (where a port is no address). Exercised in a browser:
    choosing it on 127.0.0.1:8198 landed on 127.0.0.1:8203; here, that the page is wired so."""
    page = client.get("/").text
    assert "function worldAddress(w)" in page and "if (w.url) return w.url;" in page
    assert 'h === "localhost" || h === "[::1]"' in page and "${location.protocol}//${h}:${w.port}/" in page      # this machine, or a numbered address
    assert "const address = worldAddress(chosen);" in page and "location.href = address;" in page
    assert "has no public address yet" in page and "await refreshWorldPicker();" in page                          # said, and the picker put back
    assert "/worlds/select" in page                                                                                # an unpinned registry world still switches


def test_root_serves_the_web_ui_without_touching_world_state(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Smritilila" in resp.text
    # Serving the page must never auto-bootstrap a world as a side
    # effect -- the page's own JS only ever calls GET /world/exists on
    # load, which is deliberately side-effect-free (see its own docstring).
    assert not hasattr(main.app.state, "world")


def test_world_exists_is_false_when_nothing_persisted(client: TestClient) -> None:
    resp = client.get("/world/exists")
    assert resp.status_code == 200
    assert resp.json() == {"exists": False, "playable_entities": {}}


def test_world_exists_never_bootstraps_a_world_as_a_side_effect(client: TestClient, fake_store: _FakeWorldStore) -> None:
    resp = client.get("/world/exists")
    assert resp.json() == {"exists": False, "playable_entities": {}}
    assert fake_store.commit_calls == []
    assert not hasattr(main.app.state, "world")


def test_entity_recap_returns_404_for_an_unknown_entity(client: TestClient) -> None:
    _seed_square_world(client)
    resp = client.get("/entities/nobody-by-this-id/recap")
    assert resp.status_code == 404


def test_entity_recap_last_narration_is_none_without_any_memories(client: TestClient) -> None:
    world = _seed_square_world(client)
    world.canon.entities["mara"] = Entity("mara", ["Mara"], 0, "genesis")  # no tuples yet: stands nowhere
    resp = client.get("/entities/mara/recap")
    assert resp.status_code == 200
    assert resp.json() == {"location_id": None, "last_narration": None}


def test_submit_pulse_missing_action_text_is_rejected_without_needing_a_working_llm_client(client: TestClient) -> None:
    resp = client.post("/pulses", json={})
    assert resp.status_code == 422


def test_an_action_with_nothing_said_is_refused_before_any_model_call(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Ravenmoor 508: "{" ran a whole pulse and the Narrator invented the
    deed. An action needs a letter or digit, in any script."""
    # 2026-10-01: the "看" pulse below once reached the real the second model with
    # the second model_API_KEY in the environment, and failed a full run when the
    # provider erred. A fake Narrator and no keys: never the network.
    monkeypatch.delenv("THE_FIRST_API_KEY", raising=False)
    monkeypatch.delenv("the second model_API_KEY", raising=False)
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _SmrtiClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    _seed_square_world(client)
    for junk in ("{", "   ", "?!", "...", "}{"):
        r = client.post("/pulses", json={"action_text": junk})
        assert r.status_code == 422 and r.json()["detail"] == "say what you do"
    assert roles.smrti_client.calls == []  # refused before any model call
    r = client.post("/pulses", json={"action_text": "看"})  # a word in any script is an action
    assert r.status_code == 200, r.text
    assert roles.smrti_client.calls and "看" in roles.smrti_client.calls[0]


def test_submit_pulse_without_an_api_key_returns_503_for_an_otherwise_valid_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Explicitly cleared, not just assumed absent: a provider's key can leak into the test process from the real OS environment (e.g. set once
    # via `setx` for live manual testing), which would make _resolve_roles() construct a real client and fail later with a raw network error
    # instead of the clean 503 this test is actually asserting.
    for k in [k for k in os.environ if k.endswith("_API_KEY")] + ["SMRITILILA_LOCAL_BASE_URL", "SMRITILILA_LOCAL_MODEL"]:
        monkeypatch.delenv(k, raising=False)   # no key of any provider, no local model: no narrator
    # a world with someone to act as (2026-09-24: an empty world used to
    # count "" as playable, which is what this test leaned on)
    _seed_square_world(client)
    resp = client.post("/pulses", json={"action_text": "steps into the lane"})
    assert resp.status_code == 503


def test_genesis_rejects_an_empty_self_description(client: TestClient) -> None:
    resp = client.post("/genesis", json={"self_description": "   "})
    assert resp.status_code == 422


def test_genesis_rejects_when_canon_is_not_empty(client: TestClient) -> None:
    _seed_square_world(client)  # canon now has the starter world
    resp = client.post("/genesis", json={"self_description": "A traveler arrives."})
    assert resp.status_code == 409


def test_world_clear_calls_the_stores_clear(client: TestClient, fake_store: _FakeWorldStore) -> None:
    _seed_square_world(client)
    client.post("/world/clear")
    assert fake_store.clear_calls == 1


def _install_fake_registry(monkeypatch: pytest.MonkeyPatch, entries):
    from smritilila.worlds_registry import WorldRegistryEntry

    registry = [WorldRegistryEntry(name, url) for name, url in entries]
    monkeypatch.setattr(adhisthana, "get_worlds_registry", lambda: registry)
    return registry


def test_list_worlds_returns_empty_list_when_no_registry_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_registry(monkeypatch, [])

    resp = client.get("/worlds")

    assert resp.status_code == 200
    assert resp.json() == []


def test_list_worlds_marks_the_database_url_matching_entry_active(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_registry(
        monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila"), ("Ravenmoor Test", "postgresql+asyncpg://x/smritilila_ravenmoor")]
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x/smritilila")

    resp = client.get("/worlds")

    assert resp.json() == [
        {"name": "Frontier", "active": True, "port": None, "description": None, "url": None},
        {"name": "Ravenmoor Test", "active": False, "port": None, "description": None, "url": None},
    ]


def test_list_worlds_reflects_an_explicit_selection_over_the_env_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_registry(
        monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila"), ("Ravenmoor Test", "postgresql+asyncpg://x/smritilila_ravenmoor")]
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x/smritilila")
    main.app.state.active_database_url = "postgresql+asyncpg://x/smritilila_ravenmoor"

    resp = client.get("/worlds")

    assert resp.json() == [
        {"name": "Frontier", "active": False, "port": None, "description": None, "url": None},
        {"name": "Ravenmoor Test", "active": True, "port": None, "description": None, "url": None},
    ]


def test_select_world_with_an_unknown_name_returns_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_registry(monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila")])

    resp = client.post("/worlds/select", json={"name": "Nowhere"})

    assert resp.status_code == 404


def test_select_world_switches_the_active_database_and_reloads_from_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this guards: selecting a world must force a real
    reload from the NEWLY selected database, not just keep serving
    whatever WorldSession/store happened to already be cached from
    before the switch."""
    _install_fake_registry(
        monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila"), ("Ravenmoor Test", "postgresql+asyncpg://x/smritilila_ravenmoor")]
    )
    stale_canon = Maya(world_time=0)
    stale_canon.entities["stale"] = Entity("stale", ["Stale Entity"], 0, "genesis")
    main.app.state.world = WorldSession(canon=stale_canon, scene_owner_entity_id="stale")

    fresh_canon = Maya(world_time=0)
    fresh_canon.entities["fresh"] = Entity("fresh", ["Fresh Entity"], 0, "genesis")
    _install_fake_store(monkeypatch, initial=fresh_canon, session_config=("fresh", [], ["fresh"], None))

    resp = client.post("/worlds/select", json={"name": "Ravenmoor Test"})

    assert resp.status_code == 200
    assert resp.json()["scene_owner_entity_id"] == "fresh"
    assert [e["id"] for e in resp.json()["entities"]] == ["fresh"]
    assert main.app.state.active_database_url == "postgresql+asyncpg://x/smritilila_ravenmoor"


def test_world_clear_backs_up_the_active_world_first_when_it_is_registered(
    client: TestClient, fake_store: _FakeWorldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = _install_fake_registry(monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila")])
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x/smritilila")
    backup_calls: list = []
    monkeypatch.setattr(loka, "backup_world", lambda entry, reason: backup_calls.append((entry, reason)))

    resp = client.post("/world/clear")

    assert resp.status_code == 204
    assert backup_calls == [(entries[0], "pre-clear")]
    assert fake_store.clear_calls == 1


def test_world_clear_skips_backup_when_the_active_database_is_not_registered(
    client: TestClient, fake_store: _FakeWorldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_registry(monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila")])
    monkeypatch.delenv("DATABASE_URL", raising=False)
    backup_calls: list = []
    monkeypatch.setattr(loka, "backup_world", lambda entry, reason: backup_calls.append((entry, reason)))

    resp = client.post("/world/clear")

    assert resp.status_code == 204
    assert backup_calls == []
    assert fake_store.clear_calls == 1


def test_world_clear_refuses_to_proceed_when_the_backup_itself_fails(
    client: TestClient, fake_store: _FakeWorldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    _install_fake_registry(monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila")])
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x/smritilila")

    def failing_backup(entry, reason):
        raise subprocess.CalledProcessError(1, ["pg_dump"], stderr="connection refused")

    monkeypatch.setattr(loka, "backup_world", failing_backup)

    resp = client.post("/world/clear")

    assert resp.status_code == 503
    assert fake_store.clear_calls == 0  # never reached -- the whole point


def test_backup_world_now_returns_the_backup_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_registry(monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila")])
    monkeypatch.setattr(loka, "backup_world", lambda entry, reason: f"/backups/{entry.name}/{reason}.dump")

    resp = client.post("/worlds/Frontier/backup")

    assert resp.status_code == 200
    assert resp.json() == {"world_name": "Frontier", "backup_path": "/backups/Frontier/manual.dump"}


def test_backup_world_now_returns_404_for_an_unregistered_world(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_registry(monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila")])

    resp = client.post("/worlds/Nowhere/backup")

    assert resp.status_code == 404


def test_backup_world_now_returns_502_when_pg_dump_fails(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    _install_fake_registry(monkeypatch, [("Frontier", "postgresql+asyncpg://x/smritilila")])

    def failing_backup(entry, reason):
        raise subprocess.CalledProcessError(1, ["pg_dump"], stderr="disk full")

    monkeypatch.setattr(loka, "backup_world", failing_backup)

    resp = client.post("/worlds/Frontier/backup")

    assert resp.status_code == 502


def test_submit_pulse_rejects_an_entity_id_not_in_playable_entity_ids(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: _fake_roles())
    _seed_square_world(client)
    resp = client.post("/pulses", json={"action_text": "waits", "entity_id": "mara"})
    assert resp.status_code == 422


def test_entity_scene_is_where_the_tuples_last_record_you(client: TestClient) -> None:
    """v3: where you stand and who is there come from tuples (s11); no
    adjacency, no graph."""
    world = _seed_square_world(client)
    world.canon.entities["mara"] = Entity("mara", ["Mara"], 0, "genesis")
    world.recent_memories.append(Memory("m-mara", "mara", frozenset(), "I crossed the square to the well.", "PRIMARY", 5, "square", 3, "p3", None))
    world.recent_memories.append(Memory("m-sq", "square", frozenset({"player", "mara"}), "Two people in the square.", "AMBIENT", 5, "square", 3, "p3", None))
    body = client.get("/entities/player/scene").json()
    assert body["location_id"] == "square" and set(body["present"]) >= {"mara"} and body["adjacent"] == []
    assert client.get("/entities/nobody/scene").status_code == 404


class _SmrtiClient:
    """A v3 Narrator fake: answers the append_tuples tool for whoever acts."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, system, user, *, max_tokens=8192):  # pragma: no cover
        raise NotImplementedError

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
        self.calls.append(user)
        actor = next(l for l in user.splitlines() if l.startswith("ACTOR: ")).split(": ", 1)[1]
        place = next(l for l in user.splitlines() if l.startswith("SCENE: ")).split(": ", 1)[1]
        return {
            "narration": f"{actor} acts.",
            "perceiver_tuples": [{"perceiver": actor, "text": "I look around the square in the morning light.", "salience": 4},
                                 {"perceiver": "NEW:stranger_in_a_grey_cloak", "text": "Someone near me in the square looks about.", "salience": 6}],
            "scene_tuples": [{"scene": place, "text": "Someone looks around the square this morning.", "present": [actor, "NEW:stranger_in_a_grey_cloak"]}],
            "suggested_next_action": "Look around.",
        }


def test_smrti_live_mode_runs_the_v3_loop_and_bypasses_the_v2_pipeline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    smrti_client = _SmrtiClient()
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = smrti_client
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    _seed_square_world(client)
    world = client.app.state.world
    owner = world.scene_owner_entity_id
    before = len(world.recent_memories)

    resp = client.post("/pulses", json={"action_text": "looks around the square"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["narrations"][owner] == f"{owner} acts."
    assert "committed_deltas" not in body and body["suggested_next_action"] == "Look around."  # v2's deltas are gone
    # the v2 Narrator never ran; the v3 one did for the player -- and, in the
    # background, for the scheduler's activation of the new stranger (the
    # same call with no action supplied: v3 autonomy)
    from smritilila.smrti.narrator import AUTONOMOUS_ACTION
    player_calls = [c for c in smrti_client.calls if AUTONOMOUS_ACTION not in c]
    autonomous_calls = [c for c in smrti_client.calls if AUTONOMOUS_ACTION in c]
    assert roles.narrator.resolved_events == [] and len(player_calls) == 1 and len(autonomous_calls) <= 1
    # tuples appended by the player's pulse: the actor's, the new stranger's, the place's -- and the stranger is now an entity with an opaque id
    assert len(world.recent_memories) >= before + 3
    new_id_ = body["promoted_entity_ids"][0]
    assert world.canon.entities[new_id_].labels == ["stranger in a grey cloak"] and world.canon.entities[new_id_].origin == "smrti"
    # persisted through the store, as memories + a promoted entity, no deltas
    canon, result, _ = fake_store.commit_calls[0]
    assert [e.id for e in result.promoted_entities] == [new_id_]
    # a second pulse retrieves the first's tuples: the stranger is now in the memory handed to the Narrator -- by
    # handle, its own words without the minted tail (2026-10-02: one handle rule for every id)
    resp2 = client.post("/pulses", json={"action_text": "greets the stranger"})
    assert resp2.status_code == 200
    second = [c for c in smrti_client.calls if AUTONOMOUS_ACTION not in c][1]
    assert "stranger-in-a-grey-cloak" in second and new_id_ not in second


def test_smrti_activation_overtaken_by_the_players_pulse_is_adopted_beside_it_not_dropped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    # Vadhu tick 110: the activation committed, then the live world_time
    # had moved (the player's own pulse landed first) and the result was
    # dropped from memory -- store and server disagreed until a restart.
    import asyncio
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.smrti.pulse import SmrtiPulseResult
    _seed_square_world(client)
    world = client.app.state.world
    owner = world.scene_owner_entity_id
    live_canon = world.canon
    t0 = world.canon.world_time
    minted = Entity("newcomer_ab12", ["a newcomer"], t0 + 1, "smrti")

    def overtaken(canon, actor, action, world_time, pulse_id, mems, *a, **k):
        live_canon.world_time += 1  # the player's pulse lands while this one computes
        live_canon.entities[owner].last_simulated_tick = world_time
        return SmrtiPulseResult("She moves.", [Memory("m-act", actor, frozenset(), "I moved to the kettle.", "PRIMARY", 4, "square", world_time, pulse_id, None)],
                                [minted], "", 3, [actor])

    monkeypatch.setattr(spanda, "run_smrti_pulse", overtaken)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(spanda, "log_event", lambda event, **f: events.append((event, f)))
    before = len(world.recent_memories)
    asyncio.run(spanda._run_smrti_activation_in_background(world, _fake_roles(_SpyNarrator()), owner, 1))
    # committed, and adopted beside the player's pulse: memory appended, entity merged, live canon kept
    assert fake_store.commit_calls[-1][1].memories_written[0].id == "m-act"
    assert world.canon is live_canon and world.canon.world_time == t0 + 1
    assert world.recent_memories[-1].id == "m-act" and len(world.recent_memories) == before + 1
    assert world.canon.entities["newcomer_ab12"] is minted
    assert [e for e, _ in events if e.startswith("activation_pulse")] == ["activation_pulse.adopted_beside", "activation_pulse.timing"]


def _meanwhile_world(client: TestClient):
    """The square world with an NPC in it, whose scheduler bookkeeping the
    live world moves on while a pulse computes on its own copy."""
    import copy
    from smritilila.graph.entity import Entity
    world = _seed_square_world(client)
    world.canon.entities["npc"] = Entity("npc", ["a carter"], 0, "genesis")
    working = copy.deepcopy(world.canon)
    working.world_time += 1
    return world, working


@pytest.mark.parametrize("another_commit", [True, False])
def test_a_players_commit_keeps_what_the_live_world_moved_on_meanwhile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store, another_commit: bool
) -> None:
    # TODO 09-25, the stale snapshot: the commit swapped in the pulse's own
    # copy of EVERY entity (setdefault kept only what others had minted), so
    # the scheduler's in-place bookkeeping -- a drawn NPC's last_simulated_tick
    # -- rolled back in memory while the store kept it, with or without a
    # commit in another scene meanwhile.
    import asyncio
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.smrti.pulse import SmrtiPulseResult
    world, working = _meanwhile_world(client)
    t0 = world.canon.world_time
    # meanwhile, on the live world: the scheduler draws the NPC; perhaps a turn next door commits and mints
    world.canon.entities["npc"].last_simulated_tick = 7
    if another_commit:
        world.canon.world_time += 1
        world.canon.entities["pedlar_x1"] = Entity("pedlar_x1", ["a pedlar"], t0 + 1, "smrti")
    # the pulse: it binds the player and mints a coin, on its own copy
    working.entities["player"].last_simulated_tick = working.world_time
    coin = Entity("coin_c1", ["a coin"], working.world_time, "smrti")
    working.entities["coin_c1"] = coin
    result = SmrtiPulseResult("You pick up a coin.", [Memory("m-p", "player", frozenset(), "I pick up a coin.", "PRIMARY", 4, "square",
                                                              working.world_time, "p1", None)], [coin], "", 3, ["player"])
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(spanda, "log_event", lambda event, **f: events.append((event, f)))
    asyncio.run(spanda._commit_smrti_result(world, working, result, "player", "p1"))
    assert [f["ids"] for e, f in events if e == "smrti.adopt_kept_live"] == [["npc"]]  # witnessed
    assert world.canon.entities["npc"].last_simulated_tick == 7  # the live world's, not the pulse's stale copy
    assert world.canon.entities["player"].last_simulated_tick == working.world_time  # the pulse's own
    assert world.canon.entities["coin_c1"] is coin
    assert ("pedlar_x1" in world.canon.entities) == another_commit
    assert world.canon.world_time == t0 + 1
    assert world.recent_memories[-1].id == "m-p"


def test_an_overtaken_activation_adopts_what_it_bound_not_only_what_it_minted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    # The mirror of the stale snapshot: an NPC's turn overtaken by the
    # player's pulse kept only its minted entities, so its changes to what it
    # bound (the store wrote them) were lost from memory.
    import asyncio
    import dataclasses
    from smritilila.memory.memory import Memory
    from smritilila.smrti.pulse import SmrtiPulseResult
    world, _ = _meanwhile_world(client)
    live_canon = world.canon

    def overtaken(canon, actor, action, world_time, pulse_id, mems, *a, **k):
        live_canon.world_time += 1  # the player's pulse lands while this one computes
        live_canon.entities["player"].last_simulated_tick = world_time
        # a pulse's working world shares the live entities (Maya.fork): a change
        # of its own is a new entity in its own table, never an edit in place
        canon.entities[actor] = dataclasses.replace(canon.entities[actor], last_simulated_tick=world_time)  # this turn's own
        canon.entities["player"] = dataclasses.replace(canon.entities["player"], labels=["a stale copy"])  # not bound: must not come back
        return SmrtiPulseResult("He loads the cart.", [Memory("m-npc", actor, frozenset(), "I load the cart.", "PRIMARY", 4, "square",
                                                              world_time, pulse_id, None)], [], "", 3, [actor])

    monkeypatch.setattr(spanda, "run_smrti_pulse", overtaken)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    t = world.canon.world_time + 1
    asyncio.run(spanda._run_smrti_activation_in_background(world, _fake_roles(_SpyNarrator()), "npc", 1))
    assert world.canon is live_canon
    assert world.canon.entities["npc"].last_simulated_tick == t  # adopted, as the store has it
    assert world.canon.entities["player"].last_simulated_tick == t and world.canon.entities["player"].labels == ["the player"]
    assert world.recent_memories[-1].id == "m-npc"


def test_a_pulse_changes_only_the_entities_it_binds_or_mints(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    # What _adopt rests on: the store writes only what a pulse bound or
    # minted, and the live world adopts only that -- so a pulse must change
    # nothing else on its copy. Checked on the real pulse.
    import copy
    real = spanda.run_smrti_pulse
    strays: list[str] = []

    def watched(canon, actor, *a, **k):
        before = copy.deepcopy(canon.entities)
        result = real(canon, actor, *a, **k)
        mine = set(result.bound_ids) | {e.id for e in result.new_entities} | {actor}
        strays.extend(eid for eid, ent in before.items() if eid not in mine and canon.entities.get(eid) != ent)
        return result

    smrti_client = _SmrtiClient()
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = smrti_client
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(spanda, "run_smrti_pulse", watched)
    world, _ = _meanwhile_world(client)
    for action in ("looks around the square", "greets the stranger"):
        assert client.post("/pulses", json={"action_text": action}).status_code == 200
    assert strays == []


def test_entity_memories_returns_a_perceivers_own_recent_tuples(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    from smritilila.memory.memory import Memory
    _seed_square_world(client)
    world = client.app.state.world
    owner = world.scene_owner_entity_id
    for i in range(3):
        world.recent_memories.append(Memory(f"mm{i}", owner, frozenset(), f"I remembered thing {i}.", "PRIMARY", 4, "square", 10 + i, f"pp{i}", None))
    resp = client.get(f"/entities/{owner}/memories?limit=2")
    assert resp.status_code == 200
    assert [m["experience"] for m in resp.json()["memories"]] == ["I remembered thing 1.", "I remembered thing 2."]
    assert client.get("/entities/nobody/memories").status_code == 404


def test_smrti_pool_sleeps_a_quiet_scene_no_player_is_near() -> None:
    """v5 §38 (2026-09-23): a scene sleeps when its last QUIET_TURNS shared
    records all gave nobody anything (use: none) and no player is in it or
    a neighbouring scene -- a group as well as one person alone (Ravenmoor's
    town street, four NPCs writing 45 tuples of nothing)."""
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    canon = Maya(world_time=10, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="building"),
        "road": Entity("road", ["the east road"], 0, "genesis", avarana="open"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
        "ilo": Entity("ilo", ["Ilo"], 0, "genesis"),
        "drover": Entity("drover", ["a drover"], 0, "smrti"),
        "mule": Entity("mule", ["his mule"], 0, "smrti"),
    })
    def mem(i, who, t, loc, use=None, tier="PRIMARY"):
        return Memory(f"m{i}", who, frozenset(), "x", tier, 4, loc, t, f"p{t}", None, use)
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    # ilo's last tuple is not the player's tick, so the engaged draw does not decide it
    world.recent_memories = [mem(1, "teo", 1, "inn"), mem(2, "ilo", 2, "inn"), mem(3, "drover", 4, "road"), mem(4, "mule", 4, "road")]
    world.recent_memories += [mem(10 + t, "road", t, "road", "none", "AMBIENT") for t in (7, 8, 9)]
    adhisthana.FAR_EVERY, far = 1, adhisthana.FAR_EVERY  # every tick far, so distance does not decide it
    try:
        pool = set(pravrtti._smrti_exploration_pool(world))
        assert "drover" not in pool and "mule" not in pool and "ilo" in pool  # the road gave nobody anything: both sleep
        world.recent_memories.append(mem(20, "road", 9, "road", "useful", "AMBIENT"))  # anything of use wakes it
        assert {"drover", "mule"} <= set(pravrtti._smrti_exploration_pool(world))
        world.recent_memories.pop()
        world.recent_memories.append(mem(21, "teo", 9, "road"))  # a player coming into it wakes it
        assert {"drover", "mule"} <= set(pravrtti._smrti_exploration_pool(world))
    finally:
        adhisthana.FAR_EVERY = far


def test_smrti_pool_draws_a_road_at_most_once_in_the_quiet_window() -> None:
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    canon = Maya(world_time=30, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="building"),
        "road": Entity("road", ["the west road"], 0, "genesis", avarana="road"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
    })
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.world_data["pacing"] = {"arrivals_on_traffic": False}  # asked on the clock: the ask itself, not its traffic gate
    world.recent_memories = [Memory("m1", "teo", frozenset(), "x", "PRIMARY", 4, "inn", 1, "p1", None),
                             Memory("m0", "teo", frozenset(), "x", "PRIMARY", 4, "road", 0, "p0", None)]  # Teo lately stood on the road: near
    assert pravrtti._smrti_exploration_pool(world) == ["road"]          # never asked yet
    canon.entities["road"].last_simulated_tick = 25
    assert pravrtti._smrti_exploration_pool(world) == []                # asked 5 ticks ago: quiet
    canon.entities["road"].last_simulated_tick = 10
    assert pravrtti._smrti_exploration_pool(world) == ["road"]          # 20 ticks ago: may be asked again


def test_the_draws_pool_is_witnessed_with_what_each_last_turn_gave(monkeypatch: pytest.MonkeyPatch) -> None:
    """Marc, 2026-09-28: history could not be replayed faithfully (22 of 86 picks rebuilt), so the pool is logged at
    each draw -- each candidate's last turn and the uses of its own tuples then -- for a draw rule to be scored."""
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    events = []
    monkeypatch.setattr(main, "log_event", lambda e, **kw: events.append((e, kw)))
    monkeypatch.setattr("smritilila.pravrtti.log_event", lambda e, **kw: events.append((e, kw)))  # the draw logs from its own module
    canon = Maya(world_time=9, entities={"inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene"),
                                         "ilo": Entity("ilo", ["Ilo"], 0, "genesis"),
                                         "rat": Entity("rat", ["a rat"], 0, "genesis")})
    canon.entities["ilo"].last_simulated_tick = 5
    world = WorldSession(canon=canon, scene_owner_entity_id="ilo", entry_point_location_ids=["inn"], playable_entity_ids=[])
    world.recent_memories = [Memory("m1", "ilo", frozenset(), "I swore to find her.", "PRIMARY", 3, "inn", 5, "p5", None, use="key"),
                             Memory("m2", "ilo", frozenset(), "I sat.", "PRIMARY", 6, "inn", 5, "p5", None, use="none"),
                             Memory("m3", "ilo", frozenset(), "I slept.", "PRIMARY", 6, "inn", 7, "p7", None, use="none"),
                             Memory("m4", "rat", frozenset(), "I ran.", "PRIMARY", 6, "inn", 4, "p4", None)]
    pravrtti._log_pool(world, ["ilo", "rat"])
    assert [kw for e, kw in events if e == "tick.pool"] == [{"tick": 9, "pool": [["ilo", 5, ["key", "none"]], ["rat", None, []]]}]


def test_smrti_pool_draws_far_actors_only_every_eighth_tick() -> None:
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    canon = Maya(world_time=77, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="building"),
        "market": Entity("market", ["the hill market"], 0, "smrti", avarana="open"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
        "ilo": Entity("ilo", ["Ilo"], 0, "genesis"),
        "drover": Entity("drover", ["a drover"], 0, "smrti"),
        "seller": Entity("seller", ["a cloth seller"], 0, "smrti"),
    })
    def mem(i, who, t, loc):
        return Memory(f"m{i}", who, frozenset(), "x", "PRIMARY", 4, loc, t, f"p{t}", None)
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.world_data["pacing"] = {"arrivals_on_traffic": False}  # asked on the clock: the ask itself, not its traffic gate
    world.recent_memories = [mem(1, "teo", 70, "inn"), mem(2, "ilo", 71, "inn"), mem(3, "drover", 76, "market"), mem(4, "seller", 76, "market")]
    assert set(pravrtti._smrti_exploration_pool(world)) == {"ilo"}            # tick 77: far actors wait
    canon.world_time = 80
    assert set(pravrtti._smrti_exploration_pool(world)) == {"ilo", "drover", "seller", "market"}  # a far tick -- and the quiet market is asked whether anyone came (v5 §38)


def test_a_branch_that_keeps_its_own_pace_breathes_with_no_player_near() -> None:
    """v4.61, breath (pratiṣṭhā): pacing {"far_every": 1} on a branch -- the market here -- draws its beings on every
    tick though no player is near, and its quiet scene does not sleep; the rest of the world keeps the world's far
    ticks."""
    import json
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.memory.record import Record
    canon = Maya(world_time=77, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="building"),
        "market": Entity("market", ["the hill market"], 0, "smrti", avarana="open"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
        "ilo": Entity("ilo", ["Ilo"], 0, "genesis"),
        "drover": Entity("drover", ["a drover"], 0, "smrti"),
    })
    def mem(i, who, t, loc, use="useful"):
        return Memory(f"m{i}", who, frozenset(), "x", "PRIMARY", 4, loc, t, f"p{t}", None, use)
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.world_data["pacing"] = {"arrivals_on_traffic": False}
    world.recent_memories = [mem(1, "teo", 70, "inn"), mem(2, "ilo", 71, "inn"), mem(3, "drover", 76, "market"),
                             mem(4, "market", 76, "market", "none"), mem(5, "market", 75, "market", "none"), mem(6, "market", 74, "market", "none")]
    assert "drover" not in pravrtti._smrti_exploration_pool(world)                  # tick 77: far, the world's pace
    world.records = [Record("r1", "prakriti", "market", json.dumps({"far_every": 1}), 1, "p1", "pacing")]
    world._samskara = None
    assert "drover" in pravrtti._smrti_exploration_pool(world)                      # the market breathes: drawn now


def test_smrti_pool_admits_a_thing_the_world_gave_a_stream() -> None:
    """s0 (2026-09-20): whoness is only 'this id has a stream'. A v2 world
    typed 'his money' as an object and then gave it 28 tuples; the engine
    does not weigh what kind of thing it is before letting it act."""
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    canon = Maya(world_time=10, entities={
        "saloon": Entity("saloon", ["the saloon"], 0, "genesis", avarana="building"),
        "wind": Entity("wind", ["Wind"], 0, "genesis"),
        "money": Entity("money", ["his money"], 0, "smrti"),
    })
    world = WorldSession(canon=canon, scene_owner_entity_id="wind", entry_point_location_ids=["saloon"], playable_entity_ids=["wind"])
    world.recent_memories = [Memory("m1", "wind", frozenset(), "x", "PRIMARY", 4, "saloon", 1, "p1", None),
                             Memory("m2", "money", frozenset(), "x", "PRIMARY", 4, "saloon", 2, "p2", None)]
    assert pravrtti._smrti_exploration_pool(world) == ["money"]


def test_a_player_witnesses_the_place_tuple_of_a_turn_taken_where_they_stand(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """s19 (2026-09-20, Ravenmoor 88-94): Corwin's turns demanded the
    transcript to Nathaniel's face and Marc never saw it -- the NPC's
    narration went to the NPC's card and no tuple is written for a
    player in another's turn. The place tuple is what the player saw."""
    import asyncio
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.smrti.pulse import SmrtiPulseResult
    _seed_square_world(client)
    world = client.app.state.world
    owner = world.scene_owner_entity_id
    world.canon.entities["lab"] = Entity("lab", ["Corwin's lab"], 0, "genesis", avarana="building")
    world.canon.entities["corwin"] = Entity("corwin", ["Corwin"], 0, "genesis")
    world.recent_memories.append(Memory("m-own", owner, frozenset(), "I handed Corwin the tins.", "PRIMARY", 4, "lab", world.canon.world_time, "p80", None))
    t0 = world.canon.world_time

    def corwin_speaks(canon, actor, action, world_time, pulse_id, mems, *a, **k):
        return SmrtiPulseResult("Corwin sets down the scalpel and demands the transcript tonight.", [
            Memory("m-c", "corwin", frozenset(), "I demanded the transcript tonight.", "PRIMARY", 3, "lab", world_time, pulse_id, None),
            Memory("m-lab", "lab", frozenset(), "The biologist set down his scalpel and demanded the transcript be produced that night.", "AMBIENT", 5, "lab", world_time, pulse_id, None),
            Memory("m-else", "elsewhere", frozenset(), "Rain on an empty road.", "AMBIENT", 5, "elsewhere", world_time, pulse_id, None),
        ], [], "", 3, ["corwin"])

    monkeypatch.setattr(spanda, "run_smrti_pulse", corwin_speaks)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    monkeypatch.setattr(main, "log_event", lambda event, **f: None)
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _SmrtiClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    asyncio.run(spanda._run_smrti_activation_in_background(world, roles, "corwin", 1))
    assert world.pending_witnessed == {owner: [{"place": "Corwin's lab",
        "text": "The biologist set down his scalpel and demanded the transcript be produced that night.", "said": "", "said_by": "", "world_time": t0 + 1}]}
    # the elsewhere place tuple is not the player's to see; and the queue drains into the player's next pulse
    monkeypatch.setattr(spanda, "run_smrti_pulse", lambda canon, actor, action, world_time, pulse_id, mems, *a, **k: SmrtiPulseResult(
        "You say it.", [Memory("m-2", actor, frozenset(), "I said it.", "PRIMARY", 4, "lab", world_time, pulse_id, None)], [], "", 3, [actor]))
    monkeypatch.setattr(spanda, "advance_world_tick", lambda *a, **k: type("T", (), {"pulses_run": 0, "activated_entity_id": None})())
    resp = client.post("/pulses", json={"action_text": "I say it."})
    assert resp.status_code == 200, resp.text
    assert resp.json()["witnessed"] == {owner: [{"place": "Corwin's lab",
        "text": "The biologist set down his scalpel and demanded the transcript be produced that night.", "said": "", "said_by": "", "world_time": t0 + 1}]}
    assert world.pending_witnessed == {}


def test_a_waiting_player_takes_no_turn_until_someone_touches_them() -> None:
    """Marc, 2026-10-01: "keep teo active, just don't take any turns until
    someone interacts with him". Waiting, Teo stays in the world (not absent)
    and is not drawn; a tuple of his from someone else's pulse wakes him; his
    own player's actions do not."""
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.memory.record import Record
    canon = Maya(world_time=10, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="building"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
    })
    canon.entities["teo"].last_simulated_tick = 6
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.autonomous, world.logged_off, world.waiting = {"teo"}, {"teo"}, {"teo"}
    world.recent_memories = [Memory("m1", "teo", frozenset(), "I eased down by the fire.", "PRIMARY", 4, "inn", 5, "p5", None)]
    assert "teo" not in world.absent() and pravrtti._smrti_exploration_pool(world) == []
    # his own player's pulse after the mark is not a touch
    world.recent_memories.append(Memory("m2", "teo", frozenset(), "I rubbed my ribs.", "PRIMARY", 4, "inn", 7, "p7", None))
    world.records = [Record("a7", "action", "teo", "I rub my ribs.", 7, "p7", note="typed")]
    world._samskara = None
    assert pravrtti._smrti_exploration_pool(world) == []
    # someone else's pulse writes him a tuple: awake
    world.recent_memories.append(Memory("m3", "teo", frozenset(), "Ilo asked me to check the lines.", "PRIMARY", 4, "inn", 8, "p8", None))
    world._samskara = None
    assert pravrtti._smrti_exploration_pool(world) == ["teo"]
    world.waiting = set()  # active: drawn as before, touched or not
    assert pravrtti._smrti_exploration_pool(world) == ["teo"]


def test_waiting_is_set_through_the_toggle_and_kept(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    world = _seed_square_world(client)
    r = client.post("/players/player/autonomy", json={"allow": True, "wait": True}).json()
    assert r["autonomous"] and r["waiting"] and world.waiting == {"player"} and fake_store._waiting == ["player"]
    assert world.canon.entities["player"].last_simulated_tick == world.canon.world_time  # quiet from now
    # v4.58: "active" is gone -- allowed is waiting, whatever an old page sends
    r = client.post("/players/player/autonomy", json={"allow": True, "wait": False}).json()
    assert r["autonomous"] and r["waiting"] and fake_store._waiting == ["player"]
    r = client.post("/players/player/autonomy", json={"allow": False}).json()
    assert not r["autonomous"] and not r["waiting"]


def test_smrti_pool_draws_among_those_engaged_in_the_players_scene_first() -> None:
    """Vadhu 351-357: Ilo, just asked for the rest of it, was the least
    overdue in the pool and the road and the inn took the activations
    while he 'started twice and couldn't'."""
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    canon = Maya(world_time=60, entities={
        "landing": Entity("landing", ["the east landing"], 0, "genesis", avarana="open"),
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="building"),
        "road": Entity("road", ["the west road"], 0, "genesis", avarana="road"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
        "ilo": Entity("ilo", ["Ilo"], 0, "genesis"),
        "marren": Entity("marren", ["Marren"], 0, "genesis"),
    })
    canon.entities["ilo"].last_simulated_tick = 58      # just spoke
    canon.entities["marren"].last_simulated_tick = 40   # long overdue
    def mem(i, who, t, loc):
        return Memory(f"m{i}", who, frozenset(), "x", "PRIMARY", 4, loc, t, f"p{t}", None)
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["landing"], playable_entity_ids=["teo"])
    world.world_data["pacing"] = {"arrivals_on_traffic": False}  # asked on the clock: the ask itself, not its traffic gate
    world.recent_memories = [mem(1, "marren", 40, "inn"), mem(2, "teo", 59, "landing"), mem(3, "ilo", 59, "landing"), mem(4, "landing", 59, "landing")]
    assert pravrtti._smrti_exploration_pool(world) == ["ilo"]   # engaged: wrote a tuple where the player acted, in that pulse
    # a near road past its quiet window interrupts the scene
    canon.entities["road"].last_simulated_tick = None
    world.recent_memories.append(mem(9, "teo", 45, "road"))  # Teo lately stood on the road: near
    assert pravrtti._smrti_exploration_pool(world) == ["road"]
    canon.entities["road"].last_simulated_tick = 58          # asked lately: the scene keeps the draw
    assert pravrtti._smrti_exploration_pool(world) == ["ilo"]
    world.recent_memories.pop()
    # nobody engaged (the player stood alone): the ordinary pool, roads last
    world.recent_memories = [mem(1, "marren", 40, "inn"), mem(2, "teo", 59, "landing"), mem(4, "landing", 59, "landing"), mem(5, "ilo", 50, "inn")]
    canon.world_time = 64  # a far tick: the inn is not where the player lately stood
    assert set(pravrtti._smrti_exploration_pool(world)) >= {"marren", "ilo"}


class _BijaClient(_SmrtiClient):
    """Answers write_seed and write_arrival as well as append_tuples."""

    no_account = False  # True: the world at large comes back empty (a failed writing)

    def complete(self, system, user, *, max_tokens=8192):
        self.calls.append(user)
        assert "WORLD AT LARGE" in system
        if self.no_account:
            return ""
        return "THE YEAR. A mill on a river, high summer. THE REGIONS. The valley of the mill, and beyond it the downs to the north where the chieftain Ossur holds Hof; the sea a day west. THE CENTRAL FIGURES. Ossur the godi at Hof. THE WAYS. The north road. THE STARTING REGION. The mill ford." + " More of the same." * 12

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
        self.calls.append(user)
        if tool_name == "write_seed" and "EXPANDING" not in system:
            return {"name": "The Mill Ford", "setting": "A mill on a river, high summer.", "opening": "You stand at the ford, the wheel turning behind you.",
                    "places": [{"key": "ford", "label": "the mill ford", "kind": "open", "tuple": "A shallow ford below the mill; the north road climbs away from it."},
                               {"key": "north_road", "label": "the north road", "kind": "road", "tuple": "A cart track north out of the valley; a trader a week."}],
                    "people": [{"key": "wren", "label": "Wren, the miller's daughter", "place": "ford",
                                "tuples": [{"text": "I was born in the mill and have never slept away from the wheel.", "weight": "marking"},
                                           {"text": "This morning I came down to the ford to wait for the north trader.", "weight": "ordinary"}]}],
                    "player": "wren"}
        if tool_name == "write_seed" and "EXPANDING" in system:
            return {"name": "unchanged", "setting": "unchanged", "opening": "unchanged", "player": "unchanged",
                    "places": [{"key": "hof", "label": "Hof, the chieftain's hall", "kind": "building", "tuple": "The largest hall on the water, north along the north road from the mill ford."},
                               {"key": "the_mill_ford", "label": "The Mill Ford", "kind": "open", "tuple": "a repeat of a known place under another key"},
                               {"key": "sea_road", "label": "the sea road", "kind": "road", "tuple": "A track from Hof down to the coast; fishermen and the odd trader."}],
                    "people": [{"key": "ossur", "label": "Ossur the godi", "place": "hof", "tuples": [{"text": "I keep the peace on this water by memory more than by law.", "weight": "weighty"}]},
                               {"key": "gudrun", "label": "Gudrun Hallsdóttir, Wren's aunt", "place": "the_mill_ford", "tuples": [{"text": "I came down to the ford to see my brother's girl.", "weight": "ordinary"}]},
                               {"key": "lost", "label": "a man nowhere", "place": "no_such", "tuples": [{"text": "I stood where nothing was named for me to stand.", "weight": "ordinary"}]},
                               {"key": "hob2", "label": "Hob the ferryman", "name": "Hob", "place": "hof", "tuples": [{"text": "I have poled the ford crossing since before the mill was built.", "weight": "ordinary"}]}]}
        if tool_name == "place_start":  # a described start: the ford if it is the ford, else a new scene by it
            import re as _re
            said = _re.search(r"STARTS, IN THEIR PLAYER'S WORDS: (.*)", user)[1]
            keys = dict(_re.findall(r"^  (\w+): ([^(]+?) \(", user, _re.M))
            if "ford" in said:
                return {"existing": next(k for k, v in keys.items() if v == "the mill ford")}
            if "nowhere" in said:
                return {"existing": "", "label": "a hut", "in": "no_such", "tuple": "A hut."}
            return {"existing": "", "label": "the shed" if "shed" in said else "the willow hut", "name": "",
                    "in": next(k for k, v in keys.items() if v == "the mill ford"),
                    "tuple": "A willow hut on the bank above the mill ford, its door toward the water."}
        if tool_name == "write_arrival":
            import re as _re  # the newcomer is who they say they are ("I am Ann."): a name is never handed out twice
            who = (_re.search(r"I am (\w+)", user) or [None, "Hob"])[1]
            return {"label": "Hob, a tinker" if who == "Hob" else who, "name": who, "opening": "You come down to the ford with your pack.",
                    "tuples": [{"text": "I have walked the north road with my pack of pins for eleven years.", "weight": "weighty"},
                               {"text": "I came down to the ford this morning and found a girl waiting there.", "weight": "ordinary"}],
                    "place_tuple": "A tinker came down the north road to the ford with a pack on his back."}
        return super().complete_with_tool(system, user, tool_name=tool_name, tool_description=tool_description, input_schema=input_schema)


def test_a_newcomer_may_describe_where_they_start(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """Marc, 2026-10-01: "a free text entry option where a player can describe
    where they start". A listed scene that is the place is that place; else one
    new scene, in a listed place, planted with them in one commit, at the
    nearest online player's time (v5.24); nowhere the world has is refused, and so is a
    region within the world closed to strangers."""
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    client.post("/genesis", json={"self_description": "I am Wren.", "setting": "A mill on a river, high summer.", "breadth": "here"})
    world = client.app.state.world
    ford = next(e.id for e in world.canon.entities.values() if e.labels[0] == "the mill ford")
    r = client.post("/world/join", json={"self_description": "I am Hob.", "start_description": "at the mill ford, by the wheel"})
    assert r.status_code == 200, r.text
    hob = r.json()["player_entity_id"]
    assert {m.location_id for m in world.recent_memories if m.entity_id == hob} == {ford}  # the place it is, not a second
    places = sum(1 for e in world.canon.entities.values() if e.avarana)
    monkeypatch.setattr(spanda, "_nearest_players_clock", lambda w, start, new_place=False: 4242)  # the nearest online player's time (v5.24)
    r = client.post("/world/join", json={"self_description": "I am Ann.", "start_description": "in a hut of willow above the water"})
    assert r.status_code == 200, r.text
    ann = r.json()["player_entity_id"]
    hut = next(e for e in world.canon.entities.values() if e.labels[0] == "the willow hut")
    assert hut.avarana == "scene" and sum(1 for e in world.canon.entities.values() if e.avarana) == places + 1
    k = world.kosha()
    assert k.container_at(hut.id, world.canon.world_time + 1) == ford and k.scene_of(ann, world.canon.world_time + 1) == hut.id
    own = [m for m in world.recent_memories if m.entity_id in (ann, hut.id)]
    assert {m.location_id for m in own} == {hut.id}
    assert world.samskara().kala is not None and {m.began for m in own} == {4242}  # born with a clock (v4.49)
    assert len({m.pulse_id for m in own}) == 1  # one commit: the place and the person
    # a place the world has: the nearest online player's present too, never its own older clock (Aramus, Millhaven)
    r = client.post("/world/join", json={"self_description": "I am Cai.", "start_description": "at the mill ford"})
    cai = r.json()["player_entity_id"]
    assert {m.began for m in world.recent_memories if m.entity_id == cai} == {4242}
    assert client.post("/world/join", json={"self_description": "I am Ida.", "start_description": "nowhere at all"}).status_code == 502
    # a closed region within the world refuses a stranger
    world.records.append(Record("closed1", "prakriti", ford, json.dumps({"arrivals": False}), world.canon.world_time, "x", note="pacing"))
    world._samskara = None
    r = client.post("/world/join", json={"self_description": "I am Bo.", "start_description": "a shed beside the willow hut"})
    assert r.status_code == 409 and "admits no strangers" in r.json()["detail"]


def _join_world(clocks: dict, standing: dict, online: set, up: dict):
    """A stand-in world for spanda._nearest_players_clock: `clocks` player -> minute, `standing` player -> scene, `up` scene -> its
    containers nearest first, `online` the players who are."""
    class K:
        def scene_of(self, p, t):
            return standing[p]

        def above(self, scene, t):
            return list(up.get(scene, []))

    class Kala:
        def now(self, p):
            return clocks[p]

    class W:
        playable_entity_ids = frozenset(clocks)
        canon = type("C", (), {"world_time": 9})()

        def kosha(self):
            return K()

        def samskara(self):
            return type("S", (), {"kala": Kala()})()

        def online(self, p):
            return p in online
    return W()


def test_a_newcomer_joins_at_the_clock_of_the_nearest_online_player_in_the_tree() -> None:
    """v5.24 (Marc, 2026-10-07). Stonemarch, tick 1154: Kypazec joined the common room at minute 12959, 25 minutes ahead of Elivian and
    Filonius who stood in it at about 12934, because Aeon had just made a long move to the courtyard (ended 12959) and the rule was the
    furthest along player in the whole world. Now: the nearest online player in the containment tree, the latest of several at the same
    distance."""
    up = {"common_room": ["inn", "millhaven", "county"], "courtyard": ["estate", "county"], "wharf": ["river", "county"]}
    clocks = {"filonius": 12934, "elivian": 12938, "aeon": 12959}
    standing = {"filonius": "common_room", "elivian": "common_room", "aeon": "courtyard"}
    w = _join_world(clocks, standing, {"filonius", "elivian", "aeon"}, up)
    assert spanda._nearest_players_clock(w, "common_room") == 12938    # the two in the room: the later of them, not the far traveller's
    # counter-test: the old rule, the furthest along player anywhere, is exactly what put the newcomer 25 minutes out
    assert max(clocks.values()) == 12959
    # a free-text place planted in the common room: one level further, the same players are nearest
    assert spanda._nearest_players_clock(w, "common_room", new_place=True) == 12938
    # a place in the inn, beside the common room: the room's players are still nearer than the courtyard's
    up["parlour"] = ["inn", "millhaven", "county"]
    assert spanda._nearest_players_clock(w, "parlour") == 12938
    # a player who is away is not counted: their clock is stale (Elivian's page is closed)
    assert spanda._nearest_players_clock(_join_world(clocks, standing, {"filonius", "aeon"}, up), "common_room") == 12934
    # nobody near: the tree reaches a shared container (the county) and the nearest of the far ones decides, not the furthest
    up["inn"] = ["millhaven", "county"]
    far = _join_world({"aeon": 12959, "wren": 12800}, {"aeon": "courtyard", "wren": "inn"}, {"aeon", "wren"}, up)
    assert spanda._nearest_players_clock(far, "common_room") == 12800   # Wren, in the inn above the room, is nearer than Aeon: not the furthest along
    # no one online: all the players are read instead, and there is still an answer
    assert spanda._nearest_players_clock(_join_world(clocks, standing, set(), up), "common_room") == 12938
    # no players at all: no clock (the place's own time then stands, in the caller)
    assert spanda._nearest_players_clock(_join_world({}, {}, set(), up), "common_room") is None


def test_v3_genesis_plants_the_narrators_seed_and_join_adds_a_newcomer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """bija (2026-09-20): a live world is born as tuples -- entities with
    opaque ids, first tuples, session config -- with no candidates,
    relations or deltas; a newcomer joins at a place a story already is."""
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    resp = client.post("/genesis", json={"self_description": "I am Wren, the miller's daughter.", "setting": "A mill on a river.", "breadth": "wide"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    world = client.app.state.world
    wren = body["player_entity_id"]
    # bhuvana first (wide only), fixed at birth, handed to the seed and readable
    assert world.bhuvana and world.bhuvana.startswith("THE YEAR.") and fake_store.bhuvana == world.bhuvana
    seed_ask = next(u for u in roles.smrti_client.calls if "BREADTH: wide" in u)
    assert "THE WORLD AT LARGE" in seed_ask and "STARTING REGION of the world at large" in seed_ask
    assert client.get("/world/bhuvana").json()["bhuvana"] == world.bhuvana
    assert body["narration"].startswith("You stand at the ford") and world.playable_entity_ids == [wren]
    assert {e.labels[0]: e.avarana for e in world.canon.entities.values() if e.avarana is not None} == {"the mill ford": "scene", "the north road": "scene"}  # v3 kinds plant as the scene bit (2026-10-03)
    assert [m.salience for m in world.recent_memories if m.entity_id == wren] == [2, 5]
    ford = next(e.id for e in world.canon.entities.values() if e.labels[0] == "the mill ford")
    assert world.entry_point_location_ids == [ford] and world.setting_description == "A mill on a river, high summer."
    canon, result, _ = fake_store.commit_calls[-1]
    assert len(result.promoted_entities) == 4  # ford, road, Wren, and the world at large
    from smritilila.smrti.genesis import ITIHASA
    chronicle = next(e for e in world.canon.entities.values() if e.origin == ITIHASA)
    assert chronicle.labels == ["the world at large"]
    own = [m for m in world.recent_memories if m.entity_id == chronicle.id]
    assert len(own) == 1 and own[0].location_id == "nowhere" and own[0].experience.startswith("THE YEAR.")
    assert roles.narrator.resolved_events == []  # the v2 Narrator never ran
    # entry points come from tuples: where Wren stands
    assert [e["location_id"] for e in client.get("/world/entry-points").json()] == [ford]
    # a newcomer
    resp = client.post("/world/join", json={"entry_location_id": ford, "self_description": "I am Hob, a tinker with a pack of pins."})
    assert resp.status_code == 200, resp.text
    hob = resp.json()["player_entity_id"]
    assert world.playable_entity_ids == [wren, hob] and world.canon.entities[hob].labels == ["Hob, a tinker"]
    own = [m for m in world.recent_memories if m.entity_id == hob]
    assert [m.location_id for m in own] == [ford, ford] and [m.salience for m in own] == [3, 5]
    # the arrival is the scene's record, witnessed by everyone there: the newcomer and whoever stood at the ford
    assert any(m.entity_id == ford and "tinker came down" in m.experience and hob in m.subject_ids for m in world.recent_memories)
    assert world.presence[hob] == "online" and hob in world.last_seen  # seen: not idle since the server started
    assert client.get("/world/entry-points").json() == [{"location_id": ford}]
    # where a newcomer may arrive: the busiest places by tuples ever formed there, then where players stand
    locs = client.get("/world/locations").json()
    road = next(e.id for e in world.canon.entities.values() if e.labels[0] == "the north road")
    assert [l["location_id"] for l in locs] == [ford, road]
    assert locs[0]["players_here"] == ["Wren, the miller's daughter", "Hob, a tinker"] and locs[1]["players_here"] == []
    assert locs[0]["memories"] == sum(1 for m in world.recent_memories if m.location_id == ford)
    assert client.post("/world/join", json={"entry_location_id": "no_such", "self_description": "I am Ann."}).status_code == 422
    resp = client.post("/world/join", json={"entry_location_id": road, "self_description": "I am Ann."})
    assert resp.status_code == 200, resp.text  # a busy place with no player in it is open to a newcomer
    ann = resp.json()["player_entity_id"]
    assert ann in world.playable_entity_ids and {m.location_id for m in world.recent_memories if m.entity_id == ann} == {road}
    # vistara: the world grows outward; a repeat is refused, a known place is stood in by key, the unplaced are dropped
    before = set(world.canon.entities)
    t0 = world.canon.world_time
    resp = client.post("/world/expand")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"places": ["Hof, the chieftain's hall", "the sea road"], "people": ["Ossur the godi", "Gudrun Hallsdóttir, Wren's aunt"]}
    added = {e.labels[0]: e for e in world.canon.entities.values() if e.id not in before}
    assert set(added) == {"Hof, the chieftain's hall", "the sea road", "Ossur the godi", "Gudrun Hallsdóttir, Wren's aunt"}  # a second Hob refused: the name is taken
    assert added["the sea road"].avarana == "scene" and added["Hof, the chieftain's hall"].created_at_world_time == t0
    gudrun = added["Gudrun Hallsdóttir, Wren's aunt"]
    assert [m.location_id for m in world.recent_memories if m.entity_id == gudrun.id] == [ford]  # stood in a place that existed, by key
    assert all(m.world_time == t0 for m in world.recent_memories if m.entity_id in (added["the sea road"].id, added["Hof, the chieftain's hall"].id))
    assert world.playable_entity_ids == [wren, hob, ann]  # nobody new is playable
    # the second expansion is shown the outer ring only: what the first planted, not the seed
    calls_before = len(roles.smrti_client.calls)
    resp = client.post("/world/expand")
    assert resp.status_code in (200, 502)  # the fake returns the same ring again -> all refused -> 502 is fine
    ask = roles.smrti_client.calls[-1]
    assert "THE WORLD AT LARGE" in ask  # the expansion reads it too
    assert "Hof, the chieftain's hall" in ask and "the sea road" in ask and "Ossur the godi" in ask and "Gudrun Hallsdóttir, Wren's aunt" in ask
    assert "the mill ford (open)" not in ask and "Wren, the miller's daughter @" not in ask
    assert "Wren" in ask  # ... but her name is among the names taken


class _BriefClient(_BijaClient):
    """A world from a brief: the seed has no player, and begins at the ford."""

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
        if tool_name == "write_seed":
            self.calls.append(user)
            return {"name": "the Narrator's own name", "setting": "A mill on a river, high summer.", "begins": "ford",
                    "places": [{"key": "valley", "label": "the mill valley", "in": "", "scene": False, "tuple": "A green valley with a mill in it; the godi at Hof holds it."},
                               {"key": "ford", "label": "the mill ford", "in": "valley", "scene": True, "tuple": "A shallow ford below the mill; the north road climbs away from it."},
                               {"key": "loft", "label": "the mill loft", "in": "valley", "scene": True, "tuple": "Sacks of grain under the mill roof, the wheel's noise in the boards."}],
                    "people": [{"key": "wren", "label": "Wren, the miller's daughter", "place": "ford",
                                "tuples": [{"text": "I was born in the mill and have never slept away from the wheel.", "weight": "marking"}]},
                               {"key": "miller", "label": "Ottar the miller", "place": "ford",
                                "tuples": [{"text": "I have ground this valley's grain for thirty summers.", "weight": "weighty"}]}]}
        return super().complete_with_tool(system, user, tool_name=tool_name, tool_description=tool_description, input_schema=input_schema)


def test_a_world_from_a_brief_waits_for_its_first_arrival(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """The new world screen (2026-09-24): a world born with no player.
    Nobody plays it and nobody is crowned on reload; a newcomer chooses
    from its busiest places and becomes its first player."""
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BriefClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    assert client.post("/genesis", json={"name": "The Mill", "breadth": "here"}).status_code == 422  # where and when is required
    assert client.post("/genesis", json={"where_when": "Iceland, 1120", "breadth": "here"}).status_code == 422  # so is a name
    assert client.post("/genesis", json={"name": "The Mill", "where_when": "Iceland, 1120", "breadth": "here"}).status_code == 422  # and a source
    resp = client.post("/genesis", json={"name": "The Mill", "source": "History: the Icelandic Commonwealth", "where_when": "Iceland, 1120",
                                         "different": "Ossur the godi is a woman", "breadth": "here"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["player_entity_id"] == "" and resp.json()["narrations"] == {}
    ask = roles.smrti_client.calls[-1]
    assert "WORLD NAME: The Mill" in ask and "WHAT IS DIFFERENT HERE: Ossur the godi is a woman" in ask and "WHERE IT BEGINS: not given" in ask
    assert ask.index("SOURCE: History: the Icelandic Commonwealth") < ask.index("WHERE AND WHEN")   # the source leads the brief
    assert client.app.state.world.world_data["source"] == "History: the Icelandic Commonwealth" and fake_store._view[2]["source"]
    world = client.app.state.world
    ford = next(e.id for e in world.canon.entities.values() if e.labels[0] == "the mill ford")
    assert world.playable_entity_ids == [] and world.scene_owner_entity_id == "" and world.entry_point_location_ids == [ford]
    assert fake_store._session_config[:3] == ("", [ford], [])
    assert client.get("/players").json() == []
    assert client.post("/pulses", json={"action_text": "waits"}).status_code == 422  # no one to act as
    # a reload keeps it playerless: no NPC is crowned
    fake_store._loaded = world.canon
    del client.app.state.world
    assert client.get("/players").json() == []  # get_world reloads from the store
    world = client.app.state.world
    assert world.playable_entity_ids == [] and world.scene_owner_entity_id == "" and len(world.recent_memories) == 6  # and its earth's
    # where a newcomer may arrive: the seed's own places, busiest first (the ford: two people and its own tuple)
    locs = client.get("/world/locations").json()
    assert locs[0]["location_id"] == ford and locs[0]["players_here"] == []
    resp = client.post("/world/join", json={"entry_location_id": ford, "self_description": "I am Hob, a tinker with a pack of pins."})
    assert resp.status_code == 200, resp.text
    hob = resp.json()["player_entity_id"]
    assert world.playable_entity_ids == [hob] and world.scene_owner_entity_id == hob  # the first to arrive
    assert fake_store._session_config[:3] == (hob, [ford], [hob])


def test_a_world_born_without_its_age_is_given_one_once(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """POST /world/bhuvana (2026-09-21): the crossing, Tombstone, Rome and
    Bywater were born before bhuvana. The age is written from the setting
    and the player's own first tuples; a second ask is refused. A world is
    born with its earth now (Marc, 2026-09-30), quiet, perceiving by its
    setting: the age is given to it, never a second world at large."""
    from smritilila.smrti.genesis import ITIHASA
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    roles.smrti_client.no_account = True  # below wide the account is written after the seed (2026-10-05); here it fails
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    resp = client.post("/genesis", json={"self_description": "I am Wren, the miller's daughter.", "setting": "A mill on a river.", "breadth": "here"})
    assert resp.status_code == 200, resp.text
    world = client.app.state.world
    wren = resp.json()["player_entity_id"]
    earths = [e for e in world.canon.entities.values() if e.origin == ITIHASA]
    assert world.bhuvana is None and len(earths) == 1                                 # born with its earth, keeping no age
    assert sum("THE WORLD AS ALREADY SEEDED" in c for c in roles.smrti_client.calls) == 2  # two attempts, then born without
    roles.smrti_client.no_account = False
    world.canon.world_time = 40  # lived in for a while
    commits = len(fake_store.commit_calls)
    resp = client.post("/world/bhuvana")
    assert resp.status_code == 200, resp.text
    assert resp.json()["bhuvana"].startswith("THE YEAR.") and world.bhuvana == fake_store.bhuvana == resp.json()["bhuvana"]
    ask = roles.smrti_client.calls[-1]
    assert "SETTING (the player's own words" in ask and "I am Wren, the miller's daughter." in ask  # the player's own first tuple stands in
    chronicle = next(e for e in world.canon.entities.values() if e.origin == ITIHASA)
    assert chronicle is earths[0]                                                     # the same earth, given its age
    own = sorted((m for m in world.recent_memories if m.entity_id == chronicle.id), key=lambda m: m.world_time)
    assert [m.world_time for m in own] == [0, 40] and own[-1].location_id == "nowhere" and own[-1].experience.startswith("THE YEAR.")
    canon, result, _ = fake_store.commit_calls[-1]
    assert len(fake_store.commit_calls) == commits + 1 and result.promoted_entities == []
    assert world.playable_entity_ids == [wren]
    assert client.get("/world/bhuvana").json() == {"bhuvana": world.bhuvana, "chronicle": []}
    # once only
    assert client.post("/world/bhuvana").status_code == 409


def test_below_wide_the_world_at_large_follows_the_seed_and_at_wide_it_comes_first(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Marc, 2026-10-05: every world has its wider account. Written first it made Austin's seed flatter and smaller
    (11-14 places against 16-18; 10-12 siblings under one region), so below wide it follows the seed, from the
    setting and the seed as seeded; at wide it still comes first and the seed grows within it."""
    from smritilila.smrti.genesis import ITIHASA
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    resp = client.post("/genesis", json={"self_description": "I am Wren, the miller's daughter.", "setting": "A mill on a river.", "breadth": "here"})
    assert resp.status_code == 200, resp.text
    world = client.app.state.world
    calls = roles.smrti_client.calls
    seed_at = next(i for i, c in enumerate(calls) if "BREADTH: here" in c)
    account_at = next(i for i, c in enumerate(calls) if "THE WORLD AS ALREADY SEEDED" in c)
    assert seed_at < account_at and "THE WORLD AT LARGE" not in calls[seed_at]         # the seed was written without one
    asked = calls[account_at]
    assert "the mill ford" in asked and "Wren, the miller's daughter" in asked and "IT BEGINS AT: the mill ford" in asked
    assert world.bhuvana.startswith("THE YEAR.") and fake_store.bhuvana == world.bhuvana   # kept, and saved
    earth = next(e for e in world.canon.entities.values() if e.origin == ITIHASA)
    assert any(m.entity_id == earth.id and m.experience.startswith("THE YEAR.") for m in world.recent_memories)  # the earth bears it
    # at wide: the account first (one call), the seed within it
    roles.smrti_client = _BijaClient()
    client.post("/world/clear")
    resp = client.post("/genesis", json={"self_description": "I am Wren, the miller's daughter.", "setting": "A mill on a river.", "breadth": "wide"})
    assert resp.status_code == 200, resp.text
    calls = roles.smrti_client.calls
    assert not any("THE WORLD AS ALREADY SEEDED" in c for c in calls)                  # never written after a seed that grew within it
    accounts = [c for c in calls if "SETTING (the player's own words" in c and "BREADTH:" not in c]  # the seed's ask names its breadth
    seeds = [c for c in calls if "BREADTH: wide" in c]
    assert len(accounts) == 1 and calls[0] is accounts[0] and seeds and all("THE WORLD AT LARGE" in c for c in seeds)  # a floor's extension is a seed too
    assert client.app.state.world.bhuvana.startswith("THE YEAR.")


def test_a_world_from_a_brief_has_its_account_written_after_the_seed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """The route Austin was made by (the new world screen, no player): its brief and the seed as seeded are both in
    the ask for the world at large, and the seed was written without one."""
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BriefClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    resp = client.post("/genesis", json={"name": "The Mill", "source": "History: the Icelandic Commonwealth", "where_when": "Iceland, 1120",
                                         "different": "Ossur the godi is a woman", "breadth": "around"})
    assert resp.status_code == 200, resp.text
    calls = roles.smrti_client.calls
    seed = next(c for c in calls if "BREADTH: around" in c)
    asked = next(c for c in calls if "THE WORLD AS ALREADY SEEDED" in c)
    assert "THE WORLD AT LARGE" not in seed and calls.index(seed) < calls.index(asked)
    assert "WORLD NAME: The Mill" in asked and "WHAT IS DIFFERENT HERE: Ossur the godi is a woman" in asked                   # the brief
    assert "the mill ford (in the mill valley)" in asked and "Wren, the miller's daughter" in asked      # and the seed
    assert client.app.state.world.bhuvana.startswith("THE YEAR.")


def test_the_seeded_block_names_what_the_seed_built() -> None:
    from smritilila.smrti.bija import seeded_block
    block = seeded_block({"setting": "A mill on a river.", "entry_points": ["ford"],
                          "places": {"vale": {"label": "the vale"}, "ford": {"label": "the mill ford", "in": "vale"},
                                     "orphan": {"label": "the lost shed", "in": "no_such"}},
                          "people": {"wren": {"label": "Wren, the miller's daughter"}}})
    assert "SETTING AS SEEDED: A mill on a river." in block and "IT BEGINS AT: the mill ford" in block
    assert "PLACES SEEDED: the vale; the mill ford (in the vale); the lost shed" in block      # a parent not in the seed is not named
    assert "PEOPLE SEEDED: Wren, the miller's daughter" in block and "not figures heard of" in block
    assert seeded_block({}).count("\n") == 0                                            # an empty seed: only the heading


def test_smrti_pool_asks_a_far_road_only_on_far_ticks() -> None:
    """Eyri at wide, tick 4: the road to the Althing, a fortnight's ride
    from the player, took the world's second activation."""
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    canon = Maya(world_time=30, entities={
        "racks": Entity("racks", ["the drying racks"], 0, "genesis", avarana="open"),
        "althing_road": Entity("althing_road", ["the road to the Althing"], 0, "genesis", avarana="road"),
        "halla": Entity("halla", ["Halla"], 0, "genesis"),
    })
    world = WorldSession(canon=canon, scene_owner_entity_id="halla", entry_point_location_ids=["racks"], playable_entity_ids=["halla"])
    world.world_data["pacing"] = {"arrivals_on_traffic": False}  # asked on the clock: the ask itself, not its traffic gate
    world.recent_memories = [Memory("m1", "halla", frozenset(), "x", "PRIMARY", 4, "racks", 1, "p1", None)]
    assert pravrtti._smrti_exploration_pool(world) == []                # tick 30: far, waits
    canon.world_time = 32
    assert pravrtti._smrti_exploration_pool(world) == ["althing_road"]  # a far tick


def test_new_world_slug_and_registry(tmp_path) -> None:
    """POST /worlds/create (2026-09-20): a database name from the world's
    name, and the registry appended without touching other entries."""
    from smritilila.worlds_registry import add_world, load_worlds_registry
    assert loka._db_slug("Eyrarfjörður, 1120") == "smritilila_eyrarfjorur_1120"
    assert loka._db_slug("The Shire -- Bywater, 3018") == "smritilila_the_shire_bywater_3018"
    assert loka._db_slug("!!!") == "smritilila_world"
    cfg = tmp_path / "worlds.json"
    cfg.write_text('{"worlds": [{"name": "Tombstone 1880", "database_url": "postgresql+asyncpg://a:a@h/smritilila"}]}', encoding="utf-8")
    add_world("Bywater, 3018", "postgresql+asyncpg://a:a@h/smritilila_bywater_3018", path=cfg)
    names = [w.name for w in load_worlds_registry(cfg)]
    assert names == ["Tombstone 1880", "Bywater, 3018"]
    import pytest as _pt
    with _pt.raises(ValueError):
        add_world("bywater, 3018", "x", path=cfg)  # the name is taken, case aside


def test_the_chronicle_is_drawn_on_far_ticks_and_a_road_hears_its_news(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """Itihasa (2026-09-20): the world at large is a perceiver standing
    nowhere, drawn like a far actor; its turn is the chronicle of the
    age against the bhuvana; a road's turn is told the latest."""
    import asyncio
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.smrti.genesis import ITIHASA
    from smritilila.smrti.narrator import CHRONICLE_ACTION, SCENE_ARRIVAL_ACTION
    canon = Maya(world_time=32, entities={
        "bree": Entity("bree", ["Bree"], 0, "genesis", avarana="open"),
        "east_road": Entity("east_road", ["the East Road"], 0, "genesis", avarana="road"),
        "amaranth": Entity("amaranth", ["Amaranth"], 0, "genesis"),
        "age": Entity("age", ["the world at large"], 0, ITIHASA),
    })
    def mem(i, who, t, loc, text="x"):
        return Memory(f"m{i}", who, frozenset(), text, "PRIMARY", 4, loc, t, f"p{t}", None)
    world = WorldSession(canon=canon, scene_owner_entity_id="amaranth", entry_point_location_ids=["bree"], playable_entity_ids=["amaranth"])
    world.world_data["pacing"] = {"arrivals_on_traffic": False}  # asked on the clock: the ask itself, not its traffic gate
    world.bhuvana = "THE YEAR. 3018. THE REGIONS. Bree. THE CENTRAL FIGURES. A Steward far south."
    world.recent_memories = [mem(1, "amaranth", 30, "bree"), mem(2, "age", 0, "nowhere", "It is 3018 and the Nine have ridden out.")]
    # tick 32 is a far tick and the age has never moved: it takes the draw alone; at 34 it is not in the pool
    assert pravrtti._smrti_exploration_pool(world) == ["age"]
    canon.world_time = 34
    assert "age" not in pravrtti._smrti_exploration_pool(world)
    canon.world_time = 32
    canon.entities["age"].last_simulated_tick = 20  # moved 12 ticks ago: waits its window, competes as a far actor
    assert pravrtti._smrti_exploration_pool(world) != ["age"] and "age" in pravrtti._smrti_exploration_pool(world)
    canon.entities["age"].last_simulated_tick = None
    # its turn: the chronicle action carries the bhuvana; it writes one tuple standing nowhere
    calls: list[str] = []
    def fake_pulse(canon_, actor, action, world_time, pulse_id, mems, client, embed, **kw):
        from smritilila.smrti.pulse import SmrtiPulseResult
        calls.append((actor, kw.get("as_chronicle"), kw.get("news")))
        return SmrtiPulseResult("The age moves.", [Memory("m-age", "age", frozenset(), "The Steward sent his son north.", "PRIMARY", 3, "nowhere", world_time, pulse_id, None)], [], "", 1, ["age"])
    monkeypatch.setattr(spanda, "run_smrti_pulse", fake_pulse)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    monkeypatch.setattr(main, "log_event", lambda event, **f: None)
    roles = _fake_roles(_SpyNarrator()); roles.smrti_client = _SmrtiClient()
    asyncio.run(spanda._run_smrti_activation_in_background(world, roles, "age", 1))
    assert calls[-1][0] == "age" and calls[-1][1] == world.bhuvana and calls[-1][2] is None
    assert itihasa._age_at(world, None)[0][-1] == "The Steward sent his son north."  # a world keeping no clock: every entry is reached
    assert "THE CHRONICLE SINCE" in itihasa._at_large(world) and "sent his son north" in itihasa._at_large(world)
    # a road's turn is told the news
    asyncio.run(spanda._run_smrti_activation_in_background(world, roles, "east_road", 1))
    assert calls[-1][0] == "east_road" and calls[-1][1] is None and calls[-1][2] == "The Steward sent his son north."
    # and the real pulse puts it in the road's action
    from smritilila.smrti.pulse import run_smrti_pulse
    class Rec:
        def complete_with_tool(self, system, user, **k):
            self.user = user
            return {"narration": "x", "perceiver_tuples": [], "scene_tuples": [{"scene": "east_road", "text": "The road lay empty under the evening; a rider passed toward Bree with news.", "present": []}], "suggested_next_action": ""}
    rec = Rec()
    run_smrti_pulse(canon, "east_road", None, 34, "p34", world.recent_memories + [Memory("m3", "east_road", frozenset(), "The road lay quiet all day under a low sky.", "AMBIENT", 5, "east_road", 20, "p20", None)],
                    rec, None, as_origin=True, news="The Steward sent his son north.")
    assert SCENE_ARRIVAL_ACTION in rec.user and "NEWS OF THE WIDE WORLD" in rec.user and "sent his son north" in rec.user
    class Chron(Rec):
        def complete_with_tool(self, system, user, **k):
            self.user = user
            return {"narration": "Spring deepens; Denethor doubles the watch on the river and sends riders north.", "perceiver_tuples": [], "scene_tuples": [], "actor_stands": "nowhere", "suggested_next_action": ""}
    rec2 = Chron()
    r = run_smrti_pulse(canon, "age", None, 35, "p35", world.recent_memories, rec2, None, as_chronicle=world.bhuvana)
    assert CHRONICLE_ACTION in rec2.user and "A Steward far south" in rec2.user
    assert "Bree" not in rec2.user.split("ACTION:")[0].replace("It is 3018 and the Nine have ridden out.", "")  # own stream only: no Bree tuple, no Amaranth
    # the chronicle came as the narration: it is the age's tuple, standing nowhere
    assert [(m.entity_id, m.location_id) for m in r.memories_written] == [("age", "nowhere")]
    assert r.memories_written[0].experience.startswith("Spring deepens")


def test_the_chronicle_runs_ahead_but_a_scene_is_shown_it_by_its_own_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Marc, 2026-09-25 (collision run, question 1): the age's entries carry
    the span they cover; a scene hears as news only what its clock has
    passed, and the next as the age's pull; the chronicle waits while more
    than a season ahead of the latest player. The age's time never dates a
    tick it shares with the scenes' pulses (the collide copy's 666)."""
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.memory.record import Record
    from smritilila.smrti.genesis import ITIHASA
    from smritilila.smrti.kala import AGE_LEAD, Kala
    from smritilila.smrti.pulse import run_smrti_pulse
    monkeypatch.setattr(main, "log_event", lambda event, **f: None)
    D = 24 * 60
    START = 25 * 60  # Saturday 28 April 1928, 1:00 am
    canon = Maya(world_time=40, entities={
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="building"),
        "wen": Entity("wen", ["Wen Tolland"], 0, "genesis"),
        "age": Entity("age", ["the world at large"], 0, ITIHASA),
    })
    def mem(i, who, t, loc, text, pid, began=None, ended=None):
        return Memory(f"m{i}", who, frozenset(), text, "PRIMARY", 4, loc, t, pid, None, began=began, ended=ended)
    world = WorldSession(canon=canon, scene_owner_entity_id="wen", entry_point_location_ids=["inn"], playable_entity_ids=["wen"])
    world.world_data["clock"] = {"epoch": "1928-04-27T00:00", "start": START}
    world.bhuvana = "THE YEAR. 1928."
    world.recent_memories = [mem(1, "age", 0, "nowhere", "THE YEAR. 1928, spring.", "genesis"),
                             mem(2, "wen", 30, "inn", "I took a room at the inn.", "p30", START, START + 60),
                             mem(3, "age", 31, "nowhere", "MAY. The river rose.", "p31"),
                             mem(4, "age", 32, "nowhere", "JUNE. The mill burned.", "p32")]
    world.records = [Record("k1", "kosha", "wen", "inn", 0, "genesis"),
                     Record("a1", "time", "age", str(START + 34 * D), 31, "p31", note=f"{START}||"),
                     Record("a2", "time", "age", str(START + 64 * D), 32, "p32", note=f"{START + 34 * D}||")]
    # the age's time is not the scenes': it dates no tick, gives no scene a clock
    k = world.samskara().kala
    assert k.tick_minute.get(31) is None and k.clock("age") is None and world.samskara().age["p32"] == (START + 34 * D, START + 64 * D)
    assert Kala.of(world.world_data["clock"], world.records).tick_minute.get(31) == START + 34 * D  # (without aside, it would)
    # the inn in April: the year as news, May as the pull; in mid-May, the same; in June, May is news and June the pull
    assert itihasa._age_at(world, START + 60) == (["THE YEAR. 1928, spring."], "MAY. The river rose.")
    assert itihasa._age_at(world, START + 35 * D)[0][-1] == "MAY. The river rose." and itihasa._age_at(world, START + 35 * D)[1] == "JUNE. The mill burned."
    assert itihasa._age_at(world, None)[1] is None                       # a world keeping no clock: every entry reached
    at_large = itihasa._at_large(world, START + 60)
    assert "THE CHRONICLE SINCE" not in at_large and "THE AGE'S PULL" in at_large and "river rose" in at_large
    # the next entry begins where the age has reached; it waits while more than a season ahead
    assert itihasa._age_clock(world) == START + 64 * D and 64 * D - 60 < AGE_LEAD
    assert pravrtti._kala_gate(world, ["age"])[0] == ["age"]
    world.records.append(Record("a3", "time", "age", str(START + 200 * D), 33, "p33", note=f"{START + 64 * D}||"))
    world.recent_memories.append(mem(5, "age", 33, "nowhere", "AUTUMN.", "p33"))
    assert pravrtti._kala_gate(world, ["age"])[0] == []
    # the chronicle's own turn: told the age's time, it writes the span it covers
    class Chron:
        def complete_with_tool(self, system, user, **kw):
            self.user = user
            return {"narration": "JULY. The harvest came in early.", "perceiver_tuples": [], "scene_tuples": [], "actor_stands": "nowhere",
                    "suggested_next_action": "", "elapsed_minutes": 43800}
    c = Chron()
    r = run_smrti_pulse(canon, "age", None, 41, "p41", world.recent_memories, c, None, as_chronicle=world.bhuvana,
                        age_from=START + 64 * D, view={"world_data": world.world_data}, records=world.records)
    assert "the age has reached Sunday, 1 July 1928" in c.user
    [t] = [x for x in r.records_written if x.kind == "time"]
    assert (t.entity_id, t.value, t.note) == ("age", str(START + 64 * D + 43800), f"{START + 64 * D}||")
    assert all(m.began is None and m.ended is None for m in r.memories_written)  # the entry itself is not on the scenes' clock
    # a scene's turn is shown the pull as what has not happened
    class Rec:
        def complete_with_tool(self, system, user, **kw):
            self.user = user
            return {"narration": "x", "perceiver_tuples": [], "scene_tuples": [], "suggested_next_action": ""}
    rec = Rec()
    run_smrti_pulse(canon, "inn", None, 42, "p42", world.recent_memories, rec, None, as_origin=True, news="THE YEAR.",
                    pull="MAY. The river rose.")
    assert "NOT YET HAPPENED here" in rec.user and "river rose" in rec.user


def test_a_room_pauses_around_a_player_who_is_away(monkeypatch: pytest.MonkeyPatch) -> None:
    """Marc, 2026-09-26 ("the room pauses around them"): the Green Door
    parlour drew four turns while Wentworth was offline, each written for him
    and blocked. A scene holding a player who is offline or out of the world,
    and does not let the world act for them, takes no autonomous turns -- its
    NPCs, its things, the scene itself -- until they are back."""
    import time as _time
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.memory.record import Record
    monkeypatch.setattr(main, "log_event", lambda event, **f: None)
    canon = Maya(world_time=40, entities={
        "parlour": Entity("parlour", ["the parlour"], 0, "genesis", avarana="scene"),
        "shanty": Entity("shanty", ["the shanty"], 0, "genesis", avarana="scene"),
        "wen": Entity("wen", ["Wentworth"], 0, "genesis"), "nat": Entity("nat", ["Nathaniel"], 0, "genesis"),
        "aunt": Entity("aunt", ["the aunt"], 0, "genesis"), "kettle": Entity("kettle", ["the kettle"], 0, "genesis"),
        "boatman": Entity("boatman", ["the boatman"], 0, "genesis")})
    world = WorldSession(canon=canon, scene_owner_entity_id="nat", entry_point_location_ids=["shanty"], playable_entity_ids=["nat", "wen"])
    world.records = [Record(f"r{i}", "kosha", e, into, 0, "p0") for i, (e, into) in enumerate(
        [("wen", "parlour"), ("aunt", "parlour"), ("kettle", "parlour"), ("nat", "shanty"), ("boatman", "shanty")])]
    world.recent_memories = [Memory(f"m{i}", e, frozenset(), "I was here.", "PRIMARY", 4, loc, 1, "p1", None)
                             for i, (e, loc) in enumerate([("aunt", "parlour"), ("kettle", "parlour"), ("boatman", "shanty"),
                                                           ("wen", "parlour"), ("nat", "shanty")])]
    world.last_seen["nat"] = _time.monotonic()  # Nathaniel is playing; Wentworth has not been seen
    pool = ["aunt", "kettle", "parlour", "boatman", "shanty"]
    assert pravrtti._paused(world) == {"parlour"}
    assert pravrtti._hold_for_away(world, pool) == ["boatman", "shanty"]
    world.logged_off.add("wen")  # out of the world now: the room still holds for him
    assert pravrtti._hold_for_away(world, pool) == ["boatman", "shanty"]
    world.autonomous.add("wen")  # he lets the world act for him: the room goes on
    assert pravrtti._hold_for_away(world, pool) == pool
    world.autonomous.discard("wen"); world.logged_off.discard("wen")
    world.last_seen["wen"] = _time.monotonic()  # back: the room goes on
    assert pravrtti._hold_for_away(world, pool) == pool


def test_absorb_makes_two_ids_one_and_reloads(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """s10 as a mechanism (2026-09-20): gone's tuples become keep's, gone's
    label becomes keep's alias (a name record, v5 §16: no identity tuple is
    appended), the id is dropped, and the world reloads from the store."""
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    _seed_square_world(client)
    world = client.app.state.world
    owner = world.scene_owner_entity_id
    world.canon.entities["walker"] = Entity("walker", ["hooded walker eastroad"], 1, "smrti")
    world.canon.entities["walker2"] = Entity("walker2", ["hooded traveler on foot"], 3, "smrti")
    world.recent_memories.append(Memory("w1", "walker", frozenset(), "I walked in from the east.", "PRIMARY", 4, "square", 2, "p2", None))
    # refused: a place and a person; a playable character; self
    assert client.post(f"/entities/walker/absorb/square").status_code == 422
    assert client.post(f"/entities/walker/absorb/{owner}").status_code == 422
    assert client.post("/entities/walker/absorb/walker").status_code == 422
    assert client.post("/entities/walker/absorb/nobody").status_code == 404
    resp = client.post("/entities/walker/absorb/walker2")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"kept": "walker", "gone": "walker2", "label": "hooded walker eastroad", "alias": "hooded traveler on foot", "tuples": 2, "filed": 1, "presence": 3, "records": 0, "contained": 0}
    assert fake_store.absorbed == ("walker", "walker2")
    canon, result, _ = fake_store.commit_calls[-1]
    # v5 §16, §7: the alias is a name record -- no English tuple in memory, no label edited
    assert "walker2" not in canon.entities and canon.entities["walker"].labels == ["hooded walker eastroad"]
    assert result.memories_written == []
    assert [(r.kind, r.entity_id, r.value, r.note) for r in fake_store._records] == [("name", "walker", "hooded traveler on foot", "alias")]
    assert not hasattr(client.app.state, "world")  # reloaded on next use



def test_what_a_player_witnessed_is_handed_over_as_it_happens(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /entities/{id}/witnessed (2026-09-21): the pending witnessed
    tuples, drained -- the page polls it between actions so a witnessed
    tuple reaches the player when it is written, not with their next act."""
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    canon = Maya(world_time=3, entities={"teo": Entity("teo", ["Teo"], 0, "genesis")})
    world = adhisthana.WorldSession(canon=canon, scene_owner_entity_id="teo", playable_entity_ids=["teo"])
    client.app.state.world = world
    owner = world.playable_entity_ids[0]
    world.pending_witnessed = {owner: [{"place": "the inn", "text": "The pedlar woman answered the boy's question at last.", "world_time": 3}]}
    r = client.get(f"/entities/{owner}/witnessed")
    assert r.status_code == 200 and r.json() == {"witnessed": [{"place": "the inn", "text": "The pedlar woman answered the boy's question at last.", "world_time": 3}]}
    assert client.get(f"/entities/{owner}/witnessed").json() == {"witnessed": []}  # drained: shown once
    assert client.get("/entities/nobody/witnessed").status_code == 422


def test_the_scene_takes_the_draw_on_every_second_player_pulse_and_a_witnessed_tuple_is_not_a_player_pulse() -> None:
    """Cadence (Marc, 2026-09-21; Ravenmoor 137-212: Corwin reached the
    ledger, said the name, took the bible, started the car between the
    player's turns). And the engaged set reads the player's OWN pulses:
    a witnessed place tuple under the player's id is the NPC's pulse."""
    from smritilila.graph.canon import Maya
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    canon = Maya(world_time=60, entities={
        "lab": Entity("lab", ["the lab"], 0, "genesis", avarana="building"),
        "far": Entity("far", ["the far field"], 0, "genesis", avarana="open"),
        "nat": Entity("nat", ["Nathaniel"], 0, "genesis"),
        "corwin": Entity("corwin", ["Corwin"], 0, "genesis"),
        "hartley": Entity("hartley", ["Hartley"], 0, "genesis"),
        "farmer": Entity("farmer", ["a farmer"], 0, "genesis"),
    })
    for e in ("corwin", "hartley", "farmer"):
        canon.entities[e].last_simulated_tick = 40
    def mem(i, who, t, loc, text=None, tier="PRIMARY", pulse=None):
        return Memory(f"m{i}", who, frozenset(), text or f"{who} at {t}", tier, 4, loc, t, pulse or f"p{t}", None)
    world = WorldSession(canon=canon, scene_owner_entity_id="nat", entry_point_location_ids=["lab"], playable_entity_ids=["nat"])
    # tick 59: the player acts in the lab; Corwin and Hartley write there -> engaged, the scene's draw
    world.recent_memories = [mem(1, "farmer", 30, "far"), mem(2, "nat", 59, "lab"), mem(3, "corwin", 59, "lab"), mem(4, "hartley", 59, "lab"), mem(5, "lab", 59, "lab", tier="AMBIENT")]
    canon.world_time = 60
    assert set(pravrtti._smrti_exploration_pool(world)) == {"corwin", "hartley"}
    assert set(pravrtti._smrti_exploration_pool(world)) == {"corwin", "hartley"}  # the same tick re-asked answers the same
    # tick 60: Corwin's activation; its place tuple is filed as the player's (witnessed) -- NOT a player pulse
    world.recent_memories += [mem(6, "corwin", 60, "lab", "Corwin read the ledger."), mem(7, "lab", 60, "lab", "The man read the ledger.", "AMBIENT", "pc60"),
                              mem(8, "nat", 60, "lab", "The man read the ledger.", "PRIMARY", "pc60")]
    # tick 61: the player's next own pulse at the lab -> the scene WAITS; the rest of the world has the draw
    world.recent_memories += [mem(9, "nat", 61, "lab"), mem(10, "corwin", 61, "lab"), mem(11, "lab", 61, "lab", tier="AMBIENT")]
    canon.world_time = 62
    pool = pravrtti._smrti_exploration_pool(world)
    assert "corwin" not in pool and "hartley" not in pool
    # the player's very next pulse (no tick was spent: nobody else acted) -> the scene's draw again: player pulses count, not ticks
    world.recent_memories += [mem(12, "nat", 62, "lab"), mem(13, "corwin", 62, "lab"), mem(14, "lab", 62, "lab", tier="AMBIENT")]
    canon.world_time = 63
    assert pravrtti._smrti_exploration_pool(world) == ["corwin"]
    # and had the witnessed tuple counted as a player pulse, "engaged" at 60 would have been Corwin alone by his own act
    world.scene_draws.clear()
    world.recent_memories = world.recent_memories[:8]
    canon.world_time = 61
    assert set(pravrtti._smrti_exploration_pool(world)) == {"corwin", "hartley"}  # still those who wrote beside the player's OWN 59 pulse


def test_the_world_stamp_can_never_stop_the_server_starting(tmp_path, monkeypatch) -> None:
    """_name_of_the_default_world runs inside lifespan() and exists only
    to put a name on sakshi's lines. A hand-corrupted worlds.json made
    get_worlds_registry raise JSONDecodeError straight out of startup --
    the instrument breaking the engine it watches, which v5 §44
    forbids. An unnamed world is a worse answer than a named one and a
    far better one than no server."""
    import smritilila.main as main

    bad = tmp_path / "worlds.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setenv("SMRITILILA_WORLDS_CONFIG", str(bad))
    monkeypatch.setenv("SMRITILILA_ACTIVE_WORLD_MARKER", str(tmp_path / "no-such-marker"))
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_vadhu")
    monkeypatch.setattr(main.app.state, "worlds_registry", None, raising=False)

    assert adhisthana._name_of_the_default_world() is None
    adhisthana._recall_active_world()  # must not raise -- this is lifespan()


def test_a_player_chooses_whether_the_world_may_act_for_them(client: TestClient, fake_store) -> None:
    """v5 §37 (Marc, 2026-09-23). Off by default: never voiced by another's
    turn, never moved, and offline suspended -- not drawn. On: movable, and
    offline voiced -- waiting since v4.58, so not drawn until touched. Persisted."""
    world = _seed_square_world(client)
    world.canon.entities["mara"] = Entity("mara", ["Mara"], 0, "genesis")
    world.recent_memories.append(Memory("m-mara", "mara", frozenset(), "I crossed the square.", "PRIMARY", 5, "square", 0, "p0", None))
    assert client.get("/players").json() == [{"entity_id": "player", "label": "the player", "autonomous": False, "online": False, "waiting": False}]
    assert world.unvoiced() == frozenset({"player"}) and world.movable() == frozenset()
    assert "player" not in pravrtti._smrti_exploration_pool(world)  # offline, off: suspended
    r = client.post("/players/player/autonomy", json={"allow": True})
    assert r.status_code == 200 and r.json()["autonomous"] is True and fake_store._autonomy == ["player"]
    assert world.movable() == frozenset({"player"})
    # offline and allowing it: voiced, and waiting -- not drawn until someone's pulse writes them one (v4.58)
    assert world.unvoiced() == frozenset() and world.waiting == {"player"} and "player" not in pravrtti._smrti_exploration_pool(world)
    # back online: movable still, but not voiced by another's turn and not drawn
    assert client.post("/players/player/presence", json={"online": True}).json()["online"] is True
    assert world.unvoiced() == frozenset({"player"}) and "player" not in pravrtti._smrti_exploration_pool(world)
    assert client.post("/players/player/presence", json={"online": False}).json()["online"] is False
    assert client.post("/players/nobody/autonomy", json={"allow": True}).status_code == 404


def test_a_heartbeat_keeps_a_player_online_but_never_brings_one_back(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                                                     fake_store) -> None:
    """v5 §37 (Marc, 2026-10-04): a page on screen and touched sends a heartbeat every few minutes, so a player can
    wait online without acting. It refreshes an online player's idle clock; it never undoes a log-off; a guest's page
    keeps only their own character online."""
    import time as _time
    world = _seed_square_world(client)
    world.presence["player"], world.last_seen["player"] = "online", _time.monotonic() - (adhisthana.PLAYER_IDLE_SECONDS - 60)
    assert client.post("/players/player/heartbeat").status_code == 204
    assert _time.monotonic() - world.last_seen["player"] < 5 and world.online("player")   # a fresh half hour
    assert client.post("/players/player/presence", json={"online": False}).json()["online"] is False
    seen = world.last_seen["player"]
    assert client.post("/players/player/heartbeat").status_code == 204
    assert "player" in world.logged_off and world.presence["player"] == "offline"          # a log-off holds
    assert world.last_seen["player"] == seen and "player" not in world.returning           # no return narrated
    assert client.post("/players/nobody/heartbeat").status_code == 404
    monkeypatch.setattr(dvara, "_guest", lambda request: "guest@example.com")
    monkeypatch.setattr(dvara, "_plays", lambda w, guest, pid: False)
    monkeypatch.setattr(dvara, "_god_of_active", lambda guest: False)
    assert client.post("/players/player/heartbeat").status_code == 403                     # not their character


def test_a_being_the_world_holds_is_made_playable_and_waits(client: TestClient, fake_store) -> None:
    """Marc, 2026-09-28: Elmo, played by an agent from his own page. Made playable, a being is logged off and
    waiting (v4.58) -- voiced, no turn of its own until touched, and no round waits on it -- until its first presence."""
    world = _seed_square_world(client)
    world.canon.entities["mara"] = Entity("mara", ["Mara Fenn"], 0, "genesis")
    world.recent_memories.append(Memory("m-mara", "mara", frozenset(), "I crossed the square.", "PRIMARY", 5, "square", 0, "p0", None))
    r = client.post("/world/playable/mara")
    assert r.status_code == 200 and r.json() == {"entity_id": "mara", "label": "Mara Fenn", "autonomous": True, "online": False, "waiting": True}
    assert "mara" in world.playable_entity_ids and fake_store._autonomy == ["mara"] and "mara" in fake_store._logged_off
    assert "mara" not in world.unvoiced() and "mara" not in pravrtti._smrti_exploration_pool(world)  # voiced; waiting, not drawn
    assert "mara" not in sangama._round_members(world, sangama._reach_key(world.kosha(), "square", world.canon.world_time + 1))
    assert client.post("/players/mara/presence", json={"online": True}).json()["online"] is True
    assert "mara" in world.unvoiced()                                                      # played now: her own
    assert client.post("/world/playable/square").status_code == 404                       # a scene is no one to play
    assert client.post("/world/playable/mara").status_code == 200 and world.playable_entity_ids.count("mara") == 1


def test_the_npcs_are_the_beings_no_one_plays_richest_in_memory_first_and_bounded(client: TestClient, fake_store) -> None:
    """Marc, 2026-10-06: the overseer offers "the 4 highest memories NPCs" as bots. An NPC is a being that is neither playable nor a
    place; ranked by the memories it holds, at most `limit`, and only beings that hold some. Owner only."""
    world = _seed_square_world(client)
    for i, (name, n) in enumerate([("Mara Fenn", 3), ("Old Tam", 5), ("Quill", 1), ("Dolan", 4), ("Ness", 2), ("Silent", 0)]):
        eid = f"npc{i}"
        world.canon.entities[eid] = Entity(eid, [name], 0, "genesis")
        for j in range(n):
            world.recent_memories.append(Memory(f"m-{eid}-{j}", eid, frozenset(), "something", "PRIMARY", 5, "square", j, f"p{j}", None))
    world._samskara = None
    got = client.get("/world/npcs").json()
    assert [(g["name"], g["memories"]) for g in got] == [("Old Tam", 5), ("Dolan", 4), ("Mara Fenn", 3), ("Ness", 2)]     # four, richest first
    assert "Silent" not in [g["name"] for g in client.get("/world/npcs?limit=20").json()]                                # holds none: not offered
    assert "player" not in [g["id"] for g in client.get("/world/npcs?limit=20").json()]                                  # the player's own is no NPC
    assert "square" not in [g["id"] for g in client.get("/world/npcs?limit=20").json()]                                  # nor is a place
    assert "sample" not in client.get("/world/npcs").json()[0]                                                           # what each holds only when asked for
    shown = client.get("/world/npcs?sample=true").json()[0]
    assert shown["name"] == "Old Tam" and len(shown["sample"]) == 3 and all(m == "something" for m in shown["sample"])    # three, for whoever judges
    client.post("/world/playable/npc1")
    assert "Old Tam" not in [g["name"] for g in client.get("/world/npcs").json()]                                        # played now: no longer an NPC
    guest = {"Cf-Access-Authenticated-User-Email": "g@example.com", "Cf-Ray": "x"}
    assert client.get("/world/npcs", headers=guest).status_code == 403


def test_the_npcs_most_spoken_of_are_ranked_beside_those_richest_in_memory(client: TestClient, fake_store) -> None:
    """Marc, 2026-10-06: "the dragon, the head evil cleric" matter though they are met seldom: the beings other beings hold memories ABOUT.
    Counted by distinct perceivers, never by a place's rows (which list who is present) nor a being's thought of itself."""
    world = _seed_square_world(client)
    for eid, name in [("dragon", "the red dragon"), ("tam", "Old Tam"), ("mara", "Mara"), ("quiet", "Quiet")]:
        world.canon.entities[eid] = Entity(eid, [name], 0, "genesis")
    def mem(i, owner, about, tier="PRIMARY"):
        world.recent_memories.append(Memory(f"m{i}", owner, frozenset(about), "heard tell", tier, 5, "square", i, f"p{i}", None))
    for i, owner in enumerate(["tam", "mara", "player"]):
        mem(i, owner, {"dragon"})                                                       # three beings hold a thought of the dragon
    mem(10, "mara", {"dragon"})                                                         # a second memory of the same perceiver adds nothing
    mem(11, "dragon", {"dragon"})                                                       # nor does its thought of itself
    mem(12, "square", {"quiet", "dragon"}, tier="AMBIENT")                              # a place's row lists who is present: no one's thought
    mem(13, "square", {"quiet", "dragon"})                                              # a place is no perceiver, whatever the tier of its row
    mem(14, "mara", {"quiet", "dragon"}, tier="AMBIENT")                                # nor is an ambient row a being's thought, whoever holds it
    for j in range(5):
        mem(20 + j, "tam", set())                                                       # Old Tam holds the most memories of his own
    world._samskara = None
    got = {g["name"]: g for g in client.get("/world/npcs?limit=1").json()}
    assert set(got) == {"Old Tam", "the red dragon"}                                    # one from each ranking, no being twice
    assert got["the red dragon"]["known_by"] == 3 and got["the red dragon"]["memories"] == 1 and got["Old Tam"]["known_by"] == 0
    assert "Quiet" not in client.get("/world/npcs?limit=20").json().__repr__()          # named only in a place's row: unknown to anyone


def test_the_cast_is_every_being_no_one_plays_by_name_with_the_setting_for_the_overseer_alone(client: TestClient, fake_store) -> None:
    """Marc, 2026-10-06: a model picks "the important" beings from the whole cast, for the overseer's bot list."""
    world = _seed_square_world(client)
    world.setting_description = "A fading empire. " * 100
    for eid, name, n in [("dragon", "the red dragon", 1), ("tam", "Old Tam", 3), ("quiet", "Quiet", 0)]:
        world.canon.entities[eid] = Entity(eid, [name], 0, "genesis")
        for j in range(n):
            world.recent_memories.append(Memory(f"m-{eid}-{j}", eid, frozenset(), "something", "PRIMARY", 5, "square", j, f"p{j}", None))
    world._samskara = None
    got = client.get("/world/cast").json()
    assert [(b["name"], b["memories"]) for b in got["beings"]] == [("Old Tam", 3), ("the red dragon", 1), ("Quiet", 0)]    # all of them, even one holding none
    assert "player" not in [b["id"] for b in got["beings"]] and "square" not in [b["id"] for b in got["beings"]]           # no player, no place
    assert got["setting"].startswith("A fading empire.") and len(got["setting"]) <= 900                                    # the setting, bounded
    guest = {"Cf-Access-Authenticated-User-Email": "g@example.com", "Cf-Ray": "x"}
    assert client.get("/world/cast", headers=guest).status_code == 403


def test_the_owner_appends_a_becoming_and_the_newest_under_its_key_holds(client: TestClient, fake_store) -> None:
    """A gravestone repair for the measure (Marc, 2026-09-28: Jakal's first expedition, never awarded): insert-only,
    the newest record under a key is what holds, as with the Narrator's own."""
    from smritilila.memory.record import Record
    world = _seed_square_world(client)
    world.canon.entities["mara"] = Entity("mara", ["Mara Fenn"], 0, "genesis")
    world.records.append(Record("r0", "clause", "mara", "Holds ten experience points.", 0, "p0", note="self:experience"))
    r = client.post("/world/becoming/mara", json={"key": "experience", "clause": "Holds seventy experience points.",
                                                  "why": "the first expedition"})
    assert r.status_code == 204
    held = [c for c in world.kosha().clauses_of("mara", world.canon.world_time + 1) if c[0] == "experience"]
    assert held == [("experience", "self", "Holds seventy experience points.")]
    assert len([x for x in world.records if x.entity_id == "mara"]) == 2                   # appended, the old one kept
    assert client.post("/world/becoming/mara", json={"key": "Bad Key", "clause": "x"}).status_code == 422
    assert client.post("/world/becoming/nobody", json={"key": "experience", "clause": "x"}).status_code == 422


def test_the_owner_appends_a_containment_record_checked_for_structure(client: TestClient, fake_store) -> None:
    """A gravestone repair for where things are (Marc, 2026-09-28: Turuko, in no scene): insert-only, checked for
    structure -- no cycle, never a player's body."""
    world = _seed_square_world(client)
    world.canon.entities["turuko"] = Entity("turuko", ["Turuko the Baklunish"], 0, "genesis")
    world.canon.entities["pack"] = Entity("pack", ["a pack"], 0, "genesis")
    r = client.post("/world/containment/turuko", json={"into": "square", "why": "stood in no scene"})
    assert r.status_code == 204 and world.kosha().scene_of("turuko", world.canon.world_time + 1) == "square"
    assert client.post("/world/containment/pack", json={"into": "turuko"}).status_code == 204
    assert client.post("/world/containment/turuko", json={"into": "pack"}).status_code == 422   # a cycle
    assert client.post("/world/containment/player", json={"into": "square"}).status_code == 422  # a player's body
    world.canon.entities["earth"] = Entity("earth", ["the world at large"], 0, "itihasa")
    assert client.post("/world/containment/earth", json={"into": "square"}).status_code == 422   # the earth (2026-10-03)


def test_a_logged_off_player_is_out_of_the_world_and_their_return_is_narrated(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Marc, 2026-09-24: "an offline non acting player should be entirely
    suspended from the game, not present at all" -- only an explicit log
    off, and not while the world acts for them. Out of every roster;
    persisted; back, their next pulse narrates the return."""
    from smritilila.smrti.pulse import SmrtiPulseResult
    world = _seed_square_world(client)
    world.canon.entities["wren"] = Entity("wren", ["Wren"], 0, "genesis")
    world.playable_entity_ids = ["player", "wren"]
    world.recent_memories.append(Memory("m-wren", "wren", frozenset(), "I sat by the well.", "PRIMARY", 5, "square", 4, "p4", None))
    assert client.get("/entities/player/scene").json()["present"] == ["wren"]
    # idle is not absent: only an explicit log off
    assert world.absent() == frozenset()
    client.post("/players/wren/presence", json={"online": False})
    assert world.absent() == frozenset({"wren"}) and fake_store._logged_off == ["wren"]
    assert client.get("/entities/player/scene").json()["present"] == []
    assert all("Wren" not in loc["players_here"] for loc in client.get("/world/locations").json())
    # the world acting for them: back in it (and their return is theirs to narrate)
    client.post("/players/wren/autonomy", json={"allow": True})
    assert world.absent() == frozenset() and "wren" in world.returning
    client.post("/players/wren/autonomy", json={"allow": False})
    world.returning.clear()
    assert world.absent() == frozenset({"wren"})
    # a reload keeps them out
    fake_store._loaded = world.canon
    fake_store._memories = list(world.recent_memories)
    fake_store._session_config = ("player", ["square"], ["player", "wren"], None)
    del client.app.state.world
    client.get("/players")
    world = client.app.state.world
    assert world.logged_off == {"wren"} and world.absent() == frozenset({"wren"})
    # the player's own pulse: out of the world no longer, the return narrated once
    seen: list[dict] = []

    def fake_pulse(canon, actor, action, t, pid, mems, clientx, embed, **kw):
        seen.append({"actor": actor, **kw})
        return SmrtiPulseResult(narration="n", memories_written=[], new_entities=[], suggested_next_action="",
                                context_tuples=0, bound_ids=[])
    roles = _fake_roles()
    roles.smrti_client = object()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(spanda, "run_smrti_pulse", fake_pulse)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    monkeypatch.setattr(main, "_select_for_activation", lambda *a, **k: None, raising=False)
    assert client.post("/pulses", json={"action_text": "looks up", "entity_id": "player"}).status_code == 200
    assert seen[-1]["absent"] == frozenset({"wren"}) and seen[-1]["returning"] is None
    assert client.post("/pulses", json={"action_text": "stands", "entity_id": "wren"}).status_code == 200
    assert seen[-1]["actor"] == "wren" and seen[-1]["returning"] == 4 and seen[-1]["absent"] == frozenset()
    assert world.logged_off == set() and fake_store._logged_off == [] and "wren" not in world.returning
    assert client.post("/pulses", json={"action_text": "sits", "entity_id": "wren"}).status_code == 200
    assert seen[-1]["returning"] is None  # narrated once


def test_a_join_asked_again_is_the_same_newcomer_and_a_borne_name_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """The Crossing, 2026-09-28: a join answered in 13 s was sent again twice, and three Nara Millwrights stood on
    the river road. The same join asked again is answered with the newcomer it made; a different newcomer who would
    bear a name someone in the world already bears is refused and not written, as an expansion refuses one."""
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    client.post("/genesis", json={"self_description": "I am Wren.", "setting": "A mill on a river.", "breadth": "here"})
    world = client.app.state.world
    ford = next(e.id for e in world.canon.entities.values() if e.labels[0] == "the mill ford")
    first = client.post("/world/join", json={"entry_location_id": ford, "self_description": "I am Hob, a tinker."})
    again = client.post("/world/join", json={"entry_location_id": ford, "self_description": "I am  Hob, a tinker. "})
    assert first.status_code == again.status_code == 200 and first.json() == again.json()
    assert sum(1 for p in world.playable_entity_ids if world.canon.entities[p].labels == ["Hob, a tinker"]) == 1
    other = client.post("/world/join", json={"entry_location_id": ford, "self_description": "I am Hob, the other one."})
    assert other.status_code == 409 and "already called Hob" in other.json()["detail"]
    assert sum(1 for p in world.playable_entity_ids if world.canon.entities[p].labels[0].startswith("Hob")) == 1


def test_a_join_folds_in_at_the_places_time_witnessed_and_seen(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """2026-09-25 review of the world builder against the pulse: a join
    swapped a copy of the world in (another turn committing while it awaited
    the store was lost), wrote the arrival as the newcomer's alone (players
    there never witnessed it), never marked the newcomer seen (logged off by
    their next request on a server up half an hour), and started their clock
    at the clock's start -- Marc: a newcomer joins at the current time of the
    place they start in."""
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    wren = client.post("/genesis", json={"self_description": "I am Wren.", "setting": "A mill on a river.", "breadth": "here"}).json()["player_entity_id"]
    world = client.app.state.world
    ford = next(e.id for e in world.canon.entities.values() if e.labels[0] == "the mill ford")
    world.world_data["clock"] = {"epoch": "1928-04-27T00:00", "start": 1500}
    world.recent_memories.append(Memory("m-wren", wren, frozenset(), "I waited at the ford a long while.", "PRIMARY", 5, ford,
                                        world.canon.world_time, "p-wren", None, began=1560, ended=1600))
    real_commit = fake_store.commit

    async def meanwhile(canon, memories, entities, touched_ids=(), records=()):
        if any(e.labels == ["Hob, a tinker"] for e in entities):  # another turn lands while the join awaits the store
            world.canon.entities["pedlar"] = Entity("pedlar", ["a pedlar"], world.canon.world_time, "smrti")
        return await real_commit(canon, memories, entities, touched_ids, records)
    monkeypatch.setattr(fake_store, "commit", meanwhile)
    resp = client.post("/world/join", json={"entry_location_id": ford, "self_description": "I am Hob, a tinker."})
    assert resp.status_code == 200, resp.text
    hob = resp.json()["player_entity_id"]
    assert "pedlar" in world.canon.entities and hob in world.canon.entities       # folded in, not swapped
    assert {m.began for m in world.recent_memories if m.entity_id == hob} == {1600}  # the ford's time, not 1500
    assert any(m.entity_id == wren and "tinker came down" in m.experience for m in world.recent_memories)  # Wren witnessed it
    # ... and SEES it, as it happens (meeting: Stonemarch, Jakal's memory held Dunstan's arrival, his page never showed it)
    assert any("tinker came down" in w["text"] for w in world.pending_witnessed.get(wren, []))
    assert not world.pending_witnessed.get(hob)  # the newcomer has their own arrival narration
    assert world.presence[hob] == "online" and hob not in world.logged_off


def test_an_npc_turn_keeps_what_committed_meanwhile_without_moving_the_clock(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """An NPC's commit replaced the live lists with its snapshot plus its own
    unless world_time had moved -- and an expansion or the world at large
    does not move it: their first tuples vanished from the running world."""
    import asyncio
    from smritilila.memory.memory import Memory
    from smritilila.smrti.pulse import SmrtiPulseResult
    _seed_square_world(client)
    world = client.app.state.world
    owner = world.scene_owner_entity_id
    expansion = Memory("m-ring", "hof", frozenset(), "A hall on the water.", "AMBIENT", 5, "hof", world.canon.world_time, "vistara:1", None)

    def during(canon, actor, action, world_time, pulse_id, mems, *a, **k):
        world.recent_memories.append(expansion)  # an expansion commits while this turn computes; the clock does not move
        return SmrtiPulseResult("She moves.", [Memory("m-act", actor, frozenset(), "I moved to the kettle.", "PRIMARY", 4, "square",
                                                      world_time, pulse_id, None)], [], "", 3, [actor])
    monkeypatch.setattr(spanda, "run_smrti_pulse", during)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    asyncio.run(spanda._run_smrti_activation_in_background(world, _fake_roles(_SpyNarrator()), owner, 1))
    ids = [m.id for m in world.recent_memories]
    assert "m-ring" in ids and ids[-1] == "m-act"


def test_a_newcomer_is_written_under_the_worlds_own_view(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """2026-09-25 review: an arrival, an expansion and the world at large were
    written without the world's own view -- its physics -- so a newcomer
    could be born contradicting what every later pulse is shown."""
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    client.post("/genesis", json={"self_description": "I am Wren.", "setting": "A mill on a river.", "breadth": "here"})
    world = client.app.state.world
    world.darsana = "No word, song or sign a person makes up touches the old things."
    ford = next(e.id for e in world.canon.entities.values() if e.labels[0] == "the mill ford")
    assert client.post("/world/join", json={"entry_location_id": ford, "self_description": "I am Hob."}).status_code == 200
    ask = next(u for u in reversed(roles.smrti_client.calls) if "SELF-DESCRIPTION OF THE NEWCOMER" in u)
    assert "THIS WORLD'S OWN VIEW -- it wins" in ask and "No word, song or sign" in ask and "THE DEFAULT VIEW" in ask


def test_an_arrival_naming_a_player_is_counted_as_a_pulse_would_be(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """The arrival has no channel to move, voice or write as a player; only
    its prose can name one -- counted (smrti.player_named), never blocked."""
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    wren = client.post("/genesis", json={"self_description": "I am Wren.", "setting": "A mill on a river.", "breadth": "here"}).json()["player_entity_id"]
    world = client.app.state.world
    ford = next(e.id for e in world.canon.entities.values() if e.labels[0] == "the mill ford")
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr("smritilila.smrti.pulse.log_event", lambda event, **f: events.append((event, f)))
    import smritilila.smrti.bija as bija
    real = bija.write_arrival

    def naming(*a, **k):
        out = real(*a, **k)
        out["place_tuple"] = "A tinker came down to the ford, and Wren looked up from the water."
        return out
    monkeypatch.setattr(bija, "write_arrival", naming)
    assert client.post("/world/join", json={"entry_location_id": ford, "self_description": "I am Hob."}).status_code == 200
    assert [(f["player"], f["part"]) for e, f in events if e == "smrti.player_named"] == [(wren, "Wren")]


def test_two_players_together_see_each_others_turns_as_they_happen(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Stonemarch, Marc and the first model playing together (2026-09-26): Dunstan
    offered his sword under the willow; Jakal's memory holds it (v5 §31: a
    witnessed copy is the player's own tuple), and Marc's page never showed
    it -- the witnessed feed was filled after the world's own turns only.
    Together, another player's turn reaches you as an NPC's does; the one
    who acted is never handed their own turn back. (Two players in the
    world together now act as a round, v5 §36; the feed is for one who is
    logged off and lets the world act for them -- still in the scene, and
    shown it when they come back.)"""
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.smrti.pulse import SmrtiPulseResult
    _seed_square_world(client)
    world = client.app.state.world
    world.canon.entities["mara"] = Entity("mara", ["Mara"], 0, "genesis")
    world.playable_entity_ids.append("mara")
    world.recent_memories.append(Memory("g-mara", "mara", frozenset(), "I stand in the town square too.", "PRIMARY", 3, "square", 0, "genesis", None))
    world.logged_off.add("player")
    world.autonomous.add("player")

    def mara_offers(canon, actor, action, world_time, pulse_id, mems, *a, **k):
        return SmrtiPulseResult("You offer your sword to the player.", [
            Memory("m-m", actor, frozenset(), "I offered my sword.", "PRIMARY", 4, "square", world_time, pulse_id, None),
            Memory("m-sq", "square", frozenset(), "A newcomer offered her sword in the square.", "AMBIENT", 5, "square", world_time, pulse_id, None),
        ], [], "", 3, [actor])

    monkeypatch.setattr(spanda, "run_smrti_pulse", mara_offers)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    monkeypatch.setattr(spanda, "advance_world_tick", lambda *a, **k: type("T", (), {"pulses_run": 0, "activated_entity_id": None})())
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _SmrtiClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    resp = client.post("/pulses", json={"action_text": "I offer my sword.", "entity_id": "mara"})
    assert resp.status_code == 200, resp.text
    assert [w["text"] for w in world.pending_witnessed.get("player", [])] == ["A newcomer offered her sword in the square."]
    assert not world.pending_witnessed.get("mara")
    r = client.get("/entities/player/witnessed").json()["witnessed"]
    assert [w["text"] for w in r] == ["A newcomer offered her sword in the square."]


def _round_world(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Two players online in the town square; run_smrti_pulse faked to tell the round."""
    import time as _time
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.smrti.pulse import SmrtiPulseResult
    _seed_square_world(client)
    world = client.app.state.world
    world.canon.entities["mara"] = Entity("mara", ["Mara"], 0, "genesis")
    world.playable_entity_ids.append("mara")
    world.recent_memories.append(Memory("g-mara", "mara", frozenset(), "I stand in the town square too.", "PRIMARY", 3, "square", 0, "genesis", None))
    for p in ("player", "mara"):
        world.presence[p], world.last_seen[p] = "online", _time.monotonic()
    calls = []

    def told(canon, actor, action, world_time, pulse_id, mems, *a, sangama=None, **k):
        calls.append({"actor": actor, "action": action, "sangama": sangama})
        if k.get("wait_until") is not None:
            calls[-1]["wait_until"] = k["wait_until"]
        if "default" in (k.get("sangama_sources") or {}).values():
            calls[-1]["sources"] = k["sangama_sources"]
        tell = {p: {"narration": f"You see {actor} act, as {p}.", "suggested_next_action": "Wait."} for p in (sangama or {})}
        return SmrtiPulseResult(f"You act, {actor}.", [
            Memory(f"m-{actor}", actor, frozenset(), "I acted.", "PRIMARY", 4, "square", world_time, pulse_id, None),
            *[Memory(f"m-{p}", p, frozenset(), "I was there.", "PRIMARY", 4, "square", world_time, pulse_id, None) for p in (sangama or {})],
            Memory("m-sq", "square", frozenset(), "Two people did things in the square.", "AMBIENT", 5, "square", world_time, pulse_id, None),
        ], [], "", 3, [actor], round_narrations=tell)

    monkeypatch.setattr(spanda, "run_smrti_pulse", told)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    monkeypatch.setattr(spanda, "advance_world_tick", lambda *a, **k: type("T", (), {"pulses_run": 0, "activated_entity_id": None})())
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _SmrtiClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    return world, calls


def test_players_together_act_as_one_round_and_each_is_told_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Saṅgama (Marc, 2026-09-27). Mara acts first: nothing is written, her
    page shows the round waiting on the other. When the other player acts,
    every player there has: one pulse -- Mara's the ACTOR, the first to act
    -- tells both. Each is told from where they stand; nobody is handed the
    other's turn as a witnessed line."""
    world, calls = _round_world(client, monkeypatch)
    r = client.post("/pulses", json={"action_text": "I offer my sword.", "entity_id": "mara"}).json()
    assert r["narrations"] == {} and r["round"]["open"] and r["round"]["acted"] and not calls
    assert {m["id"]: m["acted"] for m in r["round"]["members"]} == {"mara": True, "player": False}
    seen = client.get("/rounds/player").json()
    assert seen["open"] and not seen["acted"] and seen["told"] is None
    assert "offer" not in str(seen)  # who has acted, never what
    r = client.post("/pulses", json={"action_text": "I take the sword.", "entity_id": "player"}).json()
    assert calls == [{"actor": "mara", "action": "I offer my sword.", "sangama": {"player": "I take the sword."}}]
    assert r["narrations"] == {"player": "You see mara act, as player."}  # the one whose action closed it: told at once
    hers = client.get("/rounds/mara").json()
    assert hers["told"]["narrations"]["mara"] == "You act, mara." and not hers["open"]
    assert client.get("/rounds/mara").json()["told"] is None  # handed over once
    assert not world.pending_witnessed.get("mara") and not world.pending_witnessed.get("player")
    assert world.rounds == {}


def test_go_closes_a_round_and_editing_takes_an_action_back_out(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Go is for one who has acted; changing your action takes it out of the
    round until you act again. A player who has not acted when Go is pressed
    is in the round with no action."""
    world, calls = _round_world(client, monkeypatch)
    early = client.post("/rounds/player/go")  # before acting: the round as it stands, nothing run
    assert early.status_code == 200 and early.json()["narrations"] == {} and not calls
    client.post("/pulses", json={"action_text": "I offer my sword.", "entity_id": "mara"})
    edited = client.post("/rounds/mara/edit").json()
    assert not edited["acted"] and not edited["open"] and world.rounds == {}
    client.post("/pulses", json={"action_text": "I wait.", "entity_id": "player"})
    assert not client.post("/rounds/player/edit").json()["open"]  # the only action taken back: no round
    client.post("/pulses", json={"action_text": "I sheathe it again.", "entity_id": "mara"})
    assert not calls  # the other player has not acted: the round waits
    r = client.post("/rounds/mara/go").json()
    assert calls == [{"actor": "mara", "action": "I sheathe it again.", "sangama": {"player": ""}}]
    assert r["narrations"] == {"mara": "You act, mara."}
    assert client.get("/rounds/player").json()["told"]["narrations"]["player"] == "You see mara act, as player."


def test_a_player_alone_in_the_scene_acts_as_before(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    world, calls = _round_world(client, monkeypatch)
    world.presence["mara"] = "offline"  # the only other player there has logged off
    world.logged_off.add("mara")
    r = client.post("/pulses", json={"action_text": "I look around.", "entity_id": "player"}).json()
    assert calls == [{"actor": "player", "action": "I look around.", "sangama": None}] and r["round"] is None


def test_the_world_waits_in_a_scene_with_a_round_open(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """The clock does not move under a round's players: the scene's own
    turns are not drawn while it is open; elsewhere the world goes on."""
    world, _calls = _round_world(client, monkeypatch)
    from smritilila.graph.entity import Entity
    world.canon.entities["tom"] = Entity("tom", ["Tom"], 0, "genesis")
    monkeypatch.setattr(adhisthana.WorldSession, "kosha", lambda self: type("K", (), {
        "is_scene": staticmethod(lambda e: e == "square"), "scene_of": staticmethod(lambda e, t: {"tom": "square"}.get(e, "road")),
        "container_at": staticmethod(lambda e, t: None),
        "across": staticmethod(lambda s, t: [])})())  # a round's key is the scene, and what a boundary joins to it
    assert pravrtti._hold_for_rounds(world, ["tom", "far"]) == ["tom", "far"]
    world.rounds["square"] = {"actions": {"mara": "x"}, "sources": {}, "order": ["mara"], "running": False}
    assert pravrtti._hold_for_rounds(world, ["tom", "far"]) == ["far"]


def test_the_jump_question_carries_its_minutes_and_its_end_as_numbers() -> None:
    """2026-10-22: the bot tool's time guard compares where a jump ends with where the people are, so the question says it as numbers beside its
    words; a word for it is never parsed, and what a question does not have is left out."""
    from smritilila.sangama import confirm_of
    assert confirm_of({"kind": "long", "prompt": "You are about to jump ahead 3 h (to day 28, 10:09 at night). Proceed?", "minutes": 180, "to": 40209,
                       "from": 40029, "origin": "inn", "dest": "inn"}) == {"kind": "long", "prompt": "You are about to jump ahead 3 h (to day 28, 10:09 at night). Proceed?",
                                                                            "minutes": 180, "to": 40209}
    assert confirm_of({"kind": "locked", "prompt": "That scene is closed until later.", "until": 9}) == {"kind": "locked", "prompt": "That scene is closed until later."}
    assert confirm_of({"kind": "long", "prompt": "p", "minutes": "soon", "to": None}) == {"kind": "long", "prompt": "p"}


def test_a_long_jump_in_a_round_waits_on_everyone_who_acted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """v5 §36 in a round: the jump is everyone's who acted. It asks each of
    them; one yes waits on the other; the second yes writes it and tells
    both. One no ends the round, nothing written."""
    world, calls = _round_world(client, monkeypatch)
    told = spanda.run_smrti_pulse

    def held_first(*a, **k):
        r = told(*a, **k)
        r.pending = {"kind": "long", "minutes": 300, "prompt": "You are about to jump ahead 5 h. Proceed?", "to": None}
        return r

    monkeypatch.setattr(spanda, "run_smrti_pulse", held_first)
    client.post("/pulses", json={"action_text": "We march east.", "entity_id": "mara"})
    r = client.post("/pulses", json={"action_text": "I march with her.", "entity_id": "player"}).json()
    assert r["confirm"]["prompt"].startswith("You are about to jump") and r["narrations"] == {}
    assert r["confirm"]["minutes"] == 300 and "to" not in r["confirm"]                       # numbers beside the words (a missing end is left out, not null)
    assert client.get("/rounds/mara").json()["confirm"]["prompt"].startswith("You are about to jump")
    assert client.get("/rounds/mara").json()["confirm"]["minutes"] == 300
    r = client.post("/pulses/confirm", json={"entity_id": "player", "proceed": True}).json()
    assert r["narrations"] == {} and r["round"]["waiting_on"] == ["Mara"]
    assert client.get("/rounds/player").json()["confirm"] is None  # already agreed
    r = client.post("/pulses/confirm", json={"entity_id": "mara", "proceed": True}).json()
    assert r["narrations"] == {"mara": "You act, mara."}
    assert client.get("/rounds/player").json()["told"]["narrations"]["player"] == "You see mara act, as player."
    assert world.held == {} and world.rounds == {}
    # and one no ends it
    client.post("/pulses", json={"action_text": "We march east again.", "entity_id": "mara"})
    client.post("/pulses", json={"action_text": "I follow.", "entity_id": "player"})
    client.post("/pulses/confirm", json={"entity_id": "mara", "proceed": False})
    assert world.held == {} and world.rounds == {}
    assert client.get("/rounds/player").json()["declined_by"] == "Mara"


def test_a_round_waiting_on_a_yes_is_not_run_again(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """Held for its players' consent, a round is written already: Go or an
    edit leaves it alone until everyone has answered. A new action answers
    it no (v5 §36), for the whole round, and opens the next."""
    world, calls = _round_world(client, monkeypatch)
    told = spanda.run_smrti_pulse

    def held_first(*a, **k):
        r = told(*a, **k)
        r.pending = {"kind": "long", "minutes": 300, "prompt": "Proceed?", "to": None}
        return r

    monkeypatch.setattr(spanda, "run_smrti_pulse", held_first)
    client.post("/pulses", json={"action_text": "We march east.", "entity_id": "mara"})
    client.post("/pulses", json={"action_text": "I march with her.", "entity_id": "player"})
    held = client.post("/rounds/mara/go")  # answered with the round as it stands -- waiting on the yes -- not run again
    assert held.status_code == 200 and held.json()["narrations"] == {} and len(calls) == 1
    client.post("/rounds/mara/edit")
    assert len(calls) == 1 and world.rounds["square"]["actions"] == {"mara": "We march east.", "player": "I march with her."}
    client.post("/pulses", json={"action_text": "No, wait.", "entity_id": "mara"})
    assert world.held == {} and world.rounds["square"]["actions"] == {"mara": "No, wait."} and len(calls) == 1
    assert client.get("/rounds/player").json()["declined_by"] == "Mara"


def test_a_logged_off_player_who_acts_is_back_in_the_round(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """Acting is being here (v5 §37): a player who was logged off and acts is in
    the world again, and so in the round where they stand."""
    world, calls = _round_world(client, monkeypatch)
    world.logged_off.add("mara")
    r = client.post("/pulses", json={"action_text": "I come back to the square.", "entity_id": "mara"}).json()
    assert "mara" not in world.logged_off and r["round"]["open"] and not calls


def test_go_waits_twenty_seconds_after_anyone_yet_to_act_last_typed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Marc, 2026-09-27: Go only closes a round once those yet to act have
    not typed for twenty seconds -- at once if none of them is typing."""
    import time as _time
    world, calls = _round_world(client, monkeypatch)
    client.post("/pulses", json={"action_text": "I offer my sword.", "entity_id": "mara"})
    assert client.get("/rounds/mara").json()["go_in"] == 0  # nobody writing
    client.post("/rounds/player/typing")
    st = client.get("/rounds/mara").json()
    assert 19 < st["go_in"] <= 20
    r = client.post("/rounds/mara/go")
    assert r.status_code == 200 and 19 < r.json()["round"]["go_in"] <= 20 and not calls  # the countdown, nothing run
    client.post("/rounds/mara/typing")  # one who has acted typing holds nothing
    world.typed["player"] = _time.monotonic() - 21
    assert client.get("/rounds/mara").json()["go_in"] == 0
    assert client.post("/rounds/mara/go").status_code == 200 and len(calls) == 1


def test_a_player_who_has_not_acted_when_go_closes_the_round_does_their_first_suggestion(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Marc, 2026-09-27: the round goes on with them doing the page's first
    suggestion -- recorded as a default, not as their words -- and their
    page is told what was taken for them."""
    world, calls = _round_world(client, monkeypatch)
    client.post("/pulses", json={"action_text": "I offer my sword.", "entity_id": "mara"})
    client.post("/pulses", json={"action_text": "I take it.", "entity_id": "player"})  # both acted: the pulse offers "Wait." to the other player and none to its actor (Mara)
    assert world.first_suggestion == {"mara": "", "player": "Wait."}
    client.get("/rounds/mara")
    client.post("/pulses", json={"action_text": "I buckle it on.", "entity_id": "mara"})
    client.post("/rounds/mara/go")
    assert calls[-1]["sangama"] == {"player": "Wait."} and calls[-1]["sources"] == {"player": "default"}
    told = client.get("/rounds/player").json()["told"]
    assert told["round"] == {"defaulted": "Wait."} and told["narrations"]["player"]


def test_a_perceiver_whose_own_turn_gave_nothing_sleeps_until_touched(client: TestClient, fake_store) -> None:
    """Stonemarch: a bent dart, a barred door, a dead spider were each drawn to
    act on their own. A perceiver whose last own turn moved nothing, said
    nothing and gave nothing is not drawn again until another's pulse writes
    it a tuple."""
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    _seed_square_world(client)
    world = client.app.state.world
    world.canon.entities["dart"] = Entity("dart", ["a bent dart"], 0, "genesis")
    world.canon.entities["dart"].last_simulated_tick = 5
    world.recent_memories.append(Memory("d1", "dart", frozenset(), "I was thrown into the mud.", "PRIMARY", 5, "square", 3, "p3", None))
    world.recent_memories.append(Memory("d2", "dart", frozenset(), "I lay still in the mud.", "PRIMARY", 6, "square", 6, "p6", None, "none"))
    world.canon.world_time = 7
    assert "dart" not in pravrtti._smrti_exploration_pool(world)
    # handled again: a player's pulse writes it a tuple
    world.recent_memories.append(Memory("d3", "dart", frozenset(), "A hand pulled me from the mud.", "PRIMARY", 5, "square", 7, "p7", None, "none"))
    world.canon.world_time = 8
    assert "dart" in pravrtti._smrti_exploration_pool(world)


def test_a_round_is_those_within_reach_at_the_same_moment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """v5 §36 v4.31 (Marc, 2026-09-28: "It feels like it should stop where they
    meet"; "Within reach is exactly correct"). Stonemarch t155: Jakal (5:27)
    acted first, and the round ran on his clock after Dunstan had lived the
    track to 5:42; v4.23 answered by pulling the one behind forward. Now a
    player who is not level with the others acts alone -- nobody is pulled --
    until their span meets the others' present; players level and within
    reach act as one round."""
    world, calls = _round_world(client, monkeypatch)
    clocks = {"player": 327, "mara": 342}
    class _Kala:  # a world keeping a clock, as far as the round reads it; the rest of the reading is a no-op
        def now(self, p, w=None):
            return max(clocks.get(p, 0), w or 0)

        def __getattr__(self, name):
            return lambda *a, **k: None

    world.samskara().kala = _Kala()
    monkeypatch.setattr(spanda, "_kala_gate", lambda world, pool: (pool, {}))
    client.post("/pulses", json={"action_text": "I walk over to her.", "entity_id": "player"})  # behind: alone, at once
    assert calls[-1]["actor"] == "player" and not calls[-1]["sangama"] and "wait_until" not in calls[-1]
    n = len(calls)
    clocks["player"] = 342  # met: level, within reach
    world.samskara().kala = _Kala()  # the solo pulse's commit read the world again: the stand-in clock, again
    client.post("/pulses", json={"action_text": "I greet her.", "entity_id": "player"})
    assert len(calls) == n  # a round: waiting on her
    client.post("/pulses", json={"action_text": "I greet him.", "entity_id": "mara"})
    assert calls[-1]["actor"] == "player" and "mara" in calls[-1]["sangama"] and calls[-1]["wait_until"] == 342


def test_a_pulse_answer_carries_only_the_actors_own_telling(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """2026-09-28 (Teo's return, the Crossing): every world turn's narration,
    from anywhere, went to whoever acted next -- the page shows only the
    co-located ones, but a guest's browser received scenes their character
    has never met. Then (2026-09-30, item 6) the world's turns waited on the
    world for whichever player next stood near the doer -- decided when read,
    taken by the first, an NPC's own telling in a guest's browser. A world
    turn reaches a player only as what they witnessed, decided when written;
    an answer is the actor's own."""
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    world, calls = _round_world(client, monkeypatch)
    world.logged_off.add("mara")  # the player acts alone
    world.canon.entities["baker"] = Entity("baker", ["the baker"], 0, "genesis")
    world.canon.entities["barn"] = Entity("barn", ["the far barn"], 0, "genesis", avarana="scene")
    world.canon.entities["farmer"] = Entity("farmer", ["the farmer"], 0, "genesis")
    world.recent_memories += [Memory("g-baker", "baker", frozenset(), "I bake in the square.", "PRIMARY", 3, "square", 0, "genesis", None),
                              Memory("g-barn", "barn", frozenset(), "A far barn.", "AMBIENT", 5, "barn", 0, "genesis", None),
                              Memory("g-farmer", "farmer", frozenset(), "I fork hay in the far barn.", "PRIMARY", 3, "barn", 0, "genesis", None)]
    r = client.post("/pulses", json={"action_text": "I buy a loaf.", "entity_id": "player"}).json()
    assert set(r["narrations"]) == {"player"}
    assert not hasattr(world, "pending_activation_narrations") and not hasattr(world, "pending_activation_promoted_entity_ids")


def test_guests_play_only_their_own_characters_in_the_guest_world(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Marc, 2026-09-27: "allow only stonemarch access to anyone besides us".
    Cloudflare Access names each visitor; an owner (Marc's email, or this
    machine) may do anything; a guest only plays their own characters, only
    while Stonemarch is the active world, and administers nothing."""
    _seed_square_world(client)
    world = client.app.state.world
    reg = [type("W", (), {"name": "Stonemarch", "database_url": "g", "gods": (), "port": None, "description": None})(), type("W", (), {"name": "Ravenmoor", "database_url": "l", "gods": (), "port": None, "description": None})()]
    active = {"w": reg[0]}
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry", lambda: active["w"])
    monkeypatch.setattr(adhisthana, "get_worlds_registry", lambda: reg)
    ed = {"Cf-Access-Authenticated-User-Email": "guest1@example.com", "Cf-Ray": "x"}
    marc = {"Cf-Access-Authenticated-User-Email": "owner@example.com", "Cf-Ray": "x"}
    # administration is the owner's
    assert client.post("/world/clear", headers=ed).status_code == 403
    assert client.get("/world/view", headers=ed).status_code == 403
    assert client.post("/worlds/select", json={"name": "Ravenmoor"}, headers=ed).status_code == 403
    assert client.get("/world/view", headers=marc).status_code == 200
    assert client.get("/world/view").status_code == 200  # this machine
    # someone else's character is not theirs
    assert client.post("/pulses", json={"action_text": "I look.", "entity_id": "player"}, headers=ed).status_code == 403
    assert client.get("/rounds/player", headers=ed).status_code == 403
    assert client.get("/entities/player/witnessed", headers=ed).status_code == 403
    assert client.get("/world/exists", headers=ed).json()["playable_entities"] == {}
    assert client.get("/world", headers=ed).json()["playable_entity_ids"] == []
    assert [w["name"] for w in client.get("/worlds", headers=ed).json()] == ["Stonemarch"]
    assert client.get("/players", headers=ed).json() == []
    snap = client.get("/world", headers=ed).json()
    assert snap["entities"] == [] and snap["scene_owner_entity_id"] == ""  # nothing of the world they have not met
    # their own is
    world.played_by = {"player": "guest1@example.com"}
    assert [p["entity_id"] for p in client.get("/players", headers=ed).json()] == ["player"]
    snap = client.get("/world", headers=ed).json()
    assert "player" in {e["id"] for e in snap["entities"]} and snap["scene_owner_entity_id"] == "player"
    assert client.get("/rounds/player", headers=ed).status_code == 200
    assert list(client.get("/world/exists", headers=ed).json()["playable_entities"]) == ["player"]
    # through Cloudflare with no name: owns nothing
    assert client.get("/rounds/player", headers={"Cf-Ray": "x"}).status_code == 403
    # another world active: closed to guests, open to the owner
    active["w"] = reg[1]
    assert client.get("/world", headers=ed).status_code == 403
    assert client.get("/world", headers=marc).status_code == 200


def test_retiring_cuts_the_tether_and_leaves_the_being_in_the_world(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Marc, 2026-10-01: "death seems like a special case of retiring a character, freeing it to be autonomous only.
    The character who is dead doesn't disappear." Not reversible, so the name is typed back; only the player or an
    owner may; refused mid-round; the being stays where it stands, off every player set, with nothing written -- a
    guest's held back until a god releases it (2026-10-02); an owner's straight into the world."""
    world = _seed_square_world(client)
    world.canon.entities["wren"] = Entity("wren", ["Wren"], 0, "join")
    world.playable_entity_ids = ["player", "wren"]
    world.played_by = {"wren": "guest2@example.com"}
    world.autonomous, world.logged_off = {"wren"}, {"wren"}
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry", lambda: type("W", (), {"name": "Stonemarch", "gods": ()})())
    ed = {"Cf-Access-Authenticated-User-Email": "guest1@example.com", "Cf-Ray": "x"}
    mike = {"Cf-Access-Authenticated-User-Email": "guest2@example.com", "Cf-Ray": "x"}
    records, memories = len(world.records), len(world.recent_memories)
    assert client.post("/players/wren/retire", json={"confirm": "Wren"}, headers=ed).status_code == 403  # not theirs
    assert client.post("/players/wren/retire", json={"confirm": "wen"}, headers=mike).status_code == 422  # a slip
    world.rounds["square"] = {"actions": {"wren": "I wait."}, "sources": {}, "order": ["wren"], "running": False}
    assert client.post("/players/wren/retire", json={"confirm": "Wren"}, headers=mike).status_code == 409
    world.rounds.clear()
    alerts: list[tuple] = []
    monkeypatch.setattr(dvara, "_schedule_alert", lambda title, message, click=None: alerts.append((title, message)))
    assert client.post("/players/wren/retire", json={"confirm": " wren "}, headers=mike).status_code == 204
    assert world.playable_entity_ids == ["player"] and "wren" in world.canon.entities
    assert "wren" not in world.played_by | {p: 1 for p in world.autonomous | world.logged_off}
    assert "wren" not in world.movable() | world.unvoiced()
    # a guest's retirement is held back (Marc, 2026-10-02): out of the world until a god reviews and releases it
    assert world.held_back == {"wren"} and "wren" in world.absent() and fake_store._held_back == ["wren"]
    assert alerts == [("guest2@example.com retired Wren on Stonemarch", "Wren is held back until you review it (GET /world/held).")]
    assert "wren" not in pravrtti._smrti_exploration_pool(world)
    assert [h["entity_id"] for h in client.get("/world/held").json()] == ["wren"]
    assert client.get("/world/held", headers=mike).status_code == 403                  # a guest cannot review
    assert client.post("/world/held/wren/release").status_code == 204
    assert world.held_back == set() and "wren" not in world.absent() and fake_store._held_back == []
    assert fake_store.played_by == {} and fake_store._session_config[2] == ["player"]
    assert (len(world.records), len(world.recent_memories)) == (records, memories)  # the world's record untouched
    assert client.get("/players", headers=mike).json() == []
    assert client.post("/players/wren/retire", json={"confirm": "Wren"}).status_code == 404  # never taken up again
    # the last player retired: nobody is re-crowned
    assert client.post("/players/player/retire", json={"confirm": "the player"}).status_code == 204
    assert world.playable_entity_ids == [] and world.scene_owner_entity_id == ""
    assert fake_store._session_config[0] == "" and fake_store._session_config[2] == []


def test_a_guest_coming_online_sends_one_alert(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """Marc, 2026-10-01: an alert when a guest comes online -- their first
    request, or their first after PLAYER_IDLE_SECONDS away; never for the
    owner, this machine, or a visitor Cloudflare did not name."""
    _seed_square_world(client)
    sent = []
    monkeypatch.setattr(dvara, "_schedule_alert", lambda title, message, click=None: sent.append((title, click)))
    monkeypatch.setattr(dvara, "_guest_seen", {})
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry",
                        lambda: type("W", (), {"name": "Stonemarch", "gods": (), "url": "https://stonemarch.example", "guests": True})())
    ed = {"Cf-Access-Authenticated-User-Email": "guest1@example.com", "Cf-Ray": "x"}
    client.get("/worlds", headers=ed)
    client.get("/worlds", headers=ed)  # still here: no second alert
    assert sent == [("guest1@example.com is on Stonemarch", "https://stonemarch.example")]
    dvara._guest_seen["guest1@example.com"] -= adhisthana.PLAYER_IDLE_SECONDS + 1  # away the idle span, and back
    client.get("/worlds", headers=ed)
    assert len(sent) == 2
    client.get("/worlds")  # this machine
    client.get("/worlds", headers={"Cf-Access-Authenticated-User-Email": "owner@example.com", "Cf-Ray": "x"})  # the owner
    client.get("/worlds", headers={"Cf-Ray": "x"})  # through Cloudflare, unnamed
    assert len(sent) == 2


def test_an_alert_without_a_topic_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio as _asyncio
    monkeypatch.delenv("SMRITILILA_NTFY_TOPIC", raising=False)
    called = []
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: called.append(1))
    _asyncio.run(dvara._alert("t", "m"))
    assert called == []


def test_a_guest_who_joins_plays_the_character_they_made(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    _seed_square_world(client)
    world = client.app.state.world
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry", lambda: type("W", (), {"name": "Stonemarch", "gods": ()})())
    monkeypatch.setattr(bija_routes, "_join_locations", lambda w: [type("L", (), {"location_id": "square"})()])
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: None)

    async def made(req, w, roles):
        return schemas.JoinResponse(player_entity_id="wren", narration="You arrive.", narrations={"wren": "You arrive."})

    monkeypatch.setattr(bija_routes, "_v3_join", made)
    r = client.post("/world/join", json={"self_description": "I am Wren.", "entry_location_id": "square"},
                    headers={"Cf-Access-Authenticated-User-Email": "guest2@example.com", "Cf-Ray": "x"})
    assert r.status_code == 200 and world.played_by == {"wren": "guest2@example.com"}
    assert fake_store.played_by == {"wren": "guest2@example.com"} and "played_by" not in world.world_data  # the tether's own home


def test_every_newcomer_begins_as_the_world_says(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """Marc, 2026-09-27 (Stonemarch): a new player character begins at first
    level -- "a solid rule, not optional". The world's data names what every
    newcomer begins as; the join writes it as their own becomings, whatever
    their description claims and whichever model narrates."""
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    client.post("/genesis", json={"self_description": "I am Wren.", "setting": "A mill on a river.", "breadth": "here"})
    world = client.app.state.world
    ford = next(e.id for e in world.canon.entities.values() if e.labels[0] == "the mill ford")
    world.world_data["newcomer_clauses"] = {"level": "First level, whatever they claim.", "experience": "0 experience."}
    hob = client.post("/world/join", json={"entry_location_id": ford, "self_description": "I am Hob, a tinker and archmage."}).json()["player_entity_id"]
    own = {r.note: r.value for r in world.records if r.kind == "clause" and r.entity_id == hob}
    assert own == {"self:level": "First level, whatever they claim.", "self:experience": "0 experience."}
    assert [(k, x) for _b, k, x in world.kosha().view_of(hob, world.canon.world_time + 1)] == [
        ("experience", "0 experience."), ("level", "First level, whatever they claim.")]


def test_phala_reckons_what_the_company_finished_at_rest(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """Phala (Marc, 2026-09-27: experience never paid out in play). After a
    long pulse, one call reckons what the company lived since their last
    reckoning; what it writes is the company's own becomings -- never anyone
    outside the company, only keys the world allows -- and nothing is told."""
    import asyncio
    from smritilila.graph.entity import Entity
    from smritilila.memory.memory import Memory
    from smritilila.smrti.pulse import SmrtiPulseResult
    _seed_square_world(client)
    world = client.app.state.world
    world.canon.entities["elmo"] = Entity("elmo", ["Elmo"], 0, "genesis")
    world.canon.entities["stranger"] = Entity("stranger", ["a stranger"], 0, "genesis")
    world.recent_memories += [
        Memory("p1", "player", frozenset(), "I drove the gnolls from the yard.", "PRIMARY", 3, "square", 5, "pa", None),
        Memory("e1", "elmo", frozenset(), "I shot their leader in the shoulder.", "PRIMARY", 3, "square", 5, "pa", None),
        Memory("s1", "square", frozenset(), "The gnolls fled the yard.", "AMBIENT", 5, "square", 5, "pa", None)]
    world.canon.world_time = 6
    world.world_data["phala"] = {"ask": "Award experience as the view says.", "keys": ["experience", "level"]}
    shown = {}

    class _Reckoner:
        def complete_with_tool(self, system, user, **kw):
            shown["system"], shown["user"] = system, user
            return {"finished": ["the gnolls driven from the yard"], "becomings": [
                {"entity": "player", "key": "experience", "clause": "120 experience."},
                {"entity": "elmo", "key": "experience", "clause": "12,120 experience."},
                {"entity": "stranger", "key": "experience", "clause": "a windfall"},     # not of the company
                {"entity": "player", "key": "crown", "clause": "king of the moathouse"}]}  # not a key this world keeps

    # Zert fought in the yard and has gone on to the lane: the fruit follows the doer (Stonemarch, 2026-09-28:
    # Zert took part, had left by the rest, and was reckoned nothing while five others were)
    world.canon.entities["zert"] = Entity("zert", ["Zert"], 0, "genesis")
    world.canon.entities["lane"] = Entity("lane", ["the lane"], 0, "genesis", avarana="scene")
    world.recent_memories.append(Memory("z1", "zert", frozenset(), "I held the gate against them.", "PRIMARY", 3, "square", 5, "pa", None))
    world.records = [Record("k1", "kosha", "zert", "lane", 6, "pz", None)]
    monkeypatch.setattr(spanda, "_phala_client", lambda: _Reckoner())
    asyncio.run(spanda._reckon(world, ["player"], "square"))
    written = sorted((r.entity_id, r.note, r.value) for r in world.records if r.kind == "clause")
    assert written == [("elmo", "self:experience", "12,120 experience."), ("player", "self:experience", "120 experience.")]
    assert "I drove the gnolls" in shown["user"] and "Award experience" in shown["user"]
    assert "I held the gate" in shown["user"] and "zert" in shown["user"]            # the doer who left, reckoned
    # each doer carries their own mark, as a record of the world's own: reckoned once, whoever's rest calls it
    assert {(r.entity_id, r.value) for r in world.records if r.kind == "reckoned"} == {("player", "6"), ("elmo", "6"), ("zert", "6")}
    assert "phala_last" not in world.world_data
    world.recent_memories.append(Memory("e2", "elmo", frozenset(), "I bound the prisoner's wrists.", "PRIMARY", 3, "square", 7, "pb", None))
    world.recent_memories.append(Memory("p2", "player", frozenset(), "I kept watch in the square.", "PRIMARY", 3, "square", 7, "pb", None))
    world.canon.world_time = 8
    asyncio.run(spanda._reckon(world, ["player"], "square"))
    assert "bound the prisoner" in shown["user"] and "shot their leader" not in shown["user"]
    # due only in a world that asks, after a pulse longer than consent needs
    world.samskara().kala = type("K", (), {"confirm_minutes": 30, "__getattr__": lambda self, n: (lambda *a, **k: None)})()
    long = SmrtiPulseResult("", [], [], "", 0, [], elapsed=45)
    short = SmrtiPulseResult("", [], [], "", 0, [], elapsed=10)
    assert spanda._phala_due(world, long) and not spanda._phala_due(world, short)
    world.world_data.pop("phala")
    assert not spanda._phala_due(world, long)


def test_an_action_sent_twice_with_one_id_runs_once(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """Marc, 2026-09-27 ("Load failed", "Failed to fetch"): the page retries
    a send that failed in transit, with the same id. An action already
    received is never run twice -- the retry gets its answer."""
    world, calls = _round_world(client, monkeypatch)
    world.logged_off.add("mara")  # alone: a plain pulse
    body = {"action_text": "I look around.", "entity_id": "player", "client_id": "abc-1"}
    first = client.post("/pulses", json=body).json()
    again = client.post("/pulses", json=body).json()
    assert len(calls) == 1 and again["narrations"] == first["narrations"]
    client.post("/pulses", json={**body, "client_id": "abc-2"})  # a new action, a new id: runs
    assert len(calls) == 2


def test_an_action_that_failed_may_be_sent_again(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    world, calls = _round_world(client, monkeypatch)
    world.logged_off.add("mara")
    told = spanda.run_smrti_pulse
    state = {"fail": True}

    def flaky(*a, **k):
        if state.pop("fail", False):
            raise RuntimeError("the model call broke")
        return told(*a, **k)

    monkeypatch.setattr(spanda, "run_smrti_pulse", flaky)
    body = {"action_text": "I look around.", "entity_id": "player", "client_id": "abc-3"}
    with pytest.raises(RuntimeError):
        client.post("/pulses", json=body)
    assert client.post("/pulses", json=body).status_code == 200 and len(calls) == 1


def test_go_that_lands_after_the_round_was_written_is_not_an_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Marc, 2026-09-29: the page polls, so its Go can land after the other
    player's action already closed and wrote the round -- it answered 409.
    Now it is answered with the round as it stands; nothing is run again,
    and the telling is still waiting to be collected."""
    world, calls = _round_world(client, monkeypatch)
    client.post("/pulses", json={"action_text": "I offer my sword.", "entity_id": "mara"})
    client.post("/pulses", json={"action_text": "I take the sword.", "entity_id": "player"})  # the last in: written at once
    assert len(calls) == 1 and world.rounds == {}
    late = client.post("/rounds/mara/go")
    assert late.status_code == 200 and late.json()["narrations"] == {} and len(calls) == 1
    assert client.get("/rounds/mara").json()["told"]["narrations"]["mara"] == "You act, mara."  # still hers to collect


# -- 2026-09-30: the bugs found reading the tree ---------------------------


def test_a_world_is_loaded_once_however_many_ask_at_once(monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """The page asks several things at once. Every request that found no
    world loaded used to load it whole -- 55 double loads in 143 runs, the
    last to finish kept, a write meanwhile on another lost from memory. One
    request loads; the rest wait for it and take the same world."""
    import asyncio
    loads = []
    canon = Maya(world_time=3, entities={"player": Entity("player", ["the player"], 0, "genesis")})

    async def slow_load():
        loads.append(1)
        await asyncio.sleep(0.05)
        return canon

    fake_store.load = slow_load

    async def at_once():
        monkeypatch.setattr(adhisthana, "_world_load_lock", asyncio.Lock(), raising=False)  # this loop's own
        return await asyncio.gather(adhisthana.get_world(), adhisthana.get_world(), adhisthana.get_world())

    worlds = asyncio.run(at_once())
    assert len(loads) == 1
    assert worlds[0] is worlds[1] is worlds[2] is app.state.world


def test_a_second_genesis_while_one_is_seeding_is_refused(monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """A seed takes minutes; the tunnel gives up on a request at 100s and the
    page offered Build again while the first still seeded. Each request with
    no world loaded checked its own empty session, so both would have planted
    into one world. The second is refused while the first seeds."""
    import asyncio
    from fastapi import HTTPException
    seeded = []

    async def seeding(req, world, roles):
        seeded.append(world)
        await asyncio.sleep(0.05)
        return schemas.GenesisResponse(player_entity_id="", narration="", narrations={})

    monkeypatch.setattr(bija_routes, "_v3_genesis", seeding)
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: None)
    req = schemas.GenesisRequest(self_description="A miller's daughter at the ford.")

    async def twice():
        monkeypatch.setattr(bija_routes, "_genesis_lock", asyncio.Lock(), raising=False)  # this loop's own
        return await asyncio.gather(bija_routes.genesis(req), bija_routes.genesis(req), return_exceptions=True)

    first, second = asyncio.run(twice())
    assert len(seeded) == 1 and not isinstance(first, Exception)
    assert isinstance(second, HTTPException) and second.status_code == 409


def test_the_view_is_the_worlds_own_and_set_whole(client: TestClient, fake_store) -> None:
    """World data is the world's own view, authored (v5 §7). Who plays whom is the
    tether's (outside the world, beside autonomy); a reckoning is the world's
    own history (records). A view put whole touches neither."""
    world = _seed_square_world(client)
    world.played_by = {"player": "guest@example.com"}
    world.world_data = {"clock": {"start": 60}}
    client.put("/world/view", json={"darsana": None, "becomings": {}, "world_data": {"clock": {"start": 90}}})
    assert world.world_data == {"clock": {"start": 90}} and fake_store._view[2] == {"clock": {"start": 90}}
    assert world.played_by == {"player": "guest@example.com"}


def test_a_reckoning_is_kept_when_the_view_is_set_during_it(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """A reckoning is the world's own history now (a record per doer), so a
    view set during the reckoner's call cannot take it."""
    import asyncio
    world = _seed_square_world(client)
    world.recent_memories += [
        Memory("p1", "player", frozenset(), "I drove the gnolls from the yard.", "PRIMARY", 3, "square", 5, "pa", None),
        Memory("s1", "square", frozenset(), "The gnolls fled the yard.", "AMBIENT", 5, "square", 5, "pa", None)]
    world.canon.world_time = 6
    world.world_data["phala"] = {"ask": "Award experience as the view says.", "keys": ["experience"]}

    class _Reckoner:
        def complete_with_tool(self, system, user, **kw):
            world.world_data = {"phala": dict(world.world_data["phala"]), "clock": {"start": 60}}  # the owner sets the view
            return {"finished": [], "becomings": []}

    monkeypatch.setattr(spanda, "_phala_client", lambda: _Reckoner())
    asyncio.run(spanda._reckon(world, ["player"], "square"))
    assert {(r.entity_id, r.value) for r in world.records if r.kind == "reckoned"} == {("player", "6")}
    assert "phala_last" not in world.world_data


def test_a_world_behind_the_migration_head_is_backed_up_then_brought_to_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only /worlds/create migrated, so a world made before a migration stayed
    behind it until a write needed it (a sandhi record, refused by an older
    records_kind_check). A world is brought to head when it is loaded --
    backed up first, the chain holding destructive steps; at head, untouched."""
    import asyncio
    import asyncpg
    from alembic.script import ScriptDirectory
    head = ScriptDirectory.from_config(adhisthana._alembic_config()).get_current_head()
    url = "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_old"
    _install_fake_registry(monkeypatch, [("Old World", url)])
    done: list = []
    at = {"rev": "b9c0d1e2f3a4"}

    class _Conn:
        async def fetchval(self, sql):
            return at["rev"]

        async def close(self):
            pass

    async def connect(dsn):
        assert dsn == "postgresql://smritilila:smritilila@localhost/smritilila_old"
        return _Conn()

    monkeypatch.setattr(asyncpg, "connect", connect)
    monkeypatch.setattr(adhisthana, "backup_world", lambda entry, reason: done.append(("backup", entry.name, reason)) or "old.dump")
    monkeypatch.setattr(adhisthana, "_migrate_database", lambda u: done.append(("migrate", u)))
    asyncio.run(adhisthana._bring_to_head(url))
    assert done == [("backup", "Old World", "pre-migrate"), ("migrate", url)]
    done.clear()
    at["rev"] = head
    asyncio.run(adhisthana._bring_to_head(url))
    assert done == []


def test_a_player_is_movable_only_while_away_and_allowing_it(client: TestClient) -> None:
    """v5 §37: another's turn may move a player who lets the world act for them
    WHILE AWAY. movable() counted whoever had allowed it, online or not --
    both Stonemarch players had, so each was always movable by the other's solo
    turn: 8 times while playing, once pulling Jakal out of Corl's cabin into
    Dunstan's past at the keep gate (2026-09-29 19:06)."""
    import time as _time
    world = _seed_square_world(client)
    world.autonomous.add("player")
    world.last_seen["player"] = _time.monotonic()
    assert "player" not in world.movable() and "player" in world.unvoiced()        # online: their own
    world.presence["player"] = "offline"
    assert "player" in world.movable() and "player" not in world.unvoiced()        # away, allowing it: carried


def test_a_page_is_shown_only_what_happened_in_its_players_present(client: TestClient) -> None:
    """Where a player stands says where they see (containment); whether they
    were there then is their own time (kala.apart). By where alone, Jakal's
    page showed him, on the knoll, a moment 17 hours before he came."""
    world = _seed_square_world(client)
    world.world_data = {"clock": {"calendar": "days", "start": 60}}
    world.playable_entity_ids = ["player", "other"]
    world.canon.entities["other"] = Entity("other", ["the other"], 0, "genesis")
    world.recent_memories.append(Memory("p0", "player", frozenset(), "I watched the square.", "PRIMARY", 3, "square", 1, "p1", None,
                                        None, 60, 100))
    world.recent_memories.append(Memory("o0", "other", frozenset(), "I crossed the square at dusk.", "PRIMARY", 3, "square", 2, "p2", None,
                                        None, 60, 500))                                  # lived on, far past the cart
    world._samskara = None
    row = Memory("r1", "square", frozenset({"player", "other"}), "A cart rattled across the square.", "AMBIENT", 5, "square", 3, "pw", None,
                 None, 98, 100)
    spanda._queue_witnessed(world, [row])
    assert [w["text"] for w in world.pending_witnessed.get("player", [])] == ["A cart rattled across the square."]
    assert not world.pending_witnessed.get("other")                                # its moment is in their past


# -- 2026-09-30: the earth, the tether, the world's own history of reckoning ------------------------------


def test_an_earth_in_a_world_that_keeps_no_age_takes_no_turn(fake_store) -> None:
    """Earth is the world at large (Marc, 2026-09-30): every world has one, at
    its root. A world that keeps an age (a bhuvana) has it chronicle; one that
    keeps none has it perceive by its first memory, and stay quiet -- the
    Crossing's earth must not start an age with nothing to chronicle."""
    from smritilila.smrti.genesis import ITIHASA
    canon = Maya(world_time=32, entities={
        "bree": Entity("bree", ["Bree"], 0, "genesis", avarana="scene"),
        "amaranth": Entity("amaranth", ["Amaranth"], 0, "genesis"),
        "age": Entity("age", ["the world at large"], 0, ITIHASA)})
    world = WorldSession(canon=canon, scene_owner_entity_id="amaranth", entry_point_location_ids=["bree"], playable_entity_ids=["amaranth"])
    world.recent_memories = [Memory("m1", "amaranth", frozenset(), "x", "PRIMARY", 4, "bree", 30, "p30", None),
                             Memory("m2", "age", frozenset(), "A ferry crossing, late autumn.", "PRIMARY", 3, "nowhere", 0, "genesis", None)]
    for t in (32, 33, 34):
        canon.world_time = t
        assert "age" not in pravrtti._smrti_exploration_pool(world)
    world.bhuvana = "THE YEAR. 3018."                                            # keeping an age: it chronicles
    canon.world_time = 32
    world._samskara = None
    assert pravrtti._smrti_exploration_pool(world) == ["age"]


def test_a_world_given_its_age_gives_it_to_its_earth(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """A world born quiet and given its age later (POST /world/bhuvana) gives
    it to the earth it already has -- one world at large, never two."""
    from smritilila.smrti.genesis import ITIHASA
    world = _seed_square_world(client)
    world.canon.entities["earth"] = Entity("earth", ["the world at large"], 0, ITIHASA)
    world.recent_memories.append(Memory("e0", "earth", frozenset(), "A town and its square.", "PRIMARY", 3, "nowhere", 0, "genesis", None))
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: type("R", (), {"smrti_client": None})())
    monkeypatch.setattr("smritilila.smrti.bija.write_bhuvana", lambda *a, **k: "THE YEAR. 1880, spring. THE REGIONS. The San Pedro.")
    r = client.post("/world/bhuvana", json={"setting": "A silver town."})
    assert r.status_code == 200, r.text
    assert [e for e in world.canon.entities.values() if e.origin == ITIHASA] == [world.canon.entities["earth"]]
    assert world.bhuvana.startswith("THE YEAR. 1880")
    assert any(m.entity_id == "earth" and "1880" in m.experience for m in world.recent_memories)


# -- 2026-09-30: gods per world, by Marc's invitation -----------------------------------------------------

MARC = {"Cf-Access-Authenticated-User-Email": "owner@example.com", "Cf-Ray": "x"}
AUTHOR = {"Cf-Access-Authenticated-User-Email": "author@example.com", "Cf-Ray": "x"}
STRANGER = {"Cf-Access-Authenticated-User-Email": "stranger@example.com", "Cf-Ray": "x"}


def _god_registry(monkeypatch, tmp_path):
    import json
    from smritilila.worlds_registry import load_worlds_registry
    cfg = tmp_path / "worlds.json"
    cfg.write_text(json.dumps({"worlds": [
        {"name": "Stonemarch", "database_url": "postgresql+asyncpg://x/stonemarch"},
        {"name": "Hollow Fen", "database_url": "postgresql+asyncpg://x/hollow_fen", "gods": ["author@example.com"], "port": 8200}]}),
        encoding="utf-8")
    monkeypatch.setenv("SMRITILILA_WORLDS_CONFIG", str(cfg))
    monkeypatch.setenv("SMRITILILA_INVITES_CONFIG", str(tmp_path / "invites.json"))
    monkeypatch.setattr(adhisthana, "get_worlds_registry", lambda: load_worlds_registry(cfg))
    return cfg


def test_the_registry_carries_each_worlds_gods_and_its_own_server(tmp_path) -> None:
    """Gods per world (Marc, 2026-09-30): Marc and Anu for every world, and
    whoever authors one with them; an author's world runs on its own port."""
    from smritilila.worlds_registry import add_world, load_worlds_registry
    cfg = tmp_path / "worlds.json"
    add_world("Stonemarch", "postgresql+asyncpg://x/g", path=cfg)
    add_world("Hollow Fen", "postgresql+asyncpg://x/h", path=cfg, gods=("author@example.com",), port=8200)
    g, h = load_worlds_registry(cfg)
    assert (g.gods, g.port) == ((), None) and (h.gods, h.port) == (("author@example.com",), 8200)


def test_only_marc_invites(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A world is authored only by a god's invitation, and for now only Marc
    invites: not a guest, not an author, not this machine."""
    from smritilila.worlds_registry import load_invites
    _god_registry(monkeypatch, tmp_path)
    assert client.post("/worlds/invite", json={"email": "new@example.com"}, headers=STRANGER).status_code == 403
    assert client.post("/worlds/invite", json={"email": "new@example.com"}, headers=AUTHOR).status_code == 403
    assert client.post("/worlds/invite", json={"email": "new@example.com"}).status_code == 403   # this machine
    assert client.post("/worlds/invite", json={"email": "New@Example.com"}, headers=MARC).status_code == 200
    assert load_invites() == ["new@example.com"]


def test_a_guest_does_not_make_a_world_invited_or_not(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path, fake_store) -> None:
    """Marc, 2026-10-05: "Block guests from world creation". Option 1 (2026-09-30) gave an invited author a world on a server
    of its own; no guest makes a world now, invited or not, and an invitation is not spent by a refusal. The owner still makes
    one. With the one switch (dvara.GUESTS_MAY_CREATE_WORLDS) back on, the invited-author path is as it was."""
    from smritilila import dvara
    from smritilila.worlds_registry import add_invite, load_invites, load_worlds_registry
    cfg = _god_registry(monkeypatch, tmp_path)
    _seed_square_world(client)
    before = client.app.state.world
    spawned, made, events = [], [], []
    monkeypatch.setattr(loka, "log_event", lambda event, **f: events.append((event, f)))

    async def make_db(name):
        made.append(name)
        return "tide_hall", "postgresql+asyncpg://x/tide_hall"
    monkeypatch.setattr(loka, "_make_world_database", make_db)
    monkeypatch.setattr(loka, "_spawn_world_server", lambda entry: spawned.append((entry.name, entry.port)))
    assert dvara.GUESTS_MAY_CREATE_WORLDS is False                                  # the default
    assert client.post("/worlds/create", json={"name": "Tide Hall"}, headers=STRANGER).status_code == 403    # uninvited
    add_invite("stranger@example.com")
    r = client.post("/worlds/create", json={"name": "Tide Hall"}, headers=STRANGER)                          # invited
    assert r.status_code == 403 and "only the owner makes a world" in r.text
    assert client.post("/worlds/create", json={"name": "Tide Hall"}, headers=AUTHOR).status_code == 403    # a world's own god is a guest too
    assert load_invites() == ["stranger@example.com"]                              # a refusal spends nothing
    refused = [f for e, f in events if e == "guest.refused"]
    assert [f["guest"] for f in refused] == ["stranger@example.com", "stranger@example.com", "author@example.com"]
    assert {f["reason"] for f in refused} == {"world_creation"} and {f["path"] for f in refused} == {"/worlds/create"}   # a refusal is seen
    assert made == [] and spawned == [] and not any(w.name == "Tide Hall" for w in load_worlds_registry(cfg))
    # the switch back on: the invited-author path, as it was
    monkeypatch.setattr(dvara, "GUESTS_MAY_CREATE_WORLDS", True)
    r = client.post("/worlds/create", json={"name": "Tide Hall"}, headers=STRANGER)
    assert r.status_code == 200, r.text
    entry = next(w for w in load_worlds_registry(cfg) if w.name == "Tide Hall")
    assert entry.gods == ("stranger@example.com",) and entry.port not in (None, 8198, 8199, 8200)
    assert r.json()["port"] == entry.port and spawned == [("Tide Hall", entry.port)]
    assert client.app.state.world is before                                        # the shared server shows what it showed
    assert load_invites() == []                                                    # spent
    assert client.post("/worlds/create", json={"name": "Second Hall"}, headers=STRANGER).status_code == 403
    # and the owner makes a world either way
    monkeypatch.setattr(dvara, "GUESTS_MAY_CREATE_WORLDS", False)
    assert client.post("/worlds/create", json={"name": "Owner Hall"}, headers=MARC).status_code == 200


def test_a_worlds_god_administers_it_and_nothing_else(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path, fake_store) -> None:
    """Within their own world an author does what an owner does -- its view,
    its seed, entering it though it is no guest world; never another world,
    and never switching what a shared server shows (only Marc and Anu)."""
    from smritilila.worlds_registry import load_worlds_registry
    cfg = _god_registry(monkeypatch, tmp_path)
    _seed_square_world(client)
    fen, grey = [w for w in load_worlds_registry(cfg) if w.name == "Hollow Fen"][0], [w for w in load_worlds_registry(cfg) if w.name == "Stonemarch"][0]
    active = {"w": fen}
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry", lambda: active["w"])
    assert client.get("/world/view", headers=AUTHOR).status_code == 200              # their world: theirs to shape
    assert client.get("/world", headers=AUTHOR).status_code == 200                   # though no guest world
    assert client.get("/world/view", headers=STRANGER).status_code == 403
    assert client.post("/worlds/select", json={"name": "Stonemarch"}, headers=AUTHOR).status_code == 403
    active["w"] = grey
    assert client.get("/world/view", headers=AUTHOR).status_code == 403              # not their world


# -- 2026-09-30, item 6: nothing left on the world for a later reader --------------------------------------


import asyncio  # noqa: E402 -- the item-6 tests run turns directly


def _activation_result(world, monkeypatch, narration="She moves."):
    from smritilila.memory.memory import Memory
    from smritilila.smrti.pulse import SmrtiPulseResult

    def turn(canon, actor, action, world_time, pulse_id, mems, *a, **k):
        return SmrtiPulseResult(narration, [Memory("m-act-" + actor, actor, frozenset(), "I moved.", "PRIMARY", 4, "square", world_time, pulse_id, None)],
                                [], "", 3, [actor])
    monkeypatch.setattr(spanda, "run_smrti_pulse", turn)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())


def test_a_worlds_writes_go_to_its_own_store_whatever_is_selected_meanwhile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """A turn begun in one world and committed after a switch wrote into the
    newly selected world's database: every write asked the process which
    store was current when it committed. The world carries its own."""
    assert asyncio.run(adhisthana._load_world()).store is fake_store  # a world loaded carries where it came from
    world = _seed_square_world(client)
    world.store = fake_store
    owner = world.scene_owner_entity_id
    _activation_result(world, monkeypatch)
    other = _install_fake_store(monkeypatch)  # /worlds/select, while the turn was out at the model
    n = len(fake_store.commit_calls)
    asyncio.run(spanda._run_smrti_activation_in_background(world, _fake_roles(_SpyNarrator()), owner, 1))
    assert len(fake_store.commit_calls) == n + 1 and other.commit_calls == []


def test_the_log_names_the_world_a_task_took_not_the_one_selected_since() -> None:
    """sakshi's world stamp was the process's: a turn finishing after a
    switch was logged under the new world. A task keeps the world it took."""
    import contextvars
    from smritilila import sakshi
    sakshi.set_world("Ravenmoor")
    seen = []

    def task():
        sakshi.bind_world("Stonemarch")
        sakshi.set_world("The Vadhu Crossing")  # a select, meanwhile
        seen.append(sakshi.current_world())
    contextvars.copy_context().run(task)
    assert seen == ["Stonemarch"] and sakshi.current_world() == "The Vadhu Crossing"
    sakshi.set_world(None)


def test_a_background_turn_leaves_nothing_for_a_later_reader(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    """Who saw a turn is decided once, when it is written (pending_witnessed);
    nothing waits on the world to be taken by whoever comes next."""
    _seed_square_world(client)
    world = client.app.state.world
    _activation_result(world, monkeypatch)
    asyncio.run(spanda._run_smrti_activation_in_background(world, _fake_roles(_SpyNarrator()), world.scene_owner_entity_id, 1))
    assert fake_store.commit_calls  # the turn ran
    assert not hasattr(world, "pending_activation_narrations") and not hasattr(world, "pending_activation_promoted_entity_ids")


def test_when_the_world_acts_for_a_player_what_was_offered_them_is_gone(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """A round goes on for a player who has not acted with their first
    suggestion -- one offered for where they stood when they last acted. Once
    the world has acted for them since, it belongs to another moment."""
    _seed_square_world(client)
    world = client.app.state.world
    owner = world.scene_owner_entity_id
    world.offered[owner], world.first_suggestion[owner] = {"Ask the ferryman about the toll."}, "Ask the ferryman about the toll."
    _activation_result(world, monkeypatch)
    asyncio.run(spanda._run_smrti_activation_in_background(world, _fake_roles(_SpyNarrator()), owner, 1))
    assert fake_store.commit_calls  # the turn ran
    assert owner not in world.offered and owner not in world.first_suggestion


# -- 2026-09-30: each world's one line, drawn from its root ------------------------------------------------


class _DescribeClient:
    def __init__(self, line="Late autumn on the Vadhu, where one inn and a ferry hold two quiet banks."):
        self.line, self.calls = line, []

    def complete(self, system, user, *, max_tokens=8192):
        self.calls.append((system, user))
        return self.line


def test_a_worlds_line_is_drawn_from_its_root_and_listed_with_it(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path, fake_store) -> None:
    """Marc: "each world should have a one line description drawn from its
    root and is displayed next to the world selector" -- the root is what the
    earth carries (the age, else the earth's first memory, the setting); the
    line is kept with the list of worlds, for those deciding to enter."""
    from smritilila.smrti.genesis import plant_chronicle
    from smritilila.worlds_registry import load_worlds_registry
    cfg = _god_registry(monkeypatch, tmp_path)
    world = _seed_square_world(client)
    earth, first, _ = plant_chronicle("A ferry crossing on the river Vadhu, late autumn. One inn on the east bank.", lambda t: None)
    world.canon.entities[earth.id] = earth
    world.recent_memories.append(first)
    world.name = "Stonemarch"
    fake = _DescribeClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: GenerativeRoles(smrti_client=fake))
    r = client.post("/world/describe")
    assert r.status_code == 200 and r.json()["description"] == fake.line
    assert "A ferry crossing on the river Vadhu" in fake.calls[0][1]                 # drawn from the earth's first memory
    assert next(w for w in load_worlds_registry(cfg) if w.name == "Stonemarch").description == fake.line
    listed = {w["name"]: w for w in client.get("/worlds").json()}
    assert listed["Stonemarch"]["description"] == fake.line and listed["Hollow Fen"]["description"] is None
    world.bhuvana = "THE YEAR\n\nIt is 576 in the Flanaess. THE REGIONS. Far places."
    client.post("/world/describe")
    assert fake.calls[-1][1].endswith("It is 576 in the Flanaess.")                   # the age, where the world keeps one


def test_a_line_that_fails_to_draw_keeps_the_one_it_had(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path, fake_store) -> None:
    from smritilila.worlds_registry import load_worlds_registry, set_world_description
    cfg = _god_registry(monkeypatch, tmp_path)
    set_world_description("Stonemarch", "It is 576 in the Flanaess.", path=cfg)
    world = _seed_square_world(client)
    world.name, world.setting_description = "Stonemarch", "Millhaven, in summer."
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: GenerativeRoles(smrti_client=_DescribeClient("x" * 300)))
    assert client.post("/world/describe").status_code == 409
    assert next(w for w in load_worlds_registry(cfg) if w.name == "Stonemarch").description == "It is 576 in the Flanaess."
    assert client.post("/world/describe", headers=STRANGER).status_code == 403       # a world's gods draw it


def test_the_seed_is_sown_on_the_earth_at_birth_and_a_gods_edit_appends_to_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """v5 §15 (v4.53): the seed lies at the root. Born, the world's view is on its
    earth; edited, only what changed is appended there; world data keeps what
    the gods gave at birth; a world loaded again reads its view from the earth."""
    from smritilila.smrti import prakriti
    roles = _fake_roles(_SpyNarrator())
    roles.smrti_client = _BijaClient()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: roles)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    client.post("/world/clear")
    assert client.post("/genesis", json={"self_description": "I am Wren, the miller's daughter.", "setting": "A mill on a river.",
                                         "breadth": "wide", "source": "the sagas"}).status_code == 200
    world = client.app.state.world
    earth = adhisthana._earth(world.canon)
    assert earth is not None and prakriti.sown(world.records, earth)
    birth = (world.darsana, world.becomings, world.world_data)
    assert prakriti.reap(world.records, earth) == birth and birth[2]["source"] == "the sagas"
    before = len(world.records)
    edited = {"darsana": "The dead keep the fords.", "becomings": {}, "world_data": {**birth[2], "source": "Njáls saga"}}
    assert client.put("/world/view", json=edited).status_code == 200
    added = world.records[before:]
    assert sorted((r.kind, r.note) for r in added) == [("clause", "within:darsana"), ("prakriti", "source")]
    assert all(r.entity_id == earth for r in added)
    assert fake_store._view[2]["source"] == "the sagas"            # the birth, kept as given
    assert client.get("/world/view").json()["world_data"]["source"] == "Njáls saga"
    assert adhisthana._reaped(world.canon, world.records, fake_store._view) == (
        "The dead keep the fords.", birth[1], {**birth[2], "source": "Njáls saga"})    # loaded again: from the earth


def test_a_pinned_server_shows_its_own_world_and_choosing_another_is_going_to_its_address(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Marc, 2026-10-01: "I don't want people accidentally switching a world someone is playing in." A server whose
    port a world holds is that world's own: it comes up on it, refuses to switch -- owners included -- and refuses to
    make a new world (which would switch it); /worlds names each world's own address, and guests see those open to them."""
    from smritilila.worlds_registry import WorldRegistryEntry
    registry = [WorldRegistryEntry("Stonemarch", "postgresql+asyncpg://x/smritilila_stonemarch", port=8198, url="https://stonemarch.example", guests=True),
                WorldRegistryEntry("FAFO", "postgresql+asyncpg://x/smritilila_fafo", port=8200, url="https://fafo.example", guests=True),
                WorldRegistryEntry("Ravenmoor", "postgresql+asyncpg://x/smritilila_ravenmoor", port=8201, url="https://ravenmoor.example"),
                WorldRegistryEntry("Crossing", "postgresql+asyncpg://x/smritilila_vadhu", port=8199, url="https://crossing.example",
                                   guests=("guest.a@example.com",)),
                WorldRegistryEntry("Shaped", "postgresql+asyncpg://x/smritilila_shaped", gods=("author@example.com",))]
    monkeypatch.setattr(adhisthana, "get_worlds_registry", lambda: registry)
    monkeypatch.setenv("SMRITILILA_DEV_PORT", "8200")
    monkeypatch.setattr(adhisthana, "_remember_active_world", lambda name: None)
    adhisthana._recall_active_world()
    assert main.app.state.active_database_url == "postgresql+asyncpg://x/smritilila_fafo"
    r = client.post("/worlds/select", json={"name": "Stonemarch"})
    assert r.status_code == 409 and "https://stonemarch.example" in r.json()["detail"]
    assert client.post("/worlds/create", json={"name": "Another"}).status_code == 409
    worlds = {w["name"]: w for w in client.get("/worlds").json()}
    assert worlds["FAFO"]["active"] and worlds["Stonemarch"]["url"] == "https://stonemarch.example"
    guest = {"Cf-Access-Authenticated-User-Email": "friend@example.com"}
    assert sorted(w["name"] for w in client.get("/worlds", headers=guest).json()) == ["FAFO", "Stonemarch"]  # not Ravenmoor
    # a world naming its guests is open to them alone; a world's own god sees it though it is closed
    guest_a = {"Cf-Access-Authenticated-User-Email": "Guest.A@example.com"}
    assert sorted(w["name"] for w in client.get("/worlds", headers=guest_a).json()) == ["Crossing", "FAFO", "Stonemarch"]
    author = {"Cf-Access-Authenticated-User-Email": "author@example.com"}
    assert "Shaped" in [w["name"] for w in client.get("/worlds", headers=author).json()]
    crossing = registry[3]
    assert dvara._open_to_guests(crossing, "guest.a@example.com") and not dvara._open_to_guests(crossing, "friend@example.com")
    assert not dvara._open_to_guests(crossing, "") and not dvara._open_to_guests(crossing)
    monkeypatch.setenv("SMRITILILA_DEV_PORT", "8299")  # a server no world holds: switched by its owner, as before
    assert adhisthana._pinned_world() is None


def test_the_page_is_told_whether_it_serves_an_owner(client: TestClient, tmp_path, monkeypatch) -> None:
    """Marc, 2026-10-02: the page asks /me, open to anyone, who it serves: the admin link and the new-world button are the owner's. (2026-10-06: and whether the builder is theirs.
    2026-10-15: the world pull-down is no longer one of the things /me decides -- a guest has it too.)"""
    monkeypatch.setenv("SMRITILILA_USERS_CONFIG", str(tmp_path / "users.json"))    # no registered users
    assert client.get("/me").json() == {"owner": True, "builder": True}
    guest = {"Cf-Access-Authenticated-User-Email": "guest2@example.com", "Cf-Ray": "x"}
    assert client.get("/me", headers=guest).json() == {"owner": False, "builder": False}


def test_the_page_offers_every_visitor_the_world_pull_down() -> None:
    """Marc, 2026-10-15: "restore the pull-down for guests". From 2026-10-02 the page hid the selector and its label for anyone /me called not an owner, so a
    guest could not choose among the worlds /worlds already listed to them. The picker may hide its parts when there are no worlds to offer, and for nothing else."""
    import re
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "src" / "smritilila" / "web" / "index.html").read_text(encoding="utf-8")
    start = html.index("async function refreshWorldPicker()")
    body = html[start:html.index("function worldAddress(", start)]
    assert not re.search(r"getElementById\(\"world-select(-label)?\"\)\.classList\.add\(\"hidden\"\)", body), "the pull-down is hidden for someone by name"
    assert "if (!me.owner)" not in body                                              # no branch of the picker is the owner's alone
    assert "pickerParts.forEach(el => el.classList.remove(\"hidden\"))" in body    # shown whenever the server lists a world


def test_every_online_player_is_listed_to_everyone_with_where_and_when(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Marc, 2026-10-16: "Show all online players to everyone" and "Remove logged off from the list". /world/sync (melana) lists each player who is
    online with the scene and minute they stand at -- to an owner and to a guest who plays none of them; a logged-off player is not listed, unless
    the page names it as the one it plays (`as`) and the caller may play it. Where and when only: nothing of what anyone remembers."""
    import time
    world = _seed_square_world(client)
    for pid, label in (("wren", "Wren"), ("ghost", "Ghost")):
        world.canon.entities[pid] = Entity(pid, [label], 0, "join")
    world.canon.entities["inn"] = Entity("inn", ["The Inn"], 0, "genesis", avarana="open")
    world.playable_entity_ids = ["player", "wren", "ghost"]
    world.played_by = {"ghost": "guest1@example.com"}
    world.recent_memories[1] = Memory("g-player", "player", frozenset(), "I stand in the town square.", "PRIMARY", 3, "square", 0, "genesis", None, began=100, ended=150)
    world.recent_memories += [Memory("g-wren", "wren", frozenset(), "I am at the inn.", "PRIMARY", 3, "inn", 0, "genesis", None, began=180, ended=240),
                              Memory("g-ghost", "ghost", frozenset(), "I stand in the town square.", "PRIMARY", 3, "square", 0, "genesis", None, began=100, ended=150)]
    for pid in ("player", "wren"):                      # online; the ghost is not (no presence has ever been seen)
        world.presence[pid], world.last_seen[pid] = "online", time.monotonic()
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry", lambda: type("W", (), {"name": "Stonemarch", "gods": ()})())
    mike = {"Cf-Access-Authenticated-User-Email": "guest2@example.com", "Cf-Ray": "x"}   # plays none of them
    ed = {"Cf-Access-Authenticated-User-Email": "guest1@example.com", "Cf-Ray": "x"}        # plays the ghost
    # a world with no clock says where, never when
    assert [(p["entity_id"], p["scene"], p["at"], p["at_text"]) for p in client.get("/world/sync").json()] == [("player", "Town Square", None, None), ("wren", "The Inn", None, None)]
    world.world_data["clock"] = {"calendar": "days", "start": 60}
    got = client.get("/world/sync").json()                                                       # this machine: an owner
    assert [(p["entity_id"], p["label"], p["scene_id"], p["scene"], p["at"]) for p in got] == [
        ("player", "the player", "square", "Town Square", 150), ("wren", "Wren", "inn", "The Inn", 240)]
    assert all(p["at_text"] for p in got) and "ghost" not in {p["entity_id"] for p in got}      # never seen online: not listed
    assert client.get("/world/sync", headers=mike).json() == got                                  # a guest sees every online player, as the owner does
    # the page's own character is always listed, online or not -- and only when the caller may play it
    assert [p["entity_id"] for p in client.get("/world/sync?as=ghost", headers=ed).json()] == ["player", "wren", "ghost"]
    assert [p["entity_id"] for p in client.get("/world/sync?as=ghost", headers=mike).json()] == ["player", "wren"]


def test_an_owner_may_ask_for_those_whose_page_is_closed_but_whose_time_is_still_their_own(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-10-22, for the bot tool's time guard: a person who closes the tab is offline, not yet logged off, and the world still stops at their present (v5 §37),
    yet the list of who is online no longer names them. `own` asks for them as well, with `online` false: not those logged off (frozen) or waiting (brought along),
    and never for a guest, who is shown only what the page has always shown."""
    import time
    world = _seed_square_world(client)
    for pid, label in (("wren", "Wren"), ("ghost", "Ghost"), ("sleeper", "Sleeper"), ("waiter", "Waiter")):
        world.canon.entities[pid] = Entity(pid, [label], 0, "join")
    world.playable_entity_ids = ["player", "wren", "ghost", "sleeper", "waiter"]
    world.played_by = {"ghost": "guest1@example.com"}
    world.world_data["clock"] = {"calendar": "days", "start": 60}
    world.recent_memories[1] = Memory("g-player", "player", frozenset(), "I am here.", "PRIMARY", 3, "square", 0, "genesis", None, began=100, ended=150)   # as the test above
    for pid, at in (("wren", 240), ("ghost", 150), ("sleeper", 300), ("waiter", 310)):
        world.recent_memories.append(Memory(f"g-{pid}", pid, frozenset(), "I am here.", "PRIMARY", 3, "square", 0, "genesis", None, began=at - 10, ended=at))
    world.presence["player"], world.last_seen["player"] = "online", time.monotonic()
    world.presence["wren"], world.last_seen["wren"] = "offline", time.monotonic()           # closed the tab a moment ago: offline, not logged off
    world.logged_off = {"ghost", "sleeper"}                                                 # by the button, or thirty minutes without acting
    world.waiting = {"waiter"}                                                              # offline and chose to wait: brought along, no bound
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry", lambda: type("W", (), {"name": "Stonemarch", "gods": ()})())
    mike = {"Cf-Access-Authenticated-User-Email": "guest2@example.com", "Cf-Ray": "x"}
    assert [p["entity_id"] for p in client.get("/world/sync").json()] == ["player"]            # as ever: only who is online
    got = client.get("/world/sync?own=1").json()
    assert [(p["entity_id"], p["online"], p["at"]) for p in got] == [("player", True, 150), ("wren", False, 240)]
    assert all(p["online"] for p in client.get("/world/sync").json())                          # the field is there for every row, true on the page's list
    assert [p["entity_id"] for p in client.get("/world/sync?own=1", headers=mike).json()] == ["player"]   # a guest is not shown who has gone offline


def test_the_page_shows_who_is_level_who_is_remote_and_offers_to_join_them() -> None:
    """Marc, 2026-10-16: the list above the text box -- logged-off players gone, yellow for a player in the same pulse (the same scene and minute),
    light violet for one remote in place or time, and choosing a remote one asks to advance and go there (Y/N); one who is behind you is only
    named as behind. Structure only: it was met in a browser (see the TODO entry); no JS runtime is available to the suite."""
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "src" / "smritilila" / "web" / "index.html").read_text(encoding="utf-8")
    start = html.index("let onlinePolled = 0;")
    body = html[start:html.index("// while a round waits on its telling", start)]
    assert "/world/sync?as=" in body and 'api("/players")' not in body                       # the new list, not the old one
    assert "(away)" not in body                                                              # a logged-off player is not shown
    assert 'p.scene_id === me.scene_id && (me.at == null || p.at == null || me.at === p.at)' in body   # level: the same scene and minute
    assert 's.className = level ? "same" : "remote"' in body and "if (!level) { s.dataset.id" in body   # only a remote player can be chosen
    assert "if (delta < 0) return { behind: -delta, there };" in body and "is behind you by" in body    # behind: told so, no Y/N
    assert "await askYesNo(plan.prompt)" in body and 'requestSubmit()' in body                # Y acts it as an ordinary action
    css = html[:html.index("</style>")]
    assert "--same: #f0d24a" in css and "--remote: #c9b6f2" in css and "#online-list .same" in css and "#online-list .remote" in css


def test_the_how_to_play_page_says_what_the_brief_says_to_players() -> None:
    """Marc, 2026-10-20: "Take the 'for players' section and replace the player instructions that are presented in the how to play link in the UI." The page's
    instructions (the Welcome panel) are the brief's "For Players" section, word for word; the doc is edited, the brief and the page follow, and this fails if only one did."""
    import html as htmllib
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    where = root / "release" / "Smritilila in Brief.md"
    brief = (where if where.exists() else root / "Smritilila in Brief.md").read_text(encoding="utf-8")   # the export keeps it at its root
    section = brief[brief.index("\n# For Players\n") + len("\n# For Players\n"):].split("\n# ", 1)[0]   # to the next heading: "Worlds and stories" follows it (2026-10-24)
    want =[re.sub(r"\s+", " ", p.replace("**", "")).strip() for p in section.split("\n\n") if p.strip()]
    page = (root / "src" / "smritilila" / "web" / "index.html").read_text(encoding="utf-8")
    panel = page[page.index('<p id="intro-world"></p>') + len('<p id="intro-world"></p>'):page.index('<button id="intro-ok">')]
    got = [re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", "", p))).strip() for p in re.findall(r"<p>(.*?)</p>", panel, flags=re.S)]
    assert got == want


def test_the_play_bar_ends_with_the_day_and_time_of_the_character_played() -> None:
    """Marc, 2026-10-19: "let's add day and time to the end of the playing as line". The words are the ones /world/sync already gives each
    player (`at_text`: "Monday, 5 October 2026, 7:13 in the evening"), so no route is new and every running server answers it. Structure only: it was
    met in a browser (see the TODO entry); no JS runtime is available to the suite."""
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "src" / "smritilila" / "web" / "index.html").read_text(encoding="utf-8")
    bar = html[html.index('<span class="muted">playing as</span>'):html.index('id="state-toggle"')]
    assert bar.index('id="current-location-label"') < bar.index('id="current-time-label"')       # at the end of the line, after where
    fn = html[html.index("function renderCurrentTime()"):]
    fn = fn[:fn.index("\n}\n")]
    assert "p.entity_id === actingAs" in fn and "me.at_text" in fn and ': ""' in fn                  # the played character's own clock; nothing without one
    refresh = html[html.index("async function refreshOnline"):html.index('document.getElementById("online-list").addEventListener("click"')]
    assert refresh.index("syncPlayers = players || [];") < refresh.index("renderCurrentTime();") < refresh.index("if (!others.length)")   # set before the early return for a player alone
    world = html[html.index("async function refreshWorld()"):]
    assert "refreshOnline(true)" in world[:world.index("return world;")]                           # the clock moves with every pulse


def test_the_cards_of_those_here_ask_the_store_again_after_every_pulse_and_never_go_blank() -> None:
    """Marc, 2026-10-20: Filonius's cat got a memory from each of his pulses and its card kept the first one it had loaded (`loadEntityTuples` returned if it had
    ever asked; the cache was emptied only when the scene had news, and a player's own pulse answers with the actor's narration alone). Now every open card asks
    the store after each committed pulse and each piece of scene news, keeps what it shows while it asks, and is drawn again, in that card alone, only if the answer
    differs. Structure only (no JS runtime is available to the suite); the behaviour was met in a browser, see the TODO entry."""
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "src" / "smritilila" / "web" / "index.html").read_text(encoding="utf-8").replace("\r\n", "\n")

    def body(head: str) -> str:
        i = html.index(head)
        return html[i:html.index("\n}\n", i)]
    assert html.count("entityTuples = new Map()") == 1                                  # declared once; nothing empties the cache any more
    load = body("async function loadEntityTuples(id)")
    assert "renderEntityPanel" not in load                                              # an answer never redraws the whole panel
    assert "sameTuples(had, fresh)" in load and load.index("sameTuples(had, fresh)") < load.index("refreshEntityCard(id)")   # an answer that does not differ draws nothing
    assert "entityTupleAsk.get(id) !== ask" in load                                     # a slow older answer is let go
    assert "if (had !== null) return;" in load                                          # a failed look keeps what the card shows
    assert "(remembering...)" not in load and "(remembering...)" in body("function fillEntityCardBody(id, body)")   # only a first look says it
    assert "stored && stored.length ? stored" in body("function entityCardMemories(id)")   # the store's account first, what the page was told until it has one
    refresh = body("function refreshEntityTuples()")
    assert "tupleEpoch++" in refresh and "expandedEntities.has(id)" in refresh          # every card shown open, no other
    assert "refreshEntityTuples();" in body("function routeNarrations(")                  # after every pulse
    assert "refreshEntityTuples();" in body("function appendWitnessed(seen)")             # and whenever the scene has news
    panel = body("function renderEntityPanel()")
    assert "card.dataset.entityId = id" in panel and "fillEntityCardBody(id, body)" in panel
    assert "entityTupleEpoch.set(id, -1)" in panel                                      # opening a card asks the store whatever it holds
    card = body("function refreshEntityCard(id)")
    assert "replaceChildren" in card and "renderEntityPanel" not in card                # swapped in that card alone


def test_a_running_server_sees_a_world_added_to_the_registry(tmp_path, monkeypatch) -> None:
    """Marc, 2026-10-05: Austin was missing from every older server's selector -- each read worlds.json once."""
    import os
    cfg = tmp_path / "worlds.json"
    monkeypatch.setenv("SMRITILILA_WORLDS_CONFIG", str(cfg))
    monkeypatch.setattr(app.state, "worlds_registry", None, raising=False)
    monkeypatch.setattr(app.state, "worlds_registry_stamp", None, raising=False)
    cfg.write_text(json.dumps({"worlds": [{"name": "Stonemarch", "database_url": "postgresql+asyncpg://x/g"}]}), encoding="utf-8")
    assert [w.name for w in adhisthana.get_worlds_registry()] == ["Stonemarch"]
    st = cfg.stat()
    cfg.write_text(json.dumps({"worlds": [{"name": "Stonemarch", "database_url": "postgresql+asyncpg://x/g"},
                                          {"name": "Austin", "database_url": "postgresql+asyncpg://x/a"}]}), encoding="utf-8")
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # a coarse clock must still see the change
    assert [w.name for w in adhisthana.get_worlds_registry()] == ["Stonemarch", "Austin"]
    # a half-written file keeps the list already held, and is read again once whole
    cfg.write_text('{"worlds": [', encoding="utf-8")
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000))
    assert [w.name for w in adhisthana.get_worlds_registry()] == ["Stonemarch", "Austin"]


def test_a_pinned_world_narrates_by_its_own_share(tmp_path, monkeypatch) -> None:
    """Marc, 2026-10-05: Austin is narrated by the mid-size model alone, whatever share the other servers run on."""
    cfg = tmp_path / "worlds.json"
    cfg.write_text(json.dumps({"worlds": [
        {"name": "Austin", "database_url": "postgresql+asyncpg://x/a", "port": 8202, "rotate_share": 0},
        {"name": "Stonemarch", "database_url": "postgresql+asyncpg://x/g", "port": 8198},
        {"name": "Odd", "database_url": "postgresql+asyncpg://x/o", "port": 8203, "rotate_share": "lots"}]}), encoding="utf-8")
    monkeypatch.setenv("SMRITILILA_WORLDS_CONFIG", str(cfg))
    monkeypatch.setattr(app.state, "worlds_registry", None, raising=False)
    monkeypatch.setattr(app.state, "worlds_registry_stamp", None, raising=False)
    from smritilila.llm import providers
    providers.all_providers()
    monkeypatch.setattr(providers, "_PROVIDERS", {n: providers.Provider(name=n, narrate_rank=r, best_rank=r, make_narrator=lambda: object(), make_best=lambda m: None,
                                                                      describe=lambda: None, hint="X") for r, n in enumerate(("alpha", "beta"))})   # two providers, whoever they are
    monkeypatch.setenv("SMRITILILA_ROTATE_SHARE", "0.95")
    shares = {}
    for port in (8202, 8198, 8203):
        monkeypatch.setenv("SMRITILILA_DEV_PORT", str(port))
        shares[port] = adhisthana._resolve_roles().smrti_client._share
    assert shares == {8202: 0.0, 8198: 0.95, 8203: 0.95}  # a nonsense share is the server's own
