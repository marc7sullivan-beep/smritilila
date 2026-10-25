# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""smrti/pratistha.py and its endpoints -- directed growth, proposed, reviewed, planted (v4.61)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from smritilila import main
from smritilila import adhisthana
from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.adhisthana import WorldSession
from smritilila.main import app
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.smrti.pratistha import check, plant, probe
from test_main import client, fake_store  # noqa: F401 -- the shared fixtures
from smritilila import dvara
from smritilila.routes import pratistha as pratistha_routes

SPEC = {
    "title": "The Reed Chapel", "why": "the world's talk keeps naming it",
    "places": [{"key": "chapel", "label": "a chapel of reeds", "name": "the Reed Chapel", "in": "fen", "scene": True,
                "tuple": "A chapel woven of green reeds stands on stilts over the black water of the fen."}],
    "people": [{"key": "keeper", "label": "a priest in a wet grey robe", "name": "Ossa Wend", "in": "chapel",
                "tuples": [{"text": "I keep the Reed Chapel for those who pay the fen its due, and I count who comes.", "weight": "weighty", "use": "useful"},
                           {"text": "The quiet men from the river bring me silver every new moon, and ask for names.", "weight": "notable", "use": "key"}]}],
    "things": [{"key": "ledger", "label": "a ledger of names", "in": "keeper", "tuple": "A ledger bound in eelskin.",
                "pages": ["Page of names: the miller's boy, the tinker, two drovers from the south road, all paid for.",
                          "Page of debts: what the quiet men owe the chapel for every name brought across the water."]}],
    "breath": {"in": "chapel", "far_every": 1},
}


def _world() -> tuple[Maya, list[Memory], list[Record]]:
    canon = Maya(world_time=10, entities={
        "land": Entity("land", ["the river country"], 0, "genesis"),
        "inn": Entity("inn", ["the inn"], 0, "genesis", avarana="scene"),
        "fen": Entity("fen", ["the black fen"], 0, "genesis"),
        "teo": Entity("teo", ["Teo"], 0, "genesis"),
    })
    mems = [Memory("m1", "teo", frozenset(), "I sat by the inn fire and heard talk of a chapel in the fen.", "PRIMARY", 4, "inn", 9, "p9", None),
            Memory("m2", "inn", frozenset({"teo"}), "The inn's fire burned low.", "AMBIENT", 5, "inn", 9, "p9", None)]
    recs = [Record("r1", "kosha", "inn", "land", 0, "p0"), Record("r2", "kosha", "fen", "land", 0, "p0"),
            Record("r3", "kosha", "teo", "inn", 0, "p0")]
    return canon, mems, recs


def _kosha(canon, mems, recs):
    from smritilila.smrti.samskara import Samskara
    return Samskara.build(canon, mems, recs, players=["teo"], clock=None, defaults=None).log_for(canon).kosha


def test_a_seed_that_stands_is_planted_as_inserts_and_proved_read_only() -> None:
    canon, mems, recs = _world()
    errors, warnings = check(SPEC, _kosha(canon, mems, recs), {"teo"}, 11)
    assert errors == [] and warnings == []
    ents, new_mems, new_recs, ids = plant(SPEC, 11, None, "proof", lambda text: None)
    kinds = {r.kind for r in new_recs}
    assert {"kosha", "name", "prakriti"} <= kinds and len(ents) == 5                    # chapel, keeper, ledger, two pages
    page1, page2 = [e.id for e in ents if e.labels == ["page 1"]][0], [e.id for e in ents if e.labels == ["page 2"]][0]
    into = {r.entity_id: r.value for r in new_recs if r.kind == "kosha"}
    assert into[page1] == ids["ledger"] and into[page2] == page1                        # a page inside the one before it
    breath = next(r for r in new_recs if r.kind == "prakriti")
    assert breath.entity_id == ids["chapel"] and json.loads(breath.value) == {"far_every": 1} and breath.note == "pacing"
    proof = probe(canon, mems, recs, ents, new_mems, new_recs, ["teo"], ids, SPEC)
    assert proof["leaks"] == {"teo": []}                                                # far from Teo: nothing of it in his prompt
    assert proof["stands"] == []
    assert proof["reads"] == {"a ledger of names": {"pages": 2, "shown when read": 2, "shown when idle": 0}}


def test_a_seed_that_does_not_stand_is_refused_by_the_check() -> None:
    canon, mems, recs = _world()
    k = _kosha(canon, mems, recs)
    bad = json.loads(json.dumps(SPEC))
    bad["people"][0]["in"] = "fen"                    # a region, not a scene: nowhere to stand
    bad["things"][0]["in"] = "teo"                    # never into a player
    bad["places"][0]["in"] = "atlantis"               # nowhere the world or the seed holds
    bad["breath"]["far_every"] = 20
    errors, _ = check(bad, k, {"teo"}, 11)
    assert any("atlantis" in e for e in errors) and any("thing ledger" in e for e in errors)
    assert any("person keeper" in e for e in errors) and any("breath" in e for e in errors)
    taken = json.loads(json.dumps(SPEC))
    taken["people"][0]["name"] = "Teo"
    canon.entities["teo"].labels = ["Teo"]
    k2 = _kosha(canon, mems, recs + [Record("r9", "name", "teo", "Teo", 0, "p0")])
    assert any("already called Teo" in e for e in check(taken, k2, {"teo"}, 11)[0])


class _Builder:
    """A fake model for the builder: the survey and the seed, by tool."""
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=None):
        self.calls.append(tool_name)
        if tool_name == "survey":
            return {"candidates": [{"what": "the Reed Chapel the inn talks of", "why": "spoken of, not held", "source": "talk",
                                    "importance": 4, "nearness": 3, "anchor": "fen"}], "choice": 0}
        return json.loads(json.dumps(SPEC))


def test_a_proposal_waits_for_review_and_is_planted_only_when_a_god_says(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                                                          fake_store, tmp_path) -> None:
    """Marc, 2026-10-02: "on demand first ... view each new build as a proposal first so we can check it over"."""
    canon, mems, recs = _world()
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.recent_memories, world.records = list(mems), list(recs)
    client.app.state.world = world
    builder = _Builder()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: adhisthana.GenerativeRoles(smrti_client=builder))
    monkeypatch.setattr(pratistha_routes, "PROPOSALS_DIR", tmp_path)
    r = client.post("/world/pratistha", json={})
    assert r.status_code == 200, r.text
    p = r.json()
    assert builder.calls == ["survey", "pratistha"] and p["status"] == "proposed" and p["errors"] == []
    assert any("Ossa Wend" in line for line in p["summary"]) and fake_store.commit_calls == []   # nothing planted yet
    assert [x["id"] for x in client.get("/world/pratistha").json()] == [p["id"]]
    planted: dict = {}
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry", lambda: type("E", (), {"database_url": "postgresql://x/y"})())
    monkeypatch.setattr("smritilila.backup.backup_world", lambda entry, reason="": planted.setdefault("backup", reason))
    monkeypatch.setattr(pratistha_routes, "_db_proof", lambda *a: planted.setdefault("copy", True))
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("Em", (), {"embed": staticmethod(lambda text: None)})())
    r = client.post(f"/world/pratistha/{p['id']}/plant")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "planted" and planted == {"backup": "pratistha", "copy": True}
    _canon, result, _ = fake_store.commit_calls[-1]
    assert {e.labels[0] for e in result.promoted_entities} >= {"a chapel of reeds", "a priest in a wet grey robe", "a ledger of names"}
    assert client.post(f"/world/pratistha/{p['id']}/plant").status_code in (409, 404)        # planted once
    client.app.state.world = world  # the endpoint drops the world to reload it; the fake store has no loader here
    world.recent_memories += list(result.memories_written)
    w = client.get(f"/world/pratistha/{p['id']}/watch").json()                             # tracked from here on
    assert set(w["beings"]) == {"chapel", "keeper", "ledger", } and w["breath"] == {"in": "chapel", "far_every": 1}
    assert all(b["turns_since"] == 0 and b["met_by_players"] == [] for b in w["beings"].values())


def test_a_proposal_touching_a_secret_is_withheld_from_every_god_outside_it(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                                                            fake_store, tmp_path) -> None:
    """Option 1 (Marc, 2026-10-02): the builder reads the whole world, a secret seed's lived consequences too, so a
    proposal can carry a secret. One that names a secret entity is withheld -- listed, never shown, read, planted or
    rejected -- from any god outside the secret (a request through Cloudflare); what it plants is born secret."""
    canon, mems, recs = _world()
    canon.entities["quiet"] = Entity("quiet", ["the quiet men"], 0, "genesis")
    recs = recs + [Record("rs", "kosha", "quiet", "inn", 0, "p0"), Record("rx", "secret", "quiet", "anu", 0, "seed")]
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.recent_memories, world.records = list(mems), list(recs)
    client.app.state.world = world
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: adhisthana.GenerativeRoles(smrti_client=_Builder()))
    monkeypatch.setattr(pratistha_routes, "PROPOSALS_DIR", tmp_path)
    away = {"cf-ray": "remote", "cf-access-authenticated-user-email": sorted(dvara.OWNERS)[0]}  # a god through Cloudflare
    p = client.post("/world/pratistha", json={}, headers=away).json()
    assert p["title"] == "(withheld: it touches a secret)" and p["summary"] == [] and p["survey"] == []
    assert client.get("/world/pratistha", headers=away).json()[0]["title"].startswith("(withheld")
    assert client.get(f"/world/pratistha/{p['id']}", headers=away).status_code == 403
    assert client.post(f"/world/pratistha/{p['id']}/plant", headers=away).status_code == 403
    here = client.get("/world/pratistha").json()[0]                               # the god who holds it
    assert here["title"] == "The Reed Chapel" and here["summary"]
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry", lambda: type("E", (), {"database_url": "postgresql://x/y"})())
    monkeypatch.setattr("smritilila.backup.backup_world", lambda entry, reason="": None)
    monkeypatch.setattr(pratistha_routes, "_db_proof", lambda *a: None)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("Em", (), {"embed": staticmethod(lambda text: None)})())
    assert client.post(f"/world/pratistha/{p['id']}/plant").status_code == 200
    _canon, result, _ = fake_store.commit_calls[-1]
    born = {e.id for e in result.promoted_entities}
    secret = {r.entity_id for r in fake_store._records if r.kind == "secret"} if hasattr(fake_store, "_records") else None
    assert secret is None or born <= secret                                      # grown from a secret: born secret


def test_a_single_sheet_carries_its_words_as_its_own_memory_not_a_one_link_chain() -> None:
    """The Temple's first proposal, 2026-10-02: a one-page thing planted as a chain was never read (v4.60 reads two
    links or more) -- caught by the proof. One sheet's words are its own memory."""
    spec = json.loads(json.dumps(SPEC))
    spec["things"][0]["pages"] = ["Only this: the miller's boy, paid for."]
    ents, mems, _recs, ids = plant(spec, 11, None, "p", lambda text: None)
    assert not [e for e in ents if e.labels[0].startswith("page")]
    assert [m.experience for m in mems if m.entity_id == ids["ledger"]] == ["A ledger bound in eelskin.", "Only this: the miller's boy, paid for."]


def test_in_a_world_that_holds_a_secret_every_proposal_goes_to_the_holder_first(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                                                                fake_store, tmp_path) -> None:
    """Holder first (Marc, 2026-10-02): the Temple's first proposal paraphrased a secret no word of its names, so in a
    world holding any secret a proposal is withheld until its holder releases it -- and only the holder may."""
    canon, mems, recs = _world()
    canon.entities["hidden"] = Entity("hidden", ["a sealed jar"], 0, "genesis")
    recs = recs + [Record("rh", "kosha", "hidden", "inn", 0, "p0"), Record("rx", "secret", "hidden", "anu", 0, "seed")]
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.recent_memories, world.records = list(mems), list(recs)
    client.app.state.world = world
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: adhisthana.GenerativeRoles(smrti_client=_Builder()))
    monkeypatch.setattr(pratistha_routes, "PROPOSALS_DIR", tmp_path)
    away = {"cf-ray": "remote", "cf-access-authenticated-user-email": sorted(dvara.OWNERS)[0]}
    p = client.post("/world/pratistha", json={}).json()                         # names nothing secret, still withheld
    assert client.get("/world/pratistha", headers=away).json()[0]["title"].startswith("(withheld")
    assert client.post(f"/world/pratistha/{p['id']}/release", headers=away).status_code == 403
    assert client.post(f"/world/pratistha/{p['id']}/release").status_code == 200
    assert client.get("/world/pratistha", headers=away).json()[0]["title"] == "The Reed Chapel"


def test_a_place_that_reads_as_one_the_world_already_holds_is_warned_of_never_refused() -> None:
    """Found 2026-10-10 on Stonemarch: a focused build made "Lower Market" beside the existing "approach to the lower market" and the check said nothing."""
    canon, mems, recs = _world()
    canon.entities["mkt"] = Entity("mkt", ["the approach to the lower market"], 0, "genesis", avarana="scene")
    recs = recs + [Record("r8", "kosha", "mkt", "land", 0, "p0")]
    k = _kosha(canon, mems, recs)

    def spec_with(label, name):
        s = json.loads(json.dumps(SPEC))
        s["places"][0].update({"label": label, "name": name})
        return s
    errors, warnings = check(spec_with("the lower market", "Lower Market"), k, {"teo"}, 11)
    assert errors == [] and any("reads as mkt" in w and "second one" in w for w in warnings)         # two rare words of the existing label: warned, not refused
    assert check(spec_with("the lower fen path", "the Lower Fen Path"), k, {"teo"}, 11)[1] == []     # "lower" alone is one word: not enough
    assert check(SPEC, k, {"teo"}, 11)[1] == []                                                       # an unrelated place: silent, as before
    assert check(spec_with("a salt flat", "the Salt Flat"), k, {"teo"}, 11)[1] == []
