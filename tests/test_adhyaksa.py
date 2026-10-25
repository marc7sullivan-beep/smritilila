# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""smritilila.routes.adhyaksa (2026-10-06): the owner's admin screen. The app is the real one, with its real gate; the registry is a
file in a temporary folder and the servers it probes are faked."""
import json

import pytest
from fastapi.testclient import TestClient

from smritilila import adhisthana
from smritilila.main import app
from smritilila.routes import adhyaksa
from smritilila.worlds_registry import load_worlds_registry

OWNER = {}
GUEST = {"Cf-Access-Authenticated-User-Email": "guest@example.com", "Cf-Ray": "x"}


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    cfg = tmp_path / "worlds.json"
    cfg.write_text(json.dumps({"worlds": [
        {"name": "Austin", "database_url": "postgresql+asyncpg://x/a", "port": 8202, "url": "https://austin.example", "description": "Texas.", "guests": True},
        {"name": "Crossing", "database_url": "postgresql+asyncpg://x/c", "port": 8199, "guests": ["a@x.com"], "gods": ["g@x.com"]},
        {"name": "Local", "database_url": "postgresql+asyncpg://x/l"},
    ]}), encoding="utf-8")
    monkeypatch.setenv("SMRITILILA_WORLDS_CONFIG", str(cfg))
    monkeypatch.setattr(adhisthana, "get_worlds_registry", lambda: load_worlds_registry(cfg))
    answers = {"/health": {"status": "ok"}, "/players": [{"label": "Franklin", "online": True}, {"label": "Ada", "online": False}]}
    monkeypatch.setattr(adhyaksa, "_ask", lambda port, path: answers[path] if port == 8202 else None)
    return cfg


@pytest.fixture()
def client():
    return TestClient(app)


def test_the_owner_sees_each_world_its_server_and_who_may_enter(client, registry):
    d = client.get("/admin/worlds", headers=OWNER).json()
    by = {w["name"]: w for w in d["worlds"]}
    assert by["Austin"]["up"] is True and by["Austin"]["online"] == ["Franklin"] and by["Austin"]["guests"] == {"mode": "all", "emails": []}
    assert by["Crossing"]["up"] is False and by["Crossing"]["guests"] == {"mode": "list", "emails": ["a@x.com"]} and by["Crossing"]["gods"] == ["g@x.com"]
    assert by["Local"]["up"] is None and by["Local"]["guests"]["mode"] == "none"      # no server of its own: not asked
    assert "Stonemarch" in d["always_open"]


@pytest.mark.parametrize("mode,emails,expect", [("all", [], True), ("none", [], False), ("list", ["B@X.com", " a@x.com", "b@x.com"], ("a@x.com", "b@x.com"))])
def test_a_guest_setting_is_kept_in_worlds_json(client, registry, mode, emails, expect):
    r = client.put("/admin/worlds/Crossing/guests", json={"mode": mode, "emails": emails}, headers=OWNER)
    assert r.status_code == 200, r.text
    kept = {w.name: w for w in load_worlds_registry(registry)}
    assert kept["Crossing"].guests == expect and kept["Austin"].guests is True and kept["Crossing"].gods == ("g@x.com",)   # the rest untouched


@pytest.mark.parametrize("body,code", [({"mode": "some"}, 422), ({"mode": "list", "emails": []}, 422), ({"mode": "list", "emails": ["not an email"]}, 422)])
def test_a_bad_setting_is_refused_and_changes_nothing(client, registry, body, code):
    before = registry.read_text(encoding="utf-8")
    assert client.put("/admin/worlds/Crossing/guests", json=body, headers=OWNER).status_code == code
    assert registry.read_text(encoding="utf-8") == before


def test_an_unknown_world_is_a_404(client, registry):
    assert client.put("/admin/worlds/Nowhere/guests", json={"mode": "all"}, headers=OWNER).status_code == 404


@pytest.mark.parametrize("method,path", [("GET", "/admin"), ("GET", "/admin/worlds"), ("PUT", "/admin/worlds/Crossing/guests")])
def test_a_guest_is_refused_every_admin_route_and_nothing_changes(client, registry, method, path):
    before = registry.read_text(encoding="utf-8")
    assert client.request(method, path, json={"mode": "all"}, headers=GUEST).status_code == 403
    assert registry.read_text(encoding="utf-8") == before


def test_the_endpoints_refuse_a_guest_on_their_own_even_if_the_gate_were_open(client, registry, monkeypatch):
    from smritilila import main
    monkeypatch.setattr(main, "_OPEN_TO_ANY", {"/admin", "/admin/worlds", "/admin/worlds/Crossing/guests", "/me"})
    assert client.get("/admin/worlds", headers=GUEST).status_code == 403
    assert client.put("/admin/worlds/Crossing/guests", json={"mode": "all"}, headers=GUEST).status_code == 403
    assert client.get("/admin", headers=GUEST).status_code == 403
    assert load_worlds_registry(registry)[1].guests == ("a@x.com",)


def test_the_owner_gets_the_page_and_a_link_to_it():
    page = TestClient(app).get("/admin", headers=OWNER)
    assert page.status_code == 200 and "Smritilila admin" in page.text and "/admin/worlds" in page.text
    index = TestClient(app).get("/").text
    assert '<button id="admin-link"' in index and 'location.href = "/admin"' in index


def test_the_new_player_intro_is_on_the_page_for_a_guest_once():
    index = TestClient(app).get("/").text
    assert 'id="intro-panel"' in index and "how to play" in index and "smritilila.introSeen" in index
    panel = index.split('id="intro-panel"', 1)[1].split('id="continue-view"', 1)[0]
    assert "<b>Act</b>" in panel and "<b>Go</b>" in panel                   # the two buttons are introduced
    assert "<b>Time.</b>" in panel and "Proceed? Y / N" in panel                  # time, and the jump-ahead question
    assert "half an hour" in panel and "ten minutes" not in panel                  # the threshold is 30 (2026-10-16), each world its own
    assert "not outside it" not in panel and "Talk to people" not in panel  # and the line about talking only in the world is gone
    assert "!isOwner && !introSeen()" in index                              # an owner is not greeted; a guest who has seen it is not again


# -- registered users and bots (2026-10-06) ----------------------------------------------------------------------------------------------

@pytest.fixture()
def users(tmp_path, monkeypatch):
    p = tmp_path / "users.json"
    monkeypatch.setenv("SMRITILILA_USERS_CONFIG", str(p))
    return p


def test_the_owner_adds_lists_enables_and_removes_users(client, users):
    assert client.get("/admin/users", headers=OWNER).json() == {"users": []}
    assert client.post("/admin/users", json={"email": " Ann@Example.com ", "builder": False}, headers=OWNER).status_code == 201
    assert client.post("/admin/users", json={"email": "ann@example.com"}, headers=OWNER).status_code == 409          # once
    assert client.post("/admin/users", json={"email": "nope"}, headers=OWNER).status_code == 422
    assert client.post("/admin/users", json={"email": "owner@example.com"}, headers=OWNER).status_code == 409   # an owner needs no registering
    u = client.get("/admin/users", headers=OWNER).json()["users"]
    assert [(x["email"], x["builder"]) for x in u] == [("ann@example.com", False)] and u[0]["added"]
    assert client.put("/admin/users/ann@example.com/builder", json={"builder": True}, headers=OWNER).status_code == 200
    assert client.get("/admin/users", headers=OWNER).json()["users"][0]["builder"] is True
    assert client.put("/admin/users/zed@example.com/builder", json={"builder": True}, headers=OWNER).status_code == 404
    assert client.delete("/admin/users/ann@example.com", headers=OWNER).status_code == 200
    assert client.delete("/admin/users/ann@example.com", headers=OWNER).status_code == 404
    assert client.get("/admin/users", headers=OWNER).json() == {"users": []}


@pytest.mark.parametrize("method,path,body", [("GET", "/admin/users", None), ("POST", "/admin/users", {"email": "a@b.co"}),
                                              ("PUT", "/admin/users/a@b.co/builder", {"builder": True}), ("DELETE", "/admin/users/a@b.co", None),
                                              ("GET", "/admin/bots", None), ("PUT", "/admin/bots", {"world": "Austin", "entity_id": "e", "active": True})])
def test_a_guest_cannot_touch_users_or_bots_even_a_registered_builder(client, users, registry, tmp_path, monkeypatch, method, path, body):
    monkeypatch.setenv("SMRITILILA_BOTS_CONFIG", str(tmp_path / "bots.json"))
    from smritilila import users_registry
    users_registry.add_user("guest@example.com", builder=True)
    assert client.request(method, path, json=body, headers=GUEST).status_code == 403
    assert users_registry.load_users()[0].email == "guest@example.com" and not (tmp_path / "bots.json").exists()


@pytest.fixture()
def austin(registry, tmp_path, monkeypatch):
    """Austin's server, faked: two people's characters (playable), four NPCs by memory (one a notebook); POST /world/playable is recorded."""
    monkeypatch.setenv("SMRITILILA_BOTS_CONFIG", str(tmp_path / "bots.json"))

    class Austin:
        playable = {"p1": ["Franklin"], "p2": ["Ada"]}
        npcs = [{"id": "n0", "name": "grid-lined notebook", "memories": 12, "sample": ["it lay open on the table"]},
                {"id": "n1", "name": "Maya", "memories": 9, "sample": ["I told Franklin I would come"]},
                {"id": "n2", "name": "Walt", "memories": 5, "sample": ["I put the kettle on"]},
                {"id": "n3", "name": "Cass", "memories": 2, "sample": ["I watched from the door"]}]
        made, posts_ok = [], True
        alive = {"n1", "n2", "n3"}          # what the fake model says is alive: everything but the notebook
        asked, model_down = [], False
        setting, cast, important = "Austin, Texas, today", [], set()     # the cast the overseer reads, and who the fake model says matters
        asked_important = []

    a = Austin()
    a.made, a.asked, a.asked_important, a.cast, a.important = [], [], [], [], set()

    class Model:
        def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
            if a.model_down:
                raise RuntimeError("no model")
            if tool_name == "judge_important":
                a.asked_important.append(user)
                return {"important": sorted(a.important) + ["not-a-candidate"]}
            a.asked.append(user)
            return {"alive": sorted(a.alive) + ["not-a-candidate"]}
    monkeypatch.setattr(adhyaksa, "_alive_client", lambda: Model())

    def ask(port, path):
        if port != 8202:
            return None
        if path == "/world/exists":
            return {"exists": True, "playable_entities": a.playable}
        if path == "/world/cast":
            return {"setting": a.setting, "beings": a.cast}
        if path.startswith("/world/npcs"):
            assert "sample=true" in path                                   # the judge is shown what each holds
            return a.npcs
        return None

    def post(port, path):
        a.made.append(path)
        if a.posts_ok and path.startswith("/world/playable/"):
            eid = path.rsplit("/", 1)[1]
            a.playable[eid] = [next(n["name"] for n in a.npcs if n["id"] == eid)]
            a.npcs = [n for n in a.npcs if n["id"] != eid]            # played now: no longer an NPC
        return a.posts_ok
    monkeypatch.setattr(adhyaksa, "_ask", ask)
    monkeypatch.setattr(adhyaksa, "_post", post)
    return a


def beings(client):
    return {w["name"]: w for w in client.get("/admin/bots", headers=OWNER).json()["worlds"]}["Austin"]["characters"]


def test_the_bot_list_is_the_worlds_npcs_richest_in_memory_and_never_a_persons_character(client, austin):
    d = {w["name"]: w for w in client.get("/admin/bots", headers=OWNER).json()["worlds"]}
    assert "Local" not in d                                                          # no server of its own: nothing to ask
    assert [(c["id"], c["name"], c["bot"], c["memories"]) for c in d["Austin"]["characters"]] == [("n1", "Maya", False, 9), ("n2", "Walt", False, 5), ("n3", "Cass", False, 2)]   # not the notebook: it is not alive
    assert d["Austin"]["up"] and d["Crossing"]["up"] is False and d["Crossing"]["characters"] == []
    assert "Franklin" not in str(d["Austin"]["characters"]) and "Ada" not in str(d["Austin"]["characters"])      # the people's characters are not offered...
    assert d["Austin"]["players"] == [{"id": "p1", "name": "Franklin"}, {"id": "p2", "name": "Ada"}]              # ...they are listed apart, to add by name


def test_a_model_reads_what_each_being_holds_and_only_the_living_are_offered_and_it_is_asked_once(client, austin, tmp_path):
    import json as _json
    beings(client)
    asked = austin.asked[0]
    assert "THE WORLD: Austin" in asked and "n0 | grid-lined notebook" in asked and "- it lay open on the table" in asked and "- I told Franklin I would come" in asked
    assert [c["id"] for c in beings(client)] == ["n1", "n2", "n3"] and len(austin.asked) == 1                 # judged once: the second look asks no one
    kept = _json.loads((tmp_path / "bots.json").read_text(encoding="utf-8"))["alive"]["Austin"]
    assert kept == {"n0": False, "n1": True, "n2": True, "n3": True}                                           # the judgement is kept, the notebook's too
    austin.npcs = austin.npcs + [{"id": "n4", "name": "Bea", "memories": 1, "sample": ["I sang"]}]
    austin.alive.add("n4")
    assert [c["id"] for c in beings(client)] == ["n1", "n2", "n3", "n4"]                                        # the notebook took no place...
    assert len(austin.asked) == 2 and "n4 | Bea" in austin.asked[1] and "n1 | Maya" not in austin.asked[1]    # ...and only the new one is read


def test_the_living_most_spoken_of_are_offered_beside_the_richest_in_memory_and_none_twice(client, austin, monkeypatch):
    """The dragon in its mountain holds few memories and is spoken of by many (Marc, 2026-10-06)."""
    monkeypatch.setattr(adhyaksa, "PER_RANKING", 2)                                 # two of each ranking, so the two can be told apart
    austin.npcs = [{"id": "r1", "name": "Hob", "memories": 30, "known_by": 2, "sample": ["I mended the gate"]},
                   {"id": "r2", "name": "Ula", "memories": 20, "known_by": 9, "sample": ["I sold the bread"]},
                   {"id": "d1", "name": "the red dragon", "memories": 1, "known_by": 14, "sample": ["I woke on the gold"]},
                   {"id": "c1", "name": "the high cleric", "memories": 2, "known_by": 11, "sample": ["I read the old rite"]},
                   {"id": "o1", "name": "the old map", "memories": 1, "known_by": 13, "sample": ["it hung on the wall"]}]
    austin.alive = {"r1", "r2", "d1", "c1"}                                          # the map is a thing
    got = beings(client)
    assert [c["id"] for c in got] == ["r1", "r2", "d1", "c1"]                         # the two richest, then the two most spoken of: dragon (14) before cleric (11)
    assert [c["known_by"] for c in got] == [2, 9, 14, 11] and "the old map" not in str(got)
    austin.alive = {"r1", "r2"}
    assert [c["id"] for c in beings(client)] == ["r1", "r2", "d1", "c1"]              # judged once: what was judged is kept, so the dragon stays offered
    assert len(austin.asked) == 1


def test_the_beings_that_matter_to_the_story_are_found_by_reading_the_whole_cast_and_kept(client, austin, tmp_path):
    """The dragon and the head cleric are met seldom and spoken of seldom: neither ranking finds them (Marc, 2026-10-06)."""
    import json as _json
    austin.setting = "A fading empire; a sealed temple of the old faith in the hills"
    austin.cast = [{"id": "d1", "name": "the red dragon", "memories": 1}, {"id": "c1", "name": "the high priest of the temple", "memories": 2},
                   {"id": "t1", "name": "a baker", "memories": 40}, {"id": "o1", "name": "the old map", "memories": 3}]
    austin.important = {"d1", "c1"}
    got = {c["id"]: c for c in beings(client)}
    assert got["d1"]["important"] and got["c1"]["important"] and "t1" not in got and "o1" not in got        # the baker and the map do not matter
    asked = austin.asked_important[0]
    assert "THE SETTING: A fading empire" in asked and "d1 | the red dragon | 1" in asked and "t1 | a baker | 40" in asked   # the setting and every name, with counts
    assert [c["important"] for c in beings(client)].count(True) == 2 and len(austin.asked_important) == 1      # judged once: the second look asks no one
    kept = _json.loads((tmp_path / "bots.json").read_text(encoding="utf-8"))["important"]["Austin"]
    assert kept == {"d1": True, "c1": True, "t1": False, "o1": False}                                          # every judgement kept, the negatives too
    austin.cast = austin.cast + [{"id": "n9", "name": "the new warlord", "memories": 0}]
    austin.important = {"n9"}
    got = {c["id"]: c for c in beings(client)}
    assert got["n9"]["important"] and len(austin.asked_important) == 2                                         # only the new being is read next time
    assert "n9 | the new warlord" in austin.asked_important[1] and "d1 |" not in austin.asked_important[1]
    assert got["d1"]["important"]                                                                              # and the first picks stay


def test_a_reading_names_at_most_eight_and_only_beings_it_was_shown(client, austin, monkeypatch):
    austin.cast = [{"id": f"b{i}", "name": f"Lord {i}", "memories": 0} for i in range(12)]
    austin.important = {f"b{i}" for i in range(12)}                                                           # it names all twelve
    got = [c for c in beings(client) if c["important"]]
    assert len(got) == adhyaksa.IMPORTANT_MAX == 8 and all(c["id"].startswith("b") for c in got)              # "not-a-candidate" is dropped, the rest capped


def test_a_model_that_cannot_say_who_matters_offers_none_new_and_says_so(client, austin):
    austin.cast = [{"id": "d1", "name": "the red dragon", "memories": 1}]
    austin.model_down = True
    d = {w["name"]: w for w in client.get("/admin/bots", headers=OWNER).json()["worlds"]}["Austin"]
    assert not [c for c in d["characters"] if c.get("important")] and "could not ask a model" in d["warning"]
    austin.model_down, austin.important = False, {"d1"}
    assert [c["id"] for c in beings(client) if c["important"]] == ["d1"]                                       # and it recovers by itself


def test_twenty_at_most_per_world_those_that_matter_first(client, austin):
    austin.npcs = [{"id": f"x{i}", "name": f"Being {i}", "memories": 100 - i, "known_by": 40 - i, "sample": ["I lived"]} for i in range(40)]
    austin.alive = {f"x{i}" for i in range(40)}
    austin.cast = [{"id": f"k{i}", "name": f"Lord {i}", "memories": 0} for i in range(8)]
    austin.important = {f"k{i}" for i in range(8)}
    got = [c["id"] for c in beings(client)]
    assert len(got) == adhyaksa.LIST_MAX == 20
    assert got[:8] == [f"k{i}" for i in range(8)]                                      # those that matter to the story come first
    assert got[8:18] == [f"x{i}" for i in range(10)] and got[18:] == ["x10", "x11"]    # then the richest in memory (ten), then the most spoken of


def test_what_was_switched_on_stays_on_the_list_beyond_the_cap(client, austin, monkeypatch):
    monkeypatch.setattr(adhyaksa, "LIST_MAX", 2)
    austin.npcs = [{"id": f"x{i}", "name": f"Being {i}", "memories": 100 - i, "sample": ["I lived"]} for i in range(5)]
    austin.alive = {f"x{i}" for i in range(5)}
    assert client.put("/admin/bots", json={"world": "Austin", "entity_id": "x0", "active": True}, headers=OWNER).status_code == 200
    got = [c["id"] for c in beings(client)]
    assert got == ["x0", "x1", "x2"]                                                   # x0 is known and first; the cap bounds only the fresh offers (two)
    austin.npcs = [{"id": "y0", "name": "Newcomer", "memories": 999, "sample": ["I lived"]}] + austin.npcs
    austin.alive.add("y0")
    assert [c["id"] for c in beings(client)][0] == "x0"                                # never pushed off by a richer newcomer


def test_a_model_that_cannot_be_asked_offers_none_new_says_so_and_keeps_what_was_switched_on(client, austin, tmp_path):
    austin.model_down = True
    d = {w["name"]: w for w in client.get("/admin/bots", headers=OWNER).json()["worlds"]}["Austin"]
    assert d["characters"] == [] and "could not ask a model" in d["warning"]
    assert not (tmp_path / "bots.json").exists()                                                               # nothing judged, nothing kept
    austin.model_down = False
    assert [c["id"] for c in beings(client)] == ["n1", "n2", "n3"]                                              # and it recovers by itself


def test_switching_an_npc_on_makes_it_playable_and_keeps_it_on_the_list_when_it_is_played(client, austin, tmp_path):
    r = client.put("/admin/bots", json={"world": "Austin", "entity_id": "n1", "active": True}, headers=OWNER)
    assert r.status_code == 200 and r.json() == {"world": "Austin", "bots": ["n1"]} and austin.made == ["/world/playable/n1"]
    b = beings(client)
    assert [(c["id"], c["bot"]) for c in b] == [("n1", True), ("n2", False), ("n3", False)]           # playable now, but still offered: it is ours
    client.put("/admin/bots", json={"world": "Austin", "entity_id": "n1", "active": False}, headers=OWNER)
    assert [(c["id"], c["bot"]) for c in beings(client)][0] == ("n1", False)                          # off, and still on the list to switch back on
    client.put("/admin/bots", json={"world": "Austin", "entity_id": "n1", "active": True}, headers=OWNER)
    assert austin.made == ["/world/playable/n1"]                                                      # already playable: made so once only


def test_a_character_people_play_is_made_a_bot_only_by_name_and_then_stays_on_the_list(client, austin, tmp_path):
    from smritilila import bots_registry
    put = lambda **k: client.put("/admin/bots", json={"world": "Austin", "entity_id": "p1", "active": True, **k}, headers=OWNER)
    assert put().status_code == 404                                                   # not offered: refused unless named
    assert put(existing=True, entity_id="ghost").status_code == 404                  # nor can an unknown being be named
    r = put(existing=True)
    assert r.status_code == 200 and r.json()["bots"] == ["p1"] and austin.made == []   # already playable: nothing is made
    assert bots_registry.load_bots() == {"Austin": ["p1"]} and bots_registry.load_known() == {"Austin": ["p1"]}
    d = {w["name"]: w for w in client.get("/admin/bots", headers=OWNER).json()["worlds"]}["Austin"]
    assert ("p1", True) in [(c["id"], c["bot"]) for c in d["characters"]]            # on the list, switched on, by his name
    assert d["players"] == [{"id": "p2", "name": "Ada"}]                              # and no longer among those to add
    client.put("/admin/bots", json={"world": "Austin", "entity_id": "p1", "active": False}, headers=OWNER)
    assert ("p1", False) in [(c["id"], c["bot"]) for c in beings(client)]            # off, and still there to switch back on


def test_a_persons_character_cannot_be_made_a_bot_nor_an_unknown_being(client, austin, tmp_path):
    for eid in ("p1", "ghost", "n3-not-offered"):
        assert client.put("/admin/bots", json={"world": "Austin", "entity_id": eid, "active": True}, headers=OWNER).status_code == 404
    from smritilila import bots_registry
    assert austin.made == [] and bots_registry.load_bots() == {} and bots_registry.load_known() == {}


def test_a_bot_that_could_not_be_made_playable_is_not_recorded(client, austin, tmp_path):
    austin.posts_ok = False
    assert client.put("/admin/bots", json={"world": "Austin", "entity_id": "n2", "active": True}, headers=OWNER).status_code == 502
    from smritilila import bots_registry
    assert bots_registry.load_bots() == {} and bots_registry.load_known() == {}


def test_a_world_whose_server_does_not_answer_or_that_is_unknown_refuses(client, austin):
    put = lambda w, e: client.put("/admin/bots", json={"world": w, "entity_id": e, "active": True}, headers=OWNER).status_code
    assert put("Crossing", "n1") == 409 and put("Nowhere", "n1") == 404 and put("Local", "n1") == 404


def test_the_bots_screen_can_add_an_existing_character_after_a_confirm():
    page = TestClient(app).get("/admin", headers=OWNER).text
    assert "add an existing character" in page and "existing: true" in page and "confirm(" in page and "spoken of by" in page


def test_the_admin_page_has_a_button_to_each_world_and_the_bots_and_users_screens():
    page = TestClient(app).get("/admin", headers=OWNER).text
    for needle in ('textContent: "open"', 'id="tab-bots"', "activate bots", 'id="tab-users"', "registered users", "/admin/bots", "/admin/users", "builder access"):
        assert needle in page, needle
