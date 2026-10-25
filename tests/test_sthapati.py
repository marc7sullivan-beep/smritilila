# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""smritilila.routes.sthapati (2026-10-06): the master builder, the conversation in which a world's brief is made. The model is a fake that
records what it is sent; the app is the real one, with its real gate."""
import json

import pytest
from fastapi.testclient import TestClient

from smritilila.main import app
from smritilila.routes import sthapati

OWNER = {}  # this machine: no Cloudflare headers
GUEST = {"Cf-Access-Authenticated-User-Email": "guest@example.com", "Cf-Ray": "x"}
FULL = {"name": "Austin", "source": "Present day: Austin, Texas, as it is now", "where_when": "Austin, Texas, 5 October 2026",
        "begins": "", "different": "", "feel": "", "breadth": "around"}


class _Builder:
    def __init__(self, out=None, boom=None):
        self.out, self.boom, self.calls = out, boom, []

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192, images=None):
        self.calls.append({"system": system, "user": user, "tool": tool_name, "schema": input_schema, "images": images})
        if self.boom:
            raise self.boom
        return self.out


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def builder(monkeypatch):
    b = _Builder()
    monkeypatch.setattr(sthapati, "_builder_client", lambda: b)
    return b


def msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def test_a_turn_returns_the_builders_reply_and_what_it_sets(client, builder):
    builder.out = {"message": "Which Austin: the real one, as it is now?", "patch": {"name": "Austin", "source": "Present day: Austin, Texas", "breadth": "around"}, "ready": False}
    r = client.post("/builder/turn", json={"messages": msgs(("user", "I want a world in Austin")), "draft": {}}, headers=OWNER)
    assert r.status_code == 200, r.text
    assert r.json() == {"message": "Which Austin: the real one, as it is now?", "patch": {"name": "Austin", "source": "Present day: Austin, Texas", "breadth": "around"}, "ready": False}
    call = builder.calls[0]
    assert call["tool"] == "builder_turn" and "builder of worlds" in call["system"]
    assert "Creator: I want a world in Austin" in call["user"] and "name: (not yet)" in call["user"]    # the brief and the talk both go in


def test_the_builder_is_told_todays_date_for_a_present_day_world(client, builder):
    import time
    builder.out = {"message": "ok"}
    client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {}}, headers=OWNER)
    builder.out = {"line": "a line", "opening": "an opening"}
    client.post("/builder/preview", json={"draft": FULL}, headers=OWNER)
    for call in builder.calls:                                                          # a model's own idea of "now" is a year behind
        assert call["user"].startswith("TODAY (the owner's date, for any present-day world): ") and time.strftime("%Y") in call["user"][:90]


def test_the_current_brief_is_what_the_builder_is_shown(client, builder):
    builder.out = {"message": "ok"}
    client.post("/builder/turn", json={"messages": msgs(("user", "hi"), ("assistant", "hello"), ("user", "more")), "draft": FULL}, headers=OWNER)
    user = builder.calls[0]["user"]
    assert "source: Present day: Austin, Texas, as it is now" in user and "breadth: around" in user and "begins: (not yet)" in user
    assert user.index("Creator: hi") < user.index("Builder: hello") < user.index("Creator: more")           # in order


def test_a_patch_is_only_the_briefs_fields_as_text_and_only_what_changes(client, builder):
    builder.out = {"message": "ok", "patch": {"name": "Austin", "source": "x" * 5000, "different": 12, "feel": "fair, quiet", "granted": "an old field",
                                              "breadth": "enormous", "invented_field": "no", "begins": "Zilker Park"}, "ready": False}
    r = client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {"name": "Austin"}}, headers=OWNER)
    patch = r.json()["patch"]
    assert patch == {"source": "x" * 2000, "feel": "fair, quiet", "begins": "Zilker Park"}   # name unchanged; different not text; granted is no field now; breadth not one of three; unknown dropped; source clipped


def test_ready_needs_a_name_a_source_and_where_and_when(client, builder):
    builder.out = {"message": "ready?", "patch": {"name": "Austin"}, "ready": True}
    r = client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {}}, headers=OWNER)
    assert r.json()["ready"] is False                                                # the builder said so; the brief says otherwise
    builder.out = {"message": "ready", "patch": {"source": "Present day: Austin", "where_when": "Austin, 2026"}, "ready": True}
    r = client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {"name": "Austin"}}, headers=OWNER)
    assert r.json()["ready"] is True


def test_the_conversation_is_bounded(client, builder):
    builder.out = {"message": "ok"}
    long = [("user" if i % 2 == 0 else "assistant", f"message {i} " + "word " * 600) for i in range(101)]    # ends on a user message
    r = client.post("/builder/turn", json={"messages": msgs(*long), "draft": {}}, headers=OWNER)
    assert r.status_code == 200
    user = builder.calls[0]["user"]
    assert "message 100 " in user and "message 0 " not in user and "message 60 " not in user                 # the oldest went
    convo = user.split("THE CONVERSATION:\n", 1)[1].split("\n\nReply to", 1)[0]
    assert len(convo) <= sthapati.MAX_TRANSCRIPT_CHARS + 40 * 12                                                  # the budget, plus each line's speaker label


def test_many_short_messages_are_cut_to_the_last_forty(client, builder):
    builder.out = {"message": "ok"}
    short = [("user" if i % 2 == 0 else "assistant", f"short {i} end") for i in range(101)]                   # far under the character budget
    client.post("/builder/turn", json={"messages": msgs(*short), "draft": {}}, headers=OWNER)
    user = builder.calls[0]["user"]
    assert "short 100 end" in user and "short 61 end" in user and "short 60 end" not in user and "short 0 end" not in user    # exactly the last forty


def test_the_conversation_must_end_with_the_owner(client, builder):
    r = client.post("/builder/turn", json={"messages": msgs(("user", "hi"), ("assistant", "hello")), "draft": {}}, headers=OWNER)
    assert r.status_code == 422 and builder.calls == []
    assert client.post("/builder/turn", json={"messages": [], "draft": {}}, headers=OWNER).status_code == 422


def test_a_model_failure_or_an_empty_reply_is_a_502_not_a_crash(client, builder):
    builder.boom = RuntimeError("overloaded")
    r = client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {}}, headers=OWNER)
    assert r.status_code == 502 and "could not answer" in r.text
    builder.boom, builder.out = None, {"message": "   "}
    assert client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {}}, headers=OWNER).status_code == 502


@pytest.mark.parametrize("path,body", [("/builder/turn", {"messages": msgs(("user", "hi")), "draft": {}}),
                                       ("/builder/preview", {"draft": FULL})])
def test_a_guest_does_not_use_the_builder_and_the_model_is_never_called(client, builder, path, body):
    r = client.post(path, json=body, headers=GUEST)
    assert r.status_code == 403 and builder.calls == []


def test_the_endpoint_refuses_a_guest_on_its_own_even_if_the_gate_were_open(client, builder, monkeypatch):
    from smritilila import dvara, main
    monkeypatch.setattr(dvara, "_OWNER_ONLY", set())
    monkeypatch.setattr(main, "_OWNER_ONLY", set())
    monkeypatch.setattr(main, "_OWNER_ONLY_PREFIX", ())
    monkeypatch.setattr(main, "_OPEN_TO_ANY", {"/builder/turn", "/builder/preview", "/builder/build", "/builder/build/x", "/me"})   # a gate that lets it through
    r = client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {}}, headers=GUEST)
    assert r.status_code == 403 and "builds worlds" in r.text and builder.calls == []
    r = client.post("/builder/build", json={"name": "Tide Hall", "draft": FULL}, headers=GUEST)
    assert r.status_code == 403 and "builds worlds" in r.text
    r = client.get("/builder/build/x", headers=GUEST)                                       # the status of a build is the owner's too
    assert r.status_code == 403 and "builds worlds" in r.text


def test_the_preview_shows_what_the_world_will_open_with(client, builder):
    builder.out = {"line": "It is Austin in October 2026, a warm city between festival weekends.", "opening": "It is the first week of October.",
                   "places": ["Zilker Park", "Barton Springs Pool"] + [f"place {i}" for i in range(12)], "people": ["Wat Barris, a tavern keeper, wants a quiet year"],
                   "cautions": ["Maya Lin is also a living artist's name"]}
    r = client.post("/builder/preview", json={"draft": FULL}, headers=OWNER)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["line"].startswith("It is Austin") and j["cautions"] == ["Maya Lin is also a living artist's name"] and len(j["places"]) == 8   # bounded
    assert builder.calls[0]["tool"] == "builder_preview" and "THE BRIEF:" in builder.calls[0]["user"] and "Austin, Texas, 5 October 2026" in builder.calls[0]["user"]


def test_a_preview_needs_the_three_required_fields_and_a_whole_answer(client, builder):
    assert client.post("/builder/preview", json={"draft": {"name": "Austin"}}, headers=OWNER).status_code == 422 and builder.calls == []
    builder.out = {"line": "a line", "opening": ""}
    assert client.post("/builder/preview", json={"draft": FULL}, headers=OWNER).status_code == 502
    builder.boom = RuntimeError("x")
    assert client.post("/builder/preview", json={"draft": FULL}, headers=OWNER).status_code == 502


def test_the_conversation_is_kept_with_the_world_it_made_under_a_safe_file_name(monkeypatch, tmp_path):
    monkeypatch.setattr(sthapati, "SESSIONS_DIR", tmp_path / "builder_sessions")
    name = sthapati._write_session("The Vadhū Crossing / 2", [sthapati.BuilderMessage(role="assistant", content="What world?"), sthapati.BuilderMessage(role="user", content="A ferry")], FULL)
    assert name.endswith("-the_vadhu_crossing_2.json")                               # a safe file name from any world name
    saved = json.loads((tmp_path / "builder_sessions" / name).read_text(encoding="utf-8"))
    assert saved["name"] == "The Vadhū Crossing / 2" and saved["draft"]["breadth"] == "around" and [m["content"] for m in saved["messages"]] == ["What world?", "A ferry"]


def test_the_page_has_the_conversation_and_not_the_old_intro():
    html = (sthapati.Path(__file__).resolve().parent.parent / "src" / "smritilila" / "web" / "index.html").read_text(encoding="utf-8")
    for needle in ('id="builder-log"', 'id="builder-input"', 'id="builder-send"', 'id="nw-preview-btn"', "/builder/turn", "/builder/preview", "/builder/build"):
        assert needle in html, needle
    for needle in ('id="nw-different"', 'id="nw-feel"', 'id="nw-map-btn"', 'id="nw-map-picker"', "Maps need a model that can read images", "/builder/maps"):
        assert needle in html, needle                                                       # the two questions, and the map button with its note
    for gone in ('id="nw-days"', 'id="nw-granted"', 'id="nw-who"'):
        assert gone not in html, gone                                                       # the three they replaced
    assert "A new world from history, a book or a film -- its own memory" not in html     # the old intro text is gone
    assert 'id="nw-cancel-btn"' not in html                                                 # the corner button is the way back
    assert "/builder/build" in html                                                         # Build is a job the page polls
    for gone in ('id="nw-expand-section"', 'id="expand-world-btn"', "or build a new world", "Grow <strong"):
        assert gone not in html, gone                                                       # the selected world's text, its expand button and divider are off the builder screen


# -- Building (2026-10-06): a world is made on a server of its own; Build was a 409 on every pinned server ------------------------------------

@pytest.fixture()
def building(monkeypatch, tmp_path):
    from smritilila import adhisthana
    from smritilila.routes import loka
    from smritilila.worlds_registry import load_worlds_registry

    cfg = tmp_path / "worlds.json"
    cfg.write_text(json.dumps({"worlds": [{"name": "Stonemarch", "database_url": "postgresql+asyncpg://x/stonemarch", "port": 8198},
                                          {"name": "Old Rome", "database_url": "postgresql+asyncpg://x/rome"}]}), encoding="utf-8")   # one with a server of its own, one without
    monkeypatch.setenv("SMRITILILA_WORLDS_CONFIG", str(cfg))
    monkeypatch.setattr(adhisthana, "get_worlds_registry", lambda: load_worlds_registry(cfg))
    monkeypatch.setattr(sthapati, "SESSIONS_DIR", tmp_path / "builder_sessions")
    monkeypatch.setattr(sthapati, "HEALTH_SECONDS", 0.2)
    monkeypatch.setattr(sthapati.time, "sleep", lambda s: None)
    sthapati._BUILDS.clear()

    class World:
        made, spawned, calls = [], [], []
        health, exists, genesis = (200, {"status": "ok"}), (200, {"exists": False, "playable_entities": {}}), (200, {"player_entity_id": ""})
        manacitra = (200, {"id": "p1abc"})
        expand = (200, {"places": [f"place {i}" for i in range(14)], "people": [f"person {i}" for i in range(8)]})

    w = World()
    w.made, w.spawned, w.calls = [], [], []

    async def make_db(name):
        w.made.append(name)
        return "tide_hall", "postgresql+asyncpg://x/tide_hall"
    monkeypatch.setattr(loka, "_make_world_database", make_db)
    monkeypatch.setattr(loka, "_free_world_port", lambda: 8230)
    monkeypatch.setattr(loka, "_spawn_world_server", lambda entry: w.spawned.append((entry.name, entry.port)))

    def http(method, url, body=None, timeout=30.0):
        w.calls.append((method, url, body))
        if url.endswith("/health"):
            return w.health
        if url.endswith("/world/exists"):
            return w.exists
        if url.endswith("/genesis"):
            return w.genesis
        if url.endswith("/world/manacitra"):
            return w.manacitra
        if url.endswith("/world/expand"):
            return w.expand
        return 404, {}
    monkeypatch.setattr(sthapati, "_http", http)
    w.cfg, w.sessions = cfg, tmp_path / "builder_sessions"
    return w


BRIEF = {"name": "Tide Hall", "source": "Present day: a tide mill in Suffolk", "where_when": "Suffolk, 2026", "begins": "the mill yard", "different": "", "feel": "quiet, forgiving", "breadth": "here"}


def build(client, body=None, headers=OWNER):
    return client.post("/builder/build", json={"name": "Tide Hall", "draft": BRIEF, "messages": msgs(("assistant", "What world?"), ("user", "a tide mill"))} if body is None else body, headers=headers)


def status(client, job, headers=OWNER):
    return client.get(f"/builder/build/{job}", headers=headers)


def test_build_makes_the_world_on_a_server_of_its_own_and_seeds_it_there(client, building):
    r = build(client)
    assert r.status_code == 202, r.text
    s = status(client, r.json()["job"]).json()
    assert s["done"] and s["error"] is None and s["result"] == {"name": "Tide Hall", "port": 8230, "address": "http://127.0.0.1:8230", "public": False}
    assert building.made == ["Tide Hall"] and building.spawned == [("Tide Hall", 8230)]                       # its database, its server
    reg = json.loads(building.cfg.read_text(encoding="utf-8"))["worlds"]
    assert [w["name"] for w in reg] == ["Stonemarch", "Old Rome", "Tide Hall"] and reg[2]["port"] == 8230 and "guests" not in reg[2]    # registered; closed to guests
    genesis = [c for c in building.calls if c[1].endswith("/genesis")]
    assert genesis and genesis[0][0] == "POST" and genesis[0][1] == "http://127.0.0.1:8230/genesis"           # seeded THERE, not on this server
    assert genesis[0][2]["name"] == "Tide Hall" and genesis[0][2]["breadth"] == "here" and genesis[0][2]["begins"] == "the mill yard"
    assert [c[1].rsplit("/", 1)[1] for c in building.calls][:3] == ["health", "health", "exists"]               # waited for its server before asking
    saved = list(building.sessions.glob("*-tide_hall.json"))
    assert len(saved) == 1 and [m["content"] for m in json.loads(saved[0].read_text(encoding="utf-8"))["messages"]] == ["What world?", "a tide mill"]


def test_a_failed_seed_leaves_the_world_and_a_retry_seeds_it_without_making_another(client, building):
    building.genesis = (502, {"detail": "the seed could not be planted: truncated"})
    s = status(client, build(client).json()["job"]).json()
    assert s["done"] and s["result"] is None and "seeding failed (502)" in s["error"] and "the seed could not be planted" in s["error"]
    assert building.made == ["Tide Hall"] and any(w["name"] == "Tide Hall" for w in json.loads(building.cfg.read_text(encoding="utf-8"))["worlds"])
    building.genesis = (200, {})
    s = status(client, build(client).json()["job"]).json()                                                      # the same name again
    assert s["done"] and s["error"] is None and s["result"]["port"] == 8230
    assert building.made == ["Tide Hall"] and building.spawned == [("Tide Hall", 8230)]                         # no second database, no second server


def test_a_world_that_has_its_seed_is_not_built_again(client, building):
    status(client, build(client).json()["job"])
    building.exists = (200, {"exists": True, "playable_entities": {}})
    s = status(client, build(client).json()["job"]).json()
    assert s["done"] and s["result"] is None and "already exists and has been seeded" in s["error"]
    assert len([c for c in building.calls if c[1].endswith("/genesis")]) == 1                                   # seeded once


def test_an_old_world_with_no_server_of_its_own_is_refused_not_overwritten(client, building):
    body = {"name": "old rome", "draft": {**BRIEF, "name": "old rome"}, "messages": []}                        # the name, in any case
    s = status(client, build(client, body).json()["job"]).json()
    assert s["done"] and s["result"] is None and "already exists" in s["error"]
    assert building.made == [] and building.spawned == [] and not [c for c in building.calls if c[1].endswith("/genesis")]   # nothing made, nothing seeded


def test_a_server_that_does_not_start_is_reported(client, building):
    building.health = (0, "connection refused")
    s = status(client, build(client).json()["job"]).json()
    assert s["done"] and s["result"] is None and "did not start" in s["error"]
    assert not [c for c in building.calls if c[1].endswith("/genesis")]                                         # nothing seeded into nowhere


def test_a_build_needs_a_name_a_source_where_and_when_and_a_breadth(client, building):
    assert build(client, {"name": "Tide Hall", "draft": {**BRIEF, "source": ""}}).status_code == 422
    assert build(client, {"name": "", "draft": {**BRIEF, "name": ""}}).status_code == 422
    assert build(client, {"name": "Tide Hall", "draft": {**BRIEF, "breadth": "enormous"}}).status_code == 422
    assert building.made == [] and building.calls == []


def test_the_same_world_is_not_built_twice_at_once_and_an_unknown_job_is_a_404(client, building):
    sthapati._BUILDS["running"] = {"job": "running", "name": "Tide Hall", "step": "Seeding", "done": False, "error": None, "result": None}
    assert build(client).status_code == 409 and building.made == []
    assert status(client, "nope").status_code == 404
    assert status(client, "running").json()["step"] == "Seeding"


@pytest.mark.parametrize("make", [lambda c: build(c, headers=GUEST), lambda c: status(c, "x", headers=GUEST)])
def test_a_guest_neither_builds_nor_watches_a_build(client, building, make):
    assert make(client).status_code == 403 and building.made == [] and building.calls == []


# -- Registered users with builder access (2026-10-06) ---------------------------------------------------------------------------------

USER = {"Cf-Access-Authenticated-User-Email": "ann@example.com", "Cf-Ray": "x"}
OTHER = {"Cf-Access-Authenticated-User-Email": "bob@example.com", "Cf-Ray": "x"}


@pytest.fixture()
def registered(monkeypatch, tmp_path):
    from smritilila import users_registry
    monkeypatch.setenv("SMRITILILA_USERS_CONFIG", str(tmp_path / "users.json"))
    users_registry.add_user("ann@example.com", builder=True)
    users_registry.add_user("bob@example.com", builder=False)


def test_a_user_with_builder_access_may_talk_to_the_builder_and_one_without_may_not(client, builder, registered):
    builder.out = {"message": "Which world?"}
    assert client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {}}, headers=USER).status_code == 200
    assert client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {}}, headers=OTHER).status_code == 403
    assert client.post("/builder/turn", json={"messages": msgs(("user", "hi")), "draft": {}}, headers=GUEST).status_code == 403
    assert len(builder.calls) == 1                                                  # the model was asked once, for the one allowed


def test_a_user_with_builder_access_is_a_god_of_the_world_they_make_and_sees_only_their_own_build(client, building, registered):
    r = build(client, headers=USER)
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert status(client, job, headers=USER).json()["result"]["port"] == 8230
    reg = {w["name"]: w for w in json.loads(building.cfg.read_text(encoding="utf-8"))["worlds"]}
    assert reg["Tide Hall"]["gods"] == ["ann@example.com"] and "guests" not in reg["Tide Hall"]                  # theirs, and private
    assert status(client, job, headers=OWNER).status_code == 200                                                # the owner sees every build
    assert status(client, job, headers=OTHER).status_code == 403                                                # one not enabled cannot
    from smritilila import users_registry
    users_registry.set_builder("bob@example.com", True)
    assert status(client, job, headers=OTHER).status_code == 404                                                # enabled, but not their build


def test_an_owner_builds_a_world_with_no_other_god(client, building):
    build(client)
    assert "gods" not in {w["name"]: w for w in json.loads(building.cfg.read_text(encoding="utf-8"))["worlds"]}["Tide Hall"]


def test_a_user_cannot_seed_a_world_made_by_someone_else(client, building, registered):
    from smritilila.worlds_registry import add_world
    add_world("Tide Hall", "postgresql+asyncpg://x/tide_hall", path=building.cfg, port=8231)                    # the owner's, made and not yet seeded
    s = status(client, build(client, headers=USER).json()["job"], headers=USER).json()
    assert s["done"] and s["result"] is None and "already exists" in s["error"]
    assert not [c for c in building.calls if c[1].endswith("/genesis")]


def test_me_says_whether_the_page_should_offer_the_builder(client, registered):
    assert client.get("/me", headers=USER).json() == {"owner": False, "builder": True}
    assert client.get("/me", headers=OTHER).json() == {"owner": False, "builder": False}
    assert client.get("/me", headers=OWNER).json() == {"owner": True, "builder": True}


def test_the_seed_writer_reads_the_new_fields_and_none_of_the_old():
    """2026-10-23: `different` and `feel` replaced days, granted and who. A blank field is "as the source has it", as every blank one was."""
    from smritilila.schemas import GenesisRequest
    from smritilila.smrti.bija import brief_text
    text = brief_text("Tide Hall", {"source": "S", "where_when": "W", "begins": "", "different": "no tides at all", "feel": "deadly"})
    assert "WHAT IS DIFFERENT HERE: no tides at all" in text and "WHAT PLAY SHOULD FEEL LIKE: deadly" in text and "WHERE IT BEGINS: not given" in text
    assert "TAKES FOR GRANTED" not in text and "WHO MATTERS" not in text and "DAYS ARE LIKE" not in text
    blank = brief_text("Tide Hall", {"source": "S", "where_when": "W"})
    assert "WHAT IS DIFFERENT HERE: not given -- as the record or the source has it" in blank and "WHAT PLAY SHOULD FEEL LIKE: not given" in blank
    sent = GenesisRequest(name="x", source="s", where_when="w", different="d", feel="f", days="an old field", who="an old field")
    assert sent.brief() == {"source": "s", "where_when": "w", "begins": "", "different": "d", "feel": "f"}      # a client still sending the old ones is not refused, and they are dropped


def test_the_builder_is_told_the_new_fields_and_how_to_treat_a_map():
    for needle in ("different --", "feel --", "forgiving, fair or deadly", "A MAP:", "at least one of begins, different or feel"):
        assert needle in sthapati.TURN_SYSTEM, needle
    for gone in ("days --", "granted --", "who --", "days or granted"):
        assert gone not in sthapati.TURN_SYSTEM, gone
    assert sthapati.FIELDS == ("name", "source", "where_when", "begins", "different", "feel", "breadth")


# -- Maps (2026-10-23): saved images the owner picks; the builder looks at a new one once; a build has each read ---------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64    # not a real image: listing and copying never look inside


@pytest.fixture()
def maps(monkeypatch, tmp_path):
    from smritilila.routes import pratistha
    root = tmp_path / "maps"
    (root / "Stonemarch" / "deeper").mkdir(parents=True)
    (root / "Tide Hall").mkdir()
    for path, data in ((root / "Stonemarch" / "atlas.png", PNG), (root / "Stonemarch" / "notes.txt", b"not an image"), (root / "Stonemarch" / ".hidden.png", PNG),
                       (root / "Stonemarch" / "empty.png", b""), (root / "Stonemarch" / "deeper" / "two_deep.png", PNG), (root / "Tide Hall" / "coast.JPG", PNG),
                       (root / "loose.png", PNG)):
        path.write_bytes(data)
    monkeypatch.setattr(pratistha, "MAPS_DIR", root)
    return root


class _Seer(_Builder):
    """The model that takes pictures."""


def test_the_picker_lists_the_saved_images_one_folder_deep_and_nothing_else(client, maps):
    r = client.get("/builder/maps", headers=OWNER)
    assert r.status_code == 200
    assert r.json() == [{"id": "Stonemarch/atlas.png", "folder": "Stonemarch", "name": "atlas.png", "bytes": len(PNG)},
                        {"id": "Tide Hall/coast.JPG", "folder": "Tide Hall", "name": "coast.JPG", "bytes": len(PNG)}]   # not a text file, hidden, empty, two deep, or loose
    assert sthapati.saved_maps() == r.json()


def test_the_picker_serves_only_what_it_lists(client, maps):
    ok = client.get("/builder/maps/file", params={"id": "Stonemarch/atlas.png"}, headers=OWNER)
    assert ok.status_code == 200 and ok.content == PNG and ok.headers["content-type"] == "image/png"
    assert client.get("/builder/maps/file", params={"id": "Tide Hall/coast.JPG"}, headers=OWNER).headers["content-type"] == "image/jpeg"
    for bad in ("Stonemarch/notes.txt", "Stonemarch/../Tide Hall/coast.JPG", "../maps/loose.png", "loose.png", "Stonemarch/deeper/two_deep.png", "nope/none.png", ""):
        assert client.get("/builder/maps/file", params={"id": bad}, headers=OWNER).status_code == 404, bad


def test_a_guest_neither_lists_nor_fetches_maps_and_a_user_with_builder_access_may(client, maps, registered):
    assert client.get("/builder/maps", headers=GUEST).status_code == 403
    assert client.get("/builder/maps/file", params={"id": "Stonemarch/atlas.png"}, headers=GUEST).status_code == 403
    assert client.get("/builder/maps", headers=OTHER).status_code == 403                                   # registered, not enabled
    assert client.get("/builder/maps", headers=USER).status_code == 200


def test_a_map_is_shown_to_the_builder_once_with_the_message_and_by_a_model_that_takes_images(client, builder, maps, monkeypatch):
    seer = _Seer(out={"message": "A map of the Flanaess, 576 CY.", "patch": {"where_when": "The Flanaess, 576 CY"}})
    monkeypatch.setattr(sthapati, "_vision_client", lambda: seer)
    r = client.post("/builder/turn", json={"messages": msgs(("user", "here is my map")), "draft": {}, "look": ["Stonemarch/atlas.png", "Tide Hall/coast.JPG"]}, headers=OWNER)
    assert r.status_code == 200, r.text
    assert r.json()["patch"] == {"where_when": "The Flanaess, 576 CY"}
    assert builder.calls == [] and len(seer.calls) == 1                                                       # the one that takes pictures answered, not the text one
    sent = seer.calls[0]
    assert [(p.name, p.mime, p.data) for p in sent["images"]] == [("atlas.png", "image/png", PNG), ("coast.JPG", "image/jpeg", PNG)]
    assert "shown you 2 maps with this message (atlas.png, coast.JPG): look at it first." in sent["user"] and "A MAP:" in sent["system"]
    builder.out = {"message": "ok"}
    client.post("/builder/turn", json={"messages": msgs(("user", "and then?")), "draft": {}}, headers=OWNER)
    assert builder.calls[0]["images"] is None and "shown you" not in builder.calls[0]["user"]                  # no look, no picture, no extra words


def test_a_model_that_cannot_see_is_refused_plainly_and_an_unsaved_map_never_reaches_a_model(client, builder, maps, monkeypatch):
    from smritilila.llm.client import ImagesNotSupported
    blind = _Seer(boom=ImagesNotSupported("words only"))
    monkeypatch.setattr(sthapati, "_vision_client", lambda: blind)
    r = client.post("/builder/turn", json={"messages": msgs(("user", "a map")), "draft": {}, "look": ["Stonemarch/atlas.png"]}, headers=OWNER)
    assert r.status_code == 422 and r.json()["detail"] == sthapati.NEEDS_IMAGES
    blind.boom = RuntimeError("400 invalid image input")                                                      # a text-only server that says so its own way
    r = client.post("/builder/turn", json={"messages": msgs(("user", "a map")), "draft": {}, "look": ["Stonemarch/atlas.png"]}, headers=OWNER)
    assert r.status_code == 502 and "could not answer" in r.text and sthapati.NEEDS_IMAGES in r.text
    calls = len(blind.calls)
    assert client.post("/builder/turn", json={"messages": msgs(("user", "a map")), "draft": {}, "look": ["Stonemarch/nowhere.png"]}, headers=OWNER).status_code == 404
    four = ["Stonemarch/atlas.png"] * 4
    assert client.post("/builder/turn", json={"messages": msgs(("user", "a map")), "draft": {}, "look": four}, headers=OWNER).status_code == 422
    assert len(blind.calls) == calls and builder.calls == []                                                  # neither reached a model


def test_a_build_has_each_picked_map_copied_into_the_new_world_and_read_there(client, building, maps):
    from smritilila.backup import _safe_dirname
    (maps / "Tide Hall" / "atlas.png").write_bytes(PNG + b"1")                                                # another map by the same file name
    body = {"name": "Tide Hall", "draft": BRIEF, "messages": [], "maps": ["Stonemarch/atlas.png", "Tide Hall/atlas.png"]}
    s = status(client, build(client, body).json()["job"]).json()
    assert s["done"] and s["error"] is None
    assert s["result"]["maps"] == [{"image": "atlas.png", "proposal": "p1abc", "error": None}, {"image": "Tide Hall-atlas.png", "proposal": "p1abc", "error": None}]
    mine = maps / _safe_dirname("Tide Hall")
    assert (mine / "atlas.png").read_bytes() == PNG and (mine / "Tide Hall-atlas.png").read_bytes() == PNG + b"1"
    posted = [(c[1], c[2]) for c in building.calls if c[1].endswith("/world/manacitra")]
    assert posted == [("http://127.0.0.1:8230/world/manacitra", {"image": "atlas.png"}), ("http://127.0.0.1:8230/world/manacitra", {"image": "Tide Hall-atlas.png"})]
    urls = [c[1].rsplit("/", 1)[1] for c in building.calls]
    assert urls.index("genesis") < urls.index("manacitra")                                                     # read after the world is seeded, there
    genesis = next(c for c in building.calls if c[1].endswith("/genesis"))[2]
    assert set(genesis) == {"name", "source", "where_when", "begins", "different", "feel", "breadth"} and genesis["feel"] == "quiet, forgiving"


def test_a_map_that_cannot_be_read_leaves_the_world_built_and_says_why(client, building, maps):
    building.manacitra = (409, {"detail": "this world has no earth to file a map under"})
    body = {"name": "Tide Hall", "draft": BRIEF, "messages": [], "maps": ["Stonemarch/atlas.png"]}
    s = status(client, build(client, body).json()["job"]).json()
    assert s["done"] and s["error"] is None and s["result"]["port"] == 8230
    assert s["result"]["maps"] == [{"image": "atlas.png", "proposal": None, "error": "(409) this world has no earth to file a map under"}]


def test_a_build_with_a_map_that_is_not_saved_is_refused_before_anything_is_made(client, building, maps):
    body = {"name": "Tide Hall", "draft": BRIEF, "messages": [], "maps": ["Stonemarch/nowhere.png"]}
    assert build(client, body).status_code == 404 and building.made == [] and building.calls == []
    assert build(client, {**body, "maps": ["Stonemarch/atlas.png"] * 4}).status_code == 422 and building.made == []
    s = status(client, build(client).json()["job"]).json()                                                      # no maps: no reading, and no "maps" in the result
    assert "maps" not in s["result"] and not [c for c in building.calls if c[1].endswith("/world/manacitra")]


# -- How much to build (Marc, 2026-10-23: "Wide should be labeled small. Then add a medium and large (the size of the Stonemarch recent buildout)") ------

def built(client, breadth):
    return status(client, build(client, {"name": "Tide Hall", "draft": {**BRIEF, "breadth": breadth}, "messages": []}).json()["job"]).json()


@pytest.mark.parametrize("breadth,seed,rings", [("here", "here", 0), ("around", "around", 0), ("small", "wide", 0), ("medium", "wide", 3), ("large", "wide", 7)])
def test_a_size_is_the_seeds_breadth_and_how_many_rings_grow_it_outward(client, building, breadth, seed, rings):
    s = built(client, breadth)
    assert s["done"] and s["error"] is None
    genesis = next(c for c in building.calls if c[1].endswith("/genesis"))[2]
    assert genesis["breadth"] == seed                                                                      # the engine's own word: small is its "wide"
    asked = [c for c in building.calls if c[1].endswith("/world/expand")]
    assert len(asked) == rings and all(c[0] == "POST" and c[1] == "http://127.0.0.1:8230/world/expand" for c in asked)
    urls = [c[1].rsplit("/", 1)[1] for c in building.calls]
    assert not rings or urls.index("genesis") < urls.index("expand")                                      # grown outward from the seed, there
    if rings:
        assert s["result"]["grown"] == {"rings": rings, "places": 14 * rings, "people": 8 * rings}
    else:
        assert "grown" not in s["result"]


def test_a_ring_that_adds_nothing_ends_the_growing_and_one_that_fails_leaves_the_world_built(client, building):
    building.expand = (200, {"places": [], "people": []})
    s = built(client, "large")
    assert s["error"] is None and s["result"]["grown"] == {"rings": 1, "places": 0, "people": 0, "stopped": "a ring added nothing"}
    assert len([c for c in building.calls if c[1].endswith("/world/expand")]) == 1                          # not seven
    building.calls.clear()
    building.expand = (502, {"detail": "the seed could not be planted: truncated"})
    s = built(client, "medium")
    assert s["done"] and s["error"] is None and s["result"]["port"] == 8230                                  # the world is built; the growing says why it stopped
    assert s["result"]["grown"]["rings"] == 0 and "ring 1: (502)" in s["result"]["grown"]["error"]


def test_wide_is_now_small_for_the_builder_and_an_older_page_or_model_that_says_wide_is_understood(client, builder):
    assert sthapati.BREADTHS == ("here", "around", "small", "medium", "large")
    assert sthapati.clean_draft({"breadth": "wide"})["breadth"] == "small"
    assert sthapati.clean_patch({"breadth": "wide"}, {"breadth": "here"}) == {"breadth": "small"}
    assert sthapati.clean_patch({"breadth": "medium"}, {"breadth": "here"}) == {"breadth": "medium"}
    assert sthapati.clean_patch({"breadth": "enormous"}, {"breadth": "here"}) == {}
    assert sthapati.TURN_TOOL["properties"]["patch"]["properties"]["breadth"]["enum"] == list(sthapati.BREADTHS)
    for word in ('"small"', '"medium"', '"large"', "about 120 places"):
        assert word in sthapati.TURN_SYSTEM, word
    assert '"wide"' not in sthapati.TURN_SYSTEM


def test_the_page_offers_the_sizes_small_medium_and_large(client):
    html = (sthapati.Path(__file__).resolve().parent.parent / "src" / "smritilila" / "web" / "index.html").read_text(encoding="utf-8")
    for value in ("here", "around", "small", "medium", "large"):
        assert f'<option value="{value}"' in html, value
    assert '<option value="wide"' not in html and 'value="small" selected' in html                          # small is the one it opens on, as wide was


# -- Browsing for an image (Marc, 2026-10-23: "Should allow file browsing for images"): the file chosen on the owner's machine is kept with the saved maps ---

UPLOADED = "uploaded"


def real_png(extra=b""):
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"0" * 40 + extra


def upload(client, name, data, headers=OWNER):
    return client.post("/builder/maps", params={"name": name}, content=data, headers={**headers, "Content-Type": "application/octet-stream"})


def test_a_browsed_image_is_kept_with_the_saved_maps_and_listed_and_served(client, maps):
    r = upload(client, "My Atlas.png", real_png())
    assert r.status_code == 201, r.text
    assert r.json() == {"id": "uploaded/My Atlas.png", "folder": "uploaded", "name": "My Atlas.png", "bytes": len(real_png())}
    assert (maps / "uploaded" / "My Atlas.png").read_bytes() == real_png()
    assert "uploaded/My Atlas.png" in [m["id"] for m in client.get("/builder/maps", headers=OWNER).json()]
    got = client.get("/builder/maps/file", params={"id": "uploaded/My Atlas.png"}, headers=OWNER)
    assert got.status_code == 200 and got.content == real_png() and got.headers["content-type"] == "image/png"


def test_an_image_is_kept_for_what_it_is_not_for_what_it_is_called(client, maps):
    jpeg, gif, webp = b"\xff\xd8\xff\xe0" + b"1" * 30, b"GIF89a" + b"2" * 30, b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"3" * 30
    assert upload(client, "photo.png", jpeg).json()["name"] == "photo.jpg"                                  # a jpeg called a png is kept as a jpeg
    assert upload(client, "a.bmp", gif).json()["name"] == "a.gif" and upload(client, "b", webp).json()["name"] == "b.webp"
    for name, body in (("notes.png", b"not an image at all"), ("script.png", b"<script>alert(1)</script>"), ("empty.png", b"")):
        r = upload(client, name, body)
        assert r.status_code == 422 and "not a png, jpeg, gif or webp" in r.json()["detail"], name
    assert sorted(p.name for p in (maps / "uploaded").iterdir()) == ["a.gif", "b.webp", "photo.jpg"]          # nothing else was kept


def test_a_browsed_name_cannot_reach_another_folder_and_a_clash_keeps_both(client, maps):
    before = {p for p in maps.parent.rglob("*") if p.is_file()}
    for name in ("../../evil.png", "..\\..\\evil.png", "/etc/passwd.png", "dir/inner.png"):
        r = upload(client, name, real_png(name.encode()))
        assert r.status_code == 201, name
        assert r.json()["folder"] == "uploaded" and "/" not in r.json()["name"] and ".." not in r.json()["name"], r.json()
    written = {p for p in maps.parent.rglob("*") if p.is_file()} - before
    assert written and all((maps / UPLOADED) in p.parents for p in written)                                   # nothing at all was written outside maps/uploaded
    first, same, other = upload(client, "atlas.png", real_png()), upload(client, "atlas.png", real_png()), upload(client, "atlas.png", real_png(b"different"))
    assert first.json()["name"] == "atlas.png" and same.json()["name"] == "atlas.png" and other.json()["name"] == "atlas-2.png"   # the same bytes once; another beside it
    assert (maps / "uploaded" / "atlas.png").read_bytes() == real_png() and (maps / "uploaded" / "atlas-2.png").read_bytes() == real_png(b"different")


def test_a_browsed_image_has_a_size_limit_and_the_folder_a_count_limit(client, maps, monkeypatch):
    monkeypatch.setattr(sthapati, "MAX_MAP_BYTES", 100)
    assert upload(client, "big.png", real_png(b"x" * 200)).status_code == 413 and not (maps / "uploaded").exists()
    assert upload(client, "small.png", real_png()).status_code == 201
    monkeypatch.setattr(sthapati, "MAX_UPLOADED", 2)
    assert upload(client, "two.png", real_png(b"2")).status_code == 201
    assert upload(client, "three.png", real_png(b"3")).status_code == 409                                    # full: a new one is refused ...
    assert upload(client, "two.png", real_png(b"2")).status_code == 201                                      # ... one already kept is not a new file


def test_the_size_limit_holds_for_a_body_that_does_not_say_how_long_it_is(client, maps, monkeypatch):
    monkeypatch.setattr(sthapati, "MAX_MAP_BYTES", 100)
    pieces = (real_png()[:30] if i == 0 else b"y" * 40 for i in range(5))                                    # sent in pieces, with no Content-Length
    r = client.post("/builder/maps", params={"name": "streamed.png"}, content=pieces, headers=OWNER)
    assert r.status_code == 413 and not (maps / "uploaded").exists()


def test_only_the_owner_or_an_enabled_user_may_browse_an_image_in(client, maps, registered):
    assert upload(client, "a.png", real_png(), headers=GUEST).status_code == 403
    assert upload(client, "a.png", real_png(), headers=OTHER).status_code == 403
    assert not (maps / "uploaded").exists()
    assert upload(client, "a.png", real_png(), headers=USER).status_code == 201


def test_the_page_can_browse_for_an_image(client):
    html = (sthapati.Path(__file__).resolve().parent.parent / "src" / "smritilila" / "web" / "index.html").read_text(encoding="utf-8")
    for needle in ('id="nw-map-browse"', 'id="nw-map-file"', 'accept="image/png,image/jpeg,image/webp,image/gif"', 'API + "/builder/maps?name="', "application/octet-stream"):
        assert needle in html, needle
    assert html.count("started before they were added") == 2                                                 # a server that predates maps says so, in the picker and on a browse
