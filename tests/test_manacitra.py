# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""smrti/manacitra.py and POST /world/manacitra -- the builder reads a map, and what it reads becomes a pratiṣṭhā seed (2026-10-21)."""

from __future__ import annotations

import io
import re

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from smritilila import adhisthana
from smritilila.adhisthana import WorldSession
from smritilila.graph.entity import Entity
from smritilila.llm.client import Picture
from smritilila.routes import pratistha as pratistha_routes
from smritilila.smrti import manacitra as M
from smritilila.smrti.kala import ITIHASA
from smritilila.smrti.pratistha import check, plant
from test_main import client, fake_store  # noqa: F401 -- the shared fixtures
from test_pratistha import SPEC, _kosha, _world

KNOWN = [("land", "the river country", "region"), ("inn", "the inn", "scene"), ("fen", "the black fen", "region")]


def _sheet(w: int = 400, h: int = 300) -> Picture:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return Picture(buf.getvalue(), "image/png", "sheet.png")


def _row(name, *, inside="", known="", scene=True, partial=False, text="Somewhere on the map.", kind="village"):
    return {"name": name, "kind": kind, "in": inside, "known": known, "scene": scene, "partial": partial, "tuple": text}


def test_the_parts_cover_the_sheet_and_each_overlaps_its_neighbour() -> None:
    boxes = dict(M.tile_boxes(2262, 1646, 3, 3, 0.2))
    assert len(boxes) == 9 and boxes["r1c1"][:2] == (0, 0) and boxes["r3c3"][2:] == (2262, 1646)
    assert all(0 <= l < r <= 2262 and 0 <= t < b <= 1646 for l, t, r, b in boxes.values())
    width = boxes["r1c1"][2] - boxes["r1c1"][0]
    assert boxes["r1c1"][2] - boxes["r1c2"][0] >= 0.19 * width and boxes["r1c2"][2] - boxes["r1c3"][0] >= 0.19 * width   # a name that wide lies whole in one of two
    assert all(any(l <= x < r for l, _, r, _ in boxes.values()) for x in range(0, 2262, 37))                              # no column of the sheet is unread
    one = M.tile_boxes(100, 80, 1, 1, 0.2)
    assert one == [("r1c1", (0, 0, 100, 80))]


def test_a_printed_name_is_the_place_the_world_holds_by_its_bare_words() -> None:
    known = [("v", "Village of Oldford", "region"), ("h", "Hale Downs Northern Verge", "region"), ("m", "the ruined mill", "scene"),
             ("c", "City of Hale", "region")]
    assert M.match_known("Oldford", known) == ["v"]
    assert M.match_known("Hale Downs", known) == ["h"]            # two words that open a longer label
    assert M.match_known("Hale", known) == ["c"]                  # one word is the city: never a prefix of the Downs
    assert M.match_known("Mill", known) == []                     # nor is a word inside a label the label
    assert M.match_known("Ôldford!", known) == ["v"] and M.norm("Osnabrölt") == "osnabrolt"


def test_what_the_parts_read_is_made_one_seed_that_stands_and_every_choice_is_said() -> None:
    canon, mems, recs = _world()
    canon.entities["earth"] = Entity("earth", ["the world at large"], 0, ITIHASA)
    areas = [{"name": "Greywater Forest", "kind": "forest", "known": "", "tuple": "A great forest in the north of the sheet."},
             {"name": "The Black Fen", "kind": "fen", "known": "", "tuple": "A fen in the south."},       # the world holds it already
             {"name": "IRONWOOD", "kind": "wood", "known": "", "tuple": "A wood in the west."}]           # read again, spaced, by a part
    known = KNOWN + [("wharf", "Greywater River Wharf", "scene")]
    parts = {
        "r1c1": [_row("Tulham", inside="Greywater Forest", text="A village at the forest's edge."),
                 _row("Wayside Inn", known="inn"),                                                          # the world's own, by the id the model gave
                 _row("The Greywater", inside="Greywater Forest", scene=False, kind="river", text="A river through the forest."),
                 _row("Stoneb", inside="Tulham", partial=True, text=""),                                     # cut by the edge, but a piece of a name read whole
                 _row("Quarrybo", inside="Tulham", partial=True, text=""),                                   # cut by the edge, never read whole
                 _row("Tulhan", inside="Greywater Forest"),                                                  # a letter off a name read twice: the once-read one goes
                 _row("Deepdelve", inside="Greywater Forest", known="inn"),                                  # the model says it is the inn: it shares no word with it
                 _row("Greywater River", inside="Greywater Forest", scene=False, kind="river"),              # not the wharf on it: a scene is not what it lies on
                 _row("Greywater Wharf Road", inside="Greywater Forest", kind="road"),
                 _row("IRON WOOD", inside="", scene=False, kind="forest")],
        "r1c2": [_row("Tulham", inside="Greywater Forest", text="A village at the forest's edge, with a mill and a ford on the Greywater."),
                 _row("Stonebridge", inside="Tulham", kind="bridge"),
                 _row("Stonebridqe", inside="Tulham", kind="bridge"),                                        # a letter off: flagged, both kept
                 _row("Lostwick", inside=""),                                                                # no area: filed in the earth, and said
                 _row("Ashby", inside="Brook End"), _row("Brook End", inside="Ashby"),                       # a loop of containers
                 _row("Greywater Forest", scene=False, text="again"),                                        # an area already read
                 _row("Mill Lane", inside="inn", kind="road"),                                               # in a place the world holds, by id
                 _row("Reedmarsh", inside="Black Fen", scene=False, kind="swamp")],                          # in one the world holds, by name
    }
    spec, report = M.build_spec(areas, parts, known, set(canon.entities), "earth", "sheet.png")
    by = {p["label"]: p for p in spec["places"]}
    assert "Wayside Inn" not in by and "The Black Fen" not in by and {h["name"] for h in report["already_held"]} == {"Wayside Inn", "The Black Fen"}
    assert {h["name"]: h["by"] for h in report["already_held"]} == {"Wayside Inn": "the model's word", "The Black Fen": "the same words"}
    assert "Deepdelve" in by and report["claimed_held_but_not"] == [{"name": "Deepdelve", "claimed": "inn", "label": "the inn"}]   # said, and made
    assert "IRON WOOD" not in by and "IRONWOOD" in by and by["IRONWOOD"]["in"] == "earth"                  # one wood, however it is spaced
    assert report["misreads_resolved"] == [{"kept": "Tulham", "dropped": "Tulhan"}] and "Tulhan" not in by
    assert "Greywater River" in by and not any(h["name"] == "Greywater River" for h in report["already_held"])
    assert by["Greywater Forest"]["in"] == "earth" and by["Greywater Forest"]["scene"] is False
    assert by["Tulham"]["tuple"].endswith("a mill and a ford on the Greywater.")                             # the fullest account of a name
    assert by["Tulham"]["in"] == by["Greywater Forest"]["key"] and by["Stonebridge"]["in"] == by["Tulham"]["key"]
    assert not any(p["scene"] for p in spec["places"])                                                       # every place is a region: none is a neighbour of another
    assert by["Lostwick"]["in"] == "earth" and report["unplaced"] == ["Lostwick"]
    assert by["Mill Lane"]["in"] == "inn" and by["Reedmarsh"]["in"] == "fen"
    assert report["cycles_broken"] == 1 and sorted(by[x]["in"] for x in ("Ashby", "Brook End")).count("earth") == 1
    assert report["cut_and_never_whole"] == ["Quarrybo"] and "Quarrybo" not in by and "Stoneb" not in by       # a name is never completed
    assert ["Stonebridge", "Stonebridqe"] in report["possible_misreads"]
    assert [p["label"] for p in spec["places"]].count("Greywater Forest") == 1
    assert spec["map"] == {"image": "sheet.png", "parts": 2} and spec["people"] == [] and not any("_kind" in p for p in spec["places"])
    keys = [p["key"] for p in spec["places"]]
    assert len(keys) == len(set(keys)) and not set(keys) & set(canon.entities)
    errors, _warnings = check(spec, _kosha(canon, mems, recs), {"teo"}, 11)
    assert errors == []                                                                                      # the seed stands as pratiṣṭhā asks
    ents, _mems, new_recs, ids = plant(spec, 11, None, "proof", lambda text: None)
    assert len(ents) == len(spec["places"]) and {r.kind for r in new_recs} == {"kosha", "name"}


def test_a_name_read_once_among_several_readings_is_listed_and_not_made() -> None:
    """A world is only ever added to: a place with a misread name cannot be taken back. So with `min_votes`, a name needs that many readings
    -- as an area or in a part, the two counted together -- and one read less often is listed for a god to look at."""
    canon, _mems, _recs = _world()
    areas = [{"name": "Greywater Forest", "kind": "forest", "known": "", "tuple": "x"}]
    parts = {"r1c1": [_row("Tulham", inside="Greywater Forest"), _row("Quarrow", inside="Greywater Forest"), _row("Greywater Forest", scene=False)],
             "r1c2": [_row("Tulham", inside="Greywater Forest")]}
    spec, report = M.build_spec(areas, parts, KNOWN, set(canon.entities), "land", "s.png", min_votes=2)
    assert [p["label"] for p in spec["places"]] == ["Greywater Forest", "Tulham"] and report["read_too_seldom"] == ["Quarrow"]
    spec, report = M.build_spec(areas, parts, KNOWN, set(canon.entities), "land", "s.png")                  # by default every reading counts
    assert "Quarrow" in [p["label"] for p in spec["places"]] and report["read_too_seldom"] == []


def test_a_maps_seed_may_be_large_and_holds_places_only_while_any_other_seed_stays_small() -> None:
    canon, mems, recs = _world()
    k = _kosha(canon, mems, recs)
    places = [{"key": f"p{i}", "label": f"Place {i}", "name": f"Place {i}", "in": "land", "scene": True, "tuple": f"Place {i} lies by the road."}
              for i in range(60)]
    big = {"title": "t", "why": "w", "places": places, "people": [], "things": []}
    assert any("too large" in e for e in check(big, k, {"teo"}, 11)[0])
    big["map"] = {"image": "x"}
    assert check(big, k, {"teo"}, 11)[0] == []
    big["people"] = SPEC["people"]
    assert any("places only" in e for e in check(big, k, {"teo"}, 11)[0])


class _Reader:
    """A fake model that reads a sheet: the areas once, then one village in each part, by the part the prompt names."""

    def __init__(self, broken: str | None = None, slip: tuple[str, str] | None = None) -> None:
        self.calls: list[tuple[str, str, list]] = []
        self.broken = broken
        self.slip = slip            # (the name the part reads, the spelling it slips into once, on its first reading)
        self.read_once = False
        import threading
        self._lock = threading.Lock()

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=None, images=None):
        self.calls.append((tool_name, user, list(images or [])))
        if tool_name == self.broken:
            raise RuntimeError("the model is down")
        if tool_name == "read_areas":
            return {"areas": [{"name": "Greywater Forest", "kind": "forest", "known": "", "tuple": "A forest."}]}
        r, c = re.search(r"row (\d) of \d.*column (\d) of \d", user).groups()
        name = f"Place {r}{c}"
        if self.slip and name == "Place 11":
            with self._lock:
                name, self.read_once = (self.slip[1] if not self.read_once else self.slip[0]), True
        return {"features": [_row(name, inside="Greywater Forest")]}


def test_the_whole_sheet_is_read_once_then_every_part_and_each_call_carries_its_picture() -> None:
    canon, mems, recs = _world()
    reader = _Reader()
    sheet = _sheet(400, 300)
    spec, report, readings = M.read_map(reader, "A world of reeds", sheet, KNOWN, set(canon.entities), "land", cols=2, rows=2, passes=1)
    assert readings["areas"][0]["name"] == "Greywater Forest" and sorted(readings["parts"]) == ["r1c1", "r1c2", "r2c1", "r2c2"]    # what was read, unedited
    assert [c[0] for c in reader.calls].count("read_areas") == 1 and [c[0] for c in reader.calls].count("read_part") == 4
    assert all(len(c[2]) == 1 and c[2][0].mime == "image/png" for c in reader.calls)                         # no call goes without its picture
    sizes = {Image.open(io.BytesIO(c[2][0].data)).size for c in reader.calls if c[0] == "read_part"}
    assert all(w < 400 and h < 300 for w, h in sizes) and Image.open(io.BytesIO(reader.calls[0][2][0].data)).size == (400, 300)
    assert all("A world of reeds" in c[1] for c in reader.calls) and any("row 2 of 2 (the bottom), column 1 of 2 (the left)" in c[1] for c in reader.calls)
    assert sorted(p["label"] for p in spec["places"]) == ["Greywater Forest", "Place 11", "Place 12", "Place 21", "Place 22"]
    assert report["parts"] == {"r1c1": 1, "r1c2": 1, "r2c1": 1, "r2c2": 1}


def test_the_sheet_is_read_in_passes_and_a_spelling_read_once_loses_to_the_one_read_more_often() -> None:
    """A model's reading of small print wanders (Stonemarch's map, 2026-10-21: 'Culvar' for Tulvar in one run, 'Mastyne' for Mastryne in
    another): so the sheet is read three times, and a name read once beside a neighbour spelling read more often is dropped."""
    canon, mems, recs = _world()
    reader = _Reader(slip=("Place 11", "Place 1l"))                # the first reading of one part slips a letter
    spec, report, readings = M.read_map(reader, None, _sheet(), KNOWN, set(canon.entities), "land", cols=2, rows=2, passes=3)
    names = [c[0] for c in reader.calls]
    assert names.count("read_areas") == 3 and names.count("read_part") == 12
    assert sorted(readings["parts"]) == sorted(f"r{r}c{c}{s}" for r in (1, 2) for c in (1, 2) for s in ("", "#2", "#3"))
    assert report["misreads_resolved"] == [{"kept": "Place 11", "dropped": "Place 1l"}] and report["passes"] == 3
    assert sorted(p["label"] for p in spec["places"]) == ["Greywater Forest", "Place 11", "Place 12", "Place 21", "Place 22"]   # the areas of the passes, once


def test_an_abbreviation_is_the_place_it_abbreviates() -> None:
    # invented words, so that the export's scrub of the world's own names cannot change what the initials are
    assert M.is_acronym_of("OMH", "Ruins of the Old Mill House") and M.is_acronym_of("omh", "the Old Mill House")
    assert not M.is_acronym_of("OMH", "Old House") and not M.is_acronym_of("Tomb", "Tomb of Old Bones") and not M.is_acronym_of("O M H", "Old Mill House")


def test_a_part_that_cannot_be_read_stops_the_whole_read_rather_than_leave_a_hole() -> None:
    reader = _Reader(broken="read_part")
    with pytest.raises(M.MapReadFailed):
        M.read_map(reader, None, _sheet(), KNOWN, {"land"}, "land", cols=2, rows=1, passes=1)
    assert [c[0] for c in reader.calls].count("read_part") >= M.READ_ATTEMPTS                                # tried again before it gave up


def test_a_gif_is_made_a_png_and_png_and_jpeg_are_sent_as_they_are(tmp_path) -> None:
    buf = io.BytesIO()
    Image.new("P", (20, 10)).save(buf, "GIF")
    (tmp_path / "old.gif").write_bytes(buf.getvalue())
    (tmp_path / "a.png").write_bytes(_sheet().data)
    gif, png = M.load_picture(tmp_path / "old.gif"), M.load_picture(tmp_path / "a.png")
    assert gif.mime == "image/png" and gif.name == "old.png" and Image.open(io.BytesIO(gif.data)).size == (20, 10)
    assert png.mime == "image/png" and png.data == (tmp_path / "a.png").read_bytes()


def test_a_map_is_proposed_and_proved_not_planted_and_only_a_file_in_the_worlds_maps_folder_is_read(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                                                                                  fake_store, tmp_path) -> None:
    canon, mems, recs = _world()
    canon.entities["earth"] = Entity("earth", ["the world at large"], 0, ITIHASA)
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.recent_memories, world.records, world.name = list(mems), list(recs), "testworld"
    client.app.state.world = world
    reader = _Reader()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: adhisthana.GenerativeRoles(smrti_client=reader))
    monkeypatch.setattr(pratistha_routes, "PROPOSALS_DIR", tmp_path / "proposals")
    monkeypatch.setattr(pratistha_routes, "MAPS_DIR", tmp_path / "maps")
    (tmp_path / "maps" / "testworld").mkdir(parents=True)
    (tmp_path / "maps" / "testworld" / "sheet.png").write_bytes(_sheet().data)
    assert client.post("/world/manacitra", json={"image": "../sheet.png"}).status_code == 422                 # a name, never a path
    assert client.post("/world/manacitra", json={"image": "nothing.png"}).status_code == 404
    r = client.post("/world/manacitra", json={"image": "sheet.png", "cols": 2, "rows": 1})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["status"] == "proposed" and p["errors"] == [] and p["report"]["areas_read"] == 1 and p["title"] == "The map sheet.png"
    assert any("Place 11" in line for line in p["summary"]) and fake_store.commit_calls == []              # nothing planted
    assert [x["id"] for x in client.get("/world/pratistha").json()] == [p["id"]]


def test_a_guest_cannot_start_a_map_reading_but_the_owner_and_the_worlds_god_can(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                                                                 fake_store, tmp_path) -> None:
    """The route says "Owner and the world's gods", and the gate had no line for it: a guest on a guest-open world (Stonemarch) could start a
    map reading -- several vision-model calls -- and write a proposal file (found 2026-10-22, reading the comments against the code). Refused
    before the route runs, so the reader is never called; a god of the world and this machine pass the gate (and meet "no such map")."""
    canon, mems, recs = _world()
    canon.entities["earth"] = Entity("earth", ["the world at large"], 0, ITIHASA)
    world = WorldSession(canon=canon, scene_owner_entity_id="teo", entry_point_location_ids=["inn"], playable_entity_ids=["teo"])
    world.recent_memories, world.records, world.name = list(mems), list(recs), "testworld"
    client.app.state.world = world
    reader = _Reader()
    monkeypatch.setattr(adhisthana, "_resolve_roles", lambda: adhisthana.GenerativeRoles(smrti_client=reader))
    monkeypatch.setattr(pratistha_routes, "PROPOSALS_DIR", tmp_path / "proposals")
    monkeypatch.setattr(pratistha_routes, "MAPS_DIR", tmp_path / "maps")
    (tmp_path / "maps" / "testworld").mkdir(parents=True)
    (tmp_path / "maps" / "testworld" / "sheet.png").write_bytes(_sheet().data)
    entry = type("W", (), {"name": "Stonemarch", "database_url": "g", "gods": ("god@example.com",), "port": None, "description": None})()
    monkeypatch.setattr(adhisthana, "_active_world_registry_entry", lambda: entry)
    monkeypatch.setattr(adhisthana, "get_worlds_registry", lambda: [entry])
    guest = {"Cf-Access-Authenticated-User-Email": "friend@example.com", "Cf-Ray": "x"}
    god = {"Cf-Access-Authenticated-User-Email": "god@example.com", "Cf-Ray": "x"}
    r = client.post("/world/manacitra", json={"image": "sheet.png", "cols": 2, "rows": 1}, headers=guest)
    assert r.status_code == 403 and r.json()["detail"] == "only the world's owner may do that"             # the owner-only refusal, not "world closed"
    assert reader.calls == [] and not (tmp_path / "proposals").exists()                                    # no model call, no file
    assert client.post("/world/manacitra", json={"image": "nothing.png"}, headers=god).status_code == 404  # past the gate: the map is not there
    assert client.post("/world/manacitra", json={"image": "nothing.png"}).status_code == 404                # this machine
