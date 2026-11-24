# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Ābhāsa (Marc, 2026-11-11: "image generation and presentation"): the pictures of a world -- what is made after a telling and when, a player's portrait drafted
from how they look and changed by asking until they keep it, who sees what, the two boxes a newcomer fills, and the page that shows them. The model that paints is
a fake that returns a real small JPEG, and the writer that judges whether a being has a face is a fake that answers by name; the real painter was met on Marc's
key on the day it was built (notes in the TODO)."""

from __future__ import annotations

import asyncio
import io
import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from smritilila import abhasa
from smritilila.adhisthana import WorldSession, extend_history
from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.llm.client import Picture
from smritilila.memory.chitra import Chitra
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.schemas import JoinRequest
from smritilila.world_store import InMemoryWorldStore
from smritilila import worlds_registry
from test_main import client, fake_store  # noqa: F401 -- the shared fixtures


def _jpeg(width: int, height: int, color=(120, 90, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "JPEG")
    return buf.getvalue()


class _Painter:
    """Records every request; makes a picture of the shape asked for, its colour the number of pictures made so far (so two are never alike).
    `refuse`: answers None (the model's filter) for a prompt holding the word."""
    model = "fake-painter"

    def __init__(self, refuse: str | None = None, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.refuse, self.fail = refuse, fail

    def paint(self, prompt: str, *, seed: Picture | None = None, aspect: str = "16:9"):
        self.calls.append({"prompt": prompt, "seed": seed, "aspect": aspect})
        if self.fail:
            raise RuntimeError("the service is down")
        if self.refuse and self.refuse in prompt:
            return None
        shade = (len(self.calls) * 20 % 255, 90, 60)
        return Picture(_jpeg(1376, 768, shade) if aspect == "16:9" else _jpeg(896, 1200, shade), "image/jpeg")


class _Writer:
    """The small model that says whether a being has a face: answers by the name it is shown."""

    def __init__(self, faces: dict[str, str]) -> None:
        self.faces, self.asked, self.users, self.systems = faces, [], [], []

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=500):
        name = user.split("\n", 1)[0].removeprefix("NAME: ")
        self.asked.append(name)
        self.users.append(user)
        self.systems.append(system)
        look = self.faces.get(name, "")
        return {"being": bool(look), "look": look}


class _Roles:
    def __init__(self, writer) -> None:
        self.smrti_client = writer


def _m(mid: str, who: str, text: str, tick: int, subjects=(), salience: int = 4, loc: str = "square", pid: str | None = None, tier: str = "PRIMARY") -> Memory:
    return Memory(mid, who, frozenset(subjects), text, tier, salience, loc, tick, pid or f"p{tick}", None)


def _world(store=None) -> WorldSession:
    """Two players (the hero and a friend who lives at the cellar), a being (Mara), a prop (a key), two places and a region."""
    canon = Maya(world_time=10, entities={
        "hero": Entity("hero", ["Hero"], 0, "genesis"), "friend": Entity("friend", ["Friend"], 0, "genesis"),
        "mara": Entity("mara", ["Mara the Ferrywoman"], 0, "genesis", kosha="square"), "key": Entity("key", ["an iron key"], 0, "genesis"),
        "square": Entity("square", ["Town Square"], 0, "genesis", avarana="scene"), "cellar": Entity("cellar", ["The Cellar"], 0, "genesis", avarana="scene"),
        "region": Entity("region", ["The Marches"], 0, "genesis"),
    })
    world = WorldSession(canon=canon, scene_owner_entity_id="hero", entry_point_location_ids=["square"], playable_entity_ids=["hero", "friend"])
    world.recent_memories = [
        _m("sq0", "square", "A town square, the lane running off it, a well at its middle.", 0, tier="AMBIENT", pid="g"),
        _m("h0", "hero", "I came to the square in a grey wool cloak, a staff in my hand.", 0, salience=2, pid="g"),
        _m("f0", "friend", "I keep the stall by the cellar door.", 0, salience=2, loc="cellar", pid="g2"),
    ]
    world.store = store if store is not None else InMemoryWorldStore()
    world.name = None
    return world


def _arm(monkeypatch: pytest.MonkeyPatch, painter: _Painter, **cfg) -> None:
    monkeypatch.setattr(abhasa, "settings", lambda world: dict(cfg))
    monkeypatch.setattr(abhasa, "painter_of", lambda: painter)


def _portrait(world: WorldSession, pid: str, shade=(100, 100, 140)) -> None:
    """A player with a portrait they have accepted."""
    st = asyncio.run(abhasa.ensure(world))
    asyncio.run(abhasa._keep(world, st, Chitra(f"portrait-{pid}", pid, None, 10, "how they look", "fake-painter", "image/jpeg", _jpeg(300, 400, shade),
                                              "accepted by the player after 0 changes")))


def _tell(world: WorldSession, roles, tuples: list[Memory], actor: str = "hero", scene: str | None = "square") -> None:
    """A committed telling: its tuples are in the history, and its pictures are made (the work the commit starts in the background, awaited)."""
    extend_history(world, tuples)
    told: dict[str, list[Memory]] = {}
    for m in tuples:
        if m.tier == "PRIMARY" and m.entity_id in world.playable_entity_ids:
            told.setdefault(m.entity_id, []).append(m)
    asyncio.run(abhasa._illustrate(world, roles, abhasa.settings(world), abhasa.painter_of(), told, {actor: scene}, tuples[0].pulse_id))


def _kept(world: WorldSession) -> list[Chitra]:
    return list(asyncio.run(world.store.load_chitra_index()))


LOOK_HERE = "Tall and weathered, a grey cloak."


# -- the words and the pixels, pure ---------------------------------------------------------------------------------------------------------

def test_a_picture_is_kept_as_a_jpeg_no_larger_than_the_cap_and_one_that_is_not_an_image_is_kept_as_it_came() -> None:
    data, mime = abhasa.shrink(_jpeg(3000, 1800), "image/png")
    im = Image.open(io.BytesIO(data))
    assert mime == "image/jpeg" and max(im.size) == abhasa.MAX_SIDE and im.size == (1280, 768)
    small, _ = abhasa.shrink(_jpeg(400, 300), "image/jpeg")
    assert Image.open(io.BytesIO(small)).size == (400, 300)
    assert abhasa.shrink(b"not an image", "image/webp") == (b"not an image", "image/webp")


def test_portraits_are_appended_into_one_image_side_by_side_at_one_height() -> None:
    one = abhasa.sheet([_jpeg(300, 400), _jpeg(600, 400), _jpeg(200, 400)], height=400)
    im = Image.open(io.BytesIO(one.data))
    assert one.mime == "image/jpeg" and im.size == (300 + 600 + 200, 400)


def test_the_prompts_carry_what_was_given_and_never_ask_for_a_face_a_place_does_not_have() -> None:
    style = abhasa.STYLE
    assert "No text, lettering" in style
    p = abhasa.portrait_prompt(style, "Mara", "A broad woman of fifty, a river-grey braid, a ferryman's oilskins.", [], "I ferry people.")
    assert "Mara" in p and "river-grey braid" in p and "head and shoulders" in p and "I ferry people." in p
    q = abhasa.portrait_prompt(style, "Mara", None, ["Mara poled the ferry across, her braid wet."])
    assert "What is known of them" in q and "her braid wet" in q
    place = abhasa.place_prompt(style, "Town Square", ["A well at its middle."])
    assert "empty of the people the accounts mention" in place and "no one in the foreground" in place and "Town Square" in place and "A well at its middle." in place
    assert "the people in these words are not to be shown" in place
    bare = abhasa.place_prompt(style, "Town Square", [])
    assert "Town Square" in bare and "What is known of the place" not in bare and bare.endswith("Town Square.")   # told nothing of it: its name alone
    moment = abhasa.moment_prompt(style, ["Hero", "Mara"], "I fell on the ice.")
    assert "exactly as in their reference portrait" in moment and "Hero, Mara" in moment and "I fell on the ice." in moment
    assert "reference portrait" not in abhasa.moment_prompt(style, [], "A storm broke.")
    assert abhasa.sheet_line(["Hero", "Mara"]).endswith("left to right: Hero, Mara. ")
    edit = abhasa.edit_prompt(style, "Hero", "Tall, grey cloak.", "make my hair white")
    assert "nothing else" in edit and "make my hair white" in edit and "Tall, grey cloak." in edit and "Hero" in edit


def test_the_judge_is_asked_the_name_and_the_accounts_and_answers_whether_it_has_a_face() -> None:
    w = _Writer({"Mara": "A broad woman of fifty."})
    assert abhasa.judge(w, "Mara", ["She poled us across.", "Her braid was wet."]) == {"being": True, "look": "A broad woman of fifty."}
    assert abhasa.judge(w, "an iron key", ["I took the key."]) == {"being": False, "look": ""}
    assert w.asked == ["Mara", "an iron key"]


def test_a_beings_fixed_points_are_the_same_every_time_and_spread_over_beings_so_vague_accounts_do_not_all_get_one_face() -> None:
    """Marc, 2026-11-11: the Austin bartender, whose accounts say nothing of their look, was painted with Charisse's face. The model's habitual face for a
    vague account is one face, and independent judge calls fall into one archetype as well, so the points that decide a face come from the id."""
    assert abhasa.fixed_points("mara") == abhasa.fixed_points("mara")
    ids = [f"being_{i}_{i * 7919:06x}" for i in range(300)]
    seen = {(abhasa._pick(e, "build", abhasa.LOOK_BUILD), abhasa._pick(e, "complexion", abhasa.LOOK_COMPLEXION), abhasa._pick(e, "hair", abhasa.LOOK_HAIR),
             abhasa._pick(e, "face", abhasa.LOOK_FACE)) for e in ids}
    assert len(seen) >= 280                                              # 300 beings, at most a few alike in all four points
    for salt, options in (("build", abhasa.LOOK_BUILD), ("complexion", abhasa.LOOK_COMPLEXION), ("hair", abhasa.LOOK_HAIR), ("face", abhasa.LOOK_FACE)):
        assert {abhasa._pick(e, salt, options) for e in ids} == set(options)      # every option is used, none is dead
    assert not any("bald" in h for h in abhasa.LOOK_HAIR)                # a drawn point must not contradict a youth or a woman as "bald" did in the replay
    for e in ("mara", "the_bartender_cf257d"):                           # what is handed over is exactly the picks, named
        p = abhasa.fixed_points(e)
        assert abhasa._pick(e, "build", abhasa.LOOK_BUILD) in p and abhasa._pick(e, "hair", abhasa.LOOK_HAIR) in p and abhasa._pick(e, "face", abhasa.LOOK_FACE) in p
    assert "sex" in abhasa.JUDGE_SYSTEM and "COMMIT" in abhasa.JUDGE_SYSTEM and "FIXED POINTS" in abhasa.JUDGE_SYSTEM and "never write around the sex" in abhasa.JUDGE_SYSTEM


def test_the_judge_is_handed_the_world_and_the_beings_fixed_points_only_when_it_has_them_and_the_name_stays_first() -> None:
    w = _Writer({"Mara": "A broad woman of fifty."})
    abhasa.judge(w, "Mara", ["She poled us across."])
    assert w.users[0] == "NAME: Mara\n\nACCOUNTS:\n1. She poled us across."          # the old call, unchanged: nothing added for a caller that gives none
    abhasa.judge(w, "Mara", ["She poled us across."], eid="mara", setting="A river town in 1928.")
    user = w.users[1]
    assert user.startswith("NAME: Mara\n") and "THE WORLD: A river town in 1928." in user and abhasa.fixed_points("mara") in user
    assert user.index("ACCOUNTS:") < user.index("THE WORLD:") < user.index("FIXED POINTS")
    assert w.asked == ["Mara", "Mara"]


def test_the_registry_reads_a_worlds_picture_setting_and_a_typo_never_turns_spending_on(tmp_path: Path) -> None:
    path = tmp_path / "worlds.json"
    path.write_text(json.dumps({"worlds": [
        {"name": "A", "database_url": "x", "pictures": True}, {"name": "B", "database_url": "y", "pictures": {"daily": 5, "style": "ink"}},
        {"name": "C", "database_url": "z"}, {"name": "D", "database_url": "w", "pictures": "yes"}, {"name": "E", "database_url": "v", "pictures": 1}]}))
    got = {w.name: w.pictures for w in worlds_registry.load_worlds_registry(path)}
    assert got == {"A": {}, "B": {"daily": 5, "style": "ink"}, "C": None, "D": None, "E": None}


# -- what is made, and when --------------------------------------------------------------------------------------------------------------

def test_a_registry_that_cannot_be_read_turns_pictures_off_and_never_fails_a_telling_or_an_arrival(monkeypatch: pytest.MonkeyPatch) -> None:
    """Met on the first real join (2026-11-11): worlds.json came back with a byte-order mark, the settings read raised inside the join's own request, and the
    arrival of a newcomer already committed failed with it. The hooks answer to nothing but themselves."""
    from smritilila import adhisthana
    painter = _Painter()
    monkeypatch.setattr(abhasa, "painter_of", lambda: painter)

    def broken():
        raise ValueError("Unexpected UTF-8 BOM")
    monkeypatch.setattr(adhisthana, "get_worlds_registry", broken)
    monkeypatch.setattr(adhisthana, "active_world_registry_entry", broken)
    world = _world()
    assert abhasa.settings(world) is None
    abhasa.after_telling(world, None, [_m("t1", "hero", "I fell.", 11, salience=1)], "square", "hero", "p11")
    abhasa.after_join(world, "hero", LOOK_HERE, "I am the hero.")
    assert painter.calls == [] and not abhasa._tasks
    # and a hook that fails for any other reason is let go the same way
    monkeypatch.setattr(abhasa, "settings", broken)
    abhasa.after_telling(world, None, [_m("t1", "hero", "I fell.", 11, salience=1)], "square", "hero", "p11")
    abhasa.after_join(world, "hero", LOOK_HERE, "I am the hero.")
    assert painter.calls == [] and not abhasa._tasks


def test_a_world_that_does_not_ask_for_pictures_makes_none_and_starts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    monkeypatch.setattr(abhasa, "settings", lambda world: None)
    monkeypatch.setattr(abhasa, "painter_of", lambda: painter)
    world = _world()

    async def go():
        abhasa.after_telling(world, None, [_m("t1", "hero", "I fell.", 11, salience=1)], "square", "hero", "p11")
        abhasa.after_join(world, "hero", "Tall, weathered, a grey cloak.", "I am the hero.")
        await asyncio.gather(*abhasa._tasks)
    asyncio.run(go())
    assert painter.calls == [] and _kept(world) == [] and not abhasa._tasks
    # and one that asks, with no painter configured, makes none either
    monkeypatch.setattr(abhasa, "settings", lambda world: {})
    monkeypatch.setattr(abhasa, "painter_of", lambda: None)
    asyncio.run(go())
    assert painter.calls == [] and not abhasa._tasks


def test_a_place_is_pictured_once_with_no_one_in_it_and_a_player_is_not_painted_for_being_in_a_telling(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    _tell(world, _Roles(_Writer({})), [_m("t1", "hero", "I walked to the well.", 11, subjects={"square"}, salience=4)])
    assert [c["aspect"] for c in painter.calls] == ["16:9"]                      # the place; the hero's own portrait is theirs to ask for, not made for them
    assert "Town Square" in painter.calls[0]["prompt"] and "a well at its middle" in painter.calls[0]["prompt"] and "empty of the people the accounts mention" in painter.calls[0]["prompt"]
    assert painter.calls[0]["seed"] is None
    kept = _kept(world)
    assert [(c.entity_id, c.pulse_id) for c in kept] == [("square", None)] and kept[0].has_picture() and kept[0].mime == "image/jpeg" and kept[0].seq == 1
    _tell(world, _Roles(_Writer({})), [_m("t2", "hero", "I drew water.", 12, subjects={"square"})])
    assert len(painter.calls) == 1


def _place_world(rows: list[tuple[int, str]]) -> WorldSession:
    """The test world with the square's own record replaced by these rows, oldest first."""
    world = _world()
    world.recent_memories = [m for m in world.recent_memories if m.entity_id != "square"] + \
        [_m(f"sq{t}", "square", text, t, tier="AMBIENT", pid=f"g{t}") for t, text in rows]
    return world


CLEAN = "Lamps are lit along the lane and a dog sleeps under the cart."


def test_a_place_is_told_the_first_row_of_its_own_record_that_names_no_one_and_its_name_alone_if_none_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stonemarch, 2026-11-12 (Marc: "It got filonious quite wrong"): the view was painted from the place's first and latest rows, the latest naming the player,
    and the painter drew the player with a face of its own. Nobody's face is in a place's picture, so nobody's name is in what it is told."""
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _place_world([(0, "Hero comes to the well in a grey cloak."), (1, "Hero asks Friend about the well."), (2, CLEAN), (3, "A second plain line about the well.")])
    _tell(world, _Roles(_Writer({})), [_m("t1", "hero", "I walked to the well.", 11, subjects={"square"})])
    prompt = painter.calls[0]["prompt"]
    assert CLEAN in prompt and "Hero" not in prompt and "Friend" not in prompt and "second plain line" not in prompt   # the first that names no one, and only that
    assert "walked to the well" not in prompt                                                                           # and never the telling's own words

    painter = _Painter()
    _arm(monkeypatch, painter)
    rows = [(i, f"Hero does thing number {i} by the well.") for i in range(abhasa.PLACE_ROWS)] + [(abhasa.PLACE_ROWS, CLEAN)]
    world = _place_world(rows)
    _tell(world, _Roles(_Writer({})), [_m("t1", "hero", "I walked to the well.", 11, subjects={"square"})])
    prompt = painter.calls[0]["prompt"]
    assert "Town Square" in prompt and CLEAN not in prompt and "Hero" not in prompt and "walked to the well" not in prompt and "What is known of the place" not in prompt
    assert _kept(world)[0].has_picture()                                                                                # none within the bound: its name alone, and it is still painted


def test_the_names_a_place_is_kept_from_are_those_whose_faces_are_held_not_every_perceiver_or_every_being(monkeypatch: pytest.MonkeyPatch) -> None:
    """Austin's apartment: its first row names "electric bike" (a carried thing that perceives), and was skipped as naming someone."""
    def accounts(world: WorldSession) -> list[str]:
        return abhasa._place_accounts(world, asyncio.run(abhasa.ensure(world)), "square")

    _arm(monkeypatch, _Painter())
    first = "Mara the Ferrywoman poles her ferry in below the well."
    world = _place_world([(0, first), (1, CLEAN)])
    assert accounts(world) == [first]                                    # Mara is a being no one has a portrait of: her name does not keep the row out
    _portrait(world, "mara")
    assert accounts(world) == [CLEAN]                                    # her face is held now: the row that names her is not told, and the next is
    world = _place_world([(0, "Friend keeps a stall by the well."), (1, CLEAN)])
    assert accounts(world) == [CLEAN]                                    # a player's name keeps a row out, portrait or none


def test_a_moment_of_weighty_or_heavier_is_painted_from_the_players_portrait_and_an_ordinary_one_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    _portrait(world, "hero")
    roles = _Roles(_Writer({}))
    _tell(world, roles, [_m("t1", "hero", "I counted the coins.", 11, subjects={"square"}, salience=4)])           # the place
    painter.calls.clear()
    _tell(world, roles, [_m("t2", "hero", "I counted again.", 12, subjects={"square"}, salience=4)])
    assert painter.calls == []
    _tell(world, roles, [_m("t3", "hero", "The ice gave way under me and I went in.", 13, subjects={"square"}, salience=3)])
    assert len(painter.calls) == 1
    call = painter.calls[0]
    assert call["aspect"] == "16:9" and "ice gave way" in call["prompt"] and "reference portrait of Hero" in call["prompt"]
    assert call["seed"] is not None and call["seed"].mime == "image/jpeg"
    moment = next(c for c in _kept(world) if c.pulse_id == "p13")
    assert moment.entity_id is None and moment.seeds == ("hero",)
    _tell(world, roles, [_m("t4", "hero", "I thought about the ice.", 14, subjects={"square"}, salience=4)])   # weighty is the line: notable is not a moment
    assert len(painter.calls) == 1


def test_a_moment_waits_for_the_players_face_and_the_place_may_still_be_pictured(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    _tell(world, _Roles(_Writer({})), [_m("t1", "hero", "The ice gave way under me and I went in.", 11, subjects={"square"}, salience=1)])
    assert [c["aspect"] for c in painter.calls] == ["16:9"] and "Town Square" in painter.calls[0]["prompt"]   # no portrait accepted: no moment, the place instead
    assert [c.pulse_id for c in _kept(world)] == [None]


def test_a_scene_with_several_characters_is_seeded_with_their_portraits_appended_into_one_image(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    _portrait(world, "hero")
    roles = _Roles(_Writer({"Mara the Ferrywoman": "A broad woman of fifty, a river-grey braid."}))
    # Mara is named in the hero's tuples in three tellings: her face is asked about, and she is painted
    for i, t in enumerate((11, 12, 13)):
        _tell(world, roles, [_m(f"r{i}", "hero", f"Mara poled the ferry across, telling {i}.", t, subjects={"mara", "square"}, salience=4)])
    by_entity = {c.entity_id: c for c in _kept(world) if c.pulse_id is None}
    assert set(by_entity) == {"hero", "square", "mara"} and by_entity["mara"].has_picture()
    mara_call = next(c for c in painter.calls if "river-grey braid" in c["prompt"])
    assert mara_call["aspect"] == "3:4" and mara_call["seed"] is None
    painter.calls.clear()
    _tell(world, roles, [_m("big", "hero", "Mara reached for me as the ferry rolled and I fell.", 14, subjects={"mara", "key", "square"}, salience=2)])
    assert len(painter.calls) == 1
    call = painter.calls[0]
    assert call["prompt"].startswith("The image above is one reference sheet of portraits, left to right: Hero, Mara the Ferrywoman. ")
    sheet = Image.open(io.BytesIO(call["seed"].data))
    assert sheet.height == 768 and sheet.width > 768     # two portraits side by side, the actor first
    moment = next(c for c in _kept(world) if c.pulse_id == "p14")
    assert moment.seeds == ("hero", "mara")               # the prop, with no portrait, is not in it


def test_at_most_four_portraits_seed_one_picture_the_actor_first(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    _portrait(world, "hero")
    for name in ("a", "b", "c", "d", "e"):
        world.canon.entities[name] = Entity(name, [name.upper()], 0, "genesis")
        _portrait(world, name)    # five beings already have portraits
    _tell(world, _Roles(_Writer({})), [_m("t2", "hero", "They all came at me at once.", 12, subjects={"a", "b", "c", "d", "e", "square"}, salience=1)])
    moment = next(c for c in _kept(world) if c.pulse_id == "p12")
    assert moment.seeds == ("hero", "a", "b", "c") and len(moment.seeds) == abhasa.SEEDS_MAX


def _round(world: WorldSession, roles, hero_weight: int, friend_weight: int, tick: int = 12, actor: str = "hero", friend_loc: str = "square") -> str:
    """One round: one telling, one pulse id, each of two players told their own tuple (as saṅgama writes it). Returns its pulse id."""
    pid = f"round{tick}"
    _tell(world, roles, [_m(f"rh{tick}", "hero", "The ice gave way under us and I went in.", tick, subjects={"friend", "square"}, salience=hero_weight, pid=pid),
                         _m(f"rf{tick}", "friend", "The ice gave way and Hero went in; I reached for him.", tick, subjects={"hero", "square"}, salience=friend_weight,
                            loc=friend_loc, pid=pid)], actor=actor)
    return pid


def test_a_round_is_one_telling_and_has_one_moment_seeded_with_every_player_in_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Marc, 2026-11-11: "Yes, one per round". Before it, each heavy player made their own picture of the same moment (two, seeded hero-friend and friend-hero)."""
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    _online(world, "hero", "friend")
    _portrait(world, "hero")
    _portrait(world, "friend")
    roles = _Roles(_Writer({}))
    _tell(world, roles, [_m("warm", "hero", "I stood in the square.", 11, subjects={"square"}, salience=4, pid="p0")])      # the square is pictured
    _tell(world, roles, [_m("warm2", "friend", "I kept my stall.", 11, subjects={"cellar"}, salience=4, loc="cellar", pid="p00")])  # and so is the cellar
    painter.calls.clear()

    pid = _round(world, roles, 2, 3, tick=12)                      # both weighty or heavier: one picture, of the heavier tuple (the hero's), both faces in it
    assert len(painter.calls) == 1
    moments = [c for c in _kept(world) if c.pulse_id == pid]
    assert len(moments) == 1 and moments[0].seeds == ("hero", "friend")
    assert "ice gave way under us" in painter.calls[0]["prompt"] and "Hero, Friend" in painter.calls[0]["prompt"]
    assert abhasa.visible_to(world, asyncio.run(abhasa.ensure(world)), "friend")[-1].pulse_id == pid   # a round's picture is each of its players' own

    painter.calls.clear()
    _round(world, roles, 3, 1, tick=13)                            # the friend's tuple is the heavier: its words, the friend's face first
    assert len(painter.calls) == 1 and "Hero went in; I reached" in painter.calls[0]["prompt"]
    assert next(c for c in _kept(world) if c.pulse_id == "round13").seeds == ("friend", "hero")

    painter.calls.clear()
    _round(world, roles, 2, 2, tick=14, actor="friend")            # a tie goes to the one who acted
    assert len(painter.calls) == 1 and next(c for c in _kept(world) if c.pulse_id == "round14").seeds == ("friend", "hero")
    _round(world, roles, 2, 2, tick=17, actor="hero")
    assert next(c for c in _kept(world) if c.pulse_id == "round17").seeds == ("hero", "friend")

    painter.calls.clear()                                          # a partner the heavy tuple does not name is in the seed all the same: they were told
    _tell(world, roles, [_m("ua", "hero", "A crack ran across the whole pond.", 18, subjects={"square"}, salience=2, pid="round18"),
                         _m("ub", "friend", "I saw the crack run.", 18, subjects={"square"}, salience=4, pid="round18")])
    assert len(painter.calls) == 1 and next(c for c in _kept(world) if c.pulse_id == "round18").seeds == ("hero", "friend")

    painter.calls.clear()
    _round(world, roles, 4, 2, tick=15)                            # only one of them heavy: still the one picture, and the other is in it
    assert len(painter.calls) == 1 and next(c for c in _kept(world) if c.pulse_id == "round15").seeds == ("friend", "hero")

    painter.calls.clear()
    _round(world, roles, 4, 4, tick=16)                            # neither heavy: no moment, and both places already pictured
    assert painter.calls == []


def test_the_moment_stands_for_the_places_of_the_players_it_is_of(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    _portrait(world, "hero")
    _portrait(world, "friend")
    _round(world, _Roles(_Writer({})), 2, 3, tick=12, friend_loc="cellar")      # neither the square nor the cellar has a picture, and neither is made
    assert len(painter.calls) == 1 and [c.pulse_id for c in _kept(world) if c.entity_id not in ("hero", "friend")] == ["round12"]


def test_a_moment_that_fails_is_tried_again_only_a_few_times_and_never_fails_the_telling(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter(fail=True)
    _arm(monkeypatch, painter)
    world = _world()
    _portrait(world, "hero")
    roles = _Roles(_Writer({}))
    pid = _round(world, roles, 2, 3, tick=12)
    told = {"hero": [_m("again", "hero", "The ice gave way under us and I went in.", 12, subjects={"square"}, salience=2, pid=pid)]}
    for _ in range(abhasa.TRIES + 3):
        asyncio.run(abhasa._illustrate(world, roles, abhasa.settings(world), abhasa.painter_of(), told, {"hero": "square"}, pid))
    assert len([c for c in painter.calls if "ice gave way" in c["prompt"]]) == abhasa.TRIES and [c for c in _kept(world) if c.pulse_id == pid] == []


def test_a_telling_is_never_pictured_twice_and_a_heavy_player_with_no_face_does_not_stop_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    _portrait(world, "hero")                                       # the friend has no accepted portrait
    roles = _Roles(_Writer({}))
    _tell(world, roles, [_m("warm", "hero", "I stood in the square.", 11, subjects={"square"}, salience=4, pid="p0")])
    painter.calls.clear()
    pid = _round(world, roles, 2, 1, tick=12, friend_loc="cellar")
    # one moment, of the hero (the friend, heavier, has no face to give it), the hero's face alone in it; the friend's own place, which has none, is pictured
    assert [c["aspect"] for c in painter.calls] == ["16:9", "16:9"]
    assert "ice gave way under us" in painter.calls[0]["prompt"]
    assert "The Cellar" in painter.calls[1]["prompt"]
    assert next(c for c in _kept(world) if c.pulse_id == pid).seeds == ("hero",)

    painter.calls.clear()
    tuples = [_m("again", "hero", "The ice gave way under us and I went in.", 12, subjects={"friend", "square"}, salience=2, pid=pid)]
    asyncio.run(abhasa._illustrate(world, roles, abhasa.settings(world), abhasa.painter_of(), {"hero": tuples}, {"hero": "square"}, pid))
    assert painter.calls == [] and len([c for c in _kept(world) if c.pulse_id == pid]) == 1   # the same telling told again makes nothing

    painter2 = _Painter()
    _arm(monkeypatch, painter2)
    world2 = _world()                                              # nobody has a portrait: no moment, only the places
    _round(world2, _Roles(_Writer({})), 2, 3, tick=12, friend_loc="cellar")
    assert all(c.pulse_id is None for c in _kept(world2)) and len(_kept(world2)) == 2


def test_a_being_is_asked_about_on_its_third_telling_and_a_thing_judged_to_have_no_face_is_never_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    writer = _Writer({"Mara the Ferrywoman": "A broad woman of fifty."})
    roles = _Roles(writer)
    for i, t in enumerate((11, 12)):
        _tell(world, roles, [_m(f"a{i}", "hero", f"I took the iron key, and Mara watched, {i}.", t, subjects={"mara", "key", "square"})])
    assert writer.asked == []                                               # twice is not enough
    _tell(world, roles, [_m("a2", "hero", "I took the iron key, and Mara watched, 2.", 13, subjects={"mara", "key", "square"})])
    assert writer.asked == ["an iron key"]                                  # one being a telling, the most named first, then by id: the key
    key_row = next(c for c in _kept(world) if c.entity_id == "key")
    assert not key_row.has_picture() and key_row.note == "judged to have no face"
    _tell(world, roles, [_m("a3", "hero", "Mara again.", 14, subjects={"mara", "square"})])
    assert writer.asked == ["an iron key", "Mara the Ferrywoman"]
    for t in (15, 16):
        _tell(world, roles, [_m(f"b{t}", "hero", "The key again, and Mara.", t, subjects={"key", "mara", "square"})])
    assert writer.asked == ["an iron key", "Mara the Ferrywoman"]           # each asked once, and the portrait made once
    assert sum(1 for c in _kept(world) if c.entity_id == "mara") == 1 and next(c for c in _kept(world) if c.entity_id == "mara").has_picture()


def test_a_beings_portrait_is_painted_from_the_look_the_judge_wrote_after_being_handed_the_world_and_the_fixed_points(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    world.world_data = {"source": "A river town in 1928."}
    writer = _Writer({"Mara the Ferrywoman": "A lean woman of sixty, hair white and combed flat, a crooked nose."})
    roles = _Roles(writer)
    for i, t in enumerate((11, 12, 13)):
        _tell(world, roles, [_m(f"z{i}", "hero", f"Mara poled the ferry, {i}.", t, subjects={"mara", "square"})])
    assert writer.asked == ["Mara the Ferrywoman"]
    assert "THE WORLD: A river town in 1928." in writer.users[0] and abhasa.fixed_points("mara") in writer.users[0]   # the id of the being, not its label
    assert "Mara poled the ferry, 0." in writer.users[0] and "Mara poled the ferry, 2." in writer.users[0]            # and still the accounts
    portrait = [c for c in painter.calls if "character reference portrait" in c["prompt"]]
    assert len(portrait) == 1 and "crooked nose" in portrait[0]["prompt"] and "Mara the Ferrywoman" in portrait[0]["prompt"]
    assert abhasa.fixed_points("mara") not in portrait[0]["prompt"]       # the painter is given the look, never the instruction to the judge


def test_a_players_own_other_characters_and_places_are_never_counted_as_beings_to_ask_about(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    writer = _Writer({"Friend": "x", "Town Square": "x", "The Marches": "x", "Hero": "x"})
    for i, t in enumerate((11, 12, 13, 14)):
        _tell(world, _Roles(writer), [_m(f"c{i}", "hero", "I saw my friend in the square.", t, subjects={"friend", "square", "region", "hero"})])
    assert "Friend" not in writer.asked and "Town Square" not in writer.asked and "Hero" not in writer.asked
    assert writer.asked == ["The Marches"]   # a region is not a scene, and the engine knows no more of it: the judge says


def test_the_days_cap_stops_pictures_and_the_telling_is_unharmed(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter, daily=2)
    world = _world()
    _portrait(world, "hero")                                                # one picture made today already
    _tell(world, _Roles(_Writer({})), [_m("t1", "hero", "The ice gave way.", 11, subjects={"square"}, salience=1)])
    assert len(painter.calls) == 1                                          # the moment: the day's second
    _tell(world, _Roles(_Writer({})), [_m("t2", "hero", "I fell again.", 12, subjects={"square"}, salience=1)])
    assert len(painter.calls) == 1 and len([c for c in _kept(world) if c.has_picture()]) == 2


def test_a_refusal_is_kept_so_it_is_not_asked_again_and_a_failure_is_tried_again_only_a_few_times(monkeypatch: pytest.MonkeyPatch) -> None:
    refusing = _Painter(refuse="establishing view")
    _arm(monkeypatch, refusing)
    world = _world()
    roles = _Roles(_Writer({}))
    _tell(world, roles, [_m("t1", "hero", "I walked.", 11, subjects={"square"}, salience=4)])
    declined = next(c for c in _kept(world) if c.entity_id == "square")
    assert not declined.has_picture() and declined.note == "declined by the model"
    n = len(refusing.calls)
    _tell(world, roles, [_m("t2", "hero", "I walked on.", 12, subjects={"square"}, salience=4)])
    assert len(refusing.calls) == n

    failing = _Painter(fail=True)
    _arm(monkeypatch, failing)
    other = _world()
    for i in range(6):
        _tell(other, roles, [_m(f"f{i}", "hero", "I walked.", 11 + i, subjects={"square"}, salience=4)])
    assert len(failing.calls) == abhasa.TRIES and _kept(other) == []


def test_a_telling_that_finds_too_many_waiting_is_let_go(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    abhasa.state_of(world).waiting = abhasa.MAX_WAITING
    asyncio.run(abhasa._illustrate(world, None, {}, painter, {"hero": [_m("t1", "hero", "I fell.", 11, salience=1)]}, {"hero": "square"}, "p11"))
    assert painter.calls == [] and abhasa.state_of(world).waiting == abhasa.MAX_WAITING


# -- a player's portrait: a draft, changed by asking, kept when they accept it ------------------------------------------------------------------

LOOK = "Tall and weathered, close-cropped grey hair, a scar through the left eyebrow; a patched green cloak."


def test_a_draft_is_painted_from_how_they_look_changed_by_what_they_ask_and_kept_only_when_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    asyncio.run(abhasa.draft_start(world, "hero", LOOK, "I am the hero, a courier."))
    assert len(painter.calls) == 1 and painter.calls[0]["aspect"] == "3:4" and painter.calls[0]["seed"] is None
    assert "scar through the left eyebrow" in painter.calls[0]["prompt"] and "a courier" in painter.calls[0]["prompt"]
    s = abhasa.draft_state(world, "hero")
    assert s["enabled"] and not s["accepted"] and s["version"] == 1 and s["asks"] == [] and s["edits_left"] == abhasa.MAX_EDITS and not s["busy"] and s["error"] is None
    first = abhasa.draft_image(world, "hero")
    assert first is not None and _kept(world) == []                         # a draft is nobody's yet: nothing is kept

    asyncio.run(abhasa.draft_edit(world, "hero", "make my hair white and a little longer"))
    call = painter.calls[1]
    assert call["seed"] is not None and call["seed"].data == first          # the model is given the picture shown, and the request
    assert "make my hair white and a little longer" in call["prompt"] and "nothing else" in call["prompt"] and "scar through the left eyebrow" in call["prompt"]
    asyncio.run(abhasa.draft_edit(world, "hero", "a darker cloak"))
    s = abhasa.draft_state(world, "hero")
    assert s["version"] == 3 and s["asks"] == ["make my hair white and a little longer", "a darker cloak"] and s["edits_left"] == abhasa.MAX_EDITS - 2
    assert abhasa.draft_image(world, "hero") != first and _kept(world) == []

    kept = asyncio.run(abhasa.draft_accept(world, "hero"))
    assert kept.entity_id == "hero" and kept.pulse_id is None and kept.has_picture() and kept.note == "accepted by the player after 2 changes"
    assert [c.id for c in _kept(world)] == [kept.id] and kept.size > 0
    assert LOOK in asyncio.run(world.store.load_chitra_index())[0].prompt and "a darker cloak" in _kept(world)[0].prompt
    mime, data = asyncio.run(world.store.load_chitra_data(kept.id))
    assert mime == "image/jpeg" and data == abhasa.state_of(world).portraits["hero"] and Image.open(io.BytesIO(data)).format == "JPEG"
    assert abhasa.draft_state(world, "hero") == {"enabled": True, "accepted": True} and abhasa.draft_image(world, "hero") is None
    with pytest.raises(abhasa.Refused):                                      # final: not painted again
        asyncio.run(abhasa.draft_start(world, "hero", LOOK))
    # and it is what every later picture of them is seeded with
    _tell(world, _Roles(_Writer({})), [_m("t1", "hero", "The ice gave way.", 11, subjects={"square"}, salience=1)])
    assert next(c for c in _kept(world) if c.pulse_id == "p11").seeds == ("hero",)


def test_going_back_undoes_the_last_change_and_the_conversation_with_it(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    asyncio.run(abhasa.draft_start(world, "hero", LOOK))
    first = abhasa.draft_image(world, "hero")
    asyncio.run(abhasa.draft_edit(world, "hero", "add a hat"))
    assert abhasa.draft_image(world, "hero") != first
    rev = abhasa.draft_state(world, "hero")["rev"]
    abhasa.draft_back(world, "hero")
    s = abhasa.draft_state(world, "hero")
    assert abhasa.draft_image(world, "hero") == first and s["version"] == 1 and s["asks"] == [] and s["rev"] != rev
    assert s["edits_left"] == abhasa.MAX_EDITS - 1                          # the change was made: it is counted
    with pytest.raises(abhasa.Refused):
        abhasa.draft_back(world, "hero")                                    # nothing before the first


def test_a_refused_change_leaves_the_portrait_shown_as_it_was_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter(refuse="What is asked: make them bleed")
    _arm(monkeypatch, painter)
    world = _world()
    asyncio.run(abhasa.draft_start(world, "hero", LOOK))
    shown = abhasa.draft_image(world, "hero")
    asyncio.run(abhasa.draft_edit(world, "hero", "make them bleed"))
    s = abhasa.draft_state(world, "hero")
    assert abhasa.draft_image(world, "hero") == shown and s["version"] == 1 and s["asks"] == [] and "would not make that" in s["error"] and not s["busy"]
    assert s["edits_left"] == abhasa.MAX_EDITS
    painter.refuse = None
    asyncio.run(abhasa.draft_edit(world, "hero", "make them smile"))         # and the next try is clean
    assert abhasa.draft_state(world, "hero")["error"] is None and abhasa.draft_state(world, "hero")["version"] == 2
    painter.fail = True
    asyncio.run(abhasa.draft_edit(world, "hero", "a beard"))
    assert "failed" in abhasa.draft_state(world, "hero")["error"] and abhasa.draft_state(world, "hero")["version"] == 2


def test_a_draft_needs_a_description_a_player_and_a_world_that_makes_pictures_and_the_changes_are_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    with pytest.raises(abhasa.Refused, match="describe how you look"):
        asyncio.run(abhasa.draft_start(world, "hero", "tall"))
    with pytest.raises(abhasa.Refused):
        asyncio.run(abhasa.draft_start(world, "mara", LOOK))                 # not a player
    with pytest.raises(abhasa.Refused, match="no portrait yet"):
        asyncio.run(abhasa.draft_edit(world, "hero", "a hat"))
    with pytest.raises(abhasa.Refused):
        asyncio.run(abhasa.draft_accept(world, "hero"))
    asyncio.run(abhasa.draft_start(world, "hero", LOOK))
    with pytest.raises(abhasa.Refused, match="say what to change"):
        asyncio.run(abhasa.draft_edit(world, "hero", "   "))
    monkeypatch.setattr(abhasa, "MAX_EDITS", 2)
    asyncio.run(abhasa.draft_edit(world, "hero", "a hat"))
    asyncio.run(abhasa.draft_edit(world, "hero", "a scarf"))
    with pytest.raises(abhasa.Refused, match="as many changes"):
        asyncio.run(abhasa.draft_edit(world, "hero", "a beard"))
    assert len(painter.calls) == 3
    monkeypatch.setattr(abhasa, "settings", lambda w: None)                  # a world that makes none
    with pytest.raises(abhasa.PicturesOff):
        asyncio.run(abhasa.draft_start(world, "friend", LOOK))
    assert abhasa.draft_state(world, "hero") == {"enabled": False}


def test_a_players_changes_are_pictures_against_the_days_cap_and_a_new_draft_starts_over(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter, daily=2)
    world = _world()
    asyncio.run(abhasa.draft_start(world, "hero", LOOK))
    asyncio.run(abhasa.draft_edit(world, "hero", "a hat"))
    asyncio.run(abhasa.draft_edit(world, "hero", "a scarf"))                 # the third picture of the day: not made
    s = abhasa.draft_state(world, "hero")
    assert len(painter.calls) == 2 and s["version"] == 2 and "used up" in s["error"]
    monkeypatch.setattr(abhasa, "settings", lambda w: {"daily": 10})
    asyncio.run(abhasa.draft_start(world, "hero", "Short, red hair in a knot, a leather apron."))   # begun again
    s = abhasa.draft_state(world, "hero")
    assert s["version"] == 1 and s["asks"] == [] and s["look"].startswith("Short, red hair") and s["error"] is None and s["edits_left"] == abhasa.MAX_EDITS


def test_a_joining_player_is_given_a_first_draft_from_how_they_wrote_that_they_look_and_none_with_nothing_written(monkeypatch: pytest.MonkeyPatch) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    world.canon.entities["tamsin"] = Entity("tamsin", ["Tamsin"], 0, "genesis")
    world.playable_entity_ids.append("tamsin")

    async def go(look: str):
        abhasa.after_join(world, "tamsin", look, "I am Tamsin, a tinker from the hills.")
        await asyncio.gather(*abhasa._tasks)
    asyncio.run(go(""))
    assert painter.calls == [] and abhasa.draft_state(world, "tamsin")["version"] == 0     # nothing written: the page asks them to describe themselves
    asyncio.run(go("Short and wiry, red hair in a knot; a patched green cloak."))
    assert len(painter.calls) == 1 and painter.calls[0]["aspect"] == "3:4"
    prompt = painter.calls[0]["prompt"]
    assert "Tamsin" in prompt and "red hair in a knot" in prompt and "a tinker from the hills" in prompt
    assert abhasa.draft_state(world, "tamsin")["version"] == 1 and _kept(world) == []      # a draft: theirs to change and keep


def test_the_portrait_routes_walk_a_player_from_a_draft_to_a_kept_portrait_and_refuse_what_cannot_be_done(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    client.app.state.world = world
    assert client.get("/players/hero/portrait").json() == {"enabled": True, "accepted": False, "busy": False, "version": 0, "rev": 0, "asks": [], "look": "", "error": None,
                                                           "edits_left": abhasa.MAX_EDITS}
    assert client.get("/players/hero/portrait/image").status_code == 404
    short = client.post("/players/hero/portrait/draft", json={"look": "tall"})
    assert short.status_code == 409 and "describe how you look" in short.json()["detail"]
    drafted = client.post("/players/hero/portrait/draft", json={"look": LOOK}).json()
    assert drafted["version"] == 1 and drafted["rev"] == 1
    image = client.get("/players/hero/portrait/image?rev=1")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg" and image.headers["cache-control"] == "no-store"
    edited = client.post("/players/hero/portrait/edit", json={"ask": "white hair"}).json()
    assert edited["version"] == 2 and edited["asks"] == ["white hair"]
    assert client.get("/players/hero/portrait/image?rev=2").content != image.content
    backed = client.post("/players/hero/portrait/back").json()
    assert backed["version"] == 1 and backed["asks"] == [] and client.get("/players/hero/portrait/image").content == image.content
    assert client.post("/players/hero/portrait/edit", json={"ask": ""}).status_code == 409
    assert client.post("/players/nobody/portrait/draft", json={"look": LOOK}).status_code == 404
    accepted = client.post("/players/hero/portrait/accept").json()
    assert accepted == {"enabled": True, "accepted": True}
    assert client.post("/players/hero/portrait/accept").status_code == 409 and client.get("/players/hero/portrait/image").status_code == 404
    pictures = client.get("/players/hero/pictures").json()
    assert [(p["caption"], p["place"], p["moment"]) for p in pictures] == [("Hero", False, False)]    # the kept portrait is the player's first picture
    off = client.get("/players/friend/portrait")
    monkeypatch.setattr(abhasa, "settings", lambda w: None)
    assert client.get("/players/hero/portrait").json() == {"enabled": False} and client.post("/players/hero/portrait/draft", json={"look": LOOK}).status_code == 409
    assert off.status_code == 200 and client.get("/players/hero/pictures").json() == []


# -- who sees what ------------------------------------------------------------------------------------------------------------------------

def _online(world: WorldSession, *pids: str) -> None:
    for pid in pids:
        world.last_seen[pid] = time.monotonic()


def _pictured_world(monkeypatch: pytest.MonkeyPatch, beside: bool = False):
    """`beside`: the square and the cellar stand side by side in one region, so each is the other's neighbour (v5 §12), and there is a gate in the square."""
    painter = _Painter()
    _arm(monkeypatch, painter)
    world = _world()
    if beside:
        for sid, name in (("square", "Town Square"), ("cellar", "The Cellar")):
            world.canon.entities[sid] = Entity(sid, [name], 0, "genesis", avarana="scene", kosha="region")
        world.canon.entities["gate"] = Entity("gate", ["the cellar gate"], 0, "genesis", kosha="square")
    _online(world, "hero", "friend")
    _portrait(world, "hero")
    _portrait(world, "friend")
    roles = _Roles(_Writer({"Mara the Ferrywoman": "A broad woman of fifty."}))
    for i, t in enumerate((11, 12, 13)):
        _tell(world, roles, [_m(f"r{i}", "hero", f"Mara poled the ferry across, telling {i}.", t, subjects={"mara", "square"})])
    _tell(world, roles, [_m("big", "hero", "The ferry rolled and I went in.", 14, subjects={"mara", "square"}, salience=2)])
    _tell(world, roles, [_m("fr", "friend", "I sold bread at the cellar door.", 15, subjects={"cellar"}, loc="cellar")], actor="friend", scene="cellar")
    return world, roles


def _seen(world: WorldSession, pid: str) -> set:
    return {(c.entity_id, c.pulse_id) for c in abhasa.visible_to(world, abhasa.state_of(world), pid)}


def _away_for_the_idle_time(world: WorldSession, pid: str) -> None:
    """The player is asked about again after as long a gap as makes them away: the next ask begins a new session."""
    abhasa.state_of(world).sessions[pid].asked -= timedelta(seconds=abhasa.adhisthana.PLAYER_IDLE_SECONDS + 1)
    time.sleep(0.05)   # Windows' clock moves in steps of about 15 ms: what was made before must be seen to be before the session that begins on the next ask


def test_a_player_sees_their_own_portrait_those_in_their_scene_that_place_and_what_this_session_made_of_what_they_lived_and_nothing_of_anyone_elses(monkeypatch: pytest.MonkeyPatch) -> None:
    world, _ = _pictured_world(monkeypatch)
    st = abhasa.state_of(world)
    kept = {(c.entity_id, c.pulse_id) for c in st.index if c.has_picture()}
    assert kept == {("hero", None), ("friend", None), ("mara", None), ("square", None), (None, "p14"), ("cellar", None)}
    hero = {(c.entity_id, c.pulse_id) for c in abhasa.visible_to(world, st, "hero")}
    assert hero == {("hero", None), ("mara", None), ("square", None), (None, "p14")}              # not the friend, nor the cellar they never stood in
    friend = {(c.entity_id, c.pulse_id) for c in abhasa.visible_to(world, st, "friend")}
    assert friend == {("friend", None), ("cellar", None)}                                           # not the hero's moment, not Mara (never named in their tuples)
    asked = abhasa.listing(world, st, "hero", after=0)
    assert [r["seq"] for r in asked] == sorted(r["seq"] for r in asked) and {r["caption"] for r in asked} == {"Hero", "Mara the Ferrywoman", "Town Square", "this moment"}
    by = {r["caption"]: r for r in asked}
    assert by["this moment"]["moment"] and not by["Hero"]["moment"] and not by["Hero"]["place"] and by["Town Square"]["place"]
    last = max(r["seq"] for r in asked)
    assert abhasa.listing(world, st, "hero", after=last) == []


def test_a_being_met_in_an_earlier_session_is_shown_only_while_they_are_in_the_scene_and_the_places_picture_always(monkeypatch: pytest.MonkeyPatch) -> None:
    world, roles = _pictured_world(monkeypatch)
    assert _seen(world, "hero") == {("hero", None), ("mara", None), ("square", None), (None, "p14")}
    _away_for_the_idle_time(world, "hero")                                  # the hero is away as long as makes them away, and comes back: a new session
    assert _seen(world, "hero") == {("hero", None), ("mara", None), ("square", None)}   # the moment was another session's; Mara and the square are here
    _tell(world, roles, [_m("walk", "hero", "I went down to the cellar, and the friend was there; Mara had come down with me.", 16, subjects={"friend", "cellar", "mara"}, loc="cellar")], scene="cellar")
    assert _seen(world, "hero") == {("hero", None), ("friend", None), ("cellar", None)}   # the friend's portrait now they are in the scene, and the cellar's picture; not Mara's, made before this session (a being is shown while here)
    _tell(world, roles, [_m("back", "hero", "I climbed back up to the square.", 17, subjects={"square"})])
    assert _seen(world, "hero") == {("hero", None), ("mara", None), ("square", None)}     # and Mara and the square, not the friend, when they are back


def test_what_was_made_this_session_of_the_scene_they_left_goes_with_it_and_the_moment_stays_until_the_session_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Marc, 2026-11-24, of what they lived in the scenes they are not in ("Hide them"): the picture panel follows where they stand, not where they have been."""
    world, roles = _pictured_world(monkeypatch)
    _tell(world, roles, [_m("walk", "hero", "I went down to the cellar, and the friend was there.", 16, subjects={"friend", "cellar"}, loc="cellar")], scene="cellar")
    assert _seen(world, "hero") == {("hero", None), ("friend", None), ("cellar", None), (None, "p14")}   # Mara and the square, made this session, were left behind; the telling they lived is theirs
    _away_for_the_idle_time(world, "hero")
    assert _seen(world, "hero") == {("hero", None), ("friend", None), ("cellar", None)}   # and the moment is another session's after it


def _session_begun_before_the_pictures(world: WorldSession, pid: str = "hero") -> None:
    """`_pictured_world` makes the friend's portrait before anything asks about the hero's session, and a session begins when it is first asked about, so the portrait
    was made a moment BEFORE it: on Windows' clock, which moves in steps of about 15 ms, the two fell in one step (made 'this session') or in two (not), and a test that
    wanted the friend seen through a portal passed or failed by chance (one run in six, 2026-11-24). Begin the session earlier, as one that has been going a while."""
    s = abhasa._session(world, abhasa.state_of(world), pid)
    assert s is not None
    s.began -= timedelta(seconds=5)


def _down_to_the_cellar_and_back_up(world: WorldSession, roles) -> tuple[set, set]:
    """The hero goes down to the cellar (what is seen there), and climbs back to the square (what is seen then)."""
    _tell(world, roles, [_m("walk", "hero", "I went down to the cellar, and the friend was there.", 16, subjects={"friend", "cellar"}, loc="cellar")], scene="cellar")
    down = _seen(world, "hero")
    _tell(world, roles, [_m("back", "hero", "I climbed back up to the square.", 17, subjects={"square"})])
    return down, _seen(world, "hero")


def test_what_they_lived_in_a_scene_beside_theirs_is_not_shown_while_nothing_open_joins_the_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Marc, 2026-11-21: "pictures from adjacent containers are also presented in the picture panel. Those pictures shouldn't be shown unless there is a portal
    (window/door) between them". Stonemarch: Jakal on the east track was shown the village green and the lane before the Welcome Wench, scenes of his own village's
    container with no boundary to his."""
    world, roles = _pictured_world(monkeypatch, beside=True)
    down, up = _down_to_the_cellar_and_back_up(world, roles)
    assert down == {("hero", None), ("friend", None), ("cellar", None), (None, "p14")}          # here: the cellar and the friend; the square they left is beside it, shut
    assert up == {("hero", None), ("mara", None), ("square", None), (None, "p14")}              # and back: the square and Mara; the cellar and the friend, lived but beside, are not
    assert ("friend", None) not in up and ("cellar", None) not in up


def test_a_scene_beside_theirs_is_seen_through_a_portal_that_is_open_and_not_once_it_is_shut(monkeypatch: pytest.MonkeyPatch) -> None:
    world, roles = _pictured_world(monkeypatch, beside=True)
    _session_begun_before_the_pictures(world)
    world.records.append(Record("s1", "sandhi", "gate", "cellar", 15, "p15", "open"))           # the gate is in the square and also in the cellar
    down, up = _down_to_the_cellar_and_back_up(world, roles)
    assert up == {("hero", None), ("mara", None), ("square", None), (None, "p14"), ("friend", None), ("cellar", None)}   # the cellar is across the gate: its picture and the friend's stay
    assert down == up
    world.records.append(Record("s2", "sandhi", "gate", "cellar", 17, "p17", "closed"))         # shut it: nothing joins them now
    assert _seen(world, "hero") == {("hero", None), ("mara", None), ("square", None), (None, "p14")}


def test_a_scene_that_is_not_beside_theirs_is_not_shown_either_for_what_they_lived_there_this_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Marc, 2026-11-24 ("Hide them"): the rule was about neighbours only, and what was lived in a scene far from theirs (Stonemarch: the Common Room, the keep gate,
    Filonius's portrait) still showed. The square and the cellar of the plain world share no container, so they are not beside each other: now what was lived below
    is gone when they are back up, as it is for a neighbour, and a place is seen from where one stands or through an open portal, or not at all."""
    world, roles = _pictured_world(monkeypatch)
    down, up = _down_to_the_cellar_and_back_up(world, roles)
    assert down == {("hero", None), ("friend", None), ("cellar", None), (None, "p14")}
    assert up == {("hero", None), ("mara", None), ("square", None), (None, "p14")}              # the cellar and the friend are lived, and far
    assert ("friend", None) not in up and ("cellar", None) not in up


def test_the_far_rule_does_not_reach_their_own_scene_a_being_they_lived_there_is_not_far_though_held_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behaviour kept from before 2026-11-24: Mara, lived this session and standing in the hero's own scene, is shown by the lived path even when she is not
    counted as 'here' (held back, so not in the world to be shown with the scene). The far rule is about other scenes."""
    world, _ = _pictured_world(monkeypatch)
    assert ("mara", None) in _seen(world, "hero")
    world.held_back.add("mara")
    assert abhasa._here(world, abhasa.state_of(world), "hero")[1] == frozenset()                 # not 'here' now ...
    assert ("mara", None) in _seen(world, "hero")                                                # ... and lived, in their scene: not far


def test_a_player_in_no_scene_is_not_shown_what_stands_in_no_scene_as_if_the_two_were_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nowhere is not a place two things share: a player in no scene and a lived being in no scene are not 'here' with each other."""
    world, _ = _pictured_world(monkeypatch)
    assert ("mara", None) in _seen(world, "hero")
    monkeypatch.setattr(abhasa, "_here", lambda w, st, pid: (None, frozenset()))
    monkeypatch.setattr(abhasa, "_stands_in", lambda w, eid: None)
    assert ("mara", None) not in _seen(world, "hero")


def test_a_portal_shows_the_scene_it_opens_on_though_it_is_not_beside_theirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A far scene is no different from a neighbour here: what is open joins it, and it shuts again when the gate does."""
    world, roles = _pictured_world(monkeypatch)
    _session_begun_before_the_pictures(world)
    world.canon.entities["gate"] = Entity("gate", ["the cellar gate"], 0, "genesis", kosha="square")
    world.records.append(Record("s1", "sandhi", "gate", "cellar", 15, "p15", "open"))
    _, up = _down_to_the_cellar_and_back_up(world, roles)
    assert up == {("hero", None), ("mara", None), ("square", None), (None, "p14"), ("friend", None), ("cellar", None)}
    world.records.append(Record("s2", "sandhi", "gate", "cellar", 17, "p17", "closed"))
    assert _seen(world, "hero") == {("hero", None), ("mara", None), ("square", None), (None, "p14")}


def test_a_portal_to_another_scene_shows_only_that_scene_and_a_third_beside_theirs_stays_shut(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gate to the cellar is no gate to anywhere else beside the square: a third scene in the same region stays shut."""
    world, roles = _pictured_world(monkeypatch, beside=True)
    world.canon.entities["lane"] = Entity("lane", ["The Lane"], 0, "genesis", avarana="scene", kosha="region")
    world.records.append(Record("s1", "sandhi", "gate", "cellar", 15, "p15", "open"))
    st = asyncio.run(abhasa.ensure(world))
    asyncio.run(abhasa._keep(world, st, Chitra("lane-picture", "lane", None, 10, "a lane", "fake-painter", "image/jpeg", _jpeg(300, 400), "view")))
    extend_history(world, [_m("l0", "hero", "I looked down the lane and back.", 16, subjects={"lane"})])
    assert abhasa._through_portals(world, "square") == frozenset({"cellar"})                     # the cellar is across the gate; the lane is only beside
    assert abhasa._through_portals(world, None) == frozenset()
    assert ("lane", None) not in _seen(world, "hero")


def test_a_portrait_made_in_their_session_by_someone_elses_act_is_not_shown_to_a_player_who_has_not_lived_with_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stonemarch, 2026-11-12 (Marc): Filonius's portrait, made while Jakal was online, was shown to Jakal, whose own tuples had named him once, long ago."""
    _arm(monkeypatch, _Painter())
    world = _world()
    _online(world, "hero", "friend")
    st = asyncio.run(abhasa.ensure(world))
    extend_history(world, [_m("old", "hero", "The friend sold me bread at the cellar.", 5, subjects={"friend", "cellar"})])   # an earlier meeting
    assert abhasa._session(world, st, "hero") is not None
    _portrait(world, "friend")                                                       # made now, in the hero's session, of someone at the cellar
    assert _seen(world, "hero") == set()                                             # not theirs to see: not here, and not lived this session
    extend_history(world, [_m("in", "friend", "I came up to the square.", 20, loc="square")])
    assert _seen(world, "hero") == {("friend", None)}                                # here, it is
    world.held_back.add("friend")
    assert _seen(world, "hero") == set()                                             # a character held back is not in the world to be seen
    world.held_back.discard("friend")
    world.logged_off.add("friend")
    assert _seen(world, "hero") == set()                                             # nor one logged off
    world.autonomous.add("friend")
    assert _seen(world, "hero") == {("friend", None)}                                # unless the world acts for them: they stay in it


def test_the_first_moment_a_session_lives_is_in_it_though_its_tuples_were_committed_before_the_session_was_asked_about(monkeypatch: pytest.MonkeyPatch) -> None:
    _arm(monkeypatch, _Painter())
    world = _world()
    _online(world, "hero")
    _portrait(world, "hero")
    _tell(world, _Roles(_Writer({})), [_m("big", "hero", "The ice gave way under me.", 11, subjects={"square"}, salience=2)])
    assert (None, "p11") in _seen(world, "hero")


def test_a_player_who_is_not_online_has_no_session_and_sees_their_own_and_what_is_here_only(monkeypatch: pytest.MonkeyPatch) -> None:
    world, _ = _pictured_world(monkeypatch)
    world.last_seen["hero"] -= abhasa.adhisthana.PLAYER_IDLE_SECONDS + 1             # idle past the time that makes them away
    assert _seen(world, "hero") == {("hero", None), ("mara", None), ("square", None)}
    assert "hero" not in abhasa.state_of(world).sessions
    world.last_seen["hero"] = time.monotonic()                                        # back online: a new session begins, with none of the old one's pictures
    assert abhasa._session(world, abhasa.state_of(world), "hero") is not None and (None, "p14") not in _seen(world, "hero")
    world.logged_off.add("hero")
    assert abhasa._session(world, abhasa.state_of(world), "hero") is None            # logged off is no session either


def test_a_beings_later_portrait_stands_in_place_of_its_first_which_is_kept_and_no_longer_shown_or_used_as_a_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    world, roles = _pictured_world(monkeypatch)
    st = abhasa.state_of(world)
    painter = abhasa.painter_of()
    first = st.of["mara"]
    assert first.has_picture() and {c.id for c in abhasa.visible_to(world, st, "hero") if c.entity_id == "mara"} == {first.id}
    made = len(painter.calls)

    second = asyncio.run(abhasa._keep(world, st, Chitra("mara-again", "mara", None, 20, "a tall rangy man", "fake-painter", "image/jpeg", _jpeg(300, 400, (200, 30, 30)))))
    assert st.of["mara"].id == second.id and abhasa.has_portrait(st, "mara")
    assert [c.id for c in abhasa.visible_to(world, st, "hero") if c.entity_id == "mara"] == [second.id]     # the first is not shown beside it
    assert abhasa.superseded(st, first) and not abhasa.superseded(st, second)
    assert asyncio.run(abhasa._portrait_bytes(world, st, "mara")) == _jpeg(300, 400, (200, 30, 30))          # and it is the one a picture of Mara is seeded with
    assert {c.id for c in _kept(world) if c.entity_id == "mara"} == {first.id, second.id}                   # nothing is deleted: the table is insert-only
    assert [r["seq"] for r in abhasa.listing(world, st, "hero", after=first.seq) if r["entity_id"] == "mara"] == [second.seq]   # a page already open is handed the new one

    asyncio.run(abhasa._keep(world, st, Chitra("mara-declined", "mara", None, 21, "", "", note="declined by the model")))
    assert st.of["mara"].id == second.id                                                                      # a refusal never takes a face's place
    for i, t in enumerate((16, 17)):
        _tell(world, roles, [_m(f"later{i}", "hero", f"Mara again, {i}.", t, subjects={"mara", "square"})])
    assert len(painter.calls) == made and st.of["mara"].id == second.id                                       # and no portrait is made of a being that has one

    world.abhasa = None                                                                                       # a restart reads the same from the store
    again = asyncio.run(abhasa.ensure(world))
    assert again.of["mara"].id == second.id and asyncio.run(abhasa._portrait_bytes(world, again, "mara")) == _jpeg(300, 400, (200, 30, 30))
    assert [c.id for c in abhasa.visible_to(world, again, "hero") if c.entity_id == "mara"] == [second.id]


def test_the_pictures_routes_list_and_serve_a_players_pictures_and_refuse_the_rest(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    world, _ = _pictured_world(monkeypatch)
    client.app.state.world = world
    listed = client.get("/players/hero/pictures").json()
    assert len(listed) == 4 and all(set(r) >= {"id", "seq", "caption", "moment", "place"} for r in listed)
    first = client.get(f"/players/hero/pictures/{listed[0]['id']}")
    assert first.status_code == 200 and first.headers["content-type"] == "image/jpeg" and "private" in first.headers["cache-control"]
    assert Image.open(io.BytesIO(first.content)).format == "JPEG"
    assert client.get(f"/players/hero/pictures?after={listed[-1]['seq']}").json() == []
    st = abhasa.state_of(world)
    cellar = next(c for c in st.index if c.entity_id == "cellar")
    assert client.get(f"/players/hero/pictures/{cellar.id}").status_code == 404            # a place the hero never stood in
    assert client.get("/players/hero/pictures/nonesuch").status_code == 404
    assert client.get("/players/nobody/pictures").status_code == 404
    monkeypatch.setattr(abhasa, "settings", lambda w: None)                                # switched off: none are shown
    assert client.get("/players/hero/pictures").json() == [] and client.get(f"/players/hero/pictures/{listed[0]['id']}").status_code == 404


# -- a newcomer, in two boxes --------------------------------------------------------------------------------------------------------------

def test_a_newcomer_told_in_two_boxes_is_given_to_every_writer_as_the_background_then_how_they_look() -> None:
    both = JoinRequest(self_description="I am Tamsin, a tinker from the hills.", appearance="Short and wiry, red hair in a knot; a patched green cloak.")
    assert both.told() == ("I am Tamsin, a tinker from the hills.\n\nPHYSICAL DESCRIPTION (appearance, clothing): "
                           "Short and wiry, red hair in a knot; a patched green cloak.")
    assert JoinRequest(self_description="  I am Tamsin.  ").told() == "I am Tamsin."           # no appearance: the request as it always was
    assert JoinRequest(self_description="I am Tamsin.", appearance="   ").told() == "I am Tamsin."


def test_the_join_form_has_two_boxes_and_sends_both_and_the_page_has_the_portrait_panel_and_the_pictures(client: TestClient) -> None:
    page = client.get("/").text
    assert 'id="nc-look"' in page and 'Physical description' in page and 'Background information' in page
    assert "appearance: look" in page and "self_description: text" in page                          # both are sent
    assert 'id="pictures-toggle"' in page and 'id="gallery"' in page and "no-pictures" in page
    ask = page.split("async function pollPictures() {")[1].split("applyPicturesMode();")[0]
    assert "}/pictures`)" in ask and "?after=" not in ask                                        # the whole list, each time: it changes as they move
    assert "picturesHere" in ask and "fig.remove()" in ask                                       # and what is no longer theirs to see leaves the gallery
    # a card in the log goes into a turn (the last when none is open: it arrives between turns), so it folds away with that turn's text behind "N earlier turns" (Marc, 2026-11-21);
    # put straight in the log it was outside every turn and stayed on screen. Met in a real page: 2 told turns, a picture, a third told turn -> the picture hidden with its turn
    assert 'const turn = currentTurn || turnElements[turnElements.length - 1] || document.getElementById("player-log");' in ask
    assert "turn.appendChild(pictureFigure(p, false))" in ask
    poll = page.split("witnessedPollTimer = setInterval(")[1].split("}, WITNESSED_POLL_MS")[0]
    assert "pollPictures();" in poll and "pollPortrait();" in poll                                 # asked on the page's own poll
    assert 'id="portrait-panel"' in page and "Keep this as your portrait? It is final." in page and 'This is me' in page
    assert "/portrait/${path}" in page and "portrait/image?rev=" in page
    assert 'portraitDo("draft"' in page and 'portraitDo("edit"' in page and 'portraitDo("back")' in page and 'portraitDo("accept")' in page
    # "Change it" with nothing typed says so and puts the cursor in the box, and does not ask the server (Marc: "Change it button doesn't do anything")
    send = page.split('document.getElementById("portrait-send").addEventListener("click"')[1].split("});")[0]
    assert 'id="portrait-nudge"' in page and "portraitNudge.classList.remove" in send and "portraitSay.focus()" in send
    assert send.index("portraitNudge.classList.remove") < send.index('portraitDo("edit"')
    # the portrait takes over the whole window while it is being painted and changed, and the page beneath is back when it is kept (Marc, 2026-11-16)
    assert "#portrait-panel:not(.hidden) { position: fixed; inset: 0;" in page and 'id="portrait-stage"' in page
    assert page.index("z-index: 40") < page.index("#lightbox { position: fixed")                        # and a click on the picture in the gallery still opens above it
    assert 'if (!s.enabled || s.accepted) { portraitDone = true; panel.classList.add("hidden"); return; }' in page   # kept: the layer goes, nothing else changes
