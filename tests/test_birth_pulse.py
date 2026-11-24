# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The fourth route to a birth memory (Marc, 2026-11-08: "Wire in the narrator in-pulse minting"): a telling the Narrator writes gives a perceiver its
first tuple -- a person it minted, or a being the world held no memory of until now -- and the engine writes that being's birth into the same commit
(routes/spanda.py `_births`; smrti/bija.py `first_memories`, `birth_rows`, `births_beside`). The builder's routes are tests/test_birth.py's."""

from __future__ import annotations

import asyncio
import copy
import time

import pytest
from fastapi.testclient import TestClient

from smritilila import adhisthana
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila import spanda
from smritilila.smrti import bija
from smritilila.smrti.bija import birth_rows, births_beside, first_memories
from smritilila.smrti.log import build_log
from smritilila.smrti.pulse import SmrtiPulseResult
from test_birth import BIRTH, _Writer
from test_main import _fake_roles, _seed_square_world, client, fake_store  # noqa: F401 -- the shared fixtures


def _m(mid: str, eid: str, text: str, tier: str = "PRIMARY", tick: int = 5, pid: str = "p1", loc: str = "square") -> Memory:
    return Memory(mid, eid, frozenset(), text, tier, 4, loc, tick, pid, None)


def test_a_first_memory_is_a_perceivers_tuple_when_the_world_held_none_of_any_tier() -> None:
    written = [_m("a1", "newcomer", "Someone near me looks about."), _m("a2", "newcomer", "I step back from the well.", tick=6),
               _m("a3", "veteran", "I greet the miller."), _m("a4", "square", "Two people in the square.", tier="AMBIENT", loc="square"),
               _m("a5", "second_wind", "I hear it.", tier="SECONDARY", tick=5)]
    got = first_memories(written, lambda eid: eid in {"veteran", "square"})
    assert {k: [m.id for m in v] for k, v in got.items()} == {"newcomer": ["a1", "a2"], "second_wind": ["a5"]}   # in stream order; a scene's row and a veteran are nobody's first
    assert first_memories([], lambda eid: False) == {} and first_memories([_m("a4", "square", "x", tier="AMBIENT")], lambda eid: False) == {}


def test_a_birth_row_is_one_tick_before_its_first_in_a_pulse_of_its_own_with_its_place_and_minutes() -> None:
    first = Memory("zz", "npc", frozenset(), "I look about.", "PRIMARY", 5, "square", 40, "p9", None, began=600, ended=615)
    rows = birth_rows("npc", BIRTH, first, "p9", lambda t: [float(len(t))])
    assert [r.experience for r in rows] == [b["text"] for b in BIRTH] and [r.salience for r in rows] == [2, 4, 1] and [r.use for r in rows] == ["useful", "useful", "key"]
    # a pulse id of its own: a weighty birth under the telling's id would make that telling heavier to the news (janasruti), and a birth is nobody's news
    assert {r.world_time for r in rows} == {39} and {r.pulse_id for r in rows} == {"birth:p9"} and {r.location_id for r in rows} == {"square"}
    assert {(r.began, r.ended) for r in rows} == {(600, 615)} and {r.tier for r in rows} == {"PRIMARY"} and {r.subject_ids for r in rows} == {frozenset()}
    assert [r.id for r in rows] == sorted(r.id for r in rows) and len({r.id for r in rows}) == 3 and all(r.embedding == [float(len(r.experience))] for r in rows)
    assert birth_rows("npc", BIRTH, first, "p9", lambda t: None)[0].id != rows[0].id   # each call its own id prefix


def test_the_writer_is_shown_at_most_three_first_tuples_and_a_failed_being_is_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bija, "log_event", lambda *a, **k: None)
    firsts = {"a": [_m(f"a{i}", "a", f"Tuple {i} of the first being, long enough to be one.") for i in range(5)],
              "b": [_m("b1", "b", "The second being's only tuple, long enough to be one.", tick=7)]}
    w = _Writer(fail=("THE BEING: the second",))
    rows = births_beside(w, firsts, {"a": ("the first being", "the square"), "b": ("the second being", "a lane")}, "p1", lambda t: None,
                         bhuvana="THE YEAR. A town.", darsana=None, source="Fiction: a town")
    assert {r.entity_id for r in rows} == {"a"} and len(rows) == 3 and {r.world_time for r in rows} == {4}
    shown = next(u for u in w.users if "THE BEING: the first being" in u)
    assert "WHERE IT IS: the square" in shown and "Tuple 0" in shown and "Tuple 2" in shown and "Tuple 3" not in shown
    assert "THIS WORLD'S SOURCE: Fiction: a town" in shown


def _telling(client: TestClient, monkeypatch: pytest.MonkeyPatch, writer=None):
    """The square world, a pulse's result that mints a stranger and gives it a first tuple, and roles that hold a birth writer."""
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: type("E", (), {"embed": staticmethod(lambda t: None)})())
    world = _seed_square_world(client)
    world.world_data = {"source": "Test: a town square"}
    working = copy.deepcopy(world.canon)
    working.world_time += 1
    t = working.world_time
    stranger = Entity("stranger_ab12", ["a stranger in grey"], t, "smrti")
    working.entities[stranger.id] = stranger
    result = SmrtiPulseResult("You look around.", [_m("m-p", "player", "I look around the square.", tick=t),
                                                   _m("m-s", "stranger_ab12", "Someone near me looks about the square.", tick=t),
                                                   _m("m-sq", "square", "Someone looks around the square.", tier="AMBIENT", tick=t)],
                              [stranger], "", 3, ["player"])
    roles = _fake_roles()
    roles.smrti_client = writer if writer is not None else _Writer()
    return world, working, result, roles


def test_a_players_telling_commits_a_minted_beings_birth_in_the_same_commit_before_its_first_tuple(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    writer = _Writer()
    world, working, result, roles = _telling(client, monkeypatch, writer)
    asyncio.run(spanda._commit_smrti_result(world, working, result, "player", "p1", roles=roles))
    assert len(writer.users) == 1 and "THE BEING: a stranger in grey" in writer.users[0] and "WHERE IT IS: Town Square" in writer.users[0]   # the player had memories already
    assert "  - Someone near me looks about the square." in writer.users[0] and "THIS WORLD'S SOURCE: Test: a town square" in writer.users[0]
    assert len(fake_store.commit_calls) == 1   # one commit, the births in it
    committed = fake_store.commit_calls[0][1].memories_written
    born = sorted((m for m in committed if m.pulse_id == "birth:p1"), key=lambda m: m.id)
    assert [m.entity_id for m in born] == ["stranger_ab12"] * 3 and [m.experience for m in born] == [b["text"] for b in BIRTH]
    assert {m.world_time for m in born} == {working.world_time - 1} and {m.location_id for m in born} == {"square"}
    assert [m.id for m in committed if m.pulse_id == "p1"] == ["m-p", "m-s", "m-sq"]   # the telling's own rows untouched
    # the live world holds them, and reads them where a restart would: its name, its beginning, its first tuple
    live = [t.memory for t in world.samskara().log.stream("stranger_ab12")]
    assert live == ["a stranger in grey"] + [b["text"] for b in BIRTH] + ["Someone near me looks about the square."]
    fresh, _ = build_log(world.canon, world.recent_memories, world.records)
    assert [t.memory for t in fresh.stream("stranger_ab12")] == live


def test_an_activation_commits_a_minted_beings_birth_too(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    writer = _Writer()
    world, working, result, roles = _telling(client, monkeypatch, writer)
    owner = world.scene_owner_entity_id
    t = world.canon.world_time + 1
    result.memories_written = [_m("m-a", owner, "I wait in the square.", tick=t, pid="pa"),
                               _m("m-s", "stranger_ab12", "Someone near me looks about the square.", tick=t, pid="pa")]
    monkeypatch.setattr(spanda, "run_smrti_pulse", lambda *a, **k: result)
    asyncio.run(spanda._run_smrti_activation_in_background(world, roles, owner, 1))
    assert len(writer.users) == 1
    born = [m for m in fake_store.commit_calls[-1][1].memories_written if m.pulse_id.startswith("birth:")]
    assert len(born) == 3 and {m.entity_id for m in born} == {"stranger_ab12"} and {m.world_time for m in born} == {t - 1}
    assert [m for m in world.recent_memories if m.pulse_id.startswith("birth:")] == born


def test_a_being_an_earlier_telling_minted_with_no_tuple_is_given_its_birth_when_it_first_has_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    writer = _Writer()
    world, working, result, roles = _telling(client, monkeypatch, writer)
    world.canon.entities["stranger_ab12"] = working.entities["stranger_ab12"]   # it is in the world already, and has never had a memory
    result.new_entities = []
    asyncio.run(spanda._commit_smrti_result(world, working, result, "player", "p1", roles=roles))
    assert len(writer.users) == 1
    # a second tuple of its own later is not a first: nothing is asked, nothing is written
    t = working.world_time + 1
    later = SmrtiPulseResult("n", [_m("m-s2", "stranger_ab12", "I follow the lane out of the square.", tick=t, pid="p2")], [], "", 3, ["player"])
    w2 = copy.deepcopy(working)
    w2.world_time = t
    asyncio.run(spanda._commit_smrti_result(world, w2, later, "player", "p2", roles=roles))
    assert len(writer.users) == 1 and not [m for m in world.recent_memories if m.pulse_id == "birth:p2"]


def test_a_telling_with_no_first_tuple_asks_nothing(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    writer = _Writer()
    world, working, result, roles = _telling(client, monkeypatch, writer)
    result.memories_written = [m for m in result.memories_written if m.entity_id != "stranger_ab12"]   # minted, with no tuple of its own: no first memory
    asyncio.run(spanda._commit_smrti_result(world, working, result, "player", "p1", roles=roles))
    assert writer.users == [] and not [m for m in fake_store.commit_calls[0][1].memories_written if m.pulse_id.startswith("birth:")]


def test_no_roles_means_no_births_and_no_call(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    world, working, result, _ = _telling(client, monkeypatch)
    asyncio.run(spanda._commit_smrti_result(world, working, result, "player", "p1"))
    assert len(fake_store.commit_calls[0][1].memories_written) == 3


def test_a_birth_that_fails_or_is_waited_on_too_long_never_fails_or_holds_the_telling(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    events: list[tuple] = []
    monkeypatch.setattr(spanda, "log_event", lambda name, **kw: events.append((name, kw)))
    monkeypatch.setattr(bija, "log_event", lambda name, **kw: events.append((name, kw)))
    # the model slips: the telling is committed as it was, and the miss is counted
    world, working, result, roles = _telling(client, monkeypatch, _Writer(fail=("THE BEING",)))
    asyncio.run(spanda._commit_smrti_result(world, working, result, "player", "p1", roles=roles))
    assert [m.id for m in fake_store.commit_calls[0][1].memories_written] == ["m-p", "m-s", "m-sq"]
    assert ("smrti.births", {"pulse_id": "p1", "asked": 1, "written": 0}) in events and any(n == "bija.birth_failed" for n, _ in events)

    # the call does not come back in time: the telling commits without it, and does not wait for it
    class Slow(_Writer):
        def complete_with_tool(self, *a, **k):
            time.sleep(0.6)
            return super().complete_with_tool(*a, **k)

    monkeypatch.setattr(spanda, "BIRTH_WAIT_SECONDS", 0.05)
    world, working, result, roles = _telling(client, monkeypatch, Slow())
    async def timed() -> float:  # inside the loop: asyncio.run itself waits for the abandoned thread when it closes, which a running server never does
        started = time.monotonic()
        await spanda._commit_smrti_result(world, working, result, "player", "p2", roles=roles)
        return time.monotonic() - started

    assert asyncio.run(timed()) < 0.4 and [m.id for m in fake_store.commit_calls[-1][1].memories_written] == ["m-p", "m-s", "m-sq"]
    assert any(n == "smrti.births_failed" and kw["pulse_id"] == "p2" for n, kw in events)
