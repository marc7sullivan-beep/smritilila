# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Live-database tests for smritilila.world_store.PostgresWorldStore (v3): the
load()/commit() round trip, the clock from the last tuple, absorb, clear,
session config and the bhuvana -- nothing else touches a database.

Skipped entirely unless TEST_DATABASE_URL is set. Deliberately its OWN
env var, not DATABASE_URL: this fixture's teardown DELETES EVERY ROW of every table
after each test, and DATABASE_URL is what the real app points at a real,
non-disposable database. A real incident, not a hypothetical -- this
suite was once run with DATABASE_URL left pointed at a live game's own
database, and its teardown wiped it with no backup to recover from.
_assert_disposable below is the second, independent layer: even a
TEST_DATABASE_URL set to the wrong thing by mistake gets refused if the
database name is literally "smritilila".
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.persistence.models import Base
from smritilila.world_store import PostgresWorldStore

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(DATABASE_URL is None, reason="TEST_DATABASE_URL not set -- no disposable live Postgres available")


def _assert_disposable(database_url: str) -> None:
    """Refuses to run this row-deleting fixture against a database
    literally named "smritilila" -- the real app's own database name -- no
    matter which env var pointed at it."""
    db_name = database_url.rsplit("/", 1)[-1].split("?", 1)[0]
    if db_name == "smritilila":
        raise RuntimeError(
            "TEST_DATABASE_URL points at a database literally named 'smritilila' -- "
            "refusing to run table-truncating tests against it. Point it at a "
            "disposable database instead (e.g. 'smritilila_test')."
        )


@pytest_asyncio.fixture
async def session_factory():
    _assert_disposable(DATABASE_URL)
    engine = create_async_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    async with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())
    await engine.dispose()


def _mem(i: int, who: str, t: int, text: str, loc: str = "square") -> Memory:
    return Memory(f"m{i}", who, frozenset({who}), text, "PRIMARY", 4, loc, t, f"p{t}", None)


@pytest.mark.asyncio
async def test_load_returns_none_when_nothing_persisted(session_factory) -> None:
    assert await PostgresWorldStore(session_factory).load() is None


@pytest.mark.asyncio
async def test_commit_then_load_round_trips_entities_memories_and_the_clock(session_factory) -> None:
    store = PostgresWorldStore(session_factory)
    canon = Maya(world_time=0)
    mara = Entity("mara", ["Mara"], 0, "genesis")
    square = Entity("square", ["Town Square"], 0, "genesis", avarana="open")
    canon.entities.update({"mara": mara, "square": square})
    await store.commit(canon, [_mem(1, "mara", 3, "I crossed the square."), _mem(2, "square", 3, "A woman crossed.")], [mara, square])
    loaded = await store.load()
    assert loaded is not None and set(loaded.entities) == {"mara", "square"}
    assert loaded.world_time == 3  # v3's clock: the last tuple's tick
    assert [m.experience for m in await store.load_memories()] == ["I crossed the square.", "A woman crossed."]
    assert (await store.last_memory_for("mara")).experience == "I crossed the square."
    assert await store.last_memory_for("nobody") is None


@pytest.mark.asyncio
async def test_touched_ids_persist_the_schedulers_bookkeeping_for_an_already_settled_entity(session_factory) -> None:
    store = PostgresWorldStore(session_factory)
    canon = Maya(world_time=0)
    mara = Entity("mara", ["Mara"], 0, "genesis")
    canon.entities["mara"] = mara
    await store.commit(canon, [], [mara])
    mara.last_simulated_tick = 9
    await store.commit(canon, [], [], touched_ids=["mara"])
    loaded = await store.load()
    assert loaded.entities["mara"].last_simulated_tick == 9


@pytest.mark.asyncio
async def test_absorb_refiles_every_tuple_and_drops_the_id(session_factory) -> None:
    store = PostgresWorldStore(session_factory)
    canon = Maya(world_time=0)
    a, b = Entity("walker", ["a walker"], 0, "smrti"), Entity("walker2", ["the same walker"], 1, "smrti")
    nook = Entity("nook", ["a nook"], 0, "genesis", kosha="walker2")  # the old way: containment as the entity's field
    canon.entities.update({"walker": a, "walker2": b, "nook": nook})
    # v4's records move too (Ravenmoor's Elias copy, 2026-09-24): its own, and what was in it
    recs = [Record("r1", "kosha", "walker2", "square", 2, "p2"), Record("r2", "name", "walker2", "the same walker", 2, "p2"),
            Record("r3", "kosha", "stick", "walker2", 2, "p2"), Record("r4", "kosha", "walker", "square", 1, "p1"),
            Record("r5", "sandhi", "gate", "walker2", 2, "p2", "open"),
            Record("r6", "said", "square", "Walk on.", 2, "p2", "walker2")]  # the gone one's line (Stonemarch's second Ostler)
    await store.commit(canon, [_mem(1, "walker", 1, "I walked."), _mem(2, "walker2", 2, "I walked on.", loc="walker2")], [a, b, nook], records=recs)
    out = await store.absorb("walker", "walker2")
    assert out == {"tuples": 1, "filed": 1, "presence": 1, "records": 2, "contained": 1, "joined": 1, "field": 1, "speaker": 1, "dropped": 1}
    assert [(r.value, r.note) for r in await store.load_records() if r.kind == "said"] == [("Walk on.", "walker")]
    loaded = await store.load()
    assert "walker2" not in loaded.entities and loaded.entities["nook"].kosha == "walker"  # not left in nothing
    assert [(m.entity_id, m.location_id) for m in await store.load_memories()] == [("walker", "square"), ("walker", "walker")]
    assert sorted((r.kind, r.entity_id, r.value) for r in await store.load_records()) == [
        ("kosha", "stick", "walker"), ("kosha", "walker", "square"), ("kosha", "walker", "square"), ("name", "walker", "the same walker"),
        ("said", "square", "Walk on."), ("sandhi", "gate", "walker")]


@pytest.mark.asyncio
async def test_clear_empties_a_previously_committed_world(session_factory) -> None:
    store = PostgresWorldStore(session_factory)
    canon = Maya(world_time=0)
    mara = Entity("mara", ["Mara"], 0, "genesis")
    canon.entities["mara"] = mara
    await store.commit(canon, [_mem(1, "mara", 1, "I stood.")], [mara])
    await store.save_session_config("mara", ["square"], ["mara"])
    assert await store.load() is not None
    await store.clear()
    assert await store.load() is None and await store.load_session_config() is None


@pytest.mark.asyncio
async def test_session_config_and_bhuvana_round_trip(session_factory) -> None:
    store = PostgresWorldStore(session_factory)
    assert await store.load_session_config() is None
    await store.save_session_config("xerxes-uuid", ["camp-uuid"], ["xerxes-uuid"], "A camp on the plain.")
    assert await store.load_session_config() == ("xerxes-uuid", ["camp-uuid"], ["xerxes-uuid"], "A camp on the plain.")
    assert await store.load_bhuvana() is None
    await store.save_bhuvana("THE YEAR. The plain in summer.")
    assert await store.load_bhuvana() == "THE YEAR. The plain in summer."


@pytest.mark.asyncio
async def test_the_player_sets_round_trip_on_the_real_store(session_factory) -> None:
    """2026-10-01: waiting shipped with its repository functions missing from
    this store's imports -- every fake store had them, and every world failed
    to load for five minutes. Each player set, through the real store."""
    store = PostgresWorldStore(session_factory)
    await store.save_session_config("teo", ["inn"], ["teo", "nara"], None)
    await store.save_autonomy(["teo"])
    await store.save_logged_off(["teo", "nara"])
    await store.save_waiting(["teo"])
    await store.save_played_by({"nara": "guest@example.com"})
    assert await store.load_autonomy() == ["teo"] and await store.load_logged_off() == ["nara", "teo"]
    assert await store.load_waiting() == ["teo"] and await store.load_played_by() == {"nara": "guest@example.com"}
    await store.save_held_back(["gold_dragon"])  # 2026-10-02: a guest's retired, out until a god reviews it
    assert await store.load_held_back() == ["gold_dragon"]


def test_migrating_a_world_in_process_leaves_the_instruments_running() -> None:
    """/worlds/create migrates the new world's database inside the server
    process; alembic's env.py used to disable every existing logger, and
    sakshi and the training capture went silent until restart (the telepath
    playtest, 2026-09-23). Migrates a database of its own, made new beside
    TEST_DATABASE_URL's and dropped after: from empty to head, so the whole
    chain runs. (It first ran against TEST_DATABASE_URL itself, "already at
    head, so the upgrade is a no-op" -- true only on the machine that had
    migrated it; on a fresh Postgres the other tests there had made the
    tables with create_all, and the first migration collided: CI, 2026-10-06.)"""
    import asyncio
    import logging
    import uuid

    import asyncpg
    from smritilila.adhisthana import _migrate_database
    for name in ("smritilila.sakshi", "smritilila.training_data"):
        logging.getLogger(name)
    base, _, _ = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1).rpartition("/")
    scratch = f"smritilila_test_migrate_{uuid.uuid4().hex[:8]}"

    async def run_sql(sql: str) -> None:
        conn = await asyncpg.connect(base + "/postgres")
        try:
            await conn.execute(sql)
        finally:
            await conn.close()
    asyncio.run(run_sql(f'CREATE DATABASE "{scratch}"'))
    try:
        _migrate_database(DATABASE_URL.rsplit("/", 1)[0] + "/" + scratch)
        assert not logging.getLogger("smritilila.sakshi").disabled
        assert not logging.getLogger("smritilila.training_data").disabled
    finally:
        assert scratch.startswith("smritilila_test_migrate_")
        asyncio.run(run_sql(f'DROP DATABASE IF EXISTS "{scratch}" WITH (FORCE)'))


@pytest.mark.asyncio
async def test_a_thing_declared_and_moved_in_one_pulse_reloads_where_it_was_moved(session_factory) -> None:
    """The crossing, 587: a bowl declared on the hearthstone and carried out
    to the tin-man in the same pulse. Two records at one tick, random ids:
    on reload the order was a coin flip and the bowl came back to the
    hearth. `declared` sorts first within a tick -- the order it was written."""
    from smritilila.memory.record import Record
    from smritilila.smrti.kosha import Kosha
    store = PostgresWorldStore(session_factory)
    canon = Maya(world_time=5)
    for eid in ("inn", "tin", "bowl"):
        canon.entities[eid] = Entity(eid, [eid], 0, "genesis")
    # ids chosen so id order is the WRONG order
    recs = [Record("z-declared", "kosha", "bowl", "inn", 5, "p5", note="declared"),
            Record("a-moved", "kosha", "bowl", "tin", 5, "p5", note="moved")]
    await store.commit(canon, [], list(canon.entities.values()), records=recs)
    loaded = await store.load_records()
    assert [r.note for r in loaded if r.entity_id == "bowl"] == ["declared", "moved"]
    assert Kosha(canon, [], loaded).container_at("bowl", 6) == "tin"


@pytest.mark.asyncio
async def test_a_sandhi_record_round_trips_and_joins_both_scenes(session_factory) -> None:
    """Live, collide 676 (2026-09-25): the first sandhi a pulse wrote was
    refused by the records kind check -- the fake store had let it through."""
    from smritilila.smrti.kosha import Kosha
    store = PostgresWorldStore(session_factory)
    canon = Maya(world_time=5)
    for eid, scene in (("landing", True), ("room", True), ("door", False)):
        canon.entities[eid] = Entity(eid, [eid], 0, "genesis", avarana="scene" if scene else None)
    recs = [Record("k1", "kosha", "door", "landing", 1, "p1", note="declared"),
            Record("s1", "sandhi", "door", "room", 4, "p4", note="open")]
    await store.commit(canon, [], list(canon.entities.values()), records=recs)
    loaded = await store.load_records()
    assert [(r.kind, r.value, r.note) for r in loaded if r.entity_id == "door"] == [("kosha", "landing", "declared"), ("sandhi", "room", "open")]
    assert Kosha(canon, [], loaded).across("landing", 5) == [("door", "room")]


@pytest.mark.asyncio
async def test_a_secret_record_round_trips_on_the_real_store(session_factory) -> None:
    """2026-10-02: the first secret records were refused by the records kind check -- the fake store had let them
    through, as it once let sandhi through. A pratiṣṭhā seed born secret writes them; the store must keep them."""
    store = PostgresWorldStore(session_factory)
    canon = Maya(world_time=5)
    canon.entities["book"] = Entity("book", ["a little book"], 0, "genesis")
    await store.commit(canon, [], [canon.entities["book"]], records=[Record("x1", "secret", "book", "anu", 5, "seed")])
    assert [(r.kind, r.entity_id, r.value) for r in await store.load_records()] == [("secret", "book", "anu")]
