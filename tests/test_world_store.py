# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from smritilila import world_store
from smritilila.world_store import InMemoryWorldStore, PostgresWorldStore, make_world_store


@pytest.fixture(autouse=True)
def _clean_engine_cache():
    saved = dict(world_store._engines_by_url)
    world_store._engines_by_url.clear()
    yield
    world_store._engines_by_url.clear()
    world_store._engines_by_url.update(saved)


def test_make_world_store_returns_in_memory_store_with_no_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    store = make_world_store()

    assert isinstance(store, InMemoryWorldStore)


def test_make_world_store_caches_one_engine_per_url() -> None:
    url_a = "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_test_a"
    url_b = "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_test_b"

    make_world_store(url_a)
    make_world_store(url_a)
    make_world_store(url_b)

    assert set(world_store._engines_by_url) == {url_a, url_b}


def test_make_world_store_does_not_keep_returning_the_first_url_it_was_ever_called_with() -> None:
    """Regression test: a bare module-level `_engine` global (fixed
    2026-09-11, alongside smritilila.worlds_registry) silently kept serving
    whichever url was passed on the FIRST call, ignoring every later
    one -- switching worlds would have kept querying the original
    database forever. Each distinct url must get its OWN engine, not
    share one cached from an earlier, different call."""
    url_a = "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_test_a"
    url_b = "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_test_b"

    make_world_store(url_a)
    engine_a = world_store._engines_by_url[url_a]
    make_world_store(url_b)

    assert world_store._engines_by_url[url_b] is not engine_a


def test_make_world_store_returns_a_postgres_store_for_a_url() -> None:
    store = make_world_store("postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_test_a")

    assert isinstance(store, PostgresWorldStore)
