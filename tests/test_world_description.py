# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""A world's one line, drawn again once per real-time day (Marc, 2026-10-20: "the one sentence world description should be updated once per 'real time' day").

The line was drawn once, from the birth year, when the root was laid: Ravenmoor's still said "April 27, 1928" after its earth had written June and July. Now each server draws its own
world's line again when a day has passed since the last draw (`description_at` in worlds.json), from the root and the earth's newest entries; a failed draw leaves the line and is tried again within the hour.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from smritilila.routes import bija as bija_routes
from smritilila.worlds_registry import add_world, load_worlds_registry, set_world_description

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def test_a_line_is_due_when_it_has_no_time_or_a_day_has_passed() -> None:
    due = lambda at: bija_routes._describe_due(SimpleNamespace(described_at=at), NOW)
    assert due(None) and due("") and due("not a time")                               # never drawn with a time, or unreadable: draw
    assert not due((NOW - timedelta(hours=23, minutes=59)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    assert due((NOW - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    assert due((NOW - timedelta(days=9)).strftime("%Y-%m-%dT%H:%M:%SZ"))


def test_the_registry_keeps_when_a_line_was_drawn(tmp_path) -> None:
    cfg = tmp_path / "worlds.json"
    cfg.write_text(json.dumps({"worlds": [{"name": "Ravenmoor", "database_url": "postgresql+asyncpg://x/l", "description": "It is April."}]}), encoding="utf-8")
    assert load_worlds_registry(cfg)[0].described_at is None
    assert set_world_description("Ravenmoor", "It is July.", path=cfg, at=NOW)
    entry = load_worlds_registry(cfg)[0]
    assert entry.description == "It is July." and entry.described_at == "2026-10-20T12:00:00Z"
    assert json.loads(cfg.read_text(encoding="utf-8"))["worlds"][0]["description_at"] == "2026-10-20T12:00:00Z"
    assert not set_world_description("Nowhere", "x", path=cfg)                      # no such world: nothing written
    assert not (tmp_path / "worlds.json.lock").exists()                               # the lock is let go


def test_servers_writing_the_registry_at_once_lose_nothing_and_never_leave_half_a_file(tmp_path) -> None:
    """Six servers each write their own line once a day, and a builder may add a world: each rewrites the whole file."""
    cfg = tmp_path / "worlds.json"
    cfg.write_text(json.dumps({"worlds": [{"name": f"W{i}", "database_url": f"postgresql+asyncpg://x/w{i}"} for i in range(6)]}), encoding="utf-8")
    errors: list[BaseException] = []

    def describe(i: int) -> None:
        try:
            for n in range(15):
                assert set_world_description(f"W{i}", f"line {i}.{n}", path=cfg)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def add(i: int) -> None:
        try:
            add_world(f"Added{i}", f"postgresql+asyncpg://x/a{i}", path=cfg)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=describe, args=(i,)) for i in range(6)] + [threading.Thread(target=add, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:2]
    got = {w.name: w for w in load_worlds_registry(cfg)}
    assert [f"Added{i}" in got for i in range(6)] == [True] * 6                       # every added world is there
    assert [got[f"W{i}"].description for i in range(6)] == [f"line {i}.14" for i in range(6)]   # and every world's last line
    assert not list(tmp_path.glob("*.tmp")) and not (tmp_path / "worlds.json.lock").exists()


def _world(bhuvana: str | None, earth_entries: list[str], setting: str = "A town."):
    """The parts of a world `_root_of` reads: its age, its earth, and the earth's tuples."""
    from smritilila.memory.memory import Memory
    mems = [Memory(f"m{i}", "earth", frozenset(), t, "PRIMARY", 3, "nowhere", i * 10, f"p{i}", None) for i, t in enumerate(earth_entries)]
    return SimpleNamespace(bhuvana=bhuvana, setting_description=setting, mems=mems)


def _root(world, monkeypatch) -> str:
    monkeypatch.setattr(bija_routes, "_chronicle_id", lambda w: "earth")
    monkeypatch.setattr(bija_routes, "_stream_of", lambda w, eid: list(w.mems))
    monkeypatch.setattr(bija_routes, "_age_tuples", lambda w: list(w.mems))
    return bija_routes._root_of(world)


def test_the_root_a_line_is_drawn_from_carries_the_ages_newest_entries(monkeypatch) -> None:
    year = "THE YEAR\n\nIt is 1928, spring, and in Arkham it is Friday, the twenty-seventh of April."
    bhuvana = year + "\n\nTHE REGIONS\n\nArkham and its hills."
    moved = _world(bhuvana, [year, "LATE APRIL AND MAY, 1928. Walpurgis Eve passed on Monday.", "JUNE AND JULY, 1928. Mr. Hoover was nominated at Kansas City."])
    root = _root(moved, monkeypatch)
    assert root.startswith(year) and "THE AGE SINCE" in root
    assert "JUNE AND JULY, 1928" in root and "LATE APRIL AND MAY, 1928" in root                # the newest two
    quiet = _world(bhuvana, [year])
    assert _root(quiet, monkeypatch) == year                                                    # the age has not moved: the root as before
    born_quiet = _world(None, ["A ferry crossing on the river, late autumn."])
    assert _root(born_quiet, monkeypatch) == "A ferry crossing on the river, late autumn."     # no age: the earth's first memory, as before


def _serve(monkeypatch, entry, calls):
    world = SimpleNamespace(name=entry.name)

    async def get_world():
        calls.append("loaded")
        return world

    async def describe(w, roles):
        calls.append("described")
        return "It is July 1928 in Arkham."
    monkeypatch.setattr(bija_routes.adhisthana, "_active_world_registry_entry", lambda: entry)
    monkeypatch.setattr(bija_routes.adhisthana, "get_world", get_world)
    monkeypatch.setattr(bija_routes.adhisthana, "_resolve_roles", lambda: object())
    monkeypatch.setattr(bija_routes, "_describe", describe)
    bija_routes._describe_attempted.clear()


def test_a_server_draws_its_line_when_a_day_has_passed_and_not_before(monkeypatch) -> None:
    calls: list[str] = []
    fresh = SimpleNamespace(name="Ravenmoor", described_at=(NOW - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _serve(monkeypatch, fresh, calls)
    assert asyncio.run(bija_routes.describe_if_due(NOW)) is None and calls == []             # not due: the world is not even loaded
    stale = SimpleNamespace(name="Ravenmoor", described_at=(NOW - timedelta(hours=25)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _serve(monkeypatch, stale, calls)
    assert asyncio.run(bija_routes.describe_if_due(NOW)) == "It is July 1928 in Arkham." and calls == ["loaded", "described"]
    unregistered = None
    monkeypatch.setattr(bija_routes.adhisthana, "_active_world_registry_entry", lambda: unregistered)
    calls.clear()
    assert asyncio.run(bija_routes.describe_if_due(NOW)) is None and calls == []              # a world outside the registry has no line to keep


def test_a_draw_that_failed_is_tried_again_within_the_hour_not_at_the_next_look(monkeypatch) -> None:
    calls: list[str] = []
    stale = SimpleNamespace(name="Stonemarch", described_at=None)
    _serve(monkeypatch, stale, calls)
    answers = iter([None, "It is autumn in the Flanaess."])

    async def describe(w, roles):
        calls.append("described")
        return next(answers)
    monkeypatch.setattr(bija_routes, "_describe", describe)
    t = [1000.0]
    clock = lambda: t[0]
    assert asyncio.run(bija_routes.describe_if_due(NOW, clock)) is None and calls.count("described") == 1       # it failed
    t[0] += bija_routes.DESCRIBE_CHECK_EVERY
    assert asyncio.run(bija_routes.describe_if_due(NOW, clock)) is None and calls.count("described") == 1       # the next look: not yet
    t[0] += bija_routes.DESCRIBE_RETRY_AFTER
    assert asyncio.run(bija_routes.describe_if_due(NOW, clock)) == "It is autumn in the Flanaess." and calls.count("described") == 2


def test_the_server_starts_the_daily_draw_and_stops_it(monkeypatch) -> None:
    """The lifespan starts `describe_daily` and cancels it when the server stops."""
    from smritilila import adhisthana
    seen: dict[str, object] = {}

    async def daily():
        seen["started"] = True
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise
    monkeypatch.setattr(bija_routes, "describe_daily", daily)
    monkeypatch.setattr(adhisthana, "get_embedder", lambda: None)
    monkeypatch.setattr(adhisthana, "_recall_active_world", lambda: None)

    async def run() -> None:
        async with adhisthana.lifespan(adhisthana.app):
            await asyncio.sleep(0.05)
            assert seen.get("started")
    asyncio.run(run())
    assert seen.get("cancelled")
