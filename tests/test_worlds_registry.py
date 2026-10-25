# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from smritilila.worlds_registry import WorldRegistryEntry, load_worlds_registry


def test_load_worlds_registry_parses_entries(tmp_path) -> None:
    config = tmp_path / "worlds.json"
    config.write_text(
        json.dumps(
            {
                "worlds": [
                    {"name": "Frontier", "database_url": "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila"},
                    {"name": "Ravenmoor Test", "database_url": "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_ravenmoor"},
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = load_worlds_registry(config)

    assert entries == [
        WorldRegistryEntry("Frontier", "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila"),
        WorldRegistryEntry("Ravenmoor Test", "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_ravenmoor"),
    ]


def test_load_worlds_registry_returns_empty_list_for_a_missing_file(tmp_path) -> None:
    entries = load_worlds_registry(tmp_path / "does_not_exist.json")

    assert entries == []


def test_load_worlds_registry_returns_empty_list_for_an_empty_worlds_key(tmp_path) -> None:
    config = tmp_path / "worlds.json"
    config.write_text(json.dumps({"worlds": []}), encoding="utf-8")

    entries = load_worlds_registry(config)

    assert entries == []


def _two_worlds(tmp_path):
    config = tmp_path / "worlds.json"
    config.write_text(json.dumps({"worlds": [{"name": "Stonemarch", "database_url": "g", "description": "a line"},
                                             {"name": "Crossing", "database_url": "c", "port": 8199}]}), encoding="utf-8")
    return config


def test_the_guest_setting_is_written_under_the_lock_like_every_other_write_to_worlds_json(tmp_path, monkeypatch) -> None:
    """worlds.json has several writers (each server's daily description, a world being added, the admin screen's guests) and each rewrites the
    whole file: one that skips the lock can write back a copy read before another's change, and the admin screen would report a setting that
    is not there (found 2026-10-22: set_world_guests alone read and wrote it bare). With the lock held by another writer it must wait, and
    give up with TimeoutError rather than write around it; it writes the file whole, as set_world_description does."""
    from smritilila import worlds_registry as wr
    config = _two_worlds(tmp_path)
    monkeypatch.setattr(wr, "_LOCK_WAIT_SECONDS", 0.2)
    before = config.read_text(encoding="utf-8")
    with wr._locked(config):
        with pytest.raises(TimeoutError):
            wr.set_world_guests("Stonemarch", ["Friend@example.com "], config)
    assert config.read_text(encoding="utf-8") == before                         # nothing written around the lock
    whole = []
    real = wr._write_whole
    monkeypatch.setattr(wr, "_write_whole", lambda p, d: (whole.append(p), real(p, d))[1])
    assert wr.set_world_guests("Stonemarch", ["Friend@example.com "], config) is True
    assert whole == [config]                                                    # through the write-aside-and-replace, not a bare write
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["worlds"][0]["guests"] == ["friend@example.com"] and data["worlds"][0]["description"] == "a line"
    assert data["worlds"][1] == {"name": "Crossing", "database_url": "c", "port": 8199}   # the other entry untouched
    assert not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp"))   # nothing left beside it
    assert wr.set_world_guests("Nowhere", True, config) is False
