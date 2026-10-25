# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess

import pytest

from smritilila.backup import backup_all_worlds, backup_world
from smritilila.worlds_registry import WorldRegistryEntry


def _fake_run_factory(monkeypatch: pytest.MonkeyPatch, calls: list):
    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("smritilila.backup.subprocess.run", fake_run)


def test_backup_world_invokes_pg_dump_with_a_converted_url(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list = []
    _fake_run_factory(monkeypatch, calls)
    entry = WorldRegistryEntry("Frontier", "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila")

    path = backup_world(entry, reason="test", backups_dir=tmp_path)

    assert path.exists() is False  # fake_run never actually wrote a file -- just confirms the invocation shape
    [args] = calls
    assert "-Fc" in args
    assert str(path) in args
    assert "postgresql://smritilila:smritilila@localhost/smritilila" in args
    assert not any("+asyncpg" in a for a in args)


def test_backup_world_writes_into_a_per_world_subdirectory_named_safely(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _fake_run_factory(monkeypatch, [])
    entry = WorldRegistryEntry("Ravenmoor Test!", "postgresql+asyncpg://x/smritilila_ravenmoor")

    path = backup_world(entry, reason="test", backups_dir=tmp_path)

    assert path.parent == tmp_path / "Ravenmoor_Test_"
    assert path.parent.is_dir()


def test_backup_world_prunes_old_backups_beyond_the_retention_limit(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _fake_run_factory(monkeypatch, [])
    entry = WorldRegistryEntry("Frontier", "postgresql+asyncpg://x/smritilila")
    world_dir = tmp_path / "Frontier"
    world_dir.mkdir()
    for i in range(35):
        (world_dir / f"Frontier_2026010{i:02d}_scheduled.dump").write_text("x")

    backup_world(entry, reason="scheduled", backups_dir=tmp_path)

    remaining = list(world_dir.glob("*.dump"))
    assert len(remaining) == 30  # KEEP_PER_WORLD (fake_run wrote no dump, so the just-created one is not among them)


def test_backup_world_raises_on_pg_dump_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="pg_dump: connection refused")

    monkeypatch.setattr("smritilila.backup.subprocess.run", fake_run)
    entry = WorldRegistryEntry("Frontier", "postgresql+asyncpg://x/smritilila")

    with pytest.raises(subprocess.CalledProcessError):
        backup_world(entry, reason="test", backups_dir=tmp_path)


def test_backup_all_worlds_is_best_effort_across_a_single_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    entries = [
        WorldRegistryEntry("Frontier", "postgresql+asyncpg://x/smritilila"),
        WorldRegistryEntry("Ravenmoor Test", "postgresql+asyncpg://x/smritilila_ravenmoor"),
    ]
    monkeypatch.setattr("smritilila.backup.load_worlds_registry", lambda: entries)

    def fake_run(args, **kwargs):
        if "smritilila_ravenmoor" in args[-1]:
            raise subprocess.CalledProcessError(1, args, stderr="connection refused")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("smritilila.backup.subprocess.run", fake_run)

    results = backup_all_worlds(reason="test", backups_dir=tmp_path)

    by_name = {r.world_name: r for r in results}
    assert by_name["Frontier"].path is not None
    assert by_name["Frontier"].error is None
    assert by_name["Ravenmoor Test"].path is None
    assert "connection refused" in by_name["Ravenmoor Test"].error


def test_backup_all_worlds_returns_empty_list_when_no_registry_configured(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("smritilila.backup.load_worlds_registry", lambda: [])

    results = backup_all_worlds(reason="test", backups_dir=tmp_path)

    assert results == []
