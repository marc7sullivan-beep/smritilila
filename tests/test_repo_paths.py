# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Every path the engine computes from its own file lands where it was meant to (2026-10-06).

When main.py was divided on 2026-10-05, routes/loka.py and routes/pratistha.py kept `Path(__file__).resolve().parent.parent.parent`,
which was the repo root in main.py and is src/ one folder deeper. Nothing failed: the tests fake the spawn and the proposals. The
first real Build then found no world could start its own server (the marker and scripts/dev_server.py were looked for under src/),
and proposals would have been written to src/proposals. A moved function is moved with its bearings."""
import subprocess
from pathlib import Path

import pytest

from smritilila import adhisthana
from smritilila.routes import loka, pratistha

REPO = Path(__file__).resolve().parent.parent


def test_the_repo_root_the_routes_use_is_the_repo():
    assert loka.REPO_ROOT == REPO
    assert (loka.REPO_ROOT / "scripts" / "dev_server.py").is_file() and (loka.REPO_ROOT / "src" / "smritilila" / "main.py").is_file()


def test_proposals_are_kept_in_the_repos_proposals_folder_not_under_src():
    assert pratistha.PROPOSALS_DIR == REPO / "proposals"


def test_the_servers_marker_and_alembic_config_are_at_the_repo_root(monkeypatch):
    monkeypatch.delenv("SMRITILILA_ACTIVE_WORLD_MARKER", raising=False)
    monkeypatch.setenv("SMRITILILA_DEV_PORT", "8198")
    assert adhisthana._active_world_marker() == REPO / ".smritilila_active_world.8198"
    assert Path(adhisthana._alembic_config().config_file_name) == REPO / "alembic.ini" and (REPO / "alembic.ini").is_file()


def test_a_world_server_is_started_from_the_repo_root(monkeypatch, tmp_path):
    """The spawn names its world in a marker beside the repo and runs scripts/dev_server.py from there -- with the root where a test
    can see it, so a wrong root fails here and not in the first world someone builds."""
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(loka, "REPO_ROOT", tmp_path)
    started = []
    monkeypatch.setattr(loka.subprocess, "Popen", lambda args, **kw: started.append((args, kw)))
    monkeypatch.setattr(loka, "log_event", lambda *a, **k: None)
    entry = loka.WorldRegistryEntry(name="Tide Hall", database_url="postgresql+asyncpg://x/tide_hall", port=8231)
    loka._spawn_world_server(entry)
    assert (tmp_path / ".smritilila_active_world.8231").read_text(encoding="utf-8") == "Tide Hall"                  # which world this port serves
    args, kw = started[0]
    assert args[1] == str(tmp_path / "scripts" / "dev_server.py") and kw["env"]["SMRITILILA_DEV_PORT"] == "8231"      # and what runs on it
    assert "SMRITILILA_ACTIVE_WORLD_MARKER" not in kw["env"]


def test_no_module_looks_for_the_repo_one_folder_too_shallow():
    """The slip itself: a `.parent.parent.parent` from a file in src/smritilila/routes/ is src/, never the repo root."""
    import re
    shallow = re.compile(r"(?<!\.parent)\.parent\.parent\.parent\s*/")                  # exactly three, not the tail of four
    for f in sorted((REPO / "src" / "smritilila" / "routes").glob("*.py")):
        assert not shallow.search(f.read_text(encoding="utf-8")), f.name


def _paths(env_extra):
    """Every default place state is kept, as a fresh interpreter sees them (they are set at import)."""
    import json
    import os
    import subprocess
    import sys
    code = ("import json,os;from smritilila import adhisthana,backup,users_registry,bots_registry,worlds_registry;"
            "from smritilila.routes import sthapati,pratistha;"
            "print(json.dumps({'worlds':str(worlds_registry._DEFAULT_CONFIG_PATH),'invites':str(worlds_registry._DEFAULT_INVITES_PATH),"
            "'users':str(users_registry._DEFAULT_PATH),'bots':str(bots_registry._DEFAULT_PATH),'backups':str(backup.BACKUPS_DIR),"
            "'sessions':str(sthapati.SESSIONS_DIR),'proposals':str(pratistha.PROPOSALS_DIR),'marker':str(adhisthana._active_world_marker())}))")
    env = {k: v for k, v in os.environ.items() if k not in ("SMRITILILA_DATA_DIR", "SMRITILILA_ACTIVE_WORLD_MARKER", "SMRITILILA_WORLDS_CONFIG")}
    env.update(env_extra, PYTHONPATH=str(REPO / "src"), PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    return {k: Path(v) for k, v in json.loads(out.stdout.strip().splitlines()[-1]).items()}


def test_state_is_kept_at_the_repository_root_unless_a_data_folder_is_named():
    got = _paths({})
    assert got["worlds"] == REPO / "worlds.json" and got["users"] == REPO / "users.json" and got["bots"] == REPO / "bots.json"
    assert got["backups"] == REPO / "backups" and got["sessions"] == REPO / "builder_sessions" and got["proposals"] == REPO / "proposals"
    assert got["invites"] == REPO / "invites.json" and got["marker"].parent == REPO and got["marker"].name.startswith(".smritilila_active_world.")


def test_a_data_folder_moves_all_of_it_to_one_place(tmp_path):
    got = _paths({"SMRITILILA_DATA_DIR": str(tmp_path)})
    for name, path in got.items():
        assert path.parent == tmp_path, (name, path)                                   # a container's volume holds every bit of state
    assert got["worlds"].name == "worlds.json" and got["backups"].name == "backups" and got["sessions"].name == "builder_sessions"


def test_a_world_server_started_by_the_builder_looks_for_its_marker_in_the_data_folder(monkeypatch, tmp_path):
    """The marker the spawn writes and the one the server reads must be the same file, wherever state lives."""
    monkeypatch.setenv("SMRITILILA_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(loka, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(loka.subprocess, "Popen", lambda args, **kw: None)
    monkeypatch.setattr(loka, "log_event", lambda *a, **k: None)
    monkeypatch.delenv("SMRITILILA_ACTIVE_WORLD_MARKER", raising=False)
    entry = loka.WorldRegistryEntry(name="Tide Hall", database_url="postgresql+asyncpg://x/t", port=8231)
    loka._spawn_world_server(entry)
    monkeypatch.setenv("SMRITILILA_DEV_PORT", "8231")
    assert (tmp_path / "data" / ".smritilila_active_world.8231").read_text(encoding="utf-8") == "Tide Hall"
    assert adhisthana._active_world_marker() == tmp_path / "data" / ".smritilila_active_world.8231"
