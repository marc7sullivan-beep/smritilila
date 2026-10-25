# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The hold runner (Marc and Anu, 2026-10-17): a held bot is scripts/agents/hold_player.py, which keeps its character present and acts for no one.

It is not chosen by SMRITILILA_BOT_RUNNER: the supervisor uses it for any bot the registry lists as held (bots_registry.load_hold), whichever runner plays the rest.
Like runner_llm.py's agents, each holds a pid file at <its folder>/agent.pid and removes it when it ends.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from runner_llm import _NOWIN, _python, running_ids  # noqa: F401 -- the same pid files, found the same way

HERE = Path(__file__).resolve().parent


def launch(world: str, eid: str, name: str, port: int, folder: Path, env: dict, max_actions: int, idle: int) -> subprocess.Popen:
    args = [_python(), str(HERE / "hold_player.py"), str(folder), "--id", eid, "--name", name, "--port", str(port)]
    log = open(folder / "agent.log", "ab")
    return subprocess.Popen(args, cwd=str(folder), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=_NOWIN)
