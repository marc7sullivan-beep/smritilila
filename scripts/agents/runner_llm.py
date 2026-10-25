# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The generic bot runner (Marc and Anu, 2026-10-17): each bot is scripts/agents/llm_player.py, which plays its character through the engine's own LLM client.

Which model is not decided here: it is the one the engine is configured with for its one-off calls (an the first provider key, else a local model, else the second model), or
the model id in SMRITILILA_BOT_MODEL. The supervisor (scripts/bots_supervisor.py) picks a runner by SMRITILILA_BOT_RUNNER; this one is `llm`, and a runner is a file
scripts/agents/runner_<name>.py with `launch` and `running_ids`. Each agent writes its process id to <its folder>/agent.pid and removes it when it ends.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # a console program started from a process with no console opens a visible window; this keeps it hidden


def _python() -> str:
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe") and Path(exe[:-5] + exe[-4:]).exists():   # the supervisor runs under pythonw; the agent wants a console-less python.exe
        return exe[:-5] + exe[-4:]
    return exe


def launch(world: str, eid: str, name: str, port: int, folder: Path, env: dict, max_actions: int, idle: int) -> subprocess.Popen:
    args = [_python(), str(HERE / "llm_player.py"), str(folder), "--id", eid, "--name", name, "--port", str(port), "--world", world,
            "--max", str(max_actions), "--idle", str(idle)]
    log = open(folder / "agent.log", "ab")
    return subprocess.Popen(args, cwd=str(folder), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=_NOWIN)


def alive(pid: int) -> bool:
    if os.name == "nt":                                    # os.kill(pid, 0) is not a test for a process on Windows
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=30,
                                 creationflags=_NOWIN).stdout
        except Exception:  # noqa: BLE001
            return False
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def running_ids(folders: Path) -> set[str]:
    """Character ids whose agent is running now, from the process ids the agents left in their folders. A supervisor that is restarted does not
    know the agents the one before it started; this tells it, so it does not start a second one for a character already being played."""
    found: set[str] = set()
    if not folders.exists():
        return found
    for pidfile in folders.glob("*/agent.pid"):
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if alive(pid):
            found.add(pidfile.parent.name)
    return found
