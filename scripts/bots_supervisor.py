# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The bot supervisor (Marc and Anu, 2026-10-06): starts an agent for each character the admin screen has switched on as a bot whenever a
person is active in its world, and stops it when the last person has gone.

    python scripts/bots_supervisor.py            # watches every world in worlds.json that has a bot in bots.json
    python scripts/bots_supervisor.py --once     # one look and what it would do; starts and stops nothing

It is run on this machine, detached, like the servers. It reads worlds.json and bots.json (the overseer writes bots.json) afresh on every
look, so a toggle takes effect within one interval. An agent is started by a runner (scripts/agents/runner_<name>.py, the name in SMRITILILA_BOT_RUNNER) and
speaks to the world through the tool that does its waiting, posting and counting (scripts/agents/player_tool.py); it plays until its action limit, its
idle timeout, or until told to stop. The `llm` runner plays through the engine's own LLM client: which model is configuration, the same as for the
engine's other calls (SMRITILILA_BOT_MODEL names another model id). It makes LLM calls, so it costs.

A bot the registry marks HELD (bots_registry.load_hold; Marc, 2026-10-17: "bot mode but not taking actions") gets scripts/agents/hold_player.py instead of an
agent that plays: it is started and stopped the same way, its character is kept online while a person is active, and nothing acts for it. No model is asked.

THE TIME GUARD (Marc, 2026-10-22): an agent never moves its character ahead of a person in time (scripts/agents/player_tool.py says how). The supervisor names the world's
other bots to each agent (SMRITILILA_BOT_PEERS: they are not persons to stay behind) and switches the guard off, by SMRITILILA_BOT_TIME_GUARD=0, for a bot the registry marks FREE
(bots_registry.load_free). A free bot is also started with no person in the world and is not stopped when the last person goes: a sim of bots alone, as in the Crossing.
The whole supervisor may be run with SMRITILILA_BOT_TIME_GUARD=0 in its environment to free every agent it starts.

A person is "active" when a character of that world that is not a bot is online (the page's heartbeat). The bots' own characters
never count -- an agent playing is online too. The supervisor never writes a world's memory or database itself: it asks the world's own server, as the
characters' pages do (stopping an agent has the tool post the character's presence and autonomy), and the agent acts through the same endpoints a person's page does.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # a console program started from a process with no console opens a visible window; this keeps it hidden (Marc, 2026-10-10)


def _run_hidden(*a, **k):
    return subprocess.run(*a, creationflags=k.pop("creationflags", _NOWIN), **k)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "agents"))

INTERVAL = 15          # seconds between looks
GRACE = 180            # seconds with no person active before the agents are stopped (a person's page may blink offline)
COOLDOWN = 600         # seconds before an agent that ended is started again (it ended at its limit, or its idle timeout, or by a fault)
MAX_ACTIONS = int(os.environ.get("SMRITILILA_BOT_MAX_ACTIONS", "25"))   # an agent's action limit; a test sets it low
IDLE_MINUTES = 30
FOLDERS = Path(os.environ.get("TEMP", "/tmp")) / "smritilila-players"   # the prompt names this folder, per character
RUNNER = os.environ.get("SMRITILILA_BOT_RUNNER", "llm")          # the file scripts/agents/runner_<RUNNER>.py starts an agent


def runner():
    import importlib
    try:
        return importlib.import_module("runner_" + RUNNER)
    except ModuleNotFoundError as e:
        if e.name != "runner_" + RUNNER:
            raise
        raise SystemExit(f"no bot runner named {RUNNER!r}: SMRITILILA_BOT_RUNNER names a file scripts/agents/runner_<name>.py")


def hold_runner():
    import runner_hold
    return runner_hold


def plan(bots: list[str], humans: bool | None, running: set[str], now: float, gone_since: float | None,
         last_start: dict[str, float], free: tuple[str, ...] | list[str] | set[str] = ()) -> tuple[list[str], list[str], float | None]:
    """What to do for one world: (ids to start, ids to stop, when the people were first seen gone or None).

    humans: True a person is active, False none is, None the world's server did not answer (nothing changes: a restart is not a
    departure). An agent switched off in the admin screen is stopped at once; the rest are stopped once the people have been gone
    for GRACE. An agent that ended is not started again before COOLDOWN, so one that fails does not spend money in a loop.
    `free` (2026-10-22): the bots free of the time guard are a sim of bots alone: started whether or not a person is here, and not
    stopped when the last person goes (they end at their own limit, as ever)."""
    stop = sorted(i for i in running if i not in bots)
    if humans is None:
        return [], stop, gone_since
    free = set(free) & set(bots)

    def due(i: str) -> bool:
        return i not in running and now - last_start.get(i, -COOLDOWN) >= COOLDOWN
    if humans:
        return [i for i in bots if due(i)], stop, None
    gone = now if gone_since is None else gone_since
    if now - gone >= GRACE:
        stop = sorted(set(stop) | {i for i in running if i not in free})
    return [i for i in sorted(free) if due(i)], stop, gone


def _get(port: int, path: str):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=4) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def people_active(port: int, bots: list[str]) -> bool | None:
    """Is a person (a character that is not a bot) online in the world on this port? None if its server does not answer."""
    players = _get(port, "/players")
    if not isinstance(players, list):
        return None
    return any(p.get("online") for p in players if p.get("entity_id") not in bots)


def launch(world: str, eid: str, name: str, port: int, held: bool = False, peers: tuple[str, ...] | list[str] = (), free: bool = False) -> subprocess.Popen:
    folder = FOLDERS / eid
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["SMRITILILA_BOT_PEERS"] = ",".join(i for i in peers if i != eid)      # the other bots of the world: not persons to stay behind (the tool's time guard)
    if free:
        env["SMRITILILA_BOT_TIME_GUARD"] = "0"                                # a free bot: the guard is off for it
    try:                                                       # the user's own keys, as the servers read them (a started agent has no login of its own)
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            for i in range(winreg.QueryInfoKey(k)[1]):
                var, value, _ = winreg.EnumValue(k, i)             # (not `name`: that is the character's)
                if var.endswith("_API_KEY") and not env.get(var):
                    env[var] = value
    except (OSError, ImportError):
        pass
    return (hold_runner() if held else runner()).launch(world, eid, name, port, folder, env, MAX_ACTIONS, IDLE_MINUTES)


def stop_agent(eid: str, port: int, proc: subprocess.Popen | None) -> None:
    """Leave the character waiting and offline, through the tool, then end the agent's process tree."""
    folder = FOLDERS / eid
    if (folder / "player.json").exists():
        _run_hidden([sys.executable, str(Path(__file__).resolve().parent / "agents" / "player_tool.py"), str(folder), "stop", "--told"],
                       capture_output=True, timeout=60)
    if proc is not None and proc.poll() is None:
        if os.name == "nt":
            _run_hidden(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            proc.terminate()


def agents_still_running() -> set[str]:
    """Character ids whose agent is running now, as the runner finds them. A supervisor that is restarted does not know the agents the one
    before it started; this tells it, so it does not start a second one for a character already being played (an agent and its twin would
    both act for one character). Empty if the runner cannot tell. A held bot's agent is found by its pid file, whichever runner plays the rest."""
    return runner().running_ids(FOLDERS) | hold_runner().running_ids(FOLDERS)


def say(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="look once, say what it would do, and do nothing")
    once = ap.parse_args().once
    from smritilila.bots_registry import load_bots, load_free, load_hold
    from smritilila.worlds_registry import load_worlds_registry
    procs: dict[str, subprocess.Popen] = {}
    last_start: dict[str, float] = {}
    if not once:
        for eid in agents_still_running():                # played already, by an agent an earlier supervisor started: not started twice
            last_start[eid] = time.time()
            say(f"agent {eid[:8]} is already running: left to finish")
    gone: dict[str, float | None] = {}
    while True:
        try:
            bots = load_bots()
            held = load_hold()
            freed = load_free()
            worlds = {w.name: w for w in load_worlds_registry() if w.port}
            now = time.time()
            for name, w in worlds.items():
                ids = bots.get(name, [])
                mine = {k.split("|", 1)[1]: p for k, p in procs.items() if k.startswith(name + "|")}
                for i, p in list(mine.items()):
                    if p.poll() is not None:                       # it ended on its own: limit, idle timeout or a fault
                        say(f"{name}: agent {i[:8]} ended (exit {p.returncode})")
                        procs.pop(f"{name}|{i}", None)
                        mine.pop(i)
                if not ids and not mine:
                    continue
                humans = people_active(w.port, ids)
                start, stop, gone[name] = plan(ids, humans, set(mine), now, gone.get(name), last_start, freed.get(name, []))
                chars = (_get(w.port, "/world/exists") or {}).get("playable_entities", {})
                for i in stop:
                    say(f"{name}: stop {i[:8]}" + (" (dry run)" if once else ""))
                    if not once:
                        stop_agent(i, w.port, mine.get(i))
                        procs.pop(f"{name}|{i}", None)
                for i in start:
                    label = (chars.get(i) or [i])[0]
                    is_held = i in held.get(name, [])
                    is_free = i in freed.get(name, [])
                    say(f"{name}: start {label} ({i[:8]})" + (" held" if is_held else "") + (" free of the time guard" if is_free and not is_held else "") + (" (dry run)" if once else ""))
                    if not once:
                        last_start[i] = now
                        procs[f"{name}|{i}"] = launch(name, i, label, w.port, held=is_held, peers=ids, free=is_free)
        except Exception as exc:  # noqa: BLE001 -- a bad look must not end the watch
            say("look failed:", repr(exc)[:200])
        if once:
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
