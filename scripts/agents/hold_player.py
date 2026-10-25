# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""A held bot: its character kept present, and never acted for (Marc and Anu, 2026-10-17).

    python hold_player.py FOLDER --id ID --name NAME --port PORT [--beat 240]

The bot supervisor (scripts/bots_supervisor.py) starts this instead of an agent that plays, for a bot the registry marks held. It puts the character
online through the tool (player_tool.py init), sends the presence beat every few minutes so the server keeps counting them online (it stops counting
thirty minutes after the last), and posts no action, ever: no model is asked, nothing costs. The character shows in everyone's players-online line
(light violet when not level with them) while a person is active in the world. The supervisor ends it as it ends any agent: the tool's `stop --told`
leaves the character waiting and offline and marks the run stopped, which this notices; the process is then ended anyway.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import player_tool  # noqa: E402

BEAT = 240.0   # seconds between presence beats (the page sends one every five minutes; the server counts thirty)


def hold(folder: Path | str, eid: str, name: str, port: int, beat: float = BEAT, tick: float = 1.0, sleep=time.sleep) -> str:
    """Hold the character until the run is marked stopped; why it ended. `tick` is how often it looks at whether it was stopped."""
    folder = Path(folder)
    player_tool.main([str(folder), "init", "--id", eid, "--name", name, "--port", str(port), "--max", "1", "--idle", "100000"])
    waited = 0.0
    while True:
        try:
            me = player_tool.Player(folder)
        except (ValueError, OSError):                      # the file is being replaced, or is for the moment unreadable: look again next tick
            sleep(tick)
            waited += tick
            continue
        if me.cfg.get("stopped"):
            return "stopped"
        if waited >= beat:
            me.presence(True)                                  # it only reads the file: a save here from a stale copy could erase the `stopped` the supervisor just wrote
            waited = 0.0
            try:
                stopped_now = player_tool.Player(folder).cfg.get("stopped")
            except (ValueError, OSError):
                stopped_now = False
            if stopped_now:                                    # a stop landed while the beat was in the air: the beat may have arrived after its `offline`, so say offline again
                me.presence(False)
                return "stopped"
        sleep(tick)
        waited += tick


def main(argv: list[str] | None = None) -> int:
    if sys.stdout is None:                                         # started without a console
        sys.stdout = open(os.devnull, "w")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--beat", type=float, default=BEAT)
    a = ap.parse_args(argv)
    folder = Path(a.folder)
    folder.mkdir(parents=True, exist_ok=True)
    pidfile = folder / "agent.pid"
    pidfile.write_text(str(os.getpid()), encoding="utf-8")
    try:
        reason = hold(folder, a.id, a.name, a.port, a.beat)
    finally:
        pidfile.unlink(missing_ok=True)
    print(time.strftime("%H:%M:%S"), f"{a.name} held until: {reason}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
