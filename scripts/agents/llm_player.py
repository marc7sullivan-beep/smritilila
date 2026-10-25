# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""A bot's agent that plays one character through the engine's own LLM client (Marc and Anu, 2026-10-17).

    python llm_player.py FOLDER --id ID --name NAME --port PORT --world "World name" [--max 25] [--idle 30]

The server is spoken to only through scripts/agents/player_tool.py, which keeps the count, the stop conditions and the character's presence; this file
only decides. Each turn it waits for the tool's `watch`, reads the character's own page with `look`, asks a model what the character does next, and posts
it with `act`; a question about jumping ahead in time is answered YES or NO. The model is whatever the engine is configured with for its one-off calls
(`best_available_client`: the first configured provider, in its own order of preference, see llm/providers.py); SMRITILILA_BOT_MODEL names another model id. It makes
LLM calls, so it costs. Whenever it ends, for any reason, the character is left waiting and offline (the tool's `stop`).

What the model is shown is the character's own page and the last few tellings: nothing from above, as the agent-player prompt has always held.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import player_tool  # noqa: E402

OTHERS = "A person may be here, and others too; you act together in rounds."
NO_NOTE = "(No note was written for this world. You know only what your character's page shows you.)"
KEEP = 4               # the last tellings and actions the model is reminded of
MAX_REPLY = 600        # characters of one action
TRIES = 3              # asks of the model before the character is left waiting
_FOOTER = re.compile(r"^\[.+: \d+ of \d+ actions\]")


class ModelUnavailable(Exception):
    pass


def call(folder: Path | str, *args: str) -> str:
    """One command of the tool, and what it printed."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            player_tool.main([str(folder), *args])
    except SystemExit as e:                                        # a refused command line must not end the run with the character online
        buf.write(f"ERROR: the tool refused its arguments ({e.code})\n")
    return buf.getvalue()


def clean(out: str) -> str:
    """The tool's output without its closing line (the rule for an agent that reads it; the model is shown the page, not the rule)."""
    lines = out.strip().splitlines()
    if lines and _FOOTER.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def note_for(world: str) -> str:
    """The world's note (scripts/agents/worlds/<name>.md, the file named in the world's name), else the line that says there is none."""
    slug = world.lower()
    for f in sorted((HERE / "worlds").glob("*.md")):
        if f.stem in slug:
            return f.read_text(encoding="utf-8").strip()
    return NO_NOTE


def system_prompt(name: str, world: str) -> str:
    text = (HERE / "core.md").read_text(encoding="utf-8")
    for k, v in {"NAME": name, "OTHERS_LINE": OTHERS, "WORLD_NOTE": note_for(world)}.items():
        text = text.replace("{" + k + "}", v)
    return text.strip()


def action_from(reply: str) -> str:
    """The reply as one action: one run of text, not wrapped in quotes, no longer than a player would type."""
    text = " ".join(reply.split()).strip()
    if len(text) > 1 and text[0] in "\"“" and text[-1] in "\"”":
        text = text[1:-1].strip()
    return text[:MAX_REPLY]


def ask(client, system: str, user: str, max_tokens: int = 400) -> str:
    last: object = "an empty reply"
    for attempt in range(TRIES):
        try:
            reply = client.complete(system, user, max_tokens=max_tokens)
            if reply and reply.strip():
                return reply
        except Exception as e:  # noqa: BLE001 -- a model that is down is a reason to leave the character waiting, not a crash
            last = repr(e)[:200]
        time.sleep(0 if player_tool.FAST else 5 * (attempt + 1))
    raise ModelUnavailable(str(last))


def compose(name: str, look: str, recent: list[str], now: str) -> str:
    parts = [look]
    if recent:
        parts.append("JUST NOW (oldest first):\n" + "\n\n".join(recent[-2 * KEEP:]))
    parts += [now, f"What does {name} do next?"]
    return "\n\n".join(p for p in parts if p)


def play(folder: Path | str, eid: str, name: str, port: int, world: str, max_actions: int, idle: int, client, run=call) -> str:
    """Play until a stop condition holds or the character is stopped from outside; the reason it ended. `run` is the tool (a test passes a fake)."""
    folder = Path(folder)
    system = system_prompt(name, world)
    run(folder, "init", "--id", eid, "--name", name, "--port", str(port), "--max", str(max_actions), "--idle", str(idle))
    recent: list[str] = []
    try:
        while True:
            try:
                me = player_tool.Player(folder)
            except (ValueError, OSError):                  # the file is being replaced: look again
                time.sleep(0.05)
                continue
            if me.cfg.get("stopped"):
                return "stopped from outside"
            why = me.stop_reason()
            if why:
                run(folder, "stop")
                return why
            now = clean(run(folder, "watch"))
            if now.startswith("STOP CONDITION"):
                continue                                           # the top of the loop sees the same condition and stops
            if now.startswith("HOLD"):
                continue                                           # the time guard: a person is not ahead of the character, so nothing is theirs to open; no model is asked
            look = clean(run(folder, "look"))
            if "A QUESTION is waiting" in now:
                answer = ask(client, system, compose(name, look, recent, now) + "\nAnswer YES or NO.", max_tokens=10)
                told = clean(run(folder, "confirm", "yes" if answer.strip().lower().startswith("y") else "no"))
                recent.append(told[:1500])
                continue
            user = compose(name, look, recent, now)
            if now.startswith("TOLD"):
                recent.append(now[:1500])                          # kept for the turns after this one; this turn shows it as `now`
            text = action_from(ask(client, system, user))
            (folder / "next_action.txt").write_text(text, encoding="utf-8")   # the tool reads it, so no action is ever read as an option
            res = clean(run(folder, "act"))
            if res.startswith("HOLD"):
                continue                                           # not posted, not counted (the tool kept the action file): nothing was done, so nothing is remembered as done
            recent.append(f"{name} did: {text}")
            if "YOU WERE LAST TO ACT" in res:
                recent.append(res[:1500])
            if "A QUESTION is waiting" in res:                     # a jump the action itself raised: answered now, since a new action in its place would count as no
                answer = ask(client, system, compose(name, look, recent, res) + "\nAnswer YES or NO.", max_tokens=10)
                told = clean(run(folder, "confirm", "yes" if answer.strip().lower().startswith("y") else "no"))
                recent.append(told[:1500])
    except ModelUnavailable as e:
        run(folder, "stop", "--told")
        return f"the model could not be reached ({e})"
    except BaseException:
        run(folder, "stop", "--told")
        raise


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
    ap.add_argument("--world", default="")
    ap.add_argument("--max", type=int, default=25)
    ap.add_argument("--idle", type=int, default=30)
    a = ap.parse_args(argv)
    from smritilila.llm import providers
    from smritilila.llm.client import best_available_client
    try:
        client = best_available_client(os.environ.get("SMRITILILA_BOT_MODEL") or None)
    except KeyError:
        print(providers.nothing_configured())
        return 2
    folder = Path(a.folder)
    folder.mkdir(parents=True, exist_ok=True)
    pidfile = folder / "agent.pid"
    pidfile.write_text(str(os.getpid()), encoding="utf-8")
    try:
        reason = play(folder, a.id, a.name, a.port, a.world, a.max, a.idle, client)
    finally:
        pidfile.unlink(missing_ok=True)
    print(time.strftime("%H:%M:%S"), f"{a.name} stopped: {reason}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
