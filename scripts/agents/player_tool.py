# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The agent player's tool (Marc and Anu, 2026-10-05).

An agent that played a character had to write its own watching and acting scripts, keep its own count and decide for
itself when to stop. On 2026-10-05 four players in a row stopped early on their own judgement (2 of 15, 13 of 15, 5 of 25,
6 of 25: "no round open", "a natural pause"), though the rule against it stood in the prompt: the rule was far from where
the decision was made, and the count was in no one's hands. Every one of them went on to finish when told, at the moment
of stopping, to act on something of their own.

So the rule is shown where the decision is, by the tool, after every command; the count is kept in a file; and `stop`
refuses until a stop condition holds. The tool speaks only to the character's own endpoints, and prints only what the
character's page shows: nothing from above is in its output.

    python player_tool.py FOLDER init --id ID --name NAME --port PORT [--max 25] [--idle 30] [--guard on|off] [--peers ID,ID]
    python player_tool.py FOLDER look                 the character's recap and what they remember
    python player_tool.py FOLDER watch                waits (up to 60 s) for: a round to act in, a telling, a question
    python player_tool.py FOLDER act ["text"]         posts the action (text, else FOLDER/next_action.txt)
    python player_tool.py FOLDER confirm yes|no       answers a jump-ahead question as the character would
    python player_tool.py FOLDER status               the count and the stop conditions
    python player_tool.py FOLDER stop [--told]        leaves the character waiting and offline -- only when a stop
                                                      condition holds (--told: the one who sent you said to stop)

FOLDER is the player's own folder; the tool keeps player.json, tellings.log and answers.log there.

THE TIME GUARD (2026-10-22, Marc: "build the time guard defaulted to never move ahead of the player but make it switchable"). A bot's character never moves ahead
of a person in time. A person is a player, other than the character itself and the bot's `--peers` (the other bots of the world, which the supervisor names in
SMRITILILA_BOT_PEERS), who is online, or whose page is closed but who has not logged off and is not waiting (their time is still their own, spec 37); where and when each stands is the page's own
list of who is here (GET /world/sync, with `own`: where and when, nothing of what anyone remembers), read by the tool, so no model is shown anyone's time. The rule is deterministic:
  - an action that joins an open round goes at once (the round is the person's moment and the character acts in it);
  - an action that would open a round alone is posted only while the character is strictly BEHIND every person online (catching up); level with or ahead of one,
    `watch` says HOLD instead of "it is yours to open", and `act` is refused: nothing is posted, nothing is counted, the next `watch` waits;
  - a jump-ahead question the character raised alone is answered NO, whatever was said, if it would end past a person's present; a question in a round is the
    round's (the people in it agree for themselves);
  - nobody online, or a world with no clock: nothing to be ahead of, nothing held.
The switch, most specific first: `init --guard off`; SMRITILILA_BOT_TIME_GUARD=0 (a whole supervisor, or a bot sim run by hand); and in the registry a bot marked
free (bots.json "free"), which the supervisor also starts with no person in the world: a sim of bots alone, as in the Crossing. Default: on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FAST = bool(os.environ.get("PLAYER_TOOL_FAST"))  # tests: no waiting
POLL = 0.01 if FAST else 2.0
QUIET = 0.4 if FAST else 60.0          # a watch that sees nothing for this long returns, saying so
PRESENCE_EVERY = 900.0                 # seconds between the presence posts `watch` makes, in the place of the page's heartbeat
RETRY_WAIT = 0 if FAST else 15         # the server may be restarting
RETRIES = 3
EXTRA_ACTIONS = 2                      # "up to 2 more only to close a moment that is truly mid-way"
GUARD_OFF = ("0", "off", "false", "no")


def guard_default() -> bool:
    """The time guard unless SMRITILILA_BOT_TIME_GUARD turns it off (0, off, false, no)."""
    return (os.environ.get("SMRITILILA_BOT_TIME_GUARD") or "on").strip().lower() not in GUARD_OFF


def must_hold(me: int | None, others: list, end: int | None = None) -> bool:
    """Is the character, at minute `me`, to be held back from acting alone among people at minutes `others`? True unless nothing is ahead to be run ahead of
    (no person, no clock) or the character is strictly behind every one; for a jump that would end at `end`, unless it ends no later than the nearest person's
    present. Pure: the whole rule."""
    present = [a for a in others if isinstance(a, int)]
    if me is None or not present:
        return False
    front = min(present)
    return end > front if end is not None else me >= front


def say(*parts: str) -> None:
    print(*parts, flush=True)


class Player:
    def __init__(self, folder: str | Path) -> None:
        self.dir = Path(folder)
        self.path = self.dir / "player.json"
        self.cfg = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self) -> None:
        """Written whole or not at all (2026-10-18): the hold agent and the playing agent read this file every second, and the supervisor's `stop` rewrites it, so
        a write in place could be read half done (CI found it: a JSONDecodeError killed the hold agent's loop)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        for attempt in range(8):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:                      # Windows: a reader has the file open at this instant
                if attempt == 7:
                    raise
                time.sleep(0.02)

    # -- the server ------------------------------------------------------------------------------------------------
    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 60.0) -> tuple[int, object]:
        url = f"http://127.0.0.1:{self.cfg['port']}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        last = None
        for attempt in range(RETRIES + 1):
            req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    raw = r.read().decode("utf-8")
                    self.cfg["errors"] = 0
                    return r.status, (json.loads(raw) if raw.strip() else {})
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")
                self.cfg["errors"] = self.cfg.get("errors", 0) + 1
                return e.code, text[:500]
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                last = e
                if attempt < RETRIES:
                    time.sleep(RETRY_WAIT)
        self.cfg["errors"] = self.cfg.get("errors", 0) + 1
        return 0, f"the server could not be reached ({type(last).__name__})"

    # -- the time guard ----------------------------------------------------------------------------------------------
    def peek(self, path: str) -> tuple[int, object]:
        """A read the guard makes: a failure of it is not one of the run's four errors."""
        errors = self.cfg.get("errors", 0)
        code, body = self.call("GET", path, timeout=30)
        self.cfg["errors"] = errors
        return code, body

    def people(self) -> tuple[int | None, list, str | None]:
        """(this character's minute, the minutes of the persons online, why it could not be read): from the page's own list of who is here."""
        code, rows = self.peek(f"/world/sync?as={self.cfg['id']}&own=1")   # `own`: also a person whose page is closed but whose time is still their own
        if code != 200 or not isinstance(rows, list):
            return None, [], f"the page's list of who is here could not be read ({code})"
        mine = next((r for r in rows if isinstance(r, dict) and r.get("entity_id") == self.cfg["id"]), None)
        peers = set(self.cfg.get("peers") or [])
        others = [r.get("at") for r in rows if isinstance(r, dict) and r.get("entity_id") != self.cfg["id"] and r.get("entity_id") not in peers]
        return (mine or {}).get("at"), others, None

    def alone_verdict(self) -> str | None:
        """Why the character may not act on their own now, or None if it may."""
        if not self.cfg.get("guard", True):
            return None
        me, others, err = self.people()
        if err:
            return f"{err}; nothing is posted on a guess. Run `watch` again."
        if must_hold(me, others):
            n = self.cfg["name"]
            return (f"{n} does not act on their own ahead of a person, and a person is level with {n} in time or behind. Nothing is yours to open. "
                    f"Do nothing: run `watch` again. When that person acts, a round opens and {n} joins it at once.")
        return None

    # -- the count and the stop conditions -------------------------------------------------------------------------
    def stop_reason(self) -> str | None:
        c = self.cfg
        if c.get("actions", 0) >= c["max"]:
            return f"{c['max']} actions taken"
        if c.get("errors", 0) >= 4:
            return "four errors in a row"
        if time.time() - c.get("last_activity", time.time()) >= c["idle"] * 60:
            return f"{c['idle']} minutes with no round activity"
        return None

    def footer(self) -> str:
        c = self.cfg
        why = self.stop_reason()
        head = f"[{c['name']}: {c.get('actions', 0)} of {c['max']} actions]"
        if why:
            return f"{head} STOP CONDITION: {why}. Run `stop`, then report."
        if c.get("guard", True):
            return (f"{head} NOT A STOP. A lull, a closed round, the other player away or out of the room, a natural pause: none of "
                    f"these ends the run. If a round is open, act at once. If `watch` says it is yours to open, decide something of {c['name']}'s own, "
                    f"write it, `act`, then `watch`. If it says HOLD, do nothing and `watch` again. Stop only at {c['max']} actions, {c['idle']} minutes "
                    f"of silence, four errors in a row, or when told to stop.")
        return (f"{head} NOT A STOP. A lull, a closed round, the other player away or out of the room, a natural pause: none of "
                f"these ends the run. If no round is open, YOUR action opens the next one: decide something of {c['name']}'s own, "
                f"write it, `act`, then `watch`. Stop only at {c['max']} actions, {c['idle']} minutes of silence, four errors in "
                f"a row, or when told to stop.")

    def log(self, name: str, text: str) -> None:
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(f"--- {time.strftime('%H:%M:%S')}\n{text}\n")

    def presence(self, online: bool) -> None:
        self.call("POST", f"/players/{self.cfg['id']}/presence", {"online": online}, timeout=30)
        self.cfg["last_presence"] = time.time()


def as_text(told: object) -> str:
    if isinstance(told, str):
        return told
    if isinstance(told, dict):
        return "\n\n".join(str(v) for v in told.values())
    if isinstance(told, list):
        return "\n\n".join(str(v) for v in told)
    return json.dumps(told, ensure_ascii=False)


def members_line(r: dict) -> str:
    ms = r.get("members") or []
    return ", ".join(f"{m.get('name', m.get('id'))} ({'has acted' if m.get('acted') else 'not yet'})" for m in ms) or "no one else"


# -- the commands ---------------------------------------------------------------------------------------------------
def archive_last_run(folder: Path) -> list[str]:
    """Put the last run's files into old/<time>/ and leave the folder clean for this one. A folder is reused (a character is played
    again and again), and 2026-10-06 the old done_action_NN.txt files made every `act` of the new run fail on a rename onto a file
    that was already there: the agent had to find that out and move them itself. Its own notes and scripts are left where they are."""
    mine = [f for f in folder.iterdir() if f.is_file() and (f.name.startswith("done_action_") or f.name in ("next_action.txt", "tellings.log", "answers.log"))]
    if not mine:
        return []
    dest = folder / "old" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for f in mine:
        f.replace(dest / f.name)
    return sorted(f.name for f in mine)


def cmd_init(p: Player, a: argparse.Namespace) -> int:
    p.dir.mkdir(parents=True, exist_ok=True)
    moved = archive_last_run(p.dir)
    peers = [x.strip() for x in (a.peers if a.peers is not None else os.environ.get("SMRITILILA_BOT_PEERS", "")).split(",") if x.strip()]
    guard = guard_default() if a.guard is None else a.guard == "on"
    p.cfg = {"id": a.id, "name": a.name, "port": a.port, "max": a.max, "idle": a.idle, "actions": 0, "errors": 0,
             "last_activity": time.time(), "last_presence": time.time(), "stopped": False, "guard": guard, "peers": peers}
    p.save()
    p.presence(True)
    p.save()
    if moved:
        say(f"({len(moved)} files of the last run were put away in old/.)")
    say(f"{a.name} is online. Run `look` to read what they remember, then `watch`." + ("" if guard else " (The time guard is off: this character may act ahead of a person.)"))
    say(p.footer())
    return 0


def cmd_look(p: Player, a: argparse.Namespace) -> int:
    code, recap = p.call("GET", f"/entities/{p.cfg['id']}/recap")
    code2, mem = p.call("GET", f"/entities/{p.cfg['id']}/anusmrti")
    if code != 200 or code2 != 200:
        say(f"ERROR reading the page: recap {code}, memories {code2}: {recap if code != 200 else mem}")
    else:
        say("WHERE THEY ARE:", json.dumps(recap, ensure_ascii=False))
        say("WHAT THEY REMEMBER (oldest first; weight, use):")
        for m in (mem.get("memories") if isinstance(mem, dict) else mem) or []:
            say(f"  [t{m.get('world_time')} {m.get('weight')}/{m.get('use')}] {str(m.get('experience', ''))[:500]}")
    p.save()
    say(p.footer())
    return 0


def cmd_watch(p: Player, a: argparse.Namespace) -> int:
    start = time.time()
    last_state = None
    p.cfg["joins"] = False
    while True:
        why = p.stop_reason()
        if why:
            say(f"STOP CONDITION: {why}.")
            break
        if time.time() - p.cfg.get("last_presence", 0) >= PRESENCE_EVERY:
            p.presence(True)
        code, r = p.call("GET", f"/rounds/{p.cfg['id']}", timeout=30)
        if code != 200 or not isinstance(r, dict):
            if p.cfg.get("errors", 0) >= 4:
                continue  # the next pass reports the stop condition
            time.sleep(POLL)
            continue
        state = (r.get("open"), r.get("acted"), r.get("running"), json.dumps(r.get("members"), sort_keys=True))
        if state != last_state:
            last_state = state
            p.cfg["last_activity"] = time.time()
        if r.get("confirm"):
            p.cfg["jump"] = {**(r["confirm"] if isinstance(r["confirm"], dict) else {}), "round": True}   # /rounds shows only a jump the round holds
            say("A QUESTION is waiting, as the character would answer it:", as_text(r["confirm"]))
            say("Answer it with `confirm yes` or `confirm no`. Never send a new action instead: it throws the telling away.")
            break
        if r.get("told"):
            text = as_text(r["told"])
            p.log("tellings.log", text)
            p.cfg["last_activity"] = time.time()
            say("TOLD (handed over once; kept in tellings.log):")
            say(text)
            say("Decide the next action now, from this and from your notes.")
            break
        if r.get("open") and not r.get("acted"):
            p.cfg["joins"] = True
            say(f"A ROUND IS OPEN and {p.cfg['name']} has not acted. In it: {members_line(r)}. Act at once.")
            break
        if time.time() - start >= QUIET:
            why = p.alone_verdict()
            if why:
                say("HOLD:", why)
            else:
                say("NOTHING OPENED for a minute. It is yours to open: act on something of the character's own.")
            break
        time.sleep(POLL)
    p.save()
    say(p.footer())
    return 0


def cmd_act(p: Player, a: argparse.Namespace) -> int:
    c = p.cfg
    taken, limit = c.get("actions", 0), c["max"]
    if taken >= limit and not (a.close and taken < limit + EXTRA_ACTIONS):
        say(f"REFUSED: {limit} actions are taken. That is a stop condition: run `stop`. (Up to {EXTRA_ACTIONS} more, with --close, "
            "only to close a moment that is truly mid-way.)")
        return 2
    nxt = p.dir / "next_action.txt"
    text = (a.text or (nxt.read_text(encoding="utf-8") if nxt.exists() else "")).strip()
    if not text:
        say("REFUSED: no action. Write it to next_action.txt in your folder, or pass it as text.")
        say(p.footer())
        return 2
    if c.get("guard", True):
        # an open round the last `watch` found (not read again here: GET /rounds hands a telling over once, and a second read would eat one) is a
        # person's moment, and the character acts in it
        why = None if c.get("joins") else p.alone_verdict()
        if why:
            p.save()
            say("HOLD:", why)
            say("NOT POSTED, and not counted: the action stays in next_action.txt.")
            say(p.footer())
            return 3
    code, ans = p.call("POST", "/pulses", {"entity_id": c["id"], "action_text": text}, timeout=300)
    if code != 200 or not isinstance(ans, dict):
        p.save()
        say(f"ERROR {code}: {ans}. The action was not counted. Errors in a row: {c.get('errors', 0)}.")
        say(p.footer())
        return 1
    c["actions"] = taken + 1
    c["joins"] = False
    c["last_activity"] = time.time()
    p.log("answers.log", f"ACTION: {text}\nANSWER: {json.dumps(ans, ensure_ascii=False)}")
    if nxt.exists():
        nxt.rename(p.dir / f"done_action_{c['actions']:02d}.txt")
    say(f"POSTED action {c['actions']}: {text}")
    own = (ans.get("narrations") or {}).get(c["id"])
    if ans.get("confirm"):
        p.cfg["jump"] = {**(ans["confirm"] if isinstance(ans["confirm"], dict) else {}), "round": isinstance(ans.get("round"), dict)}
        say("A QUESTION is waiting, as the character would answer it:", as_text(ans["confirm"]))
        say("Answer it with `confirm yes` or `confirm no`.")
    elif own:
        say("YOU WERE LAST TO ACT: the round ran, and this is the telling (the round is done):")
        say(own)
        p.log("tellings.log", own)
    elif isinstance(ans.get("round"), dict):
        say(f"The round is open, waiting for the others. In it: {members_line(ans['round'])}. Run `watch`.")
    else:
        say("The answer carried no telling and no open round. Run `watch`.")
    p.save()
    say(p.footer())
    return 0


def jump_verdict(p: Player) -> str | None:
    """Why a YES to the jump the character is asked about is answered NO instead, or None. A question in a round is the round's: the people in it agree
    for themselves. One the character raised alone is answered NO when it would end past a person's present, or when where it ends cannot be told."""
    jump = p.cfg.get("jump") or {}
    if not p.cfg.get("guard", True) or jump.get("round"):
        return None
    me, others, err = p.people()
    if err:
        return f"{err}"
    end = jump.get("to")
    if end is None and me is not None and isinstance(jump.get("minutes"), int):
        end = me + jump["minutes"]
    if not any(isinstance(x, int) for x in others) or me is None:
        return None
    if end is None or must_hold(me, others, end):
        return f"this jump would take {p.cfg['name']} past a person's present, and {p.cfg['name']} does not run ahead of a person"
    return None


def cmd_confirm(p: Player, a: argparse.Namespace) -> int:
    if a.answer == "yes":
        why = jump_verdict(p)
        if why:
            a.answer = "no"
            say(f"GUARD: {why}. The answer is NO.")
    p.cfg.pop("jump", None)
    code, ans = p.call("POST", "/pulses/confirm", {"entity_id": p.cfg["id"], "proceed": a.answer == "yes"}, timeout=300)
    if code != 200:
        say(f"ERROR {code}: {ans}")
    else:
        own = (ans.get("narrations") or {}).get(p.cfg["id"]) if isinstance(ans, dict) else None
        say("ANSWERED:", a.answer)
        if own:
            p.log("tellings.log", own)
            say("TOLD:", own)
        p.cfg["last_activity"] = time.time()
    p.save()
    say(p.footer())
    return 0 if code == 200 else 1


def cmd_status(p: Player, a: argparse.Namespace) -> int:
    say(json.dumps({k: p.cfg.get(k) for k in ("name", "actions", "max", "errors", "idle", "stopped")}, ensure_ascii=False))
    say(p.footer())
    return 0


def cmd_stop(p: Player, a: argparse.Namespace) -> int:
    why = "told to stop" if a.told else p.stop_reason()
    if not why:
        say(f"REFUSED: no stop condition holds. {p.cfg.get('actions', 0)} of {p.cfg['max']} actions; a lull is not one. "
            "Keep playing. (If the one who sent you said to stop, run `stop --told`.)")
        say(p.footer())
        return 2
    p.cfg["stopped"] = True                                 # said first (2026-10-18): an agent that beats presence sees it and does not post `online` after this stop's `offline`
    p.save()
    code1, _ = p.call("POST", f"/players/{p.cfg['id']}/autonomy", {"allow": True, "wait": True}, timeout=30)
    code2, _ = p.call("POST", f"/players/{p.cfg['id']}/presence", {"online": False}, timeout=30)
    say(f"{p.cfg['name']} is left waiting and offline (autonomy {code1}, presence {code2}). Stopped because: {why}. "
        f"Actions taken: {p.cfg.get('actions', 0)}. Now report: each action verbatim with what they were told (tellings.log has "
        "every telling); anything odd; why you stopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # the Windows console has no "ū"
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--id", required=True)
    i.add_argument("--name", required=True)
    i.add_argument("--port", type=int, required=True)
    i.add_argument("--max", type=int, default=25)
    i.add_argument("--idle", type=int, default=30)
    i.add_argument("--guard", choices=["on", "off"], default=None, help="the time guard (default: on, unless SMRITILILA_BOT_TIME_GUARD turns it off)")
    i.add_argument("--peers", default=None, help="ids of the other bots of this world, which are not persons to stay behind (default: SMRITILILA_BOT_PEERS)")
    sub.add_parser("look")
    sub.add_parser("watch")
    act = sub.add_parser("act")
    act.add_argument("text", nargs="?", default="")
    act.add_argument("--close", action="store_true")
    cf = sub.add_parser("confirm")
    cf.add_argument("answer", choices=["yes", "no"])
    sub.add_parser("status")
    st = sub.add_parser("stop")
    st.add_argument("--told", action="store_true")
    a = ap.parse_args(argv)
    p = Player(a.folder)
    if a.cmd != "init" and not p.cfg:
        say(f"No player.json in {a.folder}: run `init` first.")
        return 2
    return {"init": cmd_init, "look": cmd_look, "watch": cmd_watch, "act": cmd_act, "confirm": cmd_confirm,
            "status": cmd_status, "stop": cmd_stop}[a.cmd](p, a)


if __name__ == "__main__":
    raise SystemExit(main())
