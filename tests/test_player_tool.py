# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""scripts/agents/player_tool.py (2026-10-05): the agent player's tool, run for real against a stub server that records what
it is sent. Four players in a row stopped early on their own judgement; the tool keeps the count, shows the rule after every
command and refuses to stop until a stop condition holds."""
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "scripts" / "agents" / "player_tool.py"
ID = "teo_test_id"


class _Stub:
    """The character's own endpoints, as the server answers them; every request is recorded."""

    def __init__(self):
        self.round = {"members": [{"id": ID, "name": "Teo", "acted": False}, {"id": "nara", "name": "Nara", "acted": False}],
                      "open": False, "acted": False, "told": None, "confirm": None, "running": False}
        self.pulse_answer = {"confirm": None, "narrations": {}, "round": {"members": [{"id": ID, "name": "Teo", "acted": True},
                                                                                        {"id": "nara", "name": "Nara", "acted": False}]}}
        self.pulse_status = 200
        self.sync: object = [{"entity_id": ID, "label": "Teo", "at": 100}]   # who is here and when (the time guard's only source); nobody else by default
        self.sync_status = 200
        self.requests: list[tuple[str, str, dict | None]] = []

    def serve(self):
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                raw = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                stub.requests.append(("GET", self.path, None))
                if self.path.startswith("/rounds/"):
                    snap = dict(stub.round)
                    stub.round["told"] = None  # handed over once
                    return self._send(200, snap)
                if self.path.startswith("/world/sync"):
                    return self._send(stub.sync_status, stub.sync if stub.sync_status == 200 else {"detail": "boom"})
                if self.path.endswith("/recap"):
                    return self._send(200, {"location_id": "the_inn", "last_narration": "I sat by the fire."})
                if self.path.endswith("/anusmrti"):
                    return self._send(200, {"memories": [{"world_time": 5, "weight": "key", "use": "key", "experience": "I was pulled from the river by Nara."}]})
                self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                stub.requests.append(("POST", self.path, body))
                if self.path == "/pulses":
                    return self._send(stub.pulse_status, stub.pulse_answer if stub.pulse_status == 200 else {"detail": "boom"})
                self._send(200, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def posted(self, path_end):
        return [b for m, p, b in self.requests if m == "POST" and p.endswith(path_end)]


@pytest.fixture()
def tool(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PLAYER_TOOL_FAST", "1")
    spec = importlib.util.spec_from_file_location("player_tool_under_test", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stub = _Stub().serve()
    folder = tmp_path / "teo"

    def run(*args, capsys=None):
        return mod.main([str(folder), *args])

    run.stub, run.folder, run.mod = stub, folder, mod
    run("init", "--id", ID, "--name", "Teo", "--port", str(stub.port), "--max", "3", "--idle", "30")
    run.init_out = capsys.readouterr().out
    yield run
    stub.server.shutdown()


def _out(capsys):
    return capsys.readouterr().out


def test_init_puts_the_last_runs_files_away_so_the_new_run_can_act(tool, capsys):
    """2026-10-06: a reused folder still held done_action_NN.txt, and every `act` failed on the rename onto it."""
    f = tool.folder
    for name in ("done_action_01.txt", "done_action_02.txt", "tellings.log", "answers.log", "next_action.txt"):
        (f / name).write_text("the last run", encoding="utf-8")
    (f / "notes.md").write_text("my own notes", encoding="utf-8")
    tool("init", "--id", ID, "--name", "Teo", "--port", str(tool.stub.port), "--max", "3", "--idle", "30")
    assert "put away in old/" in capsys.readouterr().out
    assert sorted(x.name for x in f.iterdir() if x.is_file()) == ["notes.md", "player.json"]                  # clean, and the agent's own notes untouched
    kept = [x.name for x in (next((f / "old").iterdir())).iterdir()]
    assert sorted(kept) == ["answers.log", "done_action_01.txt", "done_action_02.txt", "next_action.txt", "tellings.log"]   # nothing lost
    (f / "next_action.txt").write_text("I wait by the post.", encoding="utf-8")
    assert tool("act") == 0 and (f / "done_action_01.txt").read_text(encoding="utf-8") == "I wait by the post."     # the first act of the new run works


def test_a_fresh_folder_is_not_given_an_old_directory(tool):
    assert not (tool.folder / "old").exists()


def test_init_puts_the_character_online_and_says_the_rule(tool):
    out = tool.init_out
    assert tool.stub.posted("/presence") == [{"online": True}]
    cfg = json.loads((tool.folder / "player.json").read_text(encoding="utf-8"))
    assert cfg["max"] == 3 and cfg["actions"] == 0 and cfg["id"] == ID
    assert "NOT A STOP" in out and "0 of 3 actions" in out


def test_look_reads_only_the_characters_own_page(tool, capsys):
    capsys.readouterr()
    assert tool("look") == 0
    out = _out(capsys)
    assert "I sat by the fire." in out and "pulled from the river by Nara" in out and "[t5 key/key]" in out
    assert {p for m, p, b in tool.stub.requests if m == "GET"} >= {f"/entities/{ID}/recap", f"/entities/{ID}/anusmrti"}
    assert not any(p.startswith("/players") for m, p, b in tool.stub.requests if m == "GET")  # nothing about anyone else


def test_watch_returns_the_moment_a_round_opens_for_the_character(tool, capsys):
    capsys.readouterr()
    tool.stub.round["open"] = True
    assert tool("watch") == 0
    out = _out(capsys)
    assert "A ROUND IS OPEN" in out and "Nara (not yet)" in out and "NOT A STOP" in out


def test_a_telling_is_printed_in_full_kept_and_handed_over_once(tool, capsys):
    capsys.readouterr()
    tool.stub.round["told"] = "Nara asks Kester where the weir stood. \"Vadhū\" he says."
    tool("watch")
    out = _out(capsys)
    assert "TOLD" in out and "Nara asks Kester where the weir stood." in out and "Vadhū" in out
    assert "Nara asks Kester" in (tool.folder / "tellings.log").read_text(encoding="utf-8")
    tool("watch")   # nothing now: the telling was handed over once
    assert "NOTHING OPENED" in _out(capsys)


def test_a_lull_is_not_a_stop_and_the_watch_says_whose_move_it_is(tool, capsys):
    capsys.readouterr()
    assert tool("watch") == 0
    out = _out(capsys)
    assert "NOTHING OPENED" in out and "It is yours to open" in out and "NOT A STOP" in out and "STOP CONDITION" not in out


def test_a_question_is_surfaced_and_answered_not_replaced_by_an_action(tool, capsys):
    capsys.readouterr()
    tool.stub.round["confirm"] = "You are about to jump ahead 15 min. Proceed?"
    tool("watch")
    assert "A QUESTION" in _out(capsys)
    assert tool("confirm", "yes") == 0
    assert tool.stub.posted("/pulses/confirm") == [{"entity_id": ID, "proceed": True}]
    assert json.loads((tool.folder / "player.json").read_text(encoding="utf-8"))["actions"] == 0  # an answer is not an action
    assert tool("confirm", "no") == 0
    assert tool.stub.posted("/pulses/confirm")[-1] == {"entity_id": ID, "proceed": False}       # and "no" is a no


def test_act_counts_posts_and_keeps_the_action_file(tool, capsys):
    capsys.readouterr()
    (tool.folder / "next_action.txt").write_text("I ask Ilo how long before the water drops.", encoding="utf-8")
    assert tool("act") == 0
    out = _out(capsys)
    assert tool.stub.posted("/pulses") == [{"entity_id": ID, "action_text": "I ask Ilo how long before the water drops."}]
    assert "POSTED action 1" in out and "The round is open, waiting for the others" in out and "1 of 3 actions" in out
    assert (tool.folder / "done_action_01.txt").exists() and not (tool.folder / "next_action.txt").exists()
    assert json.loads((tool.folder / "player.json").read_text(encoding="utf-8"))["actions"] == 1


def test_the_last_to_act_gets_the_telling_inline(tool, capsys):
    capsys.readouterr()
    tool.stub.pulse_answer = {"confirm": None, "round": None, "narrations": {ID: "The round ran. Ilo says tomorrow noon."}}
    tool("act", "I ask Ilo about the water.")
    out = _out(capsys)
    assert "YOU WERE LAST TO ACT" in out and "Ilo says tomorrow noon." in out
    assert "Ilo says tomorrow noon." in (tool.folder / "tellings.log").read_text(encoding="utf-8")


def test_an_empty_action_and_a_failed_post_are_not_counted(tool, capsys):
    capsys.readouterr()
    assert tool("act", "") == 2 and "REFUSED: no action" in _out(capsys)
    tool.stub.pulse_status = 500
    assert tool("act", "I wait.") == 1
    out = _out(capsys)
    assert "ERROR 500" in out and "not counted" in out
    assert json.loads((tool.folder / "player.json").read_text(encoding="utf-8"))["actions"] == 0


def test_stop_is_refused_while_no_stop_condition_holds(tool, capsys):
    capsys.readouterr()
    tool("act", "one")
    n_before = len(tool.stub.requests)
    assert tool("stop") == 2
    out = _out(capsys)
    assert "REFUSED: no stop condition holds" in out and "1 of 3 actions" in out and "a lull is not one" in out
    assert not tool.stub.posted("/autonomy")                                    # the character is left as they were
    assert [r for r in tool.stub.requests[n_before:] if r[0] == "POST"] == []


def test_at_the_limit_stop_is_allowed_and_leaves_the_character_waiting_and_offline(tool, capsys):
    for i in range(3):
        tool("act", f"action {i}")
    capsys.readouterr()
    assert tool("act", "a fourth") == 2 and "REFUSED" in _out(capsys) and len(tool.stub.posted("/pulses")) == 3
    tool("watch")
    assert "STOP CONDITION: 3 actions taken" in _out(capsys)
    assert tool("stop") == 0
    assert "left waiting and offline" in _out(capsys)
    assert tool.stub.posted("/autonomy") == [{"allow": True, "wait": True}]
    assert tool.stub.posted("/presence")[-1] == {"online": False}               # waiting, then offline: in that order
    order = [p for m, p, b in tool.stub.requests if m == "POST" and (p.endswith("/autonomy") or p.endswith("/presence"))]
    assert order[-2].endswith("/autonomy") and order[-1].endswith("/presence")


def test_two_more_actions_may_close_a_moment_that_is_truly_mid_way(tool, capsys):
    for i in range(3):
        tool("act", f"action {i}")
    assert tool("act", "closing one", "--close") == 0 and tool("act", "closing two", "--close") == 0
    assert tool("act", "closing three", "--close") == 2
    assert len(tool.stub.posted("/pulses")) == 5


def test_being_told_to_stop_is_a_stop_condition(tool, capsys):
    capsys.readouterr()
    assert tool("stop", "--told") == 0
    assert "Stopped because: told to stop" in _out(capsys)


def test_thirty_minutes_without_round_activity_is_a_stop_condition(tool, capsys):
    cfgp = tool.folder / "player.json"
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    cfg["last_activity"] = time.time() - 31 * 60
    cfgp.write_text(json.dumps(cfg), encoding="utf-8")
    capsys.readouterr()
    tool("watch")
    assert "STOP CONDITION: 30 minutes with no round activity" in _out(capsys)
    assert tool("stop") == 0


def test_four_errors_in_a_row_are_a_stop_condition(tool, capsys, monkeypatch):
    tool.stub.pulse_status = 500
    for _ in range(4):
        tool("act", "x")
    capsys.readouterr()
    assert tool("stop") == 0 and "four errors in a row" in _out(capsys)


def test_a_folder_without_init_is_told_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLAYER_TOOL_FAST", "1")
    spec = importlib.util.spec_from_file_location("player_tool_noinit", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([str(tmp_path / "nobody"), "watch"]) == 2
    assert "run `init` first" in capsys.readouterr().out


# -- the time guard (2026-10-22, Marc: "build the time guard defaulted to never move ahead of the player but make it switchable") -----------------------

def _here(tool, me, *others):
    """Who the page lists as here, and when: this character at minute `me`, then (id, minute) of each other."""
    tool.stub.sync = [{"entity_id": ID, "label": "Teo", "at": me}] + [{"entity_id": i, "label": i, "at": at} for i, at in others]


def test_the_whole_rule_is_a_pure_function(tool):
    hold = tool.mod.must_hold
    assert hold(100, [100]) is True and hold(100, [90]) is True            # level with a person, or ahead of one: held
    assert hold(100, [101]) is False and hold(100, [500, 101]) is False    # strictly behind every person: may act
    assert hold(100, [500, 100]) is True                                   # behind one, level with another: held
    assert hold(100, []) is False and hold(None, [200]) is False           # nobody, or no clock: nothing to be ahead of
    assert hold(100, [None, 90]) is True and hold(100, [None]) is False   # a person with no clock is not a time
    assert hold(100, [500], end=500) is False and hold(100, [500], end=501) is True   # a jump may end AT a person's present, never past it
    assert hold(100, [500, 300], end=301) is True                          # the nearest person's present is the limit


def test_a_person_level_with_the_character_holds_it_where_it_would_have_been_told_to_open(tool, capsys):
    capsys.readouterr()
    _here(tool, 100, ("jakal", 100))
    assert tool("watch") == 0
    out = _out(capsys)
    assert "HOLD:" in out and "does not act on their own ahead of a person" in out and "NOTHING OPENED" not in out and "It is yours to open" not in out
    assert "If it says HOLD, do nothing and `watch` again" in out and "STOP CONDITION" not in out
    assert ("GET", "/world/sync?as=" + ID + "&own=1", None) in tool.stub.requests


def test_a_person_behind_or_ahead_the_watch_says_so_and_only_a_person_ahead_frees_the_character(tool, capsys):
    capsys.readouterr()
    _here(tool, 100, ("jakal", 80))
    tool("watch")
    assert "HOLD:" in _out(capsys)                                           # a person behind: the character is already ahead
    _here(tool, 100, ("jakal", 140))
    tool("watch")
    out = _out(capsys)
    assert "NOTHING OPENED" in out and "It is yours to open" in out and "HOLD" not in out.split("NOT A STOP")[0]   # a person ahead: catching up is allowed
    _here(tool, 100, ("jakal", 140), ("thran", 100))
    tool("watch")
    assert "HOLD:" in _out(capsys)                                           # strictly behind EVERY person: one level is enough to hold


def test_the_other_bots_of_the_world_are_not_persons_and_nobody_online_holds_nothing(tool, capsys, tmp_path):
    capsys.readouterr()
    tool("init", "--id", ID, "--name", "Teo", "--port", str(tool.stub.port), "--peers", "nara,ilo")
    _here(tool, 100, ("nara", 100), ("ilo", 50))
    capsys.readouterr()
    tool("watch")
    assert "NOTHING OPENED" in _out(capsys)                                  # two bots at his time and before it: no person to stay behind
    _here(tool, 100)
    tool("watch")
    assert "NOTHING OPENED" in _out(capsys)                                  # nobody else here
    _here(tool, None, ("jakal", None))
    tool("watch")
    assert "NOTHING OPENED" in _out(capsys)                                  # a world with no clock: no time to run ahead in


def test_a_failure_to_read_who_is_here_holds_rather_than_guesses_and_is_not_one_of_the_runs_errors(tool, capsys):
    capsys.readouterr()
    tool.stub.sync_status = 500
    for _ in range(5):
        tool("watch")
    out = _out(capsys)
    assert out.count("HOLD:") == 5 and "could not be read" in out and "STOP CONDITION" not in out
    assert json.loads((tool.folder / "player.json").read_text(encoding="utf-8"))["errors"] == 0   # a guard read is not one of the four errors in a row


def test_an_action_that_would_open_a_round_alone_is_refused_when_held_and_costs_nothing(tool, capsys):
    capsys.readouterr()
    _here(tool, 100, ("jakal", 100))
    (tool.folder / "next_action.txt").write_text("I walk on to the ford.", encoding="utf-8")
    assert tool("act") == 3
    out = _out(capsys)
    assert "HOLD:" in out and "NOT POSTED, and not counted" in out
    assert tool.stub.posted("/pulses") == [] and (tool.folder / "next_action.txt").exists() and not (tool.folder / "done_action_01.txt").exists()
    assert json.loads((tool.folder / "player.json").read_text(encoding="utf-8"))["actions"] == 0
    _here(tool, 100, ("jakal", 400))                                            # the person is ahead now: catching up is allowed
    assert tool("act") == 0 and tool.stub.posted("/pulses") == [{"entity_id": ID, "action_text": "I walk on to the ford."}]


def test_an_action_that_joins_a_round_a_person_opened_goes_at_once_even_when_level(tool, capsys):
    capsys.readouterr()
    _here(tool, 100, ("jakal", 100))
    tool.stub.round["open"] = True
    tool("watch")
    assert "A ROUND IS OPEN" in _out(capsys)
    (tool.folder / "next_action.txt").write_text("I take the other oar.", encoding="utf-8")
    assert tool("act") == 0 and tool.stub.posted("/pulses") == [{"entity_id": ID, "action_text": "I take the other oar."}]   # the round is the person's moment
    capsys.readouterr()
    (tool.folder / "next_action.txt").write_text("And then I go alone.", encoding="utf-8")
    assert tool("act") == 3                                                      # the next action opens nothing: it is held again, the round is spent


def test_a_jump_the_character_raised_alone_is_answered_no_when_it_would_end_past_a_person(tool, capsys):
    capsys.readouterr()
    tool.stub.pulse_answer = {"confirm": {"kind": "long", "prompt": "You are about to jump ahead 3 h (to day 28). Proceed?", "minutes": 180, "to": 280},
                              "narrations": {}, "round": None}
    _here(tool, 100, ("jakal", 250))                                             # the character is behind him, so it may act; the jump ends at 280
    (tool.folder / "next_action.txt").write_text("I sleep till the water drops.", encoding="utf-8")
    assert tool("act") == 0
    assert "A QUESTION is waiting" in _out(capsys)
    assert tool("confirm", "yes") == 0
    out = _out(capsys)
    assert "GUARD:" in out and "The answer is NO" in out
    assert tool.stub.posted("/pulses/confirm") == [{"entity_id": ID, "proceed": False}]      # yes asked, no sent


def test_a_jump_that_ends_before_a_persons_present_is_let_through_and_so_is_one_with_nobody_here(tool, capsys):
    capsys.readouterr()
    ask = {"confirm": {"kind": "long", "prompt": "You are about to jump ahead 2 h. Proceed?", "minutes": 120, "to": 220}, "narrations": {}, "round": None}
    tool.stub.pulse_answer = ask
    _here(tool, 100, ("jakal", 250))
    (tool.folder / "next_action.txt").write_text("I rest.", encoding="utf-8")
    tool("act")
    tool("confirm", "yes")
    assert tool.stub.posted("/pulses/confirm")[-1] == {"entity_id": ID, "proceed": True}      # 220 is before 250: catching up
    _here(tool, 100)
    (tool.folder / "next_action.txt").write_text("I rest again.", encoding="utf-8")
    tool("act")
    tool("confirm", "yes")
    assert tool.stub.posted("/pulses/confirm")[-1] == {"entity_id": ID, "proceed": True}      # nobody here: nothing to pass


def test_a_jump_whose_end_cannot_be_told_is_answered_no_while_a_person_is_here_and_a_rounds_question_is_the_rounds(tool, capsys):
    capsys.readouterr()
    tool.stub.pulse_answer = {"confirm": {"kind": "long", "prompt": "You are about to jump ahead. Proceed?"}, "narrations": {}, "round": None}
    _here(tool, 100, ("jakal", 900))
    (tool.folder / "next_action.txt").write_text("I wait.", encoding="utf-8")
    tool("act")
    tool("confirm", "yes")
    assert tool.stub.posted("/pulses/confirm")[-1]["proceed"] is False                       # no minutes, no end: not guessed
    tool.stub.round["confirm"] = {"kind": "long", "prompt": "Jump ahead 5 h?", "minutes": 300, "to": 400}
    _here(tool, 100, ("jakal", 100))
    tool("watch")                                                                # a question the ROUND holds: the people in it agree for themselves
    tool("confirm", "yes")
    assert tool.stub.posted("/pulses/confirm")[-1]["proceed"] is True


def test_the_guard_is_on_by_default_and_each_switch_turns_it_off(tool, capsys, monkeypatch):
    cfg = lambda: json.loads((tool.folder / "player.json").read_text(encoding="utf-8"))
    assert cfg()["guard"] is True                                                # the fixture's init: no flag, no variable
    args = ("init", "--id", ID, "--name", "Teo", "--port", str(tool.stub.port))
    tool(*args, "--guard", "off")
    assert cfg()["guard"] is False and "time guard is off" in capsys.readouterr().out
    _here(tool, 100, ("jakal", 100))
    capsys.readouterr()
    (tool.folder / "next_action.txt").write_text("I go on alone.", encoding="utf-8")
    tool("watch")
    assert "NOTHING OPENED" in _out(capsys) and tool("act") == 0                 # a free bot acts level with a person, as before
    monkeypatch.setenv("SMRITILILA_BOT_TIME_GUARD", "0")
    tool(*args)
    assert cfg()["guard"] is False                                               # the whole run's switch
    tool(*args, "--guard", "on")
    assert cfg()["guard"] is True                                                # and the flag is the more specific
    for off in ("off", "false", "No", " 0 "):
        monkeypatch.setenv("SMRITILILA_BOT_TIME_GUARD", off)
        tool(*args)
        assert cfg()["guard"] is False
    monkeypatch.setenv("SMRITILILA_BOT_TIME_GUARD", "on")
    tool(*args)
    assert cfg()["guard"] is True


def test_the_guard_reads_only_where_and_when_never_what(tool):
    _here(tool, 100, ("jakal", 100))
    tool("watch")
    gets = {p for m, p, b in tool.stub.requests if m == "GET"}
    assert not any(p.startswith("/players") or p.endswith("/anusmrti") for p in gets)
    assert "/world/sync?as=" + ID + "&own=1" in gets                              # the page's own list of who is here and when
