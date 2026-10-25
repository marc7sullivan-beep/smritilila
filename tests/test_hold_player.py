# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""A held bot (2026-10-17, Marc: "bot mode but not taking actions"): the registry lists it, the supervisor starts the hold agent for it instead of one that
plays, and that agent keeps the character online and posts nothing. Run for real against the stub server of tests/test_player_tool.py; no model exists in it."""
import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
AGENTS = TESTS.parent / "scripts" / "agents"
sys.path.insert(0, str(AGENTS))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stubs = _load("test_player_tool_for_the_hold_stub", TESTS / "test_player_tool.py")
hp = _load("hold_player_under_test", AGENTS / "hold_player.py")
ID = stubs.ID


@pytest.fixture()
def stub():
    s = stubs._Stub().serve()
    yield s
    s.server.shutdown()


def test_the_registry_holds_an_active_bot_and_forgets_it_when_it_is_switched_off(tmp_path):
    from smritilila import bots_registry as b
    p = tmp_path / "bots.json"
    b.set_bot("Stonemarch", "dunstan", True, p)
    b.set_bot("Stonemarch", "other", True, p)
    assert b.load_hold(p) == {}
    assert b.set_hold("Stonemarch", "dunstan", True, p) == ["dunstan"] and b.load_hold(p) == {"Stonemarch": ["dunstan"]}
    b.set_alive("Stonemarch", {"dunstan": True}, p)                                    # another write keeps it
    b.set_important("Stonemarch", {"dunstan": False}, p)
    assert b.load_hold(p) == {"Stonemarch": ["dunstan"]} and b.load_bots(p) == {"Stonemarch": ["dunstan", "other"]}
    b.set_hold("Stonemarch", "ghost", True, p)                                         # not an active bot: nothing to hold
    assert b.load_hold(p) == {"Stonemarch": ["dunstan"]}
    b.set_bot("Stonemarch", "dunstan", False, p)                                       # switched off: no longer held, still known
    assert b.load_hold(p) == {} and "dunstan" in b.load_known(p)["Stonemarch"]
    b.set_bot("Stonemarch", "dunstan", True, p)
    assert b.load_hold(p) == {}                                                      # switched on again: played, unless held again
    assert b.set_hold("Stonemarch", "dunstan", True, p) == ["dunstan"] and b.set_hold("Stonemarch", "dunstan", False, p) == []


def test_a_held_character_is_online_and_beats_and_acts_for_no_one_and_leaves_waiting_when_stopped(stub, tmp_path):
    folder = tmp_path / ID
    ended = {}
    t = threading.Thread(target=lambda: ended.setdefault("why", hp.hold(folder, ID, "Teo", stub.port, beat=0.05, tick=0.01)), daemon=True)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline and len([b for b in stub.posted("/presence") if b == {"online": True}]) < 3:
        time.sleep(0.02)
    assert len([b for b in stub.posted("/presence") if b == {"online": True}]) >= 3          # on at the start, then the beat, again and again
    assert stub.posted("/pulses") == [] and stub.posted("/pulses/confirm") == []             # and nothing is ever done for them
    assert t.is_alive()
    hp.player_tool.main([str(folder), "stop", "--told"])                                     # what the supervisor does when the last person has gone
    t.join(10)
    assert not t.is_alive() and ended["why"] == "stopped"
    assert stub.posted("/autonomy") == [{"allow": True, "wait": True}] and stub.posted("/presence")[-1] == {"online": False}
    assert stub.posted("/pulses") == []


def test_a_held_character_survives_a_state_file_that_is_unreadable_for_a_moment(stub, tmp_path):
    """CI, 2026-10-18: the supervisor's stop rewrote player.json while the hold agent read it, a JSONDecodeError ended the agent's loop, and the test failed one run in four.
    The write is now whole-or-nothing, and a read that fails is tried again on the next tick."""
    folder = tmp_path / ID
    ended = {}

    def run():
        try:
            ended["why"] = hp.hold(folder, ID, "Teo", stub.port, beat=0.05, tick=0.01)
        except BaseException as e:                                                       # the loop must not die of a bad read
            ended["died"] = repr(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline and not (folder / "player.json").exists():
        time.sleep(0.01)
    (folder / "player.json").write_text("{", encoding="utf-8")                           # half a file
    time.sleep(0.15)
    assert t.is_alive() and "died" not in ended
    (folder / "player.json").write_text('{"stopped": true}', encoding="utf-8")            # readable again, and the run is over
    t.join(10)
    assert not t.is_alive() and ended == {"why": "stopped"}


def test_the_supervisor_starts_a_held_bot_with_the_hold_runner_and_finds_held_agents_too(monkeypatch, tmp_path):
    sup = _load("bots_supervisor_for_the_hold_test", TESTS.parent / "scripts" / "bots_supervisor.py")
    seen = {}

    class Hold:
        @staticmethod
        def launch(world, eid, name, port, folder, env, max_actions, idle):
            seen["hold"] = (world, eid, name, port, folder)
            return "held"

        @staticmethod
        def running_ids(folders):
            return {"h1"}

    class Plays:
        @staticmethod
        def launch(*a):
            seen["plays"] = a
            return "played"

        @staticmethod
        def running_ids(folders):
            return {"p1"}

    monkeypatch.setattr(sup, "runner", lambda: Plays)
    monkeypatch.setattr(sup, "hold_runner", lambda: Hold)
    monkeypatch.setattr(sup, "FOLDERS", tmp_path / "players")
    assert sup.launch("Stonemarch", "e1", "Dunstan", 8198, held=True) == "held" and "plays" not in seen
    assert seen["hold"] == ("Stonemarch", "e1", "Dunstan", 8198, tmp_path / "players" / "e1")
    assert sup.launch("Stonemarch", "e2", "Jakal", 8198) == "played"                           # the default is to play
    assert sup.agents_still_running() == {"h1", "p1"}


def test_the_hold_runner_starts_the_hold_agent_for_the_character(monkeypatch, tmp_path):
    rh = _load("runner_hold_under_test", AGENTS / "runner_hold.py")
    started = {}

    class Popen:
        def __init__(self, args, **kw):
            started["args"], started["kw"] = args, kw

    folder = tmp_path / "players" / "e1"
    folder.mkdir(parents=True)
    with monkeypatch.context() as m:                                                         # subprocess.run is built on Popen: the fake is only for the launch
        m.setattr(rh.subprocess, "Popen", Popen)
        rh.launch("Stonemarch", "e1", "Dunstan", 8198, folder, {"K": "v"}, 10, 30)
    a = started["args"]
    assert a[1].endswith("hold_player.py") and a[a.index("--id") + 1] == "e1" and a[a.index("--port") + 1] == "8198" and started["kw"]["env"] == {"K": "v"}
    assert (folder / "agent.log").exists()
