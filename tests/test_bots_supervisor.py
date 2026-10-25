# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""scripts/bots_supervisor.py and the bot registry (2026-10-06): the decision of what to start and stop, who counts as a person, and
(2026-10-17) that an agent is started and found through the runner the name in SMRITILILA_BOT_RUNNER points at."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bots_supervisor.py"
spec = importlib.util.spec_from_file_location("bots_supervisor", SCRIPT)
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)

NOW = 10_000.0


def test_a_bot_starts_when_a_person_is_active():
    assert sup.plan(["a", "b"], True, set(), NOW, None, {}) == (["a", "b"], [], None)


def test_a_bot_already_running_is_not_started_again():
    assert sup.plan(["a", "b"], True, {"a"}, NOW, None, {}) == (["b"], [], None)


def test_a_bot_that_ended_waits_out_the_cooldown_before_starting_again():
    assert sup.plan(["a"], True, set(), NOW, None, {"a": NOW - 60})[0] == []
    assert sup.plan(["a"], True, set(), NOW, None, {"a": NOW - sup.COOLDOWN - 1})[0] == ["a"]


def test_nobody_active_stops_the_bots_only_after_the_grace():
    start, stop, gone = sup.plan(["a"], False, {"a"}, NOW, None, {})
    assert (start, stop, gone) == ([], [], NOW)                                  # first seen gone: the clock starts, nothing stops
    assert sup.plan(["a"], False, {"a"}, NOW + sup.GRACE - 1, NOW, {})[1] == []
    assert sup.plan(["a"], False, {"a"}, NOW + sup.GRACE, NOW, {})[1] == ["a"]


def test_a_person_returning_cancels_the_departure():
    assert sup.plan(["a"], True, {"a"}, NOW + 100, NOW, {})[2] is None


def test_a_server_that_does_not_answer_is_not_a_departure():
    start, stop, gone = sup.plan(["a"], None, {"a"}, NOW + 10_000, NOW, {})
    assert (start, stop, gone) == ([], [], NOW)


def test_a_bot_switched_off_is_stopped_at_once_even_with_people_here():
    assert sup.plan(["a"], True, {"a", "b"}, NOW, None, {"a": NOW})[1] == ["b"]
    assert sup.plan([], False, {"b"}, NOW, None, {})[1] == ["b"]                   # nor does it wait out the grace
    assert sup.plan([], True, {"b"}, NOW, None, {})[1] == ["b"]


def test_a_person_is_a_character_online_that_is_not_a_bot(monkeypatch):
    players = [{"entity_id": "bot", "online": True}, {"entity_id": "p1", "online": False}]
    monkeypatch.setattr(sup, "_get", lambda port, path: players)
    assert sup.people_active(1, ["bot"]) is False                                 # the agent's own presence is no person
    players.append({"entity_id": "p2", "online": True})
    assert sup.people_active(1, ["bot"]) is True
    monkeypatch.setattr(sup, "_get", lambda port, path: None)
    assert sup.people_active(1, ["bot"]) is None


def test_the_bot_registry_switches_characters_on_and_off(tmp_path):
    from smritilila import bots_registry as b
    p = tmp_path / "bots.json"
    assert b.load_bots(p) == {}
    b.set_bot("Austin", "e1", True, p); b.set_bot("Austin", "e2", True, p); b.set_bot("Stonemarch", "g1", True, p)
    assert b.load_bots(p) == {"Austin": ["e1", "e2"], "Stonemarch": ["g1"]}
    b.set_bot("Austin", "e1", False, p); b.set_bot("Austin", "e1", False, p)    # off twice is off
    assert b.load_bots(p) == {"Austin": ["e2"], "Stonemarch": ["g1"]}
    b.set_bot("Stonemarch", "g1", False, p)
    assert "Stonemarch" not in b.load_bots(p)                                       # an empty world is not kept
    assert b.load_known(p) == {"Austin": ["e1", "e2"], "Stonemarch": ["g1"]}        # every one ever switched on stays known, active or not
    p.write_text("not json", encoding="utf-8")
    assert b.load_bots(p) == {}                                                   # a broken file is no bots, never a crash


def test_the_runner_is_the_file_the_name_points_at_and_a_wrong_name_says_so(monkeypatch, tmp_path):
    agents = Path(__file__).resolve().parent.parent / "scripts" / "agents"
    monkeypatch.setattr(sup, "RUNNER", "llm")
    assert sup.runner().__name__ == "runner_llm" and (agents / "runner_llm.py").exists()
    monkeypatch.setattr(sup, "RUNNER", "nobody")
    with pytest.raises(SystemExit, match="no bot runner named 'nobody'"):
        sup.runner()


def test_the_supervisor_starts_and_finds_agents_through_its_runner(monkeypatch, tmp_path):
    """The supervisor knows no model and no program: it makes the character's folder, hands the runner the limits, and asks it who is running."""
    seen = {}

    class Runner:
        @staticmethod
        def launch(world, eid, name, port, folder, env, max_actions, idle):
            seen["launch"] = (world, eid, name, port, folder, max_actions, idle)
            return "proc"

        @staticmethod
        def running_ids(folders):
            seen["folders"] = folders
            return {"e1"}

    monkeypatch.setattr(sup, "runner", lambda: Runner)
    monkeypatch.setattr(sup, "FOLDERS", tmp_path / "smritilila-players")
    assert sup.launch("Austin", "e2", "Maya", 8202) == "proc"
    assert seen["launch"] == ("Austin", "e2", "Maya", 8202, tmp_path / "smritilila-players" / "e2", sup.MAX_ACTIONS, sup.IDLE_MINUTES)
    assert (tmp_path / "smritilila-players" / "e2").is_dir()
    assert seen["launch"][2] == "Maya"                                             # the keys copied from the user's registry do not take the character's name
    assert sup.agents_still_running() == {"e1"} and seen["folders"] == tmp_path / "smritilila-players"


def test_a_free_bot_is_started_with_no_person_here_and_is_not_stopped_when_the_last_person_goes():
    """The time guard's switch (2026-10-22): a sim of bots alone, as in the Crossing, is a bot the registry frees."""
    assert sup.plan(["a", "b"], False, set(), NOW, None, {}, ["b"]) == (["b"], [], NOW)               # nobody here: only the free bot starts
    assert sup.plan(["a"], False, set(), NOW, None, {}) == ([], [], NOW)                               # and a guarded one never does
    assert sup.plan(["a", "b"], False, {"a", "b"}, NOW + sup.GRACE, NOW, {}, ["b"])[1] == ["a"]         # the grace ends the guarded one, not the free one
    assert sup.plan(["b"], False, set(), NOW, None, {"b": NOW - 60}, ["b"])[0] == []                    # the cooldown holds a free bot too
    assert sup.plan(["a", "b"], True, set(), NOW, None, {}, ["b"])[0] == ["a", "b"]                     # with a person here every bot starts, as ever
    assert sup.plan(["a"], None, set(), NOW, None, {}, ["a"]) == ([], [], None)                         # a server that does not answer starts nothing
    assert sup.plan([], False, {"b"}, NOW, None, {}, ["b"])[1] == ["b"]                                  # a free bot switched off is stopped at once: free only counts for an active bot


def test_the_registry_frees_an_active_bot_and_forgets_it_when_the_bot_goes(tmp_path):
    from smritilila import bots_registry as b
    p = tmp_path / "bots.json"
    b.set_bot("Crossing", "ilo", True, p)
    b.set_bot("Crossing", "teo", True, p)
    assert b.load_free(p) == {}
    assert b.set_free("Crossing", "ilo", True, p) == ["ilo"] and b.load_free(p) == {"Crossing": ["ilo"]}
    b.set_hold("Crossing", "teo", True, p)
    b.set_alive("Crossing", {"ilo": True}, p)                                                  # another write keeps the free list, as it keeps the hold list
    assert b.load_free(p) == {"Crossing": ["ilo"]} and b.load_hold(p) == {"Crossing": ["teo"]}
    assert b.set_free("Crossing", "ilo", False, p) == [] and b.load_free(p) == {}
    b.set_free("Crossing", "ilo", True, p)
    b.set_bot("Crossing", "ilo", False, p)
    assert b.load_free(p) == {}                                                                # only an active bot is free


def test_the_supervisor_names_the_worlds_other_bots_and_switches_the_guard_off_only_for_a_free_one(monkeypatch, tmp_path):
    seen = []

    class Runner:
        @staticmethod
        def launch(world, eid, name, port, folder, env, max_actions, idle):
            seen.append((eid, env.get("SMRITILILA_BOT_PEERS"), env.get("SMRITILILA_BOT_TIME_GUARD")))
            return "proc"

    monkeypatch.setattr(sup, "runner", lambda: Runner)
    monkeypatch.setattr(sup, "FOLDERS", tmp_path / "smritilila-players")
    monkeypatch.delenv("SMRITILILA_BOT_TIME_GUARD", raising=False)
    sup.launch("Crossing", "ilo", "Ilo", 8199, peers=["ilo", "nara"])
    sup.launch("Crossing", "nara", "Nara", 8199, peers=["ilo", "nara"], free=True)
    assert seen == [("ilo", "nara", None), ("nara", "ilo", "0")]                               # a bot is not its own peer; the guard is on unless freed
