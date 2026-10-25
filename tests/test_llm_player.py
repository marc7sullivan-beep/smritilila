# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""scripts/agents/llm_player.py and runner_llm.py (2026-10-17): a bot that plays a character through the engine's own LLM client. Run for real against the
stub server of tests/test_player_tool.py, with a fake model: no model is called. What matters is that the tool still keeps the count and the stop
conditions, that the model is shown the character's page and nothing of the tool's rule, and that however the run ends the character is left waiting."""
import importlib.util
import os
import sys
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


stubs = _load("test_player_tool_for_the_stub", TESTS / "test_player_tool.py")
ID = stubs.ID


class FakeModel:
    """Answers from a list (the last one repeats) and keeps what it was shown. `on_ask` lets a test change the world as the model answers."""

    def __init__(self, replies, on_ask=None):
        self.replies, self.on_ask, self.seen = list(replies), on_ask, []

    def complete(self, system, user, *, max_tokens=8192, prompt=None):
        self.seen.append((system, user))
        if self.on_ask:
            self.on_ask(user)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture()
def lp(monkeypatch):
    mod = _load("llm_player_under_test", AGENTS / "llm_player.py")
    for k, v in (("FAST", True), ("POLL", 0.01), ("QUIET", 0.4), ("RETRY_WAIT", 0)):
        monkeypatch.setattr(mod.player_tool, k, v)
    return mod


@pytest.fixture()
def stub():
    s = stubs._Stub().serve()
    yield s
    s.server.shutdown()


def _play(lp, stub, tmp_path, client, max_actions=2):
    return lp.play(tmp_path / ID, ID, "Teo", stub.port, "Austin", max_actions, 30, client)


def _left_waiting(stub):
    return stub.posted("/autonomy") == [{"allow": True, "wait": True}] and stub.posted("/presence")[-1] == {"online": False}


def test_the_footer_of_the_tool_is_not_shown_to_the_model(lp):
    out = "WHERE THEY ARE: x\n[Teo: 1 of 25 actions] NOT A STOP. A lull is not one.\n"
    assert lp.clean(out) == "WHERE THEY ARE: x"
    assert lp.clean("TOLD:\nA telling.\n[Teo: 3 of 25 actions] NOT A STOP.") == "TOLD:\nA telling."
    assert lp.clean("no footer here") == "no footer here"


def test_an_action_is_one_run_of_text_not_wrapped_in_quotes_and_not_longer_than_a_player_types(lp):
    assert lp.action_from('"I look at Nara."') == "I look at Nara."
    assert lp.action_from("I nod.\n\n  Then I wait.") == "I nod. Then I wait."
    assert lp.action_from("He said \"hello\" to her.") == 'He said "hello" to her.'
    assert len(lp.action_from("x" * 5000)) == lp.MAX_REPLY


def test_a_world_with_a_note_file_uses_it_and_one_without_says_so(lp, monkeypatch, tmp_path):
    (tmp_path / "worlds").mkdir()
    (tmp_path / "worlds" / "ferry.md").write_text("A ferry on a river.\n", encoding="utf-8")
    monkeypatch.setattr(lp, "HERE", tmp_path)
    assert lp.note_for("The Ferry Crossing") == "A ferry on a river."
    assert lp.note_for("Austin") == lp.NO_NOTE


def test_the_system_prompt_names_the_character_and_leaves_no_brace_unfilled(lp):
    text = lp.system_prompt("Teo", "Austin")
    assert "You play one character: Teo." in text and lp.NO_NOTE in text
    assert "{" not in text


def test_the_model_plays_the_character_to_the_limit_then_the_character_is_left_waiting(lp, stub, tmp_path):
    model = FakeModel(["I look at Nara and wait.", "--- I nod to her."])
    reason = _play(lp, stub, tmp_path, model)
    assert reason == "2 actions taken"
    assert [b["action_text"] for b in stub.posted("/pulses")] == ["I look at Nara and wait.", "--- I nod to her."]   # an action that starts with a dash is still an action
    assert all(b["entity_id"] == ID for b in stub.posted("/pulses"))
    assert _left_waiting(stub)
    system, user = model.seen[0]
    assert "Teo" in system and "I was pulled from the river by Nara." in user                    # the page, as the character's own
    assert "NOT A STOP" not in user and "Run `stop`" not in user                                  # the rule is for an agent that reads the tool, not for the model
    assert "I look at Nara and wait." in model.seen[1][1]                                         # the second turn remembers the first


def test_a_telling_is_shown_to_the_model_on_the_turn_that_answers_it(lp, stub, tmp_path):
    stub.round.update({"told": "Nara pulls the oars and the ferry swings.", "open": True})
    model = FakeModel(["I hold the rope."])
    _play(lp, stub, tmp_path, model, max_actions=1)
    assert "Nara pulls the oars and the ferry swings." in model.seen[0][1]


def test_a_question_about_jumping_ahead_is_answered_as_the_model_answers_it_and_never_by_an_action(lp, stub, tmp_path):
    stub.round["confirm"] = "Jump ahead two hours?"

    def clear(user):
        if "Answer YES or NO" in user:
            stub.round["confirm"] = None

    model = FakeModel(["Yes, he would.", "I sit by the fire."], on_ask=clear)
    _play(lp, stub, tmp_path, model, max_actions=1)
    assert stub.posted("/pulses/confirm") == [{"entity_id": ID, "proceed": True}]
    assert [b["action_text"] for b in stub.posted("/pulses")] == ["I sit by the fire."]
    assert "Jump ahead two hours?" in model.seen[0][1]


def test_a_model_that_cannot_be_reached_leaves_the_character_waiting_having_acted_for_no_one(lp, stub, tmp_path):
    model = FakeModel([RuntimeError("503 high demand")])
    reason = _play(lp, stub, tmp_path, model)
    assert reason.startswith("the model could not be reached") and "503" in reason
    assert len(model.seen) == lp.TRIES and stub.posted("/pulses") == []
    assert _left_waiting(stub)


def test_an_unexpected_failure_still_leaves_the_character_waiting(lp, stub, tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "compose", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        _play(lp, stub, tmp_path, FakeModel(["x"]))
    assert _left_waiting(stub)


def test_without_an_llm_configured_it_says_so_and_touches_nothing(lp, monkeypatch, tmp_path, capsys):
    import smritilila.llm.client as c

    def none(model=None):
        raise KeyError("THE_FIRST_API_KEY")

    monkeypatch.setattr(c, "best_available_client", none)
    folder = tmp_path / "teo"
    assert lp.main([str(folder), "--id", ID, "--name", "Teo", "--port", "1"]) == 2
    assert "no LLM is configured" in capsys.readouterr().out and not folder.exists()


def test_a_run_from_the_command_line_uses_the_engines_client_and_clears_its_pid_file(lp, monkeypatch, tmp_path, stub):
    import smritilila.llm.client as c

    seen = {}
    monkeypatch.setattr(c, "best_available_client", lambda model=None: seen.setdefault("model", model) or FakeModel(["I wait."]))
    monkeypatch.setenv("SMRITILILA_BOT_MODEL", "some-model-id")
    folder = tmp_path / "teo"
    assert lp.main([str(folder), "--id", ID, "--name", "Teo", "--port", str(stub.port), "--world", "Austin", "--max", "1"]) == 0
    assert seen["model"] == "some-model-id"                                       # the model is the configuration's, or the one the variable names
    assert not (folder / "agent.pid").exists() and _left_waiting(stub)


def test_the_runner_starts_the_agent_with_the_character_and_finds_agents_by_their_pid_files(monkeypatch, tmp_path):
    rl = _load("runner_llm_under_test", AGENTS / "runner_llm.py")
    started = {}

    class Popen:
        def __init__(self, args, **kw):
            started["args"], started["kw"] = args, kw

    folder = tmp_path / "smritilila-players" / "e1"
    folder.mkdir(parents=True)
    with monkeypatch.context() as m:                                                       # subprocess.run is built on Popen: the fake is only for the launch
        m.setattr(rl.subprocess, "Popen", Popen)
        rl.launch("Austin", "e1", "Maya", 8202, folder, {"K": "v"}, 10, 30)
    a = started["args"]
    assert a[1].endswith("llm_player.py") and a[a.index("--id") + 1] == "e1" and a[a.index("--port") + 1] == "8202"
    assert a[a.index("--world") + 1] == "Austin" and a[a.index("--max") + 1] == "10" and started["kw"]["env"] == {"K": "v"}
    assert (folder / "agent.log").exists()
    (folder / "agent.pid").write_text(str(os.getpid()), encoding="utf-8")                  # an agent that is running
    gone = tmp_path / "smritilila-players" / "e2"
    gone.mkdir()
    (gone / "agent.pid").write_text("99999999", encoding="utf-8")                          # one that ended without clearing its file
    (tmp_path / "smritilila-players" / "e3").mkdir()                                            # a folder that was never played
    assert rl.running_ids(tmp_path / "smritilila-players") == {"e1"}
    assert rl.running_ids(tmp_path / "nowhere") == set()


def test_a_held_character_asks_the_model_nothing_and_acts_the_moment_a_person_is_ahead(lp, stub, tmp_path):
    """The time guard (2026-10-22): `watch` says HOLD, and the loop does nothing, costing no model call, until the tool lets it act."""
    stub.sync = [{"entity_id": ID, "label": "Teo", "at": 100}, {"entity_id": "jakal", "label": "Jakal", "at": 100}]
    holds = {"n": 0}
    real = lp.call

    def run(folder, *args):
        out = real(folder, *args)
        if args and args[0] == "watch" and out.startswith("HOLD"):
            holds["n"] += 1
            if holds["n"] == 3:                                                  # the person moves on
                stub.sync[1]["at"] = 400
        return out

    model = FakeModel(["I follow the road."])
    reason = lp.play(tmp_path / ID, ID, "Teo", stub.port, "Austin", 1, 30, model, run=run)
    assert reason == "1 actions taken" and holds["n"] == 3
    assert len(model.seen) == 1                                                  # three HOLDs cost nothing: the model was asked once, to act
    assert [b["action_text"] for b in stub.posted("/pulses")] == ["I follow the road."]


def test_an_action_the_tool_held_is_not_remembered_as_done(lp, stub, tmp_path):
    stub.sync = [{"entity_id": ID, "label": "Teo", "at": 100}]
    state = {"first": True}
    real = lp.call

    def run(folder, *args):
        if args and args[0] == "act" and state["first"]:
            state["first"] = False
            return "HOLD: a person is level with Teo.\nNOT POSTED, and not counted: the action stays in next_action.txt.\n[Teo: 0 of 1 actions] NOT A STOP."
        return real(folder, *args)

    model = FakeModel(["I knock.", "I knock again."])
    lp.play(tmp_path / ID, ID, "Teo", stub.port, "Austin", 1, 30, model, run=run)
    assert "Teo did: I knock." not in model.seen[1][1]                           # refused, so the next turn is not told it happened
    assert [b["action_text"] for b in stub.posted("/pulses")] == ["I knock again."]


def test_a_jump_question_that_comes_back_with_an_action_is_answered_not_thrown_away(lp, stub, tmp_path):
    stub.pulse_answer = {"confirm": {"kind": "long", "prompt": "You are about to jump ahead 1 h. Proceed?", "minutes": 60, "to": 160}, "narrations": {}, "round": None}
    model = FakeModel(["I sleep.", "YES"])
    lp.play(tmp_path / ID, ID, "Teo", stub.port, "Austin", 1, 30, model)
    assert stub.posted("/pulses/confirm") == [{"entity_id": ID, "proceed": True}]           # nobody else is here, so the model's yes stands
    assert "You are about to jump ahead 1 h" in model.seen[1][1]
