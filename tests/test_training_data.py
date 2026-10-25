# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging

import pytest

import smritilila.sakshi as sakshi
import smritilila.training_data as training_data
from smritilila.training_data import log_example

# code_version itself moved to smritilila.sakshi on 2026-09-22 -- the
# instrument layer owns a fact about the observed process, rather than
# this one stream owning it. Its own resolution tests moved with it, to
# tests/test_sakshi.py; what stays here is that this stream carries the
# value, caches it, and never spawns git when the feature is off.


@pytest.fixture(autouse=True)
def _reset_lazy_configuration(monkeypatch):
    """_configure() latches once per process (see training_data.py's own
    docstring, same shape as smritilila.sakshi) -- reset the latch, strip
    any FileHandler it attached, and reset the resolved code_version
    around every test in this file, or an earlier test's configuration
    (or lack of it) would leak into a later one via shared module state.
    Also stubs code_version to a fixed value by default -- most tests
    don't care about the real git state of whatever repo happens to be
    running them, and a real git subprocess call would make those
    assertions environment-dependent; tests that DO care override this
    stub themselves. Patched on THIS module, not on smritilila.sakshi: the
    name is bound here at import, so patching sakshi's own attribute
    would leave this module's bound reference untouched."""
    training_data._configured = False
    sakshi._code_version = None
    training_data._logger.propagate = True   # _configure turns it off once it has a file
    for handler in list(training_data._logger.handlers):
        training_data._logger.removeHandler(handler)
    monkeypatch.setattr(training_data, "code_version", lambda: "abc1234")
    # the pulse stamps are sakshi's facts; fixed here for the same reason
    monkeypatch.setattr(training_data, "run_id", lambda: "20260923T000000Z-1")
    monkeypatch.setattr(training_data, "current_world", lambda: "Ravenmoor")
    monkeypatch.setattr(training_data, "now", lambda: "2026-09-23T00:00:00.000+00:00")
    yield
    training_data._configured = False
    sakshi._code_version = None
    training_data._logger.propagate = True   # _configure turns it off once it has a file
    for handler in list(training_data._logger.handlers):
        training_data._logger.removeHandler(handler)


def test_log_example_writes_a_json_line_to_the_configured_file_path(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "training.jsonl"
    monkeypatch.setenv("SMRITILILA_TRAINING_DATA_LOG", str(log_path))

    log_example(
        backend="the second model", tool_name="narrate_pulse", system="sys prompt", user="user prompt",
        input_schema={"type": "object"}, output={"world_deltas": []},
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "backend": "the second model", "model": None, "reasoning": None, "tool_name": "narrate_pulse", "code_version": "abc1234",
        "run": "20260923T000000Z-1", "world": "Ravenmoor", "pulse_id": None, "when": "2026-09-23T00:00:00.000+00:00",
        "system": "sys prompt", "user": "user prompt",
        "input_schema": {"type": "object"}, "output": {"world_deltas": []},
    }


def test_log_example_says_which_model_settings_and_pulse_wrote_it(tmp_path, monkeypatch) -> None:
    """2026-09-23: finding the capture of Ravenmoor's t=342 meant searching
    for its action text, and its model could only be learned by joining
    to sakshi's llm.call. A line now says what wrote it and where it
    belongs."""
    log_path = tmp_path / "training.jsonl"
    monkeypatch.setenv("SMRITILILA_TRAINING_DATA_LOG", str(log_path))

    log_example(
        backend="the first provider", tool_name="append_tuples", system="s", user="u", input_schema={}, output={},
        model="model-a", reasoning={"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}},
        pulse_id="p342",
    )

    line = json.loads(log_path.read_text(encoding="utf-8"))
    assert line["model"] == "model-a"
    assert line["reasoning"] == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    assert (line["pulse_id"], line["run"], line["world"]) == ("p342", "20260923T000000Z-1", "Ravenmoor")
    assert line["when"] == "2026-09-23T00:00:00.000+00:00"


def test_log_example_without_the_env_var_creates_no_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SMRITILILA_TRAINING_DATA_LOG", raising=False)
    log_path = tmp_path / "training.jsonl"

    log_example(backend="the second model", tool_name=None, system="s", user="u", input_schema=None, output="text")

    assert not log_path.exists()


def test_log_example_appends_across_calls_without_attaching_a_duplicate_handler(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "training.jsonl"
    monkeypatch.setenv("SMRITILILA_TRAINING_DATA_LOG", str(log_path))

    log_example(backend="the second model", tool_name="t1", system="s", user="u1", input_schema=None, output="a")
    log_example(backend="the second model", tool_name="t1", system="s", user="u2", input_schema=None, output="b")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # not 4 -- a second _configure() call must not attach a second handler
    assert [json.loads(line)["user"] for line in lines] == ["u1", "u2"]


def test_log_example_never_raises_even_on_unserializable_output(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "training.jsonl"
    monkeypatch.setenv("SMRITILILA_TRAINING_DATA_LOG", str(log_path))

    log_example(backend="the second model", tool_name="t", system="s", user="u", input_schema=None, output=object())

    # default=str (see training_data.py) stringifies it rather than
    # raising -- confirmed by the line actually landing, not by catching.
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_log_example_skips_building_the_payload_when_unset(monkeypatch) -> None:
    """The `output`/`user` payload must not even be serialized when the
    feature is off -- the early handlers check exists specifically to
    skip that cost, not just skip the write. json.dumps(default=str)
    would call __str__ on an unserializable object if it ran at all;
    log_example's own broad try/except would silently swallow that
    either way, so a plain "did not raise" assertion can't distinguish
    "skipped" from "attempted and swallowed" -- this tracks whether
    __str__ was actually invoked instead."""
    monkeypatch.delenv("SMRITILILA_TRAINING_DATA_LOG", raising=False)
    calls: list[bool] = []

    class _Tracked:
        def __str__(self) -> str:
            calls.append(True)
            return "tracked"

    log_example(backend="the second model", tool_name="t", system="s", user="u", input_schema=None, output=_Tracked())

    assert calls == []


def test_log_example_skips_resolving_code_version_when_unset(monkeypatch) -> None:
    """git subprocess calls are real process spawns -- must not happen
    at all when the feature is off, not just be wasted."""
    monkeypatch.delenv("SMRITILILA_TRAINING_DATA_LOG", raising=False)
    calls: list[bool] = []
    monkeypatch.setattr(training_data, "code_version", lambda: calls.append(True) or "abc1234")

    log_example(backend="the second model", tool_name="t", system="s", user="u", input_schema=None, output="x")

    assert calls == []


def _counting_code_version(calls: list[bool]):
    """A stand-in for sakshi.code_version that resolves once and caches,
    the way the real one does -- so the caching assertion below is about
    log_example not defeating the cache, which is what this stream can
    actually get wrong."""
    cached: list[str] = []

    def resolve() -> str:
        if not cached:
            calls.append(True)
            cached.append("abc1234")
        return cached[0]

    return resolve


def test_log_example_reuses_the_resolved_code_version_across_calls(tmp_path, monkeypatch) -> None:
    """Resolved once per process (git doesn't change mid-process), not
    re-run on every single call."""
    log_path = tmp_path / "training.jsonl"
    monkeypatch.setenv("SMRITILILA_TRAINING_DATA_LOG", str(log_path))
    calls: list[bool] = []
    monkeypatch.setattr(training_data, "code_version", _counting_code_version(calls))

    log_example(backend="the second model", tool_name="t", system="s", user="u1", input_schema=None, output="a")
    log_example(backend="the second model", tool_name="t", system="s", user="u2", input_schema=None, output="b")

    assert len(calls) == 1
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["code_version"] for line in lines] == ["abc1234", "abc1234"]


def test_no_example_is_lost_to_calls_racing_the_very_first_one(tmp_path, monkeypatch) -> None:
    """The same latch race smritilila.sakshi had, and it bites harder here:
    log_example returns early when no handler is attached, so a call
    crossing the window has its training example thrown away, not merely
    unlogged. Measured before the fix: eight concurrent calls captured
    one of eight. Generative calls genuinely overlap -- the routes put
    genesis, bhuvana, expand and the deferred activation pulse through
    asyncio.to_thread."""
    import threading

    log_path = tmp_path / "training.jsonl"
    monkeypatch.setenv("SMRITILILA_TRAINING_DATA_LOG", str(log_path))
    released = threading.Barrier(8)

    def hammer(n: int) -> None:
        released.wait()
        log_example(backend="the first provider", tool_name="append_tuples",
                    system="s", user=f"u{n}", input_schema=None, output=f"o{n}")

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    captured = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(training_data._logger.handlers) == 1
    assert sorted(line["user"] for line in captured) == sorted(f"u{n}" for n in range(8))
