# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

import smritilila.sakshi as sakshi
from smritilila.sakshi import log_event, run_id, set_world


@pytest.fixture(autouse=True)
def _reset_lazy_configuration():
    """_configure() latches once per process, and the run id and the
    active world are process globals besides (see sakshi.py's own
    docstring) -- reset all three and strip any FileHandler around every
    test in this file, or an earlier test's configuration (or lack of
    it) would leak into a later one via shared module state."""
    def clear() -> None:
        sakshi._configured = False
        sakshi._run = None
        sakshi._world = None
        sakshi._logger.propagate = True   # _configure turns it off once it has a file
        for handler in list(sakshi._logger.handlers):
            sakshi._logger.removeHandler(handler)
    clear()
    yield
    clear()


def test_log_event_emits_one_json_line_with_the_event_name(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="smritilila.sakshi"):
        log_event("scope.participant_count", pulse_id="p1", count=3)

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload.pop("when")  # the three stamps are asserted on their own below
    assert payload.pop("run")
    assert payload.pop("world") is None
    assert payload == {"event": "scope.participant_count", "pulse_id": "p1", "count": 3}


def test_log_event_never_raises_even_on_unserializable_fields(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="smritilila.sakshi"):
        log_event("scope.participant_count", weird=object())  # not JSON-serializable by default

    # default=str (see sakshi.py) stringifies it rather than raising --
    # confirmed by the line actually landing, not by catching an exception.
    assert len(caplog.records) == 1


def test_log_event_writes_a_json_line_into_the_run_s_own_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SMRITILILA_SAKSHI_DIR", str(tmp_path))

    log_event("tick.source", tick=5, source="heap")

    written = tmp_path / f"{run_id()}.jsonl"
    lines = [json.loads(line) for line in written.read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == ["sakshi.run", "tick.source"]
    event = lines[1]
    assert event.pop("when") and event.pop("run") == run_id() and event.pop("world") is None
    assert event == {"event": "tick.source", "tick": 5, "source": "heap"}


def test_the_run_s_file_opens_with_one_line_naming_the_run_itself(tmp_path, monkeypatch) -> None:
    """Everything true of the whole run and of no single event in it is
    written once, at the top, rather than carried by every later line."""
    monkeypatch.setenv("SMRITILILA_SAKSHI_DIR", str(tmp_path))
    set_world("the crossing")

    log_event("tick.source", tick=1, source="heap")

    opening = json.loads((tmp_path / f"{run_id()}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert opening["event"] == "sakshi.run"
    assert opening["run"] == run_id()
    assert opening["world"] == "the crossing"
    assert opening["pid"] == os.getpid()
    assert "code_version" in opening  # None outside a git checkout; the key is always present


def test_a_run_id_sorts_by_start_time_and_carries_the_pid(tmp_path, monkeypatch) -> None:
    """The run id is also the file's name, so `ls` over the directory
    reads as a history of the server -- which needs it to sort. The pid
    is what keeps two servers started in the same second apart."""
    minted = run_id()
    stamp, _, pid = minted.partition("-")
    assert pid == str(os.getpid())
    started = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    assert abs((datetime.now(timezone.utc) - started).total_seconds()) < 60
    assert run_id() == minted  # minted once, never re-minted


def test_log_event_appends_across_calls_without_attaching_a_duplicate_handler(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SMRITILILA_SAKSHI_DIR", str(tmp_path))

    log_event("tick.source", tick=1, source="heap")
    log_event("tick.source", tick=2, source="exploration")

    lines = (tmp_path / f"{run_id()}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # the run line and two events -- not 5, a second _configure() must not attach a second handler
    assert [json.loads(line).get("tick") for line in lines] == [None, 1, 2]


def test_log_event_without_the_env_var_creates_no_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SMRITILILA_SAKSHI_DIR", raising=False)

    log_event("tick.source", tick=1, source="heap")

    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_directory_silences_the_witness_without_raising(tmp_path, monkeypatch) -> None:
    """The witness never affects what it witnesses -- the same D5
    tolerance the rest of this module keeps, applied to the one piece of
    setup that touches the filesystem."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("SMRITILILA_SAKSHI_DIR", str(blocker))

    log_event("tick.source", tick=1, source="heap")  # must not raise

    assert sakshi._logger.handlers == []


def test_no_line_is_lost_to_threads_racing_the_very_first_one(tmp_path, monkeypatch) -> None:
    """The window that made this fail was between setting the latch and
    attaching the handler -- a mkdir and a file open wide -- and a
    thread crossing it logged into a handler-less logger and was
    discarded. The routes run genesis, bhuvana, expand and the deferred
    activation pulse through asyncio.to_thread, all of which log, so the
    race sits exactly at process start. Measured before the fix: eight
    threads lost seven of nine lines, every attempt."""
    monkeypatch.setenv("SMRITILILA_SAKSHI_DIR", str(tmp_path))
    released = threading.Barrier(8)

    def hammer(n: int) -> None:
        released.wait()  # let them all arrive at _configure together
        log_event("probe", n=n)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    written = [json.loads(line) for line in
               (tmp_path / f"{run_id()}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(sakshi._logger.handlers) == 1  # one handler, not one per racing thread
    assert [line["event"] for line in written].count("sakshi.run") == 1
    assert sorted(line["n"] for line in written if line["event"] == "probe") == list(range(8))


def test_log_event_stamps_every_line_with_a_utc_wall_clock(caplog) -> None:
    """Added 2026-09-22: every line was timeless until then, so a run
    could not be asked when anything happened (see sakshi._now)."""
    with caplog.at_level(logging.INFO, logger="smritilila.sakshi"):
        log_event("tick.source", tick=1, source="heap")

    when = json.loads(caplog.records[0].message)["when"]
    stamped = datetime.fromisoformat(when)
    assert stamped.tzinfo is not None and stamped.utcoffset() == timedelta(0)
    assert abs((datetime.now(timezone.utc) - stamped).total_seconds()) < 60


def test_every_line_says_which_world_was_active_when_it_was_written(caplog) -> None:
    """Per line, not per run: /worlds/select switches the active world
    mid-process, and before this stamp existed four worlds' runs were
    indistinguishable in one undivided file."""
    with caplog.at_level(logging.INFO, logger="smritilila.sakshi"):
        set_world("the crossing")
        log_event("tick.source", tick=1, source="heap")
        set_world("Eyri")
        log_event("tick.source", tick=2, source="heap")

    assert [json.loads(r.message)["world"] for r in caplog.records] == ["the crossing", "Eyri"]


def test_an_explicit_when_field_wins_over_the_stamp(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="smritilila.sakshi"):
        log_event("tick.source", when="supplied", tick=1)

    assert json.loads(caplog.records[0].message)["when"] == "supplied"


# -- code_version: a fact about the observed process, moved here from
# smritilila.training_data on 2026-09-22 along with the function itself.


def test_resolve_code_version_appends_dirty_suffix_when_the_working_tree_has_changes(monkeypatch) -> None:
    class _FakeCompletedProcess:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    def _fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return _FakeCompletedProcess(0, "abc1234\n")
        if cmd[:2] == ["git", "diff"]:
            return _FakeCompletedProcess(0, b"-old line\n+new line\n")
        if cmd[:2] == ["git", "ls-files"]:
            return _FakeCompletedProcess(0, b"")
        return _FakeCompletedProcess(0, " M src/smritilila/main.py\n")  # git status --porcelain, one changed file

    monkeypatch.setattr(sakshi.subprocess, "run", _fake_run)

    stamp = sakshi._resolve_code_version()
    assert stamp is not None and stamp.startswith("abc1234-dirty.")
    assert len(stamp.split(".")[-1]) == 6


def _dirty_stamp(monkeypatch, diff: bytes) -> str | None:
    class _Done:
        def __init__(self, stdout) -> None:
            self.returncode, self.stdout = 0, stdout

    def _fake_run(cmd, **kwargs):
        return {"rev-parse": _Done("abc1234\n"), "status": _Done(" M x.py\n"),
                "diff": _Done(diff), "ls-files": _Done(b"")}[cmd[1]]

    monkeypatch.setattr(sakshi.subprocess, "run", _fake_run)
    return sakshi._resolve_code_version()


def test_two_different_uncommitted_states_get_two_different_stamps(monkeypatch) -> None:
    """2026-09-23: a bare "-dirty" stamped every unsaved state of one
    commit identically -- two captures under two different unsaved
    prompts claimed the same build. The same edit must still stamp the
    same in every process."""
    a = _dirty_stamp(monkeypatch, b"+rule 6, version A\n")
    b = _dirty_stamp(monkeypatch, b"+rule 6, version B\n")
    again = _dirty_stamp(monkeypatch, b"+rule 6, version A\n")
    assert a != b
    assert a == again


def test_a_dirty_tree_whose_diff_cannot_be_read_still_says_dirty(monkeypatch) -> None:
    class _Done:
        def __init__(self, code, stdout) -> None:
            self.returncode, self.stdout = code, stdout

    def _fake_run(cmd, **kwargs):
        return {"rev-parse": _Done(0, "abc1234\n"), "status": _Done(0, " M x.py\n"),
                "diff": _Done(128, b""), "ls-files": _Done(0, b"")}[cmd[1]]

    monkeypatch.setattr(sakshi.subprocess, "run", _fake_run)
    assert sakshi._resolve_code_version() == "abc1234-dirty"


def test_resolve_code_version_has_no_dirty_suffix_on_a_clean_tree(monkeypatch) -> None:
    class _FakeCompletedProcess:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    def _fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return _FakeCompletedProcess(0, "abc1234\n")
        return _FakeCompletedProcess(0, "")  # git status --porcelain, clean

    monkeypatch.setattr(sakshi.subprocess, "run", _fake_run)

    assert sakshi._resolve_code_version() == "abc1234"


def test_resolve_code_version_returns_none_when_git_is_unavailable(monkeypatch) -> None:
    def _fake_run(cmd, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(sakshi.subprocess, "run", _fake_run)

    assert sakshi._resolve_code_version() is None


# -- the reader (scripts/v3/sakshi.py). Imported by path: it is a script,
# deliberately outside the package, and importing it here is what keeps
# the one rule it must not break under test -- that two worlds' clocks
# are never averaged into one range.


def _reader():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "v3" / "sakshi.py"
    spec = importlib.util.spec_from_file_location("sakshi_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_span_is_one_world_s_clock_and_never_two() -> None:
    """The first thing the live instrument produced was a run that
    recalled the crossing at 456 and then loaded Ravenmoor at 221, and
    the listing reported "221-456" -- a range no clock ever traversed.
    world_time is a world's own clock; two of them share no scale."""
    reader = _reader()
    lines = [
        {"event": "world.loaded", "world": "The Vadhu Crossing", "world_time": 456},
        {"event": "world.loaded", "world": "Ravenmoor", "world_time": 221},
    ]
    assert reader._worlds(lines) == ["The Vadhu Crossing", "Ravenmoor"]
    assert reader._span(reader._within(lines, "The Vadhu Crossing")) == "456"
    assert reader._span(reader._within(lines, "Ravenmoor")) == "221"


def test_the_listing_gives_a_run_that_switched_worlds_a_row_for_each(tmp_path, capsys) -> None:
    """The guarantee is in the caller, not in _span: _show must never
    hand two worlds' lines to it at once. One row per world, each
    carrying that world's own clock and its own counts."""
    reader = _reader()
    record = tmp_path / "20260922T184718Z-1.jsonl"
    record.write_text("\n".join(json.dumps(line) for line in [
        {"event": "sakshi.run", "when": "2026-09-22T18:47:18.000+00:00", "world": "The Vadhu Crossing"},
        {"event": "world.loaded", "when": "2026-09-22T18:47:32.000+00:00",
         "world": "The Vadhu Crossing", "world_time": 456},
        {"event": "world.loaded", "when": "2026-09-22T18:47:44.000+00:00",
         "world": "Ravenmoor", "world_time": 221},
    ]) + "\n", encoding="utf-8")

    reader._show([record])

    rows = [line for line in capsys.readouterr().out.splitlines() if line.strip()][1:]  # drop the header
    assert len(rows) == 2
    assert "The Vadhu Crossing" in rows[0] and rows[0].split()[-4] == "456"
    assert "Ravenmoor" in rows[1] and rows[1].split()[-4] == "221"
    assert "221-456" not in "".join(rows)  # the range no clock ever traversed
    assert record.stem in rows[0] and record.stem not in rows[1]  # named once


def test_a_record_with_no_world_stamp_is_still_one_group() -> None:
    """The archive predates `run` and `world`. It still has to read."""
    reader = _reader()
    lines = [{"event": "smrti.pulse", "world_time": 5}, {"event": "smrti.pulse", "world_time": 9}]
    assert reader._worlds(lines) == []
    assert reader._within(lines, None) == lines
    assert reader._span(lines) == "5-9"


def test_a_run_with_no_pulses_hides_nothing(capsys) -> None:
    """Rarity is measured against the pulse, so a run with no pulses has
    no baseline and everything in it is what happened to it. The old
    max(pulses, 1) quietly made the threshold "twice or more is routine"
    for such a run: the first live record held four lines, two of them
    the world.loaded pair that IS the world switch, and the reader
    showed the other two and hid exactly those -- with no sign anything
    had been left out."""
    reader = _reader()
    lines = [
        {"event": "sakshi.run", "world": "The Vadhu Crossing"},
        {"event": "world.recalled", "world": "The Vadhu Crossing", "name": "The Vadhu Crossing"},
        {"event": "world.loaded", "world": "The Vadhu Crossing", "world_time": 456},
        {"event": "world.loaded", "world": "Ravenmoor", "world_time": 221},
    ]

    reader.ledger(lines, False)

    shown = capsys.readouterr().out
    assert "world.loaded" in shown  # fired twice; the switch itself
    assert "world.recalled" in shown and "sakshi.run" in shown
    assert "no pulses" in shown


def test_a_run_with_pulses_still_filters_what_is_routine(capsys) -> None:
    """The other direction: with a baseline, an event on every pulse is
    the engine working and stays out of the way."""
    reader = _reader()
    lines = []
    for n in range(4):
        lines.append({"event": "smrti.pulse", "pulse_id": f"p{n}", "world_time": n,
                      "actor": "teo", "autonomous": True, "tuples": 1, "written": 1})
        lines.append({"event": "llm.call", "pulse_id": f"p{n}", "input_tokens": 1, "output_tokens": 1})
    lines.append({"event": "smrti.stayed_behind", "pulse_id": "p2", "actor": "teo"})

    reader.ledger(lines, True)

    shown = capsys.readouterr().out
    assert "smrti.stayed_behind" in shown       # once in four pulses
    assert "llm.call" not in shown              # once per pulse: the engine working
    assert "4 pulses" in shown


def test_the_witness_does_not_also_print_to_the_console(tmp_path, monkeypatch) -> None:
    """A logger propagates to its ancestors by default and uvicorn puts
    a console handler on the root, so every line written here was also
    printed to the server's terminal -- an endless scroll under a live
    session (Marc, 2026-09-22). Off only once there IS a file: with none,
    records still propagate, which is what lets caplog see them and what
    keeps a misconfigured run from going quietly dark."""
    monkeypatch.setenv("SMRITILILA_SAKSHI_DIR", str(tmp_path))
    root = logging.getLogger()
    reached = []

    class Console(logging.Handler):
        def emit(self, record): reached.append(record.name)

    handler = Console()
    root.addHandler(handler)
    try:
        log_event("smrti.pulse", pulse_id="p1")
        assert reached == []                                   # nothing to the console
        assert (tmp_path / f"{run_id()}.jsonl").read_text(encoding="utf-8").strip()   # all of it to the file
    finally:
        root.removeHandler(handler)


def test_without_a_file_the_records_still_propagate(caplog) -> None:
    """The other half: silencing propagation unconditionally would make
    a run with no SMRITILILA_SAKSHI_DIR write nowhere AND say nothing."""
    with caplog.at_level(logging.INFO, logger="smritilila.sakshi"):
        log_event("smrti.pulse", pulse_id="p1")
    assert len(caplog.records) == 1
