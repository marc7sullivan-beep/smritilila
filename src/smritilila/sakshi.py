# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Sākṣī -- the witness. The instrument layer: what the engine owes an
observer, and the only place a run is written down.

Named for the witness-consciousness that observes without participating
and is unaffected by what it observes, which is exactly the contract
this module already kept before it had a name: every log_event call is
fire-and-forget, read by nothing in engine code (the engine does ask
current_world() for the name of the world being served, and
adhisthana.get_world binds it), and a logging failure
must never affect the pulse it is instrumenting (the D5 tolerance
already applied to memory inheritance, late arrival and unresolved
speakers elsewhere in the pipeline). The engine cannot see sākṣī; sākṣī
sees the engine.

(The name was v2's, for a corroboration layer retired on 2026-09-21 --
see v3.0 §"What Is Retired". Lines in the archived telemetry carrying
`sakshi.disagreement` or `sakshi.unbound` are that other thing, not
this one. The archive is the only place they appear.)

One JSON object per log line, on the "smritilila.sakshi" logger. Three
stamps ride on every line, none of which the caller supplies:

  `when`  -- when the MACHINE did this, UTC, to the millisecond.
             Deliberately not `world_time`, and deliberately not a
             Sanskrit name: world_time is the world's own clock and
             belongs to the world, while this belongs to the observer
             standing outside it. Added 2026-09-22; until then every
             event was timeless.
  `run`   -- which process. Minted once, on first ask (in practice the
             first log_event), never changes.
  `world` -- which world was active when the line was written. Per
             line, not per run, because /worlds/select can switch the
             active world mid-process.

A run's lines land in their own file: SMRITILILA_SAKSHI_DIR names a
DIRECTORY, and each process writes `<dir>/<run>.jsonl`. This is what
bounds the store -- one file per run rather than one file forever --
and it is why the run id is built to sort and to be read at a glance
rather than to be short.

Unset (the default), events still fire but land nowhere durable --
nothing here forces every process (including the test suite) to write a
file just for importing this module. The handler is attached lazily, on
the first log_event() call, not at import time, and only once per
process.

Superseded 2026-09-22: SMRITILILA_TELEMETRY_LOG, which named one file that
every run of every world appended to forever. By the day it was
retired, `telemetry.jsonl` held 47,386 lines over 134 event names, of
which only 43 names still had an emitter anywhere in the tree: 71% of
the file had been written by machinery deleted on 2026-09-21, and one
dead event alone (`world_builder.malformed_entry`) was 29% of it. No
line said which world it belonged to, so four worlds' runs were
indistinguishable, and no line said which run, so every restart since
the beginning ran together with no seam. That file is kept, as the only
surviving record of how v2 behaved, and moved to
`archive/telemetry-through-2026-09-22.jsonl`; it is not appended to
again. See archive/README.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path


_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # a console program started from a process with no console opens a visible window; this keeps it hidden (Marc, 2026-10-10)


def _run_hidden(*a, **k):
    return subprocess.run(*a, creationflags=k.pop("creationflags", _NOWIN), **k)

_logger = logging.getLogger("smritilila.sakshi")
_configured = False
_configuring = threading.Lock()
_run: str | None = None
_world: str | None = None
# the world a request took, carried into the background tasks it starts (item 6, 2026-09-30)
_taken: ContextVar[str | None] = ContextVar("sakshi_world", default=None)
_code_version: str | None = None

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _now() -> str:
    """`when`, above. One datetime.now() per event: O(1), no growth with
    the world.

    Named `when` and not `at` because `at` was already taken, twice, for
    something else: smrti.narration_markup logs `at` as an offset into
    the narration, and smrti.actor_stands (retired with actor_stands in
    v4) logged it as a place id. A stamp called `at` would have been
    silently overwritten on those events and, worse, a reader would find
    `"at": "the-taproom"` on one line and a timestamp on the next with
    nothing to distinguish them. The stamp got the free name.

    An explicit `when` from a caller still wins, same as any field."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _mint_run() -> str:
    """`run`, above: when this process started and which process it was.
    Sortable, and legible without opening the file -- it is also the
    file's own name, so `ls` over the run directory reads as a history
    of the server.

    The pid is what makes it unique rather than merely descriptive: two
    servers run on this machine at once (Marc's 8198 and the crossing's
    8199, scripts/dev_server.py), and they can start within the same
    second."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"


def code_version() -> str | None:
    """The git short SHA this process is running, plus a "-dirty.<hash>"
    suffix if the working tree had uncommitted changes when it was
    resolved -- the hash naming WHICH changes (2026-09-23: a bare
    "-dirty" stamped every uncommitted state of a commit identically, so
    two captures under different unsaved prompts claimed the same build).

    An observation about the observed, so it lives here rather than in
    any one stream's own module (smritilila.training_data reads it from here;
    it was written there first, 2026-09-16, before there was an
    instrument layer to put it in). This project's prompts and rules
    change session to session, so a line captured under one version can
    silently misrepresent what the current tree would produce. Resolved
    once per process -- git does not change mid-process."""
    global _code_version
    if _code_version is None:
        _code_version = _resolve_code_version() or ""
    return _code_version or None


def _resolve_code_version() -> str | None:
    try:
        commit = _run_hidden(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        if commit.returncode != 0:
            return None
        dirty = _run_hidden(
            ["git", "status", "--porcelain"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        suffix = ""
        if dirty.returncode == 0 and dirty.stdout.strip():
            which = _dirty_hash()
            suffix = "-dirty" + (f".{which}" if which else "")
        return commit.stdout.strip() + suffix
    except Exception:
        return None


# An untracked file bigger than this is hashed by path and size only:
# the stamp has to be cheap at process start, and a multi-megabyte
# stray file is never the prompt change the stamp exists to tell apart.
_DIRTY_FILE_BYTES = 1 << 20


def _dirty_hash() -> str | None:
    """Six hex characters naming the uncommitted state: the diff against
    HEAD plus every untracked, non-ignored file. Two different unsaved
    edits of one commit get two different stamps; the same edit gets the
    same stamp in every process. Cost is one diff and one file listing at
    process start -- bounded by the size of the uncommitted change, never
    by the world or the run."""
    try:
        h = hashlib.sha1()
        diff = _run_hidden(["git", "diff", "HEAD", "--binary"], cwd=_REPO_ROOT, capture_output=True, timeout=10)
        if diff.returncode != 0:
            return None
        h.update(_as_bytes(diff.stdout))
        listed = _run_hidden(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=_REPO_ROOT, capture_output=True, timeout=10,
        )
        if listed.returncode != 0:
            return None
        for name in sorted(n for n in _as_bytes(listed.stdout).split(b"\0") if n):
            h.update(b"\0untracked\0" + name)
            path = _REPO_ROOT / name.decode("utf-8", "replace")
            try:
                size = path.stat().st_size
                h.update(path.read_bytes() if size <= _DIRTY_FILE_BYTES else f"size:{size}".encode())
            except OSError:
                h.update(b"unreadable")
        return h.hexdigest()[:6]
    except Exception:
        return None


def _as_bytes(out: object) -> bytes:
    return out if isinstance(out, bytes) else str(out or "").encode("utf-8")


def current_world() -> str | None:
    """The world stamped on this line: the one this task took (bind_world),
    else the one this process is serving (set_world)."""
    return _taken.get() or _world


def now() -> str:
    """The `when` stamp, for another instrument (smritilila.training_data)."""
    return _now()


def set_world(name: str | None) -> None:
    """Restamp the active world. Called wherever a route (routes/loka.py) changes which
    world this process is serving -- selection, creation, and the
    recall a restart does -- never by the engine.

    A process-global rather than a ContextVar, which is not a shortcut:
    the server holds exactly one world at a time by design (database per
    world, one active in-process -- routes/loka.py's /worlds/select), so there
    is no second world for a background activation pulse or a scheduler
    tick to belong to. A ContextVar would have to be set on every entry
    point that is not a request, and would still only ever hold this
    same value.

    Not quite (item 6, 2026-09-30): a turn begun in one world can finish
    after a select, and was then logged under the new one. So a task also
    keeps the world it took -- bind_world, below -- and the process-global
    is what a line carries only where no task took one."""
    global _world
    _world = name


def bind_world(name: str | None) -> None:
    """This task's lines, and those of the tasks it starts, name `name`
    whatever the process selects meanwhile (adhisthana.get_world binds it)."""
    _taken.set(name)


def run_id() -> str:
    """This process's run id, minted on first ask."""
    global _run
    if _run is None:
        _run = _mint_run()
    return _run


def _configure() -> None:
    """Attach the run's file handler, once, and hold every other caller
    at the door until it is attached.

    The lock is not ceremony. Until 2026-09-22 this set `_configured`
    at the TOP and attached the handler a dozen lines later, across a
    mkdir and a file open -- so any thread that called log_event inside
    that window saw the latch already set, returned immediately, and
    logged into a logger with no handler on it. Its line was discarded,
    silently, which is the one failure an instrument must not have.
    Measured: eight threads released together lost seven of their nine
    lines, every attempt. Inherited from the original Phase 0
    instrumentation (7aec707) and never triggered on purpose, but
    the routes (routes/bija.py, routes/spanda.py) run several things through asyncio.to_thread -- genesis,
    bhuvana, expand, the deferred activation pulse -- and all of them
    log, so the window is real and it sits exactly at process start,
    where the run line and the first world are written."""
    global _configured
    if _configured:
        return
    with _configuring:
        if _configured:  # another thread finished while this one waited
            return
        try:
            directory = os.environ.get("SMRITILILA_SAKSHI_DIR")
            if directory:
                path = Path(directory)
                path.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(path / f"{run_id()}.jsonl", mode="a", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(message)s"))
                _logger.addHandler(handler)
                _logger.setLevel(logging.INFO)
                # Ours alone from here (2026-09-22). A logger propagates
                # to its ancestors by default, and uvicorn puts a console
                # handler on the root -- so every line written here was
                # also printed to the terminal the server runs in, an
                # endless scroll under a live session. Turned off only
                # once we HAVE a file: with no file the records still
                # propagate, which is what lets caplog see them in tests
                # and what keeps a misconfigured run from going silent.
                _logger.propagate = False
                # The run's own first line: everything true of the whole
                # run and of no single event in it, written once so no
                # later line has to carry it. Emitted inside the lock so
                # it cannot be overtaken by the event that triggered it.
                _logger.info(json.dumps({
                    "event": "sakshi.run", "when": _now(), "run": run_id(), "world": current_world(),
                    "pid": os.getpid(), "code_version": code_version(),
                }))
        except OSError:
            pass  # an unwritable directory silences the witness; it never stops the engine
        finally:
            _configured = True  # last, so nobody leaves before the handler is on

def log_event(event: str, **fields: object) -> None:
    try:
        _configure()
        _logger.info(json.dumps(
            {"event": event, "when": _now(), "run": run_id(), "world": current_world(), **fields},
            default=str,
        ))
    except Exception:
        pass
