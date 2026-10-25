# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The telescope: what a run can be asked, from the outside.

smritilila.sakshi writes; this reads. Nothing here imports the engine, opens
a database, or calls a model -- a run's file is the whole input, and
answering a question about a run costs one pass over that run and
nothing at all over any other. That is the bound: the store grows one
file per run, and a question is asked of one file.

Three questions, which is all there is so far:

    python scripts/v3/sakshi.py                  which runs are there
    python scripts/v3/sakshi.py <run>            what happened in one
    python scripts/v3/sakshi.py <run> <pulse>    everything about one pulse

Times are shown twice: the record's own UTC, and the reader's local
clock. sakshi writes UTC because a record that moves with whoever reads
it is not a record; a session is remembered in local time.

<run> is a run id, a prefix of one, or a path to any .jsonl in sakshi's
own shape. Closed records in archive/ answer to their own names the
same way -- their lines predate the `run` and `world` stamps and are
read anyway, with those columns left blank.

On what counts as worth looking at (`--rare`). Deliberately NOT a list
of bad event names. This project has just spent a day discovering that
71% of its own telemetry named machinery that no longer exists; a
curated taxonomy of anomalies would rot exactly the same way, silently,
and would have to be edited every time the engine learns a new way to
complain. Rarity needs no maintenance and no judgment about the world:
in a run of n pulses, the events that fire on every pulse are the
engine working, and the ones that fire a handful of times are the ones
that happened TO this run. An event carrying a non-null `error` is
always shown, however common it is.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

# Windows hands a console stdout encoded cp1252, which cannot represent
# a world named "The Vadhū Crossing" -- the reader died on its own first
# real run, printing the listing, on the name of the world this project
# plays in most. The files are UTF-8 by construction (sakshi writes them
# that way); only the terminal was narrow. `errors="replace"` so a
# console that still cannot show a glyph prints a box rather than
# refusing to answer the question.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent.parent
_SAKSHI_DIR = _ROOT / "sakshi"
_ARCHIVE_DIR = _ROOT / "archive"

# A pulse's own summary line, and the two events that bracket it. Read
# by name because these three ARE the ledger's columns -- every other
# event is shown by rarity or by pulse, never by name.
_PULSE = "smrti.pulse"
_ACTION = "pulse.player_action"
_REQUEST = "request.timing"


def read(path: Path) -> list[dict]:
    """One pass, skipping anything unparseable -- a run cut off
    mid-write by a killed server ends in half a line, and that is not a
    reason to refuse to read the rest of it."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
    return out


def runs() -> list[Path]:
    return sorted(_SAKSHI_DIR.glob("*.jsonl")) if _SAKSHI_DIR.is_dir() else []


def archived() -> list[Path]:
    """Closed records (archive/README.md). Named alongside the live runs
    rather than behind a flag: a question about a closed record is the
    same question, and the whole point of keeping one is being able to
    ask it."""
    return sorted(_ARCHIVE_DIR.glob("*.jsonl")) if _ARCHIVE_DIR.is_dir() else []


def resolve(name: str) -> Path:
    """A run id, a prefix of one, or a path to any file in this shape."""
    as_path = Path(name)
    if as_path.is_file():
        return as_path
    matches = [p for p in runs() + archived() if p.stem.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit(f"nothing named {name!r} in {_SAKSHI_DIR} or {_ARCHIVE_DIR} "
                 "(and no file at that path)")
    sys.exit("that prefix names several runs:\n  " + "\n  ".join(p.stem for p in matches))


def _clock(when: str | None) -> str:
    """`when` is ISO UTC; the run listing wants it human and narrow."""
    return (when or "").replace("T", " ")[:16]


def _local(when: str | None, fmt: str = "%m-%d %H:%M") -> str:
    """The same instant on the reader's own clock.

    sakshi records UTC on purpose -- a world is played from wherever its
    player is, and a record that moves with the reader is not a record.
    But a session is REMEMBERED in local time: "the run after dinner",
    not "the run at 23:24Z". Both columns, then, and no ambiguity about
    which is which.

    (Written 2026-09-22 after I spent an evening reading this tool's own
    UTC stamps as local and telling Marc it was nearly midnight. It was
    a quarter to seven.)

    Falls back to the raw string if `when` is not a stamp we wrote -- an
    explicit `when` from a caller wins over the stamp everywhere else,
    so it may be anything at all."""
    try:
        return datetime.fromisoformat(when).astimezone().strftime(fmt)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return (when or "")[:16]


def _time_of(when: str | None) -> str:
    """Within one pulse, on the reader's clock -- to the millisecond,
    because two events regularly land in the same second and the
    question there is what happened before what."""
    try:
        return datetime.fromisoformat(when).astimezone().strftime("%H:%M:%S.%f")[:-3]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return (when or "").partition("T")[2][:12]


def _worlds(lines: list[dict]) -> list[str]:
    """Every world this record touched, first seen first. A run that
    switched mid-way names both rather than reporting whichever it
    happened to start in."""
    return [w for w in dict.fromkeys(line.get("world") for line in lines) if w]


def _within(lines: list[dict], world: str | None) -> list[dict]:
    """The lines belonging to one world, or all of them for a record
    that predates the stamp (the archive) and has no worlds to divide
    by."""
    return lines if world is None else [ln for ln in lines if ln.get("world") == world]


def _span(lines: list[dict]) -> str:
    """How far a world's clock moved. Only ever asked of ONE world's
    lines: world_time is that world's own clock, and two worlds' ticks
    share no scale. A run that recalled the crossing at 456 and then
    loaded Ravenmoor at 221 reported "221-456" -- a range no clock ever
    traversed, and the first thing the live instrument produced. The
    listing splits such a run into a row per world instead."""
    ticks = sorted({line["world_time"] for line in lines
                    if isinstance(line.get("world_time"), int)})
    if not ticks:
        return "--"
    return str(ticks[0]) if ticks[0] == ticks[-1] else f"{ticks[0]}-{ticks[-1]}"


def list_runs() -> None:
    found = runs()
    if not found:
        print(f"no runs in {_SAKSHI_DIR}")
        print("(a server writes there when SMRITILILA_SAKSHI_DIR is set -- scripts/dev_server.py sets it)")
    _show(found)
    closed = archived()
    if closed:
        print("\nclosed records (archive/README.md):")
        _show(closed)


def _show(found: list[Path]) -> None:
    if not found:
        return
    print(f"{'name':<34} {'world':<22} {'started (UTC)':<20} {'local':<12} "
          f"{'ticks':<12} {'pulses':>6} {'events':>7} {'errors':>6}")
    for path in found:
        lines = read(path)
        # A row per world, not per file. Every column but the name is a
        # fact about one world -- its clock, its pulses, its failures --
        # and a run that switched worlds has two of each, not an average
        # and not whichever came first.
        for nth, world in enumerate(_worlds(lines) or [None]):
            own = _within(lines, world)
            errors = sum(1 for line in own if line.get("error"))
            pulses = sum(1 for line in own if line.get("event") == _PULSE)
            # The earliest stamp present -- prefixed ">=" when the record
            # also holds timeless lines, which the archive mostly is:
            # `when` arrived 2026-09-22, so a file written before that
            # would otherwise report the afternoon it was closed as the
            # morning it began.
            stamps = [line["when"] for line in own if line.get("when")]
            started = _clock(min(stamps)) if stamps else ""
            if started and len(stamps) < len(own):
                started = ">= " + started
            local = _local(min(stamps)) if stamps else ""
            print(f"{(path.stem if nth == 0 else ''):<34} {(world or '--')[:22]:<22} "
                  f"{started:<20} {local:<12} {_span(own):<12} {pulses:>6} {len(own):>7} {errors:>6}")


def ledger(lines: list[dict], rare_only: bool) -> None:
    """A run, pulse by pulse. One row per pulse the engine actually ran,
    in the order it ran them, with what the player typed underneath."""
    actions = {line.get("pulse_id"): line for line in lines if line.get("event") == _ACTION}
    requests = {line.get("pulse_id"): line for line in lines if line.get("event") == _REQUEST}
    calls: dict[str, list[dict]] = defaultdict(list)
    for line in lines:
        if line.get("event") == "llm.call":
            calls[line.get("pulse_id")].append(line)

    pulses = [line for line in lines if line.get("event") == _PULSE]
    if not rare_only:
        print(f"{'pulse':<10} {'tick':>5} {'':<1} {'actor':<28} {'tuples':>6} {'wrote':>5} "
              f"{'llm':>3} {'in':>7} {'out':>6} {'ms':>7}")
        for pulse in pulses:
            pid = pulse.get("pulse_id") or ""
            own = calls.get(pid, [])
            actor = str(pulse.get("actor") or "")
            # Who acted, marked in its own column rather than appended:
            # a v3 id is a readable label ("sela_pedlar_blue_pack_b21cf6")
            # and a v2 one is 32 hex characters, so any suffix on the
            # long ones was being truncated away exactly where it was
            # most needed -- the player's own rows.
            mark = " " if pulse.get("autonomous") else "*"
            took = requests.get(pid, {}).get("duration_ms") or pulse.get("duration_ms") or 0
            print(f"{pid[:8]:<10} {pulse.get('world_time', ''):>5} {mark:<1} {actor[:28]:<28} "
                  f"{pulse.get('tuples', ''):>6} {pulse.get('written', ''):>5} {len(own):>3} "
                  f"{sum(c.get('input_tokens') or 0 for c in own):>7} "
                  f"{sum(c.get('output_tokens') or 0 for c in own):>6} {took:>7}")
            said = (actions.get(pid) or {}).get("action_text")
            if said:
                print(f"           > {said[:96]}")
        print()

    # What happened TO this run, by rarity -- see this module's docstring
    # for why this is a frequency question and not a list of names.
    # Rarity is measured against the pulse -- so a run with no pulses has
    # no baseline to be ordinary against, and everything in it is what
    # happened to it. `max(pulses, 1)` was a fudge that silently made the
    # threshold "twice or more is routine" for such a run: the first live
    # record held four lines, two of them the world.loaded pair that IS
    # the world switch, and the reader hid exactly those two and showed
    # the other two. The one thing worth seeing, omitted, with no sign
    # anything had been left out.
    counts = Counter(line.get("event") for line in lines)
    ordinary = len(pulses)
    notable = sorted(
        (name for name, n in counts.items()
         # STRICTLY fewer than one per pulse, which is what the heading
         # has always claimed. `<=` let through exactly the events that
         # fire on every pulse -- llm.call above all, the most routine
         # line the engine writes -- so the list meant to hold what
         # happened TO a run led with what happens in every run.
         if name not in (_PULSE, _ACTION, _REQUEST) and (ordinary == 0 or n < ordinary)),
        key=lambda name: counts[name],
    )
    failed = sorted({line.get("event") for line in lines if line.get("error")})
    if not notable and not failed:
        return
    print(f"every event ({len(lines)} lines, no pulses):" if ordinary == 0
          else f"rarer than one per pulse ({ordinary} pulses):")
    for name in notable:
        example = next(line for line in lines if line.get("event") == name)
        shown = {k: v for k, v in example.items()
                 if k not in ("event", "when", "run", "world") and v not in (None, [], {})}
        print(f"  {counts[name]:>4}x  {name:<38} {json.dumps(shown)[:88]}")
    for name in failed:
        n = sum(1 for line in lines if line.get("event") == name and line.get("error"))
        print(f"  {n:>4}x  {name:<38} ERROR: "
              f"{next(line['error'] for line in lines if line.get('event') == name and line.get('error'))[:60]}")


def one_pulse(lines: list[dict], prefix: str) -> None:
    """Everything the witness saw of one pulse, in the order it saw it.
    The pulse_id is the correlation spine and always has been -- this
    just follows it."""
    ids = sorted({line.get("pulse_id") for line in lines
                  if isinstance(line.get("pulse_id"), str) and line["pulse_id"].startswith(prefix)})
    if not ids:
        sys.exit(f"no pulse in this run starts with {prefix!r}")
    for pid in ids:
        own = [line for line in lines if line.get("pulse_id") == pid]
        print(f"--- {pid}  ({len(own)} events)")
        for line in own:
            shown = {k: v for k, v in line.items()
                     if k not in ("event", "when", "run", "world", "pulse_id")}
            # To the millisecond here, unlike the run listing: inside one
            # pulse the question is what happened before what, and two
            # of these land in the same second regularly.
            print(f"  {_time_of(line.get('when')):<13} {line.get('event','?'):<38} "
                  f"{json.dumps(shown, default=str)[:110]}")
        print()


def main(argv: list[str]) -> None:
    rare_only = "--rare" in argv
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        list_runs()
        return
    path = resolve(args[0])
    lines = read(path)
    where = ", ".join(f"{w} ({_span(_within(lines, w))})" for w in _worlds(lines)) or "--"
    stamps = [ln["when"] for ln in lines if ln.get("when")]
    began = f"  --  began {_local(min(stamps), '%Y-%m-%d %H:%M')} local" if stamps else ""
    print(f"{path.stem}  --  {where}  --  {len(lines)} events{began}\n")
    if len(args) > 1:
        one_pulse(lines, args[1])
    else:
        ledger(lines, rare_only)


if __name__ == "__main__":
    main(sys.argv[1:])
