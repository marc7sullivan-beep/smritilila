# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Which characters are bots (Marc, 2026-10-06): the beings of each world the overseer's screen has switched on to be played by an
agent whenever a person is active in that world. Kept in bots.json beside worlds.json -- outside every world, so no world's memory
is written (a world's memories are append-only).

    {"bots":  {world name: [active character ids]},
     "known": {world name: [every id ever switched on]},
     "hold":  {world name: [ids]},               the active bots that are only HELD: kept present and never acted for (2026-10-17)
     "free":  {world name: [ids]},               the active bots FREE of the time guard, and started with no person in the world: a sim of bots alone (2026-10-22)
     "alive": {world name: {id: true|false}},    what a model read of each candidate and judged, kept so it is asked once
     "important": {world name: {id: true|false}}}   the same, of who matters to the world's story (the dragon, the high priest)

An NPC switched on is made playable in its world, so it must stay on the overseer's list when it is switched off, though a playable
being is otherwise no NPC (the list is of NPCs, not of the people's characters). The supervisor (scripts/bots_supervisor.py) reads
the active ones."""
from __future__ import annotations

import json
import os
from pathlib import Path

from smritilila.paths import data_dir

_DEFAULT_PATH = data_dir() / "bots.json"


def _path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else Path(os.environ.get("SMRITILILA_BOTS_CONFIG", _DEFAULT_PATH))


def _read(path: str | Path | None, key: str) -> dict[str, list[str]]:
    try:
        data = json.loads(_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    got = data.get(key, {}) if isinstance(data, dict) else {}
    return {str(w): [str(i) for i in ids] for w, ids in got.items() if isinstance(ids, list) and ids}


def load_bots(path: str | Path | None = None) -> dict[str, list[str]]:
    """The active bots by world."""
    return _read(path, "bots")


def load_hold(path: str | Path | None = None) -> dict[str, list[str]]:
    """The active bots that are held: the supervisor keeps their character online while a person is active, and nothing acts for them
    (Marc, 2026-10-17: "bot mode but not taking actions", so that he is seen in the players-online line)."""
    return _read(path, "hold")


def load_free(path: str | Path | None = None) -> dict[str, list[str]]:
    """The active bots that are free (Marc, 2026-10-22: "build the time guard defaulted to never move ahead of the player but make it switchable. Sometimes we want
    to run a bot sim with no players like in the crossing"): the guard in scripts/agents/player_tool.py is off for them, and the supervisor starts them whether or
    not a person is in the world and does not stop them when the last person goes. Every other bot never moves ahead of a person."""
    return _read(path, "free")


def load_known(path: str | Path | None = None) -> dict[str, list[str]]:
    """Every being ever switched on as a bot, by world, active or not."""
    return _read(path, "known")


def _judged(path: str | Path | None, key: str) -> dict[str, dict[str, bool]]:
    try:
        data = json.loads(_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    got = data.get(key, {}) if isinstance(data, dict) else {}
    return {str(w): {str(i): bool(v) for i, v in d.items()} for w, d in got.items() if isinstance(d, dict)}


def load_alive(path: str | Path | None = None) -> dict[str, dict[str, bool]]:
    return _judged(path, "alive")


def load_important(path: str | Path | None = None) -> dict[str, dict[str, bool]]:
    return _judged(path, "important")


def _write(path, bots, known, alive, important, hold=None, free=None) -> None:
    hold = load_hold(path) if hold is None else hold
    hold = {w: [i for i in ids if i in bots.get(w, [])] for w, ids in hold.items()}   # only an active bot can be held
    free = load_free(path) if free is None else free
    free = {w: [i for i in ids if i in bots.get(w, [])] for w, ids in free.items()}   # nor be free
    data = {"bots": bots, "known": known, "alive": alive, "important": important}
    if any(hold.values()):
        data["hold"] = {w: ids for w, ids in hold.items() if ids}
    if any(free.values()):
        data["free"] = {w: ids for w, ids in free.items() if ids}
    _path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def set_alive(world: str, judged: dict[str, bool], path: str | Path | None = None) -> None:
    """Keep what was judged of a world's beings (true: alive)."""
    alive = load_alive(path)
    alive[world] = {**alive.get(world, {}), **judged}
    _write(path, load_bots(path), load_known(path), alive, load_important(path))


def set_important(world: str, judged: dict[str, bool], path: str | Path | None = None) -> None:
    """Keep what was judged of who matters to a world's story (true: important)."""
    important = load_important(path)
    important[world] = {**important.get(world, {}), **judged}
    _write(path, load_bots(path), load_known(path), load_alive(path), important)


def set_hold(world: str, entity_id: str, held: bool, path: str | Path | None = None) -> list[str]:
    """Hold an active bot (present, never acted for) or release it to be played; the world's held list after."""
    hold = load_hold(path)
    hold[world] = [i for i in hold.get(world, []) if i != entity_id] + ([entity_id] if held else [])
    _write(path, load_bots(path), load_known(path), load_alive(path), load_important(path), hold)
    return load_hold(path).get(world, [])


def set_free(world: str, entity_id: str, free: bool, path: str | Path | None = None) -> list[str]:
    """Free an active bot of the time guard (and start it with no person in the world) or put the guard back; the world's free list after."""
    held = load_free(path)
    held[world] = [i for i in held.get(world, []) if i != entity_id] + ([entity_id] if free else [])
    _write(path, load_bots(path), load_known(path), load_alive(path), load_important(path), free=held)
    return load_free(path).get(world, [])


def set_bot(world: str, entity_id: str, active: bool, path: str | Path | None = None) -> list[str]:
    """Switch a being on or off as a bot; the world's active list after."""
    bots, known, alive, important = load_bots(path), load_known(path), load_alive(path), load_important(path)
    ids = [i for i in bots.get(world, []) if i != entity_id]
    if active:
        ids.append(entity_id)
        known[world] = [i for i in known.get(world, []) if i != entity_id] + [entity_id]
    if ids:
        bots[world] = ids
    else:
        bots.pop(world, None)
    _write(path, bots, known, alive, important)
    return ids
