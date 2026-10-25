# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Persistence gateway for the HTTP API's world session (smritilila.adhisthana), v3
(2026-09-21): a world is entities, memories, records (v4) and one session row, and a
pulse commits the tuples and records it wrote and the entities it minted or touched
-- one transaction. The v2 delta-append store (deltas, relations,
candidates, field state, samskara) is history.

Two implementations:
  - InMemoryWorldStore: never touches a database. load() always reports
    nothing persisted; commit() is a no-op. Default whenever DATABASE_URL
    isn't set.
  - PostgresWorldStore: loads/commits via smritilila.persistence.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from smritilila.graph.canon import Maya
from smritilila.graph.entity import Entity
from smritilila.memory.memory import Memory
from smritilila.memory.record import Record
from smritilila.persistence.engine import make_engine, make_session_factory
from smritilila.persistence.repository import (
    absorb_entity,
    clear_all,
    load_all_memories,
    load_all_records,
    load_autonomy,
    load_logged_off,
    load_played_by,
    load_waiting,
    load_held_back,
    load_view,
    load_bhuvana,
    load_canon,
    load_last_memory,
    load_session_config,
    save_autonomy,
    save_logged_off,
    save_played_by,
    save_view,
    save_bhuvana,
    save_memories,
    save_records,
    save_session_config,
    save_waiting,
    save_held_back,
    upsert_entities,
)


class WorldStore(Protocol):
    async def load(self) -> Maya | None:
        """None means nothing persisted yet -- the caller has an empty
        world waiting on /genesis."""
        ...

    async def commit(self, canon: Maya, memories: Sequence[Memory], entities: Sequence[Entity], touched_ids: Sequence[str] = (), records: Sequence[Record] = ()) -> None:
        """One pulse's writes: the tuples and records appended, the entities minted or
        relabeled (upserted), and every entity in touched_ids upserted from
        the canon so the scheduler's bookkeeping (last_simulated_tick)
        persists. One transaction."""
        ...

    async def clear(self) -> None:
        """Wipes all persisted world state -- the only way to get back to
        an empty world, since load() would otherwise keep reloading
        whatever world already exists."""
        ...

    async def save_session_config(
        self,
        scene_owner_entity_id: str,
        entry_point_location_ids: list[str],
        playable_entity_ids: list[str],
        setting_description: str | None = None,
    ) -> None:
        """WorldSession's own bookkeeping (who plays, where a newcomer may
        land, the setting) -- persisted so a restart recovers it."""
        ...

    async def load_session_config(self) -> tuple[str, list[str], list[str], str | None] | None:
        """None means nothing was ever saved."""
        ...

    async def save_bhuvana(self, text: str | None) -> None:
        """The world at large (2026-09-20): written once at birth."""
        ...

    async def absorb(self, keep: str, gone: str) -> dict[str, int]:
        """s10: `gone` was `keep` all along. Re-files every tuple, drops
        the id; the alias (a name record) is the caller's."""
        ...

    async def load_bhuvana(self) -> str | None:
        ...

    async def save_autonomy(self, player_ids: list[str]) -> None:
        """v5 §37: the players who allow the world to act for them."""
        ...

    async def save_view(self, darsana: str | None, becomings: dict, world_data: dict) -> None:
        """v5 §7-§8: the world's own view, library of becomings, world data."""
        ...

    async def load_view(self) -> tuple[str | None, dict, dict]:
        ...

    async def load_autonomy(self) -> list[str]:
        ...

    async def save_logged_off(self, player_ids: list[str]) -> None:
        """The players who logged off (Marc, 2026-09-24): frozen as well,
        they are out of the world until they come back."""
        ...

    async def load_logged_off(self) -> list[str]:
        ...

    async def save_waiting(self, player_ids: list[str]) -> None:
        """The players who wait (Marc, 2026-10-01): in the world, taking no
        turn of their own until someone touches them."""
        ...

    async def load_waiting(self) -> list[str]:
        ...

    async def save_held_back(self, ids: list[str]) -> None:
        """A guest's retired characters, out of the world until a god reviews them (Marc, 2026-10-02)."""
        ...

    async def load_held_back(self) -> list[str]:
        ...

    async def save_played_by(self, played_by: dict[str, str]) -> None:
        """The tether (Marc, 2026-09-30): which character each guest plays.
        Its name: tantu (तन्तु), a thread (v5 §15)."""
        ...

    async def load_played_by(self) -> dict[str, str]:
        ...

    async def last_memory_for(self, entity_id: str) -> Memory | None:
        """The single most recent memory an entity has of its own
        experience, from the durable record -- None if it never perceived
        anything, or doesn't exist."""
        ...

    async def load_memories(self) -> list[Memory]:
        """Every persisted memory, oldest first -- the durable history
        WorldSession.recent_memories is seeded from on load."""
        ...

    async def load_records(self) -> list[Record]:
        """v4: every containment, name and action record, oldest first."""
        ...


class InMemoryWorldStore:
    def __init__(self) -> None:
        self._bhuvana: str | None = None

    async def load(self) -> Maya | None:
        return None

    async def commit(self, canon: Maya, memories: Sequence[Memory], entities: Sequence[Entity], touched_ids: Sequence[str] = (), records: Sequence[Record] = ()) -> None:
        return None

    async def clear(self) -> None:
        return None

    async def save_session_config(
        self,
        scene_owner_entity_id: str,
        entry_point_location_ids: list[str],
        playable_entity_ids: list[str],
        setting_description: str | None = None,
    ) -> None:
        return None

    async def load_session_config(self) -> tuple[str, list[str], list[str], str | None] | None:
        return None

    async def save_bhuvana(self, text: str | None) -> None:
        self._bhuvana = text

    async def absorb(self, keep: str, gone: str) -> dict[str, int]:
        return {"tuples": 0, "filed": 0, "presence": 0, "dropped": 0}

    async def load_bhuvana(self) -> str | None:
        return self._bhuvana

    async def save_autonomy(self, player_ids: list[str]) -> None:
        self._autonomy = sorted(set(player_ids))

    async def load_autonomy(self) -> list[str]:
        return list(getattr(self, "_autonomy", []))

    async def save_logged_off(self, player_ids: list[str]) -> None:
        self._logged_off = sorted(set(player_ids))

    async def load_logged_off(self) -> list[str]:
        return list(getattr(self, "_logged_off", []))

    async def save_waiting(self, player_ids: list[str]) -> None:
        self._waiting = sorted(set(player_ids))

    async def load_waiting(self) -> list[str]:
        return list(getattr(self, "_waiting", []))

    async def save_held_back(self, ids: list[str]) -> None:
        self._held_back = sorted(set(ids))

    async def load_held_back(self) -> list[str]:
        return list(getattr(self, "_held_back", []))

    async def save_played_by(self, played_by: dict[str, str]) -> None:
        self._played_by = dict(played_by)

    async def load_played_by(self) -> dict[str, str]:
        return dict(getattr(self, "_played_by", {}))

    async def save_view(self, darsana: str | None, becomings: dict, world_data: dict) -> None:
        self._view = (darsana, dict(becomings), dict(world_data))

    async def load_view(self) -> tuple[str | None, dict, dict]:
        return getattr(self, "_view", (None, {}, {}))

    async def last_memory_for(self, entity_id: str) -> Memory | None:
        return None

    async def load_memories(self) -> list[Memory]:
        return []

    async def load_records(self) -> list[Record]:
        return []


class PostgresWorldStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load(self) -> Maya | None:
        async with self._session_factory() as session:
            canon = await load_canon(session)
            if not canon.entities:
                return None  # nothing persisted yet -- a fresh database
            return canon

    async def commit(self, canon: Maya, memories: Sequence[Memory], entities: Sequence[Entity], touched_ids: Sequence[str] = (), records: Sequence[Record] = ()) -> None:
        async with self._session_factory() as session:
            written = {e.id for e in entities}
            touched = [canon.entities[eid] for eid in set(touched_ids) if eid in canon.entities and eid not in written]
            await upsert_entities(session, list(entities) + touched)
            await save_memories(session, list(memories))
            await save_records(session, list(records))
            await session.commit()

    async def clear(self) -> None:
        async with self._session_factory() as session:
            await clear_all(session)
            await session.commit()

    async def save_session_config(
        self,
        scene_owner_entity_id: str,
        entry_point_location_ids: list[str],
        playable_entity_ids: list[str],
        setting_description: str | None = None,
    ) -> None:
        async with self._session_factory() as session:
            await save_session_config(
                session, scene_owner_entity_id, entry_point_location_ids, playable_entity_ids, setting_description
            )
            await session.commit()

    async def load_session_config(self) -> tuple[str, list[str], list[str], str | None] | None:
        async with self._session_factory() as session:
            return await load_session_config(session)

    async def save_bhuvana(self, text: str | None) -> None:
        async with self._session_factory() as session:
            await save_bhuvana(session, text)
            await session.commit()

    async def absorb(self, keep: str, gone: str) -> dict[str, int]:
        async with self._session_factory() as session:
            out = await absorb_entity(session, keep, gone)
            await session.commit()
            return out

    async def load_bhuvana(self) -> str | None:
        async with self._session_factory() as session:
            return await load_bhuvana(session)

    async def save_autonomy(self, player_ids: list[str]) -> None:
        async with self._session_factory() as session:
            await save_autonomy(session, player_ids)
            await session.commit()

    async def load_autonomy(self) -> list[str]:
        async with self._session_factory() as session:
            return await load_autonomy(session)

    async def save_logged_off(self, player_ids: list[str]) -> None:
        async with self._session_factory() as session:
            await save_logged_off(session, player_ids)
            await session.commit()

    async def load_logged_off(self) -> list[str]:
        async with self._session_factory() as session:
            return await load_logged_off(session)

    async def save_waiting(self, player_ids: list[str]) -> None:
        async with self._session_factory() as session:
            await save_waiting(session, player_ids)
            await session.commit()

    async def load_waiting(self) -> list[str]:
        async with self._session_factory() as session:
            return await load_waiting(session)

    async def save_held_back(self, ids: list[str]) -> None:
        async with self._session_factory() as session:
            await save_held_back(session, ids)
            await session.commit()

    async def load_held_back(self) -> list[str]:
        async with self._session_factory() as session:
            return await load_held_back(session)

    async def save_played_by(self, played_by: dict[str, str]) -> None:
        async with self._session_factory() as session:
            await save_played_by(session, played_by)
            await session.commit()

    async def load_played_by(self) -> dict[str, str]:
        async with self._session_factory() as session:
            return await load_played_by(session)

    async def save_view(self, darsana: str | None, becomings: dict, world_data: dict) -> None:
        async with self._session_factory() as session:
            await save_view(session, darsana, becomings, world_data)
            await session.commit()

    async def load_view(self) -> tuple[str | None, dict, dict]:
        async with self._session_factory() as session:
            return await load_view(session)

    async def last_memory_for(self, entity_id: str) -> Memory | None:
        async with self._session_factory() as session:
            return await load_last_memory(session, entity_id)

    async def load_memories(self) -> list[Memory]:
        async with self._session_factory() as session:
            return await load_all_memories(session)

    async def load_records(self) -> list[Record]:
        async with self._session_factory() as session:
            return await load_all_records(session)


_engines_by_url: dict[str, AsyncEngine] = {}


def make_world_store(database_url: str | None = None) -> WorldStore:
    """Keyed by URL, not a single cached engine -- a bare global would
    silently keep serving the FIRST url it was ever called with, ignoring
    every later one (a real bug once world selection, smritilila.worlds_registry,
    made calling this with a DIFFERENT url an expected, live thing to do).
    One AsyncEngine (and its own connection pool) per distinct database,
    reused across calls."""
    url = database_url or os.environ.get("DATABASE_URL")
    if url is None:
        return InMemoryWorldStore()
    engine = _engines_by_url.get(url)
    if engine is None:
        engine = make_engine(url)
        _engines_by_url[url] = engine
    return PostgresWorldStore(make_session_factory(engine))
