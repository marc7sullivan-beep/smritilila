# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Postgres schema, v3 (2026-09-21): entities (the id registry -- the
label IS the first tuple), memories (the log: every tuple ever written),
and session_config (who plays, where they may join, the setting, the
bhuvana); records (v4: structural facts, below) came after. The v2 tables -- candidates, delta_log, current_relations,
entity_field_state, samskara -- are dropped by the migration of the same
date; nothing here reads or writes them.

embedding is a plain JSONB float array, not a native pgvector column: the
Postgres SERVER extension isn't installed on this machine (winget's vanilla
Postgres 17 build doesn't include it). Similarity is computed in Python
(smrti/context.py's MMR), a deliberate choice at this scale.
"""

from __future__ import annotations

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EntityRow(Base):
    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    labels: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    created_at_world_time: Mapped[int] = mapped_column(Integer, nullable=False)
    origin: Mapped[str] = mapped_column(Text, nullable=False)  # one of graph.entity.Origin
    due_tick: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_simulated_tick: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_touched_tick: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ever_acted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    svabhava: Mapped[str | None] = mapped_column(Text, nullable=True)  # retired 2026-10-03: no longer on Entity; old rows keep it, unread
    avarana: Mapped[str | None] = mapped_column(Text, nullable=True)  # the scene bit (graph.entity.Avarana); v3 rows hold v3's words, read as the bit
    kosha: Mapped[str | None] = mapped_column(Text, nullable=True)  # the enclosing building's id (v3.10)


class SessionConfigRow(Base):
    """smritilila.adhisthana's WorldSession config (scene_owner_entity_id,
    entry_point_location_ids, playable_entity_ids) -- app-session
    bookkeeping, not world canon, so it lives in its own table rather than
    as a relation on the graph. A single row, id always SESSION_CONFIG_ID
    (persistence/repository.py): this
    is deliberately a single-WorldSession app (adhisthana.py's docstring),
    so there's nothing to key multiple rows by. Without this, a process
    restart on a genesis-created world (real UUID entity ids, not the
    starter world's literal "player") had no way to recover which entity
    was the player.

    playable_entity_ids: every entity a POST /pulses caller may act as
    (via its entity_id field) -- scene_owner_entity_id is a member whenever it is set,
    but not necessarily the only one once /world/join has been used to
    add a genuinely switchable second character.

    setting_description: the setting description a world was seeded
    from, NULL for a plain /genesis world -- read back when the world at
    large, an expansion or the world's one-line description is written
    from it (routes/bija.py)."""

    __tablename__ = "session_config"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    scene_owner_entity_id: Mapped[str] = mapped_column(Text, nullable=False)
    entry_point_location_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    playable_entity_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    setting_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # bhuvana (2026-09-20): the world at large, written once at birth (before
    # the seed at wide breadth, after it below, from 2026-10-05); read by every seed, expansion and newcomer call.
    bhuvana: Mapped[str | None] = mapped_column(Text, nullable=True)
    # v5 §37 (2026-09-23): the players who allow the world to act for them.
    autonomous_players: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    # Marc, 2026-09-24: the players who logged off -- while also frozen,
    # entirely out of the world until they come back.
    logged_off_players: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    waiting_players: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)  # 2026-10-01: in the world, no turn until touched
    held_back: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)  # 2026-10-02: a guest's retired, out until reviewed
    # Marc, 2026-09-30: the tether -- which character each guest plays (an avatar from outside,
    # tethered to it). Beside autonomy and logged-off; world data kept it until then.
    played_by: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # v5 §7-§8 (2026-09-23): the world's own view (NULL: the default view),
    # its library of becomings {key: text}, and its world data (pacing,
    # capability defaults). Read by the prompt and the scheduler only.
    darsana: Mapped[str | None] = mapped_column(Text, nullable=True)
    becomings: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    world_data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class MemoryRow(Base):
    """v1.4 S8-S11: immutable per-perceiver memory, insert-only (but for
    repository.absorb_entity, which re-points entity_id, location_id and
    subject_ids when one entity is merged into another). entity_id
    is a real FK -- only an Entity has a stream.

    embedding is a plain JSONB float array, not a native pgvector column:
    the Postgres SERVER extension still isn't installed on this machine
    (winget's vanilla Postgres 17 build doesn't include it, and building
    from source needs a Visual Studio C toolchain this environment
    doesn't have). Similarity is computed in Python (smrti/context.py),
    same as the fake-embedding unit tests already did -- a deliberate
    choice, not a stopgap, at this project's current memory-count scale.
    Nullable: memories written before an Embedder existed (or with none
    supplied) simply have none and aren't retrieval-eligible."""

    __tablename__ = "memories"
    __table_args__ = (
        CheckConstraint("tier IN ('PRIMARY','SECONDARY','AMBIENT')", name="memories_tier_check"),
        CheckConstraint("salience BETWEEN 1 AND 7", name="memories_salience_check"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id"), nullable=False)
    subject_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    experience: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    salience: Mapped[int] = mapped_column(Integer, nullable=False)
    location_id: Mapped[str] = mapped_column(Text, nullable=False)
    world_time: Mapped[int] = mapped_column(Integer, nullable=False)
    pulse_id: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
    use: Mapped[str | None] = mapped_column(Text, nullable=True)  # v5 §3: key | useful | none
    # v5 §19 kāla: the minutes the pulse began and ended, in a world that keeps a clock
    began: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ended: Mapped[int | None] = mapped_column(Integer, nullable=True)


class RecordRow(Base):
    """v4's structural records (memory/record.py): containment, declared
    names, a player's own words. Insert-only, like memories (but for
    repository.absorb_entity, which re-points entity_id, a kosha or sandhi
    value, a said note), and apart from them so no reader of experience
    ever meets one."""

    __tablename__ = "records"
    __table_args__ = (CheckConstraint("kind IN ('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned','prakriti','secret','follows','closes')", name="records_kind_check"),)

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    entity_id: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    world_time: Mapped[int] = mapped_column(Integer, nullable=False)
    pulse_id: Mapped[str] = mapped_column(Text, nullable=False)
