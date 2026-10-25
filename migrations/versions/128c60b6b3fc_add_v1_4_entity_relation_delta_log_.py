# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add v1.4 entity relation delta log memories

Revision ID: 128c60b6b3fc
Revises: bf4a83a2b1de
Create Date: 2026-08-31 17:57:42.635809

Additive alongside the v0.9 tables (locations/actors/world_objects/
promoted_details/episodes/beat_records/beliefs) -- none of those are
touched. These five tables back the new graph/deltalog/memory packages
(persistence/repository_v2.py) built in the v1.4 migration; nothing in
the still-fully-working v0.9 system reads or writes them yet.

Verified offline first via `alembic upgrade head --sql`, same as every
prior migration in this project, then applied for real against the local
Postgres 17 instance set up earlier this session.

No pgvector extension or embedding column here: the Postgres SERVER
extension isn't installed on this machine (winget's vanilla Postgres 17
build doesn't include it, and building from source needs a Visual Studio
C toolchain this environment doesn't have), and no Embedder
implementation exists yet to populate one anyway. Add it in a follow-up
migration once both are actually in place.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '128c60b6b3fc'
down_revision: Union[str, Sequence[str], None] = 'bf4a83a2b1de'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "entities",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("labels", postgresql.JSONB(), nullable=False),
        sa.Column("created_at_world_time", sa.Integer(), nullable=False),
        sa.Column("origin", sa.Text(), nullable=False),
        sa.Column("due_tick", sa.Integer(), nullable=True),
        sa.Column("last_simulated_tick", sa.Integer(), nullable=True),
    )

    op.create_table(
        "candidates",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("handle", sa.Text(), nullable=False),
        sa.Column("mention", sa.Text(), nullable=False),
        sa.Column("first_seen_world_time", sa.Integer(), nullable=False),
        sa.Column("mention_count", sa.Integer(), nullable=False),
        sa.Column("last_mentioned_pulse_id", sa.Text(), nullable=True),
        sa.Column("discarded", sa.Boolean(), nullable=False),
    )

    op.create_table(
        "delta_log",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("global_seq", sa.Integer(), nullable=False, unique=True),
        sa.Column("pulse_id", sa.Text(), nullable=False),
        sa.Column("world_time", sa.Integer(), nullable=False),
        sa.Column("subject_id", sa.Text(), nullable=False),
        sa.Column("subject_kind", sa.Text(), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("target_id", sa.Text(), nullable=False),
        sa.Column("target_kind", sa.Text(), nullable=False),
        sa.Column("op", sa.Text(), nullable=False),
        sa.Column("expected_current", sa.Text(), nullable=True),
        sa.Column("anchor", sa.Text(), nullable=False),
    )
    op.create_check_constraint("delta_log_op_check", "delta_log", "op IN ('ADD','MUTATE','RETRACT')")

    op.create_table(
        "current_relations",
        sa.Column("subject_id", sa.Text(), primary_key=True),
        sa.Column("subject_kind", sa.Text(), nullable=False),
        sa.Column("relation", sa.Text(), primary_key=True),
        sa.Column("target_id", sa.Text(), primary_key=True),
        sa.Column("target_kind", sa.Text(), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("committed_delta_id", sa.Text(), sa.ForeignKey("delta_log.id"), nullable=False),
    )

    op.create_table(
        "memories",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("entity_id", sa.Text(), sa.ForeignKey("entities.id"), nullable=False),
        sa.Column("subject_ids", postgresql.JSONB(), nullable=False),
        sa.Column("experience", sa.Text(), nullable=False),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("salience", sa.Integer(), nullable=False),
        sa.Column("location_id", sa.Text(), nullable=False),
        sa.Column("world_time", sa.Integer(), nullable=False),
        sa.Column("pulse_id", sa.Text(), nullable=False),
    )
    op.create_check_constraint("memories_tier_check", "memories", "tier IN ('PRIMARY','SECONDARY','AMBIENT')")
    op.create_check_constraint("memories_salience_check", "memories", "salience BETWEEN 1 AND 7")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("memories")
    op.drop_table("current_relations")
    op.drop_table("delta_log")
    op.drop_table("candidates")
    op.drop_table("entities")
