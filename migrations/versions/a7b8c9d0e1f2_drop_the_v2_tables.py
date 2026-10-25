# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""drop the v2 tables

Revision ID: a7b8c9d0e1f2
Revises: f1a2b3c4d5e6
Create Date: 2026-09-21 00:00:00.000000

v2 is history (Marc, 2026-09-21). What v3 keeps: entities, memories,
session_config. What goes: candidates (s5's bookkeeping), delta_log and
current_relations (the graph), entity_field_state (wave propagation),
samskara (co-presence). Every v2-born world had its rooms absorbed and
its svabhava set from the graph before this runs
(scripts/v3/convert_v2_world.py, no longer in the tree); a pg_dump of each was taken then.

Also dropped: the v0.9 orphans (actors, beat_records, beliefs, episodes,
locations, promoted_details, world_objects) -- empty in every world,
flagged for exactly this follow-up when the v1.4 schema replaced them.

The downgrade recreates the five v2 tables empty -- the shapes as they
last stood -- so a dump can be restored into them; it does not bring the
rows back, and does not recreate the v0.9 orphans.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a7b8c9d0e1f2'
down_revision: Union[str, Sequence[str], None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    for table in ("current_relations", "entity_field_state", "samskara", "delta_log", "candidates",
                  "beat_records", "beliefs", "promoted_details", "world_objects", "episodes", "actors", "locations"):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


def downgrade() -> None:
    """Downgrade schema: the five tables, empty."""
    op.create_table(
        "candidates",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("handle", sa.Text(), nullable=False),
        sa.Column("mention", sa.Text(), nullable=False),
        sa.Column("first_seen_world_time", sa.Integer(), nullable=False),
        sa.Column("mention_count", sa.Integer(), nullable=False),
        sa.Column("last_mentioned_pulse_id", sa.Text(), nullable=False),
        sa.Column("discarded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("ever_significant", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_addressed_world_time", sa.Integer(), nullable=True),
    )
    op.create_table(
        "delta_log",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("global_seq", sa.BigInteger(), sa.Identity(), nullable=False, unique=True),
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
        sa.Column("signature", sa.Text(), nullable=True),
        sa.Column("magnitude", sa.Float(), nullable=False, server_default="0.2"),
        sa.Column("provisional", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
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
        "entity_field_state",
        sa.Column("entity_id", sa.Text(), sa.ForeignKey("entities.id"), primary_key=True),
        sa.Column("signature_sum", postgresql.JSONB(), nullable=False),
        sa.Column("magnitude_sum", sa.Float(), nullable=False),
    )
    op.create_table(
        "samskara",
        sa.Column("entity_a_id", sa.Text(), sa.ForeignKey("entities.id"), primary_key=True),
        sa.Column("entity_b_id", sa.Text(), sa.ForeignKey("entities.id"), primary_key=True),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("pulse_count", sa.Integer(), nullable=False),
        sa.Column("first_world_time", sa.Integer(), nullable=False),
        sa.Column("last_world_time", sa.Integer(), nullable=False),
    )
