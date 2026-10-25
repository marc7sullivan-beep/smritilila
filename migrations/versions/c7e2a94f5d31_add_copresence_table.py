# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add copresence table (memory/copresence.py interpersonal familiarity)

Revision ID: c7e2a94f5d31
Revises: b3e6f01a9c2d
Create Date: 2026-09-10 00:00:00.000000

canon.copresence was in-memory only, with nothing here to restore it
from -- a real, observed durability gap: every process restart silently
reset every character's accumulated familiarity with every other,
reintroducing exactly the "greeted as a stranger" failure the module's
own docstring names as its motivating incident (confirmed live,
2026-09-10: Amber and Jasper, ~15 real shared pulses, both narrated the
other as a stranger the first pulse after a restart). One row per pair
that has ever shared a pulse, keyed in memory/copresence.py's own
_pair_key canonical order (a <= b) so a pair is never stored both ways.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7e2a94f5d31'
down_revision: Union[str, Sequence[str], None] = 'b3e6f01a9c2d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "copresence",
        sa.Column("entity_a_id", sa.Text(), nullable=False),
        sa.Column("entity_b_id", sa.Text(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("pulse_count", sa.Integer(), nullable=False),
        sa.Column("first_world_time", sa.Integer(), nullable=False),
        sa.Column("last_world_time", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["entity_a_id"], ["entities.id"]),
        sa.ForeignKeyConstraint(["entity_b_id"], ["entities.id"]),
        sa.PrimaryKeyConstraint("entity_a_id", "entity_b_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("copresence")
