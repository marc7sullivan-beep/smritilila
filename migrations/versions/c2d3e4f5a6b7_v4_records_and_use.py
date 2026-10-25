# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""v4: the records table, and use on memories

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-09-23 12:00:00.000000

Design Principles v4 (draft, 2026-09-23). `records` holds the facts that
are not anyone's experience: containment (§23, kośa), declared names
(§33), and a player's own words for a pulse (§38-39) -- insert-only.
`memories.use` is the second of a tuple's two words (§35): what the
moment gives its perceiver. Nullable: every tuple before v4 has none.
Nothing existing is rewritten.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c2d3e4f5a6b7'
down_revision: Union[str, Sequence[str], None] = 'b1c2d3e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("use", sa.Text(), nullable=True))
    op.create_table(
        "records",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("world_time", sa.Integer(), nullable=False),
        sa.Column("pulse_id", sa.Text(), nullable=False),
        sa.CheckConstraint("kind IN ('kosha','name','action')", name="records_kind_check"),
    )
    op.create_index("records_entity_idx", "records", ["entity_id"])


def downgrade() -> None:
    op.drop_index("records_entity_idx", table_name="records")
    op.drop_table("records")
    op.drop_column("memories", "use")
