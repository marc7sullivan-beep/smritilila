# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add smritilila.field state (wave-propagation spec Phases 0-2)

Revision ID: a1c4d92f7e6b
Revises: ebbce748487a
Create Date: 2026-09-08 00:00:00.000000

Adds delta_log.signature (a committed delta's own field signature vector,
nullable -- a delta committed with no embedder available never gets one)
and delta_log.magnitude (non-nullable, defaulted -- every delta this app
constructs already carries one, see deltalog/delta.py), plus a new
entity_field_state table (one row per entity currently carrying a
nonzero field; smritilila.field.tick.teotl's own pruning deletes a
row once its magnitude_sum drops below threshold, rather than zeroing
it in place).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a1c4d92f7e6b'
down_revision: Union[str, Sequence[str], None] = 'ebbce748487a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("delta_log", sa.Column("signature", postgresql.JSONB(), nullable=True))
    op.add_column(
        "delta_log", sa.Column("magnitude", sa.Float(), nullable=False, server_default="0.2")
    )
    op.create_table(
        "entity_field_state",
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column("signature_sum", postgresql.JSONB(), nullable=False),
        sa.Column("magnitude_sum", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["entity_id"], ["entities.id"]),
        sa.PrimaryKeyConstraint("entity_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("entity_field_state")
    op.drop_column("delta_log", "magnitude")
    op.drop_column("delta_log", "signature")
