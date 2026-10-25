# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add provisional to delta_log

Revision ID: b7c1e2d3f4a5
Revises: ad360a5eb6bc
Create Date: 2026-09-15 00:00:02.000000

Design Principles v2_1, "ask, don't guess": deltalog/delta.py's
CommittedDelta.provisional marks a fact the ENGINE inferred (a default
location, place-on-entry containment, a kinship edge off a possessive,
a drift-corrected advance) as pencil, not ink -- a later Narrator ADD on
the same functional relation is rewritten into a MUTATE and wins,
instead of dying as a cardinality_violation against a guess it never
saw (Appendix D Open item 43). Server default false: every pre-existing
row reads as asserted.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7c1e2d3f4a5'
down_revision: Union[str, Sequence[str], None] = 'ad360a5eb6bc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "delta_log",
        sa.Column("provisional", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("delta_log", "provisional")
