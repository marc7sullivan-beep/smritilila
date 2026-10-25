# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add ever_significant to candidates

Revision ID: ebbce748487a
Revises: 91fda35e85b2
Create Date: 2026-09-06 00:00:00.000000

Whether ANY mention of this candidate was ever judged significant --
distinct from mention_count > 0, which is trivially true for every
candidate from the moment it's minted regardless of that founding
mention's own significance. Gates pulse/pipeline.py's own world_state
surfacing of not-yet-promoted candidates as passive facts, without
giving them a perceiver stream. Added NOT NULL with a default of false:
an already-persisted candidate has no historical record of whether an
earlier mention was significant, and false (invisible until next
mentioned) is the safe default, same spirit as this codebase's other
Boolean columns (discarded).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ebbce748487a'
down_revision: Union[str, Sequence[str], None] = '91fda35e85b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("candidates", sa.Column("ever_significant", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("candidates", "ever_significant")
