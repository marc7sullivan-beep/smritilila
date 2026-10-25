# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add bhuvana to session_config

Revision ID: f1a2b3c4d5e6
Revises: c9d0e1f2a3b4
Create Date: 2026-09-20 00:00:00.000000

Bhuvana (भुवन, the world at large): the world's own text of its whole
-- regions and how they lie, seats, the central figures of the age and
where they are, the state of the year -- written once at birth (v3.0
§14, 2026-09-20) and read by every seed, expansion and newcomer call so
the rings grow into a shape the world already has. Session bookkeeping
like setting_description, not canon: the engine carries it unread (§0).
Nullable: worlds born before this, or at a breadth below wide, have
none.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'c9d0e1f2a3b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("session_config", sa.Column("bhuvana", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("session_config", "bhuvana")
