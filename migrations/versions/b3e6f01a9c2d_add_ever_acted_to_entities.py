# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add ever_acted to entities

Revision ID: b3e6f01a9c2d
Revises: a1c4d92f7e6b
Create Date: 2026-09-08 00:00:00.000000

Live investigation, 2026-09-08: mirrors candidates.ever_significant's own
shape, one behavioral level up -- has this entity, when given its own
autonomous activation episode, ever committed a delta about itself beyond
the purely structural (loc/description/blocking)? Nullable-free (default
False), same as ever_significant -- an already-persisted entity simply
hasn't proven anything yet.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3e6f01a9c2d'
down_revision: Union[str, Sequence[str], None] = 'a1c4d92f7e6b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("entities", sa.Column("ever_acted", sa.Boolean(), nullable=False, server_default="false"))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("entities", "ever_acted")
