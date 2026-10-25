# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add last_touched_tick to entities

Revision ID: 91fda35e85b2
Revises: 8dacce97c126
Create Date: 2026-09-06 00:00:00.000000

Change Spec Phase 0 S1.2: set to the current tick on commit of any
delta naming the entity as either endpoint, and on any memory write for
it. Nullable, same as due_tick/last_simulated_tick before it -- an
already-persisted entity has no historical value to backfill. Read by
nothing until Phase 2.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '91fda35e85b2'
down_revision: Union[str, Sequence[str], None] = '8dacce97c126'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("entities", sa.Column("last_touched_tick", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("entities", "last_touched_tick")
