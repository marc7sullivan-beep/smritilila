# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""the players who wait: in the world, taking no turn of their own until touched

Revision ID: a3b4c5d6e7f8
Revises: e2f3a4b5c6d7
Create Date: 2026-10-01 23:00:00.000000

Marc, 2026-10-01: "keep teo active, just don't take any turns until someone
interacts with him". Additive: every existing world starts with nobody
waiting, which is exactly the build before this.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'a3b4c5d6e7f8'
down_revision: Union[str, None] = 'e2f3a4b5c6d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("session_config", sa.Column("waiting_players", postgresql.JSONB(), nullable=False,
                                              server_default=sa.text("'[]'::jsonb")))


def downgrade() -> None:
    op.drop_column("session_config", "waiting_players")
