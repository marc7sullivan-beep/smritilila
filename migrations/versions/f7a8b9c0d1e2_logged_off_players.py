# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""the players who logged off: frozen as well, out of the world

Revision ID: f7a8b9c0d1e2
Revises: e5f6a7b8c9d0
Create Date: 2026-09-24 14:00:00.000000

Marc, 2026-09-24: "an offline non acting player should be entirely
suspended from the game, not present at all" -- and only an explicit log
off makes them so. Persisted, so a restart does not bring a logged-off
player back into a scene. Additive: every existing world starts with
nobody logged off, which is exactly the build before this.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'f7a8b9c0d1e2'
down_revision: Union[str, None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("session_config", sa.Column("logged_off_players", postgresql.JSONB(), nullable=False,
                                              server_default=sa.text("'[]'::jsonb")))


def downgrade() -> None:
    op.drop_column("session_config", "logged_off_players")
