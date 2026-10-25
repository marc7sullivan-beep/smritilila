# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""v4 §43: which players allow the world to act for them

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-09-23 18:00:00.000000

Design Principles v4 (draft) §43, decided 2026-09-23: each player chooses
whether the world may act for them -- another's turn may move them, and
while they are offline they are drawn like anyone. Off by default. A
player preference, not a world fact, so it lives on session_config with
the rest of who plays. Additive: every existing world starts with nobody
allowing it, which is exactly the build before this.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'd3e4f5a6b7c8'
down_revision: Union[str, None] = 'c2d3e4f5a6b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("session_config", sa.Column("autonomous_players", postgresql.JSONB(), nullable=False,
                                              server_default=sa.text("'[]'::jsonb")))


def downgrade() -> None:
    op.drop_column("session_config", "autonomous_players")
