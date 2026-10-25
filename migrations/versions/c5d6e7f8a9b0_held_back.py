# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""held back: a guest's retired character, out of the world until a god reviews it

Revision ID: c5d6e7f8a9b0
Revises: b4c5d6e7f8a9
Create Date: 2026-10-02 22:00:00.000000

Marc, 2026-10-02: "when a player retires a character they should be held in back
until we review. I don't want guests just releasing anything willy nilly. A
regular character is fine, creating gold dragons is questionable." Additive:
every existing world starts with nobody held back, which is the build before this.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'c5d6e7f8a9b0'
down_revision: Union[str, None] = 'b4c5d6e7f8a9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("session_config", sa.Column("held_back", postgresql.JSONB(), nullable=False,
                                              server_default=sa.text("'[]'::jsonb")))


def downgrade() -> None:
    op.drop_column("session_config", "held_back")
