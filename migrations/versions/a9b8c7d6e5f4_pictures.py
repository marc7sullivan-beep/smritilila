# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""pictures: the world's pictures, kept in the world (citra, ābhāsa)

Revision ID: a9b8c7d6e5f4
Revises: e7f8a9b0c1d2
Create Date: 2026-11-11 12:00:00.000000

Marc, 2026-11-11: pictures for a new player, a being met three times, a place entered, a moment of the heaviest weight. A picture is of an
entity or of a telling, its bytes in the row (so a world's backup carries them), insert-only; a being judged to have no face is a row with
no bytes. Additive: every existing world starts with none, which is exactly the build before this.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'a9b8c7d6e5f4'
down_revision: Union[str, None] = 'e7f8a9b0c1d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pictures",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("seq", sa.BigInteger(), sa.Identity(always=False), nullable=False, unique=True),
        sa.Column("entity_id", sa.Text(), nullable=True),
        sa.Column("pulse_id", sa.Text(), nullable=True),
        sa.Column("world_time", sa.Integer(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("mime", sa.Text(), nullable=False, server_default=""),
        sa.Column("data", sa.LargeBinary(), nullable=True),
        sa.Column("size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("seeds", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("made_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("pictures_entity_id", "pictures", ["entity_id"])
    op.create_index("pictures_pulse_id", "pictures", ["pulse_id"])


def downgrade() -> None:
    op.drop_index("pictures_pulse_id", table_name="pictures")
    op.drop_index("pictures_entity_id", table_name="pictures")
    op.drop_table("pictures")
