# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add avarana to entities

Revision ID: c9d0e1f2a3b4
Revises: b7c1e2d3f4a5
Create Date: 2026-09-17

Design Principles v2.12 (Open item 48): a place's enclosure --
open | building | district -- assigned once at mint and never re-derived
from what later stands in it. Nullable: a non-place carries none, and a
place minted before v2.12 carries none until tools/migrate_avarana.py
(no longer in the tree) assigns it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c9d0e1f2a3b4'
down_revision: Union[str, Sequence[str], None] = 'b7c1e2d3f4a5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("entities", sa.Column("avarana", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("entities", "avarana")
