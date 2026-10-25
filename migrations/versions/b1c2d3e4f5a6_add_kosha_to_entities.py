# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add kosha to entities

Revision ID: b1c2d3e4f5a6
Revises: a7b8c9d0e1f2
Create Date: 2026-09-23 00:00:00.000000

Kośa, the sheath (Design Principles v3.10): the building that encloses a
place, as the Narrator stated it at mint. Nullable -- every place minted
before today has none, and a place enclosed by nothing has none forever.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, Sequence[str], None] = 'a7b8c9d0e1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("entities", sa.Column("kosha", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("entities", "kosha")
