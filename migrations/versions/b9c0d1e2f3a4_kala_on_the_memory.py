# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""kāla on the memory: the pulse's minutes, carried by each memory of it

Revision ID: b9c0d1e2f3a4
Revises: a8b9c0d1e2f3
Create Date: 2026-09-24 23:50:00.000000

Marc, 2026-09-24: "aren't time records just another field associated with
a memory?" A clocked pulse's time is the pulse's own: each memory written
in it carries the minutes it began and ended. People's spans, scenes'
clocks and the tick's minute are all read from memories (kala.py). The
`time` records already written stay, read for their pulses; none are
written after this. Additive: two nullable columns, and no row changes.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b9c0d1e2f3a4'
down_revision: Union[str, None] = 'a8b9c0d1e2f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("began", sa.Integer(), nullable=True))
    op.add_column("memories", sa.Column("ended", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("memories", "ended")
    op.drop_column("memories", "began")
