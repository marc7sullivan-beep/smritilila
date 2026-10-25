# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""rename copresence table to samskara

Revision ID: d4f8a1c39b02
Revises: c7e2a94f5d31
Create Date: 2026-09-11 00:00:00.000000

Pure rename, no schema change -- memory/copresence.py (and the module,
class, and every function in it) was renamed to memory/samskara.py per
this project's own naming methodology (a philosophically-apt name for a
significant data store, not a plain literal one): "samskara" is the
Vedantic/yogic term for a mental impression left by repeated experience,
one that strengthens into a durable disposition with reinforcement and
fades without it -- exactly what decayed_weight's own quadratic-by-
weight half-life scaling (2026-09-11) models. Preserves every existing
row (an ALTER TABLE RENAME, not a drop-and-recreate) -- this table
already held real accumulated interpersonal history by the time of the
rename and losing it would have been the exact "greeted as a stranger"
failure the table itself exists to prevent.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'd4f8a1c39b02'
down_revision: Union[str, Sequence[str], None] = 'c7e2a94f5d31'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table('copresence', 'samskara')


def downgrade() -> None:
    """Downgrade schema."""
    op.rename_table('samskara', 'copresence')
