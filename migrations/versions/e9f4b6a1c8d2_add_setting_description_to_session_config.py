# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add setting_description to session_config

Revision ID: e9f4b6a1c8d2
Revises: d4f8a1c39b02
Create Date: 2026-09-12 00:00:00.000000

The exact setting description a World Builder world was seeded from
(main.py's POST /world/seed), NULL for a /world/reset or plain /genesis
world. Session-config bookkeeping (SessionConfigRow's own docstring),
not world canon -- read back every pulse to build the Narrator's own
WORLD SETTING prompt section (llm/narrator.py) instead of leaving it to
decay out of recency/similarity-ranked memory context the way a genesis
self-description alone does.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9f4b6a1c8d2'
down_revision: Union[str, Sequence[str], None] = 'd4f8a1c39b02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("session_config", sa.Column("setting_description", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("session_config", "setting_description")
