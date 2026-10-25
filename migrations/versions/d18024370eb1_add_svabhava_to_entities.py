# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add svabhava to entities

Revision ID: d18024370eb1
Revises: c58c62bfcda2
Create Date: 2026-09-14 00:00:00.000000

Live design review, 2026-09-14 (Design Principles v2_0 Appendix D, v2.0.7):
a soft, evidence-inferred CONCLUSION about an entity's own inherent
nature -- place/actor/object -- distinct from the hard entity-type tag
S5 bans, since it's never assigned at creation and nothing branches
control flow on it (graph/svabhava.py's own module docstring). Nullable,
no server_default -- unlike ever_acted's boolean-with-default, this
field's natural absent-value IS None/NULL, not a sentinel.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd18024370eb1'
down_revision: Union[str, Sequence[str], None] = 'c58c62bfcda2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("entities", sa.Column("svabhava", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("entities", "svabhava")
