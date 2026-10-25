# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add embedding column to memories

Revision ID: 4b96f61f53c2
Revises: 128c60b6b3fc
Create Date: 2026-08-31 19:32:33.352873

Plain JSONB float array, not a native pgvector column: the Postgres
SERVER extension still isn't installed on this machine (winget's vanilla
Postgres 17 build doesn't include it, and building from source needs a
Visual Studio C toolchain this environment doesn't have). Similarity is
computed in Python (memory/retrieval.py) at this project's current
memory-count scale -- a deliberate choice, not a stopgap.

Nullable and additive: every existing memories row gets NULL, meaning
"written before an Embedder existed" -- consistent with
memory/memory.py's own Memory.embedding default of None.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '4b96f61f53c2'
down_revision: Union[str, Sequence[str], None] = '128c60b6b3fc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("memories", sa.Column("embedding", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("memories", "embedding")
