# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add last_addressed_world_time to candidates

Revision ID: ad360a5eb6bc
Revises: d18024370eb1
Create Date: 2026-09-14 00:00:01.000000

Live investigation, 2026-09-14: graph/candidate.py's own
Candidate.last_addressed_world_time (the field behind this session's own
extra-participant crash fix, pulse/spanda.py's own
_directly_addressed_candidate_ids) was added to the dataclass but never
wired into persistence/repository.py's Candidate<->row mapping -- a
real, live gap: without this column the field silently reset to None on
every process restart or canon reload from Postgres, quietly
reintroducing the exact crash it was built to fix. Nullable, no default
-- matches the field's own "None only before its own first mint" shape.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ad360a5eb6bc'
down_revision: Union[str, Sequence[str], None] = 'd18024370eb1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("candidates", sa.Column("last_addressed_world_time", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("candidates", "last_addressed_world_time")
