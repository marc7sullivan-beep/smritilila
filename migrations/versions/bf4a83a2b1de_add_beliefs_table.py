# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add beliefs table

Revision ID: bf4a83a2b1de
Revises: abfc28612d9b
Create Date: 2026-08-29 05:53:46.255725

D7's migration policy in practice: additive. A new table, no existing
data touched -- resolver.belief.Belief (S13/S14) now exists as a domain
type, closing the gap the initial migration's docstring flagged.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bf4a83a2b1de"
down_revision: Union[str, Sequence[str], None] = "abfc28612d9b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "beliefs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("holder_id", sa.Text(), sa.ForeignKey("actors.id"), nullable=False),
        sa.Column("subject_id", sa.Text(), sa.ForeignKey("actors.id"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("consolidation_episode_id", sa.Text(), sa.ForeignKey("episodes.id"), nullable=False),
        sa.UniqueConstraint("holder_id", "subject_id", "version", name="beliefs_holder_subject_version_uq"),
    )


def downgrade() -> None:
    op.drop_table("beliefs")
