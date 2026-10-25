# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add actor goal

Revision ID: abfc28612d9b
Revises: e3a927664c30
Create Date: 2026-08-28 21:30:58.721733

D7's migration policy in practice: additive while the schema is still
churning. A nullable column addition, applying to no existing data
differently -- llm.actor.ActorCognition needed Actor.goal (S04) to have
anything to decide from beyond raw mechanical stats.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "abfc28612d9b"
down_revision: Union[str, Sequence[str], None] = "e3a927664c30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("actors", sa.Column("goal", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("actors", "goal")
