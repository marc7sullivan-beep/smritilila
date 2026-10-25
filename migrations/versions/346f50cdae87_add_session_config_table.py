# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add session config table

Revision ID: 346f50cdae87
Revises: 4b96f61f53c2
Create Date: 2026-09-02 07:36:19.744862

The persistence side of the genesis-reload gap fix: smritilila.main.WorldSession's
own bookkeeping (scene_owner_entity_id, entry_point_location_ids) was never
persisted, so a process restart on a genesis-created world (real UUID entity
ids, not the starter world's literal "player") had no way to recover which
entity was the player. A single row (id always "singleton" -- this is
deliberately a single-WorldSession app), not part of the world Entity/
Relation graph itself.

Verified offline via `alembic upgrade head --sql` first, same as every prior
migration in this project, then applied for real against the local
Postgres 17 instance.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '346f50cdae87'
down_revision: Union[str, Sequence[str], None] = '4b96f61f53c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "session_config",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("scene_owner_entity_id", sa.Text(), nullable=False),
        sa.Column("entry_point_location_ids", postgresql.JSONB(), nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("session_config")
