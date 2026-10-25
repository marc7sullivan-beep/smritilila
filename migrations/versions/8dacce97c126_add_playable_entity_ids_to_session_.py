# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""add playable_entity_ids to session config

Revision ID: 8dacce97c126
Revises: 346f50cdae87
Create Date: 2026-09-02 15:21:22.034760

The persistence side of multi-character support: session_config previously
tracked a single scene_owner_entity_id (the one entity a POST /pulses caller
could ever act as). playable_entity_ids widens that to every entity a caller
may act as via the request's new entity_id field -- always includes
scene_owner_entity_id, but /world/join can now add more.

Added nullable first, backfilled from the existing row's own
scene_owner_entity_id (a single-element array -- the only playable entity any
already-persisted world could have had before this migration), then made
NOT NULL -- not added NOT NULL outright, since that would break against the
one row this table already has on a live database.

Verified offline via `alembic upgrade head --sql` first, same as every prior
migration in this project.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '8dacce97c126'
down_revision: Union[str, Sequence[str], None] = '346f50cdae87'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("session_config", sa.Column("playable_entity_ids", postgresql.JSONB(), nullable=True))
    op.execute("UPDATE session_config SET playable_entity_ids = jsonb_build_array(scene_owner_entity_id)")
    op.alter_column("session_config", "playable_entity_ids", nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("session_config", "playable_entity_ids")
