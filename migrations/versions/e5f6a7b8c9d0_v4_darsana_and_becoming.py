# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""v4 §45-46: a world's own view, and becoming

Revision ID: e5f6a7b8c9d0
Revises: d3e4f5a6b7c8
Create Date: 2026-09-23 22:00:00.000000

Design Principles v4 (draft) §45 (darśana: the ontology is the prompt's)
and §46 (becoming: what a being is can change, in every world).

records gains four kinds, all insert-only like the rest:
  clause     -- entity bears value (text) under note "<reach>:<key>";
                value NULL is the clause withdrawn
  capability -- entity is given (note "on") or loses (note "off") the
                capability named in value ("perceives", "reaches")
  said       -- the line spoken aloud in a scene's record: entity is the
                scene, value the line, note the speaker -- structured, so
                the engine writes no language into memory text
  mark       -- a fact about a memory the engine must not write into it:
                value "as_narrated" for an actor tuple stored as narrated

session_config gains the world's own view (NULL: the default view), its
library of becomings, and its world data (pacing, capability defaults).
Additive: every existing world keeps today's view and today's pacing.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = 'd3e4f5a6b7c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KINDS = "('kosha','name','action','clause','capability','said','mark')"


def upgrade() -> None:
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {KINDS}")
    op.add_column("session_config", sa.Column("darsana", sa.Text(), nullable=True))
    op.add_column("session_config", sa.Column("becomings", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")))
    op.add_column("session_config", sa.Column("world_data", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")))


def downgrade() -> None:
    op.drop_column("session_config", "world_data")
    op.drop_column("session_config", "becomings")
    op.drop_column("session_config", "darsana")
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", "kind IN ('kosha','name','action')")
