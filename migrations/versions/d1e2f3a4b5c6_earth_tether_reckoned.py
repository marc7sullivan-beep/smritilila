# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""the tether's home and the world's own history of reckoning

Revision ID: d1e2f3a4b5c6
Revises: c0d1e2f3a4b5
Create Date: 2026-09-30 19:00:00.000000

Marc, 2026-09-30: world data is the world's own view, authored; players are
avatars from outside, tethered to it. Two things the engine kept inside the
view move out of it:

- who plays which character (world data "played_by") is the tether's --
  a `played_by` column beside `autonomous_players` and `logged_off_players`;
- each doer's last reckoning (world data "phala_last") is the world's own
  history -- a `reckoned` record per doer, value the tick reckoned to.

Insert-only for the history (ids derived from what they record, so a rerun
writes nothing new); the view loses the two keys it should never have held.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision: str = 'd1e2f3a4b5c6'
down_revision: Union[str, None] = 'c0d1e2f3a4b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KINDS = "('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned')"
BEFORE = "('kosha','name','action','clause','capability','said','mark','time','sandhi')"


def upgrade() -> None:
    op.add_column("session_config", sa.Column("played_by", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")))
    op.execute("UPDATE session_config SET played_by = world_data->'played_by', world_data = world_data - 'played_by' "
               "WHERE world_data ? 'played_by'")
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {KINDS}")
    op.execute(
        "INSERT INTO records (id, kind, entity_id, value, world_time, pulse_id, note) "
        "SELECT md5('reckoned:' || m.key || ':' || (m.value #>> '{}')), 'reckoned', m.key, m.value #>> '{}', "
        "(m.value #>> '{}')::int, 'phala_last', 'moved from world data' "
        "FROM session_config s, jsonb_each(s.world_data->'phala_last') m "
        "WHERE m.key IN (SELECT id FROM entities) "
        "AND NOT EXISTS (SELECT 1 FROM records r WHERE r.id = md5('reckoned:' || m.key || ':' || (m.value #>> '{}')))")
    op.execute("UPDATE session_config SET world_data = world_data - 'phala_last' WHERE world_data ? 'phala_last'")


def downgrade() -> None:
    op.execute(
        "UPDATE session_config SET world_data = world_data || jsonb_build_object('phala_last', "
        "(SELECT coalesce(jsonb_object_agg(entity_id, value::int), '{}'::jsonb) FROM "
        "(SELECT DISTINCT ON (entity_id) entity_id, value FROM records WHERE kind = 'reckoned' "
        "ORDER BY entity_id, world_time DESC) latest))")
    op.execute("DELETE FROM records WHERE kind = 'reckoned'")
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {BEFORE}")
    op.execute("UPDATE session_config SET world_data = world_data || jsonb_build_object('played_by', played_by) "
               "WHERE played_by <> '{}'::jsonb")
    op.drop_column("session_config", "played_by")
