# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""closes: one of a perceiver's experiences closes an earlier one of theirs (head and tail); follows is retired

Revision ID: e7f8a9b0c1d2
Revises: d6e7f8a9b0c1
Create Date: 2026-10-13 19:00:00.000000

Marc, 2026-10-13 (spec v5.33): the Narrator says when a tail closes a head. A `closes` record
says, at world_time t, entity's experience at t closes entity's own experience at tick `value`;
retrieval shows a head and its tail together. `follows` is retired: its kind stays allowed
(records are insert-only, existing rows stay as scars, never read). Additive: the check gains a kind.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'e7f8a9b0c1d2'
down_revision: Union[str, None] = 'd6e7f8a9b0c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KINDS = "('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned','prakriti','secret','follows','closes')"
BEFORE = "('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned','prakriti','secret','follows')"


def upgrade() -> None:
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {KINDS}")


def downgrade() -> None:
    op.execute("DELETE FROM records WHERE kind = 'closes'")
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {BEFORE}")
