# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""follows: one of a perceiver's experiences follows from an earlier one of theirs

Revision ID: d6e7f8a9b0c1
Revises: c5d6e7f8a9b0
Create Date: 2026-10-10 20:00:00.000000

Marc and Anu, 2026-10-10 (spec v5.30): the Narrator writes what a moment
continues, answers or ends. A `follows` record says, at world_time t,
entity's experience at t follows from entity's own experience at tick `value`;
retrieval shows a cause and what came of it together. Additive: the check
gains a kind; no row changes.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'd6e7f8a9b0c1'
down_revision: Union[str, None] = 'c5d6e7f8a9b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KINDS = "('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned','prakriti','secret','follows')"
BEFORE = "('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned','prakriti','secret')"


def upgrade() -> None:
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {KINDS}")


def downgrade() -> None:
    op.execute("DELETE FROM records WHERE kind = 'follows'")
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {BEFORE}")
