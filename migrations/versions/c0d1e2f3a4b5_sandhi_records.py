# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""sandhi: a boundary thing is in both scenes it joins

Revision ID: c0d1e2f3a4b5
Revises: b9c0d1e2f3a4
Create Date: 2026-09-25 17:00:00.000000

Marc, 2026-09-25 ("aren't opening and containment very closely related?";
the name sandhi, his): a window, a door, a gate is IN both scenes it joins.
Its first holder is its containment; a `sandhi` record gives it the other
side (value: that scene; note: "open" or "closed"). Additive: the check
gains a kind; no row changes, and a world whose Narrator declares no
boundaries writes none.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'c0d1e2f3a4b5'
down_revision: Union[str, None] = 'b9c0d1e2f3a4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KINDS = "('kosha','name','action','clause','capability','said','mark','time','sandhi')"
BEFORE = "('kosha','name','action','clause','capability','said','mark','time')"


def upgrade() -> None:
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {KINDS}")


def downgrade() -> None:
    op.execute("DELETE FROM records WHERE kind = 'sandhi'")
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {BEFORE}")
