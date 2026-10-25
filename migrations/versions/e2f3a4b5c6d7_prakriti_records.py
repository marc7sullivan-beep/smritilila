# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""prakriti: the world's nature, written onto the earth

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-10-01 12:00:00.000000

Marc and Anu, 2026-10-01 (spec v4.53, §50): the seed lies at the root and
grows through the branches. A world's structured givens -- pacing, fortune,
clock, reckoning, source, ... -- are `prakriti` records (note: the key,
value: its JSON) on the earth or on a branch, read from the tree, nearest
that says. Additive: the check gains a kind; no row changes. The seed is
written onto each world's earth by scripts/v4/sow_seed.py, insert-only.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'e2f3a4b5c6d7'
down_revision: Union[str, None] = 'd1e2f3a4b5c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KINDS = "('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned','prakriti')"
BEFORE = "('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned')"


def upgrade() -> None:
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {KINDS}")


def downgrade() -> None:
    op.execute("DELETE FROM records WHERE kind = 'prakriti'")
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {BEFORE}")
