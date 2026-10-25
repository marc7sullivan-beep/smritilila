# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""kāla: a record of time -- spec v4.2 §47

Revision ID: a8b9c0d1e2f3
Revises: f7a8b9c0d1e2
Create Date: 2026-09-24 23:00:00.000000

Marc, 2026-09-24: people carry time. A world that keeps a clock writes a
`time` record for each person a pulse passes for (value: the minute it
ended; note: "start|origin|dest") and for each scene it touched (note:
"scene"). Additive: the check gains a kind; no row changes, and a world
without a clock writes none.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'a8b9c0d1e2f3'
down_revision: Union[str, None] = 'f7a8b9c0d1e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KINDS = "('kosha','name','action','clause','capability','said','mark','time')"
BEFORE = "('kosha','name','action','clause','capability','said','mark')"


def upgrade() -> None:
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {KINDS}")


def downgrade() -> None:
    op.execute("DELETE FROM records WHERE kind = 'time'")
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {BEFORE}")
