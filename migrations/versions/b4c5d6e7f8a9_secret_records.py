# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""secret: an entity kept from every god outside its holder

Revision ID: b4c5d6e7f8a9
Revises: a3b4c5d6e7f8
Create Date: 2026-10-02 18:00:00.000000

Marc and Anu, 2026-10-02 (spec v4.61, pratiṣṭhā, option 1): a seed may be a
secret -- the book Anu seeded into Stonemarch is one. A `secret` record (value:
who holds it) marks an entity; a proposal naming any secret entity is
withheld from every god outside the secret, and what such a proposal plants
is born secret. Additive: the check gains a kind; no row changes.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'b4c5d6e7f8a9'
down_revision: Union[str, None] = 'a3b4c5d6e7f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KINDS = "('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned','prakriti','secret')"
BEFORE = "('kosha','name','action','clause','capability','said','mark','time','sandhi','reckoned','prakriti')"


def upgrade() -> None:
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {KINDS}")


def downgrade() -> None:
    op.execute("DELETE FROM records WHERE kind = 'secret'")
    op.drop_constraint("records_kind_check", "records", type_="check")
    op.create_check_constraint("records_kind_check", "records", f"kind IN {BEFORE}")
