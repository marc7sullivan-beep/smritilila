# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Id generation for what is newly minted (canonical entities, records, pulse ids)."""

from __future__ import annotations

import uuid
from typing import Callable

IdFactory = Callable[[], str]


def new_id() -> str:
    return uuid.uuid4().hex
