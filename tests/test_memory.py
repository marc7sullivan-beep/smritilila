# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The tuple (smritilila.memory.memory.Memory): immutable once written."""
from __future__ import annotations

import dataclasses

import pytest

from smritilila.memory.memory import Memory


def test_memory_is_immutable() -> None:
    m = Memory("m1", "teo", frozenset({"ilo"}), "I poled across.", "PRIMARY", 4, "landing", 3, "p3", None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.salience = 1  # type: ignore[misc]
    assert dataclasses.replace(m, salience=1).salience == 1 and m.salience == 4
