# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Citra (चित्र, a picture): one picture a world keeps (2026-11-11, Marc: "I want to try adding image generation and presentation").

A picture is of an entity (a being's portrait, a place's view) or of a telling (a moment), never both: the engine knows no quality (nirguna), so
there is no `kind`, and a place is told from a being by the scene bit its entity already carries. Insert-only, like a memory. It lives beside
the world's memories and records and is never one of them: the Narrator is never shown a picture, and nothing here is retrieved (the echo is
never fed back: the Narrator would read its own picture's prompt back as if it were the world).

A being the writer judged to have no face (a key, a drawbridge) is kept as a row with no data and a note, so it is asked about once.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Chitra:
    id: str
    entity_id: str | None          # whose likeness, or which place: None for a moment
    pulse_id: str | None           # the telling a moment is of: None for an entity's picture
    world_time: int
    prompt: str                    # the words that were sent (never the bytes)
    model: str
    mime: str = ""
    data: bytes | None = None      # None: a being judged to have no face (`note` says so), or the index's row, which carries no bytes
    note: str | None = None
    seeds: tuple[str, ...] = ()    # the entities whose portraits were the seed of this one, the actor first
    seq: int = 0                   # the order pictures were kept in, set by the store; a page asks for those after the last it saw
    made_at: datetime | None = None
    size: int = 0                  # bytes of `data`, kept when the index is read without them

    def has_picture(self) -> bool:
        return bool(self.size or self.data)

