# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Persistent Entities -- whatever the world has minted: seeded, or met in play.

Per v1.4 Design Principles S5, entityhood has one mechanism: everything
that becomes an Entity (a person, a storm, a rumor, an institution) uses
the same type. What differs is memory, relationships, capabilities, and
behavior -- not ontology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Provenance (identity, v5 §1's sanctioned exception): how an entity came to be. Rows from v2 may still carry its
# words (promoted_path1/2, minted_geometry, minted_overflow, macro_seed, sub_location): stored, read by nothing.
Origin = Literal[
    "genesis",  # planted by a seed: a birth, an expansion, a pratiṣṭhā
    "smrti",    # v3: minted at first appearance by the Narrator (smrti/pulse.py); its label is its first tuple
    "itihasa",  # the world at large (genesis.py plants it): the earth
]

# Āvaraṇa: the scene bit (v5 §11) -- would everyone inside and nobody outside share it. v3's kinds of place (open,
# building, district, room, road) are no longer accepted (2026-10-03, the nirguna audit); rows that hold them are
# scenes, as every v3 place was (v5 Appendix B).
Avarana = Literal["scene"]


@dataclass
class Entity:
    id: str
    labels: list[str]
    created_at_world_time: int
    origin: Origin
    # the scheduler's stamp: the tick this entity last had a turn of its own
    # (scheduler/episode.py). v2's due_tick, last_touched_tick,
    # label_embeddings and ever_acted went on 2026-09-25 -- nothing had
    # written them since v2; the columns of those that had one stay in
    # persistence/models.py, unread.
    last_simulated_tick: int | None = None
    # svabhava ("own-being": place / actor / object) is GONE from the entity
    # (2026-10-03, the nirguna audit). Retired from every reader 2026-09-30
    # -- it decided nothing the scene bit did not, and made persons of a
    # wallet, a derringer and "my horse" -- it was still carried and
    # rewritten on every save. Its column stays in persistence/models.py,
    # unread: old worlds keep what they stored, and nothing reads or writes it.
    # The scene bit (v5 §11). v3 rows still hold v3's words for it (open,
    # building, road, district, room): read as the bit -- any value is a
    # scene -- and never written again.
    avarana: Avarana | None = None
    # Kośa, the sheath (v3.10, 2026-09-23): the id of the building that
    # encloses this place, as the Narrator stated it at mint -- set once,
    # never inferred, never edited. The engine reads no word of a label
    # to decide what is inside what (s0); it asks. A place's scene is the
    # root of its kośa chain, which is s5's enclosure grain stated as
    # structure rather than as a definition of "room".
    kosha: str | None = None
