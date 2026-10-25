# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Smṛti (स्मृति -- that which is remembered): the v3.0 substrate.

One structure, one loop. A tuple is [perceiver, seq, chronology, memory];
the runtime process is retrieve -> narrate -> append. See `Smritilila Design
Principles v3_0.md`. Identity is the perceiver; scenes are filed as streams
of their own (an impersonal record), and perceive only where they have the
capability (v5 sections 8, 11); the identifier is opaque.

`context.py` is the s7 retrieval, `narrator.py` the Narrator call,
`pulse.py` the pulse that appends what it returns; `kosha.py` is
containment (v4), `kala.py` each person's time, `log.py` the store rows
read as tuples, `samskara.py` the kept reading of them, `bija.py` and
`genesis.py` a world's seeding, `phala.py` what lived time earns,
reckoned at rest.
"""
