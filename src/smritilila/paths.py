# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Where Smritilila keeps what it keeps (2026-10-06, for an install in a container).

The code lives in the repository; what a server writes -- the registry of worlds, the registered users and bots, the witness logs, the
builder's conversations, proposals, backups, the marker that says which world a port serves -- is state, and in a container it belongs on a
volume, not in the image. SMRITILILA_DATA_DIR names that one folder; unset, state sits at the repository's root, as it always has.

Individual settings still win over it where they exist (SMRITILILA_WORLDS_CONFIG, SMRITILILA_SAKSHI_DIR, ...)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir(default: Path | None = None) -> Path:
    """The folder state is kept in: SMRITILILA_DATA_DIR, else `default`, else the repository's root."""
    env = (os.environ.get("SMRITILILA_DATA_DIR") or "").strip()
    return Path(env) if env else (default or REPO_ROOT)
