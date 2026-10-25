# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The registered users (Marc, 2026-10-06): who the overseer's screen (routes/adhyaksa.py) has registered, beside worlds.json, in
users.json -- not committed (.gitignore): it holds people's emails. A user is an email, whether they may use the world builder
(`builder`), and the day they were added. Registering a user does not let them through Cloudflare Access; that stays its own step.
The owners are not in this file: they may do everything."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from smritilila.paths import data_dir

_DEFAULT_PATH = data_dir() / "users.json"


@dataclass(frozen=True)
class RegisteredUser:
    email: str
    builder: bool = False
    added: str = ""


def _path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else Path(os.environ.get("SMRITILILA_USERS_CONFIG", _DEFAULT_PATH))


def _norm(email: str) -> str:
    return (email or "").strip().lower()


def load_users(path: str | Path | None = None) -> list[RegisteredUser]:
    """A missing or unreadable file, or one of the wrong shape, is no users: never an error that takes a request down. An entry that is not
    an object with an email string is skipped; the rest stand."""
    p = _path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [RegisteredUser(email=_norm(u["email"]), builder=bool(u.get("builder")), added=str(u.get("added", "")))
                for u in data.get("users", []) if isinstance(u, dict) and isinstance(u.get("email"), str) and _norm(u["email"])]
    except (OSError, ValueError, AttributeError, TypeError):
        return []


def _save(users: list[RegisteredUser], path: str | Path | None) -> None:
    _path(path).write_text(json.dumps({"users": [{"email": u.email, "builder": u.builder, "added": u.added} for u in users]}, indent=2) + "\n", encoding="utf-8")


def add_user(email: str, builder: bool = False, path: str | Path | None = None) -> RegisteredUser:
    email = _norm(email)
    users = load_users(path)
    if any(u.email == email for u in users):
        raise ValueError(f"{email} is already registered")
    user = RegisteredUser(email=email, builder=builder, added=time.strftime("%Y-%m-%d"))
    _save(users + [user], path)
    return user


def set_builder(email: str, builder: bool, path: str | Path | None = None) -> bool:
    """False if no such user."""
    email = _norm(email)
    users = load_users(path)
    if not any(u.email == email for u in users):
        return False
    _save([RegisteredUser(u.email, builder, u.added) if u.email == email else u for u in users], path)
    return True


def remove_user(email: str, path: str | Path | None = None) -> bool:
    email = _norm(email)
    users = load_users(path)
    kept = [u for u in users if u.email != email]
    if len(kept) == len(users):
        return False
    _save(kept, path)
    return True


def may_build(email: str | None, path: str | Path | None = None) -> bool:
    """A registered user the owner has enabled for the world builder."""
    email = _norm(email or "")
    return bool(email) and any(u.email == email and u.builder for u in load_users(path))
