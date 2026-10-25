# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Async engine/session factory. Reads DATABASE_URL (asyncpg DSN, e.g.
postgresql+asyncpg://user:pass@host/smritilila) at call time rather than import
time, so importing this module never requires the env var to be set --
only actually connecting does.
"""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def make_engine(database_url: str | None = None) -> AsyncEngine:
    url = database_url or os.environ["DATABASE_URL"]
    return create_async_engine(url)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
