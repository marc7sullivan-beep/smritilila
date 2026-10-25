# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Set up and run an Smritilila server on any machine (2026-10-06).

    python scripts/serve.py check              what is configured, and what is missing, said plainly
    python scripts/serve.py setup              make the database if there is none, bring it to the current schema, and plant the example
                                               world (worlds/vadhu_crossing.json) if the database is empty
    python scripts/serve.py run                start the server (http://127.0.0.1:8198)

Settings are read from the environment, and from a `.env` file in the repository's root (copy `.env.example`): a variable already in the
environment wins over the file. Nothing here needs more than the standard library and Smritilila's own requirements.

Smritilila treats a request that carries no identity header as the owner's, because the server is meant to sit on the owner's own machine,
or behind a proxy that names every visitor (Cloudflare Access, an authenticating reverse proxy). So `run` listens on this machine only,
and refuses another address unless SMRITILILA_ALLOW_REMOTE=1 says you have put such a proxy in front of it: anyone who can reach an open
server is its owner."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_DATABASE_URL = "postgresql+asyncpg://smritilila:smritilila@localhost:5432/smritilila"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def load_dotenv(path: Path | None = None) -> list[str]:
    """KEY=VALUE lines of `.env` into the environment, never over a variable that is already set. Quotes around a value are dropped;
    blank lines and # comments are skipped. Returns the names it set."""
    path = path or REPO / ".env"
    if not path.is_file():
        return []
    set_names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            set_names.append(key)
    return set_names


def models_configured() -> list[str]:
    """Which ways of reaching a model the environment names."""
    from smritilila.llm import providers
    return providers.configured()


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def _plain(url: str) -> str:
    """The URL asyncpg takes: the SQLAlchemy driver suffix dropped."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _split(url: str) -> tuple[str, str]:
    """(the URL of the server's maintenance database, the name of this database)."""
    base, _, name = _plain(url).rpartition("/")
    return base + "/postgres", name.split("?", 1)[0]


async def _database_state(url: str) -> str:
    """'ready' (it answers), 'missing' (the server answers, the database is not there), or the reason it cannot be reached."""
    import asyncpg
    try:
        conn = await asyncpg.connect(_plain(url), timeout=5)
        await conn.close()
        return "ready"
    except asyncpg.InvalidCatalogNameError:
        return "missing"
    except Exception as exc:  # noqa: BLE001 -- said to the person setting up, whatever it is
        return f"{type(exc).__name__}: {exc}"


def check() -> int:
    """Say what is configured and what is not. Returns the number of problems."""
    problems = 0
    print(f"Python {sys.version.split()[0]}", end="")
    if sys.version_info < (3, 12):
        print("  -- Smritilila needs Python 3.12 or newer")
        problems += 1
    else:
        print("  ok")
    models = models_configured()
    if models:
        print("Model: " + "; ".join(models) + "  ok")
    else:
        from smritilila.llm import providers
        todo = providers.nothing_configured().split(": ", 1)[1]
        print("Model: NONE. " + todo[:1].upper() + todo[1:] + " "
              "(an OpenAI-compatible server such as Ollama or vLLM). Every pulse of a world is a call to a model, so nothing runs without one.")
        problems += 1
    state = asyncio.run(_database_state(database_url()))
    if state == "ready":
        print(f"Database: {_split(database_url())[1]} answers  ok")
    elif state == "missing":
        print(f"Database: the server answers but there is no database {_split(database_url())[1]!r}. Run: python scripts/serve.py setup")
        problems += 1
    else:
        print(f"Database: cannot reach it ({state}). Is Postgres running, and is DATABASE_URL right? Now: {database_url()}")
        problems += 1
    print("Ready." if not problems else f"{problems} thing(s) to fix.")
    return problems


async def _setup(seed: bool) -> int:
    import asyncpg
    url = database_url()
    maintenance, name = _split(url)
    state = await _database_state(url)
    if state == "missing":
        try:
            conn = await asyncpg.connect(maintenance, timeout=10)
            try:
                await conn.execute(f'CREATE DATABASE "{name}"')
            finally:
                await conn.close()
            print(f"Made the database {name!r}.")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not make the database {name!r}: {type(exc).__name__}: {exc}\n"
                  f"Make it yourself (createdb {name}) or give the role the right to, and run setup again.")
            return 1
    elif state != "ready":
        print(f"Cannot reach the database ({state}). Is Postgres running, and is DATABASE_URL right? Now: {url}")
        return 1
    else:
        print(f"The database {name!r} is there.")
    from smritilila import adhisthana
    await adhisthana._bring_to_head(url)
    print("The schema is at the current version.")
    if seed:
        from sqlalchemy import func, select
        from smritilila.persistence.engine import make_engine, make_session_factory
        from smritilila.persistence.models import EntityRow
        engine = make_engine(url)
        async with make_session_factory(engine)() as session:
            held = (await session.execute(select(func.count()).select_from(EntityRow))).scalar_one()
        await engine.dispose()
        if held:
            print(f"The database already holds {held} beings: the example world is left alone.")
        else:
            from smritilila.smrti.genesis import seed as plant
            await plant(str(REPO / "worlds" / "vadhu_crossing.json"), url)
    print("Setup done. Start the server with: python scripts/serve.py run")
    return 0


def run(host: str, port: int) -> int:
    if host not in LOOPBACK and os.environ.get("SMRITILILA_ALLOW_REMOTE") != "1":
        print(f"Refusing to listen on {host}: Smritilila treats a request with no identity header as the owner's, so an open server hands "
              "ownership to anyone who can reach it. Put an authenticating proxy in front (Cloudflare Access, an identity-aware reverse "
              "proxy) and set SMRITILILA_ALLOW_REMOTE=1, or run on 127.0.0.1.")
        return 2
    if not models_configured():
        print("No model is configured (see: python scripts/serve.py check). The page will load, but no world can act.")
    os.environ.setdefault("DATABASE_URL", database_url())
    from smritilila.paths import data_dir
    os.environ.setdefault("SMRITILILA_SAKSHI_DIR", str(data_dir() / "sakshi"))
    os.environ.setdefault("SMRITILILA_TRAINING_DATA_LOG", str(data_dir() / "training_data.jsonl"))   # every prompt the Narrator is shown, kept (spec §44); scripts/dev_server.py has always set it, and this launcher did not
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("SMRITILILA_NO_SCENES", "1")   # no scene tag: a container is a scene once someone who perceives has stood in it (the same default as scripts/dev_server.py); 0 reads the tag as before
    import uvicorn
    uvicorn.run("smritilila.main:app", host=host, port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("setup")
    s.add_argument("--no-example", action="store_true", help="do not plant the example world")
    r = sub.add_parser("run")
    r.add_argument("--host", default=os.environ.get("SMRITILILA_HOST", "127.0.0.1"))
    r.add_argument("--port", type=int, default=int(os.environ.get("SMRITILILA_PORT", "8198")))
    load_dotenv()
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return 1 if check() else 0
    if a.cmd == "setup":
        return asyncio.run(_setup(seed=not a.no_example))
    return run(a.host, a.port)


if __name__ == "__main__":
    sys.exit(main())
