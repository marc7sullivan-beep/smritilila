# Smṛtilīlā

*Smritilila · स्मृतिलीला · the play of memory · Apache 2.0*

> **Note:** analyzing this code with a code agent will be grossly inaccurate on a single pass, due to its non-standard ontology.

A persistent, multiplayer, generative world engine. A world is a set of memories: places and people each hold what they have lived,
and a narrator model writes what happens next from what each of them remembers. The engine keeps memory and knows nothing else;
everything a world *is* (its weather, its roads, its manners) is text written once at its birth and then by play.

**Start with `Smritilila in Brief.md`**: what a world is, how it works, and how to play one, on a page or two. Two tales of played worlds, written from the memories of the characters in
them, are in `stories/`. They, and more tales like them, can be read as pages in the [Smṛtilīlā Chronicles](https://world-builder-stories.netlify.app/).

Smritilila is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE.md`). Rough edges are expected.

**It has been tested with both cloud LLM API calls and local models.**

## What you need

- **Linux or Windows** with **Python 3.12**. It has been run on Ubuntu 24.04 (the tests, in continuous integration) and on Windows 11; macOS has not been tried, and neither has a Python newer than 3.12.
- **Postgres 17**, the version the tests and the containers use; no other version has been tried. Either install it, or use the included `docker-compose.yml` (`docker compose up -d`). The login in the examples
  (`smritilila` / `smritilila`) is a local default for a throwaway database: choose your own password before you keep anything you care about in it.
  With your own Postgres, make that role first (`CREATE ROLE smritilila LOGIN PASSWORD '...' CREATEDB;`): `setup` makes the database if it is missing, and
  needs the right to. A role without it can have the database made for it (`createdb smritilila`). `pg_dump` should be on the `PATH` (or named in
  `PG_DUMP_PATH`): a world is backed up with it before its schema is changed or it is cleared, and whenever a backup is asked for.
- **A language model, which you wire in.** Smritilila includes no model, names none and chooses none for you. Every pulse of a world is a call to a
  model, so nothing acts until you connect one, and a world plays only as well as the model behind it. Connect a model, served by you or by a
  hosted service, at an OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp's server and LM Studio all offer one): its address and the model id.
  Another kind of backend is a small module you add (see "Another backend", below).
- Disk and patience for the first start: it downloads a small embedding model (about 100 MB) the first time.

## Quick start with containers (an Ubuntu server with Docker)

You need Docker Engine and the Compose plugin, v2.24 or newer (`docker compose version`). Podman's `podman compose` should work and has not been tried.

```bash
git clone https://github.com/marc7sullivan-beep/smritilila.git && cd smritilila
cp .env.example .env              # set one way to reach a model (see below); leave DATABASE_URL alone, compose sets the container's own
docker compose up -d --build      # Postgres and Smritilila; the first start downloads an embedding model (about 100 MB) into the volume
docker compose logs -f smritilila      # until it says it is running
curl http://127.0.0.1:8198/health
```

Smritilila is published on **127.0.0.1 of the server only**, on purpose: whoever can reach it is its owner (see "Safety" below). To use it from
your own machine, tunnel to it: `ssh -L 8198:127.0.0.1:8198 you@server`, then open <http://127.0.0.1:8198>.

**A model, wired in by you.** Nothing is included. Either:
- Ollama as a container beside it: `docker compose --profile ollama up -d`, then `docker compose exec ollama ollama pull <model>`, and in `.env`
  `SMRITILILA_LOCAL_BASE_URL=http://ollama:11434/v1` and `SMRITILILA_LOCAL_MODEL=<model>` (for an NVIDIA GPU, see the comment in `docker-compose.yml`);
- a model server already running on the host or elsewhere, or a hosted service with an OpenAI-compatible endpoint:
  `SMRITILILA_LOCAL_BASE_URL=http://host.docker.internal:11434/v1` (or its address), `SMRITILILA_LOCAL_MODEL=<model>`, and `SMRITILILA_LOCAL_API_KEY` if it asks for one.
After editing `.env`: `docker compose up -d` again. `docker compose exec smritilila python scripts/serve.py check` says what is configured.

**Where things live.** The database is in the volume `smritilila-pgdata`. Everything else Smritilila writes (the list of worlds, the witness logs, backups, the
builder's conversations) is in `smritilila-data`, mounted at `/data`. A world built from the page runs on a server of its own, on the next free
port from 8200; those ports (8200-8230) are published on the server's loopback too, and the page tells you the address.
`docker compose down` keeps both volumes; `docker compose down -v` deletes them.

## Quick start without containers

```bash
git clone https://github.com/marc7sullivan-beep/smritilila.git && cd smritilila
python3 -m venv .venv && . .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch    # first, unless you have a GPU to use: the default wheel brings a GPU stack of gigabytes
pip install -e ".[dev]"

cp .env.example .env            # then edit .env: set DATABASE_URL and one way to reach a model
python scripts/serve.py check   # says what is configured and what is missing
python scripts/serve.py setup   # makes the database, brings it to the current schema, plants the example world
python scripts/serve.py run     # http://127.0.0.1:8198
```

Open <http://127.0.0.1:8198>, pick a character (Teo, the ferryman's apprentice, is in the example world), and write what he does.

The engine is run from a clone of this repository, as above: its pages, its schema and its scripts sit beside the package, not inside it. The name `smritilila` on PyPI is a placeholder
that holds the name, so `pip install smritilila` does not install the engine.

On **Windows**, in PowerShell: `py -3.12 -m venv .venv`, then `.venv\Scripts\Activate.ps1`, and `copy .env.example .env`; the rest is the same. To set a variable for one session
rather than in `.env`: `$env:TEST_DATABASE_URL = "..."`, where the commands here say `export`. The first start downloads the embedding model, so allow it a few minutes.

### With a local model

In `.env`:

```
SMRITILILA_LOCAL_BASE_URL=http://localhost:11434/v1     # Ollama's address; vLLM is usually :8000/v1
SMRITILILA_LOCAL_MODEL=<model>                           # the id of whatever you serve
SMRITILILA_LOCAL_TIMEOUT=600
```

Smritilila asks the model for structured JSON on every pulse, with a long prompt (the memories that matter to the scene). A small model
may fail to keep to the shape, or be slow; the engine records what it asked and what came back (`sakshi/`, one file per run) so a
failure can be seen rather than guessed.

**The model needs a long context.** A pulse's prompt is about 10,000 tokens in a fresh world and grows with the world: over a few hundred calls in worlds that had
been played for a while, the median was about 23,000 tokens, one in ten was over 31,000, and the largest was about 40,000. The answer is small by comparison (about
1,200 tokens at the median, at most about 5,000). Give the model a context of at least 32,000 tokens, and more for an old world. A server whose context is smaller
will usually **cut the prompt without saying so**, and the narration then comes back thin, wrong, or malformed. Ollama's default is only a few thousand tokens, and
its OpenAI-compatible endpoint ignores a context asked for in the request, so set `OLLAMA_CONTEXT_LENGTH` for the server (or `num_ctx` in a Modelfile) and check the
value its log reports. Other servers have a setting of their own (`--max-model-len` in vLLM, `-c` in llama.cpp's server).

## Making your own world

The page's **new world** button opens a conversation with a builder model: say what you want ("Treasure Island, 1765", "the
Arizona Territory in 1881", "my own village"), it asks a couple of questions and fills in a brief, shows a preview, and builds the
world on a server of its own. This uses the model heavily and works better with a stronger one.

How much to build is a choice: *here* (only where it begins), *around* (a day's walk), *small* (a region, about 25 places), *medium* (about 70) or *large* (about 120). Medium and large grow the world outward, ring by ring, after the first seed, which takes many more model calls and several minutes.

A map image can be attached (**Attach a saved map**, from a folder under `maps/` in the data folder, or **Browse for an image...** on your own computer, which saves it there; up to three; a png, jpeg, webp or gif, at most 25 MB): the builder looks at it with your next message, and when the world is built it is read place by place into a proposal for you to review, which plants nothing until you say. **Maps need a model that can read images**; a model that takes words only is refused, and the builder says so.

### Worlds made and tried

Eight worlds have been made and played so far (the brief lists them). Only the first ships with the code (`worlds/vadhu_crossing.json`); the rest are described here as examples of what a seed and the builder can make. Two tales of played worlds, one of Lovecraft and one of Greyhawk, are in `stories/`.

- **The Vadhū Crossing:** a ferry crossing on a river in late autumn, with one inn, a ferry and two landings, where every small thing is remembered by someone.
- **Greyhawk:** a classic fantasy campaign setting, with a village, a keep, a moathouse and dragons, played by several people at once, with experience and levels kept as play goes on.
- **Lovecraft:** a 1920s cosmic-horror world of a New England university town, where being near one of the old things leaves lapses of mind.
- **The Braided Isles:** an archipelago ninety years after a great sundering, where the sea shifts and every crossing grows dangerous.
- **Austin:** present-day Austin, Texas, with no magic: phones, texts and letters, in a tired lull between festival weekends.
- **The Breakthrough:** a timeless realm of shifting color and geometry, where a single breath feels like a whole life.
- **Early Rome:** Rome under Augustus, early in the Principate and after decades of civil war: the Forum, the Curia and the Palatine, senators and freedmen and the stewards of great houses.
- **The American West, 1880:** a frontier town of the American West in 1880.

## Bots

A character can be switched on as a **bot** in the admin screen. `python scripts/bots_supervisor.py` then starts an agent for it whenever a person is
online in its world, and stops it when the last person has gone, leaving the character waiting. The agent plays through the same model setup as the
engine's other calls: whichever model you have wired in.
It is stopped by an action limit (`SMRITILILA_BOT_MAX_ACTIONS`, 25 by default) and by silence, and every action is a model call, so it costs. A world's
note for its bots can go in `scripts/agents/worlds/<name>.md`; without one the agent knows only what its character's page shows.
A bot can also be **held**: `hold` in `bots.json` lists the bots whose character is kept online while someone is there, with nothing acting for it and no model asked.

**A bot never moves ahead of a person in time.** By default, while a person is in the world, a bot's character acts alone only while it is behind every person on the clock
(catching up), joins a round a person opened, and answers no to a question of jumping ahead that would carry it past one; otherwise its agent is told to hold, and the action
is refused without a model call. The rule is the agent tool's, not the model's. To turn it off for a run, `SMRITILILA_BOT_TIME_GUARD=0` (or `init --guard off` for one agent). For a
simulation of bots alone, mark them **free** in `bots.json` (`"free": {"<world>": ["<character id>"]}`, or `bots_registry.set_free`): a free bot has the guard off and is started
with no person in the world, and kept running when the last person goes.

## What it costs

On a paid service, every pulse is a call with a long prompt (see "The model needs a long context" above: roughly 10,000 to 40,000 tokens in, about 1,000 out); a
session of a few dozen actions costs real money. A local model costs time instead.

## Another backend

The engine speaks to a model only through the `LLMClient` protocol (`src/smritilila/llm/client.py`) and knows no vendor. A backend is a module
`<name>_provider.py` in `src/smritilila/llm/` that calls `register(Provider(...))` from `llm/providers.py` when it is imported: how to build its client, how to say it is
configured, and in what order it is asked. The one that ships is the OpenAI-compatible endpoint above. With more than one configured, `SMRITILILA_PRIMARY_PROVIDER` names the
one asked first, and `SMRITILILA_ROTATE_SHARE` sends a share of the narration to the second (0 to 1; 0 means only when the first fails).

## What it keeps, and where your play goes

**Play goes to the model you wire in.** Each pulse sends it the scene, what the characters in it remember, and the words the player wrote, including a character's
written backstory. With a model on your own machine or network that goes nowhere else. With a hosted service it goes to that service, under its terms and not these.
Nothing of it is sent anywhere else. Two other connections exist: the embedding model is fetched once, from Hugging Face, the first time the
server starts; and, only if you set `SMRITILILA_NTFY_TOPIC`, a short push naming a guest and the world they entered goes to <https://ntfy.sh> (or `SMRITILILA_NTFY_URL`) when that
guest arrives. With the topic unset, which is the default, nothing is sent.

**A prompt log is kept by default.** `scripts/serve.py run` writes `training_data.jsonl` in the data folder (`SMRITILILA_DATA_DIR`; `/data` in the containers): one line for every
model call, holding the whole prompt and the whole answer. It is there so that a past pulse can be explained from exactly what the model was shown, and so that a model can
be tuned on real play. The engine only writes it and never reads it back, and nothing sends it anywhere. It grows without limit: tens of kilobytes for one call in a fresh
world, up to about 100 KB in an old one (the author's, over many worlds, has reached 274 MB), and it holds every word players have written. Keep it as private as the
database itself, and leave it out of anything you publish or back up somewhere shared. **To turn it off,** put `SMRITILILA_TRAINING_DATA_LOG=` (empty) in `.env`; to put it somewhere
else, give it a path.

The witness logs (`sakshi/`, one file per run) and the builder's conversations are in the same folder and also hold play. No world depends on any of these: they can be deleted
whenever the server is stopped.

## Safety: this is the owner's server

Smritilila treats a request that carries no identity header as the **owner's**. That is right on your own machine, which is why
`scripts/serve.py run` listens on `127.0.0.1` only and refuses another address. To let other people in, put an authenticating proxy
in front of it (Cloudflare Access, an identity-aware reverse proxy) that names every visitor in
`Cf-Access-Authenticated-User-Email`, set `SMRITILILA_OWNERS` to your own email, and only then set `SMRITILILA_ALLOW_REMOTE=1`. Anyone who can
reach an open server is its owner.

**The Host allow-list.** A page in your own browser can reach `127.0.0.1` too (DNS rebinding points another site's name at it), and can send any header it likes. What it
cannot change is the `Host` it is reached by, so every request is first checked against the names this server answers to: this machine (`localhost`, `127.0.0.1`, `::1`) and
the domains in `SMRITILILA_ALLOWED_HOSTS` (comma-separated; each allows itself and its subdomains). **Any other Host is refused with a 400.** The default list is a placeholder
domain, `example.com`, so **set `SMRITILILA_ALLOWED_HOSTS` to your own domain before you put a tunnel or proxy in front of the server**, or its visitors are refused.
`SMRITILILA_HOST_GUARD` is `enforce` (the default), `report` (logs what it would refuse and lets it through: for the first run on a real tunnel) or `off`.

## Layout

| | |
|---|---|
| `src/smritilila/` | the engine and its HTTP API (`main.py` is the map) |
| `src/smritilila/web/` | the page (`index.html`) and the admin screen (`admin.html`) |
| `migrations/` | the database schema (alembic) |
| `worlds/vadhu_crossing.json` | the example world, a ferry crossing in late autumn |
| `scripts/serve.py` | check, set up and run |
| `scripts/bots_supervisor.py`, `scripts/agents/` | the bots: who runs them, and the agent that plays a character through the engine's model |
| `scripts/v3/sakshi.py` | reads the witness log of a run: `python scripts/v3/sakshi.py` lists runs, `... <run>` says what happened |
| `tests/` | the test suite: `pytest` (it needs no database; the tests that use a live Postgres are skipped unless `TEST_DATABASE_URL` is set, see below) |
| `Smritilila in Brief.md` | the idea in brief: what a world is, how it works, and how to play one |
| `stories/` | two tales of played worlds, one of cosmic horror and one of a fantasy adventure, written from the memories of the characters in them |

## Tests

```bash
pytest -q
```

That needs no database: the tests that use a live Postgres are skipped. To run them too, give them a database of their own:

```bash
createdb smritilila_test        # the live tests use their own database and empty it: never point TEST_DATABASE_URL at one you care about
export TEST_DATABASE_URL=postgresql+asyncpg://smritilila:smritilila@localhost/smritilila_test
pytest -q
```

## Questions

The code's dated comments are its memory: read one near anything surprising. The author is not interested in bug reports or code questions.
Questions about the ontology (what a world is, what a being is, how memory, time and becoming work, and why) are welcome: marc7sullivan@icloud.com.

## License, authors and contributing

Smritilila is licensed under the Apache License 2.0 (`LICENSE`); the brief, `Smritilila in Brief.md`, and the two stories in `stories/` under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Copyright 2026 Marc Sullivan.

Smritilila was designed and written by Marc Sullivan together with Anu.

A change sent in is made under the Developer Certificate of Origin: sign off each commit (`git commit -s`). `CONTRIBUTING.md` says how.
