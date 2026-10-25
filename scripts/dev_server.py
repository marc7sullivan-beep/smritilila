# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Dev-server launcher for the Browser-pane preview tool, which spawns a
fresh process that doesn't inherit env vars set in any already-running
shell. DATABASE_URL is hardcoded to the local Postgres instance this
project's migrations already target; every *_API_KEY is read straight
from the Windows user registry (set via `setx`, so it's durable but not
visible to already-running processes -- only to freshly-spawned ones that
re-read it, which is exactly what this script does on their behalf).
SMRITILILA_SAKSHI_DIR defaults on for the same reason -- a preview-launched
process doesn't inherit a shell's own `$env:SMRITILILA_SAKSHI_DIR` either,
so the witness (pulse.timing/activation_pulse.timing/request.timing
included) was silently firing into nowhere for a preview-started server
until this default existed. Pinned to an ABSOLUTE path derived from this
file's own location, not a bare relative one -- confirmed live,
2026-09-07: a preview-tool-launched process's own cwd is NOT this
project's root (it's whatever directory the launch session itself is
rooted in), so a relative path silently wrote the file into that other
directory instead, nowhere near this repo.

A DIRECTORY, not a file, since 2026-09-22: sakshi writes one file per
run, named for the run. Two servers run here at once (8198 and 8199)
and each restart starts a new run, so what used to be one undivided
file is now a directory that reads as a history of the server.

SMRITILILA_TRAINING_DATA_LOG defaults on too, same reasoning, by explicit
request while this project is still in active development -- every
generative-role call this dev server ever serves is worth capturing as
potential LoRA training data (smritilila.training_data) by default, not
opt-in per session. Revisit this default (opt-in instead) once the
project is no longer in this exploratory phase, given the real prompt/
completion volume this writes compared to sakshi's own lightweight
metadata.

Console output (uvicorn's own request logs and any traceback) is teed to
logs/dev_server.log as well as this process's own stdout -- so a crash
can be read straight off disk instead of copy-pasted out of whichever
terminal happened to be running the server.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://smritilila:smritilila@localhost/smritilila")
# Assigned, not setdefault (2026-09-15): after the repo moved to
# another folder, a preview-launched server inherited the OLD absolute paths
# from the previous project directory's session environment and wrote
# a whole run's sakshi lines and captures into the stale copy. These two
# are pinned to THIS file's own location, whatever the environment says.
# (SMRITILILA_DATA_DIR, in a container, is the volume state lives on; unset, it is the repository, as before)
_STATE = Path(os.environ.get("SMRITILILA_DATA_DIR") or Path(__file__).resolve().parent.parent)
os.environ["SMRITILILA_SAKSHI_DIR"] = str(_STATE / "sakshi")
os.environ["SMRITILILA_TRAINING_DATA_LOG"] = str(_STATE / "training_data.jsonl")
# One pulse in ten goes to the second API (2026-09-22, Marc's call).
# A recourse path nobody exercises is a recourse path nobody knows is
# broken -- this one sat on a preview model id for months, and the
# obvious stable replacement turned out to be 404 for new users. Both
# facts surfaced only when something finally called them. Now a share of
# pulses calls it, and llm.call carries route and model so the
# record says which one answered. Those pulses go through
# complete_json, the shape path THE PORT needs, so they prove that too.
# Expect a different voice on them: v3 makes one Narrator call per
# pulse, so the share IS the fraction of pulses another model wrote.
# Set to 0 to stop; SMRITILILA_SECONDARY_MODEL picks a different one.
# 0.5 from 2026-09-25 (Marc): an even split, so the second model is measured on the
# day's rules (whole names, the age's pull, who stands elsewhere) as often
# as the larger model -- they were replayed on the larger model alone.
# 0.8 from 2026-09-26 (Marc): the the first provider bill. Flash wrote 13 of the
# 50/50 run's 26 pulses (Ravenmoor 676-701) with no silent loss, no error,
# no drop; kept over the mid-size model 5 as primary, which replayed 16/8 on the mint
# questions and 5 of 12 floating containers against the larger model's ~20/2 and 0.
# 0.95 from 2026-10-01 (Marc: "change the mid-size model to be only called 5% of the
# time"): replayed on saṅkalpa (v5 §28), the second model let a ruinous draw land as
# ruinous every time where the mid-size model softened it until given its own reading.
# the mid-size model stays the fallback when the second model fails, and births stay the second model-first.
# 1 from 2026-10-13 (Marc: "Let's make the small model backup only. It's really not performing well"): the primary, the small model since v5.28, answers
# only when the second model fails. In the 2026-10-13 run on Stonemarch its two player pulses took 29.7 s and 36.9 s, with 4.6k and 6.4k tokens
# of thinking, where the second model's took 7-15 s.
os.environ.setdefault("SMRITILILA_ROTATE_SHARE", "0")
# No scene tag (2026-10-19, Marc: "turn it on for all worlds and for world creation"): a container is a scene once a perceiver has stood
# directly in it (kosha.no_scenes). Austin and Stonemarch ran it for a day through scripts/restart_worlds.ps1's per-port table first. Set here
# so that every world server, a reboot's logon task and the creation routes in them (/worlds/create, /genesis, the expansions) read the same
# rule; SMRITILILA_NO_SCENES=0 in the environment turns it off for one server.
os.environ.setdefault("SMRITILILA_NO_SCENES", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# SMRITILILA_SHADOW_ASK, SMRITILILA_JIJNASA_MODE and SMRITILILA_SMRTI_MODE were set
# here until 2026-09-22. Jijnasa (the shadow-ask experiment, one the small model
# call per player pulse) and the v2 pipeline they switched between both
# went on 2026-09-21; every pulse is now retrieve -> v3 Narrator ->
# append, unconditionally, with no second loop to opt a world into.
# Nothing read any of the three.

if sys.platform == "win32":
    import winreg

    # The user-scope registry value WINS over an inherited one (2026-09-17):
    # a server launched from the desktop app inherited the app's own
    # environment, captured before the key was rotated; the old key
    # expired that afternoon and every Narrator call fell back to the second model
    # for seventeen ticks while the valid key sat in HKCU unread.
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        # SMRITILILA_NTFY_TOPIC (2026-10-01): the guest-online alerts' private topic -- read here so every restart
        # carries it, however the server is started; unset, no alert is sent
        # (2026-10-17: every *_API_KEY the user has set, whichever provider it is for, rather than two named ones)
        for i in range(winreg.QueryInfoKey(key)[1]):
            name, value, _ = winreg.EnumValue(key, i)
            if not (name.endswith("_API_KEY") or name == "SMRITILILA_NTFY_TOPIC"):
                continue
            if os.environ.get(name) not in (None, value):
                print(f"{name}: inherited value differs from the user-scope registry; using the registry", flush=True)
            os.environ[name] = value

LOG_PATH = _STATE / "logs" / "dev_server.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# SMRITILILA_DEV_PORT: Marc plays on 8198; a second server (the senator's
# runs) takes another port rather than the same one (2026-09-16).
PORT = os.environ.get("SMRITILILA_DEV_PORT", "8198")
# SMRITILILA_HOST: 127.0.0.1 unless a container says otherwise (its published ports are what decide who can reach it)
HOST = os.environ.get("SMRITILILA_HOST", "127.0.0.1")

# UTF-8 both ways, undecodable bytes replaced (2026-09-30): read in the
# locale's cp1252, a migration's log line ("kāla") raised in the loop
# below, and its `finally` killed the server mid-migration -- the sample
# world's /worlds/create, and any in-server migration at load.
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "smritilila.main:app", "--host", HOST, "--port", PORT],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    # no console window of its own (2026-10-08): started detached, this script has no console, so a console child got a visible one, and Marc closing it killed the world
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
)
sys.stdout.reconfigure(errors="replace")  # a console that cannot show a character shows a stand-in, not a crash

with LOG_PATH.open("a", encoding="utf-8") as log_file:
    log_file.write(f"\n=== dev server started, pid={proc.pid} ===\n")
    log_file.flush()
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        proc.wait()
        log_file.write(f"=== dev server exited, returncode={proc.returncode} ===\n")

sys.exit(proc.returncode)
