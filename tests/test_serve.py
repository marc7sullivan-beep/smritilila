# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""scripts/serve.py (2026-10-06): the portable way to set up and run Smritilila, for a machine that is not the author's."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "serve.py"
spec = importlib.util.spec_from_file_location("serve_under_test", SCRIPT)
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)

KEYS = ("SMRITILILA_LOCAL_BASE_URL", "SMRITILILA_LOCAL_MODEL", "DATABASE_URL", "SMRITILILA_ALLOW_REMOTE", "SMRITILILA_HOST")


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    import os
    for k in KEYS + tuple(k for k in os.environ if k.endswith("_API_KEY")):   # every provider's key, whatever it is called
        monkeypatch.delenv(k, raising=False)
    # run() sets these for the process it starts: set here first so monkeypatch puts them back
    # and no later test inherits them
    monkeypatch.setenv("SMRITILILA_SAKSHI_DIR", str(tmp_path / "sakshi"))
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setenv("SMRITILILA_NO_SCENES", "0")   # run() defaults it to 1 for the process it starts: put back after each test, so no later test reads the derived rule


def test_the_env_file_is_read_without_overriding_what_is_already_set(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# a comment\n\nDATABASE_URL="postgresql+asyncpg://u:p@h/db"\nSOME_API_KEY=from-file\nSMRITILILA_LOCAL_MODEL = \'a-model\'\nnot a setting\nEMPTY=\n',
                 encoding="utf-8")
    monkeypatch.setenv("SOME_API_KEY", "from-shell")
    import os
    for k in ("DATABASE_URL", "SMRITILILA_LOCAL_MODEL", "EMPTY"):   # load_dotenv sets these for good: registered with monkeypatch to be put back
        monkeypatch.setenv(k, "placeholder")
        monkeypatch.delenv(k)
    set_names = serve.load_dotenv(f)
    assert os.environ["DATABASE_URL"] == "postgresql+asyncpg://u:p@h/db"            # quotes dropped
    assert os.environ["SOME_API_KEY"] == "from-shell"                              # the shell wins
    assert os.environ["SMRITILILA_LOCAL_MODEL"] == "a-model" and os.environ["EMPTY"] == ""
    assert "SOME_API_KEY" not in set_names and "DATABASE_URL" in set_names
    assert serve.load_dotenv(tmp_path / "missing.env") == []                         # no file is no error


def test_the_models_the_environment_names(monkeypatch):
    assert serve.models_configured() == []
    monkeypatch.setenv("SMRITILILA_LOCAL_BASE_URL", "http://localhost:11434/v1")
    assert serve.models_configured() == []                                           # a URL alone names no model
    monkeypatch.setenv("SMRITILILA_LOCAL_MODEL", "a-model")
    assert len(serve.models_configured()) == 1 and "a-model" in serve.models_configured()[0]


def test_the_database_url_splits_into_its_server_and_its_name():
    assert serve._split("postgresql+asyncpg://smritilila:pw@localhost:5432/smritilila_x") == ("postgresql://smritilila:pw@localhost:5432/postgres", "smritilila_x")
    assert serve._split("postgresql+asyncpg://a@h/db?ssl=off")[1] == "db"
    assert serve.database_url() == serve.DEFAULT_DATABASE_URL


@pytest.mark.parametrize("state,model,problems,says", [
    ("ready", True, 0, "Ready."),
    ("missing", True, 1, "Run: python scripts/serve.py setup"),
    ("OSError: refused", True, 1, "Is Postgres running"),
    ("ready", False, 1, "Model: NONE"),
])
def test_check_says_what_is_missing_and_counts_it(monkeypatch, capsys, state, model, problems, says):
    async def fake_state(url):
        return state
    monkeypatch.setattr(serve, "_database_state", fake_state)
    if model:
        monkeypatch.setenv("SMRITILILA_LOCAL_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("SMRITILILA_LOCAL_MODEL", "m")
    assert serve.check() == problems
    assert says in capsys.readouterr().out


def test_the_server_refuses_an_open_address_unless_told_a_proxy_stands_in_front(monkeypatch, capsys):
    started = []
    import types, sys
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/serve_test")   # run() would set it for good: set here, so it is put back
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: started.append((a, k))))
    assert serve.run("0.0.0.0", 8198) == 2 and not started
    out = capsys.readouterr().out
    assert "no identity header as the owner's" in out and "SMRITILILA_ALLOW_REMOTE=1" in out            # says why, and what to do
    assert serve.run("127.0.0.1", 8198) == 0 and started[-1][1] == {"host": "127.0.0.1", "port": 8198}
    assert serve.run("localhost", 8199) == 0 and serve.run("::1", 8199) == 0                      # this machine, however it is spelled
    monkeypatch.setenv("SMRITILILA_ALLOW_REMOTE", "1")
    assert serve.run("0.0.0.0", 8198) == 0 and started[-1][1]["host"] == "0.0.0.0"
    monkeypatch.setenv("SMRITILILA_ALLOW_REMOTE", "yes")
    assert serve.run("192.168.1.5", 8198) == 2                                                     # only "1" is the word


def test_a_server_with_no_model_says_so_and_still_starts(monkeypatch, capsys):
    import types, sys
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/serve_test")
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))
    assert serve.run("127.0.0.1", 8198) == 0 and "No model is configured" in capsys.readouterr().out


def test_the_launcher_keeps_the_training_log_as_dev_server_does_and_leaves_an_explicit_choice_alone(monkeypatch, tmp_path):
    """Spec §44: the training log (every prompt the Narrator was shown) is written wherever Smritilila runs. scripts/dev_server.py always set it; `serve.py run` did not, so a newcomer's server kept none."""
    import os
    import sys
    import types
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))
    monkeypatch.setenv("SMRITILILA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SMRITILILA_TRAINING_DATA_LOG", raising=False)
    assert serve.run("127.0.0.1", 8198) == 0
    assert os.environ["SMRITILILA_TRAINING_DATA_LOG"] == str(tmp_path / "training_data.jsonl")
    monkeypatch.setenv("SMRITILILA_TRAINING_DATA_LOG", str(tmp_path / "mine.jsonl"))
    assert serve.run("127.0.0.1", 8198) == 0
    assert os.environ["SMRITILILA_TRAINING_DATA_LOG"] == str(tmp_path / "mine.jsonl")             # what was chosen stands


def test_both_launchers_run_with_no_scene_tag_unless_told_otherwise(monkeypatch, tmp_path):
    """2026-10-19 (Marc: "turn it on for all worlds and for world creation"): a container is a scene once a perceiver has stood in it, for every server this machine starts
    (scripts/dev_server.py: the worlds, the logon task, the creation routes inside them) and for a newcomer's (`serve.py run`). SMRITILILA_NO_SCENES=0 reads the tag as before."""
    import ast
    import os
    import sys
    import types
    from smritilila.smrti import kosha
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))
    monkeypatch.setenv("SMRITILILA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SMRITILILA_NO_SCENES")                      # the fixture put "0" there; unset is the default being asked about
    assert serve.run("127.0.0.1", 8198) == 0
    assert os.environ["SMRITILILA_NO_SCENES"] == "1" and kosha.no_scenes()
    monkeypatch.setenv("SMRITILILA_NO_SCENES", "0")
    assert serve.run("127.0.0.1", 8198) == 0
    assert os.environ["SMRITILILA_NO_SCENES"] == "0" and not kosha.no_scenes()   # what was chosen stands
    # dev_server.py starts a subprocess the moment it is imported, so it is read, not run: the one setdefault, before the server is spawned
    tree = ast.parse((SCRIPT.parent / "dev_server.py").read_text(encoding="utf-8"))
    sets = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "setdefault"
            and [getattr(a, "value", None) for a in n.args] == ["SMRITILILA_NO_SCENES", "1"]]
    spawn = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "Popen"]
    assert len(sets) == 1 and spawn and sets[0].lineno < spawn[0].lineno


def test_an_empty_setting_in_the_env_file_turns_the_prompt_log_off_and_a_path_turns_it_on(monkeypatch, tmp_path):
    """The README (2026-10-18) tells a reader the prompt log is on by default and that `SMRITILILA_TRAINING_DATA_LOG=` (empty) in `.env` turns it off. Each half is met in a
    process of its own, since the log's file handler is set once per process: with the path a line is written (so the probe can show the failure), with the empty value none is."""
    import os
    import subprocess
    import sys
    import types
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))
    monkeypatch.setenv("SMRITILILA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SMRITILILA_TRAINING_DATA_LOG", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SMRITILILA_TRAINING_DATA_LOG=\n", encoding="utf-8")
    serve.load_dotenv(env_file)
    assert serve.run("127.0.0.1", 8198) == 0
    assert os.environ["SMRITILILA_TRAINING_DATA_LOG"] == ""                                       # run() leaves an explicit empty choice alone
    code = ("from smritilila.training_data import log_example\n"
            "log_example(backend='b', tool_name=None, system='s', user='u', input_schema=None, output={'a': 1})\n")
    src = str(Path(__file__).resolve().parent.parent / "src")
    for value, written in (("", False), (str(tmp_path / "on.jsonl"), True)):
        child = {k: v for k, v in os.environ.items() if k != "SMRITILILA_TRAINING_DATA_LOG"}
        child.update({"SMRITILILA_TRAINING_DATA_LOG": value, "PYTHONPATH": src + os.pathsep + child.get("PYTHONPATH", "")})
        done = subprocess.run([sys.executable, "-c", code], env=child, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr[-400:]
        assert (tmp_path / "on.jsonl").exists() is written, (value, done.stderr[-200:])
        (tmp_path / "on.jsonl").unlink(missing_ok=True)
