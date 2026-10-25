# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Raw (prompt, completion) capture for constructing LoRA/fine-tuning
training data -- one JSON line per SUCCESSFUL generative-role call.

Opt-in via SMRITILILA_TRAINING_DATA_LOG (both launchers, scripts/dev_server.py and scripts/serve.py, set it by default), same lazy-file-handler pattern as
smritilila.sakshi, but deliberately a SEPARATE stream/file: a Narrator
prompt alone runs ~8k tokens, so folding this into sakshi's own
lightweight operational lines would bloat and pollute them (and the two
have different audiences -- ops/debugging vs. training data). The size
gap is not theoretical: on the day sakshi was named, the two streams
standing side by side were 12.8 MB and 105.8 MB.

Wired into each client's complete/complete_with_tool (llm/client.py's
OpenAICompatClient, llm/the first provider_provider.py, llm/the second model_provider.py) -- the one choke
point every generative call (the Narrator, the world builder, everything)
already passes through regardless of caller, so capturing there gets every role for free
without a second instrumentation point per role.

Captures are provider-agnostic (system/user/input_schema/output), not
any one vendor's own fine-tuning JSONL shape -- which shape is actually
needed depends on which vendor/tool trains the eventual LoRA, a
decision this capture point shouldn't have to anticipate. `user` is the
role's own prompt as sent, the same shape from either backend (the second model's
"(Respond as: ...)" suffix, its stand-in for a forced tool call, is left
out).

A call that errors, or comes back empty, is not logged -- but an example
is written the moment a completion arrives, before anything has parsed or
accepted it. So a completion that fails to parse (llm/client.py's
_complete_json) is already in the log when the rotation retries it
through the other client, and so is an answer a pulse later throws away
(log_discarded tags that one): not every line is one to imitate. A
logging failure must never affect the call it's instrumenting, same D5
tolerance as smritilila.sakshi.

Every line also carries `code_version` -- this project's own prompts/
schemas keep changing session to session, so an example captured under
one prompt version can silently misrepresent what the CURRENT prompt
would produce. Resolved by smritilila.sakshi, which is where it moved on
2026-09-22 when there was finally an instrument layer to own it: a fact
about the observed process, not about this one stream. Still skipped
entirely alongside the rest of this module's work when the feature is
off."""

from __future__ import annotations

import json
import logging
import os
import threading

from smritilila.sakshi import code_version, current_world, now, run_id

_logger = logging.getLogger("smritilila.training_data")
_configured = False
_configuring = threading.Lock()


def _configure() -> None:
    """Same latch race smritilila.sakshi had, and the same fix -- see its
    _configure for the measurement. It bites harder here: log_example
    returns early when no handler is attached, so a call crossing the
    window is not merely unlogged, its training example is thrown away.
    Measured 2026-09-22: eight concurrent calls captured ONE of eight.
    Generative calls genuinely overlap (the routes run genesis, bhuvana,
    expand and the deferred activation pulse through asyncio.to_thread),
    and a discarded example is corpus this project intends to fine-tune
    on, gone with no trace that it ever existed."""
    global _configured
    if _configured:
        return
    with _configuring:
        if _configured:
            return
        try:
            path = os.environ.get("SMRITILILA_TRAINING_DATA_LOG")
            if path:
                handler = logging.FileHandler(path, mode="a", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(message)s"))
                _logger.addHandler(handler)
                _logger.setLevel(logging.INFO)
                # Same as smritilila.sakshi's, and it matters more here: these
                # lines carry the FULL prompt, ~8k tokens of Narrator
                # context each, and every one of them was being printed
                # to the server's terminal as well as written to the
                # file.
                _logger.propagate = False
        except OSError:
            pass
        finally:
            _configured = True  # last, so nobody leaves before the handler is on


def log_example(
    *,
    backend: str,
    tool_name: str | None,
    system: str,
    user: str,
    input_schema: dict | None,
    output: object,
    model: str | None = None,
    reasoning: dict | None = None,
    pulse_id: str | None = None,
) -> None:
    """output is the raw parsed dict for a tool call, or the raw string
    for a plain completion -- whichever complete/complete_with_tool
    actually returned to its own caller, unmodified.

    Self-describing since 2026-09-23: which model and reasoning settings
    wrote it, and which pulse, run and world it belongs to, with a
    timestamp. Until then a line said only "the first provider" or "the second model", and
    finding one pulse's capture meant searching for its action text; the
    model could be learned only by joining to sakshi's llm.call."""
    try:
        _configure()
        if not _logger.handlers:
            return  # opt-in only -- skip building the (potentially large) payload entirely when unset
        _logger.info(
            json.dumps(
                {
                    "backend": backend,
                    "model": model,
                    "reasoning": reasoning,
                    "tool_name": tool_name,
                    "code_version": code_version(),
                    "run": run_id(),
                    "world": current_world(),
                    "pulse_id": pulse_id,
                    "when": now(),
                    "system": system,
                    "user": user,
                    "input_schema": input_schema,
                    "output": output,
                },
                default=str,
            )
        )
    except Exception:
        pass


def log_discarded(*, pulse_id: str, reason: str, detail: str | None = None) -> None:
    """A tag, appended when a pulse throws away an answer it already
    captured (2026-09-24, Marc: "tag it"). The example was written during
    the call, before anyone knew it would be discarded, and lines are
    never edited -- so the tag is its own line: every example for
    `pulse_id` written BEFORE this line was discarded, for `reason`. The
    discarded answer is kept, not deleted: beside its replacement it is a
    (wrong, corrected) pair. A tuner reading for what to imitate skips it.
    First use: the pulse's arrival rerun, whose first pass wrote a player out of
    a scene they were in."""
    try:
        _configure()
        if not _logger.handlers:
            return
        _logger.info(json.dumps({"discarded": True, "pulse_id": pulse_id, "reason": reason, "detail": detail,
                                 "code_version": code_version(), "run": run_id(), "world": current_world(), "when": now()},
                                default=str))
    except Exception:
        pass
