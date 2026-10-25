# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""LLM clients: the seam between the engine and whatever answers (split from the hosted-service clients, 2026-10-17).

The engine speaks to a model only through the LLMClient Protocol below and knows no vendor. This module holds the Protocol; the way to get a
shape out of a model that cannot be forced to one (complete_json, JsonShapedClient); the OpenAI-compatible client, for a model served at
/v1/chat/completions (Ollama, vLLM, llama.cpp's server, LM Studio, a hosted service of that kind); RotatingClient, which sends a share of
calls to a second client and falls back to the other when one fails; Anurupa, a prompt fitted to the model that receives it; and the telemetry
every client shares (llm.call, one line per call, stamped with the pulse and the route).

Which clients exist is llm/providers.py's business. A backend is a module `<name>_provider.py` in this package that registers itself there, and
registers with `register_backend` how to read its responses, so that llm.call says how a call stopped and what it cost.
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import itertools
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from smritilila.sakshi import log_event
from smritilila.training_data import log_example

# Real, observed attribution gap (live investigation, 2026-09-14): llm.call
# telemetry carries no pulse_id at all, unlike every other pulse-scoped
# event -- a response-time deep dive tried to attribute individual calls to
# pulses by POSITION in telemetry.jsonl (between consecutive pulse.timing
# markers) and got it wrong for exactly the pulses where the PREVIOUS
# pulse's own deferred activation (routes/spanda.py's _run_smrti_activation_in_
# background, genuinely running on a separate thread via asyncio.to_thread)
# was still finishing: two threads' log lines interleaving in true
# execution order, not the clean sequential blocks positional attribution
# assumed. A ContextVar, not a new parameter threaded through every call
# site and everything that calls them: bound once, at the top of the
# pulse, and read here with zero signature changes anywhere in between.
# (Written 2026-09-14 against ~9 call sites across 8 role classes --
# Narrator, ActorCognition, WorldGeometry, TriggerEvaluator,
# MemoryInheritance, LateArrivalNarrator, UnresolvedSpeakerDetector,
# WorldBuilder -- and bound in spanda's run_pulse/run_activation_pulse.
# Seven of those roles went with v2 on 2026-09-21; GenerativeRoles is one
# client now, and the binding site is spanda.py. The argument for a
# ContextVar over a threaded parameter is unchanged, and so is every word
# that follows.) Correct across the asyncio.to_thread boundary
# specifically because contextvars.copy_context() (asyncio.to_thread's
# own documented mechanism) propagates the CALLING context into the new
# thread -- setting this before entering to_thread, or as the very first
# statement inside the threaded function itself, both work identically;
# each thread's own
# context is independent, so this can never leak between a player pulse
# on the main thread and a concurrently-running activation pulse on its
# own worker thread. None outside any bound scope (world seeding at
# genesis, a world's expansion) -- accurate,
# not a guess: those calls genuinely aren't part of any one pulse.
_pulse_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("smritilila_llm_pulse_id", default=None)
# Which door this call came through: "primary", "secondary" (a rotation
# sent it there on purpose) or "fallback" (the other one failed). Same
# ContextVar mechanism as the pulse id, and for the same reason -- the
# alternative is inferring it from the position of two event types in
# the record, which is the attribution this project has already been
# burned by. One field on llm.call answers it instead.
_route_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("smritilila_llm_route", default="primary")
# Shared by every RotatingClient in the process -- see RotatingClient._pick.
_rotation_n = itertools.count(1)
# the pick a caller planned for its next call (plan(), then the call): the model that will narrate is known before
# the prompt is built -- fortune's spread is chosen by it (2026-09-28)
_planned_ctx: contextvars.ContextVar = contextvars.ContextVar("smritilila_llm_planned", default=None)


def bind_pulse_id(pulse_id: str | None) -> None:
    """Call once, as early as possible in a pulse's own processing --
    every llm.call this
    context (or a thread spawned via asyncio.to_thread from within it)
    goes on to make will carry this pulse_id, with no further plumbing.
    No context-manager/reset pairing needed: each top-level pulse call
    always sets its own value before doing any work, and nothing ever
    reads a stale value BETWEEN distinct top-level calls in the same
    context -- the only failure mode a reset would guard against.
    (A per-pulse call count rode along here until 2026-09-30; nothing had
    read it since v2's call budget went.)"""
    _pulse_id_ctx.set(pulse_id)


def current_pulse_id() -> str | None:
    """The pulse whatever is running now belongs to, for code that wants
    to stamp a line and has no pulse_id threaded to it.

    Same binding as llm.call's own stamp, so a line stamped this way
    sorts with that pulse's calls and needs no plumbing of its own. It
    propagates into asyncio.to_thread, which is what makes it safe here:
    the alternative is attributing an event to a pulse by its POSITION
    in the record, and that is the mistake this ContextVar was added to
    end -- two threads' lines interleave in true execution order, not in
    clean per-pulse blocks (see this module's own note above).

    None outside a pulse, which is an honest answer and not a gap.
    """
    return _pulse_id_ctx.get()


# Seconds a hosted call may take before it is given up on. Raised from 60 on 2026-09-13 after a call for an 85-participant scene timed out: a
# populated scene is a long prompt, and whole pulses were already measured at 90-150 s. 180 is headroom over the one failure seen, not a measured
# floor; recalibrate against per-call durations if they are ever recorded.
DEFAULT_TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True)
class Picture:
    """An image shown to a model beside the prompt (a map the builder reads): its bytes, its media type and a name. A call that carries
    pictures is a call that cannot be answered without them, so a client that cannot take one says so (ImagesNotSupported) and is never
    handed the words alone."""
    data: bytes
    mime: str = "image/png"
    name: str = ""
    fine: bool = False      # small print to be read (a map's labels): the finest reading the model offers, at the cost of more of its attention

    def mark(self) -> str:
        """The picture as the prompt log keeps it: its name and a hash, never its bytes."""
        return f"[picture {self.name or 'unnamed'} {self.mime} sha256:{hashlib.sha256(self.data).hexdigest()[:12]} {len(self.data)} bytes]"


class ImagesNotSupported(RuntimeError):
    """The model that was asked takes words only."""


def logged_user(user: str, images: "Sequence[Picture] | None") -> str:
    """The user turn as the prompt log keeps it: the words, then a mark for each picture that came with them."""
    return user if not images else user + "\n" + "\n".join(p.mark() for p in images)


# -- complete_json: shape without tool-calling -------------------------
#
# The Narrator's contract is a nested schema -- arrays of objects with
# enums -- and the provider enforces it through tool-calling. That works
# on the frontier models and does NOT work on the open ones: measured
# 2026-09-22, both an open model sizes answer a flat two-field schema in ~3s
# and return 504 DEADLINE_EXCEEDED, server-side, on the real
# append_tuples schema. Since THE PORT's whole premise is running this
# engine on open weights, shape has to stop depending on the provider.
#
# So: ask for the JSON in the text, and parse it here. The engine is
# already built for this -- it tolerates misshapen rows today (unfold,
# rows(), smrti.malformed_tuple) because the Narrator has always
# occasionally returned something odd. This just moves the shape
# guarantee from the provider to us, where it can be the same on every
# backend.

JSON_INSTRUCTION = """
Answer with ONE JSON object and nothing else -- no prose before or after
it, no markdown fence, no commentary. It must match this schema:

{schema}

Every required field must be present. Do not add fields the schema does
not name."""


class Anurupa(str):
    """Anurūpa (अनुरूप, "fitting, according to form" -- the name Marc's,
    2026-09-26): a prompt fitted to whoever receives it. Per-model variants
    ("different length suggestions based on model called"). As a str it IS the shared
    default -- any caller, wrapper or fake that treats it as text sees
    exactly what it saw before. Only the leaf client that answers resolves
    it for its own model (resolve_for), so the rotation, which picks the
    model inside the call, gets each model its own prompt; a model with no
    variant is sent the default byte for byte (its cache untouched).
    `variants`: {model-name fragment: whole text}, first match wins."""

    def __new__(cls, default: str, variants: dict[str, str] | None = None) -> "Anurupa":
        self = super().__new__(cls, default)
        self.variants = dict(variants or {})
        return self


def resolve_for(system: str, model: str) -> tuple[str, str | None]:
    """The text to send `model`, and which variant it was: None for a plain
    str, "default" for a Anurupa with no variant for this model."""
    if not isinstance(system, Anurupa):
        return system, None
    for key, text in system.variants.items():
        if key in (model or "").lower():
            return text, key
    return str(system), "default"


def _prompt_field(prompt: str | None) -> dict:
    """llm.call's `prompt`: which of a Anurupa's variants was sent --
    absent for a plain prompt, so the record of every other role is as it
    was."""
    return {"prompt": prompt} if prompt else {}


def _json_instruction(schema: dict) -> str:
    return JSON_INSTRUCTION.format(schema=json.dumps(schema, indent=1))


def _complete_json(client, system: str, user: str, *, input_schema: dict, max_tokens: int, images: "Sequence[Picture] | None" = None) -> dict:
    """complete_json for any provider: the same answer complete_with_tool
    returns, without asking the provider to enforce the shape (see the
    note above LLMClient). Built on the client's own complete(), so there
    is no second copy of any provider's request code to drift.

    The schema goes on the END of the system prompt, where it is as
    static as the rest of it -- the cached prefix keeps caching."""
    system, variant = resolve_for(system, client.model)  # before the schema is added: a str + str is a plain str
    text = client.complete(system + _json_instruction(input_schema), user, max_tokens=max_tokens,
                           **({"prompt": variant} if variant else {}),  # a plain prompt: the call as it always was
                           **({"images": images} if images else {}))
    try:
        out = _parse_json_object(text)
    except Exception as exc:
        log_event("llm.json", pulse_id=_pulse_id_ctx.get(), model=client.model,
                  route=_route_ctx.get(), ok=False, error=repr(exc)[:200], chars=len(text or ""))
        raise
    log_event("llm.json", pulse_id=_pulse_id_ctx.get(), model=client.model,
              route=_route_ctx.get(), ok=True, chars=len(text or ""), fields=sorted(out))
    return out


def _parse_json_object(text: str) -> dict:
    """The model's text as one JSON object, tolerantly.

    Deliberately forgiving in the two ways models actually miss -- a
    ```json fence, and prose wrapped around the object -- and strict
    about everything else, because a silently half-parsed pulse is worse
    than a failed one that falls back. Scans for the outermost balanced
    braces rather than regexing, so a brace inside a narration string
    cannot truncate the object."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    start = raw.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON object in the completion (started {raw[:60]!r})")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(raw[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(raw[start:i + 1])
                if not isinstance(parsed, dict):
                    raise RuntimeError(f"completion parsed to {type(parsed).__name__}, not an object")
                return parsed
    raise RuntimeError("the JSON object never closed (truncated completion?)")


_JSON_TYPES = {"array": list, "object": dict}


def _unstring(out: object, schema: dict, tool_name: str, model: str) -> object:
    """A top-level field the schema declares an array or an object that came
    back as JSON text is decoded (2026-10-01: the mid-size model returned a seed's whole
    `places` array as a string in 3 of 12 draws, and the birth failed with
    "the seed names no place"). Only when the text parses to exactly the
    declared type; anything else is left as it came. What is decoded is
    counted (llm.unstringed)."""
    if not isinstance(out, dict):
        return out
    fixed = []
    for name, spec in (schema.get("properties") or {}).items():
        want = _JSON_TYPES.get(spec.get("type")) if isinstance(spec, dict) else None
        value = out.get(name)
        if want is None or not isinstance(value, str):
            continue
        try:
            decoded = json.loads(value)
        except ValueError:
            continue
        if isinstance(decoded, want):
            out = {**out, name: decoded}
            fixed.append(name)
    if fixed:
        log_event("llm.unstringed", pulse_id=_pulse_id_ctx.get(), model=model, tool_name=tool_name, fields=fixed)
    return out


class LLMClient(Protocol):
    """Each call may also take `images`, a sequence of Picture, shown to the model with the words (the builder reads a map this way). A client
    that cannot take them raises ImagesNotSupported; one given none is called as it always was."""

    def complete(self, system: str, user: str, *, max_tokens: int = 8192) -> str: ...

    def complete_with_tool(
        self,
        system: str,
        user: str,
        *,
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        max_tokens: int = 8192,
    ) -> dict: ...

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        input_schema: dict,
        max_tokens: int = 8192,
    ) -> dict: ...


# -- what a response says about how the call stopped and what it cost, per backend (llm.call telemetry)
#
# Each backend registers two readers for its own response shape. Both are best-effort and never raise: a missing figure is far cheaper than
# a telemetry bug taking down a pulse. Field names are deliberately NOT unified across backends: they report different things.
_BACKENDS: dict[str, tuple[Callable[[object], "str | None"], Callable[[object], dict]]] = {}


def register_backend(name: str, finish: Callable[[object], "str | None"], usage: Callable[[object], dict]) -> None:
    _BACKENDS[name] = (finish, usage)


def _finish_of(response: object | None, backend: str) -> str | None:
    """Why the model stopped, for llm.call (added 2026-09-22, from a rotated pulse whose JSON came back truncated: the record could say only THAT it
    stopped early, not why). Never raises."""
    if response is None or backend not in _BACKENDS:
        return None
    try:
        return _BACKENDS[backend][0](response)
    except Exception:
        return None


def _usage_fields(response: object | None, backend: str) -> dict[str, object]:
    """Best-effort token and cache usage for llm.call (added 2026-09-14). response=None (the call raised before returning one) or any unexpected
    shape gives {} and never raises."""
    if response is None or backend not in _BACKENDS:
        return {}
    try:
        return _BACKENDS[backend][1](response)
    except Exception:
        return {}


def _openai_finish(response) -> str | None:
    return str(((response.get("choices") or [{}])[0]).get("finish_reason") or "") or None


def _openai_usage(response) -> dict[str, object]:
    usage = response.get("usage") or {}
    return {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


register_backend("openai", _openai_finish, _openai_usage)


LOCAL_TIMEOUT_SECONDS = 600.0   # a local model on modest hardware is slow: minutes for a populated scene, not seconds


class OpenAICompatClient:
    """A model served at an OpenAI-compatible endpoint (2026-10-06, for a friend who runs local models): Ollama, vLLM, llama.cpp's
    server, LM Studio, or any `/v1/chat/completions`. No new dependency: the standard library speaks to it.

    It does plain completion, and answers `complete_json` as text JSON built on its own `complete` (the one path the engine has for a
    model that cannot be forced to a shape; see `_complete_json`). Wrap it in `JsonShapedClient` to sit where a tool-calling model
    sits: `local_client_from_env` does. An open model's "thinking" (`<think>...</think>`) is dropped from the answer.

        SMRITILILA_LOCAL_BASE_URL   e.g. http://localhost:11434/v1 (Ollama), http://localhost:8000/v1 (vLLM)
        SMRITILILA_LOCAL_MODEL      the model's name at that server
        SMRITILILA_LOCAL_API_KEY    only if the server asks for one
        SMRITILILA_LOCAL_TIMEOUT    seconds a call may take (default 600)"""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = LOCAL_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._timeout = timeout

    def _post(self, body: dict) -> dict:
        import urllib.request
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    @staticmethod
    def _user_content(user: str, images: "Sequence[Picture] | None"):
        """The user turn: the words alone, or the words and each picture as a data URI (the form this endpoint takes an image in). A model
        that is not a vision model answers such a request with an error of its own, which is the right answer."""
        if not images:
            return user
        return [{"type": "text", "text": user},
                *({"type": "image_url", "image_url": {"url": f"data:{p.mime};base64,{base64.b64encode(p.data).decode('ascii')}",
                                                      **({"detail": "high"} if p.fine else {})}} for p in images)]

    def complete(self, system: str, user: str, *, max_tokens: int = 8192, prompt: str | None = None, images: "Sequence[Picture] | None" = None) -> str:
        system, variant = resolve_for(system, self.model)
        prompt = prompt or variant
        _start = time.monotonic()
        _error: str | None = None
        response = None
        try:
            response = self._post({"model": self.model, "max_tokens": max_tokens, "stream": False,
                                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": self._user_content(user, images)}]})
            text = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if not text:
                raise RuntimeError(f"empty completion from model (finish_reason={_finish_of(response, 'openai')!r})")
            log_example(backend="openai", tool_name=None, system=system, user=logged_user(user, images), input_schema=None, output=text,
                        model=self.model, reasoning={}, pulse_id=_pulse_id_ctx.get())
            return text
        except Exception as exc:
            _error = type(exc).__name__
            raise
        finally:
            log_event("llm.call", backend="openai", model=self.model, reasoning={}, route=_route_ctx.get(), finish=_finish_of(response, "openai"),
                      method="complete", duration_ms=int((time.monotonic() - _start) * 1000), error=_error, pulse_id=_pulse_id_ctx.get(),
                      **_prompt_field(prompt), **_usage_fields(response, "openai"))

    def complete_json(self, system: str, user: str, *, input_schema: dict, max_tokens: int = 8192, images: "Sequence[Picture] | None" = None) -> dict:
        return _complete_json(self, system, user, input_schema=input_schema, max_tokens=max_tokens, images=images)

    def complete_with_tool(self, system: str, user: str, *, tool_name: str, tool_description: str, input_schema: dict, max_tokens: int = 8192,
                           images: "Sequence[Picture] | None" = None) -> dict:
        # a local model is not asked for a forced tool call: the schema goes in the prompt and the answer is parsed (see JsonShapedClient)
        return self.complete_json(system, user, input_schema=input_schema, max_tokens=max_tokens, images=images)


def local_client_from_env() -> "LLMClient | None":
    """The local model the environment names (SMRITILILA_LOCAL_BASE_URL and SMRITILILA_LOCAL_MODEL), or None if it names none."""
    base, model = (os.environ.get("SMRITILILA_LOCAL_BASE_URL") or "").strip(), (os.environ.get("SMRITILILA_LOCAL_MODEL") or "").strip()
    if not (base and model):
        return None
    try:
        timeout = float(os.environ.get("SMRITILILA_LOCAL_TIMEOUT") or LOCAL_TIMEOUT_SECONDS)
    except ValueError:
        timeout = LOCAL_TIMEOUT_SECONDS
    return JsonShapedClient(OpenAICompatClient(base, model, api_key=os.environ.get("SMRITILILA_LOCAL_API_KEY") or None, timeout=timeout))

def best_available_client(model: str | None = None) -> "LLMClient":
    """The best model this machine is configured with, for the one-off judgements and the builder (not the narration's rotation): the first
    registered provider, in its own order of preference for these calls, that is configured (`model`, if given, names the model it uses; a
    provider that is not configured is passed over). Raises KeyError if there is none."""
    from smritilila.llm import providers
    return providers.best(model)


class JsonShapedClient:
    """Presents complete_with_tool over a model that cannot do it.

    Every caller in the engine asks for `complete_with_tool`, because
    that is how shape has always been guaranteed. An open model cannot
    answer it: measured 2026-09-22, both an open model sizes return 504
    DEADLINE_EXCEEDED on the real append_tuples schema through
    tool-calling, and answer the same schema as text JSON in 11-34s.
    This wrapper routes the one to the other, so an open model can sit
    anywhere a client sits without a single call site knowing.

    Composition rather than a flag on the second modelClient on purpose: "this
    model cannot do tool-calling" is a fact about the model, not a mode
    of the provider, and the next open model to go in the secondary slot
    will need the same wrapper whoever serves it.

    `model` is forwarded so llm.call still reports which model actually
    answered -- the wrapper must be invisible in the record except for
    the shape of the request that got there."""

    def __init__(self, inner: "LLMClient") -> None:
        self._inner = inner

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "unknown")

    def complete(self, system: str, user: str, *, max_tokens: int = 8192, images: "Sequence[Picture] | None" = None) -> str:
        return self._inner.complete(system, user, max_tokens=max_tokens, **({"images": images} if images else {}))

    def complete_json(self, *args, **kwargs) -> dict:
        return self._inner.complete_json(*args, **kwargs)

    def complete_with_tool(
        self,
        system: str,
        user: str,
        *,
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        max_tokens: int = 8192,
        images: "Sequence[Picture] | None" = None,
    ) -> dict:
        # tool_name and tool_description are the provider's framing for a
        # shape this path states in the prompt instead; the schema is the
        # whole of what matters and it carries over unchanged.
        return self._inner.complete_json(
            system, user, input_schema=input_schema, max_tokens=max_tokens, **({"images": images} if images else {}),
        )


class RotatingClient:
    """Sends a SHARE of calls to the secondary on purpose, and falls back
    either way (2026-09-22, Marc: "let's rotate calls between the second model and
    the first provider and make sure we have instrumentation").

    Why rotate rather than keep the secondary in reserve: a recourse path
    that is never exercised is a recourse path nobody knows is broken.
    The the second model secondary was pinned to a preview id for months and the
    obvious stable replacement turned out to be 404 for new users -- both
    facts only surfaced because someone called them. Rotating means the
    secondary is proven continuously, on real prompts, instead of on the
    afternoon the primary fails. It is also the dress rehearsal for the
    port: put an open-weight model here (an open model runs on the the second model API
    and now works, see the second modelClient.thinking) and every rotated call is
    evidence the engine does not need a frontier model.

    `share` is the fraction of calls the secondary gets, by deterministic
    round-robin rather than a coin flip -- so a run is reproducible and
    "one in five" means one in five, not five in a row by chance. 0.0
    disables rotation: the primary every time, the secondary only when
    it fails (never the other way back -- the primary had its shot).

    The cost of rotating is that consecutive pulses can be narrated by
    different models, and their prose does not match. v3 makes one
    Narrator call per pulse, so `share` IS the fraction of pulses in
    another voice. Keep it low for a world being played; raise it when
    the point is comparison.

    Every call is stamped `route` on llm.call: "primary", "secondary"
    (rotation sent it there) or "fallback" (the other one failed), so
    the two are never confused in the record."""

    def __init__(self, primary: "LLMClient", secondary: "LLMClient", share: float = 0.0) -> None:
        self._primary = primary
        self._secondary = secondary
        self._share = max(0.0, min(1.0, share))

    def _pick(self) -> tuple["LLMClient", "LLMClient", str]:
        """Round-robin by accumulated share: the secondary's turn comes
        when its running claim crosses the next whole call.

        The count is the PROCESS's, not this instance's (fixed
        2026-09-22, same evening it was written). adhisthana.py's
        _resolve_roles builds a fresh RotatingClient on every request, so
        an instance counter reset to 1 each time and, at share 0.1,
        int(1 * 0.1) never exceeded int(0 * 0.1) -- the secondary's turn
        never came and ten live pulses rotated zero times. It passed
        verification because the probe held ONE client across ten calls,
        which is not what the server does. A rotation policy is a
        property of the process; so is its counter.

        itertools.count is atomic in CPython, which matters because
        activation pulses run on their own threads."""
        n = next(_rotation_n)
        if self._share and int(n * self._share) > int((n - 1) * self._share):
            return self._secondary, self._primary, "secondary"
        return self._primary, self._secondary, "primary"

    def prefer(self, word: str) -> "RotatingClient":
        """The same two models, the one whose name holds `word` asked first and
        the other only when it fails -- no rotation. For calls one model is
        better at (Marc, 2026-10-01: births go to the second model first -- replayed,
        the second model wrote a seed's nature on every completed draw; the mid-size model's seed
        call broke on some briefs, `places` coming back as text or markup)."""
        if word in (getattr(self._secondary, "model", "") or ""):
            return RotatingClient(self._secondary, self._primary, 0.0)
        return RotatingClient(self._primary, self._secondary, 0.0)

    def plan(self) -> str | None:
        """Pick the next call's model now and hold it for that call: the name of the model that will answer."""
        pick = self._pick()
        _planned_ctx.set(pick)
        return getattr(pick[0], "model", None)

    def _run(self, method: str, *args, **kwargs):
        planned = _planned_ctx.get()
        if planned is not None:
            _planned_ctx.set(None)
        first, other, route = planned if planned is not None else self._pick()
        token = _route_ctx.set(route)
        try:
            return getattr(first, method)(*args, **kwargs)
        except Exception as exc:
            print(f"RotatingClient: {route} failed ({exc!r}), retrying via the other", flush=True)
            log_event("llm.rotation_fallback", pulse_id=_pulse_id_ctx.get(), tried=route, error=repr(exc)[:200])
            _route_ctx.set("fallback")
            return getattr(other, method)(*args, **kwargs)
        finally:
            _route_ctx.reset(token)

    def complete_json(self, *args, **kwargs) -> dict:
        return self._run("complete_json", *args, **kwargs)

    def complete(self, *args, **kwargs) -> str:
        return self._run("complete", *args, **kwargs)

    def complete_with_tool(self, *args, **kwargs) -> dict:
        return self._run("complete_with_tool", *args, **kwargs)
