# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Providers: how the engine finds whatever answers (2026-10-17).

The engine speaks to a model through the LLMClient Protocol (llm/client.py) and knows no vendor. A provider is one way of reaching a model: how to
build its client, how to say whether it is configured, in what order it is asked. A provider is a module in this package named `<name>_provider.py`
that calls `register(Provider(...))` when it is imported; the first time anything asks, every such module found here is imported. The one provider
built in is `local`: a model served at an OpenAI-compatible endpoint (SMRITILILA_LOCAL_BASE_URL and SMRITILILA_LOCAL_MODEL).

To reach another backend, write a module that registers one; nothing else in the engine changes. A provider may also hand the engine, if its model
needs them:
  * `swaps`: prompt text for the Narrator in that model's own words ({model-name fragment: {shared text: its text}}, see smrti/narrator.system_for);
  * `seed_word`: a word in the model name of the one a world's seeds are written by first.

`make_narrator()` builds the client the narration is given (or raises KeyError / ValueError, or returns None, when it is not configured), and
`make_best(model)` the one for one-off judgements and the builder, where `model`, if given, names the model to use. `narrate_rank` is the order the
narration asks providers in when none is named first (SMRITILILA_PRIMARY_PROVIDER); `best_rank` the order for the one-off calls.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

from smritilila.llm.client import LLMClient, local_client_from_env
from smritilila.sakshi import log_event


@dataclass(frozen=True)
class Provider:
    name: str
    narrate_rank: int
    best_rank: int
    make_narrator: Callable[[], "LLMClient | None"]
    make_best: Callable[[str | None], "LLMClient | None"]
    describe: Callable[[], "str | None"]
    hint: str
    seed_word: str | None = None
    swaps: Callable[[], dict[str, dict[str, str]]] | None = None


_PROVIDERS: dict[str, Provider] = {}
_found = False


def register(provider: Provider) -> None:
    _PROVIDERS[provider.name] = provider


def all_providers() -> list[Provider]:
    """Every registered provider, the plugin modules in this package imported first (once). A plugin that fails to import is left out and said so."""
    global _found
    if not _found:
        _found = True
        import smritilila.llm as package
        for module in sorted(m.name for m in pkgutil.iter_modules(package.__path__) if m.name.endswith("_provider")):
            try:
                importlib.import_module(f"smritilila.llm.{module}")
            except Exception as exc:  # noqa: BLE001 -- a backend whose library is not installed must not take the engine down
                log_event("llm.provider_unavailable", module=module, error=repr(exc)[:200])
    return list(_PROVIDERS.values())


def narration_order(first: str | None = None) -> list[Provider]:
    """The providers in the order narration asks them: by `narrate_rank`, the one named first (SMRITILILA_PRIMARY_PROVIDER) ahead of the rest."""
    ordered = sorted(all_providers(), key=lambda p: p.narrate_rank)
    word = (first or "").strip().lower()
    named = [p for p in ordered if p.name == word]
    return named + [p for p in ordered if p.name != word] if named else ordered


def narrators(first: str | None = None) -> list[LLMClient]:
    """The clients for the narration that are configured, in order."""
    out: list[LLMClient] = []
    for provider in narration_order(first):
        try:
            client = provider.make_narrator()
        except (KeyError, ValueError):
            continue
        if client is not None:
            out.append(client)
    return out


def best(model: str | None = None) -> LLMClient:
    """The first configured provider, in `best_rank` order, for a one-off call. Raises KeyError if none is."""
    for provider in sorted(all_providers(), key=lambda p: p.best_rank):
        try:
            client = provider.make_best(model)
        except (KeyError, ValueError):
            continue
        if client is not None:
            return client
    raise KeyError(nothing_configured())


def configured() -> list[str]:
    """What each configured provider says it is, for `scripts/serve.py check`."""
    found = [(p.narrate_rank, p.describe()) for p in all_providers()]
    return [text for _, text in sorted(found, key=lambda t: t[0]) if text]


def nothing_configured() -> str:
    """The sentence for a server with no model: what to set, per provider, in narration order."""
    hints = [p.hint for p in sorted(all_providers(), key=lambda p: p.narrate_rank)]
    if len(hints) > 2:
        joined = ", ".join(hints[:-1]) + ", or " + hints[-1]
    else:
        joined = " or ".join(hints)
    return f"no LLM is configured: set {joined}"


def seed_word() -> str | None:
    """A word in the model name of the client a world's seeds are written by first, if a provider asks for one."""
    return next((p.seed_word for p in sorted(all_providers(), key=lambda p: p.narrate_rank) if p.seed_word), None)


def prompt_swaps() -> dict[str, dict[str, str]]:
    """The Narrator's prompt swaps every provider offers, by model-name fragment, in narration order."""
    out: dict[str, dict[str, str]] = {}
    for provider in sorted(all_providers(), key=lambda p: p.narrate_rank):
        if provider.swaps is not None:
            out.update(provider.swaps())
    return out


def _local_describe() -> str | None:
    import os
    base, model = os.environ.get("SMRITILILA_LOCAL_BASE_URL"), os.environ.get("SMRITILILA_LOCAL_MODEL")
    return f"a local model ({model} at {base})" if base and model else None


register(Provider(
    name="local", narrate_rank=2, best_rank=1, hint="SMRITILILA_LOCAL_BASE_URL and SMRITILILA_LOCAL_MODEL",
    make_narrator=local_client_from_env, make_best=lambda model: local_client_from_env(), describe=_local_describe,
))
