# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Local sentence-embedding backend: the embedding each tuple carries, which
smrti/context.py's retrieval compares. Runs entirely on this machine -- no external API key, no per-call
cost -- since the first provider has no embeddings API of its own and this
project's chosen tradeoff is a heavier local dependency over a second
hosted API key.

sentence-transformers (and its own dependency, torch) is a real, heavy
install and the model weights are a real download on first use -- the
import is deferred into __init__ so constructing anything else in this
module tree never pays that cost, and importing this specific class is
the only place that does.

Live investigation, 2026-09-08: "runs entirely on this machine" was not
actually true as shipped. Traced at the socket level -- every
construction (i.e. every fresh server process, not per-call) made 18
plain HEAD requests to huggingface.co, one per model component file,
purely to check the locally-cached files hadn't changed. No game data,
no player data, no credentials in them (just library/version User-Agent
strings and a random session id), but real, unwanted network traffic for
a module that promises none, on every process start, and >9s of the
~9.2s total construction cost -- construction with local_files_only=True
dropped to 140ms with zero outbound connections, confirmed by the same
socket-level trace. local_files_only=True is now unconditional: the
model this project pins by name (DEFAULT_MODEL_NAME) is expected to
already be cached from a prior run/deploy step, matching this module's
own "no per-call cost" contract taken literally -- a genuinely
never-downloaded model on a fresh machine raises OSError, caught below
and retried WITHOUT local_files_only exactly once (a real network fetch,
logged so it's visible rather than silently slow), so a first-ever
deploy still works and every run after it is fast and silent.
"""

from __future__ import annotations

import logging

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, small and fast; good default for short narration/memory text

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            logger.warning(
                "%r not found in the local cache -- fetching from Hugging Face Hub "
                "(one-time; every process after this one on this machine will be local-only).",
                model_name,
            )
            self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()
