# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""Tests the real local embedding backend. This loads an actual
sentence-transformers model (already cached locally after first use, no
network needed thereafter) -- slower than this project's other unit
tests, but a real model load, not something worth faking here specifically
since the whole point of choosing a local model was to make this testable
without an external API key.
"""

from __future__ import annotations

import math
import socket

import pytest

from smritilila.llm.embedder import SentenceTransformerEmbedder


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    # Module-scoped: the real model load is the slow part of this file --
    # share one instance across both tests rather than paying for it twice.
    return SentenceTransformerEmbedder()


def test_embed_returns_a_list_of_floats(embedder: SentenceTransformerEmbedder) -> None:
    vector = embedder.embed("the sword lies in the square")
    assert isinstance(vector, list)
    assert len(vector) == 384  # all-MiniLM-L6-v2's dimension
    assert all(isinstance(x, float) for x in vector)


def test_semantically_similar_text_scores_higher_than_dissimilar_text(embedder: SentenceTransformerEmbedder) -> None:
    query = embedder.embed("Mara draws her sword and readies for a fight.")
    similar = embedder.embed("Mara grips her blade, preparing to battle.")
    dissimilar = embedder.embed("The bakery smells of fresh bread this morning.")

    assert _cosine_similarity(query, similar) > _cosine_similarity(query, dissimilar)


def test_construction_makes_no_network_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    # Live investigation, 2026-09-08: this module's own docstring claims
    # "runs entirely on this machine, no external API key, no per-call
    # cost" -- traced at the socket level and found that wasn't actually
    # true (18 HEAD requests to huggingface.co per construction, just to
    # check locally-cached files hadn't changed). local_files_only=True
    # is supposed to make the docstring true again -- verified here the
    # same way the live investigation was: no socket.socket.connect call
    # at all during construction, not just "the requests looked idle."
    calls: list[object] = []
    original_connect = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", lambda self, address: calls.append(address))
    try:
        SentenceTransformerEmbedder()
    finally:
        socket.socket.connect = original_connect
    assert calls == []


def test_falls_back_to_a_network_fetch_when_the_model_is_not_yet_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, local_files_only: bool = False) -> None:
            calls.append(local_files_only)
            if local_files_only:
                raise OSError("not cached")

    import sentence_transformers

    # __init__ does `from sentence_transformers import SentenceTransformer`
    # itself (a deferred import) -- that resolves the module attribute
    # fresh on every call, so patching it here is enough; no need to
    # patch anything inside smritilila.llm.embedder directly.
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = SentenceTransformerEmbedder()

    assert calls == [True, False]  # tried local-only first, fell back without it
    assert isinstance(embedder._model, _FakeSentenceTransformer)
