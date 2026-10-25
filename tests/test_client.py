# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The generic LLM client (llm/client.py): rotation, the shape-from-text path (JsonShapedClient), the pulse id a call is stamped with, usage fields."""

from __future__ import annotations

import pytest

from smritilila.llm.client import RotatingClient, _pulse_id_ctx, _usage_fields, bind_pulse_id


class _FakeClient:
    def __init__(self, *, fail: bool = False, text: str = "ok", tool_output: dict | None = None):
        self.fail = fail
        self.text = text
        self.tool_output = tool_output if tool_output is not None else {"k": "v"}
        self.complete_calls = 0
        self.complete_with_tool_calls = 0

    def complete(self, system, user, *, max_tokens=8192):
        self.complete_calls += 1
        if self.fail:
            raise RuntimeError("primary is down")
        return self.text

    def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192):
        self.complete_with_tool_calls += 1
        if self.fail:
            raise RuntimeError("primary is down")
        return self.tool_output


def test_complete_uses_primary_when_it_succeeds() -> None:
    primary = _FakeClient(text="from primary")
    secondary = _FakeClient(text="from secondary")
    client = RotatingClient(primary=primary, secondary=secondary, share=0.0)

    assert client.complete("sys", "user") == "from primary"
    assert secondary.complete_calls == 0


def test_complete_falls_back_to_secondary_when_primary_raises() -> None:
    primary = _FakeClient(fail=True)
    secondary = _FakeClient(text="from secondary")
    client = RotatingClient(primary=primary, secondary=secondary, share=0.0)

    assert client.complete("sys", "user") == "from secondary"
    assert secondary.complete_calls == 1


def test_complete_propagates_when_both_fail() -> None:
    primary = _FakeClient(fail=True)
    secondary = _FakeClient(fail=True)
    client = RotatingClient(primary=primary, secondary=secondary, share=0.0)

    with pytest.raises(RuntimeError, match="primary is down"):
        client.complete("sys", "user")


def test_complete_with_tool_uses_primary_when_it_succeeds() -> None:
    primary = _FakeClient(tool_output={"from": "primary"})
    secondary = _FakeClient(tool_output={"from": "secondary"})
    client = RotatingClient(primary=primary, secondary=secondary, share=0.0)

    result = client.complete_with_tool("sys", "user", tool_name="t", tool_description="d", input_schema={})
    assert result == {"from": "primary"}
    assert secondary.complete_with_tool_calls == 0


def test_complete_with_tool_falls_back_to_secondary_when_primary_raises() -> None:
    primary = _FakeClient(fail=True)
    secondary = _FakeClient(tool_output={"from": "secondary"})
    client = RotatingClient(primary=primary, secondary=secondary, share=0.0)

    result = client.complete_with_tool("sys", "user", tool_name="t", tool_description="d", input_schema={})
    assert result == {"from": "secondary"}
    assert secondary.complete_with_tool_calls == 1


def test_secondary_failing_does_not_fall_back_to_primary_again() -> None:
    # Not a retry loop -- the primary only ever gets one attempt per call.
    primary = _FakeClient(fail=True)
    secondary = _FakeClient(fail=True)
    client = RotatingClient(primary=primary, secondary=secondary, share=0.0)

    with pytest.raises(RuntimeError):
        client.complete("sys", "user")
    assert primary.complete_calls == 1
    assert secondary.complete_calls == 1


def test_bind_pulse_id_propagates_into_an_asyncio_to_thread_call() -> None:
    # The whole design rests on this: _run_one_activation
    # (routes/spanda.py) dispatches run_smrti_pulse via asyncio.to_thread, and
    # contextvars.copy_context() propagation is what lets bind_pulse_id's
    # value survive the hop onto that worker thread with no plumbing.
    import asyncio

    seen: list[str | None] = []

    def _read_pulse_id_from_ctx_var() -> None:
        seen.append(_pulse_id_ctx.get())

    async def _run() -> None:
        bind_pulse_id("p_thread")
        await asyncio.to_thread(_read_pulse_id_from_ctx_var)

    asyncio.run(_run())

    assert seen == ["p_thread"]


def test_usage_fields_returns_empty_for_a_none_response() -> None:
    assert _usage_fields(None, "the first provider") == {}
    assert _usage_fields(None, "the second model") == {}


def test_usage_fields_swallows_an_unexpected_response_shape() -> None:
    # A response object with no usage attribute at all (an SDK shape this
    # module wasn't written against) must degrade to {}, never raise --
    # sakshi's usage extraction must never mask the real completion or error
    # already being handled by its own caller.
    class _BareResponse:
        pass

    assert _usage_fields(_BareResponse(), "the first provider") == {}
    assert _usage_fields(_BareResponse(), "the second model") == {}


def test_rotation_sends_exactly_the_configured_share_to_the_secondary() -> None:
    """Deterministic round-robin, not a coin flip: "one in five" must
    mean one in five, not five in a row by chance. A run has to be
    reproducible for the comparison to mean anything."""
    from smritilila.llm.client import RotatingClient

    class Named:
        def __init__(self, name): self.name = name
        def complete(self, *a, **k): return self.name
        def complete_with_tool(self, *a, **k): return {"who": self.name}

    for share, n in ((0.2, 100), (0.5, 100), (0.0, 20), (1.0, 20)):
        c = RotatingClient(primary=Named("A"), secondary=Named("B"), share=share)
        served = [c.complete("s", "u") for _ in range(n)]
        assert served.count("B") == int(n * share), (share, served.count("B"))


def test_rotation_falls_back_whichever_way_it_was_headed(monkeypatch) -> None:
    """The secondary's turn is still a turn that can fail. Rotation must
    not remove the recourse it exists to exercise."""
    from smritilila.llm.client import RotatingClient
    events = []
    monkeypatch.setattr("smritilila.llm.client.log_event", lambda e, **f: events.append((e, f)))

    class Dead:
        def complete(self, *a, **k): raise RuntimeError("down")
        def complete_with_tool(self, *a, **k): raise RuntimeError("down")

    class Live:
        def complete(self, *a, **k): return "served"
        def complete_with_tool(self, *a, **k): return {"ok": True}

    # every call rotates to the secondary, and the secondary is dead
    assert RotatingClient(primary=Live(), secondary=Dead(), share=1.0).complete("s", "u") == "served"
    assert [f["tried"] for e, f in events if e == "llm.rotation_fallback"] == ["secondary"]
    events.clear()
    # and the ordinary direction still works
    assert RotatingClient(primary=Dead(), secondary=Live(), share=0.0).complete("s", "u") == "served"
    assert [f["tried"] for e, f in events if e == "llm.rotation_fallback"] == ["primary"]


def test_complete_json_falls_back_and_rotates_like_the_other_calls() -> None:
    """It must be interchangeable at the call site, or the port path is
    a second code path that drifts from the one in use."""
    class Dead:
        def complete_json(self, *a, **k): raise RuntimeError("down")
    class Live:
        def __init__(self, name): self.name = name
        def complete_json(self, *a, **k): return {"served_by": self.name}

    assert RotatingClient(primary=Dead(), secondary=Live("B")).complete_json(
        "s", "u", input_schema={}) == {"served_by": "B"}
    r = RotatingClient(primary=Live("A"), secondary=Live("B"), share=0.5)
    served = [r.complete_json("s", "u", input_schema={})["served_by"] for _ in range(4)]
    assert served in (["A", "B", "A", "B"], ["B", "A", "B", "A"])  # the count is the process's: the phase depends on calls before


def test_a_model_that_cannot_tool_call_still_answers_complete_with_tool() -> None:
    """JsonShapedClient is what lets an open model sit in a slot every
    caller addresses with complete_with_tool. The wrapper must be
    invisible except for the shape of the request that reaches the
    model -- including `model`, so llm.call still names who answered."""
    from smritilila.llm.client import JsonShapedClient

    class OpenModel:
        model = "model-c"
        def __init__(self): self.seen = {}
        def complete_json(self, system, user, *, input_schema, max_tokens=8192):
            self.seen = {"schema": input_schema, "max_tokens": max_tokens}
            return {"narration": "x", "actor_stands": "hollow"}
        def complete_with_tool(self, *a, **k):
            raise AssertionError("the wrapper must never call tool-use on a model that cannot do it")

    inner = OpenModel()
    shaped = JsonShapedClient(inner)
    out = shaped.complete_with_tool(
        "RULES", "ACTION: bank the ashes", tool_name="append_tuples",
        tool_description="ignored by this path", input_schema={"type": "object"}, max_tokens=4096)

    assert out == {"narration": "x", "actor_stands": "hollow"}
    assert inner.seen["schema"] == {"type": "object"}   # the schema carries over unchanged
    assert inner.seen["max_tokens"] == 4096
    assert shaped.model == "model-c"          # the record still names the real model


def test_rotation_survives_a_fresh_client_per_request() -> None:
    """The bug this exists to prevent (2026-09-22): adhisthana.py's
    _resolve_roles builds a NEW RotatingClient on every request, so an
    instance counter reset to 1 each time and at share 0.1 the
    secondary's turn never came -- ten live pulses rotated zero times.
    It passed its first test because that test held ONE client across
    ten calls, which is not what the server does. So: build it the way
    the server builds it."""
    from smritilila.llm.client import RotatingClient

    class Named:
        def __init__(self, name): self.name = name
        def complete(self, *a, **k): return self.name

    def one_request():                      # exactly what _resolve_roles does
        return RotatingClient(primary=Named("A"), secondary=Named("B"), share=0.1)

    served = [one_request().complete("s", "u") for _ in range(40)]
    assert served.count("B") == 4, served   # one in ten, across forty separate clients


def test_a_client_preferring_a_word_asks_the_model_holding_it_first_every_time_and_the_other_only_when_it_fails() -> None:
    """Marc, 2026-10-01: births go to one model first (a provider names the word, llm/providers.seed_word). No rotation: the preferred one every call."""
    plain, preferred = _FakeClient(tool_output={"from": "plain"}), _FakeClient(tool_output={"from": "preferred"})
    plain.model, preferred.model = "model-one", "model-two"
    seeder = RotatingClient(primary=plain, secondary=preferred, share=0.8).prefer("two")
    for _ in range(5):
        assert seeder.complete_with_tool("s", "u", tool_name="t", tool_description="d", input_schema={}) == {"from": "preferred"}
    assert plain.complete_with_tool_calls == 0
    preferred.fail = True
    assert seeder.complete_with_tool("s", "u", tool_name="t", tool_description="d", input_schema={}) == {"from": "plain"}
    # a pair with no such word in it keeps its own order
    assert RotatingClient(primary=plain, secondary=_FakeClient(), share=0.8).prefer("two")._share == 0.0
