# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""A model served locally (2026-10-06): any OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp, LM Studio), for a friend who runs open
models on their own machine. The server here is a stub that records what it is sent and answers as one does."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import HTTPException

from smritilila import adhisthana
from smritilila.llm.client import JsonShapedClient, OpenAICompatClient, RotatingClient, best_available_client, local_client_from_env


class _Server:
    def __init__(self):
        self.requests, self.answer, self.status = [], "hello from a local model", 200
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if outer.status != 200:
                    self.wfile.write(b'{"error":"boom"}')
                    return
                self.wfile.write(json.dumps({"choices": [{"message": {"content": outer.answer}, "finish_reason": "stop"}],
                                             "usage": {"prompt_tokens": 11, "completion_tokens": 7}}).encode("utf-8"))

            def log_message(self, *a):
                pass
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture()
def server():
    s = _Server()
    yield s
    s.httpd.shutdown()


@pytest.fixture()
def clean_env(monkeypatch):
    import os
    for k in [k for k in os.environ if k.endswith("_API_KEY")] + ["SMRITILILA_LOCAL_BASE_URL", "SMRITILILA_LOCAL_MODEL", "SMRITILILA_LOCAL_API_KEY",
                                                                 "SMRITILILA_LOCAL_TIMEOUT", "SMRITILILA_PRIMARY_PROVIDER", "SMRITILILA_ROTATE_SHARE"]:
        monkeypatch.delenv(k, raising=False)       # every provider's key, whatever it is called: a machine's own must not change what these tests find


def test_a_completion_goes_to_chat_completions_with_the_system_and_user_and_comes_back_as_text(server):
    c = OpenAICompatClient(server.url, "a-model:8b", api_key="sekret")
    assert c.complete("You are the Narrator.", "The door opens.", max_tokens=300) == "hello from a local model"
    req = server.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sekret"
    assert req["body"]["model"] == "a-model:8b" and req["body"]["max_tokens"] == 300 and req["body"]["stream"] is False
    assert req["body"]["messages"] == [{"role": "system", "content": "You are the Narrator."}, {"role": "user", "content": "The door opens."}]


def test_a_picture_goes_with_the_words_as_a_data_uri_and_a_call_without_one_is_as_it_was(server):
    """Marc, 2026-10-21: the builder reads a map. A picture rides the user turn as an image_url data URI (the form this endpoint takes one in);
    a call with none sends its words as a plain string, byte for byte as before; the prompt log keeps a mark, never the bytes."""
    import base64
    from smritilila.llm.client import Picture, logged_user
    pic = Picture(b"\x89PNG-bytes", "image/png", "a map.png")
    c = OpenAICompatClient(server.url, "m")
    c.complete("s", "u")
    assert server.requests[0]["body"]["messages"][1]["content"] == "u"
    c.complete("s", "look at this", images=[pic])
    content = server.requests[1]["body"]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "look at this"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"\x89PNG-bytes").decode()}}
    server.answer = '{"x": 1}'
    assert JsonShapedClient(c).complete_with_tool("s", "u", tool_name="t", tool_description="d", input_schema={"type": "object"}, images=[pic]) == {"x": 1}
    assert isinstance(server.requests[2]["body"]["messages"][1]["content"], list)         # through the wrapper and the JSON path, the picture still goes
    from dataclasses import replace
    c.complete("s", "look", images=[replace(pic, fine=True)])
    assert server.requests[3]["body"]["messages"][1]["content"][1]["image_url"]["detail"] == "high" and "detail" not in content[1]["image_url"]
    mark = logged_user("words", [pic])
    assert mark.startswith("words\n[picture a map.png image/png sha256:") and "PNG-bytes" not in mark and logged_user("words", None) == "words"


def test_a_picture_reaches_a_client_that_takes_none_as_an_error_not_as_words_alone(monkeypatch):
    """A call that carries a picture cannot be answered without it: a model that takes words only says so (ImagesNotSupported), and the rotation's
    fallback does not quietly send the other model the words alone."""
    from smritilila.llm.client import ImagesNotSupported, Picture

    class Words:
        model = "words-only"
        def complete(self, system, user, *, max_tokens=8192, images=None):
            if images:
                raise ImagesNotSupported("no")
            return "ok"
        def complete_with_tool(self, system, user, *, images=None, **k):
            if images:
                raise ImagesNotSupported("no")
            return {}

    class Sees(Words):
        model = "sees"
        def complete_with_tool(self, system, user, *, tool_name, tool_description, input_schema, max_tokens=8192, images=None):
            raise RuntimeError("down")

    pic = Picture(b"x")
    rot = RotatingClient(Sees(), Words(), 0.0)
    with pytest.raises(ImagesNotSupported):
        rot.complete_with_tool("s", "u", tool_name="t", tool_description="d", input_schema={}, images=[pic])


def test_a_server_that_asks_for_no_key_is_sent_none(server):
    OpenAICompatClient(server.url, "m").complete("s", "u")
    assert server.requests[0]["auth"] is None


def test_a_models_thinking_is_dropped_from_the_answer(server):
    server.answer = "<think>she probably expects a lantern\nand a draught</think>\nA lantern gutters in the draught."
    assert OpenAICompatClient(server.url, "m").complete("s", "u") == "A lantern gutters in the draught."


def test_the_tool_call_is_answered_as_json_the_schema_in_the_prompt_and_the_fence_forgiven(server):
    server.answer = '```json\n{"narration": "The door opens.", "elapsed_minutes": 2}\n```'
    schema = {"type": "object", "properties": {"narration": {"type": "string"}, "elapsed_minutes": {"type": "integer"}}, "required": ["narration"]}
    out = OpenAICompatClient(server.url, "m").complete_with_tool("You narrate.", "He knocks.", tool_name="narrate", tool_description="tell it",
                                                                 input_schema=schema)
    assert out == {"narration": "The door opens.", "elapsed_minutes": 2}
    sent = server.requests[0]["body"]["messages"][0]["content"]
    assert "You narrate." in sent and '"elapsed_minutes"' in sent                      # the shape travels in the system prompt


def test_an_empty_completion_and_a_server_error_are_errors_not_silence(server):
    server.answer = "<think>nothing to say</think>"
    with pytest.raises(RuntimeError, match="empty completion"):
        OpenAICompatClient(server.url, "m").complete("s", "u")
    server.status = 500
    with pytest.raises(Exception):
        OpenAICompatClient(server.url, "m").complete("s", "u")


def test_the_environment_names_the_local_model_or_none(monkeypatch, clean_env, server):
    assert local_client_from_env() is None
    monkeypatch.setenv("SMRITILILA_LOCAL_BASE_URL", server.url)
    assert local_client_from_env() is None                                            # a URL alone names no model
    monkeypatch.setenv("SMRITILILA_LOCAL_MODEL", "another-model")
    c = local_client_from_env()
    assert isinstance(c, JsonShapedClient) and c.model == "another-model"                  # wrapped: it sits where a tool-calling model sits
    monkeypatch.setenv("SMRITILILA_LOCAL_TIMEOUT", "banana")
    assert local_client_from_env() is not None                                        # a nonsense timeout falls back, never a crash


def test_a_machine_with_only_a_local_model_narrates_with_it_alone(monkeypatch, clean_env, server):
    monkeypatch.setenv("SMRITILILA_LOCAL_BASE_URL", server.url)
    monkeypatch.setenv("SMRITILILA_LOCAL_MODEL", "m")
    c = adhisthana._resolve_roles().smrti_client
    assert isinstance(c, JsonShapedClient) and not isinstance(c, RotatingClient)


def test_with_nothing_configured_the_error_says_what_to_set(clean_env):
    with pytest.raises(HTTPException) as e:
        adhisthana._resolve_roles()
    assert e.value.status_code == 503 and "SMRITILILA_LOCAL_BASE_URL" in e.value.detail and "SMRITILILA_LOCAL_MODEL" in e.value.detail


def test_the_one_off_judgements_and_the_builder_use_the_local_model_when_it_is_all_there_is(monkeypatch, clean_env, server):
    monkeypatch.setenv("SMRITILILA_LOCAL_BASE_URL", server.url)
    monkeypatch.setenv("SMRITILILA_LOCAL_MODEL", "m")
    assert best_available_client().model == "m"
    from smritilila.routes import sthapati
    assert sthapati._builder_client().model == "m"


def test_the_llm_call_is_recorded_with_the_local_backend_and_its_tokens(monkeypatch, server):
    from smritilila.llm import client as llm
    events = []
    monkeypatch.setattr(llm, "log_event", lambda name, **kw: events.append((name, kw)))
    OpenAICompatClient(server.url, "m").complete("s", "u")
    call = next(kw for name, kw in events if name == "llm.call")
    assert call["backend"] == "openai" and call["model"] == "m" and call["finish"] == "stop" and call["error"] is None
    assert call["input_tokens"] == 11 and call["output_tokens"] == 7
