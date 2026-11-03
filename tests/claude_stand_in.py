"""Anthropic's Messages API on loopback, as CrewAI's own Anthropic provider calls it.

A test that runs a real crew on Claude needs CrewAI and its ``anthropic`` extra
and nothing else: every message answers at once with a final answer and the
usage a cached prompt reports, 100 plain input tokens, 1,000 written to the
prompt cache and 500 read from it. CrewAI counts those as one input of 1,600 and
keeps the two cache counts beside it, which is the arithmetic a run's usage has
to survive. The provider asks ``/v1/messages/count_tokens`` before each call in
the releases that preflight, and that answer is not usage.

A workforce's router asks its manager which unit owns a request; the stand-in
answers that question with the unit it was told to route to, so a dispatch
runs as it would against a model that chose.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

#: The model a crew is set to, as the runtime sets it (``OPENAI_MODEL_NAME``),
#: and the model id CrewAI then reports, the provider's prefix gone.
MODEL_SETTING = "anthropic/claude-sonnet-5-5"
MODEL_ID = "claude-sonnet-5-5"

#: Per message: what Anthropic's API answers, and what CrewAI counts of it.
USAGE = {"input_tokens": 100, "output_tokens": 300, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 500}
INPUT_PER_CALL = 1600
OUTPUT_PER_CALL = 300
WRITTEN_PER_CALL = 1000
READ_PER_CALL = 500

#: The question the workforce's router puts to its manager.
ROUTE_QUESTION = "Reply with exactly one unit key from:"

_ANSWER = "Thought: I now can give a great answer\nFinal Answer: {text}"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # quiet
        pass

    def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
        server: Any = self.server
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/count_tokens"):
            self._answer({"input_tokens": INPUT_PER_CALL})
            return
        with server.lock:
            server.messages += 1
        text = server.route_to if ROUTE_QUESTION in json.dumps(request) else "The work is done."
        reply = _ANSWER.format(text=text)
        model = request.get("model") or "unknown"
        if request.get("stream"):
            self._stream(model, reply)
            return
        self._answer(
            {
                "id": "msg_standin",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": reply}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": USAGE,
            }
        )

    def _answer(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _stream(self, model: str, reply: str) -> None:
        message = {
            "id": "msg_standin",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {**USAGE, "output_tokens": 1},
        }
        events = [
            ("message_start", {"type": "message_start", "message": message}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": reply}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": OUTPUT_PER_CALL},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for name, data in events:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())


class StandIn:
    """The running stand-in, and how many messages it has answered."""

    def __init__(self, server: ThreadingHTTPServer) -> None:
        self._server: Any = server

    @property
    def messages(self) -> int:
        with self._server.lock:
            return int(self._server.messages)


@contextmanager
def claude_on_loopback(monkeypatch: pytest.MonkeyPatch, home: Path, *, route_to: str = "none") -> Iterator[StandIn]:
    """Serve the stand-in and point this process's environment at it.

    The runner hands a run this environment, so a crew it starts calls the
    stand-in with a key that is not one. ``home`` becomes the run's home: CrewAI
    keeps every kickoff's task outputs in a database under the user's data
    folder, named after the working directory, which is "bundle" for every
    bundle these tests build, and another CrewAI's rows there failed a run with
    a ``JSONDecodeError``.
    """
    pytest.importorskip("crewai")
    pytest.importorskip("anthropic")
    server: Any = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.messages = 0
    server.lock = threading.Lock()
    server.route_to = route_to
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-standin")
        # The fixture crews declare OpenAI's key, as a marketplace crew does, and
        # the runner refuses a run whose declared credentials are unset; the run
        # itself never reads it, since its model is Claude.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL_NAME", MODEL_SETTING)
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
        monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
        monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
        yield StandIn(server)
    finally:
        server.shutdown()
        server.server_close()
