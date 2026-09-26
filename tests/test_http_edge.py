"""The runner's raw HTTP boundary, driven over a bare socket.

``http.client`` opens one connection per request and never sends what these
send: a request line ``http.server`` cannot parse, a body refused and left on
the wire, a length written ``1_3``, a connection that goes quiet. The suite was
green through all four defects the review found here, which is why this file's
client is a socket and its reader is written out by hand.
"""

from __future__ import annotations

import json
import logging
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from sigrix_runtime.postern import entitlement as ent  # noqa: E402
from sigrix_runtime.postern import server as server_module  # noqa: E402
from sigrix_runtime.postern.server import PosternServer, Runner, RunnerConfig  # noqa: E402
from tests import postern_schema  # noqa: E402
from tests.support import BUNDLE_VERSION, CREW_CONFIG, RUNTIME_ROOT

HOST = b"Host: 127.0.0.1\r\n"


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    shutil.copytree(RUNTIME_ROOT, root)
    shutil.copytree(CREW_CONFIG, root / "config", dirs_exist_ok=True)
    (root / "VERSION").write_text(BUNDLE_VERSION, encoding="utf-8")
    return root


@contextmanager
def _serve(
    bundle_root: Path,
    *,
    entitlement: ent.Entitlement | None = None,
    inbound_token: str = "",
    max_connections: int = server_module.MAX_CONNECTIONS,
) -> Iterator[int]:
    config = RunnerConfig(bundle_root=bundle_root, port=0, inbound_token=inbound_token)
    gate = entitlement or ent.Entitlement(base_url="", token="", agent_id="")
    server = PosternServer(config, Runner(config, gate), max_connections=max_connections)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


class _Wire:
    """One connection, read one response at a time -- so a second one is visible."""

    def __init__(self, port: int, *, timeout: float = 10.0) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.file = self.sock.makefile("rb")

    def send(self, data: bytes) -> _Wire:
        self.sock.sendall(data)
        return self

    def response(self) -> tuple[int, dict[str, str], bytes]:
        status_line = self.file.readline().decode("latin-1")
        assert status_line.startswith("HTTP/1.1 "), f"no status line, got {status_line!r}"
        headers: dict[str, str] = {}
        while (line := self.file.readline()) not in (b"\r\n", b""):
            name, _, value = line.decode("latin-1").partition(":")
            headers[name.strip().lower()] = value.strip()
        return int(status_line.split()[1]), headers, self.file.read(int(headers.get("content-length") or 0))

    def closed(self) -> bool:
        """Whether the runner ended the connection. A timeout here means it did not."""
        try:
            return self.file.read(1) == b""
        except ConnectionResetError:
            return True

    def close(self) -> None:
        self.file.close()
        self.sock.close()


def _post(path: str, body: bytes, *, content_type: str = "application/json", extra: bytes = b"") -> bytes:
    head = f"POST /postern/v0{path} HTTP/1.1\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\n"
    return head.encode("latin-1") + HOST + extra + b"\r\n" + body


def _get(path: str) -> bytes:
    return f"GET /postern/v0{path} HTTP/1.1\r\n".encode("latin-1") + HOST + b"\r\n"


def _envelope(body: bytes) -> dict:
    payload = json.loads(body)
    postern_schema.validate(payload, postern_schema.load("error"), name="error")
    return payload["error"]


# ---------------------------------------------------------------------------
# 1. What http.server refuses on its own answers in the envelope
# ---------------------------------------------------------------------------

_REFUSED_BY_HTTP_SERVER = {
    # Refused before a version is read, which is why the stdlib's answer to
    # these two carried no status line at all -- only an HTML body.
    "an HTTP/2.0 request line": b"GET /postern/v0/status HTTP/2.0\r\n" + HOST + b"\r\n",
    "a request line that is not one": b"NOT A REQUEST LINE\r\n\r\n",
    "more than 100 headers": _get("/status")[:-2] + b"".join(b"X-%d: y\r\n" % n for n in range(101)) + b"\r\n",
    # Exactly the stdlib's 65,537-byte read limit, so no byte is left unread to
    # turn the runner's close into a reset before the answer is read.
    "an over-long request line": b"GET /" + b"a" * 65_532,
}


@pytest.mark.parametrize("raw", _REFUSED_BY_HTTP_SERVER.values(), ids=_REFUSED_BY_HTTP_SERVER.keys())
def test_what_http_server_refuses_itself_answers_in_the_envelope(bundle: Path, raw: bytes) -> None:
    """SPEC 2.1: one error shape, even for a request the handler never saw.

    ``parse_request`` refused these with ``http.server``'s HTML page -- with
    codes (414, 431, 505) SPEC 2.1's table does not have. Each is now a
    ``400`` envelope, and the last answer on its connection.
    """
    with _serve(bundle) as port:
        wire = _Wire(port).send(raw)
        status, headers, body = wire.response()
        assert status == 400
        assert headers["content-type"] == "application/json; charset=utf-8"
        assert _envelope(body)["code"] == "bad_request"
        assert headers["connection"] == "close"
        assert wire.closed()


def test_a_method_with_no_handler_is_the_not_found_a_wrong_method_already_earns(bundle: Path) -> None:
    """``PUT`` has no ``do_PUT``; the stdlib answered an HTML ``501``.

    SPEC 2.1's ``501`` means a verb above the runner's level, which ``PUT``
    is not; a defined verb reached with the wrong method is this runner's
    ``404``, naming the method that works.
    """
    with _serve(bundle) as port:
        wire = _Wire(port).send(b"PUT /postern/v0/run HTTP/1.1\r\n" + HOST + b"Content-Length: 0\r\n\r\n")
        status, _, body = wire.response()
    assert status == 404
    assert _envelope(body)["message"] == "/run is served over POST, not PUT."


def test_a_method_with_no_handler_meets_the_gates_before_a_route_is_named(bundle: Path) -> None:
    """The token gate sits before the route lookup so a stranger cannot map the paths.

    Answering ``PUT`` from ``send_error`` directly would have named ``/run``'s
    method to a caller the gate refuses; it is dispatched instead, so both
    gates see it first.
    """
    with _serve(bundle, inbound_token="s3cret") as port:
        wire = _Wire(port).send(b"DELETE /postern/v0/run HTTP/1.1\r\n" + HOST + b"\r\n")
        status, _, body = wire.response()
    assert (status, _envelope(body)["code"]) == (401, "unauthorized")

    with _serve(bundle) as port:
        wire = _Wire(port).send(b"DELETE /postern/v0/run HTTP/1.1\r\nHost: evil.example\r\n\r\n")
        status, _, body = wire.response()
    assert status == 400
    assert _envelope(body)["detail"] == {"host": "evil.example"}


# ---------------------------------------------------------------------------
# 2. A body refused unread ends the connection
# ---------------------------------------------------------------------------


def _revoked() -> ent.Entitlement:
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id="acme/x")
    gate._store(
        ent.CheckAnswer(
            state=ent.STATE_REVOKED, checked_at=ent.datetime.now(ent.UTC), stale_after_seconds=60, grace_seconds=0
        )
    )
    return gate


_BODY = b'{"inputs": {"prompt": "go"}}'


@pytest.mark.parametrize(
    ("request_bytes", "entitlement", "expected"),
    [
        (_post("/run", _BODY, content_type="text/plain"), None, (400, "bad_request")),
        (_post("/run", _BODY), "revoked", (403, "not_entitled")),
        (_post("/nothing-here", _BODY), None, (404, "not_found")),
    ],
    ids=["refused on its media type", "refused on the entitlement", "sent to no route"],
)
def test_a_body_refused_unread_ends_the_connection_rather_than_becoming_the_next_request(
    bundle: Path, request_bytes: bytes, entitlement: str | None, expected: tuple[int, str]
) -> None:
    """The desync the review measured: the refused body parsed as a request line.

    Each refusal here is made before the body is read, which is its point --
    the media type is checked before reading because a ``text/plain`` body is
    the browser's no-preflight shape. So those bytes are still on the wire, and
    the client's next ``GET`` on the kept-alive connection came back an HTML
    ``400`` quoting them. The refusal now closes the connection instead.
    """
    gate = _revoked() if entitlement == "revoked" else None
    with _serve(bundle, entitlement=gate) as port:
        wire = _Wire(port).send(request_bytes + _get("/status"))
        status, headers, body = wire.response()
        assert (status, _envelope(body)["code"]) == expected
        assert headers["connection"] == "close"
        assert wire.closed(), "the unread body would have been read as the next request"


def test_a_body_that_was_read_leaves_the_connection_open(bundle: Path) -> None:
    """The control for the test above: closing is for an *unread* body only.

    This one is read and then refused (no ``prompt``), so the next request on
    the connection is still a request, and is answered.
    """
    with _serve(bundle) as port:
        wire = _Wire(port).send(_post("/run", b'{"inputs": {}}') + _get("/status"))
        status, headers, body = wire.response()
        assert (status, _envelope(body)["code"]) == (400, "bad_request")
        assert "connection" not in headers
        status, _, body = wire.response()
        assert status == 200
        assert json.loads(body)["postern"] == "0.1"


# ---------------------------------------------------------------------------
# 3. Content-Length is digits, once; Transfer-Encoding is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("declared", [b"1_3", b"+13", b"0x0d", b"13 13", "²".encode("latin-1"), b""])
def test_a_content_length_that_is_not_digits_is_refused(bundle: Path, declared: bytes) -> None:
    """``int()`` read ``1_3`` and ``+13`` as thirteen and read the body.

    ``²`` is the case ``isdigit()`` alone would let through to an ``int()``
    that then raises -- an unhandled error rather than a refusal.
    """
    body = b'{"inputs":{}}'
    raw = (
        b"POST /postern/v0/run HTTP/1.1\r\n"
        + HOST
        + b"Content-Type: application/json\r\nContent-Length: "
        + declared
        + b"\r\n\r\n"
        + body
    )
    with _serve(bundle) as port:
        status, headers, response = _Wire(port).send(raw).response()
    error = _envelope(response)
    assert (status, error["code"]) == (400, "bad_request")
    assert "Content-Length" in error["message"]
    assert headers["connection"] == "close"


def test_two_content_lengths_are_refused(bundle: Path) -> None:
    body = b'{"inputs":{}}'
    with _serve(bundle) as port:
        wire = _Wire(port).send(_post("/run", body, extra=b"Content-Length: 14\r\n"))
        status, _, response = wire.response()
    assert status == 400
    assert "Content-Length" in _envelope(response)["message"]


def test_a_content_length_with_surrounding_whitespace_is_still_a_length(bundle: Path) -> None:
    """RFC 9110 lets whitespace surround a field value; the strictness is on what is between."""
    body = b'{"inputs":{}}'
    raw = b"POST /postern/v0/run HTTP/1.1\r\n" + HOST + b"Content-Type: application/json\r\n"
    raw += b"Content-Length:  " + str(len(body)).encode() + b" \r\n\r\n" + body
    with _serve(bundle) as port:
        status, _, response = _Wire(port).send(raw).response()
    # Read, then refused for what it says -- not for how its length was written.
    assert (status, _envelope(response)["detail"]) == (400, {"key": "prompt"})


@pytest.mark.parametrize("coding", [b"chunked", b"gzip, chunked", b"identity"])
def test_any_transfer_encoding_is_refused(bundle: Path, coding: bytes) -> None:
    """Only ``chunked`` exactly was refused, so ``gzip, chunked`` fell through to Content-Length."""
    raw = _post("/run", b"", extra=b"Transfer-Encoding: " + coding + b"\r\n")
    with _serve(bundle) as port:
        wire = _Wire(port).send(raw + b"0\r\n\r\n")
        status, headers, response = wire.response()
        assert (status, _envelope(response)["code"]) == (400, "bad_request")
        assert headers["connection"] == "close"
        assert wire.closed()


# ---------------------------------------------------------------------------
# 4. A silent connection is dropped, and connections are bounded
# ---------------------------------------------------------------------------


def test_a_body_that_stops_arriving_is_refused_and_its_thread_freed(bundle: Path, monkeypatch) -> None:
    """Declared 1,000 bytes, sent 20, stalled: that thread was held for good."""
    monkeypatch.setattr(server_module._Handler, "timeout", 0.5)
    stalled = b"POST /postern/v0/run HTTP/1.1\r\n" + HOST + b"Content-Type: application/json\r\n"
    stalled += b"Content-Length: 1000\r\n\r\n" + b"x" * 20
    with _serve(bundle) as port:
        wire = _Wire(port).send(stalled)
        status, headers, body = wire.response()
        assert status == 400
        assert "stopped arriving" in _envelope(body)["message"]
        assert wire.closed()


def test_an_idle_kept_alive_connection_is_closed(bundle: Path, monkeypatch) -> None:
    """The bound is on by default -- it was ``None`` -- and it is what closes an idle connection.

    The first assertion is the one that failed before the bound existed; the rest
    shows the mechanism at a length a test can wait for.
    """
    assert server_module._Handler.timeout == server_module.SOCKET_TIMEOUT_SECONDS
    assert 0 < server_module.SOCKET_TIMEOUT_SECONDS <= 60
    monkeypatch.setattr(server_module._Handler, "timeout", 0.5)
    with _serve(bundle) as port:
        wire = _Wire(port).send(_get("/status"))
        assert wire.response()[0] == 200
        started = time.monotonic()
        assert wire.closed()
        assert time.monotonic() - started < 5


def test_connections_past_the_bound_are_closed_unserved_and_said_once(bundle: Path, caplog) -> None:
    """CORS gates reading a reply, not opening a socket, so any page can open these."""
    caplog.set_level(logging.WARNING, logger="sigrix_runtime.postern")
    with _serve(bundle, max_connections=2) as port:
        held = [_Wire(port).send(_get("/status")) for _ in range(2)]
        for wire in held:
            assert wire.response()[0] == 200  # both accepted, both now idle and held

        for _ in range(3):
            refused = _Wire(port)
            assert refused.closed()
            refused.close()
        saturated = [r for r in caplog.records if "connections are open" in r.getMessage()]
        assert len(saturated) == 1

        for wire in held:
            wire.close()
        deadline = time.monotonic() + 10
        while True:
            wire = _Wire(port)
            try:
                wire.send(_get("/status"))
                assert wire.response()[0] == 200
                break
            except (AssertionError, OSError):
                # The held connections' threads have not seen their EOF yet.
                assert time.monotonic() < deadline, "no slot came back after the held connections closed"
                time.sleep(0.05)
            finally:
                wire.close()


def test_a_request_line_cannot_write_control_characters_into_the_log(bundle: Path, caplog) -> None:
    """The stdlib's ``log_message`` escapes them; the override that routes it to ``logging`` did not."""
    caplog.set_level(logging.INFO, logger="sigrix_runtime.postern")
    with _serve(bundle) as port:
        wire = _Wire(port).send(b"GET /postern/v0/\x1b[2Jcleared HTTP/1.1\r\n" + HOST + b"\r\n")
        assert wire.response()[0] == 404
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "\x1b" not in logged
    assert "\\x1b[2Jcleared" in logged


# ---------------------------------------------------------------------------
# 5. What the edge says about itself
# ---------------------------------------------------------------------------


def test_the_server_header_names_the_runner_and_not_the_interpreter(bundle: Path) -> None:
    """``http.server`` appends ``Python/3.11.15``, which tells a caller only what to try."""
    with _serve(bundle) as port:
        _, headers, _ = _Wire(port).send(_get("/status")).response()
    assert headers["server"] == "postern-sigrix-runner/0.1"


def test_a_preflight_admits_no_header_the_runner_does_not_read(bundle: Path) -> None:
    """SPEC 2.3 names ``Idempotency-Key`` "where the runner honours it", and nothing here reads it.

    Admitted, it told a page its retry was deduplicated -- which, on a verb
    that spends money, is a second charge the page thought it had avoided.
    """
    config = RunnerConfig(bundle_root=bundle, port=0, allowed_origins=("https://app.example",))
    server = PosternServer(config, Runner(config, ent.Entitlement(base_url="", token="", agent_id="")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        preflight = b"OPTIONS /postern/v0/run HTTP/1.1\r\n" + HOST
        preflight += b"Origin: https://app.example\r\nAccess-Control-Request-Method: POST\r\n\r\n"
        status, headers, _ = _Wire(server.server_address[1]).send(preflight).response()
    finally:
        server.shutdown()
        server.server_close()
    assert status == 204
    assert headers["access-control-allow-headers"] == "Content-Type"
