"""A stand-in for the distributor half of Postern, and the bundle it serves.

Shared by the pull's own tests and the image's, because the two ask the same
questions of it — one fake, so a change to what a distributor answers cannot
become true in one file only. It speaks over a real loopback socket rather
than by patching: the rules being tested are about connections (SPEC 7), a
bearer header (5.3), a digest header (5.6) and a body read in chunks, and a
mock of ``open_response`` would assert none of them.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import zipfile
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from tests.support import CREW_CONFIG, RUNTIME_ROOT

AGENT_ID = "acme/market-research-crew"
BUNDLE_MEDIA_TYPE = "application/zip"
REPR_DIGEST_HEADER = "Repr-Digest"


def zip_bytes(files: dict[str, bytes], *, stored: bool = False) -> bytes:
    """``stored`` keeps the archive as large as its contents, which is how a
    test says "a big download" rather than "a big unpacking"."""
    buffer = io.BytesIO()
    method = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buffer, "w", method) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def real_bundle_zip() -> bytes:
    """The shipped starter tree plus a demo's ``config/`` — a servable bundle."""
    files: dict[str, bytes] = {}
    for path in sorted(RUNTIME_ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[path.relative_to(RUNTIME_ROOT).as_posix()] = path.read_bytes()
    for path in sorted(CREW_CONFIG.iterdir()):
        if path.is_file():
            files[f"config/{path.name}"] = path.read_bytes()
    # What a distributor's bundle build adds on top of the static tree.
    # The README matters to more than shape: it is the entry most likely to
    # collide with something a buyer left in the folder.
    files["README.md"] = b"# The agent\n"
    return zip_bytes(files)


def digest_header(payload: bytes) -> str:
    return f"sha-256=:{base64.b64encode(hashlib.sha256(payload).digest()).decode('ascii')}:"


def check_body(state: str = "active", *, agent_id: str = AGENT_ID, access_ends_at: str = "") -> bytes:
    """SPEC 5.3's answer, stamped now, with its one optional member when given.

    Deliberately not a fixed timestamp: ``checked_at`` is the distributor's
    clock and the runner measures staleness against it, so an answer written
    into a test as a literal is *already stale* by the time it is read — and
    a stale ``active`` decides as ``unknown``, which is a different branch
    from the one under test.
    """
    body: dict[str, Any] = {
        "postern": "0.1",
        "state": state,
        "agent_id": agent_id,
        "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stale_after_seconds": 60,
        "grace_seconds": 86400,
    }
    if access_ends_at:
        body["access_ends_at"] = access_ends_at
    return json.dumps(body).encode("utf-8")


def envelope(code: str, message: str, detail: Any = None) -> bytes:
    """SPEC 2.1's error object, whose root is closed."""
    return json.dumps({"error": {"code": code, "message": message, "detail": detail}}).encode("utf-8")


def serving(payload: bytes, *, digest: str | None = None, status: int = 200):
    """A handler that answers everything with ``payload``."""
    headers = {"Content-Type": BUNDLE_MEDIA_TYPE}
    if digest is None:
        digest = digest_header(payload)
    if digest:
        headers[REPR_DIGEST_HEADER] = digest
    return lambda path: (status, headers, payload)


def version_body(*, agent_id: str = AGENT_ID, version: str = "1.0.0") -> bytes:
    """SPEC 8's answer, stamped with a version nothing else here uses."""
    return json.dumps({"postern": "0.1", "agent_id": agent_id, "version": version}).encode("utf-8")


def licensed_bundle(payload: bytes | None = None):
    """A handler that answers the check, the pull, and the version check —
    the three requests a real boot makes (SPEC 5.3, 5.6, 8)."""

    def _handler(path: str):
        if "/entitlements/" in path:
            return 200, {"Content-Type": "application/json"}, check_body("active")
        if "/versions/" in path:
            return 200, {"Content-Type": "application/json"}, version_body()
        body = real_bundle_zip() if payload is None else payload
        return 200, {"Content-Type": BUNDLE_MEDIA_TYPE, REPR_DIGEST_HEADER: digest_header(body)}, body

    return _handler


class Distributor:
    """SPEC 5.3 and 5.6's endpoints, on loopback, answering whatever a test says."""

    def __init__(self, handler, *, send_length: bool = True, http_1_1: bool = True) -> None:
        self.requests: list[tuple[str, str]] = []
        #: Every header of every request, for the tests that care about what
        #: a runner does *not* send.
        self.exchanges: list[tuple[str, dict[str, str]]] = []
        outer = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1" if http_1_1 else "HTTP/1.0"

            def do_GET(self) -> None:  # noqa: N802 - http.server's naming
                outer.requests.append((self.path, self.headers.get("Authorization") or ""))
                outer.exchanges.append((self.path, {key.lower(): value for key, value in self.headers.items()}))
                status, headers, body = handler(self.path)
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                if send_length:
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:  # noqa: A002
                pass

        self._server = HTTPServer(("127.0.0.1", 0), _H)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def paths(self) -> list[str]:
        return [path for path, _ in self.requests]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def distributor():
    """A factory, so one test can stand up more than one and none leaks."""
    made: list[Distributor] = []

    def _make(handler, **kwargs) -> Distributor:
        instance = Distributor(handler, **kwargs)
        made.append(instance)
        return instance

    yield _make
    for instance in made:
        instance.close()
