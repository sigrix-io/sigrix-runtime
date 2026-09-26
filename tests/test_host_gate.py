"""The ``Host`` gate: what stops a DNS-rebound page driving a runner.

A page on ``evil.com`` whose name is re-resolved to ``127.0.0.1`` is
*same-origin* with a loopback runner. The browser therefore applies no CORS
and sends ``application/json`` with no preflight, so the two controls the
runner already had are not defeated --- they are never consulted. The run
executes, spending money and invoking ``write_tools``, and the page reads the
result. ``Host`` is the one header the page cannot forge, and so the one
thing telling that request apart from the buyer's own.

Three halves are tested here and none is worth much alone.

**The attack is refused.** On every verb, not only ``run``: ``describe`` and
``status`` carry the agent's identifier and somebody's entitlement state, and
a page that can read those has already learned more than it should.

**The buyer is not.** This package ships inside every bundle a buyer
downloads, so the failure that matters most is a gate that refuses its own
owner --- and the shapes it could break are not one. Loopback by address and
by name, IPv6, an address literal a LAN operator reached it at, an absent
header, a host the operator declared.

**The hosted runner is untouched.** The platform's own hosted runners bind
``0.0.0.0`` inside a container and are reached at an ingress name, which
is a name this process was never told. They carry ``POSTERN_INBOUND_TOKEN``, and a bearer token defeats
rebinding outright, so such a runner is exempt --- without that, this change
would refuse every hosted run to re-close a hole the token had already
closed. That exemption is the one line here most likely to read as a
weakening and be "tightened" away, so it is asserted from both sides.
"""

from __future__ import annotations

import json
import logging
import shutil
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from sigrix_runtime.postern import PATH_PREFIX  # noqa: E402
from sigrix_runtime.postern import entitlement as ent  # noqa: E402
from sigrix_runtime.postern import server as server_module  # noqa: E402
from sigrix_runtime.postern.server import PosternServer, Runner, RunnerConfig  # noqa: E402
from tests import postern_schema  # noqa: E402
from tests.support import CREW_CONFIG, RUNTIME_ROOT

TOKEN = "s3cret-inbound-token"
REBOUND = "evil.com"
HOSTED_FQDN = "crew-abc123.kindhill-9f2c1d40.westeurope.azurecontainerapps.io"


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    shutil.copytree(RUNTIME_ROOT, root)
    shutil.copytree(CREW_CONFIG, root / "config", dirs_exist_ok=True)
    return root


class _Client:
    """A hand-written request, because ``Host`` is what is under test.

    ``http.client`` supplies a ``Host`` of its own whenever the caller names
    none, so it cannot send the one shape that has to be covered --- a request
    carrying no ``Host`` at all. Writing the bytes settles what is on the wire
    rather than what a library decided to put there.
    """

    def __init__(self, port: int) -> None:
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        *,
        host: str | None = "",
        extra_host: str = "",
        payload: Any = None,
        bearer: str = "",
    ) -> tuple[int, dict[str, str], bytes]:
        # `host=""` means "the real one"; `host=None` means "send none".
        lines = [f"{method} {PATH_PREFIX}{path} HTTP/1.1"]
        if host is not None:
            lines.append(f"Host: {host or f'127.0.0.1:{self.port}'}")
        if extra_host:
            lines.append(f"Host: {extra_host}")
        if bearer:
            lines.append(f"Authorization: Bearer {bearer}")
        body = b""
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            lines += ["Content-Type: application/json", f"Content-Length: {len(body)}"]
        lines += ["Connection: close", "", ""]
        connection = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        try:
            connection.sendall("\r\n".join(lines).encode("utf-8") + body)
            chunks = []
            while True:
                received = connection.recv(65536)
                if not received:
                    break
                chunks.append(received)
        finally:
            connection.close()
        head, _, rest = b"".join(chunks).partition(b"\r\n\r\n")
        status = int(head.split()[1])
        headers = {}
        for line in head.decode("latin-1").split("\r\n")[1:]:
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        return status, headers, rest

    def json(self, method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        status, _, raw = self.request(method, path, **kwargs)
        return status, json.loads(raw.decode("utf-8")) if raw else {}


@contextmanager
def _serve(
    bundle_root: Path,
    *,
    inbound_token: str = "",
    origins: tuple[str, ...] = (),
    declares_host: str = "",
) -> Iterator[_Client]:
    """A real runner on a real socket.

    ``declares_host`` is set *after* the bind rather than passed to it: the
    names worth testing (an internal hostname, the container's ``0.0.0.0``)
    either do not resolve here or would bind an interface a test has no
    business opening. What the gate reads is the configured value, and a name
    does not have to resolve to have been declared.
    """
    config = RunnerConfig(bundle_root=bundle_root, port=0, inbound_token=inbound_token, allowed_origins=origins)
    server = PosternServer(config, Runner(config, ent.Entitlement(base_url="", token="", agent_id="")))
    if declares_host:
        config.host = declares_host
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Client(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# The attack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("POST", "/run", {"inputs": {"prompt": "hi"}}),
        ("POST", "/stream", {"inputs": {"prompt": "hi"}}),
        ("GET", "/describe", None),
        ("GET", "/status", None),
    ],
)
def test_a_rebound_host_is_refused_on_every_verb(bundle: Path, method: str, path: str, payload: Any) -> None:
    """The issue's own reproduction, and its three neighbours.

    ``run`` is where the money is, but ``describe`` names the agent and
    ``status`` carries an entitlement snapshot, and neither is a fact for a
    page the buyer never meant to hand this runner to.
    """
    with _serve(bundle) as client:
        status, body = client.json(method, path, host=REBOUND, payload=payload)
    assert status == 400
    assert body["error"]["code"] == "bad_request"
    postern_schema.validate(body, postern_schema.load("error"), name=f"{method} {path}")


def test_the_refusal_names_the_host_and_the_remedy(bundle: Path) -> None:
    """An operator who meets this legitimately has to be able to act on it.

    The remedy is the token rather than a flag of this gate's own: a runner
    reached by a name is off-machine, which is the deployment SPEC 7 already
    requires to authenticate its callers.
    """
    with _serve(bundle) as client:
        _, body = client.json("GET", "/describe", host=REBOUND)
    message = body["error"]["message"]
    assert REBOUND in message
    assert "POSTERN_INBOUND_TOKEN" in message
    assert body["error"]["detail"]["host"] == REBOUND


def test_the_refusal_is_logged_as_a_warning(bundle: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The buyer's only signal that a page tried this.

    ``caplog`` is filtered by logger name rather than counted: this runner is
    threaded and the suite's other daemon threads log into whatever window is
    open (see CLAUDE.md), so a bare record count would be a claim about the
    whole process.
    """
    with caplog.at_level(logging.WARNING, logger="sigrix_runtime.postern"), _serve(bundle) as client:
        client.json("GET", "/describe", host=REBOUND)
    warnings = [r for r in caplog.records if r.name == "sigrix_runtime.postern" and r.levelno >= logging.WARNING]
    assert [r for r in warnings if REBOUND in r.getMessage()]


def test_a_folded_host_cannot_forge_a_log_line(bundle: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The value is the caller's, and it is quoted before it is logged.

    An obs-fold continuation survives header parsing as an embedded newline,
    so a bare ``%s`` would let whoever sends one write their own log records.
    """
    with caplog.at_level(logging.WARNING, logger="sigrix_runtime.postern"), _serve(bundle) as client:
        status, _ = client.json("GET", "/describe", host="evil.com\r\n INFO all-clear")
    assert status == 400
    for record in caplog.records:
        assert "\n" not in record.getMessage()


def test_more_than_one_host_header_is_refused(bundle: Path) -> None:
    """Found by driving it: the second header was accepted on the first's word.

    ``self.headers.get("Host")`` answers with the first of two, so the gate
    was checking a value while a different one sat beside it unread. RFC 9112
    3.2 makes that a MUST-reject, and nothing legitimate sends two --- a
    browser cannot, ``Host`` being forbidden to ``fetch``.
    """
    with _serve(bundle) as client:
        status, body = client.json("GET", "/describe", host="127.0.0.1", extra_host=REBOUND)
    assert status == 400
    assert body["error"]["code"] == "bad_request"
    assert "one Host header" in body["error"]["message"]


def test_a_long_host_is_bounded_before_it_is_quoted_back(bundle: Path) -> None:
    """A header can be long; a refusal that echoes all of it is a megaphone."""
    with _serve(bundle) as client:
        _, body = client.json("GET", "/describe", host="a" * 4000 + ".example")
    assert len(body["error"]["detail"]["host"]) <= server_module.MAX_QUOTED_HOST_CHARS + 3


# ---------------------------------------------------------------------------
# The buyer, who must not notice any of this
# ---------------------------------------------------------------------------


def test_the_shapes_a_buyer_reaches_their_own_runner_at_all_answer(bundle: Path) -> None:
    """Loopback by address and by name, IPv6, and no header at all.

    An absent ``Host`` passes because a browser always sends one, so its
    absence cannot arrive from the attack --- refusing it would enforce
    HTTP/1.1 conformance this runner has never enforced, against clients that
    are not the threat.
    """
    with _serve(bundle) as client:
        port = client.port
        for host in (f"127.0.0.1:{port}", "127.0.0.1", f"localhost:{port}", "localhost", f"[::1]:{port}", "[::1]"):
            assert client.json("GET", "/describe", host=host)[0] == 200, host
        assert client.json("GET", "/describe", host=None)[0] == 200


def test_an_address_literal_passes_wherever_it_points(bundle: Path) -> None:
    """Rebinding needs a name; an address has nothing to re-point.

    A browser sends an address literal only for a page whose own origin is
    that address, which this runner cannot serve --- it answers JSON and no
    document. So refusing an operator who reached their runner at its LAN
    address would cost a real deployment and stop nothing, and this is the
    assertion that says so out loud before somebody "tightens" it to loopback.
    """
    with _serve(bundle) as client:
        for host in ("192.168.1.50:8787", "10.4.2.9", "203.0.113.5:8787", "[2001:db8::1]:8787"):
            assert client.json("GET", "/describe", host=host)[0] == 200, host


def test_a_name_that_merely_ends_in_localhost_is_a_different_name(bundle: Path) -> None:
    """``localhost`` is matched exactly, so a subdomain of it is not it."""
    with _serve(bundle) as client:
        assert client.json("GET", "/describe", host="evil.localhost")[0] == 400
        assert client.json("GET", "/describe", host="localhost.evil.com")[0] == 400
        assert client.json("GET", "/describe", host="localhost.")[0] == 200


def test_a_host_the_operator_declared_answers(bundle: Path) -> None:
    """The two declarations a runner already has: the bind, and an origin."""
    with _serve(bundle, declares_host="runner.internal") as client:
        assert client.json("GET", "/describe", host="runner.internal:8787")[0] == 200
        assert client.json("GET", "/describe", host=REBOUND)[0] == 400

    with _serve(bundle, origins=("https://app.example.com:8443",)) as client:
        assert client.json("GET", "/describe", host="app.example.com")[0] == 200
        assert client.json("GET", "/describe", host=REBOUND)[0] == 400


def test_allowing_every_origin_is_not_a_way_past_this(bundle: Path) -> None:
    """``*`` says who may read a reply, not what this runner is called.

    The attack this gate is about does not need a permitted origin --- it is
    same-origin --- so reading ``*`` as a bypass would hand it back to every
    operator who ever set one.
    """
    config = RunnerConfig(bundle_root=bundle, port=0, allow_any_origin=True)
    server = PosternServer(config, Runner(config, ent.Entitlement(base_url="", token="", agent_id="")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _Client(server.server_address[1]).json("GET", "/describe", host=REBOUND)[0] == 400
    finally:
        server.shutdown()
        server.server_close()


def test_the_preflight_is_still_answered(bundle: Path) -> None:
    """SPEC 2.3 makes answering ``OPTIONS`` a MUST.

    A preflight has no side effect and the request behind it meets this gate
    anyway, so refusing one would trade a rule of the specification for
    nothing --- the same reason the token gate skips it.
    """
    with _serve(bundle) as client:
        status, _, _ = client.request("OPTIONS", "/run", host=REBOUND)
    assert status == 204


# ---------------------------------------------------------------------------
# The hosted runner, which is reached at a name on purpose
# ---------------------------------------------------------------------------


def test_a_runner_that_authenticates_answers_to_any_name(bundle: Path) -> None:
    """The platform's hosted runners, which this must not break.

    ``0.0.0.0`` inside a container, reached at an ingress FQDN nothing here
    was told about. The token is the stronger control and has already closed
    this hole; checking the name as well would refuse every hosted run.
    """
    with _serve(bundle, inbound_token=TOKEN, declares_host="0.0.0.0") as client:  # noqa: S104 - the deployment under test
        assert client.json("GET", "/describe", host=HOSTED_FQDN, bearer=TOKEN)[0] == 200
        assert client.json("POST", "/status", host=HOSTED_FQDN, bearer=TOKEN)[0] == 404


def test_the_exemption_admits_nobody_the_token_would_not(bundle: Path) -> None:
    """The half that makes the exemption safe rather than a hole.

    A rebound page cannot read the token --- it is not in the page and a
    browser never attaches one --- so on a protected runner the request this
    gate exists to stop is refused before the gate is reached. Asserting the
    exemption alone would pass just as well if it admitted everyone.
    """
    with _serve(bundle, inbound_token=TOKEN, declares_host="0.0.0.0") as client:  # noqa: S104 - the deployment under test
        for host in (HOSTED_FQDN, REBOUND, "127.0.0.1"):
            status, body = client.json("GET", "/describe", host=host)
            assert status == 401, host
            assert body["error"]["code"] == "unauthorized"


# ---------------------------------------------------------------------------
# The parsing, where the interesting cases are unreachable over a socket
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("127.0.0.1:8787", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("[::1]:8787", "::1"),
        ("[::1]", "::1"),
        ("::1", "::1"),  # no brackets: more colons than a port can explain
        ("EVIL.COM:8787", "evil.com"),
        ("evil.com.", "evil.com"),
        ("  localhost:8787  ", "localhost"),
        ("", ""),
    ],
)
def test_the_host_label_is_read_off_the_header(raw: str, expected: str) -> None:
    """Ports come off, brackets come off, the DNS root label comes off.

    ``127.0.0.1.`` is the one that matters: left on, the trailing dot makes an
    address fail to parse as one and the request is refused as a name.
    """
    assert server_module._host_label(raw) == expected


def test_a_wildcard_bind_declares_no_name(bundle: Path) -> None:
    """An operator who asked for every interface has said nothing about names.

    ``0.0.0.0`` still passes as an *address literal*, which is the other rule;
    what must not happen is a wildcard bind quietly declaring a host, since
    that is how every container would come to answer to one.
    """
    for wildcard in ("", "*", "0.0.0.0", "::", "[::]"):  # noqa: S104 - naming a bind, not making one
        config = RunnerConfig(bundle_root=bundle, host=wildcard)
        assert server_module._declared_hosts(config) == frozenset(), wildcard

    assert server_module._declared_hosts(RunnerConfig(bundle_root=bundle, host="Runner.Internal.")) == {
        "runner.internal"
    }
