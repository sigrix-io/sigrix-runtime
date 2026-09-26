"""``POSTERN_INBOUND_TOKEN``: the gate a runner bound off-machine needs.

SPEC 7 obliges a runner binding a non-loopback interface to authenticate its
callers and specifies no scheme for it; SPEC 2.1 fixes only the refusal, as
``401`` with ``unauthorized`` (upstreamed to ``sigrix-io/postern`` alongside
this story, since the code did not exist and the schema's enum is closed).

Two halves are tested here and neither is worth much alone.

**Unset must be byte-identical to before.** This package ships inside every
bundle a buyer downloads, so the failure that matters most is a runner that
starts demanding a token nobody set and refuses its own owner on their own
machine. ``tests/test_postern_runner_server.py`` is the real proof of that —
it drives all four verbs over a socket and is unchanged by this story — so
what is asserted here is the narrower thing that file cannot see: that
turning the gate *off* leaves no trace of it in the preflight either.

**Set must refuse before it does anything else.** Not merely "``describe``
answers 401": the gate precedes the route lookup and the entitlement gate, so
an unauthenticated caller cannot map the runner's paths or read the state of
somebody else's licence out of ``status``. Each of those is a separate test
because each is a separate line that a later tidy-up could reorder.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest

from sigrix_runtime.postern import PATH_PREFIX  # noqa: E402
from sigrix_runtime.postern import entitlement as ent  # noqa: E402
from sigrix_runtime.postern import errors as errors_module  # noqa: E402
from sigrix_runtime.postern import server as server_module  # noqa: E402
from sigrix_runtime.postern.server import PosternServer, Runner, RunnerConfig, build_runner  # noqa: E402
from tests import postern_schema  # noqa: E402
from tests.support import CREW_CONFIG, RUNTIME_ROOT

TOKEN = "s3cret-inbound-token"


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    shutil.copytree(RUNTIME_ROOT, root)
    shutil.copytree(CREW_CONFIG, root / "config", dirs_exist_ok=True)
    return root


class _Client:
    """The smallest HTTP client that can see what this test needs to see."""

    def __init__(self, port: int) -> None:
        self.port = port

    def request(
        self, method: str, path: str, *, body: bytes | None = None, headers: dict[str, str] | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request(method, PATH_PREFIX + path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
        finally:
            connection.close()

    def json(self, method: str, path: str, payload: Any = None, *, bearer: str = "") -> tuple[int, dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        if body is not None:
            headers["Content-Length"] = str(len(body))
        status, _, raw = self.request(method, path, body=body, headers=headers)
        return status, json.loads(raw.decode("utf-8")) if raw else {}


@contextmanager
def _serve(
    bundle_root: Path,
    *,
    inbound_token: str = "",
    entitlement: ent.Entitlement | None = None,
    origins: tuple[str, ...] = (),
) -> Iterator[_Client]:
    config = RunnerConfig(bundle_root=bundle_root, port=0, inbound_token=inbound_token, allowed_origins=origins)
    gate = entitlement or ent.Entitlement(base_url="", token="", agent_id="")
    server = PosternServer(config, Runner(config, gate))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Client(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Unset: the bundle on a buyer's own machine must not change
# ---------------------------------------------------------------------------


def test_with_no_token_configured_every_verb_answers_unauthenticated(bundle: Path) -> None:
    """The whole of the opt-in. A loopback runner refuses nobody."""
    with _serve(bundle) as client:
        assert client.json("GET", "/describe")[0] == 200
        assert client.json("GET", "/status")[0] == 200


def test_a_runner_reading_no_token_does_not_admit_the_header_in_its_preflight(bundle: Path) -> None:
    """SPEC 2.3: ``Authorization`` is off the browser's safelist.

    Naming it makes a client preflight a ``describe`` that would otherwise go
    without one — a cost paid for a credential this runner never reads.
    """
    with _serve(bundle, origins=("https://app.example",)) as client:
        _, headers, _ = client.request(
            "OPTIONS",
            "/describe",
            headers={"Origin": "https://app.example", "Access-Control-Request-Method": "GET"},
        )
        assert "Authorization" not in headers["access-control-allow-headers"]


# ---------------------------------------------------------------------------
# Set: the refusal, and what it refuses before
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [("GET", "/describe", None), ("GET", "/status", None), ("POST", "/run", {"inputs": {"prompt": "x"}})],
)
def test_an_unauthenticated_request_is_refused_in_the_envelope(
    bundle: Path, method: str, path: str, payload: Any
) -> None:
    """SPEC 2.1's envelope, with the code SPEC 7's refusal is defined as."""
    with _serve(bundle, inbound_token=TOKEN) as client:
        status, body = client.json(method, path, payload)
        assert status == 401
        assert set(body) == {"error"}
        assert body["error"]["code"] == errors_module.UNAUTHORIZED
        postern_schema.validate(body, postern_schema.load("error"), name=f"{method} {path}")


def test_the_refusal_names_the_scheme_rather_than_the_variable(bundle: Path) -> None:
    """The message is read by whoever is holding the wrong credential.

    Naming the *environment variable* would be addressed to the operator who
    already knows; naming the header is what the caller can act on.
    """
    with _serve(bundle, inbound_token=TOKEN) as client:
        message = client.json("GET", "/describe")[1]["error"]["message"]
    assert "Bearer" in message


def test_a_correct_token_answers_exactly_as_an_ungated_runner_does(bundle: Path) -> None:
    """Byte-for-byte, so the gate is a doorway rather than a filter."""
    with _serve(bundle) as open_client:
        open_status, open_body = open_client.json("GET", "/describe")
    with _serve(bundle, inbound_token=TOKEN) as gated:
        gated_status, gated_body = gated.json("GET", "/describe", bearer=TOKEN)
    assert (open_status, open_body) == (gated_status, gated_body)


@pytest.mark.parametrize(
    "header",
    [
        "",
        "Bearer",
        "Bearer ",
        f"Bearer {TOKEN}x",
        f"Basic {TOKEN}",
        TOKEN,
        f"bearer {TOKEN}"[:-1],
    ],
)
def test_a_near_miss_is_refused(bundle: Path, header: str) -> None:
    """Only the exact token, presented in the exact scheme, gets through."""
    with _serve(bundle, inbound_token=TOKEN) as client:
        status, _, _ = client.request("GET", "/describe", headers={"Authorization": header})
    assert status == 401


def test_more_than_one_space_after_the_scheme_is_still_conformant(bundle: Path) -> None:
    """RFC 9110 spells the separator ``1*SP``, so a second space is legal.

    Written down because the tidy-looking implementation — split once on a
    single space and compare the remainder — refuses a client that is within
    the grammar, and does it with a message about the credential rather than
    about the spacing. ``token68`` admits no space itself, so stripping what
    the separator left cannot make a wrong token right.
    """
    with _serve(bundle, inbound_token=TOKEN) as client:
        status, _, _ = client.request("GET", "/describe", headers={"Authorization": f"Bearer   {TOKEN}"})
    assert status == 200


def test_a_non_ascii_credential_is_refused_rather_than_dropping_the_connection(bundle: Path) -> None:
    """One junk byte must not turn a refusal into a dropped request.

    ``hmac.compare_digest`` raises ``TypeError`` on a ``str`` carrying a
    non-ASCII character, and ``http.server`` decodes headers as latin-1, so a
    caller supplies one by sending a single high byte. The gate runs before
    ``_dispatch``'s ``try``, so that raise reaches ``socketserver``, which
    logs a traceback and closes the socket — the client sees a connection
    error rather than the ``401`` envelope, from a request it chose.

    Written against the wire because that is the only place the byte exists:
    an assertion about ``_authorized`` in isolation cannot show the response
    that never arrives.
    """
    with _serve(bundle, inbound_token=TOKEN) as client:
        status, _, raw = client.request("GET", "/describe", headers={"Authorization": "Bearer t\u00ffken"})
    assert status == 401
    assert json.loads(raw.decode("utf-8"))["error"]["code"] == errors_module.UNAUTHORIZED


def test_the_scheme_name_is_matched_case_insensitively(bundle: Path) -> None:
    """RFC 9110 makes it so, and a client is entitled to spell it either way."""
    with _serve(bundle, inbound_token=TOKEN) as client:
        status, _, _ = client.request("GET", "/describe", headers={"Authorization": f"bEaReR {TOKEN}"})
    assert status == 200


def test_an_unknown_path_is_refused_before_it_is_looked_up(bundle: Path) -> None:
    """So an unauthenticated caller cannot map the runner's paths.

    Without the gate this answers ``404`` naming the four verbs it serves,
    which is a small map handed to somebody holding no credential.
    """
    with _serve(bundle, inbound_token=TOKEN) as client:
        assert client.json("GET", "/no-such-verb")[1]["error"]["code"] == errors_module.UNAUTHORIZED
        # ...and the map is still there for a caller who may have it.
        assert client.json("GET", "/no-such-verb", bearer=TOKEN)[1]["error"]["code"] == errors_module.NOT_FOUND


def test_the_gate_precedes_the_entitlement_gate(bundle: Path) -> None:
    """``status`` carries an entitlement snapshot and the agent's identifier.

    Neither is a fact for a caller who may not talk to this runner at all, so
    the credential is decided before ``prepare_run``'s own ordering (SPEC 4.6)
    is reached. A ``403`` here would report somebody else's licence state.
    """
    revoked = ent.Entitlement(base_url="https://d.example", token="t", agent_id="acme/x")
    revoked._store(
        ent.CheckAnswer(
            state=ent.STATE_REVOKED,
            checked_at=ent.datetime.now(ent.UTC),
            stale_after_seconds=60,
            grace_seconds=0,
        )
    )
    with _serve(bundle, inbound_token=TOKEN, entitlement=revoked) as client:
        assert client.json("POST", "/run", {"inputs": {"prompt": "go"}})[0] == 401
        # The entitlement refusal is still there for a caller who gets past it.
        assert client.json("POST", "/run", {"inputs": {"prompt": "go"}}, bearer=TOKEN)[0] == 403


def test_a_head_request_is_gated_too(bundle: Path) -> None:
    """It routes through the same dispatch, so it must not be a way around."""
    with _serve(bundle, inbound_token=TOKEN) as client:
        assert client.request("HEAD", "/describe")[0] == 401


# ---------------------------------------------------------------------------
# The preflight is not the verb behind it (SPEC 2.3)
# ---------------------------------------------------------------------------


def test_a_preflight_is_answered_without_a_credential(bundle: Path) -> None:
    """SPEC 2.3: a preflight MUST NOT require credentials.

    A browser cannot attach one to it in any case, so gating it would make a
    gated runner unreachable from a page rather than merely authenticated.
    """
    with _serve(bundle, inbound_token=TOKEN, origins=("https://app.example",)) as client:
        status, headers, raw = client.request(
            "OPTIONS",
            "/run",
            headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST"},
        )
    assert status == 204
    assert raw == b""
    assert headers["access-control-allow-origin"] == "https://app.example"


def test_a_gated_runner_admits_the_header_it_reads(bundle: Path) -> None:
    """A browser cannot send a header its preflight did not admit.

    Without this a page is the one client kind that could never authenticate —
    and the failure reads as a wrong credential while being a missing header.
    """
    with _serve(bundle, inbound_token=TOKEN, origins=("https://app.example",)) as client:
        _, headers, _ = client.request(
            "OPTIONS",
            "/run",
            headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST"},
        )
    admitted = headers["access-control-allow-headers"]
    assert "Authorization" in admitted
    # The one it already admitted is not displaced by it.
    assert "Content-Type" in admitted
    # SPEC 2.3 names ``Idempotency-Key`` "where the runner honours it", and
    # this one does not (no ``status.idempotent_retry``), so it is not admitted
    # -- a page told its retry is safe would double-spend on one.
    assert "Idempotency-Key" not in admitted


# ---------------------------------------------------------------------------
# The token never travels back out
# ---------------------------------------------------------------------------


def test_the_token_appears_in_neither_describe_nor_status(bundle: Path) -> None:
    """SPEC 4.1.3's reasoning, applied to the runner's own credential.

    A runner that answered with its inbound token would publish, to a reader
    who had just presented it, the value every other reader is missing.
    """
    with _serve(bundle, inbound_token=TOKEN) as client:
        for path in ("/describe", "/status"):
            assert TOKEN not in json.dumps(client.json("GET", path, bearer=TOKEN)[1])


def test_a_token_set_in_the_bundles_env_stays_out_of_describe_too(bundle: Path) -> None:
    """The realistic hosted shape, and the one the previous test cannot reach.

    ``_serve`` hands the token to the config directly, so it never proves the
    *file* is not also read back out. A bundle carrying the value on disk is
    what a deployment actually has, and ``describe`` derives itself from that
    same directory when no ``postern.json`` was generated for it.
    """
    (bundle / ".env").write_text(f"POSTERN_INBOUND_TOKEN={TOKEN}\n", encoding="utf-8")
    config = RunnerConfig(bundle_root=bundle, port=0)
    runner = build_runner(config, {})
    assert config.inbound_token == TOKEN, "the gate should be on for this to mean anything"
    assert TOKEN not in json.dumps(runner.describe())
    assert TOKEN not in json.dumps(runner.status())


def test_a_refusal_is_logged_without_the_presented_value(bundle: Path, caplog) -> None:
    """A log is read by more people than a response is.

    A near miss is the common case — a stale token, a copied trailing space —
    so logging what was presented publishes a working credential whenever the
    failure was a typo rather than an attack.
    """
    presented = f"{TOKEN}-but-wrong"
    with (
        caplog.at_level(logging.WARNING, logger="sigrix_runtime.postern"),
        _serve(bundle, inbound_token=TOKEN) as client,
    ):
        client.request("GET", "/describe", headers={"Authorization": f"Bearer {presented}"})
    refusals = [r for r in caplog.records if "refused an unauthenticated request" in r.getMessage()]
    assert refusals, "a refused request should say so at WARNING"
    for record in refusals:
        assert presented not in record.getMessage()
        assert TOKEN not in record.getMessage()


def test_the_comparison_is_constant_time() -> None:
    """A source scan, because a timing property has no behavioural tell.

    ``==`` on two strings returns as soon as they differ, so the time a
    refusal takes reports how much of a guess was right.
    """
    source = (RUNTIME_ROOT / "sigrix_runtime" / "postern" / "server.py").read_text(encoding="utf-8")
    assert "hmac.compare_digest(" in source


# ---------------------------------------------------------------------------
# Where the token is resolved from
# ---------------------------------------------------------------------------


def test_the_token_is_read_from_the_environment(bundle: Path) -> None:
    """So an operator who sets the variable gets the gate."""
    runner = build_runner(RunnerConfig(bundle_root=bundle, port=0), {"POSTERN_INBOUND_TOKEN": TOKEN})
    assert runner.config.inbound_token == TOKEN


def test_the_token_is_read_from_the_bundles_env_file(bundle: Path) -> None:
    """The same reader the entitlement credentials use.

    A bundle on a machine is configured by its ``.env``; a hosted runner is
    configured by its host. One resolver is what keeps those from being two
    answers.
    """
    (bundle / ".env").write_text(f'POSTERN_INBOUND_TOKEN="{TOKEN}"\n', encoding="utf-8")
    runner = build_runner(RunnerConfig(bundle_root=bundle, port=0), {})
    assert runner.config.inbound_token == TOKEN


def test_a_real_environment_variable_beats_a_stale_env_file(bundle: Path) -> None:
    """The container exports them; a stale ``.env`` must not override an operator."""
    (bundle / ".env").write_text("POSTERN_INBOUND_TOKEN=stale\n", encoding="utf-8")
    runner = build_runner(RunnerConfig(bundle_root=bundle, port=0), {"POSTERN_INBOUND_TOKEN": TOKEN})
    assert runner.config.inbound_token == TOKEN


def test_an_explicit_config_value_is_not_overridden(bundle: Path) -> None:
    """Filled rather than overridden: a caller who set it has already decided."""
    config = RunnerConfig(bundle_root=bundle, port=0, inbound_token="chosen")
    build_runner(config, {"POSTERN_INBOUND_TOKEN": TOKEN})
    assert config.inbound_token == "chosen"


def test_no_token_anywhere_leaves_the_gate_off(bundle: Path) -> None:
    """Empty is the only default that can ship inside every bundle."""
    runner = build_runner(RunnerConfig(bundle_root=bundle, port=0), {})
    assert runner.config.inbound_token == ""


# ---------------------------------------------------------------------------
# The operator signal: SPEC 7's MUST, reported rather than enforced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.0.0.5", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("0.0.0.0", False),  # noqa: S104 - naming the bind address is the point
        ("", False),
        ("10.0.0.4", False),
        ("runner.internal", False),
    ],
)
def test_a_bind_address_is_classified_by_address_not_by_name(host: str, loopback: bool) -> None:
    """SPEC 7 forbids settling this by matching a hostname.

    A name is what a resolver decides, so anything that is not an address
    reads as off-machine — the answer that warns rather than the one that
    stays quiet.
    """
    assert server_module._is_loopback_host(host) is loopback


@contextmanager
def _boot(config: RunnerConfig, environ: dict[str, str]) -> Iterator[None]:
    """Run ``serve()``'s announcement without serving.

    The one thing worth exercising here is what an operator is told at boot,
    and ``serve_forever`` never returns. Patching it is what makes the rest of
    that function — including the branch under test — reachable.

    ``shutdown`` has to be patched with it, and that is not belt-and-braces.
    ``socketserver`` clears an event on the way into ``serve_forever`` and sets
    it on the way out; ``shutdown`` waits on that event. A no-op
    ``serve_forever`` therefore leaves it unset from construction, and
    ``serve()``'s own ``finally`` blocks there forever — a hung harness rather
    than a failing test, which is a slow thing to read. ``server_close`` is
    left real so the port is genuinely released between tests.
    """
    originals = (PosternServer.serve_forever, PosternServer.shutdown)
    PosternServer.serve_forever = lambda self, *a, **k: None  # type: ignore[method-assign]
    PosternServer.shutdown = lambda self: None  # type: ignore[method-assign]
    try:
        server_module.serve(config, environ)
        yield
    finally:
        PosternServer.serve_forever, PosternServer.shutdown = originals  # type: ignore[method-assign]


def test_binding_off_machine_with_no_token_warns(bundle: Path, caplog) -> None:
    """SPEC 7's MUST, reported rather than enforced.

    Refusing to bind would take the decision off an operator who may have put
    their own gateway in front, which this runner cannot see. It is a WARNING
    because the runner looks identical either way and ``status`` deliberately
    says nothing about it — the boot log is the only place the person who can
    fix it will look.
    """
    config = RunnerConfig(bundle_root=bundle, host="0.0.0.0", port=0)  # noqa: S104 - the case under test
    with caplog.at_level(logging.WARNING, logger="sigrix_runtime.postern"), _boot(config, {}):
        pass
    assert any("no inbound authentication" in r.getMessage() for r in caplog.records)


def test_binding_off_machine_with_a_token_does_not_warn(bundle: Path, caplog) -> None:
    with (
        caplog.at_level(logging.DEBUG, logger="sigrix_runtime.postern"),
        _boot(
            RunnerConfig(bundle_root=bundle, host="0.0.0.0", port=0),  # noqa: S104 - the case under test
            {"POSTERN_INBOUND_TOKEN": TOKEN},
        ),
    ):
        pass
    assert not any("no inbound authentication" in r.getMessage() for r in caplog.records)
    assert any("must carry POSTERN_INBOUND_TOKEN" in r.getMessage() for r in caplog.records)


def test_a_loopback_runner_is_never_warned_at(bundle: Path, caplog) -> None:
    """The default deployment: a bundle a buyer unzipped on their own machine."""
    with (
        caplog.at_level(logging.DEBUG, logger="sigrix_runtime.postern"),
        _boot(RunnerConfig(bundle_root=bundle, port=0), {}),
    ):
        pass
    assert not any("no inbound authentication" in r.getMessage() for r in caplog.records)


def test_the_boot_log_never_carries_the_token(bundle: Path, caplog) -> None:
    with (
        caplog.at_level(logging.DEBUG, logger="sigrix_runtime.postern"),
        _boot(
            RunnerConfig(bundle_root=bundle, host="0.0.0.0", port=0),  # noqa: S104 - the case under test
            {"POSTERN_INBOUND_TOKEN": TOKEN},
        ),
    ):
        pass
    assert all(TOKEN not in r.getMessage() for r in caplog.records)
