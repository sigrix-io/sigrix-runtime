"""The HTTP surface: four verbs, the CORS rules, and the SSE framing.

Serves ``/postern/v0/{describe,run,stream,status}`` for the bundle it was
started inside. HTTP/1.1 on a TCP port, loopback by default (SPEC 2, 7).

**Every non-2xx body is the specification's envelope** (SPEC 2.1), unknown
paths and unsupported methods included. A client parsing this namespace
should never meet a second error shape, which is why nothing here lets
``http.server``'s own HTML error page out.

**The ``Content-Type`` gate on ``run`` and ``stream`` is a security
control, not parsing hygiene** (SPEC 2.3, 7). What makes those two verbs
preflight in a browser is ``application/json``: a cross-origin ``POST``
carrying ``text/plain`` is on the browser's safelist and is sent with no
preflight at all. A runner that parses whatever it is handed therefore has
no preflight — any page on any origin can start an agent, spending money
and invoking ``write_tools``, before an origin decision has been reached.
That the page cannot read the reply is no consolation; the side effect was
the attack. So the media type is checked **before the body is read**, and
this holds even on a runner nobody will ever point a browser at, because
the runner does not get to know that.

**CORS defaults to refusing every origin.** A runner defines no
authentication, so the origin check is the entirety of its access control
against a browser — and a loopback port is one ``fetch`` away from every
page a user visits. ``--allow-origin`` is how an operator opts one in;
``*`` is a configuration an operator may choose and never one this runner
chooses for them (SPEC 2.3). ``Origin: null`` is refused outright: it is
what sandboxed documents, ``file://`` pages and several redirect chains
all send, so admitting it is a wildcard wearing one origin's clothes.

**A refused origin gets a plain ``204``**, not an error. The browser blocks
the call either way and the page can read neither body, so the envelope
buys nothing — while a ``403`` invites whoever reads the network log to go
looking for an entitlement problem that does not exist.

**A preflight is answered at any conformance level**, so a ``POST`` behind
it arrives and can be refused ``501`` with something a client can act on
(SPEC 2.3). It requires no credentials and no entitlement, and it is not
the verb behind it.

**Inbound authentication is opt-in and off by default** (SPEC 7). This
package ships inside every bundle a buyer downloads, so a runner that
demanded a token nobody had set would refuse its own owner on their own
machine. Setting ``POSTERN_INBOUND_TOKEN`` turns the gate on, and it exists
because SPEC 7 obliges a runner binding a non-loopback interface to
authenticate its callers: the origin check above is access control against
a *browser*, and says nothing about anything else that can reach the port.

Three things about where that gate sits, each of which is the point:

* **Before the route lookup**, so an unauthenticated caller cannot map the
  runner's paths. It also means the gate covers a path this runner does not
  serve, which is the answer that leaks least.
* **Before entitlement**, so a caller who may not talk to this runner at all
  never learns the state of somebody else's licence — ``status`` carries an
  entitlement snapshot and the agent's identifier, and neither is a fact for
  a stranger. It and the ``Host`` check below are the two gates that precede
  ``prepare_run``'s ordering (SPEC 4.6): both settle whether this exchange
  should be happening at all, rather than what the request says.
* **Not on the preflight.** SPEC 2.3 says a preflight requires no
  credentials, and a browser could not send one there in any case.

**``Host`` is checked, and it is the whole of the DNS-rebinding defence.**
A page on ``evil.com`` whose name is re-resolved to ``127.0.0.1``
is *same-origin* with a loopback runner, so the browser applies no CORS and
sends ``application/json`` with no preflight. Both controls above are
satisfied because neither was ever consulted: the run executes, spending
money and invoking ``write_tools``, and the page reads the result. It needs
only that the buyer is running the runner --- its normal state --- and
visits one page.

What the browser cannot forge is ``Host``: ``fetch`` may not set it, and
the value comes from the page's own URL. So a rebound request arrives
naming a host this runner was never told about, and that is the single
thing telling it apart from the buyer's own.

The rule is therefore shaped around what a *browser* can be made to send,
because against anything else it is worth nothing --- a direct client sets
``Host`` to whatever it likes, so refusing one buys nothing there:

* **Any address literal passes.** Rebinding needs a name to re-resolve; an
  address has nothing to re-point, and a browser sends one only for a page
  whose own origin is that address --- which this runner cannot serve, as it
  answers JSON and no document. A LAN runner reached at its own IP is not
  this attack, and refusing it would cost a real deployment to stop nothing.
* **A name passes only if it is ``localhost`` or one the operator named** ---
  the bound ``--host``, or the host of an ``--allow-origin``. Anything else
  is a ``400`` envelope. ``localhost`` is safe to name because a browser
  will not resolve it anywhere but loopback (RFC 6761), and the match is
  exact, so ``evil.localhost`` is a different name and is refused.
* **An absent ``Host`` passes.** A browser always sends one, so its absence
  cannot arrive from the attack; refusing it would enforce HTTP/1.1
  conformance this runner has never enforced, against clients that are not
  the threat.

**A runner that authenticates is exempt, and that is the design rather than
an escape hatch.** This check substitutes for authentication; it does not
add to it. A bearer token defeats rebinding outright --- the attacker's page
cannot read one, and a browser never attaches it --- so ``_authorized``
already refuses that request before this could. Meanwhile the names such a
runner is legitimately reached at are its operator's business and were
never told to this process: the platform's own hosted runners bind
``0.0.0.0`` inside a container and are reached at an ingress FQDN, so
checking ``Host`` there would refuse every hosted run to re-close a hole the
token has already closed. Set against that, the runner this attack is about
is precisely the one with no token --- and SPEC 7 already says that runner
must not be off-machine, which the boot log already warns about.

**Not on the preflight, for the same reason as the token gate.** SPEC 2.3
makes answering ``OPTIONS`` a MUST, a preflight has no side effect, and the
request behind it meets this check anyway.
"""

from __future__ import annotations

import hmac
import http.client
import ipaddress
import json
import logging
import os
import selectors
import socket
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from sigrix_runtime.postern import CONFORMANCE_LEVEL, DEFAULT_HOST, DEFAULT_PORT, PATH_PREFIX, POSTERN_VERSION, errors
from sigrix_runtime.postern import describe as describe_module
from sigrix_runtime.postern import entitlement as entitlement_module
from sigrix_runtime.postern import mcp as mcp_module
from sigrix_runtime.postern import transport as transport_module
from sigrix_runtime.postern import version_check as version_check_module
from sigrix_runtime.postern.engine import Engine, Limits

logger = logging.getLogger("sigrix_runtime.postern")

JSON_MEDIA_TYPE = "application/json"

# A ``run`` body is a small map of declared inputs. Bounded because reading
# a client-declared length into memory is otherwise a one-line denial of
# service against a process the buyer is relying on.
MAX_REQUEST_BYTES = 1 * 1024 * 1024

# How long a connection may go silent -- mid-request, or idle between two on
# a kept-alive one -- before it is dropped. Never reached by a client on the
# same machine; reached by one that declared a body and stopped sending it,
# which otherwise held its thread for good.
SOCKET_TIMEOUT_SECONDS = 30

# Connections served at once, each a thread. CORS gates reading a reply, not
# opening a socket, so any page a buyer visits can open these.
MAX_CONNECTIONS = 64

# ``http.server`` refuses these itself, before ``do_*``. Reworded rather than
# relayed: its own messages quote the caller's request line back.
_UNREADABLE: dict[int, str] = {
    HTTPStatus.REQUEST_URI_TOO_LONG: "The request line is longer than this runner reads.",
    HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE: "The request's headers exceed what this runner reads.",
    HTTPStatus.HTTP_VERSION_NOT_SUPPORTED: "This runner speaks HTTP/1.1.",
}

# A logged request line is the caller's text, so its control characters go
# out as escapes -- as ``http.server``'s own ``log_message`` writes them.
_LOG_ESCAPES = str.maketrans({code: f"\\x{code:02x}" for code in (*range(0x20), *range(0x7F, 0xA0))})

# A client that has gone, as a write or read meets it: a broken pipe, a reset
# or aborted connection (all ``ConnectionError``), or a write it stopped
# reading for a whole socket timeout. ``BrokenPipeError`` alone missed the
# reset an ordinary tab-close sends.
_CLIENT_GONE = (ConnectionError, TimeoutError)

# How often a run's disconnect watch looks up to see whether the run is over.
_WATCH_TICK_SECONDS = 0.5

# Cached by the browser, so a user waiting on a run is not also waiting on
# a preflight for every one of them (SPEC 2.3).
PREFLIGHT_MAX_AGE_SECONDS = 600

# How much of a refused ``Host`` is quoted back and logged. The value is the
# caller's, so it is bounded before it reaches either -- a header can be long,
# and a log line nobody can read is the diagnostic lost.
MAX_QUOTED_HOST_CHARS = 80

# Bind addresses that name no host at all. An operator who asked for every
# interface has declined to say what they are reached at, so there is nothing
# here to match a ``Host`` against -- ``0.0.0.0`` still passes as an address
# literal, which is the rule above and not this one.
_WILDCARD_BIND_ADDRESSES = frozenset({"", "*", "0.0.0.0", "::", "[::]"})  # noqa: S104 - recognising a bind, not making one

# What a preflight admits. ``Authorization`` is added only by a runner that
# reads one (SPEC 2.3): it is off the browser's safelist, so naming it makes
# a client preflight a ``describe`` that would otherwise go without one — and
# a runner requiring no token would be buying that for a header it ignores.
# ``Idempotency-Key`` is named "where the runner honours it" (SPEC 2.3), and
# nothing here reads it yet: admitting it told a page its retry was safe.
_SAFE_REQUEST_HEADERS = "Content-Type"
_AUTHENTICATED_REQUEST_HEADERS = f"{_SAFE_REQUEST_HEADERS}, Authorization"


@dataclass
class RunnerConfig:
    """Everything the server needs, resolved once at startup."""

    bundle_root: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    allowed_origins: tuple[str, ...] = ()
    allow_any_origin: bool = False
    limits: Limits = field(default_factory=Limits)
    agent_id: str = ""
    # SPEC 7: what a caller must present to reach any of the four verbs.
    # Empty is no gate, which is what a bundle on a buyer's own machine wants;
    # `build_runner` fills it from the environment when the caller left it
    # unset, so an operator who sets the variable gets the gate whichever
    # entry point started the runner.
    inbound_token: str = ""
    # SPEC 8: the boot-time version check's held result. ``None``
    # rather than defaulting to an ``UpdateCheck`` means "checked-on-start
    # never ran" (a caller that built this config by hand, chiefly tests) —
    # distinct from ``not_required``, which means it ran and found no
    # distributor configured.
    update_check: version_check_module.UpdateCheck | None = None
    # Serve this MCP server's tools rather than the bundle.
    mcp: mcp_module.McpServer | None = None


@dataclass
class PreparedRun:
    """What ``prepare_run`` hands the engine: a bundle's prompt and variables, or an MCP call."""

    prompt: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    mcp: dict[str, Any] | None = None


class Runner:
    """The four verbs, as plain functions over a bundle.

    Separated from the HTTP handler so the protocol can be exercised
    without a socket, and so the container entrypoint and ``python -m``
    share one object rather than one server class.
    """

    def __init__(self, config: RunnerConfig, entitlement: entitlement_module.Entitlement) -> None:
        self.config = config
        self.entitlement = entitlement
        self.engine = Engine(config.bundle_root, limits=config.limits)
        self._describe_lock = threading.Lock()
        self._describe: dict[str, Any] | None = None
        self.toolbox: mcp_module.Toolbox | None = None

    # --- describe ----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """SPEC 4.1. Side-effect free, and answerable with nothing at all.

        Read once and held: the document is a property of the bundle, and
        re-reading it per request would make a verb the specification calls
        side-effect free into a disk read a client can trigger at will.
        """
        if self.config.mcp is not None:
            if self.toolbox is None:
                raise errors.unavailable("This runner has not read its MCP server's tools yet.")
            return self.toolbox.document
        with self._describe_lock:
            if self._describe is None:
                # The distributor client's identifier when the config carries
                # none. They are the same value from the same source, and the
                # one place they could diverge is a caller that constructed
                # this object directly — where a ``describe`` naming one agent
                # beside a check asking about another is not worth allowing.
                agent_id = self.config.agent_id or self.entitlement.agent_id
                self._describe = describe_module.load_describe(self.config.bundle_root, agent_id=agent_id)
            return self._describe

    def load_tools(self, *, timeout: float = mcp_module.LIST_TIMEOUT_SECONDS) -> None:
        """An MCP runner's ``describe``: the server's ``tools/list``, read once at boot and held."""
        server = self.config.mcp
        if server is None:
            return
        answer = self.engine.inspect(self._mcp_request(server, {"list": True}), timeout=timeout)
        agent_id = self.config.agent_id or self.entitlement.agent_id or mcp_module.local_agent_id(server.command)
        toolbox = mcp_module.Toolbox(answer, agent_id=agent_id)
        for warning in toolbox.warnings:
            logger.warning("Postern: %s", warning)
        if not toolbox.names:
            raise errors.unavailable("The MCP server lists no tools this runner can offer.")
        self.toolbox = toolbox

    def _mcp_request(self, server: mcp_module.McpServer, member: dict[str, Any]) -> dict[str, Any]:
        """A worker request. Only the launcher, which checks the purchase, gets the token."""
        env: dict[str, str] = {}
        if server.launcher:
            gate = self.entitlement
            env = {k: v for k, v in (("SIGRIX_TOKEN", gate.token), ("POSTERN_DISTRIBUTOR", gate.base_url)) if v}
        return {"command": list(server.command), "env": env, **member}

    # --- status ------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """SPEC 4.4. Answers at Level 1 and requires no credentials."""
        document = self.describe()
        missing = describe_module.missing_credentials(document, bundle_root=self.config.bundle_root)
        raw_agent = document.get("agent")
        agent = raw_agent if isinstance(raw_agent, dict) else {}
        body: dict[str, Any] = {
            "postern": POSTERN_VERSION,
            "level": CONFORMANCE_LEVEL,
            "state": self._state(missing),
            "entitlement": self.entitlement.snapshot(),
            "credentials": {"satisfied": not missing, "missing": missing},
            "limits": self.config.limits.as_dict(),
        }
        agent_block = {
            key: value for key, value in (("id", agent.get("id")), ("version", agent.get("version"))) if value
        }
        if agent_block:
            body["agent"] = agent_block
        if self.config.update_check is not None:
            body["update"] = self.config.update_check.as_dict()
        return body

    def _state(self, missing: list[str]) -> str:
        """``running`` observes, ``degraded`` diagnoses, ``ready`` is neither.

        ``running`` is checked first because it is the more specific fact
        and the one a polling client is asking about; the two barely
        overlap in practice, since a run cannot start while a credential it
        declared is missing.
        """
        if self.engine.running:
            return "running"
        if missing:
            return "degraded"
        return "ready"

    # --- run / stream ------------------------------------------------

    def prepare_run(self, read_body: Callable[[], Any]) -> PreparedRun:
        """Gate, validate, and split a request into prompt and variables, or a tool call.

        SPEC 4.6 orders the refusals one ``run`` can earn at once, and this
        function owns steps 2 to 5 of that sequence: the entitlement
        (``403``/``503``), the media type (``400``), the inputs (``400``),
        then the environment (``424``). Step 1, the level check, is moot for
        a runner that declares Level 3 and implements every verb.

        **``read_body`` is a callable, and that is the whole of 2 before 3.**
        The media-type gate lives on the handler, because it reads request
        headers; passing its *result* made it an argument, and Python
        evaluates arguments before the call --- so a mistyped body was
        refused with ``400`` before this function reached the gate at all.
        Taking the reader instead keeps the order in one place rather than in
        every caller, and restoring the parentheses reinstates the bug
        silently. That is why the test driving this goes over a socket:
        the old ordering spanned two functions in different classes, which
        a source scan of this one could never see.

        Neither seam is cosmetic. A revoked runner owes ``not_entitled``
        rather than a ``400`` naming something the caller could fix, since
        SPEC 5.7.4 forbids it to imply the answer might change on a retry.
        And 4 before 5 settles what the request says before this runner
        inspects what it holds, so a request that is both malformed and
        unservable is a ``bad_request`` --- which is what keeps SPEC 4.2
        exercisable on a runner whose environment is still incomplete, the
        ordinary state of one being brought up.
        """
        self.entitlement.gate()
        body = read_body()
        document = self.describe()

        if not isinstance(body, dict):
            raise errors.bad_request("The request body must be a JSON object.")
        resolved = describe_module.validate_run_inputs(document, body.get("inputs"))
        if self.toolbox is not None and self.config.mcp is not None:
            # Still step 4: the chosen tool's own arguments, checked as its schema says.
            prepared = PreparedRun(mcp=self._mcp_request(self.config.mcp, self.toolbox.call_for(resolved)))
        else:
            prompt = resolved.pop("prompt", "")
            if not isinstance(prompt, str):
                raise errors.bad_request("'prompt' must be text.", {"key": "prompt"})
            variables = {key: value for key, value in resolved.items() if value is not None}
            prepared = PreparedRun(prompt=prompt, variables=variables)

        missing = describe_module.missing_credentials(document, bundle_root=self.config.bundle_root)
        if missing:
            raise errors.missing_credential(missing)
        return prepared


class _Handler(BaseHTTPRequestHandler):
    """One connection and each request it carries. The runner is on the server object."""

    protocol_version = "HTTP/1.1"
    server_version = "postern-sigrix-runner/0.1"
    timeout = SOCKET_TIMEOUT_SECONDS

    # --- plumbing ----------------------------------------------------

    @property
    def runner(self) -> Runner:
        return cast("Runner", self.server.runner)  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s %s", self.address_string(), (fmt % args).translate(_LOG_ESCAPES))

    def version_string(self) -> str:
        # ``Server`` names this runner, not the interpreter the stdlib appends
        # (``Python/3.11.15``), which tells a caller only what to try.
        return self.server_version

    def handle_one_request(self) -> None:
        # Reset per request rather than per connection: a kept-alive one
        # carries several, and a refusal made before the headers are parsed
        # must not read the previous request's.
        self.headers = http.client.HTTPMessage()
        self._body_read = False
        super().handle_one_request()

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """What ``http.server`` refuses before ``do_*``, in the envelope.

        A malformed request line, ``HTTP/2.0``, over 100 headers, an over-long
        line: the stdlib answers each with an HTML page -- and, before it has
        read a version, with no status line either. SPEC 2.1's table has none
        of its 414, 431 or 505, so each is ``bad_request``, and the last answer
        on the connection: nothing says where the next request would start.

        A method with no ``do_*`` is a request that parsed, so it is
        dispatched: the ``Host`` and token gates see it before any answer
        names a route, and the answer is ``_no_such_route``'s ``404``.
        """
        if code == HTTPStatus.NOT_IMPLEMENTED and self.command:
            self._dispatch(self.command)
            return
        if self.request_version == "HTTP/0.9":
            self.request_version = self.protocol_version
        self.close_connection = True
        self._send_error(errors.bad_request(_UNREADABLE.get(code, "This runner could not read that request.")))

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:  # noqa: N802
        route = self._route()
        methods = _PREFLIGHT_METHODS.get(route)
        if methods is None:
            self._send_error(errors.not_found())
            return
        # SPEC 2.3: side-effect free, no credentials, no entitlement, and
        # independent of conformance level.
        self._send(204, b"", extra_headers=self._preflight_headers(methods))

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("GET", head=True)

    def _dispatch(self, method: str, *, head: bool = False) -> None:
        # First, because it asks whether the request is addressed to this
        # runner at all -- every other question here presupposes that it is.
        # The two gates are mutually exclusive by construction (a runner with
        # a token is exempt from this one, and one without authorises
        # everything), so the order below is a statement rather than a
        # behaviour; it is the reading that would otherwise be reconstructed
        # wrongly.
        refusal = self._host_refusal()
        if refusal is not None:
            self._send_error(refusal)
            return
        if not self._authorized():
            self._send_error(errors.unauthorized())
            return
        route = self._route()
        handler = _ROUTES.get((method, route))
        if handler is None:
            self._send_error(_no_such_route(route, method))
            return
        try:
            handler(self, head)
        except errors.PosternError as exc:
            self._send_error(exc)
        except _CLIENT_GONE:
            # The client went away mid-response. SPEC 4.5 calls that a
            # cancelled run, and the engine has already killed the worker
            # on the way out of the generator.
            logger.info("postern: client disconnected")
        except Exception:  # noqa: BLE001 - never leak a traceback to a client
            logger.exception("postern: unhandled failure serving %s %s", method, route)
            self._send_error(errors.unavailable("This runner hit an internal error. Try again shortly."))

    def _route(self) -> str:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if not path.startswith(PATH_PREFIX):
            return path
        return path[len(PATH_PREFIX) :] or "/"

    # --- verbs -------------------------------------------------------

    def _handle_describe(self, head: bool) -> None:
        self._send_json(200, self.runner.describe(), head=head)

    def _handle_status(self, head: bool) -> None:
        self._send_json(200, self.runner.status(), head=head)

    def _handle_run(self, head: bool) -> None:
        events, run_id = self._start_run()
        body: dict[str, Any] | None = None
        try:
            next(events)  # `start`: the worker is running, so from here it can be stopped
            with self._watching(run_id) as watch:
                try:
                    for event in events:
                        if event.name == "done":
                            body = event.payload
                except errors.PosternError:
                    if not watch.gone.is_set():
                        raise
            if watch.gone.is_set():
                return  # nobody is left to answer, and SPEC 4.5 allows no one else
        finally:
            events.close()
        if body is None:  # pragma: no cover - stream raises or yields done
            raise errors.agent_error("The agent produced no result.")
        self._send_json(200, body, head=head)

    def _handle_stream(self, head: bool) -> None:
        events, run_id = self._start_run()

        # Nothing is committed until the first event is written, so a
        # failure raised while starting the run is still answerable with a
        # status code. Past that point the response is a 200 and SPEC 4.5
        # requires the failure to arrive as an `error` event instead.
        try:
            first = next(events)
        except StopIteration:  # pragma: no cover - start is always yielded
            raise errors.agent_error("The agent produced no result.") from None

        try:
            self._begin_stream(head=head)
            if head:
                return
            with self._watching(run_id) as watch:
                try:
                    self._write_event(first.name, first.payload)
                    for event in events:
                        self._write_event(event.name, event.payload)
                except errors.PosternError as exc:
                    if not watch.gone.is_set():
                        self._write_event("error", exc.envelope())
                except _CLIENT_GONE:
                    raise
                except Exception as exc:  # noqa: BLE001 - the stream owes exactly one ending
                    logger.exception("postern: stream failed")
                    self._write_event("error", errors.agent_error(f"{type(exc).__name__}: {exc}").envelope())
        finally:
            events.close()

    def _start_run(self) -> tuple[Generator[Any, None, None], str]:
        run = self.runner.prepare_run(self._read_json_body)
        run_id = self.runner.engine.new_run_id()
        engine = self.runner.engine
        events = engine.stream(
            prompt=run.prompt, variables=run.variables, mcp=run.mcp, run_id=run_id, on_notice=_log_notice
        )
        return events, run_id

    @contextmanager
    def _watching(self, run_id: str) -> Iterator[_DisconnectWatch]:
        """Stop ``run_id`` if its client leaves while it is in flight (SPEC 4.5)."""
        watch = _DisconnectWatch(self.connection, on_gone=lambda: self._client_left(run_id))
        try:
            yield watch
        finally:
            watch.stop()

    def _client_left(self, run_id: str) -> None:
        logger.info("postern: the client left; stopping run %s.", run_id)
        self.runner.engine.cancel(run_id)

    # --- request ------------------------------------------------------

    def _read_json_body(self) -> Any:
        """The body, after the media-type gate. Refuses before reading it.

        Parameters do not enter into it: ``application/json`` and
        ``application/json; charset=utf-8`` are the same media type, and
        SPEC 2 requires a client to send the second without making the
        first nonconformant to receive.

        A refusal here leaves the body unread, and ``_send`` then closes the
        connection rather than read those bytes as the next request.
        """
        media_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if media_type != JSON_MEDIA_TYPE:
            raise errors.bad_request(
                f"This request must be sent as {JSON_MEDIA_TYPE}."
                + (f" It arrived as {media_type!r}." if media_type else " It arrived with no Content-Type.")
            )
        if self.headers.get("Transfer-Encoding") is not None:
            raise errors.bad_request("Send the request body with a Content-Length and no Transfer-Encoding.")
        length = self._declared_length()
        if length > MAX_REQUEST_BYTES:
            raise errors.bad_request(f"The request body must be {MAX_REQUEST_BYTES} bytes or fewer.")
        try:
            raw = self.rfile.read(length) if length else b""
        except TimeoutError:
            raise errors.bad_request(f"The request body stopped arriving for {self.timeout} seconds.") from None
        self._body_read = True
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise errors.bad_request(f"The request body is not valid JSON: {exc}") from None

    def _declared_length(self) -> int:
        """``Content-Length`` as RFC 9110 writes it: one value, ASCII digits.

        ``int()`` alone read ``1_3`` and ``+13`` as thirteen, and two
        headers that disagree is request smuggling's shape.
        """
        values = self.headers.get_all("Content-Length") or ["0"]
        raw = values[0].strip(" \t")
        if len(values) > 1 or not (raw.isascii() and raw.isdigit()):
            raise errors.bad_request("Content-Length must be one number of bytes, written in digits.")
        return int(raw) if len(raw) <= 12 else MAX_REQUEST_BYTES + 1

    def _body_unread(self) -> bool:
        """Whether this request may carry body bytes that nothing has read.

        Any length but ``0`` counts, a malformed one included: its bytes are
        on the wire whether or not the header could be read.
        """
        if self._body_read:
            return False
        lengths = self.headers.get_all("Content-Length") or []
        return any(value.strip(" \t") != "0" for value in lengths) or self.headers.get("Transfer-Encoding") is not None

    # --- response -----------------------------------------------------

    def _send_json(self, status: int, payload: dict[str, Any], *, head: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, content_type=f"{JSON_MEDIA_TYPE}; charset=utf-8", head=head)

    def _send_error(self, error: errors.PosternError) -> None:
        try:
            self._send_json(error.status, error.envelope())
        except _CLIENT_GONE:  # pragma: no cover - the client is gone
            pass

    def _send(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str = "",
        extra_headers: dict[str, str] | None = None,
        head: bool = False,
    ) -> None:
        self.send_response(status)
        if self.close_connection or self._body_unread():
            # Bytes nobody read would be parsed as the next request on a
            # kept-alive connection: a refused ``text/plain`` body was, and
            # the client's next ``GET`` got an HTML 400 about it.
            self.send_header("Connection", "close")
        if content_type:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        for name, value in self._cors_headers().items():
            self.send_header(name, value)
        self.end_headers()
        if body and not head:
            self.wfile.write(body)

    def _begin_stream(self, *, head: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        # A proxy that buffers turns a progress stream into a single late
        # delivery, which is the one thing `stream` exists not to be.
        self.send_header("X-Accel-Buffering", "no")
        for name, value in self._cors_headers().items():
            self.send_header(name, value)
        self.end_headers()
        # No Content-Length is possible, so the framing is the connection
        # itself; http.server would otherwise keep it alive and wait.
        self.close_connection = True

    def _write_event(self, name: str, payload: dict[str, Any]) -> None:
        frame = f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()

    # --- inbound authentication (SPEC 7) --------------------------------

    def _authorized(self) -> bool:
        """``True`` unless this runner demands a token the request lacks.

        A runner with no configured token authorises everything, which is
        what keeps a bundle on a buyer's own machine behaving exactly as it
        did — the whole of the opt-in.

        The scheme name is matched case-insensitively (RFC 9110 makes it so)
        and the token octet-for-octet, through :func:`hmac.compare_digest`,
        so the time this takes says nothing about how much of a guess was
        right.

        **The comparison is over bytes, and that is not tidiness.**
        ``compare_digest`` raises ``TypeError`` on a ``str`` carrying any
        non-ASCII character, and this one is supplied by the caller —
        ``http.server`` decodes a header as latin-1, so a single high byte in
        an ``Authorization`` value produces one. This check runs *before*
        ``_dispatch``'s ``try``, so that raise would escape to
        ``socketserver``, which logs a traceback and drops the connection:
        anyone could turn a refusal into a dropped request by sending one
        junk byte. Encoding back with latin-1 recovers exactly the bytes that
        arrived on the wire, and a refusal stays a refusal.

        A token outside ASCII cannot be carried by a conformant header at all
        (RFC 9110's ``token68``), so an operator who configures one has built
        a runner nothing can reach. That is their error to see rather than
        one to paper over by matching approximately.
        """
        expected = self.runner.config.inbound_token
        if not expected:
            return True
        scheme, _, presented = (self.headers.get("Authorization") or "").partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(
            presented.strip().encode("latin-1", "replace"), expected.encode("utf-8")
        ):
            return True
        # Never the presented value: a log is read by more people than a
        # response is, and this one would carry a working credential whenever
        # the failure was a stray character rather than an attack.
        logger.warning(
            "postern: refused an unauthenticated request from %s (%s)",
            self.address_string(),
            f"{self.command} {self._route()}".translate(_LOG_ESCAPES),
        )
        return False

    # --- DNS rebinding (Host) -------------------------------------------

    def _host_refusal(self) -> errors.PosternError | None:
        """``None`` unless ``Host`` names something this runner was not told of.

        The module docstring carries the reasoning; this is the shape of it.
        A runner requiring a token has the stronger control and is exempt,
        an absent header and any address literal pass, and a name passes
        only when it is ``localhost`` or one the operator declared.

        Deliberately not routed through :func:`_is_loopback_host`, which
        answers about a *bind* address -- where ``""`` and ``0.0.0.0`` mean
        "every interface" and a literal would have to be rejected to earn the
        SPEC 7 warning. Here they are neither, and the collapse would tie two
        questions whose wildcard readings are opposite.
        """
        config = self.runner.config
        if config.inbound_token:
            return None
        supplied = self.headers.get_all("Host") or []
        if not supplied:
            return None
        quoted = _clip(supplied[0])
        if len(supplied) > 1:
            # RFC 9112 3.2 makes more than one ``Host`` a MUST-reject, and the
            # reason is this gate's own: with two, "the host this was addressed
            # to" has two answers, and a check that silently takes one is a
            # check on a value nobody sent. Nothing legitimate sends two --- a
            # browser cannot, ``Host`` being forbidden to ``fetch`` --- so this
            # costs no client anything. Measured before it was written: without
            # it, a second header was accepted on the strength of the first.
            return self._refuse_host(quoted, f"A request must carry one Host header; this one carried {len(supplied)}.")
        host = _host_label(supplied[0])
        if not host or _is_address_literal(host) or host == "localhost":
            return None
        if host in _declared_hosts(config):
            return None
        return self._refuse_host(
            quoted,
            f"This runner does not answer to {quoted!r}. It answers on loopback, at any address "
            "literal, and at a host named by --host or --allow-origin. A runner reached by name "
            "must authenticate its callers (SPEC 7): set POSTERN_INBOUND_TOKEN.",
        )

    def _refuse_host(self, quoted: str, message: str) -> errors.PosternError:
        """Say it in the log and in the envelope.

        At ``WARNING`` because this is the buyer's only signal that a page
        tried it, rather than the ``INFO`` a merely wrong route earns.

        **The ``%r`` is load-bearing, not formatting.** An obs-fold
        continuation survives header parsing as an embedded carriage return
        and newline --- measured, not assumed --- so a bare ``%s`` would let
        whoever sends one write their own records into this log. Truncation is
        the separate protection beside it: quoting a header back at its full
        length makes a refusal into a megaphone.
        """
        logger.warning(
            "postern: refused %s %s addressed to %r. A page that rebound its own name to "
            "this machine looks exactly like this.",
            self.command,
            self._route(),
            quoted,
        )
        return errors.bad_request(message, {"host": quoted})

    # --- CORS ---------------------------------------------------------

    def _allowed_origin(self) -> str:
        origin = (self.headers.get("Origin") or "").strip()
        if not origin or origin.lower() == "null":
            return ""
        config = self.runner.config
        if config.allow_any_origin:
            return origin
        return origin if origin in config.allowed_origins else ""

    def _cors_headers(self) -> dict[str, str]:
        """Ride the *actual* response, not only the preflight (SPEC 2.3).

        The two are refused separately: a ``run`` whose response arrives
        without the header is discarded by the browser exactly as an
        unpermitted one would be — the agent having run. ``Vary: Origin``
        goes on the real response too, so a shared cache in front of the
        runner cannot hand the first caller's permission to the second.
        """
        headers = {"Vary": "Origin"}
        origin = self._allowed_origin()
        if origin:
            headers["Access-Control-Allow-Origin"] = origin
        return headers

    def _preflight_headers(self, methods: str) -> dict[str, str]:
        if not self._allowed_origin():
            # A refused preflight is a bare 204 (see the module docstring).
            return {}
        return {
            "Access-Control-Allow-Methods": methods,
            "Access-Control-Allow-Headers": (
                _AUTHENTICATED_REQUEST_HEADERS if self.runner.config.inbound_token else _SAFE_REQUEST_HEADERS
            ),
            "Access-Control-Max-Age": str(PREFLIGHT_MAX_AGE_SECONDS),
        }


class _DisconnectWatch:
    """Notices a client leaving while its run is in flight (SPEC 4.5).

    A write is too late to be the signal: ``run`` writes nothing until it
    ends and ``stream`` only between steps, so a client gone during a long
    model call was noticed when the call returned -- or, on ``run``, never,
    and the worker ran to the end for nobody. The socket says so at once.
    The request is read before this starts and nothing more is owed on the
    connection, so its turning readable means the peer closed or reset it.
    Data instead is a pipelined request: no departure, and nothing more to
    learn here. A half-close reads as a close; only a write could tell the
    two apart, and treating the end of input as the client leaving is what
    a proxy does too.
    """

    def __init__(self, sock: socket.socket, *, on_gone: Callable[[], None]) -> None:
        self.gone = threading.Event()
        self._stopped = threading.Event()
        self._sock = sock
        self._on_gone = on_gone
        threading.Thread(target=self._watch, name="postern-disconnect-watch", daemon=True).start()

    def stop(self) -> None:
        self._stopped.set()

    def _watch(self) -> None:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self._sock, selectors.EVENT_READ)
                while not self._stopped.is_set():
                    if not selector.select(_WATCH_TICK_SECONDS):
                        continue
                    if self._sock.recv(1, socket.MSG_PEEK):
                        return
                    break
        except (OSError, ValueError):
            pass  # reset, or already closed: gone either way
        if not self._stopped.is_set():
            self.gone.set()
            self._on_gone()


def _is_loopback_host(host: str) -> bool:
    """Whether a *bind* address is loopback-only.

    Deliberately not :func:`transport._peer_is_loopback`, which answers about a
    connected peer and is the rule SPEC 7 states for carrying a token outward.
    This is the other direction and a different question — what this process
    told the kernel to listen on — and the two must not be collapsed: the
    empty string and ``0.0.0.0`` mean "every interface" here and are not peer
    addresses at all.
    """
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        # A name rather than an address. A resolver decides what it means, and
        # SPEC 7 says not to settle this by matching a hostname — so the only
        # safe reading of an unresolvable answer is the one that warns.
        return host.strip().lower() == "localhost"


def _host_label(raw: str) -> str:
    """The name or address in a ``Host`` header, with the port taken off.

    Bracketed IPv6 is read first, because its own colons are not a port's --
    and a bare ``::1`` (which no conformant client sends, and which arrives
    all the same) is left whole for the same reason: more than one colon and
    no brackets cannot be split on the port without eating an address.

    A trailing dot is the DNS root label and names the same host, so it comes
    off; otherwise ``127.0.0.1.`` would read as a name rather than an address.
    """
    host = (raw or "").strip()
    if host.startswith("["):
        host = host[1 : host.index("]")] if "]" in host else host[1:]
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host.rstrip(".").strip().lower()


def _is_address_literal(host: str) -> bool:
    """Whether a ``Host`` names an address rather than something resolvable.

    An address literal is accepted wholesale because rebinding has nothing to
    work with there -- see the module docstring. Note this is *not* asking
    whether the address is loopback: a runner an operator reached at its LAN
    address is not an attack, and pretending otherwise would refuse it.
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _declared_hosts(config: RunnerConfig) -> frozenset[str]:
    """The names the operator told this runner it would be reached at.

    The bound ``--host`` when it names something (a wildcard names nothing),
    and the host of every ``--allow-origin`` -- an operator serving a page
    from the same name they run the runner at has declared that name, and the
    ordinary cross-origin shape is unaffected either way, since a page calling
    a loopback runner sends ``Host: 127.0.0.1`` whatever its own origin is.

    ``*`` is not a bypass. It says which origins may *read* a reply; it says
    nothing about what this runner is called, and the attack it would reopen
    does not need it.
    """
    names = set()
    bound = _host_label(config.host)
    if bound and config.host.strip() not in _WILDCARD_BIND_ADDRESSES:
        names.add(bound)
    for origin in config.allowed_origins:
        hostname = urlsplit(origin.strip()).hostname
        if hostname:
            names.add(hostname.rstrip(".").lower())
    return frozenset(names)


def _clip(value: str) -> str:
    """A caller's string, bounded, for a log line and an error body."""
    text = (value or "").strip()
    return text if len(text) <= MAX_QUOTED_HOST_CHARS else text[:MAX_QUOTED_HOST_CHARS] + "..."


def _log_notice(text: str) -> None:
    if text:
        logger.info("postern: %s", text)


def _no_such_route(route: str, method: str) -> errors.PosternError:
    """SPEC 2.1: a runner's ``404`` is "a path it does not implement".

    A defined verb reached with the wrong method is exactly that, and
    answering it with a code outside the specification's table would hand a
    client something it has no rule for. The message names the method that
    would have worked, which is the part that makes this diagnosable.
    """
    expected = _PREFLIGHT_METHODS.get(route)
    if expected is not None:
        wanted = expected.split(",")[0].strip()
        return errors.not_found(f"{route} is served over {wanted}, not {method}.")
    return errors.not_found(f"This runner serves {PATH_PREFIX}/describe, /run, /stream and /status.")


_ROUTES: dict[tuple[str, str], Any] = {
    ("GET", "/describe"): _Handler._handle_describe,
    ("GET", "/status"): _Handler._handle_status,
    ("POST", "/run"): _Handler._handle_run,
    ("POST", "/stream"): _Handler._handle_stream,
}

# The methods each route answers, which is also what a preflight is told.
_PREFLIGHT_METHODS: dict[str, str] = {
    "/describe": "GET, OPTIONS",
    "/status": "GET, OPTIONS",
    "/run": "POST, OPTIONS",
    "/stream": "POST, OPTIONS",
}


class PosternServer(ThreadingHTTPServer):
    """One agent, several clients. Threaded so a run cannot block a status."""

    daemon_threads = True
    # A restart on the same port during development should not have to wait
    # out TIME_WAIT; there is one agent per runner and nothing to collide
    # with (SPEC 2.2).
    allow_reuse_address = True

    def __init__(self, config: RunnerConfig, runner: Runner, *, max_connections: int = MAX_CONNECTIONS) -> None:
        self.runner = runner
        self.max_connections = max_connections
        self._connections = threading.BoundedSemaphore(max_connections)
        self._quiet_until = 0.0
        super().__init__((config.host, config.port), _Handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        """A thread per connection, and at most ``max_connections`` of them.

        One past the bound is closed unserved rather than queued, since a queue
        is a thread by another name. Said at most once a timeout's length: a
        page opening sockets in a loop would otherwise be writing the log.
        """
        if not self._connections.acquire(blocking=False):
            if time.monotonic() >= self._quiet_until:
                self._quiet_until = time.monotonic() + SOCKET_TIMEOUT_SECONDS
                logger.warning(
                    "postern: %d connections are open, the most this runner serves; closing new ones until one ends.",
                    self.max_connections,
                )
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


def build_runner(config: RunnerConfig, environ: dict[str, str] | None = None) -> Runner:
    """A :class:`Runner` with its distributor client resolved from the environment."""
    if not config.inbound_token:
        # Filled rather than overridden, and filled *here* rather than in
        # ``__main__``, so that the gate is on for every entry point that
        # builds a runner from the environment — an embedder who set the
        # variable and called ``serve()`` directly would otherwise get a
        # runner that reads as protected and is not. A caller who set the
        # field explicitly has already answered the question.
        config.inbound_token = entitlement_module.inbound_token_from_environment(config.bundle_root, environ)
    # The configured identifier wins outright, rather than only filling a gap.
    # ``__main__`` has already resolved it from the argument, the flag and the
    # environment in that order, and a gate checking a different listing than
    # the one this runner is serving is the one disagreement nothing
    # downstream could notice. Handed in, not set after: the persisted answer
    # is read at construction, keyed on it.
    gate = entitlement_module.from_environment(config.bundle_root, environ, agent_id=config.agent_id)
    if config.mcp is not None and not (environ if environ is not None else os.environ).get("SIGRIX_DELIVERY_MODE"):
        # A local runner fronting a connector: the launcher's own shape.
        gate.delivery_mode = entitlement_module.DELIVERY_MODE_CONNECTOR_LOCAL
    if not gate.listing_url:
        # The one address a refused buyer can be sent to: the
        # bundle's own ``plugin.json`` names its listing page, and nothing
        # in the environment does.
        gate.listing_url = describe_module.listing_url(config.bundle_root)
    return Runner(config, gate)


def serve(config: RunnerConfig, environ: dict[str, str] | None = None) -> None:
    """Bind, announce, and serve until interrupted."""
    runner = build_runner(config, environ)
    if config.mcp is not None:
        # Before the port is announced, so there is something to describe.
        logger.info("Postern: starting %s to read its tools.", " ".join(config.mcp.command))
        runner.load_tools()
    server = PosternServer(config, runner)
    host, port = cast("tuple[str, int]", server.server_address[:2])
    # SPEC 2: a runner launched as a subprocess reports its port on stdout
    # as a single line before any other output, so a parent that asked for
    # port 0 can find it. Harmless when nobody is reading.
    sys.stdout.write(f"POSTERN_PORT={port}\n")
    sys.stdout.flush()
    served = " ".join(config.mcp.command) if config.mcp is not None else config.bundle_root
    logger.info("Postern v%s serving %s on http://%s:%s%s", POSTERN_VERSION, served, host, port, PATH_PREFIX)
    if config.inbound_token:
        logger.info("Postern: inbound requests must carry POSTERN_INBOUND_TOKEN as a bearer token.")
    elif not _is_loopback_host(host):
        # SPEC 7's MUST, reported rather than enforced: refusing to bind would
        # take the decision off an operator who may have put their own gateway
        # in front, and this runner cannot see one. Said at WARNING because the
        # runner looks identical either way, and `status` deliberately carries
        # nothing about it — the person who can fix this is reading the boot log.
        logger.warning(
            "Postern: serving %s with no inbound authentication. SPEC 7 requires a runner "
            "bound off-machine to authenticate its callers; set POSTERN_INBOUND_TOKEN.",
            host,
        )
    if runner.entitlement.configured and runner.entitlement.base_url.startswith("http://"):
        # SPEC 7's SHOULD. Deliberately not a field in `status`: a client
        # cannot fix a distributor base URL, and the person who can is not
        # reading `status`.
        logger.warning("%s", transport_module.plaintext_notice(runner.entitlement.base_url))
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        # Each worker leads its own process group, so the Ctrl+C that
        # stopped this process did not reach it: stopped here instead.
        runner.engine.stop_all()
        server.shutdown()
        server.server_close()


__all__ = [
    "JSON_MEDIA_TYPE",
    "MAX_CONNECTIONS",
    "MAX_REQUEST_BYTES",
    "SOCKET_TIMEOUT_SECONDS",
    "PosternServer",
    "PreparedRun",
    "Runner",
    "RunnerConfig",
    "build_runner",
    "serve",
]
