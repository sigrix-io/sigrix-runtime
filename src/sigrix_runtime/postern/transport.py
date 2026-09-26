"""The one connection this runner makes outward, and the rules on it.

Two callers reach the distributor — the entitlement check (SPEC 5.3) and
the bundle pull (SPEC 5.6) — and they ask for different things: a sentence
of JSON that fits in memory twice over, and a zip that has to be read with
a bound on it. What they must **not** differ about is who they are willing
to hand a bearer token to, which is why that decision lives here rather
than once per caller.

**A token crosses plaintext only to a loopback peer** (SPEC 7), and the
condition is the address rather than the name. ``localhost`` is a name and
a resolver decides what it means; this module opens the connection first
and reads the peer address off the socket, so a name that resolved to
loopback and then connected elsewhere cannot carry a token. Written twice
that rule would be relaxed once by accident — the pull is the caller most
likely to be pointed at a mirror during development, and it is the caller
carrying the same token as the check.

**Everything that goes wrong out here is one exception.** ``TransportError``
means *we did not get an answer*, and each caller translates it into its
own vocabulary: the check calls it unreachable and keeps its previous
answer (SPEC 5.7), the pull calls it unavailable and invites a retry.
Neither treats it as a refusal, because a refusal is something a
distributor says rather than something a socket does.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

#: Sent on both requests. One string, so a distributor reading its logs sees
#: one client rather than two that happen to agree.
USER_AGENT = "postern-sigrix-runner/0.1"

#: The most a distributor's JSON answer may be. A check or a version answer is
#: a few hundred bytes (SPEC 5.3, 8); the bound is for a distributor that sends
#: more, since this runner can be pointed at any -- a 60 MB check body cost it
#: 367 MB of memory, and a larger one would have taken it down.
MAX_ANSWER_BYTES = 64 * 1024


class TransportError(RuntimeError):
    """No answer was obtained — DNS, TLS, a refused connection, a timeout.

    Also raised for a base URL this module will not speak to at all: an
    unsupported scheme, no host, or plaintext to a peer that is not
    loopback. Those are refusals by *this* runner rather than by the
    distributor, and they belong on the same side of the line as an
    unreachable endpoint: nothing was learned about the entitlement.
    """


@contextmanager
def open_response(
    base_url: str,
    path: str,
    *,
    token: str = "",
    accept: str,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
) -> Iterator[http.client.HTTPResponse]:
    """``GET base_url + path``, as a context manager.

    ``token`` is sent as a bearer header when given. It defaults to empty for
    the one caller that has none — the version check (SPEC 8) is deliberately
    unauthenticated — rather than have that caller invent a value to satisfy
    a required parameter.

    ``extra_headers`` is for headers a caller wants sent alongside the fixed
    two (``Accept``, ``User-Agent``) — today that's the entitlement check's
    ``X-Sigrix-Delivery-Mode``. Not part of the specification, so it
    stays out of ``USER_AGENT`` and out of every other caller's request.

    The response is yielded with its connection still open so a caller can
    read the body in bounded chunks rather than in one string — the pull
    reads megabytes and must be able to stop early. The connection closes
    when the block exits, however it exits.
    """
    split = urlsplit(base_url)
    host, port, scheme = split.hostname or "", split.port, (split.scheme or "https").lower()
    if not host:
        raise TransportError(f"distributor base URL has no host: {base_url!r}")
    prefix = split.path.rstrip("/")

    try:
        if scheme == "https":
            connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port, timeout=timeout, context=ssl.create_default_context()
            )
        elif scheme == "http":
            connection = http.client.HTTPConnection(host, port, timeout=timeout)
        else:
            raise TransportError(f"unsupported distributor scheme {scheme!r}")
        connection.connect()
    except TransportError:
        raise
    except (OSError, ssl.SSLError) as exc:
        raise TransportError(f"could not reach {base_url}: {exc}") from exc

    try:
        if scheme == "http" and token and not _peer_is_loopback(connection.sock):
            # SPEC 7. Refused after connecting rather than before, because
            # the rule is about the address actually reached: a name that
            # resolves to loopback and then connects elsewhere would pass a
            # check made on the hostname. A request carrying no token has
            # nothing for the rule to protect -- the version check.
            raise TransportError(
                f"refusing to send the distributor token to {base_url} over plaintext HTTP "
                "— the connection did not land on a loopback address"
            )
        headers = {"Accept": accept, "User-Agent": USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if extra_headers:
            headers.update(extra_headers)
        connection.request("GET", prefix + path, headers=headers)
        yield connection.getresponse()
    except TransportError:
        raise
    except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
        # Raised from the caller's own read as well as from ours: a
        # connection that dies halfway through a bundle is unreachable,
        # not a distributor that said something.
        raise TransportError(f"could not reach {base_url}: {exc}") from exc
    finally:
        connection.close()


def read_answer(response: http.client.HTTPResponse) -> bytes:
    """A ``200``'s body, read to :data:`MAX_ANSWER_BYTES`; ``b""`` for any other status.

    Only a ``200`` is parsed by either caller, so only a ``200`` past the
    bound is an error -- this module's one exception, which each caller
    already reads as *unreachable*. Another status's body is read to the same
    bound and dropped, so closing is a close rather than a reset.
    """
    if response.status != 200:
        response.read(MAX_ANSWER_BYTES)
        return b""
    body = response.read(MAX_ANSWER_BYTES + 1)
    if len(body) > MAX_ANSWER_BYTES:
        raise TransportError(f"the distributor's answer ran past {MAX_ANSWER_BYTES:,} bytes")
    return body


def _peer_is_loopback(sock: Any) -> bool:
    try:
        peer = sock.getpeername()
    except (AttributeError, OSError):
        return False
    address = peer[0] if isinstance(peer, tuple) and peer else ""
    try:
        return ipaddress.ip_address(str(address).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def plaintext_notice(base_url: str) -> str:
    """SPEC 7's SHOULD: name the base a token is about to cross in the clear."""
    return (
        f"Postern: talking to {base_url} over plaintext HTTP. This is permitted only because "
        "the connection lands on a loopback address; the token is refused over any other."
    )


def resolves_to_loopback(base_url: str) -> bool:
    """Whether ``base_url``'s host resolves only to loopback, best effort.

    Used for the startup notice alone. The decision that matters is made on
    the connected socket at request time, not here — this cannot be, since
    a resolver can answer differently a moment later, which is the whole
    reason SPEC 7 phrases the rule as the address rather than the name.
    """
    host = urlsplit(base_url).hostname or ""
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False
    for address in addresses:
        try:
            if not ipaddress.ip_address(str(address).split("%", 1)[0]).is_loopback:
                return False
        except ValueError:
            return False
    return True


__all__ = [
    "MAX_ANSWER_BYTES",
    "USER_AGENT",
    "TransportError",
    "open_response",
    "plaintext_notice",
    "read_answer",
    "resolves_to_loopback",
]
