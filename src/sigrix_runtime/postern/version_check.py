"""The boot-time version check (SPEC 8): "update available, re-pull".

Check-on-start, once. This module does not poll and this runner does not
self-update — a stale bundle keeps running exactly as it would have, and
the only thing this adds is a line in the boot log and a field in
``status`` a client can render.

**Unauthenticated on purpose**, which is why :func:`fetch_latest_version`
carries no token: the distributor's endpoint (SPEC 8's own AC) is "safe to
call without leaking listing content", so there is nothing here for
:mod:`transport` to protect. The distributor answers only for a listing a
stranger can already see, which is what makes that safe rather than merely
convenient.

**Must never fail a boot.** The container has to work offline after the
initial pull, so a network failure here is not a
boot failure and not even a loud one: :func:`check_for_update` catches
everything :mod:`transport` can raise and reports ``unreachable`` rather
than propagating, the same posture ``entitlement.py``'s own
``CheckUnreachable`` handling takes for the same reason.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from sigrix_runtime.postern import PATH_PREFIX
from sigrix_runtime.postern.transport import TransportError, open_response, read_answer

logger = logging.getLogger("sigrix_runtime.postern")

STATE_NOT_REQUIRED = "not_required"
STATE_UNREACHABLE = "unreachable"
STATE_CURRENT = "current"
STATE_UPDATE_AVAILABLE = "update_available"

# Matches entitlement.py's CHECK_TIMEOUT_SECONDS: short on purpose, because
# an unreachable check is a survivable condition and a boot that hangs on a
# dead endpoint has turned "check for an update" into an outage of its own.
CHECK_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class UpdateCheck:
    """What ``status``'s ``update`` block reports (status.schema.json)."""

    state: str
    current: str = ""
    latest: str = ""

    def as_dict(self) -> dict[str, Any]:
        block: dict[str, Any] = {"state": self.state}
        if self.current:
            block["current"] = self.current
        if self.latest:
            block["latest"] = self.latest
        return block


def fetch_latest_version(base_url: str, agent_id: str, *, timeout: float = CHECK_TIMEOUT_SECONDS) -> str | None:
    """One ``GET`` to SPEC 8's endpoint. ``None`` on any failure — never raises.

    Every failure shape collapses to the same answer, deliberately: a
    boot-time notice has no reader who can act differently on "the
    distributor is down" versus "it answered something this runner cannot
    parse" versus "no such listing" (this endpoint 404s a type it has
    nothing to say about, same as an unresolvable identifier) — only
    "could not learn a newer version exists" is actionable, and it is not
    actionable tonight.
    """
    owner, _, name = agent_id.partition("/")
    path = f"{PATH_PREFIX}/versions/{owner}/{name}"
    try:
        with open_response(base_url, path, accept="application/json", timeout=timeout) as response:
            # Bounded: the endpoint is unauthenticated, so a distributor
            # answering it with megabytes reached the runner whatever its token.
            status, body = response.status, read_answer(response)
    except TransportError as exc:
        logger.info("postern.version_check.unreachable: %s", exc)
        return None
    if status != 200:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    echoed = str(payload.get("agent_id") or "")
    if echoed != agent_id:
        # Same rule entitlement.py's _parse_check_body follows: an answer
        # for a different identifier is not an answer to this question.
        return None
    version = payload.get("version")
    return str(version) if isinstance(version, str) and version.strip() else None


def check_for_update(*, base_url: str, agent_id: str, current_version: str) -> UpdateCheck:
    """Compare once, at boot. Never raises, and never blocks past the timeout.

    ``current_version`` is the caller's own ``describe`` document read —
    this module does not read the bundle itself, so a check for an agent
    with no describe document to speak of (the derived-document case) is
    simply a comparison against whatever that document already resolved to.
    """
    if not (base_url and agent_id):
        return UpdateCheck(state=STATE_NOT_REQUIRED)
    latest = fetch_latest_version(base_url, agent_id)
    if latest is None:
        return UpdateCheck(state=STATE_UNREACHABLE, current=current_version)
    state = STATE_UPDATE_AVAILABLE if latest != current_version else STATE_CURRENT
    return UpdateCheck(state=state, current=current_version, latest=latest)


__all__ = [
    "STATE_CURRENT",
    "STATE_NOT_REQUIRED",
    "STATE_UNREACHABLE",
    "STATE_UPDATE_AVAILABLE",
    "UpdateCheck",
    "check_for_update",
    "fetch_latest_version",
]
