"""The specification's error envelope (SPEC 2.1), as an exception.

Every non-2xx body in Postern is::

    {"error": {"code": "…", "message": "…", "detail": null}}

and the root of that object is **closed** — it is the only schema in the
protocol that is. Nothing may sit beside ``error``, so anything an
implementation wants to attach to a failure goes inside ``detail``, which
takes arbitrary structure. That is why :class:`PosternError` carries a
detail rather than growing fields.

``message`` is shown to a user verbatim, which is what the issue behind
this work asks for when it says failure modes must produce plain-language
errors. Two rules follow, and both are easy to break by writing an error
message the way one writes a log line:

* Say what the reader can do, not what the code did. "Your access to this
  agent has been withdrawn" is actionable; "entitlement check returned
  state=revoked" is a stack frame with a full stop.
* Do not vary a message by a cause the reader is not entitled to know.
  Only the distributor half of the protocol has that hazard (SPEC 5.5) and
  this file is the runner half, but a runner relaying a distributor's
  refusal can reintroduce it — which is why :data:`NOT_ENTITLED_MESSAGE` is
  one constant for three different upstream causes.
"""

from __future__ import annotations

from typing import Any

# --- Codes (SPEC 2.1). Only the runner side; a distributor's are its own.

BAD_REQUEST = "bad_request"
UNAUTHORIZED = "unauthorized"
NOT_FOUND = "not_found"
NOT_ENTITLED = "not_entitled"
MISSING_CREDENTIAL = "missing_credential"
AGENT_ERROR = "agent_error"
NOT_IMPLEMENTED = "not_implemented"
UNAVAILABLE = "unavailable"
RUN_TIMEOUT = "run_timeout"

# SPEC 5.7.4: a runner cannot tell a withdrawn entitlement from one that
# never existed, from a token that no longer resolves. The specification
# makes those three deliberately indistinguishable at the distributor, and
# "that indistinguishability reaches the runner's vocabulary too" — so one
# sentence covers all three, and it says only what is common to them.
NOT_ENTITLED_MESSAGE = (
    "This agent is not licensed to run here. Your purchase may have been refunded, "
    "or the access token this runner is using may no longer be valid."
)

# SPEC 5.3: the one refusal a distributor does tell apart. A `revoked`
# carrying `access_ends_at` says access ran out on a date rather than being
# taken back, so the sentence above would name the wrong causes. The pull's
# `410` and the run's `403` open with this one, so the two agree about why.
WITHDRAWN_MESSAGE = "This agent was withdrawn from sale."


class PosternError(Exception):
    """A failure that has an HTTP status, a code, and something to say."""

    def __init__(self, status: int, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail

    def envelope(self) -> dict[str, Any]:
        """The body, exactly. Nothing sits beside ``error``."""
        return {"error": {"code": self.code, "message": self.message, "detail": self.detail}}


def bad_request(message: str, detail: Any = None) -> PosternError:
    return PosternError(400, BAD_REQUEST, message, detail)


def unauthorized() -> PosternError:
    """SPEC 2.1's ``401``, and SPEC 7's: this runner demands a credential.

    Only reachable on a runner an operator configured with an inbound token.
    SPEC 7 leaves the *scheme* to the runner and fixes the *refusal* here, so
    a client meets one error shape across the protocol rather than a second
    one at its edge.

    **One sentence for both causes** — no credential presented, and a wrong
    one — for the reason this module's own docstring gives: a message must
    not vary by a cause the reader is not entitled to know. Here that reader
    is whoever reached the runner's port, and telling them "that token was
    close" is a bit of an oracle they should not have. It is a different
    argument from SPEC 5.5's, which is about enumerating a catalogue, and it
    lands in the same place.

    The message is written for the operator who will actually read it, since
    the only legitimate caller of such a runner is a platform proxy rather
    than a person.
    """
    return PosternError(
        401,
        UNAUTHORIZED,
        "This runner requires an authorization token. Present it as 'Authorization: Bearer <token>'.",
    )


def not_found(message: str = "No such path on this runner.") -> PosternError:
    """SPEC 2.1: a runner has no agent identifier to miss.

    It serves exactly one agent (SPEC 2.2), so its only use of ``404`` is a
    path it does not implement — which says nothing about the agent, and is
    a different fact from the distributor's ``404``.
    """
    return PosternError(404, NOT_FOUND, message)


def not_entitled(*, pointer: str = "", detail: Any = None) -> PosternError:
    """SPEC 5.5's one sanctioned ``403``: a local runner, its own state.

    There is nothing to enumerate here — the client is already talking to
    the runner for the one agent it serves — so the answer a distributor
    must never give is the right one to give here.

    The sentence stays :data:`NOT_ENTITLED_MESSAGE`, naming no cause. A
    caller may follow it with a *pointer* — where the buyer can act, which
    is the same wherever the three indistinguishable causes apply — and
    hand a ``detail`` that says in fields what ``status`` says too.
    Both are the distributor client's to compose; this module stays the
    protocol's vocabulary and nothing about any one distributor.
    """
    message = f"{NOT_ENTITLED_MESSAGE} {pointer}" if pointer else NOT_ENTITLED_MESSAGE
    return PosternError(403, NOT_ENTITLED, message, detail)


def access_ended(ended_on: str, *, pointer: str = "", detail: Any = None) -> PosternError:
    """The same ``403``, for an entitlement that ran out on its date (SPEC 5.3).

    A ``revoked`` answer carrying a past ``access_ends_at`` is not one of the
    three causes :data:`NOT_ENTITLED_MESSAGE` names: nothing was refunded and
    the token is fine. So the sentence says what happened and when, and the
    code stays ``not_entitled``, the one SPEC 5.7.4 gives every ``revoked``.
    """
    sentence = f"{WITHDRAWN_MESSAGE} Your access ended on {ended_on}."
    message = f"{sentence} {pointer}" if pointer else sentence
    return PosternError(403, NOT_ENTITLED, message, detail)


def missing_credential(names: list[str]) -> PosternError:
    listed = ", ".join(names)
    plural = "s" if len(names) != 1 else ""
    return PosternError(
        424,
        MISSING_CREDENTIAL,
        f"This agent needs the environment variable{plural} {listed}, which "
        f"{'are' if len(names) != 1 else 'is'} not set. Add "
        f"{'them' if len(names) != 1 else 'it'} to the bundle's .env file and start the runner again.",
        {"missing": list(names)},
    )


def agent_error(message: str) -> PosternError:
    return PosternError(500, AGENT_ERROR, message)


def not_implemented(verb: str) -> PosternError:
    return PosternError(
        501,
        NOT_IMPLEMENTED,
        f"This runner does not implement {verb}. Retrying will not help.",
    )


def unavailable(message: str) -> PosternError:
    return PosternError(503, UNAVAILABLE, message)


def requirements_not_installed(module: str) -> PosternError:
    """The agent never ran: this Python lacks a package the bundle needs.

    ``unavailable`` rather than ``agent_error``: installing the requirements
    is the retry that succeeds, where ``agent_error`` would send the buyer
    to report a bug against a crew that is fine. One sentence for the engine,
    which meets this before it can spawn, and the worker, which meets it after.
    """
    return unavailable(
        f"This bundle's requirements are not installed in this Python ({module!r} is missing). "
        "Run `pip install -r requirements.txt` in the bundle folder, then `python doctor.py`."
    )


def run_timeout(max_run_seconds: int) -> PosternError:
    """SPEC 4.5. The bound rides in ``detail`` because the root is closed."""
    return PosternError(
        504,
        RUN_TIMEOUT,
        f"This run passed the runner's limit of {max_run_seconds} seconds and was stopped. "
        "Try again with less to do, or start the runner with a longer limit.",
        {"max_run_seconds": max_run_seconds},
    )


__all__ = [
    "AGENT_ERROR",
    "BAD_REQUEST",
    "MISSING_CREDENTIAL",
    "NOT_ENTITLED",
    "NOT_ENTITLED_MESSAGE",
    "NOT_FOUND",
    "NOT_IMPLEMENTED",
    "RUN_TIMEOUT",
    "UNAUTHORIZED",
    "UNAVAILABLE",
    "WITHDRAWN_MESSAGE",
    "PosternError",
    "access_ended",
    "agent_error",
    "bad_request",
    "missing_credential",
    "not_entitled",
    "not_found",
    "not_implemented",
    "requirements_not_installed",
    "run_timeout",
    "unauthorized",
    "unavailable",
]
