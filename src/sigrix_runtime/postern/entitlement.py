"""The distributor client, and the state machine of SPEC 5.7.

Everything here answers one question — *may this agent run right now* —
and the specification is unusually precise about it, because the
interesting cases are not "yes" and "no" but the ones in between: an
answer that has expired, a distributor having a bad afternoon, a machine
that rebooted with no network.

**The table is the specification's own** (SPEC 5.7.4, "Every case, in one
table"), transcribed as data rather than as an ``if`` chain, for the same
reason the platform's dispatcher keeps its authorities in a table: a
situation nobody wrote a branch for should fail a test that enumerates the
vocabulary, not fall through to whichever answer the last ``else`` gives.
:data:`SITUATIONS` is that vocabulary and :func:`decide` is the only reader.

Three rules underneath it are worth stating plainly, because each is a
place a reasonable implementation goes wrong:

* **Unreachable answers ``unavailable``; refused answers ``not_entitled``.**
  A runner that cannot find out says so and invites a retry. A runner that
  has been told no does not pretend the answer might change on the next
  request. That is why a held ``revoked`` keeps refusing while the
  distributor is unreachable rather than drifting into grace: grace exists
  so one party's downtime is not everybody's, and nothing is down here.

* **``checked_at`` is the distributor's clock and is never re-stamped**
  (SPEC 4.4, 5.3). It means *when the distributor last consulted the
  authority*, so the distributor's cache and this one expire together
  instead of stacking. Re-stamping it on receipt would silently double the
  real staleness window while every field still validated.

* **A restart is not a new window** (SPEC 5.7.2). The answer is persisted
  with its ``checked_at`` and both deadlines are evaluated against that
  value on load — which is exactly what makes an offline restart
  survivable, and what stops one being a way to mint fresh grace on demand.

**On the wire.** One ``GET``, one bearer header, and a body this module
refuses unless it is complete. A distributor that answers with a different
``agent_id`` than the one addressed has answered a different question, and
SPEC 5.3 says to treat that as a failed check rather than reconcile it —
so it lands as *unreachable*, which is the conservative of the two
failures: the runner keeps its previous answer and its deadlines rather
than acting on an answer that is about something else.

**The connection itself belongs to** :mod:`sigrix_runtime.postern.transport`,
including SPEC 7's rule that a token crosses plaintext only to a loopback
peer. The bundle pull carries the same token to the same distributor, so
that rule has one implementation and both callers translate its one
exception into their own vocabulary — here, ``unreachable``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sigrix_runtime.postern import PATH_PREFIX
from sigrix_runtime.postern.describe import AGENT_ID_SHAPE, is_agent_id
from sigrix_runtime.postern.errors import WITHDRAWN_MESSAGE, PosternError, access_ended, not_entitled, unavailable
from sigrix_runtime.postern.transport import TransportError, open_response, read_answer

logger = logging.getLogger(__name__)

STATE_ACTIVE = "active"
STATE_REVOKED = "revoked"
STATE_UNKNOWN = "unknown"
STATE_NOT_REQUIRED = "not_required"

GATE_RUN = "run"
GATE_NOT_ENTITLED = "not_entitled"
GATE_UNAVAILABLE = "unavailable"

# Not part of the specification — a self-reported hint the pilot instrumentation
# reads off the entitlement check to tell a container run apart from a
# raw unzipped one, since the wire request is otherwise identical either way.
# ``connector-local`` names a shape nothing in this tree runs yet (the
# standalone connector described in the strategy doc), so it can never be
# produced today — declared anyway so the vocabulary the platform records
# doesn't have to grow again the day that changes.
DELIVERY_MODE_ZIP = "zip"
DELIVERY_MODE_CONTAINER = "container"
DELIVERY_MODE_CONNECTOR_LOCAL = "connector-local"

#: Header the entitlement check carries the delivery mode on. Not a
#: specification field, so it travels as a header rather than a body member
#: — a distributor too old to know about it simply never reads it.
DELIVERY_MODE_HEADER = "X-Sigrix-Delivery-Mode"

# Where a check answer is kept between restarts, beside the bundle it is
# about. Not in the workspace: that is the agent's to write in, and this is
# the runner's own state.
CACHE_FILENAME = ".postern_entitlement.json"

# How long to wait on the distributor before calling it unreachable. Short
# on purpose — an unreachable check is a survivable condition with a
# declared grace behind it, and a runner that blocks a buyer's run for a
# minute on a dead endpoint has turned a graceful degradation into an
# outage of its own.
CHECK_TIMEOUT_SECONDS = 10

# The bound a ``status`` re-check waits, and how long ``status`` holds off
# after one that failed. SPEC 5.4's "first request after the window"
# does not exempt this verb, so a polled ``status`` past its window has to
# ask — but a client polling every second must not find each poll blocked
# for ten seconds on a dead distributor, nor turn into a request storm at
# it. A timed-out attempt lands where ``run``'s would: unreachable, held
# answer kept, grace running. ``run`` itself is never held off — SPEC 5.7.1
# wants the check kept up, and a run is the request that matters.
STATUS_CHECK_TIMEOUT_SECONDS = 3.0
STATUS_RECHECK_HOLD_OFF_SECONDS = 15

# The fallback window for an answer that carries none. Only a 404 does:
# SPEC 5.5 requires that body to be constant, so it cannot carry fields,
# and SPEC 5.7.4 has the runner "reuse the last stale_after_seconds the
# distributor gave it" — this is what it uses when there is no last one.
DEFAULT_STALE_AFTER_SECONDS = 60

# The longest either window is held, whatever a distributor declares: ten
# years. A declared window is otherwise unbounded, and one past what
# ``timedelta`` can count was accepted, persisted, and then failed every
# ``status`` and ``run`` with a 503 until its cache file was deleted.
# Holding a window shorter than declared only re-checks sooner, which the
# specification always allows.
MAX_WINDOW_SECONDS = 10 * 365 * 24 * 60 * 60

# How often a run repeats the warning that access is ending. Once per
# run would print a line a minute from a busy loop for a year, and once per
# process would print it at a container's first run and never again.
ACCESS_WARNING_INTERVAL = timedelta(days=1)


@dataclass(frozen=True)
class Verdict:
    """What a situation means for ``status`` and for ``run``/``stream``."""

    situation: str
    state: str
    gate: str
    carries_checked_at: bool


# SPEC 5.7.4's table, in its own order. ``decide`` names a situation and
# reads its row; nothing else in this module branches on entitlement.
SITUATIONS: tuple[Verdict, ...] = (
    Verdict("no_distributor", STATE_NOT_REQUIRED, GATE_RUN, carries_checked_at=False),
    Verdict("active_fresh", STATE_ACTIVE, GATE_RUN, carries_checked_at=True),
    Verdict("refused", STATE_REVOKED, GATE_NOT_ENTITLED, carries_checked_at=True),
    Verdict("unreachable_fresh", STATE_ACTIVE, GATE_RUN, carries_checked_at=True),
    Verdict("unreachable_in_grace", STATE_UNKNOWN, GATE_RUN, carries_checked_at=True),
    Verdict("unreachable_past_grace", STATE_UNKNOWN, GATE_UNAVAILABLE, carries_checked_at=True),
    Verdict("never_checked", STATE_UNKNOWN, GATE_UNAVAILABLE, carries_checked_at=False),
)

_BY_SITUATION = {verdict.situation: verdict for verdict in SITUATIONS}


@dataclass(frozen=True)
class CheckAnswer:
    """One answer from the distributor, with the clock it was anchored to.

    ``anchored`` records whether ``checked_at`` came from the distributor.
    A 404 carries no body to take one from (SPEC 5.5), so the runner stamps
    its own receipt time and says so — the specification permits exactly
    that, and only there, "because it forbids discarding an anchor the
    distributor supplied, and here there is none to discard".
    """

    state: str
    checked_at: datetime
    stale_after_seconds: int
    grace_seconds: int
    anchored: bool = True
    # SPEC 5.3's one optional member: when access ends, on an ``active``
    # answer, or when it ended, on a ``revoked`` one. A 404 has none.
    access_ends_at: datetime | None = None

    def expires_at(self) -> datetime:
        return self.checked_at + timedelta(seconds=self.stale_after_seconds)

    def grace_ends_at(self) -> datetime:
        return self.expires_at() + timedelta(seconds=self.grace_seconds)

    def countable(self) -> bool:
        """Whether both deadlines are dates at all.

        A ``checked_at`` near year 9999 puts them past what ``datetime``
        holds, and every later ``status`` and ``run`` raised on it. Asked
        once, where an answer arrives, rather than guarded wherever it is read.
        """
        try:
            self.grace_ends_at()
        except OverflowError:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        stored: dict[str, Any] = {
            "state": self.state,
            "checked_at": rfc3339(self.checked_at),
            "stale_after_seconds": self.stale_after_seconds,
            "grace_seconds": self.grace_seconds,
            "anchored": self.anchored,
        }
        if self.access_ends_at is not None:
            stored["access_ends_at"] = rfc3339(self.access_ends_at)
        return stored

    @classmethod
    def from_dict(cls, raw: Any) -> CheckAnswer | None:
        if not isinstance(raw, dict):
            return None
        state = str(raw.get("state") or "")
        checked_at = parse_rfc3339(raw.get("checked_at"))
        if state not in {STATE_ACTIVE, STATE_REVOKED} or checked_at is None:
            return None
        try:
            stale = _window(raw.get("stale_after_seconds"))
            grace = _window(raw.get("grace_seconds"))
        except (TypeError, ValueError, OverflowError):
            return None
        answer = cls(
            state=state,
            checked_at=checked_at,
            stale_after_seconds=stale,
            grace_seconds=grace,
            anchored=bool(raw.get("anchored", True)),
            access_ends_at=parse_rfc3339(raw.get("access_ends_at")),
        )
        # A file an older runner persisted can still hold either
        # shape, and read back it would fail every request after a restart.
        return answer if answer.countable() else None


class CheckUnreachable(RuntimeError):
    """The distributor could not be asked, or answered something unusable.

    SPEC 5.7's definition: a transport failure, a ``5xx``, or a response
    whose body is not a valid check answer. A ``404`` is deliberately not
    one of these — it is an answer.
    """


class CheckMisaddressed(CheckUnreachable):
    """The identifier is not one (SPEC 1.5), so nothing can be asked about it.

    Refused here before sending, or by the distributor's ``400`` (SPEC
    5.3.1), which is computed from the request string alone. No retry and no
    grace changes either, so the gate says it is configuration rather than
    blaming the network. A subclass, so every caller that survives an
    unreachable distributor survives this.
    """


def decide(answer: CheckAnswer | None, *, configured: bool, now: datetime) -> Verdict:
    """Name the situation this runner is in, per SPEC 5.7.4's table."""
    if not configured:
        return _BY_SITUATION["no_distributor"]
    if answer is None:
        return _BY_SITUATION["never_checked"]
    if answer.state == STATE_REVOKED:
        # No window and no grace. The distributor was reached and declined
        # to vouch; a later re-check is what may change that, not the clock.
        return _BY_SITUATION["refused"]
    if now < answer.expires_at():
        return _BY_SITUATION["active_fresh"]
    if now < answer.grace_ends_at():
        return _BY_SITUATION["unreachable_in_grace"]
    return _BY_SITUATION["unreachable_past_grace"]


class Entitlement:
    """The runner's entitlement state: refreshed on demand, persisted, gated.

    Construct it once at startup. :meth:`gate` refreshes before each
    ``run``/``stream`` and :meth:`snapshot` before each ``status`` — both
    re-check only once the held answer has expired, which is the cadence
    SPEC 5.4 requires and no more often. ``status`` asks under a shorter
    bound, never waits on a check another request already has in flight,
    and holds off for a while after an attempt that failed.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        agent_id: str,
        cache_path: Path | None = None,
        timeout: float = CHECK_TIMEOUT_SECONDS,
        delivery_mode: str = DELIVERY_MODE_ZIP,
        listing_url: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.agent_id = agent_id
        self.cache_path = cache_path
        self.timeout = timeout
        # Sent verbatim on every check; the distributor is the one
        # that validates it against the known set, not this client — a
        # buyer who hand-edits the environment gets whatever they typed
        # rather than this client silently correcting or dropping it.
        self.delivery_mode = delivery_mode
        # Where a refused buyer is sent: the listing's own page, from
        # the bundle's ``plugin.json`` when it has one. A pointer, never a
        # cause — see :meth:`_refusal_pointer`.
        self.listing_url = str(listing_url or "").strip()
        # Why the last check could not be asked, when the identifier is why;
        # read by ``gate`` ahead of the table, which has no row for it.
        self.misaddressed = ""
        self._answer: CheckAnswer | None = None
        self._last_stale_after = DEFAULT_STALE_AFTER_SECONDS
        # One check in flight at a time: ``run`` waits for it, ``status``
        # reports the held answer instead.
        self._check_lock = threading.Lock()
        self._status_hold_off_until: datetime | None = None
        # The day last warned about and when, for ``_warn_while_ending``.
        self._ending_warned: tuple[str, datetime] | None = None
        if self.configured:
            self._answer = self._load_cached()
            if self._answer is not None:
                self._last_stale_after = self._answer.stale_after_seconds

    @property
    def configured(self) -> bool:
        """Whether a distributor is configured at all (SPEC 5.1).

        All three are required together. A base URL with no token cannot
        ask anything, and a token with no agent id has nothing to ask
        about — so half a configuration is no configuration, and reporting
        ``not_required`` for it would be a lie a buyer cannot see.
        """
        return bool(self.base_url and self.token and self.agent_id)

    def refresh(self, *, now: datetime | None = None, force: bool = False, timeout: float | None = None) -> Verdict:
        """Re-check if the held answer has expired, then name the situation.

        Serialised: a second request arriving while a check is in flight
        waits for that answer rather than asking the same question twice.
        """
        moment = now or datetime.now(UTC)
        if not self.configured:
            return _BY_SITUATION["no_distributor"]
        with self._check_lock:
            self._refresh_held(moment, force=force, timeout=timeout)
        return decide(self._answer, configured=True, now=moment)

    def snapshot(self, *, now: datetime | None = None, timeout: float | None = None) -> dict[str, Any]:
        """The ``entitlement`` block of a ``status`` body (SPEC 4.4).

        Past the window this asks too — a polling client is the one
        reader who learns of a lapsed licence *only* here — but on
        ``status``'s terms: under :data:`STATUS_CHECK_TIMEOUT_SECONDS`, never
        queued behind a check ``run`` already has in flight (that answer is
        moments away, and this call reports the one held meanwhile), and not
        again within :data:`STATUS_RECHECK_HOLD_OFF_SECONDS` of an attempt
        that came back unreachable.
        """
        moment = now or datetime.now(UTC)
        if self.configured and self._due(moment) and not self._held_off(moment):
            bound = STATUS_CHECK_TIMEOUT_SECONDS if timeout is None else timeout
            if self._check_lock.acquire(blocking=False):
                try:
                    self._refresh_held(moment, force=False, timeout=bound)
                    if self._due(moment):
                        # Still nothing usable: the attempt failed. Hold
                        # ``status`` off rather than let every poll retry.
                        self._status_hold_off_until = moment + timedelta(seconds=STATUS_RECHECK_HOLD_OFF_SECONDS)
                finally:
                    self._check_lock.release()
        verdict = decide(self._answer, configured=self.configured, now=moment)
        block: dict[str, Any] = {"state": verdict.state}
        if self.misaddressed:
            block["state"] = STATE_UNKNOWN  # never asked, so no checked_at to report
        elif verdict.carries_checked_at and self._answer is not None:
            block["checked_at"] = rfc3339(self._answer.checked_at)
            block["stale_after_seconds"] = self._answer.stale_after_seconds
            block["grace_seconds"] = self._answer.grace_seconds
            if self._answer.access_ends_at is not None:
                # SPEC 4.4: the distributor's date, unchanged, through grace
                # too, since it is still the last thing the distributor said.
                block["access_ends_at"] = rfc3339(self._answer.access_ends_at)
        if self.listing_url:
            block["listing_url"] = self.listing_url
        return block

    def gate(self, *, now: datetime | None = None) -> None:
        """Raise the error ``run``/``stream`` owes this situation, or return.

        Refreshing first is the point: a client's ``run`` is when a stale
        answer has to be replaced, and doing it here rather than on a timer
        is what keeps the declared window an upper bound rather than an
        average.
        """
        moment = now or datetime.now(UTC)
        verdict = self.refresh(now=moment)
        if self.misaddressed:
            raise unavailable(self.misaddressed)
        if verdict.gate == GATE_NOT_ENTITLED:
            ended_on = self.ended_on(moment)
            if ended_on:
                # Not a refund and not a token to replace (SPEC 5.3): say what
                # did happen, and when, as the pull's 410 does.
                raise access_ended(ended_on, pointer=self._listing_pointer(), detail=self._refusal_detail())
            raise not_entitled(pointer=self._refusal_pointer(), detail=self._refusal_detail())
        if verdict.gate == GATE_UNAVAILABLE:
            raise unavailable(
                "This runner has not been able to confirm your licence for this agent. "
                "Check the machine's network connection and try again."
            )
        self._warn_while_ending(moment)

    def ends_on(self) -> str:
        """The day access ends, while an ``active`` answer says it will, else ``""``.

        Read off the answer held, through grace included: the date is the
        last thing the distributor said, and SPEC 5.3 adds no deadline to it,
        so this is only ever something to tell the buyer.
        """
        answer = self._answer
        if answer is None or answer.state != STATE_ACTIVE or answer.access_ends_at is None:
            return ""
        return rfc3339(answer.access_ends_at)[:10]

    def ended_on(self, now: datetime | None = None) -> str:
        """The day access ended, when a ``revoked`` answer says it ran out, else ``""``.

        A ``revoked`` carrying a date that has not come yet contradicts itself,
        and is read as the plain refusal rather than dated.
        """
        answer = self._answer
        moment = now or datetime.now(UTC)
        if answer is None or answer.state != STATE_REVOKED or answer.access_ends_at is None:
            return ""
        if answer.access_ends_at > moment:
            return ""
        return rfc3339(answer.access_ends_at)[:10]

    def _warn_while_ending(self, moment: datetime) -> None:
        """Log that access is ending, at most once per :data:`ACCESS_WARNING_INTERVAL`.

        The runner's console is the one place its operator reads, and
        ``status`` carries the same date for a client (SPEC 4.4).
        """
        ends_on = self.ends_on()
        if not ends_on:
            return
        last = self._ending_warned
        if last is not None and last[0] == ends_on and moment - last[1] < ACCESS_WARNING_INTERVAL:
            return
        self._ending_warned = (ends_on, moment)
        logger.warning(
            "Postern: %s Your access ends on %s, and this runner will refuse to run it after that.%s",
            WITHDRAWN_MESSAGE,
            ends_on,
            f" {self._listing_pointer()}" if self.listing_url else "",
        )

    def _refresh_held(self, moment: datetime, *, force: bool, timeout: float | None) -> None:
        """Ask the distributor if an answer is due. Called with the lock held."""
        if force or self._due(moment):
            try:
                self._store(self.check(agent_id=self.agent_id, timeout=timeout))
            except CheckUnreachable as exc:
                # SPEC 5.7.1: keep attempting for the whole grace period
                # rather than waiting it out — the answer that ends grace
                # early is also the answer that renews the entitlement.
                logger.info("postern.entitlement.unreachable: %s", exc)

    def _due(self, moment: datetime) -> bool:
        return self._answer is None or moment >= self._answer.expires_at()

    def _held_off(self, moment: datetime) -> bool:
        return self._status_hold_off_until is not None and moment < self._status_hold_off_until

    def _refusal_pointer(self) -> str:
        """Where a refused buyer can act — a pointer, never a cause.

        SPEC 5.5 keeps refund, rotation and never-owned indistinguishable at
        the distributor, and that reaches this vocabulary too; what a buyer
        is owed instead is somewhere to go. The wording is the pull's own
        (``pull.py``), so the two refusals a buyer can meet read alike.
        """
        if self.listing_url:
            return (
                f"See the listing at {self.listing_url}, and check SIGRIX_TOKEN against "
                "the runner tokens on your account's Plugins page."
            )
        return (
            "Check SIGRIX_TOKEN against the runner tokens on your account's Plugins page and "
            "POSTERN_AGENT_ID against the listing's own page."
        )

    def _listing_pointer(self) -> str:
        """Where a buyer whose access ended can look, with no token to check."""
        return f"See the listing at {self.listing_url}." if self.listing_url else ""

    def _refusal_detail(self) -> dict[str, Any]:
        """The machine half of the same pointer: fields ``status`` also carries."""
        detail: dict[str, Any] = {"state": STATE_REVOKED}
        if self._answer is not None:
            detail["checked_at"] = rfc3339(self._answer.checked_at)
            if self._answer.access_ends_at is not None:
                # SPEC 5.3's MAY, under the member SPEC 5.6 already defines.
                detail["access_ends_at"] = rfc3339(self._answer.access_ends_at)
        if self.listing_url:
            detail["listing_url"] = self.listing_url
        return detail

    def check(self, *, agent_id: str, timeout: float | None = None) -> CheckAnswer:
        """One call to SPEC 5.3's endpoint. Raises :class:`CheckUnreachable`.

        ``timeout`` overrides the client's own for this one call — how
        ``status`` asks under a shorter bound than ``run``.
        """
        if not is_agent_id(agent_id):
            # Refused rather than sent: with no slash it was addressed as
            # ``…/entitlements/<id>/``, which a distributor's catch-all answers
            # 404 -- read here as "not licensed", the wrong diagnosis.
            self.misaddressed = (
                f"This runner's agent identifier, {agent_id!r}, is not one: an identifier is "
                f"{AGENT_ID_SHAPE}. Set POSTERN_AGENT_ID to the one on the listing's page and start the runner again."
            )
            raise CheckMisaddressed(self.misaddressed)
        owner, _, name = agent_id.partition("/")
        path = f"{PATH_PREFIX}/entitlements/{owner}/{name}"
        status, body = self._request(path, timeout=timeout)

        if status == 400:
            # SPEC 5.3.1: computed from the request string alone. It was folded
            # into unreachable, so every run said to check the network.
            self.misaddressed = (
                f"The distributor refused this runner's agent identifier, {agent_id!r}, as malformed. "
                "Set POSTERN_AGENT_ID to the one on the listing's page and start the runner again."
            )
            raise CheckMisaddressed(self.misaddressed)
        self.misaddressed = ""
        if status == 404:
            # SPEC 5.7.4. An answer, not an outage: stop honouring the
            # entitlement now, with no grace. The body is constant by
            # design, so the timestamp is this runner's own and the window
            # is whatever the distributor last told us.
            return CheckAnswer(
                state=STATE_REVOKED,
                checked_at=datetime.now(UTC),
                stale_after_seconds=self._last_stale_after,
                grace_seconds=0,
                anchored=False,
            )
        if status != 200:
            raise CheckUnreachable(f"distributor answered {status}")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise CheckUnreachable(f"distributor answered an unreadable body: {exc}") from exc
        return _parse_check_body(payload, addressed=agent_id)

    # --- Internals ---------------------------------------------------

    def _request(self, path: str, *, timeout: float | None = None) -> tuple[int, bytes]:
        """One ``GET``, through the shared transport, as status and bytes.

        Read through :func:`transport.read_answer`, to its bound.
        """
        try:
            with open_response(
                self.base_url,
                path,
                token=self.token,
                accept="application/json",
                timeout=self.timeout if timeout is None else timeout,
                extra_headers={DELIVERY_MODE_HEADER: self.delivery_mode} if self.delivery_mode else None,
            ) as response:
                return response.status, read_answer(response)
        except TransportError as exc:
            raise CheckUnreachable(str(exc)) from exc

    def _store(self, answer: CheckAnswer) -> None:
        self._answer = answer
        if answer.anchored:
            self._last_stale_after = answer.stale_after_seconds
        if self.cache_path is None:
            return
        payload = {"agent_id": self.agent_id, "token_fingerprint": self._fingerprint(), **answer.as_dict()}
        # Staged beside it and swapped in whole: written in place, a
        # write that died halfway left a torn file, which reads as never
        # checked -- the offline restart SPEC 5.7.2 exists for, defeated.
        staged = self.cache_path.with_name(f"{self.cache_path.name}.{os.getpid()}.{threading.get_ident()}")
        try:
            staged.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(staged, self.cache_path)
        except OSError as exc:  # noqa: BLE001 - a runner that cannot cache still runs
            logger.info("postern.entitlement.cache_write_failed: %s", exc)
            staged.unlink(missing_ok=True)

    def keep_in(self, cache_path: Path) -> None:
        """Persist at *cache_path* from now on, starting with the answer held now."""
        self.cache_path = cache_path
        if self._answer is not None:
            self._store(self._answer)

    def _load_cached(self) -> CheckAnswer | None:
        """A persisted answer, if it is about this agent and this token.

        Both are checked because neither alone is enough: a bundle
        re-pointed at another listing, and a buyer who rotated their token
        after a leak, are each a reason the stored answer is about somebody
        else's question. The token is stored as a fingerprint — this file
        sits in the bundle directory, and a secret does not belong in it.
        """
        if self.cache_path is None:
            return None
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        if raw.get("agent_id") != self.agent_id or raw.get("token_fingerprint") != self._fingerprint():
            return None
        return CheckAnswer.from_dict(raw)

    def _fingerprint(self) -> str:
        return hashlib.sha256(self.token.encode("utf-8")).hexdigest()[:16]


#: The four variables this runner reads, and the only ones it will take from
#: a bundle's ``.env``. Provider keys are deliberately absent: SPEC 4.1.3 gives
#: them nowhere to travel and the run process loads them itself, so nothing is
#: served by lifting them into the server's environment as a side effect.
#:
#: ``POSTERN_INBOUND_TOKEN`` is the odd one out and belongs here anyway. The
#: other three are credentials this runner *presents* to a distributor; that
#: one is the credential it *demands* of its own callers (SPEC 7). What puts
#: it on this list rather than beside ``SIGRIX_DELIVERY_MODE`` is that the
#: server process is the thing that needs it — the rule above is about not
#: lifting a *run's* secrets into this process, and this one has nowhere else
#: to be read.
_ENV_FILE_KEYS = (
    "SIGRIX_TOKEN",
    "POSTERN_AGENT_ID",
    "POSTERN_DISTRIBUTOR",
    "POSTERN_INBOUND_TOKEN",
)


def _env_file_values(bundle_root: Path) -> dict[str, str]:
    """This runner's own keys out of the bundle's ``.env``, if it has one.

    A four-line parser rather than ``python-dotenv`` because this package is
    standard library only — that is what lets ``describe`` and ``status``
    answer with nothing installed. It reads the same shape ``doctor.py`` reads
    (``KEY=value``, ``#`` comments skipped, surrounding quotes stripped) and
    ignores everything outside :data:`_ENV_FILE_KEYS`.
    """
    values: dict[str, str] = {}
    try:
        lines = (bundle_root / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in _ENV_FILE_KEYS:
            values[key] = value.strip().strip("'\"")
    return values


#: Where a runner asks when nobody says otherwise, so an ordinary buyer sets
#: two variables rather than three.
DEFAULT_DISTRIBUTOR = "https://sigrix.io"


@dataclass(frozen=True)
class Settings:
    """The three values a runner needs to talk to a distributor at all."""

    base_url: str
    token: str
    agent_id: str


def settings_from_environment(bundle_root: Path, environ: dict[str, str] | None = None) -> Settings:
    """Resolve the three, environment first and the bundle's ``.env`` second.

    Split out of :func:`from_environment` because the boot sequence needs
    them *before* there is a bundle to construct a client for: the image
    pulls with these same values, and a second resolver would be a second
    answer to "which token" the moment one of them grew a default.
    """
    env = os.environ if environ is None else environ
    from_file = _env_file_values(bundle_root)

    def resolve(name: str) -> str:
        return str(env.get(name) or from_file.get(name) or "").strip()

    return Settings(
        base_url=resolve("POSTERN_DISTRIBUTOR") or DEFAULT_DISTRIBUTOR,
        token=resolve("SIGRIX_TOKEN"),
        agent_id=resolve("POSTERN_AGENT_ID"),
    )


def inbound_token_from_environment(bundle_root: Path, environ: dict[str, str] | None = None) -> str:
    """``POSTERN_INBOUND_TOKEN``: what this runner demands of its callers.

    SPEC 7 obliges a runner binding a non-loopback interface to authenticate
    its callers and specifies no scheme; SPEC 2.1 fixes only the refusal
    (``401`` with ``unauthorized``). This is the scheme: a shared bearer
    value, set by whoever deployed the runner.

    Resolved the same way :func:`settings_from_environment` resolves its
    three — process environment first, the bundle's ``.env`` second — because
    a hosted runner is configured by its host while a bundle on a machine is
    configured by its ``.env``, and one resolver is what keeps those from
    being two answers.

    **Empty means no gate**, which is the only default that can ship: this
    package is inside every bundle a buyer downloads, and a runner on
    loopback that suddenly demanded a token nobody set would refuse its own
    owner. Requiring one is the deployment's decision, made by setting it.
    """
    env = os.environ if environ is None else environ
    from_file = _env_file_values(bundle_root)
    return str(env.get("POSTERN_INBOUND_TOKEN") or from_file.get("POSTERN_INBOUND_TOKEN") or "").strip()


def delivery_mode_from_environment(environ: dict[str, str] | None = None) -> str:
    """``SIGRIX_DELIVERY_MODE``, or the zip default.

    Read from the process environment only, deliberately not from the
    bundle's ``.env`` the way :func:`settings_from_environment` reads its
    three: this names a deployment shape an *operator* sets (the image's own
    ``ENV``), not a buyer credential, and it is not in ``_ENV_FILE_KEYS``.

    An unrecognised value is passed through rather than coerced to the zip
    default — the distributor is the one that validates it (this client has
    no way to know a value is wrong that the distributor doesn't also have),
    and silently rewriting an operator's typo to "zip" would misfile it as
    the one shape it demonstrably is not.
    """
    env = os.environ if environ is None else environ
    raw = str(env.get("SIGRIX_DELIVERY_MODE") or "").strip()
    return raw or DELIVERY_MODE_ZIP


def from_environment(bundle_root: Path, environ: dict[str, str] | None = None, *, agent_id: str = "") -> Entitlement:
    """Build the client from the container's documented variables.

    ``agent_id``, when given, is one the caller resolved already and wins over
    the environment's. It is passed in rather than set afterwards because the
    persisted answer is read at construction, keyed on it: set later, the
    container's ``sigrix/runner acme/my-crew`` shape never read its cache,
    and restarting offline was never checked.

    ``SIGRIX_TOKEN`` is the buyer's marketplace token, ``POSTERN_AGENT_ID``
    the listing it is being presented for, and ``POSTERN_DISTRIBUTOR`` the
    base URL to ask — defaulted, so an ordinary buyer sets two variables
    rather than three.

    **The bundle's ``.env`` is read as a fallback, and that is load-bearing
    rather than a convenience.** The generated ``.env.example`` ships
    ``POSTERN_AGENT_ID`` already filled in and its first line tells the buyer
    to copy the file to ``.env``, so a buyer who does exactly what the bundle
    says would otherwise still be running unlicensed: ``load_dotenv`` happens
    inside :mod:`sigrix_runtime.execution`, which is the *worker* subprocess,
    and this resolver runs in the server process that never imports it. The
    failure is silent in the worst way — the runner starts, serves all four
    verbs and reports ``entitlement.state: not_required``, which is
    indistinguishable from a bundle nobody licensed.

    A real environment variable wins over the file, so an operator exporting
    ``SIGRIX_TOKEN`` (the container shape, per ``__main__``) is never
    overridden by a stale ``.env`` a bundle happened to ship with.
    """
    settings = settings_from_environment(bundle_root, environ)
    return Entitlement(
        base_url=settings.base_url,
        token=settings.token,
        agent_id=agent_id or settings.agent_id,
        cache_path=bundle_root / CACHE_FILENAME,
        delivery_mode=delivery_mode_from_environment(environ),
    )


def rfc3339(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_rfc3339(value: Any) -> datetime | None:
    """A timestamp in UTC, or ``None``.

    Converted here rather than when written back out: ``9999-12-31T22:00:00-05:00``
    parses, and is past year 9999 in UTC, so converting it later raised in
    every ``status`` that carried it.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return (parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)).astimezone(UTC)
    except (ValueError, OverflowError):
        return None


def _parse_check_body(payload: Any, *, addressed: str) -> CheckAnswer:
    """SPEC 5.3's body, refused unless every required member is there.

    "Every member above is REQUIRED", and a partial answer is not a lenient
    success — a missing ``stale_after_seconds`` would leave a revoked
    entitlement with no stated moment at which it stops being honoured, and
    a missing ``checked_at`` would put the anchor back on this runner's
    clock. Both land as unreachable, which keeps the last good answer and
    its deadlines rather than replacing them with a worse one.
    """
    if not isinstance(payload, dict):
        raise CheckUnreachable("distributor answered a body that is not an object")
    state = str(payload.get("state") or "")
    if state not in {STATE_ACTIVE, STATE_REVOKED}:
        raise CheckUnreachable(f"distributor answered an unknown state {state!r}")
    echoed = str(payload.get("agent_id") or "")
    if echoed != addressed:
        # SPEC 5.3: the echo is the question, not the result of a lookup.
        raise CheckUnreachable(f"distributor answered for {echoed!r}, not for {addressed!r}")
    checked_at = parse_rfc3339(payload.get("checked_at"))
    if checked_at is None:
        raise CheckUnreachable("distributor answered without a usable checked_at")
    try:
        stale = _window(payload["stale_after_seconds"])
        grace = _window(payload["grace_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CheckUnreachable(f"distributor answered without usable windows ({exc})") from exc
    answer = CheckAnswer(
        state=state,
        checked_at=checked_at,
        stale_after_seconds=stale,
        grace_seconds=grace,
        # Optional (SPEC 5.3), so one that does not parse is dropped rather
        # than costing the answer: the state and the windows are what count.
        access_ends_at=parse_rfc3339(payload.get("access_ends_at")),
    )
    if not answer.countable():
        raise CheckUnreachable("distributor answered a checked_at its windows cannot be counted from")
    return answer


def _window(raw: Any) -> int:
    """One declared window in whole seconds, clamped to :data:`MAX_WINDOW_SECONDS`.

    Raises for a value that is not a non-negative number of seconds --
    ``OverflowError`` among them, which is what ``int()`` of the ``inf`` a
    ``1e999`` in the body parses to raises, and which no guard here caught.
    """
    seconds = int(raw)
    if seconds < 0:
        raise ValueError("a negative window")
    return min(seconds, MAX_WINDOW_SECONDS)


__all__ = [
    "CACHE_FILENAME",
    "DEFAULT_DISTRIBUTOR",
    "DEFAULT_STALE_AFTER_SECONDS",
    "GATE_NOT_ENTITLED",
    "GATE_RUN",
    "GATE_UNAVAILABLE",
    "SITUATIONS",
    "STATE_ACTIVE",
    "STATE_NOT_REQUIRED",
    "STATE_REVOKED",
    "STATE_UNKNOWN",
    "CheckAnswer",
    "CheckMisaddressed",
    "CheckUnreachable",
    "Entitlement",
    "PosternError",
    "Settings",
    "Verdict",
    "decide",
    "from_environment",
    "inbound_token_from_environment",
    "parse_rfc3339",
    "rfc3339",
    "settings_from_environment",
]
