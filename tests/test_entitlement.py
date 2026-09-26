"""The runner's entitlement client and the state machine of SPEC 5.7.

The interesting cases here are not "yes" and "no". They are the ones in
between — an answer that has expired, a distributor having a bad
afternoon, a machine that rebooted with no network — and the
specification's own table (SPEC 5.7.4, "Every case, in one table") is what
this file walks.

Three properties are worth naming, because each is a way an implementation
passes its happy-path tests and is still wrong:

* **Unreachable and refused are opposite answers.** ``5xx`` and a dropped
  connection let a running agent keep running inside its declared window
  and grace; ``404`` stops it now and for good. Swap the two and either a
  distributor's blip revokes every agent on the platform, or a refunded
  buyer runs for another day.
* **``checked_at`` is the distributor's clock.** Re-stamping it on receipt
  makes every field still validate while silently doubling the real
  staleness window, because the distributor's cache and the runner's then
  run back to back instead of sharing a deadline.
* **A restart is not a new window.** A persisted answer that reset its own
  deadlines would make restarting a way to mint grace on demand.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from sigrix_runtime.postern import entitlement as ent  # noqa: E402
from sigrix_runtime.postern import transport  # noqa: E402
from sigrix_runtime.postern.errors import (  # noqa: E402
    NOT_ENTITLED,
    NOT_ENTITLED_MESSAGE,
    UNAVAILABLE,
    WITHDRAWN_MESSAGE,
    PosternError,
)
from tests import postern_schema  # noqa: E402

AGENT_ID = "acme/market-research-crew"
NOW = datetime(2026, 8, 15, 9, 14, 2, tzinfo=UTC)


def _answer(state: str = ent.STATE_ACTIVE, *, checked_at: datetime = NOW, stale: int = 60, grace: int = 86400):
    return ent.CheckAnswer(state=state, checked_at=checked_at, stale_after_seconds=stale, grace_seconds=grace)


# ---------------------------------------------------------------------------
# SPEC 5.7.4's table
# ---------------------------------------------------------------------------


def test_the_table_covers_every_situation_the_specification_names() -> None:
    """The vocabulary, asserted as a set rather than reached by a branch.

    The dispatcher on the platform side keeps its authorities in a table for
    the same reason: a situation nobody wrote a row for should fail
    here, not fall through to whatever the last ``else`` returns.
    """
    assert {verdict.situation for verdict in ent.SITUATIONS} == {
        "no_distributor",
        "active_fresh",
        "refused",
        "unreachable_fresh",
        "unreachable_in_grace",
        "unreachable_past_grace",
        "never_checked",
    }
    assert len({verdict.situation for verdict in ent.SITUATIONS}) == len(ent.SITUATIONS)


@pytest.mark.parametrize(
    ("label", "answer", "configured", "at", "state", "gate"),
    [
        (
            "no distributor configured",
            None,
            False,
            NOW,
            ent.STATE_NOT_REQUIRED,
            ent.GATE_RUN,
        ),
        (
            "answered active, within the window",
            _answer(),
            True,
            NOW + timedelta(seconds=30),
            ent.STATE_ACTIVE,
            ent.GATE_RUN,
        ),
        (
            "answered revoked",
            _answer(ent.STATE_REVOKED),
            True,
            NOW + timedelta(seconds=1),
            ent.STATE_REVOKED,
            ent.GATE_NOT_ENTITLED,
        ),
        (
            "answered revoked, long past its window",
            _answer(ent.STATE_REVOKED),
            True,
            NOW + timedelta(days=30),
            ent.STATE_REVOKED,
            ent.GATE_NOT_ENTITLED,
        ),
        (
            "unreachable, past the window, within grace",
            _answer(),
            True,
            NOW + timedelta(seconds=120),
            ent.STATE_UNKNOWN,
            ent.GATE_RUN,
        ),
        (
            "unreachable, past the window and grace",
            _answer(),
            True,
            NOW + timedelta(seconds=60 + 86400 + 1),
            ent.STATE_UNKNOWN,
            ent.GATE_UNAVAILABLE,
        ),
        (
            "never checked at all",
            None,
            True,
            NOW,
            ent.STATE_UNKNOWN,
            ent.GATE_UNAVAILABLE,
        ),
    ],
)
def test_every_row_of_the_specifications_table(
    label: str, answer, configured: bool, at: datetime, state: str, gate: str
) -> None:
    verdict = ent.decide(answer, configured=configured, now=at)
    assert verdict.state == state, label
    assert verdict.gate == gate, label


def test_a_zero_grace_stops_at_the_window() -> None:
    """SPEC 5.7.1: ``0`` is a valid declaration meaning *stop at the window*.

    Declared rather than inferred from an absent field, which is why the
    parser refuses a body missing ``grace_seconds`` rather than defaulting
    it — a distributor that wants strictness has to say so.
    """
    strict = _answer(grace=0)
    assert ent.decide(strict, configured=True, now=NOW + timedelta(seconds=59)).gate == ent.GATE_RUN
    assert ent.decide(strict, configured=True, now=NOW + timedelta(seconds=61)).gate == ent.GATE_UNAVAILABLE


def test_a_refused_answer_never_drifts_into_grace() -> None:
    """ "Unreachable answers unavailable; refused answers not_entitled."

    Grace exists so a distributor's downtime is not everybody's. Nothing is
    down when the answer was ``revoked``, so a runner that let a refusal age
    into ``unknown`` would hand a refunded buyer the whole grace period.
    """
    for offset in (0, 61, 86401, 10 * 86400):
        verdict = ent.decide(_answer(ent.STATE_REVOKED), configured=True, now=NOW + timedelta(seconds=offset))
        assert verdict.gate == ent.GATE_NOT_ENTITLED, offset


def test_the_two_shapes_of_unknown_are_told_apart_by_checked_at() -> None:
    """SPEC 4.4: a runner inside grace, or one that cannot start.

    "A client can say which without a further field" — so the runner has to
    emit ``checked_at`` for the first and omit it for the second.
    """
    inside_grace = _in_grace().snapshot(now=NOW + timedelta(seconds=120))
    assert inside_grace["state"] == ent.STATE_UNKNOWN
    assert "checked_at" in inside_grace

    never = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID)
    cannot_start = never.snapshot(now=NOW)
    assert cannot_start["state"] == ent.STATE_UNKNOWN
    assert "checked_at" not in cannot_start


def _in_grace() -> ent.Entitlement:
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID)
    gate._store(_answer())
    return gate


# ---------------------------------------------------------------------------
# What status reports (SPEC 4.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", [ent.STATE_ACTIVE, ent.STATE_REVOKED])
def test_active_and_revoked_both_carry_the_window_and_the_timestamp(state: str) -> None:
    """SPEC 4.4 requires both for both, and says why for ``revoked``.

    "A timestamp with no duration beside it tells a runner when it was
    refused and never when to ask again", so the restoration SPEC 5.4
    obliges a distributor to support could not be observed.
    """
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID)
    gate._store(_answer(state))
    block = gate.snapshot(now=NOW + timedelta(seconds=1))
    assert block["state"] == state
    assert block["checked_at"] == "2026-08-15T09:14:02Z"
    assert block["stale_after_seconds"] == 60
    assert block["grace_seconds"] == 86400


def test_no_distributor_reports_not_required_and_never_refuses() -> None:
    """SPEC 5.1: an agent may be free, self-authored, or locally developed."""
    gate = ent.Entitlement(base_url="", token="", agent_id="")
    assert gate.configured is False
    assert gate.snapshot(now=NOW) == {"state": ent.STATE_NOT_REQUIRED}
    gate.gate(now=NOW)  # must not raise


@pytest.mark.parametrize(
    ("base_url", "token", "agent_id"),
    [("https://d.example", "", AGENT_ID), ("https://d.example", "t", ""), ("", "t", AGENT_ID)],
)
def test_half_a_configuration_is_not_a_configuration(base_url: str, token: str, agent_id: str) -> None:
    """Reporting ``not_required`` for it would be a lie a buyer cannot see.

    A base URL with no token cannot ask anything and a token with no agent
    id has nothing to ask about, so neither is a distributor that answered
    "this agent needs no licence".
    """
    assert ent.Entitlement(base_url=base_url, token=token, agent_id=agent_id).configured is False


# ---------------------------------------------------------------------------
# What run and stream do about it
# ---------------------------------------------------------------------------


def test_a_refusal_is_the_one_sanctioned_403() -> None:
    """SPEC 5.5: correct for a local runner reporting its own state.

    There is nothing to enumerate — the client is already talking to the
    runner for the one agent it serves.
    """
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID)
    gate._store(_answer(ent.STATE_REVOKED))
    with pytest.raises(PosternError) as caught:
        gate.gate(now=NOW + timedelta(seconds=1))
    assert (caught.value.status, caught.value.code) == (403, NOT_ENTITLED)


def test_never_having_been_told_anything_is_503_rather_than_a_refusal() -> None:
    """SPEC 5.7.3: ``not_entitled`` would assert something no distributor said."""
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, timeout=0.05)
    with pytest.raises(PosternError) as caught:
        gate.gate(now=NOW)
    assert (caught.value.status, caught.value.code) == (503, UNAVAILABLE)


def test_the_refusal_message_names_no_cause() -> None:
    """SPEC 5.7.4: three causes the runner cannot tell apart, one sentence.

    A withdrawn entitlement, one that never existed and a token that no
    longer resolves are made indistinguishable at the distributor by SPEC
    5.5, and "that indistinguishability reaches the runner's vocabulary
    too". A message that guessed would undo the rule in prose.
    """
    message = ent.not_entitled().message
    for leak in ("revoked", "refund", "token is invalid", "not found", "expired"):
        assert leak.lower() not in message.lower().replace("refunded", "")


# ---------------------------------------------------------------------------
# The wire: what a check answer has to carry to be believed
# ---------------------------------------------------------------------------


class _Distributor:
    """A stand-in for SPEC 5.3's endpoint, on loopback."""

    def __init__(self, handler) -> None:
        self.requests: list[tuple[str, str]] = []
        # captured separately rather than widening the tuple above,
        # which every pre-existing test in this file unpacks positionally.
        # `None` when the header is genuinely absent, distinct from a sent
        # empty string — the two are different claims about the request.
        self.delivery_mode_headers: list[str | None] = []
        outer = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                outer.requests.append((self.path, self.headers.get("Authorization") or ""))
                outer.delivery_mode_headers.append(self.headers.get("X-Sigrix-Delivery-Mode"))
                status, payload = handler(self.path)
                body = json.dumps(payload).encode("utf-8") if payload is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except ConnectionError:
                    pass  # a runner that stopped reading at its bound, and hung up
                # One request per connection, as the runner makes them: a handler
                # left waiting for a second meets that hang-up as a reset.
                self.close_connection = True

            def log_message(self, *args) -> None:  # noqa: A002
                pass

        self._server = HTTPServer(("127.0.0.1", 0), _H)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def distributor():
    made: list[_Distributor] = []

    def _make(handler) -> _Distributor:
        instance = _Distributor(handler)
        made.append(instance)
        return instance

    yield _make
    for instance in made:
        instance.close()


def _good_body(**overrides):
    body = {
        "postern": "0.1",
        "state": "active",
        "agent_id": AGENT_ID,
        "checked_at": "2026-08-15T09:14:02Z",
        "stale_after_seconds": 60,
        "grace_seconds": 86400,
    }
    body.update(overrides)
    return body


def test_a_good_answer_is_addressed_as_two_path_segments_and_carries_the_bearer(distributor) -> None:
    """SPEC 5.3.1: an identifier occupies two path segments, never ``%2F``."""
    served = distributor(lambda path: (200, _good_body()))
    gate = ent.Entitlement(base_url=served.base_url, token="secret-token", agent_id=AGENT_ID)
    answer = gate.check(agent_id=AGENT_ID)

    assert answer.state == ent.STATE_ACTIVE
    assert answer.checked_at == NOW
    assert answer.stale_after_seconds == 60 and answer.grace_seconds == 86400
    path, authorization = served.requests[-1]
    assert path == "/postern/v0/entitlements/acme/market-research-crew"
    assert "%2F" not in path
    assert authorization == "Bearer secret-token"


def test_checked_at_is_the_distributors_clock_and_is_not_re_stamped(distributor) -> None:
    """SPEC 4.4, 5.3. The anchor is what stops the two caches stacking.

    Re-stamping on receipt leaves every field validating while the real
    worst case becomes the sum of the distributor's cache and the runner's,
    where ``stale_after_seconds`` claims to be the whole of it.
    """
    upstream = "2020-01-01T00:00:00Z"
    served = distributor(lambda path: (200, _good_body(checked_at=upstream)))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate.refresh(now=NOW, force=True)
    assert gate.snapshot(now=NOW)["checked_at"] == upstream


def test_an_answer_for_a_different_agent_is_a_failed_check_not_a_reconciliation(distributor) -> None:
    """SPEC 5.3: the echo is the question, not the result of a lookup."""
    served = distributor(lambda path: (200, _good_body(agent_id="someone/else")))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    with pytest.raises(ent.CheckUnreachable):
        gate.check(agent_id=AGENT_ID)


@pytest.mark.parametrize("dropped", ["checked_at", "stale_after_seconds", "grace_seconds"])
def test_a_partial_answer_is_unreachable_rather_than_a_lenient_success(distributor, dropped: str) -> None:
    """ "Every member above is REQUIRED" (SPEC 5.3).

    Landing these as *unreachable* is the conservative half: the runner
    keeps its last good answer and its deadlines instead of replacing them
    with one that cannot bound anything.
    """
    body = _good_body()
    body.pop(dropped)
    served = distributor(lambda path: (200, body))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    with pytest.raises(ent.CheckUnreachable):
        gate.check(agent_id=AGENT_ID)


@pytest.mark.parametrize("status", [500, 502, 503])
def test_a_5xx_is_unreachable_and_keeps_the_previous_answer(distributor, status: int) -> None:
    """SPEC 5.7. A blink must not revoke a running agent."""
    served = distributor(lambda path: (status, {"error": {"code": "unavailable"}}))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate._store(_answer())
    verdict = gate.refresh(now=NOW + timedelta(seconds=61))
    assert verdict.situation == "unreachable_in_grace"
    assert gate.snapshot(now=NOW + timedelta(seconds=61))["state"] == ent.STATE_UNKNOWN


def test_a_404_is_an_answer_and_stops_the_agent_immediately(distributor) -> None:
    """SPEC 5.7.4: no grace applies, because nothing failed."""
    served = distributor(lambda path: (404, {"error": {"code": "not_found", "message": "…", "detail": None}}))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate._store(_answer())
    verdict = gate.refresh(now=NOW + timedelta(seconds=61))
    assert verdict.situation == "refused"
    assert verdict.gate == ent.GATE_NOT_ENTITLED


def test_a_404_reuses_the_last_window_the_distributor_gave(distributor) -> None:
    """SPEC 5.7.4: the constant body carries none, so the runner reuses one.

    Attaching fields to that body would rebuild the enumeration oracle SPEC
    5.5 exists to prevent, so a runner keeps its own cadence — and asking
    again is the only way a reversal is ever observed.
    """
    answers = [(200, _good_body(stale_after_seconds=900)), (404, {"error": {"code": "not_found"}})]
    served = distributor(lambda path: answers.pop(0))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate.refresh(now=NOW, force=True)
    gate.refresh(now=NOW + timedelta(seconds=901), force=True)
    assert gate.snapshot()["stale_after_seconds"] == 900


def test_a_first_ever_404_still_reports_a_window(distributor) -> None:
    """SPEC 5.7.4, as clarified upstream in postern c14d6af (#88).

    "A first check answering ``404`` is what a token revoked before its
    runner ever ran produces, and what a runner pointed at the wrong
    distributor sees; there is no earlier answer anywhere to reuse." §4.4
    still requires ``stale_after_seconds`` under ``revoked``, so the
    runner's own re-check cadence stands in — and a runner omitting the
    field would answer ``status`` with a payload the schema rejects, which
    is the failure this case is easy to ship.

    Note where it lands: ``refused``, not ``never_checked``. "A check that
    answered ``404`` has completed, whatever it said."
    """
    served = distributor(lambda path: (404, {"error": {"code": "not_found"}}))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    assert gate._answer is None  # nothing to reuse

    verdict = gate.refresh(now=NOW)
    assert verdict.situation == "refused"
    block = gate.snapshot(now=NOW)
    assert block["state"] == ent.STATE_REVOKED
    assert block["stale_after_seconds"] == ent.DEFAULT_STALE_AFTER_SECONDS
    assert block["checked_at"]
    # The envelope is scaffolding — this case is about ``block`` — but it has to
    # be a *valid* status for the check to mean anything, and upstream `afa595b`
    # made ``agent`` required there (SPEC 1.5: the identifier appears in status
    # as well as describe, because 2.2's one-runner-one-agent rule is only
    # checkable when it does). The runner itself already satisfies this from
    # ``load_describe``, which fills ``agent.id`` with ``local_agent_id`` when
    # nothing configured one; only this hand-built payload was short.
    postern_schema.validate(
        {
            "postern": "0.1",
            "level": 3,
            "state": "ready",
            "agent": {"id": AGENT_ID},
            "entitlement": block,
        },
        postern_schema.load("status"),
        name="status",
    )


def test_a_fresh_answer_is_not_re_checked(distributor) -> None:
    """SPEC 5.4 sets the cadence: re-check on the first request past the window."""
    served = distributor(lambda path: (200, _good_body()))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate.refresh(now=NOW, force=True)
    before = len(served.requests)
    gate.refresh(now=NOW + timedelta(seconds=30))
    assert len(served.requests) == before
    gate.refresh(now=NOW + timedelta(seconds=61))
    assert len(served.requests) == before + 1


# ---------------------------------------------------------------------------
# ``status`` asks too, on its own terms, and a refusal points somewhere
# ---------------------------------------------------------------------------


def test_status_re_checks_on_the_first_request_past_the_window(distributor) -> None:
    """SPEC 5.4's cadence does not exempt ``status``.

    The probe from the issue: an ``active`` answer past its window, the
    distributor now answering 404. Before this, three ``status`` calls made
    zero requests and reported the stale state until a ``run`` happened to
    ask — while ``__main__`` told the buyer ``status`` was how to find out.
    """
    revoked = {"now": False}
    served = distributor(lambda path: (404, None) if revoked["now"] else (200, _good_body()))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate.refresh(now=NOW, force=True)
    revoked["now"] = True
    before = len(served.requests)

    inside = gate.snapshot(now=NOW + timedelta(seconds=30))
    assert inside["state"] == ent.STATE_ACTIVE
    assert len(served.requests) == before, "inside the window, status reports the held answer"

    past = gate.snapshot(now=NOW + timedelta(seconds=61))
    assert past["state"] == ent.STATE_REVOKED
    assert len(served.requests) == before + 1, "the first status past the window asks"

    again = gate.snapshot(now=NOW + timedelta(seconds=62))
    assert again["state"] == ent.STATE_REVOKED
    assert len(served.requests) == before + 1, "a fresh answer is not asked for again"


def test_a_status_re_check_is_bounded_and_keeps_the_held_answer(distributor) -> None:
    """``status`` asks under its own bound, and a timeout is *unreachable*.

    The distributor stalls past the bound. ``status`` answers within it,
    still holding the answer it had — inside grace, so ``unknown`` with the
    old ``checked_at`` — which is where a ``run``'s failed check would leave
    it too, minus the ten-second wait.
    """
    stall = {"seconds": 0.0}

    def handler(path):
        time.sleep(stall["seconds"])
        return (200, _good_body())

    served = distributor(handler)
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate.refresh(now=NOW, force=True)
    held = gate.snapshot(now=NOW)
    stall["seconds"] = 0.8
    started = time.monotonic()
    block = gate.snapshot(now=NOW + timedelta(seconds=61), timeout=0.2)
    assert time.monotonic() - started < 0.7
    assert block["state"] == ent.STATE_UNKNOWN
    assert block["checked_at"] == held["checked_at"]


def test_status_holds_off_after_an_unreachable_re_check_and_run_does_not(distributor) -> None:
    """a polled ``status`` on a dead distributor is not a request storm.

    One attempt per hold-off window from ``status``; ``run`` (``refresh``)
    keeps asking, which is what SPEC 5.7.1 wants of the check.
    """
    down = {"now": False}
    served = distributor(lambda path: (503, None) if down["now"] else (200, _good_body()))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate.refresh(now=NOW, force=True)
    down["now"] = True
    before = len(served.requests)

    past = NOW + timedelta(seconds=61)
    assert gate.snapshot(now=past)["state"] == ent.STATE_UNKNOWN
    assert len(served.requests) == before + 1
    gate.snapshot(now=past + timedelta(seconds=1))
    assert len(served.requests) == before + 1, "held off: the next poll does not ask again"
    gate.refresh(now=past + timedelta(seconds=2))
    assert len(served.requests) == before + 2, "a run is never held off"
    gate.snapshot(now=past + timedelta(seconds=ent.STATUS_RECHECK_HOLD_OFF_SECONDS))
    assert len(served.requests) == before + 3, "the hold-off has an end"


def test_status_never_waits_on_a_check_another_request_holds(distributor) -> None:
    """a ``run``'s check in flight is not a ``status``'s to queue behind."""
    served = distributor(lambda path: (200, _good_body()))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate.refresh(now=NOW, force=True)
    before = len(served.requests)
    assert gate._check_lock.acquire(blocking=False)
    try:
        started = time.monotonic()
        block = gate.snapshot(now=NOW + timedelta(seconds=61))
        assert time.monotonic() - started < 0.5
    finally:
        gate._check_lock.release()
    assert len(served.requests) == before
    assert block["state"] == ent.STATE_UNKNOWN


def test_a_refusal_points_at_the_listing_and_still_names_no_cause() -> None:
    """what a refused buyer is owed is somewhere to go, not a diagnosis."""
    url = "https://sigrix.io/crew/market-research-crew"
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, listing_url=url)
    gate._store(_answer(ent.STATE_REVOKED))
    with pytest.raises(PosternError) as caught:
        gate.gate(now=NOW + timedelta(seconds=1))
    error = caught.value
    assert error.message.startswith(ent.not_entitled().message)
    assert url in error.message
    for leak in ("revoked", "refund", "token is invalid", "not found", "expired"):
        assert leak not in error.message.lower().replace("refunded", "")
    assert error.detail == {"state": ent.STATE_REVOKED, "checked_at": ent.rfc3339(NOW), "listing_url": url}
    postern_schema.validate(error.envelope(), postern_schema.load("error"), name="error")
    assert gate.snapshot(now=NOW + timedelta(seconds=1))["listing_url"] == url


def test_a_refusal_without_a_listing_page_still_says_where_to_look() -> None:
    """a bundle with no ``plugin.json`` gets the pull's own pointer, and no link."""
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID)
    gate._store(_answer(ent.STATE_REVOKED))
    with pytest.raises(PosternError) as caught:
        gate.gate(now=NOW + timedelta(seconds=1))
    assert "SIGRIX_TOKEN" in caught.value.message and "POSTERN_AGENT_ID" in caught.value.message
    assert "listing_url" not in caught.value.detail
    assert "listing_url" not in gate.snapshot(now=NOW + timedelta(seconds=1))


# ---------------------------------------------------------------------------
# The date a withdrawn listing's access ends (SPEC 5.3's access_ends_at)
# ---------------------------------------------------------------------------

ENDS = NOW + timedelta(days=200)
LISTING_URL = "https://sigrix.io/crew/market-research-crew"


def _dated(state: str = ent.STATE_ACTIVE, *, ends_at: datetime = ENDS, checked_at: datetime = NOW) -> ent.CheckAnswer:
    return ent.CheckAnswer(
        state=state,
        checked_at=checked_at,
        stale_after_seconds=60,
        grace_seconds=86400,
        access_ends_at=ends_at,
    )


def test_the_date_is_read_off_the_wire_and_reported_in_status(distributor) -> None:
    """SPEC 4.4: the date the distributor gave, unchanged, in ``status.entitlement``."""
    served = distributor(lambda path: (200, _good_body(access_ends_at="2027-03-01T00:00:00Z")))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate._store(gate.check(agent_id=AGENT_ID))

    block = gate.snapshot(now=NOW + timedelta(seconds=1))

    assert block["access_ends_at"] == "2027-03-01T00:00:00Z"
    status = {"postern": "0.1", "level": 3, "state": "ready", "agent": {"id": AGENT_ID}, "entitlement": block}
    postern_schema.validate(status, postern_schema.load("status"), name="status")


def test_an_answer_with_no_date_reports_none(distributor) -> None:
    served = distributor(lambda path: (200, _good_body()))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate._store(gate.check(agent_id=AGENT_ID))

    assert "access_ends_at" not in gate.snapshot(now=NOW + timedelta(seconds=1))
    assert gate.ends_on() == ""


def test_a_date_that_does_not_parse_costs_the_date_and_not_the_answer(distributor) -> None:
    """Optional, so the state and the windows it arrived beside still count."""
    served = distributor(lambda path: (200, _good_body(access_ends_at="soon")))
    answer = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID).check(agent_id=AGENT_ID)

    assert answer.state == ent.STATE_ACTIVE
    assert answer.access_ends_at is None


def test_the_date_is_kept_through_grace(distributor) -> None:
    """The last thing the distributor said, still worth saying while it cannot be asked."""
    served = distributor(lambda path: (503, None))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate._store(_dated())

    block = gate.snapshot(now=NOW + timedelta(seconds=120))

    assert block["state"] == ent.STATE_UNKNOWN
    assert block["access_ends_at"] == ent.rfc3339(ENDS)


def test_the_date_survives_a_restart(tmp_path: Path) -> None:
    cache = tmp_path / ent.CACHE_FILENAME
    ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)._store(_dated())

    restarted = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)

    assert restarted.snapshot(now=NOW + timedelta(seconds=1))["access_ends_at"] == ent.rfc3339(ENDS)


def test_a_404_carries_no_date(distributor) -> None:
    """SPEC 5.5 keeps that body constant, so there is nothing to carry one in."""
    served = distributor(lambda path: (404, {"error": {"code": "not_found", "message": "No."}}))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate._store(_dated())

    gate._store(gate.check(agent_id=AGENT_ID))

    assert gate._answer is not None and gate._answer.access_ends_at is None
    assert "access_ends_at" not in gate.snapshot()


def test_access_that_ran_out_is_refused_as_a_withdrawal_with_its_date() -> None:
    """Not a refund and not a token to replace, so neither is said (SPEC 5.3)."""
    ended = NOW - timedelta(days=1)
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, listing_url=LISTING_URL)
    gate._store(_dated(ent.STATE_REVOKED, ends_at=ended))

    with pytest.raises(PosternError) as caught:
        gate.gate(now=NOW + timedelta(seconds=1))

    error = caught.value
    assert (error.status, error.code) == (403, NOT_ENTITLED)
    assert error.message.startswith(WITHDRAWN_MESSAGE)
    assert "2026-08-14" in error.message
    assert NOT_ENTITLED_MESSAGE not in error.message
    assert "SIGRIX_TOKEN" not in error.message
    assert LISTING_URL in error.message
    assert error.detail["access_ends_at"] == ent.rfc3339(ended)
    postern_schema.validate(error.envelope(), postern_schema.load("error"), name="error")
    assert gate.ended_on(NOW + timedelta(seconds=1)) == "2026-08-14"


def test_a_revoked_answer_dated_ahead_is_the_plain_refusal() -> None:
    """A ``revoked`` whose date has not come contradicts itself; the plain sentence is safe."""
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID)
    gate._store(_dated(ent.STATE_REVOKED, ends_at=NOW + timedelta(days=5)))

    with pytest.raises(PosternError) as caught:
        gate.gate(now=NOW + timedelta(seconds=1))

    assert caught.value.message.startswith(NOT_ENTITLED_MESSAGE)


def test_the_date_adds_no_deadline() -> None:
    """SPEC 5.3: a runner MUST NOT refuse a run on the strength of the date alone.

    Only a later ``revoked`` ends access, from the one party that can still move
    the date; this answer is ``active`` and inside its window.
    """
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID)
    gate._store(_dated(ends_at=NOW - timedelta(hours=1)))

    gate.gate(now=NOW + timedelta(seconds=1))  # must not raise


def test_a_run_warns_that_access_is_ending_at_most_once_a_day(caplog: pytest.LogCaptureFixture) -> None:
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, listing_url=LISTING_URL)
    gate._store(_dated())
    day_later = NOW + ent.ACCESS_WARNING_INTERVAL

    with caplog.at_level("WARNING", logger="sigrix_runtime.postern.entitlement"):
        gate.gate(now=NOW + timedelta(seconds=1))
        gate.gate(now=NOW + timedelta(seconds=2))
        gate._store(_dated(checked_at=day_later))
        gate.gate(now=day_later + timedelta(seconds=1))

    warned = [record.getMessage() for record in caplog.records]
    assert len(warned) == 2, warned
    assert all(WITHDRAWN_MESSAGE in line and ent.rfc3339(ENDS)[:10] in line for line in warned)
    assert all(LISTING_URL in line for line in warned)


# ---------------------------------------------------------------------------
# Plaintext, and the address rather than the name (SPEC 7)
# ---------------------------------------------------------------------------


def test_a_token_crosses_plaintext_only_to_a_loopback_peer(distributor, monkeypatch) -> None:
    """SPEC 7's exception, and the condition it is evaluated on.

    "``localhost`` is a name and a resolver decides what it means", so this
    forces the connected peer to look non-loopback while the URL still says
    ``127.0.0.1`` — a check made on the hostname would pass it.
    """
    served = distributor(lambda path: (200, _good_body()))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    assert gate.check(agent_id=AGENT_ID).state == ent.STATE_ACTIVE

    monkeypatch.setattr(transport, "_peer_is_loopback", lambda sock: False)
    with pytest.raises(ent.CheckUnreachable) as caught:
        gate.check(agent_id=AGENT_ID)
    assert "plaintext" in str(caught.value)


def test_an_https_distributor_is_never_subject_to_the_exception() -> None:
    """The relaxation is plaintext's alone; TLS needs no loopback peer."""
    gate = ent.Entitlement(base_url="https://sigrix.io", token="t", agent_id=AGENT_ID, timeout=0.05)
    with pytest.raises(ent.CheckUnreachable) as caught:
        gate.check(agent_id=AGENT_ID)
    assert "plaintext" not in str(caught.value)


# ---------------------------------------------------------------------------
# A malformed identifier is configuration, not an outage and not a refund
# ---------------------------------------------------------------------------


def test_a_malformed_identifier_is_never_sent(distributor) -> None:
    """With no slash it went out as ``…/entitlements/acme/``, met the catch-all 404, and read as a refund."""
    served = distributor(lambda path: (404, {"error": {"code": "not_found", "message": "…", "detail": None}}))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id="acme")
    with pytest.raises(PosternError) as refused:
        gate.gate(now=NOW)
    assert (refused.value.status, refused.value.code) == (503, UNAVAILABLE)
    assert "POSTERN_AGENT_ID" in refused.value.message
    assert served.requests == []
    assert gate.snapshot(now=NOW) == {"state": ent.STATE_UNKNOWN}


def test_a_400_from_the_distributor_is_a_configuration_refusal_even_inside_grace(distributor) -> None:
    """SPEC 5.3.1's ``400`` is computed from the request string, so no retry and no grace changes it.

    It was folded into *unreachable*: inside grace the run went ahead, and past
    it every ``run`` told the buyer to check their network for a typo no
    network can fix.
    """
    served = distributor(lambda path: (400, {"error": {"code": "bad_request", "message": "…", "detail": None}}))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate._store(_answer())
    in_grace = NOW + timedelta(seconds=61)

    with pytest.raises(PosternError) as refused:
        gate.gate(now=in_grace)
    assert (refused.value.status, refused.value.code) == (503, UNAVAILABLE)
    assert "refused this runner's agent identifier" in refused.value.message
    assert "network" not in refused.value.message
    assert gate.snapshot(now=in_grace) == {"state": ent.STATE_UNKNOWN}


def test_an_answer_after_a_400_clears_it(distributor) -> None:
    answers = iter([(400, None), (200, _good_body())])
    served = distributor(lambda path: next(answers))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    with pytest.raises(ent.CheckMisaddressed):
        gate.check(agent_id=AGENT_ID)
    assert gate.misaddressed

    gate.refresh(now=NOW, force=True)
    assert gate.misaddressed == ""
    assert gate.snapshot(now=NOW)["state"] == ent.STATE_ACTIVE


# ---------------------------------------------------------------------------
# A distributor is not trusted with the runner's memory or its clock
# ---------------------------------------------------------------------------


def test_an_answer_past_the_read_bound_is_unreachable_rather_than_read_whole(distributor) -> None:
    """A check answer is a few hundred bytes; it was read with no bound at all.

    The review's 60 MB body cost the runner 367 MB of memory. This one is a
    well-formed ``active`` answer padded past the bound -- read whole, it
    was believed.
    """
    # Built here rather than in the handler: raised on the server's thread, a
    # failure to build it looked exactly like a refusal, and passed.
    padded = _good_body(padding="x" * (2 * transport.MAX_ANSWER_BYTES))
    served = distributor(lambda path: (200, padded))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    with pytest.raises(ent.CheckUnreachable, match="ran past"):
        gate.check(agent_id=AGENT_ID)
    assert len(served.requests) == 1


def test_a_window_past_what_a_date_can_count_is_held_at_the_ceiling(distributor) -> None:
    """``stale_after_seconds=10**18`` was accepted, persisted, then raised on every read.

    ``timedelta`` stops at 999,999,999 days, and the ``OverflowError`` was in
    no guard: every ``status`` and ``run`` answered 503 "internal error",
    surviving a restart until the cache file was deleted. Held at ten years
    instead, which only re-checks sooner than declared.
    """
    served = distributor(lambda path: (200, _good_body(stale_after_seconds=10**18, grace_seconds=10**18)))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    verdict = gate.refresh(now=NOW, force=True)

    assert verdict.situation == "active_fresh"
    snapshot = gate.snapshot(now=NOW)
    assert snapshot["stale_after_seconds"] == ent.MAX_WINDOW_SECONDS
    assert snapshot["grace_seconds"] == ent.MAX_WINDOW_SECONDS


def test_an_infinite_window_is_a_failed_check_not_an_internal_error(distributor) -> None:
    """``1e999`` parses to ``inf``, and ``int(inf)`` raises the ``OverflowError`` nothing caught."""
    served = distributor(lambda path: (200, _good_body(stale_after_seconds=float("inf"))))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate._store(_answer())
    assert gate.refresh(now=NOW + timedelta(seconds=61)).situation == "unreachable_in_grace"


@pytest.mark.parametrize("checked_at", ["9999-12-31T23:59:59Z", "9999-12-31T22:00:00-05:00"])
def test_a_checked_at_whose_deadlines_are_past_year_9999_is_a_failed_check(distributor, checked_at: str) -> None:
    """Clamping the windows is not enough: the anchor itself can carry them off the calendar.

    The first overflows adding even a 60-second window; the second is a
    moment in year 9999 locally and in 10000 in UTC, so it overflowed on the
    way out, in every ``status``. Both land as unreachable, keeping the last
    good answer rather than holding one no deadline can be read from.
    """
    served = distributor(lambda path: (200, _good_body(checked_at=checked_at)))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID)
    gate._store(_answer())
    later = NOW + timedelta(seconds=61)
    assert gate.refresh(now=later).situation == "unreachable_in_grace"
    assert gate.snapshot(now=later)["checked_at"] == "2026-08-15T09:14:02Z"


def test_a_persisted_answer_from_before_the_bound_is_read_at_the_ceiling(tmp_path: Path) -> None:
    """The failure survived a restart because the answer had been written down.

    A runner upgraded over a cache an older one wrote, before the window was
    bounded, reads the window at the ceiling instead of raising on it.
    """
    cache = tmp_path / ent.CACHE_FILENAME
    first = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)
    first._store(_answer(stale=10**18))

    restarted = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)
    assert restarted.snapshot(now=NOW)["stale_after_seconds"] == ent.MAX_WINDOW_SECONDS
    assert ent.decide(restarted._answer, configured=True, now=NOW).gate == ent.GATE_RUN


def test_a_persisted_answer_no_deadline_can_be_read_from_is_never_checked(tmp_path: Path) -> None:
    cache = tmp_path / ent.CACHE_FILENAME
    first = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)
    first._store(_answer(checked_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=UTC)))

    restarted = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)
    assert restarted._answer is None
    assert restarted.snapshot(now=NOW) == {"state": ent.STATE_UNKNOWN}


# ---------------------------------------------------------------------------
# Persistence (SPEC 5.7.2)
# ---------------------------------------------------------------------------


def test_a_restart_is_not_a_new_window(tmp_path: Path) -> None:
    """SPEC 5.7.2: both deadlines are evaluated against the stored ``checked_at``.

    Without it a runner that reboots with no network holds nothing and
    "the machine that worked before the power cut does not work after it".
    With it — and only with it — a restart cannot mint fresh grace either.
    """
    cache = tmp_path / ent.CACHE_FILENAME
    first = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)
    first._store(_answer())

    restarted = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)
    assert restarted.snapshot(now=NOW + timedelta(seconds=30))["checked_at"] == "2026-08-15T09:14:02Z"
    assert ent.decide(restarted._answer, configured=True, now=NOW + timedelta(seconds=30)).gate == ent.GATE_RUN
    past_everything = NOW + timedelta(seconds=60 + 86400 + 1)
    assert ent.decide(restarted._answer, configured=True, now=past_everything).gate == ent.GATE_UNAVAILABLE


def test_a_persisted_answer_is_ignored_for_another_agent_or_another_token(tmp_path: Path) -> None:
    """Neither key alone is enough, and both name a real situation.

    A bundle re-pointed at another listing, and a buyer who rotated their
    token after a leak, are each a reason the stored answer is about
    somebody else's question.
    """
    cache = tmp_path / ent.CACHE_FILENAME
    ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)._store(_answer())

    other_agent = ent.Entitlement(base_url="https://d.example", token="t", agent_id="acme/other", cache_path=cache)
    other_token = ent.Entitlement(base_url="https://d.example", token="u", agent_id=AGENT_ID, cache_path=cache)
    assert other_agent._answer is None
    assert other_token._answer is None


def test_the_cache_file_never_holds_the_token(tmp_path: Path) -> None:
    """It sits in the bundle directory; a bearer secret does not belong there."""
    cache = tmp_path / ent.CACHE_FILENAME
    ent.Entitlement(base_url="https://d.example", token="super-secret", agent_id=AGENT_ID, cache_path=cache)._store(
        _answer()
    )
    written = cache.read_text(encoding="utf-8")
    assert "super-secret" not in written
    assert json.loads(written)["token_fingerprint"]


def test_a_write_that_dies_halfway_leaves_the_previous_answer(tmp_path: Path, monkeypatch) -> None:
    """Written in place, a torn write read back as never checked.

    That is SPEC 5.7.2's offline restart defeated by a full disk: the one
    answer the runner had, gone the moment it tried to keep a newer one.
    """
    cache = tmp_path / ent.CACHE_FILENAME
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)
    gate._store(_answer())
    kept = cache.read_text(encoding="utf-8")

    real_write_text = Path.write_text

    def dies_halfway(self: Path, data: str, *args, **kwargs):
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", dies_halfway)
    gate._store(_answer(stale=120))
    monkeypatch.undo()

    assert cache.read_text(encoding="utf-8") == kept
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [ent.CACHE_FILENAME]
    restarted = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)
    assert restarted._answer is not None and restarted._answer.stale_after_seconds == 60


def test_a_corrupt_cache_reads_as_never_checked(tmp_path: Path) -> None:
    """Which stops the agent (SPEC 5.7.3) rather than starting it unlicensed."""
    cache = tmp_path / ent.CACHE_FILENAME
    cache.write_text("{not json", encoding="utf-8")
    gate = ent.Entitlement(base_url="https://d.example", token="t", agent_id=AGENT_ID, cache_path=cache)
    assert gate._answer is None
    assert ent.decide(gate._answer, configured=True, now=NOW).situation == "never_checked"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_environment_names_the_three_things_a_runner_needs(tmp_path: Path) -> None:
    gate = ent.from_environment(
        tmp_path,
        {"SIGRIX_TOKEN": "tok", "POSTERN_AGENT_ID": AGENT_ID, "POSTERN_DISTRIBUTOR": "https://example.test"},
    )
    assert (gate.token, gate.agent_id, gate.base_url) == ("tok", AGENT_ID, "https://example.test")
    assert gate.cache_path == tmp_path / ent.CACHE_FILENAME


def test_the_distributor_defaults_so_a_buyer_sets_two_variables(tmp_path: Path) -> None:
    gate = ent.from_environment(tmp_path, {"SIGRIX_TOKEN": "tok", "POSTERN_AGENT_ID": AGENT_ID})
    assert gate.base_url == "https://sigrix.io"
    assert gate.configured is True


# ---------------------------------------------------------------------------
# Delivery-mode pilot instrumentation
# ---------------------------------------------------------------------------


def test_an_ordinary_bundle_defaults_to_zip(tmp_path: Path) -> None:
    """No SIGRIX_DELIVERY_MODE at all is what a raw unzipped bundle sees —
    nothing sets it there, only the container image does."""
    gate = ent.from_environment(tmp_path, {"SIGRIX_TOKEN": "tok", "POSTERN_AGENT_ID": AGENT_ID})
    assert gate.delivery_mode == ent.DELIVERY_MODE_ZIP


def test_the_container_shape_is_read_from_the_environment(tmp_path: Path) -> None:
    gate = ent.from_environment(
        tmp_path,
        {"SIGRIX_TOKEN": "tok", "POSTERN_AGENT_ID": AGENT_ID, "SIGRIX_DELIVERY_MODE": "container"},
    )
    assert gate.delivery_mode == ent.DELIVERY_MODE_CONTAINER


def test_the_bundles_own_env_file_is_never_consulted_for_it(tmp_path: Path) -> None:
    """Unlike the three distributor settings, this is an operator fact
    (the image's own ENV), not a buyer credential — so the bundle's ``.env``
    fallback that ``settings_from_environment`` uses for the other three must
    not apply here."""
    (tmp_path / ".env").write_text("SIGRIX_DELIVERY_MODE=container\n", encoding="utf-8")
    gate = ent.from_environment(tmp_path, {"SIGRIX_TOKEN": "tok", "POSTERN_AGENT_ID": AGENT_ID})
    assert gate.delivery_mode == ent.DELIVERY_MODE_ZIP


def test_the_check_sends_the_delivery_mode_header(distributor) -> None:
    served = distributor(lambda path: (200, _good_body()))
    gate = ent.Entitlement(
        base_url=served.base_url, token="t", agent_id=AGENT_ID, delivery_mode=ent.DELIVERY_MODE_CONTAINER
    )
    gate.check(agent_id=AGENT_ID)
    assert served.delivery_mode_headers[-1] == "container"


def test_an_unconfigured_delivery_mode_sends_no_header_at_all(distributor) -> None:
    """An explicitly empty ``delivery_mode`` (never produced by
    ``from_environment``, whose default is ``zip``, but reachable by
    constructing :class:`ent.Entitlement` directly) omits the header rather
    than sending it empty — a distributor reading its logs sees "nothing
    sent" rather than a value that parses as present-but-blank."""
    served = distributor(lambda path: (200, _good_body()))
    gate = ent.Entitlement(base_url=served.base_url, token="t", agent_id=AGENT_ID, delivery_mode="")
    gate.check(agent_id=AGENT_ID)
    assert served.delivery_mode_headers[-1] is None
