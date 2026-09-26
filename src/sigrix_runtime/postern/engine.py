"""Driving one run, and shaping what SPEC 4.2 and 4.3 return.

The engine spawns :mod:`sigrix_runtime.postern._worker`, relays what it
writes as Postern events, and enforces the two bounds the specification
lets a runner declare: a maximum duration and a maximum number of runs in
flight.

**``usage.cost_usd`` is an estimate a bundle carries, and it can go stale.**
Postern calls it advisory and forbids a client from treating it as a
billed amount, which is the right reading of a number computed from a
price table frozen at the moment this bundle was built. :data:`MODEL_PRICING`
mirrors the platform's own table, and a model absent from it produces no
``cost_usd`` at all rather than a zero — an absent estimate is a fact
about the runner, where a zero is a claim about the run.

**No ``delta``.** SPEC 4.3 makes the event OPTIONAL and requires that, if
any is emitted, concatenating every ``delta.text`` equals the final
``output.value``. CrewAI's ``kickoff`` returns its result whole rather than
incrementally, so there is no honest text to send — and the specification
says in as many words that "a runner that cannot produce incremental text
emits none". Progress rides on ``step`` instead, which is the useful signal
for a crew anyway: a multi-agent run takes minutes, and what a buyer wants
to watch is which member is working, not characters arriving.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sigrix_runtime.postern import POSTERN_VERSION
from sigrix_runtime.postern.errors import (
    PosternError,
    agent_error,
    bad_request,
    requirements_not_installed,
    run_timeout,
    unavailable,
)

# USD per million tokens. Mirrors ``MODEL_PRICING`` in the platform's
# ``services/sigrix_example_runner.py``; a test holds the two in step, on
# the mirrored-constant rule this codebase has been bitten by before.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}

# How long to wait for a killed worker to die before giving up on it.
_REAP_SECONDS = 5

logger = logging.getLogger("sigrix_runtime.postern")


@dataclass
class RunEvent:
    """One frame from the worker, already in Postern's vocabulary."""

    name: str
    payload: dict[str, Any]


@dataclass
class Limits:
    """The bounds this deployment puts on a run (SPEC 4.4's ``limits``).

    ``max_run_seconds`` of ``0`` means no limit, and the field is then
    omitted from ``status`` rather than reported as zero — "absent, the
    field means the runner imposes no limit of its own", and a declared
    zero would read as a runner that refuses everything.
    """

    max_run_seconds: int = 0
    max_concurrent_runs: int = 1

    def as_dict(self) -> dict[str, Any]:
        block: dict[str, Any] = {"max_concurrent_runs": self.max_concurrent_runs}
        if self.max_run_seconds > 0:
            block["max_run_seconds"] = self.max_run_seconds
        return block


@dataclass
class _Slots:
    """How many runs may overlap, and how many do (SPEC 4.5)."""

    limit: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    in_flight: int = 0

    def acquire(self) -> bool:
        with self.lock:
            if self.in_flight >= self.limit:
                return False
            self.in_flight += 1
            return True

    def release(self) -> None:
        with self.lock:
            self.in_flight = max(0, self.in_flight - 1)

    def busy(self) -> bool:
        with self.lock:
            return self.in_flight > 0


class Engine:
    """Runs the bundle, one subprocess at a time."""

    def __init__(self, bundle_root: Path, *, limits: Limits | None = None) -> None:
        self.bundle_root = bundle_root
        self.limits = limits or Limits()
        self._slots = _Slots(limit=max(1, self.limits.max_concurrent_runs))
        self._workers: dict[str, subprocess.Popen[str]] = {}
        self._workers_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._slots.busy()

    def new_run_id(self) -> str:
        """Unique within this runner's lifetime (SPEC 4.2)."""
        return uuid.uuid4().hex[:16]

    def cancel(self, run_id: str) -> None:
        """Stop run ``run_id`` now, if its worker is running: its client left (SPEC 4.5).

        Callable from any thread, which is the point -- the thread feeding the
        run is blocked on the worker's output and cannot stop itself. The
        stream then ends as a stopped worker's does, and the caller, which
        knows why, decides what that means.
        """
        with self._workers_lock:
            process = self._workers.get(run_id)
        if process is not None:
            _terminate(process)

    def stop_all(self) -> None:
        """Stop every run in flight: the runner itself is stopping.

        A worker leads its own process group, so the Ctrl+C that stops this
        process no longer reaches it too, and it would run on unwatched.
        """
        with self._workers_lock:
            processes = list(self._workers.values())
        for process in processes:
            _terminate(process)

    def run(
        self, *, prompt: str, variables: dict[str, Any], run_id: str, mcp: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute and return the SPEC 4.2 response body. Raises on failure."""
        body: dict[str, Any] | None = None
        for event in self.stream(prompt=prompt, variables=variables, run_id=run_id, mcp=mcp):
            if event.name == "done":
                body = event.payload
        if body is None:  # pragma: no cover - stream() raises or yields done
            raise agent_error("The agent produced no result.")
        return body

    def stream(
        self,
        *,
        prompt: str,
        variables: dict[str, Any],
        run_id: str,
        on_notice: Callable[[str], None] | None = None,
        mcp: dict[str, Any] | None = None,
    ) -> Generator[RunEvent, None, None]:
        """Yield ``start``, zero or more ``step``, then exactly one ``done``.

        ``mcp`` makes the run an MCP server's tool call (:mod:`.mcp`) rather
        than the bundle's: the same worker, bounds and cancellation.

        A failure raises :class:`PosternError` instead of yielding an
        ``error`` event: the two verbs answer it differently — ``run`` with
        a status code, ``stream`` with a final event once its ``200`` is
        already committed — and that is the caller's decision to make, not
        this generator's.

        A consumer that stops consuming stops the run: the generator is
        closed, the ``finally`` kills the worker, and the agent goes with it.
        A consumer blocked waiting for the next frame cannot stop, so a
        client that leaves meanwhile is noticed elsewhere and the run is
        stopped through :meth:`cancel`.
        """
        if not self._slots.acquire():
            raise unavailable("This runner is already running an agent. Try again when it finishes.")
        process: subprocess.Popen[str] | None = None
        try:
            # Inside the ``try`` that gives the slot back: a spawn that
            # fails -- ``fork`` out of memory, or a bundle whose requirements
            # are not installed yet -- otherwise holds it for good, and every
            # later run is refused as overlapping one that never started.
            started = time.monotonic()
            deadline = started + self.limits.max_run_seconds if self.limits.max_run_seconds > 0 else None
            process = self._spawn(prompt=prompt, variables=variables, mcp=mcp)
            with self._workers_lock:
                self._workers[run_id] = process
            yield RunEvent("start", {"run_id": run_id})
            steps: list[dict[str, Any]] = []
            failure: PosternError | None = None
            done: dict[str, Any] | None = None
            expired = threading.Event()

            for frame in self._frames(process, deadline=deadline, expired=expired):
                kind = str(frame.get("event") or "")
                if kind == "step":
                    step = {key: value for key, value in frame.items() if key != "event"}
                    steps.append(step)
                    yield RunEvent("step", {**step, "status": "finished"})
                elif kind == "notice":
                    if on_notice is not None:
                        on_notice(str(frame.get("text") or ""))
                elif kind == "error":
                    failure = _failure_from(frame)
                elif kind == "done":
                    done = frame

            if done is None and expired.is_set():
                # The watchdog's verdict, not the clock's: read after the loop,
                # the clock called a worker that finished a moment before its
                # deadline a timeout and threw its answer away.
                raise run_timeout(self.limits.max_run_seconds)
            if failure is not None:
                raise failure
            if done is None:
                raise agent_error(
                    "The agent stopped without producing a result. `python doctor.py` in the bundle "
                    "folder reports what went wrong on the last run."
                )
            yield RunEvent("done", _run_body(run_id, done, steps))
        finally:
            with self._workers_lock:
                self._workers.pop(run_id, None)
            if process is not None:
                _terminate(process)
            self._slots.release()

    def inspect(self, mcp: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """One worker for an answer rather than a run: an MCP server's tools, at boot."""
        process = self._spawn(prompt="", variables={}, mcp=mcp)
        expired = threading.Event()
        answer: dict[str, Any] | None = None
        failure: PosternError | None = None
        try:
            for frame in self._frames(process, deadline=time.monotonic() + timeout, expired=expired):
                if frame.get("event") == "error":
                    failure = _failure_from(frame)
                elif frame.get("event") == "done":
                    answer = frame
        finally:
            _terminate(process)
        if answer is not None:
            return answer
        if expired.is_set():
            raise unavailable(f"The MCP server did not list its tools within {timeout:g} seconds.")
        raise failure or agent_error("The MCP server stopped without listing its tools.")

    # --- Internals ---------------------------------------------------

    def _spawn(
        self, *, prompt: str, variables: dict[str, Any], mcp: dict[str, Any] | None = None
    ) -> subprocess.Popen[str]:
        payload: dict[str, Any] = {"bundle_root": str(self.bundle_root), "prompt": prompt, "variables": variables}
        if mcp is not None:
            payload["mcp"] = mcp
        request = json.dumps(payload)
        # A run is a seller's code, so it is handed the buyer's environment
        # less the variables that configure *this runner* — the buyer's
        # account-wide ``SIGRIX_TOKEN`` above all. Free: nothing
        # downstream reads them. The rule is ``runner_env``'s; a bundle run
        # reaches it through ``execution``, which enforces the other half, over
        # the ``.env`` this cannot reach. Imported here rather than at module
        # level: ``execution`` pulls in PyYAML, and a run already
        # needs it to parse the bundle's own config, so paying for it here costs this verb
        # nothing it was not already going to spend — where paying for it
        # merely to *start the server* broke ``describe``/``status`` for a
        # bundle whose virtualenv is not built yet. Such a bundle meets this
        # import on its first run, and is told what the worker would tell it.
        if mcp is not None:
            # An MCP run needs no framework, so it takes the rule without PyYAML.
            from sigrix_runtime.runner_env import without_runner_settings
        else:
            try:
                from sigrix_runtime.execution import without_runner_settings
            except ModuleNotFoundError as exc:
                raise requirements_not_installed(str(exc.name)) from exc

        environment = without_runner_settings(os.environ)
        if mcp is not None:
            # No bundle holds this runtime for the worker to import from its
            # cwd, so it is told where this one came from.
            here = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, (here, environment.get("PYTHONPATH"))))
        # The worker's stdout is the protocol frame stream; anything a
        # library prints to it corrupts one. Unbuffered so a step event
        # reaches a streaming client when it happens rather than when the
        # pipe fills.
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, this interpreter
                [sys.executable, "-m", "sigrix_runtime.postern._worker"],
                cwd=str(self.bundle_root),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                # Its own process group, so a kill reaches what it starts (see
                # ``_terminate``); ignored where there are none.
                start_new_session=True,
            )
        except OSError as exc:
            raise unavailable(f"This runner could not start the agent: {exc}") from exc
        assert process.stdin is not None
        try:
            process.stdin.write(request)
            process.stdin.close()
        except OSError as exc:  # pragma: no cover - the child died immediately
            _terminate(process)
            raise unavailable(f"This runner could not start the agent: {exc}") from exc
        return process

    def _frames(
        self, process: subprocess.Popen[str], *, deadline: float | None, expired: threading.Event
    ) -> Iterator[dict[str, Any]]:
        """Relay the worker's frames, and stop it if it outstays its bound.

        The kill is on a timer rather than checked between frames, and that
        is the whole reason the bound is declarable. An agent inside a model
        call writes nothing for as long as it takes, so a deadline evaluated
        only when a line arrives is not enforced during exactly the silence
        it exists to end — the limit would then be "until the next step
        finishes", which SPEC 4.5 forbids declaring as a number. The timer
        sets ``expired`` as it kills; the caller reads the verdict there.
        """
        assert process.stdout is not None
        watchdog: threading.Timer | None = None
        if deadline is not None:
            watchdog = threading.Timer(max(0.0, deadline - time.monotonic()), _expire, args=(process, expired))
            watchdog.daemon = True
            watchdog.start()
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except ValueError:
                    continue  # not ours; a library wrote to stdout
                if isinstance(frame, dict):
                    yield frame
        finally:
            if watchdog is not None:
                watchdog.cancel()


def _expire(process: subprocess.Popen[str], expired: threading.Event) -> None:
    expired.set()
    _terminate(process)


def _failure_from(frame: dict[str, Any]) -> PosternError:
    message = str(frame.get("message") or "The agent failed.")
    trace = str(frame.get("traceback") or "").strip()
    if trace:
        # The worker's stderr goes nowhere, so this is the one place an agent
        # failure leaves its stack; the client is shown the message.
        logger.warning("postern: the agent failed:\n%s", trace)
    code = str(frame.get("code") or "")
    if code == "unavailable":
        return unavailable(message)
    if code == "bad_request":  # an MCP server refusing a call's arguments
        return bad_request(message)
    return agent_error(message)


def _run_body(run_id: str, done: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    """SPEC 4.2's response body. No ``status`` member — there is no such field."""
    body: dict[str, Any] = {
        "postern": POSTERN_VERSION,
        "run_id": run_id,
        "output": {"type": "text", "value": str(done.get("text") or "")},
    }
    usage = _usage(done, steps)
    if usage:
        body["usage"] = usage
    files = done.get("files_written")
    if isinstance(files, list) and files:
        # Not a protocol member. The bundle's tools write into
        # ``workspace/`` (SPEC 1.2's "the honest route" for anything larger
        # than an envelope carries), and a client that never learns which
        # files appeared cannot point a buyer at their own deliverable.
        # Rides under the reverse-domain namespace SPEC 6 sanctions, so a
        # client that does not know it ignores it.
        body["org.sigrix"] = {"files_written": [str(name) for name in files]}
    return body


def _usage(done: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = int(done.get("input_tokens") or 0)
    output_tokens = int(done.get("output_tokens") or 0)
    usage: dict[str, Any] = {}
    if input_tokens or output_tokens:
        usage["input_tokens"] = input_tokens
        usage["output_tokens"] = output_tokens
        cost = estimate_cost(str(done.get("model_id") or ""), input_tokens, output_tokens)
        if cost is not None:
            usage["cost_usd"] = cost
    if steps:
        usage["steps"] = steps
    return usage


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD for a run, or ``None`` for a model this bundle has no price for."""
    pricing = MODEL_PRICING.get(model_id)
    if pricing is None:
        return None
    per_million_in, per_million_out = pricing
    return round((input_tokens * per_million_in + output_tokens * per_million_out) / 1_000_000.0, 6)


def _terminate(process: subprocess.Popen[str]) -> None:
    """Kill the worker and everything it started.

    The worker leads its own process group, so the group goes: killing the
    worker alone left a tool's subprocess holding its stdout, and the relay
    blocked past the declared bound. Asked of the group even once the
    worker has exited, since what it started can outlive it.
    """
    try:
        if os.getpgid(process.pid) == process.pid:  # a group it leads, never this process's own
            os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, OSError):
        pass  # no process groups on this platform, or the group is gone
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=_REAP_SECONDS)
    except subprocess.TimeoutExpired:  # pragma: no cover - the OS has it now
        pass


__all__ = [
    "MODEL_PRICING",
    "Engine",
    "Limits",
    "RunEvent",
    "estimate_cost",
]
