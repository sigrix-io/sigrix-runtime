"""What a run has spent so far, reported after every model call it makes.

A step reports its tokens only when its task finishes, so a run stopped inside a
task (its time limit, a cancel, a crash) said nothing about what that task had
spent, and a client pricing runs priced it at nothing. Now:

1. ``execution.execute`` hands ``on_usage`` the run's running totals after each
   model call crewai reports on its event bus, summed over every crew the run
   has built, and only when they moved, with how many calls they count.
2. The worker writes each as a ``usage`` frame, and the engine relays it as an
   ``org.sigrix.usage`` stream event in the shape of a run's ``usage``. SPEC 4.3
   has a client skip an event name it does not know, so nothing else changes.
3. The error a run ends with, a timeout's, a failure's, carries the last of them
   under ``detail``, in the ``org.sigrix`` namespace, beside what ``detail``
   already says.

The listener and the engine are driven with stand-ins first; then a real crew on
crewai's own Anthropic provider, against a stand-in for Anthropic's API, which is
what fails if crewai stops reporting a finished call on its bus.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

from sigrix_runtime import execution
from sigrix_runtime.postern import engine as engine_module
from sigrix_runtime.postern.engine import SIGRIX_NAMESPACE, USAGE_EVENT, Engine, Limits
from sigrix_runtime.postern.errors import PosternError
from tests import postern_schema
from tests.claude_stand_in import (
    INPUT_PER_CALL,
    MODEL_ID,
    OUTPUT_PER_CALL,
    READ_PER_CALL,
    WRITTEN_PER_CALL,
    claude_on_loopback,
)
from tests.support import BUNDLE_VERSION, CREW_CONFIG, RUNTIME_ROOT
from tests.test_server import _events, _serve

# --- 1. the listener ------------------------------------------------------------


class _Totals:
    def __init__(
        self,
        prompt: int,
        completion: int,
        *,
        written: int | None = None,
        read: int | None = None,
        requests: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        if written is not None:
            self.cache_creation_tokens = written
        if read is not None:
            self.cached_prompt_tokens = read
        if requests is not None:
            self.successful_requests = requests


class _LLM:
    def __init__(self, model: str) -> None:
        self.model = model


class _Agent:
    def __init__(self, model: str) -> None:
        self.llm = _LLM(model)


class _Crew:
    """A crew whose running totals are whatever the test last set, as crewai's are mid-run."""

    def __init__(self, model: str = MODEL_ID) -> None:
        self.agents = [_Agent(model)]
        self.totals: Any = None
        self.fails = False

    def calculate_usage_metrics(self) -> Any:
        if self.fails:
            raise RuntimeError("the counters could not be read")
        return self.totals


def _listener() -> tuple[execution._UsageListener, list[execution.RunUsage]]:
    readings: list[execution.RunUsage] = []
    return execution._UsageListener(readings.append, model_id="from-the-environment"), readings


def test_each_reading_is_the_runs_running_totals_and_one_that_did_not_move_is_not_handed_on() -> None:
    listener, readings = _listener()
    crew = _Crew()
    listener.watch(crew)

    crew.totals = _Totals(1600, 300, written=1000, read=500)
    listener.report()
    listener.report()
    crew.totals = _Totals(3200, 600, written=2000, read=1000)
    listener.report()

    assert readings == [
        execution.RunUsage(1600, 300, cache_write_tokens=1000, cache_read_tokens=500, model_id=MODEL_ID),
        execution.RunUsage(3200, 600, cache_write_tokens=2000, cache_read_tokens=1000, model_id=MODEL_ID),
    ]


def test_a_workforce_reading_sums_every_crew_and_one_crew_that_cannot_say_leaves_none() -> None:
    """All or nothing, as the run's own totals are: a sum without one crew is a smaller number, not a known one."""
    listener, readings = _listener()
    unit, manager = _Crew(), _Crew()
    listener.watch(unit)
    listener.watch(manager)
    unit.totals, manager.totals = _Totals(1600, 300), _Totals(800, 100)

    listener.report()
    manager.fails = True
    unit.totals = _Totals(3200, 600)
    listener.report()

    assert [(reading.input_tokens, reading.output_tokens) for reading in readings] == [(2400, 400)]
    assert (readings[0].cache_write_tokens, readings[0].cache_read_tokens) == (None, None)


def test_each_reading_counts_the_calls_crewai_counted_with_its_tokens() -> None:
    """crewai's ``successful_requests``, read with the tokens. Two calls crewai counted before the first one's reading
    was read come as one reading counting two more: a client pricing one call by its prompt can tell it apart."""
    listener, readings = _listener()
    crew = _Crew()
    listener.watch(crew)

    crew.totals = _Totals(1600, 300, requests=1)
    listener.report()
    crew.totals = _Totals(4800, 900, requests=3)
    listener.report()
    crew.totals = _Totals(6400, 1200)
    listener.report()

    assert [reading.calls for reading in readings] == [1, 3, None], "a count crewai did not give was made up"
    assert [reading.input_tokens for reading in readings] == [1600, 4800, 6400]


def test_a_workforce_reading_counts_every_crews_calls_and_one_crew_that_cannot_count_leaves_none() -> None:
    listener, readings = _listener()
    unit, manager = _Crew(), _Crew()
    listener.watch(unit)
    listener.watch(manager)
    unit.totals, manager.totals = _Totals(1600, 300, requests=2), _Totals(800, 100, requests=1)
    listener.report()
    manager.totals = _Totals(1600, 200)
    listener.report()

    assert [(reading.input_tokens, reading.calls) for reading in readings] == [(2400, 3), (3200, None)]


def test_nothing_is_handed_on_before_a_crew_is_watched_or_after_the_run_has_returned() -> None:
    listener, readings = _listener()
    listener.report()
    crew = _Crew()
    listener.watch(crew)
    listener.close()
    crew.totals = _Totals(1600, 300)
    listener.report()

    assert readings == []


def test_a_reading_names_the_model_a_crew_resolved_else_the_one_it_was_given() -> None:
    listener, readings = _listener()
    crew = _Crew(model="")
    crew.totals = _Totals(10, 1)
    listener.watch(crew)
    listener.report()

    assert readings[0].model_id == "from-the-environment"


def test_a_handler_that_raises_does_not_reach_the_run() -> None:
    def _broken(_reading: execution.RunUsage) -> None:
        raise BrokenPipeError("the worker's stdout is gone")

    listener = execution._UsageListener(_broken, model_id="")
    crew = _Crew()
    crew.totals = _Totals(10, 1)
    listener.watch(crew)
    listener.report()


def test_the_frame_keeps_the_cache_split_apart_and_says_nothing_it_was_not_told() -> None:
    told = execution.RunUsage(1600, 300, cache_write_tokens=1000, cache_read_tokens=500, model_id=MODEL_ID, calls=2)
    untold = execution.RunUsage(1600, 300)

    assert told.as_frame() == {
        "input_tokens": 1600,
        "output_tokens": 300,
        "cache": {"cache_write_tokens": 1000, "cache_read_tokens": 500},
        "model_id": MODEL_ID,
        "calls": 2,
    }
    assert untold.as_frame() == {"input_tokens": 1600, "output_tokens": 300, "cache": {}}


class _Bus:
    """crewai's event bus as the listener uses it: ``on``, ``off`` and ``flush``, and what each was handed.

    crewai calls a handler on a pool of its own, after the event; ``queue`` is an event it holds and has yet to
    hand on, and ``flush`` hands every one on, as crewai's waits for its pool.
    """

    def __init__(self) -> None:
        self.handlers: list[Any] = []
        self.taken_off: list[Any] = []
        self.pending: list[Any] = []

    def queue(self, handler: Any, event: Any) -> None:
        self.pending.append(lambda: handler(object(), event))

    def flush(self, timeout: float | None = None) -> bool:
        while self.pending:
            self.pending.pop(0)()
        return True

    def on(self, event_type: Any) -> Any:
        def register(handler: Any) -> Any:
            self.handlers.append((event_type, handler))
            return handler

        return register

    def off(self, event_type: Any, handler: Any) -> None:
        self.taken_off.append((event_type, handler))


def test_the_listener_is_on_crewais_bus_for_the_run_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _Bus()

    class LLMCallCompletedEvent:  # noqa: N801 - crewai's own name
        pass

    events = types.ModuleType("crewai.events")
    events.crewai_event_bus = bus  # type: ignore[attr-defined]
    events.LLMCallCompletedEvent = LLMCallCompletedEvent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "crewai", types.ModuleType("crewai"))
    monkeypatch.setitem(sys.modules, "crewai.events", events)
    listener, readings = _listener()
    crew = _Crew()
    crew.totals = _Totals(1600, 300)
    listener.watch(crew)

    with execution._reporting_usage(listener):
        [(event_type, handler)] = bus.handlers
        assert event_type is LLMCallCompletedEvent
        # The run's last call, still in crewai's pool as the run returns (a failure, say).
        bus.queue(handler, LLMCallCompletedEvent())
    crew.totals = _Totals(3200, 600)
    handler(object(), LLMCallCompletedEvent())

    assert bus.taken_off == bus.handlers, "the handler stayed on the bus after the run"
    assert [(reading.input_tokens, reading.output_tokens) for reading in readings] == [(1600, 300)], (
        "the last call was not flushed before the listener closed, or a call after the run was handed on"
    )


def test_a_crewai_with_no_event_bus_runs_with_its_steps_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "crewai", types.ModuleType("crewai"))
    monkeypatch.setitem(sys.modules, "crewai.events", None)
    listener, readings = _listener()
    ran = []

    with execution._reporting_usage(listener):
        ran.append(True)

    assert ran == [True]
    assert readings == []


# --- 2. the engine ----------------------------------------------------------------


def _scripted(engine: Engine, monkeypatch: pytest.MonkeyPatch, frames: list[Any]) -> None:
    """Spawn a worker that writes ``frames`` and stops, a number in the list being a pause, in seconds."""
    script = textwrap.dedent(
        f"""
        import json, sys, time
        for frame in json.loads({json.dumps(json.dumps(frames))}):
            if isinstance(frame, (int, float)):
                time.sleep(frame)
                continue
            sys.stdout.write(json.dumps(frame) + "\\n")
            sys.stdout.flush()
        """
    )

    def _spawn(**_kwargs: Any) -> subprocess.Popen[str]:
        return subprocess.Popen(  # noqa: S603 - this interpreter, a fixed script
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )

    monkeypatch.setattr(engine, "_spawn", _spawn)


_SPENT = {"event": "usage", "input_tokens": 1600, "output_tokens": 300, "cache": {"cache_write_tokens": 1000}}
_SPENT_MORE = {"event": "usage", "input_tokens": 3200, "output_tokens": 600, "model_id": MODEL_ID, "calls": 2}


def _streamed(engine: Engine) -> tuple[list[tuple[str, dict[str, Any]]], PosternError | None]:
    events: list[tuple[str, dict[str, Any]]] = []
    try:
        for event in engine.stream(prompt="go", variables={}, run_id="run-1"):
            events.append((event.name, event.payload))
    except PosternError as exc:
        return events, exc
    return events, None


def test_each_usage_frame_is_relayed_as_the_namespaced_event_in_the_shape_of_a_runs_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = Engine(tmp_path)
    nothing = {"event": "usage", "input_tokens": 0, "output_tokens": 0}
    _scripted(engine, monkeypatch, [_SPENT, nothing, _SPENT_MORE, {"event": "done", "text": "ok"}])

    events, failure = _streamed(engine)

    assert failure is None
    assert USAGE_EVENT == "org.sigrix.usage"
    assert engine_module.CALLS_MEMBER == "calls", "a client reads the count by this name"
    assert [name for name, _ in events] == ["start", USAGE_EVENT, USAGE_EVENT, "done"]
    assert events[1][1] == {"input_tokens": 1600, "output_tokens": 300, SIGRIX_NAMESPACE: {"cache_write_tokens": 1000}}
    assert events[2][1] == {
        "input_tokens": 3200,
        "output_tokens": 600,
        SIGRIX_NAMESPACE: {"calls": 2},
        "model_id": MODEL_ID,
    }


@pytest.mark.parametrize("calls", [0, -1, True, "2", 2.0, None])
def test_a_call_count_that_is_not_a_count_is_left_out(calls: Any) -> None:
    frame = {"input_tokens": 1600, "output_tokens": 300, "cache": {"cache_write_tokens": 1000}, "calls": calls}
    assert engine_module._running_usage(frame) == {
        "input_tokens": 1600,
        "output_tokens": 300,
        SIGRIX_NAMESPACE: {"cache_write_tokens": 1000},
    }


def test_a_runs_usage_counts_its_calls_where_its_readings_do() -> None:
    """A reading is a run's ``usage`` without its steps or its cost, so the run's own carries the count the same way,
    and a run whose last reading was its totals ends with one ``usage`` the reading agrees with."""
    done = {"input_tokens": 4800, "output_tokens": 900, "cache": {"cache_write_tokens": 3000}, "calls": 3}
    assert engine_module._usage(done, [])[SIGRIX_NAMESPACE] == {"cache_write_tokens": 3000, "calls": 3}
    assert SIGRIX_NAMESPACE not in engine_module._usage({"input_tokens": 4800, "output_tokens": 900}, [])


@pytest.mark.parametrize(
    ("frames", "limits", "code", "detail"),
    [
        pytest.param(
            [_SPENT, _SPENT_MORE, 10],
            Limits(max_run_seconds=1),
            "run_timeout",
            {"max_run_seconds": 1},
            id="cut-off-by-the-time-limit",
        ),
        pytest.param(
            [_SPENT, {"event": "error", "code": "agent_error", "message": "boom"}],
            Limits(),
            "agent_error",
            None,
            id="the-agent-failed",
        ),
        pytest.param([_SPENT], Limits(), "agent_error", None, id="stopped-without-a-result"),
    ],
)
def test_the_error_a_run_ends_with_carries_what_it_had_spent_beside_what_detail_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: list[Any], limits: Limits, code: str, detail: Any
) -> None:
    engine = Engine(tmp_path, limits=limits)
    _scripted(engine, monkeypatch, frames)

    events, failure = _streamed(engine)

    assert failure is not None and failure.code == code
    last = [payload for name, payload in events if name == USAGE_EVENT][-1]
    assert failure.detail == {**(detail or {}), SIGRIX_NAMESPACE: {"usage": last}}
    postern_schema.validate(failure.envelope(), postern_schema.load("error"), name=code)


def test_a_run_that_spent_nothing_it_reported_ends_with_detail_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = Engine(tmp_path, limits=Limits(max_run_seconds=1))
    _scripted(engine, monkeypatch, [10])

    _events_seen, failure = _streamed(engine)

    assert failure is not None and failure.code == "run_timeout"
    assert failure.detail == {"max_run_seconds": 1}


def test_a_detail_that_is_a_sentence_is_left_a_sentence() -> None:
    error = PosternError(500, "agent_error", "boom", "a sentence")
    assert engine_module._having_spent(error, {"input_tokens": 1, "output_tokens": 1}).detail == "a sentence"


# --- 3. a real crew on Claude ---------------------------------------------------------


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    shutil.copytree(RUNTIME_ROOT, root)
    shutil.copytree(CREW_CONFIG, root / "config", dirs_exist_ok=True)
    (root / "VERSION").write_text(BUNDLE_VERSION, encoding="utf-8")
    return root


def _stream(client: Any, *, timeout: float) -> list[tuple[str, dict[str, Any]]]:
    body = json.dumps({"inputs": {"prompt": "review this"}}).encode("utf-8")
    status, _, raw = client.request(
        "POST",
        "/stream",
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        timeout=timeout,
    )
    assert status == 200
    return _events(raw)


def _per_call(calls: int) -> dict[str, Any]:
    return {
        "input_tokens": INPUT_PER_CALL * calls,
        "output_tokens": OUTPUT_PER_CALL * calls,
        SIGRIX_NAMESPACE: {
            "cache_write_tokens": WRITTEN_PER_CALL * calls,
            "cache_read_tokens": READ_PER_CALL * calls,
            "calls": calls,
        },
        "model_id": MODEL_ID,
    }


def test_a_real_crew_on_claude_reports_what_it_has_spent_after_every_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)
    # A model takes time to answer; without it the stand-in answers the next call before crewai's pool has
    # handed on the last one's reading, and two calls read as one.
    with claude_on_loopback(monkeypatch, tmp_path / "home", answer_delay=0.2) as claude, _serve(root) as client:
        events = _stream(client, timeout=240)
        calls = claude.messages

    readings = [payload for name, payload in events if name == USAGE_EVENT]
    done = dict(events)["done"]
    assert calls >= 3
    assert readings == [_per_call(n) for n in range(1, calls + 1)], "a call went unreported, or one twice"
    final = {key: done["usage"][key] for key in ("input_tokens", "output_tokens", SIGRIX_NAMESPACE)}
    assert {key: readings[-1][key] for key in final} == final, "the last reading is not the run's totals"
    assert [name for name, _ in events][-1] == "done"
    schema = postern_schema.load("stream-event")
    for name, payload in events:
        if name in {"start", "step", "delta"}:
            postern_schema.validate(payload, schema, name=name)


@pytest.mark.parametrize("verb", ["stream", "run"])
def test_a_real_crew_cut_off_inside_a_task_ends_with_what_its_calls_had_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    """The stand-in answers the first message and holds the next open, a model call that never returns, until
    the runner's time limit stops the run. Each task here makes one call, so the run is cut off inside its second
    task: the end-to-end proof is the plumbing (crewai's bus, the worker's frame, the engine's error). A run cut off
    inside its first task, with no step to say anything, is the engine's case above."""
    root = _bundle(tmp_path)
    limits = Limits(max_run_seconds=15)
    with (
        claude_on_loopback(monkeypatch, tmp_path / "home", hang_after=1) as claude,
        _serve(root, limits=limits) as client,
    ):
        if verb == "stream":
            events = _stream(client, timeout=120)
            [(name, envelope)] = [(name, payload) for name, payload in events if name == "error"]
            assert [payload for name, payload in events if name == USAGE_EVENT] == [_per_call(1)]
        else:
            status, envelope = client.json("POST", "/run", {"inputs": {"prompt": "review this"}}, timeout=120)
            assert status == 504
        calls = claude.messages

    assert calls == 1
    postern_schema.validate(envelope, postern_schema.load("error"), name="run_timeout")
    assert envelope["error"]["code"] == "run_timeout"
    assert envelope["error"]["detail"] == {"max_run_seconds": 15, SIGRIX_NAMESPACE: {"usage": _per_call(1)}}
