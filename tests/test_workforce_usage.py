"""What a workforce run reports: a step for each task, and the run's totals.

A workforce's Flow answers with text, and its unit crews and its own crews (the
router, the synthesizer, the manager answering itself) each count their own
tokens. Until each crew was handed to the run as it was built, a workforce
reported no step and no usage at all, so a client pricing runs priced every
workforce run at nothing.

Two halves. The first runs a stub CrewAI, with just enough of a Flow to drive
the three orchestration modes, through the real server and worker, so the
shape is pinned in every job. The second runs CrewAI's own Flows on its own
Anthropic provider against a loopback stand-in for Anthropic's API, which is
what proves the totals are every crew's and nobody's twice; it needs CrewAI's
``anthropic`` extra, which the floor job installs.
"""

from __future__ import annotations

import json
import shutil
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest
import yaml

from sigrix_runtime import execution, workforce
from sigrix_runtime.postern import PATH_PREFIX
from sigrix_runtime.postern.engine import estimate_cost
from tests import postern_schema
from tests.claude_stand_in import (
    INPUT_PER_CALL,
    MODEL_ID,
    OUTPUT_PER_CALL,
    READ_PER_CALL,
    WRITTEN_PER_CALL,
    claude_on_loopback,
)
from tests.support import BUNDLE_VERSION, CREWS, RUNTIME_ROOT
from tests.test_server import _DOTENV_STUB, _events, _serve

# Every task of every crew spends this, as (input, output, cache written,
# cache read), the cache counts inside the input as CrewAI counts them.
_STUB_TASK_USAGE = (1600, 300, 1000, 500)

_CREWAI_STUB = """
import os as _os

_TASK_USAGE = (1600, 300, 1000, 500)
_ROUTE_QUESTION = "Reply with exactly one unit key from:"


class Process:
    sequential = "sequential"
    hierarchical = "hierarchical"


class LLM:
    def __init__(self, model=None):
        self.model = model or _os.environ.get("STUB_AGENT_MODEL", "claude-sonnet-5-5")


class Agent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.llm = LLM()


class Task:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.name = kwargs.get("name")


class _Usage:
    def __init__(self, totals):
        self.prompt_tokens, self.completion_tokens, self.cache_creation_tokens, self.cached_prompt_tokens = totals


class _TaskOutput:
    def __init__(self, task):
        self.name = getattr(task, "name", None)
        self.description = getattr(task, "description", "") or ""


class _Output:
    def __init__(self, raw, usage):
        self.raw = raw
        self.token_usage = usage

    def __str__(self):
        return self.raw


_TASK_SECONDS = float(_os.environ.get("STUB_TASK_SECONDS") or 0)
_GATE = _os.environ.get("STUB_STEP_GATE")
_TASKS_STARTED = []


class Crew:
    task_callback = None

    def __init__(self, **kwargs):
        self.tasks = kwargs.get("tasks") or []
        self.agents = kwargs.get("agents") or []
        self.manager_agent = kwargs.get("manager_agent")
        self.before_kickoff_callbacks = []
        self._totals = [0, 0, 0, 0]

    def calculate_usage_metrics(self):
        return _Usage(tuple(self._totals))

    def kickoff(self, inputs=None):
        import pathlib as _pathlib
        import time as _time

        for callback in self.before_kickoff_callbacks:
            inputs = callback(inputs or {})
        for task in self.tasks:
            # With STUB_STEP_GATE naming a folder, the run's second task waits
            # until the test has read the first one's step.
            if _GATE and len(_TASKS_STARTED) == 1:
                go = _pathlib.Path(_GATE, "go")
                for _ in range(2000):
                    if go.exists():
                        break
                    _time.sleep(0.01)
                else:
                    raise RuntimeError("nobody read the first step")
            _TASKS_STARTED.append(task)
            if _GATE:
                _pathlib.Path(_GATE, f"started-{len(_TASKS_STARTED) - 1}").write_text("")
            if _TASK_SECONDS:
                _time.sleep(_TASK_SECONDS)
            self._totals = [total + spent for total, spent in zip(self._totals, _TASK_USAGE)]
            if self.task_callback:
                self.task_callback(_TaskOutput(task))
        asked = " ".join(str(getattr(task, "description", "")) for task in self.tasks)
        answer = _os.environ.get("STUB_ROUTE_TO", "none") if _ROUTE_QUESTION in asked else "done"
        return _Output(answer, self.calculate_usage_metrics())

    async def kickoff_async(self, inputs=None):
        # With STUB_THREADED_PANEL set, a panel's units run at once, as
        # CrewAI's own kickoff_async runs each on a thread.
        if _os.environ.get("STUB_THREADED_PANEL"):
            import asyncio as _asyncio

            return await _asyncio.to_thread(self.kickoff, inputs)
        return self.kickoff(inputs)
"""

# Just enough of ``crewai.flow.flow`` for the workforce module: a start method,
# then each method listening to the one before, a coroutine run to its end.
_FLOW_STUB = """
import asyncio
import inspect


def start():
    def mark(method):
        method._flow_start = True
        return method

    return mark


def listen(source):
    def mark(method):
        method._flow_after = source.__name__
        return method

    return mark


class Flow:
    def kickoff(self):
        methods = [getattr(type(self), name) for name in dir(type(self))]
        methods = [method for method in methods if callable(method)]
        method = next(method for method in methods if getattr(method, "_flow_start", False))
        value = self._run(method)
        while True:
            following = [m for m in methods if getattr(m, "_flow_after", None) == method.__name__]
            if not following:
                return value
            method = following[0]
            value = self._run(method, value)

    def _run(self, method, *args):
        value = method(self, *args)
        return asyncio.run(value) if inspect.iscoroutine(value) else value
"""

_UNITS = (
    ("review", "Contract review", CREWS / "contract-review" / "config"),
    ("build", "Software team", CREWS / "software-team" / "config"),
)
_REVIEW_TASKS = ["review.summarise_contract", "review.identify_risks", "review.draft_negotiation_play"]
_BUILD_TASKS = ["build.answer_request"]


def _workforce_bundle(root: Path, process: str, *, stub: bool) -> Path:
    """A workforce bundle laid out as the platform compiles one."""
    shutil.copytree(RUNTIME_ROOT, root)
    shutil.copytree(_UNITS[0][2], root / "config", dirs_exist_ok=True)  # the flattened fallback
    manifest: dict[str, Any] = {
        "name": "Review and build",
        "process": process,
        "units": [{"key": key, "name": name} for key, name, _ in _UNITS],
    }
    if process == "parallel":
        manifest["synthesizer"] = "Chief of Staff"
    (root / "config" / "workforce.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    for key, _name, source in _UNITS:
        shutil.copytree(source, root / "config" / "units" / key)
    (root / "VERSION").write_text(BUNDLE_VERSION, encoding="utf-8")
    if stub:
        (root / "crewai" / "flow").mkdir(parents=True)
        (root / "crewai" / "__init__.py").write_text(_CREWAI_STUB, encoding="utf-8")
        (root / "crewai" / "flow" / "__init__.py").write_text("", encoding="utf-8")
        (root / "crewai" / "flow" / "flow.py").write_text(_FLOW_STUB, encoding="utf-8")
        (root / "dotenv.py").write_text(_DOTENV_STUB, encoding="utf-8")
    return root


def _run(root: Path, *, timeout: float = 30) -> dict[str, Any]:
    with _serve(root) as client:
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "review this"}}, timeout=timeout)
    assert status == 200, body
    postern_schema.validate(body, postern_schema.load("run-response"), name="run")
    return body


def execution_cost(model_id: str, usage: dict[str, Any]) -> float | None:
    return estimate_cost(model_id, usage["input_tokens"], usage["output_tokens"])


def _split(step_or_usage: dict[str, Any]) -> tuple[int, int]:
    split = step_or_usage["org.sigrix"]
    return split["cache_write_tokens"], split["cache_read_tokens"]


def _assert_the_steps_add_up_to_the_run(usage: dict[str, Any]) -> None:
    steps = usage["steps"]
    assert sum(step["input_tokens"] for step in steps) == usage["input_tokens"]
    assert sum(step["output_tokens"] for step in steps) == usage["output_tokens"]
    written = sum(_split(step)[0] for step in steps)
    read = sum(_split(step)[1] for step in steps)
    assert (written, read) == _split(usage)


# ---------------------------------------------------------------------------
# The shape, on a stub CrewAI: every job runs these
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("process", "route_to", "expected"),
    [
        ("sequential", None, _REVIEW_TASKS + _BUILD_TASKS),
        ("parallel", None, _REVIEW_TASKS + _BUILD_TASKS + ["workforce-synthesizer.synthesize"]),
        ("hierarchical", "review", ["workforce-manager.route", *_REVIEW_TASKS]),
        ("hierarchical", "none", ["workforce-manager.route", "workforce-manager.answer"]),
    ],
)
def test_a_workforce_reports_a_step_for_each_task_and_its_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process: str, route_to: str | None, expected: list[str]
) -> None:
    """Each unit's tasks under the unit's key, the workforce's own under ``workforce``.

    Every crew's tasks spend the same in the stub, so each step's share is one
    task's and the run's totals are that times the steps: nothing a crew spent
    is missed, and nothing is counted twice.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    if route_to is not None:
        monkeypatch.setenv("STUB_ROUTE_TO", route_to)
    usage = _run(_workforce_bundle(tmp_path / "bundle", process, stub=True))["usage"]

    steps = usage["steps"]
    assert [step["name"] for step in steps] == expected
    spent, out, written, read = _STUB_TASK_USAGE
    for step in steps:
        assert (step["input_tokens"], step["output_tokens"], *_split(step)) == _STUB_TASK_USAGE
        assert step["model_id"] == MODEL_ID
    assert (usage["input_tokens"], usage["output_tokens"]) == (spent * len(steps), out * len(steps))
    assert _split(usage) == (written * len(steps), read * len(steps))
    _assert_the_steps_add_up_to_the_run(usage)


def test_a_unit_s_steps_are_named_as_its_describe_lists_its_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One name for a task, in what the runner says it has and what it says ran."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    root = _workforce_bundle(tmp_path / "bundle", "sequential", stub=True)
    declared = set(execution.load_run_config(root).task_names)
    usage = _run(root)["usage"]
    assert {step["name"] for step in usage["steps"]} == declared


def test_a_workforce_is_priced_on_the_model_its_crews_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """As a single crew is: the model CrewAI resolved, not a setting nobody made.

    No ``OPENAI_MODEL_NAME`` here, so only the crews can name the model, and
    ``cost_usd`` is there only if the run was told which one it was.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
    monkeypatch.setenv("STUB_AGENT_MODEL", "gpt-4o-mini")
    usage = _run(_workforce_bundle(tmp_path / "bundle", "sequential", stub=True))["usage"]
    assert usage["cost_usd"] == pytest.approx(execution_cost("gpt-4o-mini", usage))


def test_the_workforce_s_own_steps_can_never_share_a_unit_s_name() -> None:
    """A unit key is identifier-safe, so a prefix with a hyphen is no unit's.

    Under a plain ``workforce`` prefix, a unit keyed ``workforce`` with a task
    named ``route`` emitted a step the router's own was indistinguishable from.
    """
    for prefix in (workforce.MANAGER_STEP_PREFIX, workforce.SYNTHESIZER_STEP_PREFIX):
        with pytest.raises(workforce.WorkforceConfigError):
            workforce.parse_manifest({"process": "sequential", "units": [{"key": prefix}]})
    assert workforce.parse_manifest({"process": "sequential", "units": [{"key": "workforce"}]}).units


def test_a_panel_s_steps_each_time_their_own_crew(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SPEC 4.3's ``latency_ms`` is the step's own elapsed time.

    Two units run at once and each task takes 0.3 s. Timed on one clock for the
    run, the unit that finished just after the other reported almost nothing.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("STUB_THREADED_PANEL", "1")
    monkeypatch.setenv("STUB_TASK_SECONDS", "0.3")
    usage = _run(_workforce_bundle(tmp_path / "bundle", "parallel", stub=True))["usage"]
    unit_steps = [step for step in usage["steps"] if not step["name"].startswith("workforce-")]
    assert [step["name"] for step in unit_steps if step["name"].startswith("build.")] == _BUILD_TASKS
    assert all(step["latency_ms"] >= 250 for step in unit_steps), [
        (step["name"], step["latency_ms"]) for step in unit_steps
    ]


def test_a_workforce_s_step_is_streamed_before_its_next_task_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a client pricing each step needs: the step before the next task spends anything."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    gate = tmp_path / "gate"
    gate.mkdir()
    monkeypatch.setenv("STUB_STEP_GATE", str(gate))
    root = _workforce_bundle(tmp_path / "bundle", "sequential", stub=True)
    body = json.dumps({"inputs": {"prompt": "go"}}).encode("utf-8")
    names: list[str] = []
    with _serve(root) as client:
        connection = HTTPConnection("127.0.0.1", client.port, timeout=15)
        try:
            connection.request(
                "POST",
                PATH_PREFIX + "/stream",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            response = connection.getresponse()
            for line in iter(response.readline, b""):
                if line.startswith(b"event: "):
                    names.append(line[len(b"event: ") :].strip().decode())
                    if names.count("step") == 1 and names[-1] == "step":
                        assert not (gate / "started-1").exists(), "the next task started before its step was read"
                        (gate / "go").write_text("", encoding="utf-8")
        finally:
            connection.close()
    assert names == ["start"] + ["step"] * len(_REVIEW_TASKS + _BUILD_TASKS) + ["done"]


def test_a_workforce_streams_each_step_as_it_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """What a client pricing the run as it goes reads: the steps, before ``done``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    root = _workforce_bundle(tmp_path / "bundle", "parallel", stub=True)
    body = json.dumps({"inputs": {"prompt": "go"}}).encode("utf-8")
    with _serve(root) as client:
        _, _, raw = client.request(
            "POST",
            "/stream",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
    events = _events(raw)
    names = [name for name, _ in events]
    assert names[0] == "start" and names[-1] == "done"
    steps = [payload for name, payload in events if name == "step"]
    schema = postern_schema.load("stream-event")
    for step in steps:
        postern_schema.validate(step, schema, name="step")
    assert [step["name"] for step in steps] == _REVIEW_TASKS + _BUILD_TASKS + ["workforce-synthesizer.synthesize"]
    done = dict(events)["done"]
    assert done["usage"]["input_tokens"] == _STUB_TASK_USAGE[0] * len(steps)


# ---------------------------------------------------------------------------
# The totals, on CrewAI's own Flows and Anthropic provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("process", "route_to", "units"),
    [
        ("sequential", "none", {"review", "build"}),
        ("parallel", "none", {"review", "build", "workforce-synthesizer"}),
        ("hierarchical", "review", {"workforce-manager", "review"}),
    ],
)
def test_a_real_workforce_on_claude_counts_every_call_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process: str, route_to: str, units: set[str]
) -> None:
    """Every message the stand-in answered is in the run's totals, once.

    The stand-in counts what was asked of it, so a crew the run did not hear
    about, or one counted twice, moves the totals off the count.
    """
    root = _workforce_bundle(tmp_path / "bundle", process, stub=False)
    with claude_on_loopback(monkeypatch, tmp_path / "home", route_to=route_to) as claude:
        usage = _run(root, timeout=240)["usage"]
        calls = claude.messages

    assert calls >= len(usage["steps"]) > 0
    assert (usage["input_tokens"], usage["output_tokens"]) == (INPUT_PER_CALL * calls, OUTPUT_PER_CALL * calls)
    assert _split(usage) == (WRITTEN_PER_CALL * calls, READ_PER_CALL * calls)
    assert {step["name"].split(".")[0] for step in usage["steps"]} == units
    assert all(step["model_id"] == MODEL_ID for step in usage["steps"])
    _assert_the_steps_add_up_to_the_run(usage)
