"""Unit tests for the execution path's per-step usage attribution.

``_attach_task_callback``, ``_resolved_model_id`` and ``_crew_usage_snapshot``
are pure functions over duck-typed crewai objects, so they are exercised here
directly with lightweight fakes — no crewai install and no subprocess needed.
The end-to-end shape against a real HTTP run (a step's ``name``/``model_id``/
tokens on the wire, ``cost_usd`` present or absent) is
``tests/test_postern_runner_server.py``'s job; this file is the boundary
cases that are awkward to provoke through a whole run: a task name past the
120-character bound that must survive untouched (AC1 says "untruncated"), a
crew whose usage cannot be read mid-run, and the model-resolution preference
order the docstrings describe but a full run only ever exercises once.
"""

from __future__ import annotations

from typing import Any

from sigrix_runtime.execution import _attach_task_callback, _crew_usage_snapshot, _resolved_model_id  # noqa: E402


class _LLM:
    def __init__(self, model: str = "") -> None:
        self.model = model


class _Agent:
    def __init__(self, llm: _LLM | None = None) -> None:
        self.llm = llm


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Crew:
    """The subset of a real ``Crew`` these functions read."""

    def __init__(
        self,
        agents: list[_Agent] | None = None,
        manager_agent: _Agent | None = None,
        usage_answers: list[Any] | None = None,
        raises: bool = False,
    ) -> None:
        self.agents = agents if agents is not None else []
        self.manager_agent = manager_agent
        self._usage_answers = list(usage_answers or [])
        self._raises = raises
        self.task_callback = None

    def calculate_usage_metrics(self) -> Any:
        if self._raises:
            raise RuntimeError("boom")
        return self._usage_answers.pop(0)


class _TaskOutput:
    def __init__(self, name: str | None = None, description: str = "") -> None:
        self.name = name
        self.description = description


def _recorder() -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def record(name: str, **kwargs: Any) -> None:
        calls.append((name, kwargs))

    return calls, record


# ---------------------------------------------------------------------------
# _resolved_model_id
# ---------------------------------------------------------------------------


def test_resolves_the_first_worker_agent_with_a_model() -> None:
    crew = _Crew(agents=[_Agent(_LLM("")), _Agent(_LLM("gpt-4o-mini")), _Agent(_LLM("claude-sonnet-4-6"))])
    assert _resolved_model_id(crew) == "gpt-4o-mini"


def test_falls_through_to_the_manager_agent_when_no_worker_has_one() -> None:
    crew = _Crew(agents=[_Agent(_LLM(""))], manager_agent=_Agent(_LLM("gpt-4o")))
    assert _resolved_model_id(crew) == "gpt-4o"


def test_an_agent_with_no_llm_at_all_is_skipped_rather_than_fatal() -> None:
    crew = _Crew(agents=[_Agent(llm=None), _Agent(_LLM("gpt-4o"))])
    assert _resolved_model_id(crew) == "gpt-4o"


def test_no_agent_anywhere_carries_a_model_answers_empty_not_a_guess() -> None:
    crew = _Crew(agents=[_Agent(_LLM(""))], manager_agent=_Agent(llm=None))
    assert _resolved_model_id(crew) == ""


def test_a_crew_with_no_agents_attribute_at_all_still_checks_the_manager() -> None:
    class _BareCrew:
        manager_agent = _Agent(_LLM("gpt-4o"))

    assert _resolved_model_id(_BareCrew()) == "gpt-4o"


# ---------------------------------------------------------------------------
# _crew_usage_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_reads_the_running_totals() -> None:
    crew = _Crew(usage_answers=[_Usage(120, 40)])
    assert _crew_usage_snapshot(crew) == (120, 40)


def test_snapshot_is_none_when_the_crew_cannot_answer() -> None:
    crew = _Crew(raises=True)
    assert _crew_usage_snapshot(crew) is None


def test_snapshot_is_none_when_the_metrics_object_itself_is_none() -> None:
    crew = _Crew(usage_answers=[None])
    assert _crew_usage_snapshot(crew) is None


# ---------------------------------------------------------------------------
# _attach_task_callback
# ---------------------------------------------------------------------------


def test_a_task_s_own_name_is_never_truncated() -> None:
    """A task's name is its identity, so it arrives untruncated: a real task name
    is a short YAML key rather than prose, so nothing here should cut one short."""
    long_name = "x" * 200
    crew = _Crew(usage_answers=[_Usage(10, 5)])
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="gpt-4o-mini")
    crew.task_callback(_TaskOutput(name=long_name, description="short"))
    name, _ = calls[0]
    assert name == long_name


def test_falls_back_to_the_description_only_when_there_is_no_name() -> None:
    """The fallback leg is bounded — it is the seller's own prompt text,
    unbounded prose, which is what ``loader.build_crew``'s docstring says a
    missing ``name=`` degrades to."""
    crew = _Crew(usage_answers=[_Usage(10, 5)])
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="")
    long_description = "word " * 60 + "\nsecond line never reached"
    crew.task_callback(_TaskOutput(name=None, description=long_description))
    name, _ = calls[0]
    assert name == long_description.splitlines()[0][:120]
    assert len(name) <= 120


def test_neither_name_nor_description_falls_back_to_the_literal_task() -> None:
    crew = _Crew(usage_answers=[_Usage(1, 1)])
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="")
    crew.task_callback(_TaskOutput(name=None, description=""))
    assert calls[0][0] == "task"


def test_each_call_reports_the_delta_since_the_previous_one() -> None:
    crew = _Crew(usage_answers=[_Usage(100, 20), _Usage(260, 55), _Usage(260, 90)])
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="gpt-4o-mini")
    for i in range(3):
        crew.task_callback(_TaskOutput(name=f"t{i}"))
    deltas = [(kwargs["input_tokens"], kwargs["output_tokens"]) for _, kwargs in calls]
    assert deltas == [(100, 20), (160, 35), (0, 35)]
    assert all(kwargs["model_id"] == "gpt-4o-mini" for _, kwargs in calls)


def test_a_crew_whose_usage_cannot_be_read_still_reports_the_step_with_no_tokens() -> None:
    crew = _Crew(raises=True)
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="gpt-4o")
    crew.task_callback(_TaskOutput(name="only-task"))
    name, kwargs = calls[0]
    assert name == "only-task"
    assert kwargs["input_tokens"] is None
    assert kwargs["output_tokens"] is None
    assert kwargs["model_id"] == "gpt-4o"


def test_a_crew_that_refuses_the_callback_attribute_does_not_raise() -> None:
    """Progress reporting is not the run — a crewai release that renames or
    drops ``task_callback`` should cost only the step events, never the
    run itself."""

    class _Stubborn:
        agents: list[Any] = []
        manager_agent = None

        def calculate_usage_metrics(self) -> _Usage:
            return _Usage(0, 0)

        @property
        def task_callback(self) -> None:
            return None

        @task_callback.setter
        def task_callback(self, value: Any) -> None:
            raise AttributeError("frozen")

    _attach_task_callback(_Stubborn(), lambda *a, **k: None, model_id="")  # must not raise
