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

import threading
import time
from typing import Any

from sigrix_runtime.execution import (  # noqa: E402
    SIGRIX_MEMBER,
    RunOutcome,
    RunStep,
    _attach_task_callback,
    _attach_usage,
    _crew_usage_snapshot,
    _resolved_model_id,
    _StepRecorder,
    _summed_counts,
    cache_split,
)


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


class _CachedUsage(_Usage):
    """crewai's ``UsageMetrics`` as its Anthropic provider fills it.

    ``prompt_tokens`` already counts the cache writes and reads; the two
    fields beside it say how many of it each was.
    """

    def __init__(self, prompt_tokens: int, completion_tokens: int, *, written: int, read: int) -> None:
        super().__init__(prompt_tokens, completion_tokens)
        self.cache_creation_tokens = written
        self.cached_prompt_tokens = read


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
    snapshot = _crew_usage_snapshot(crew)
    assert snapshot is not None
    assert (snapshot.prompt_tokens, snapshot.completion_tokens) == (120, 40)


def test_snapshot_reads_the_cache_counts_crewai_keeps_beside_the_input() -> None:
    crew = _Crew(usage_answers=[_CachedUsage(1600, 300, written=1000, read=500)])
    snapshot = _crew_usage_snapshot(crew)
    assert snapshot is not None
    assert (snapshot.prompt_tokens, snapshot.cache_write_tokens, snapshot.cache_read_tokens) == (1600, 1000, 500)


def test_a_reading_with_no_cache_fields_says_nothing_about_the_cache() -> None:
    """Absent, not zero: a zero would claim the provider cached nothing."""
    snapshot = _crew_usage_snapshot(_Crew(usage_answers=[_Usage(120, 40)]))
    assert snapshot is not None
    assert snapshot.cache_write_tokens is None
    assert snapshot.cache_read_tokens is None


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


def test_each_step_carries_its_own_share_of_the_cache_writes_and_reads() -> None:
    """The figures R1 measured against Claude's provider: 100 plain, 1,000
    written and 500 read in the first call, counted by crewai as 1,600 input."""
    crew = _Crew(
        usage_answers=[
            _CachedUsage(1600, 300, written=1000, read=500),
            _CachedUsage(3200, 600, written=1000, read=2000),
        ]
    )
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="claude-sonnet-5-5")
    crew.task_callback(_TaskOutput(name="first"))
    crew.task_callback(_TaskOutput(name="second"))
    shares = [
        (kwargs["input_tokens"], kwargs["cache_write_tokens"], kwargs["cache_read_tokens"]) for _, kwargs in calls
    ]
    assert shares == [(1600, 1000, 500), (1600, 0, 1500)]


def test_a_crew_whose_totals_carry_no_cache_fields_reports_no_cache_split() -> None:
    crew = _Crew(usage_answers=[_Usage(100, 20)])
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="gpt-4o-mini")
    crew.task_callback(_TaskOutput(name="only"))
    _, kwargs = calls[0]
    assert kwargs["input_tokens"] == 100
    assert kwargs["cache_write_tokens"] is None
    assert kwargs["cache_read_tokens"] is None


def test_a_crew_whose_usage_cannot_be_read_reports_no_cache_split_either() -> None:
    """The step is still reported, with nothing measured."""
    calls, record = _recorder()
    crew = _Crew(raises=True)
    _attach_task_callback(crew, record, model_id="")
    crew.task_callback(_TaskOutput(name="only"))
    name, kwargs = calls[0]
    assert name == "only"
    assert kwargs["cache_write_tokens"] is None
    assert kwargs["cache_read_tokens"] is None


# ---------------------------------------------------------------------------
# What a step and a run carry
# ---------------------------------------------------------------------------


def test_a_step_carries_its_cache_split_under_the_sigrix_namespace() -> None:
    """Postern's ``usage.steps[*]`` has no member for it, and a member of our own
    name could collide with one the specification adds later."""
    step = RunStep(
        name="t", latency_ms=5, input_tokens=1600, output_tokens=300, cache_write_tokens=1000, cache_read_tokens=0
    )
    payload = step.as_dict()
    assert payload[SIGRIX_MEMBER] == {"cache_write_tokens": 1000, "cache_read_tokens": 0}
    assert SIGRIX_MEMBER == "org.sigrix"
    assert "cache_write_tokens" not in payload and "cache_read_tokens" not in payload


def test_a_step_with_no_cache_counts_carries_no_namespace() -> None:
    payload = RunStep(name="t", latency_ms=5, input_tokens=10, output_tokens=2).as_dict()
    assert SIGRIX_MEMBER not in payload


def test_the_split_names_each_count_it_has_and_none_it_does_not() -> None:
    assert cache_split(None, None) == {}
    assert cache_split(0, None) == {"cache_write_tokens": 0}
    assert cache_split(None, 7) == {"cache_read_tokens": 7}


def test_a_run_takes_its_cache_totals_from_crewai_s_own_result() -> None:
    class _Result:
        token_usage = _CachedUsage(4210, 918, written=1000, read=1200)

    outcome = RunOutcome(text="done")
    _attach_usage(outcome, _Result())
    assert (outcome.input_tokens, outcome.output_tokens) == (4210, 918)
    assert (outcome.cache_write_tokens, outcome.cache_read_tokens) == (1000, 1200)


def test_a_run_whose_result_carries_no_cache_fields_leaves_them_unset() -> None:
    class _Result:
        token_usage = _Usage(4210, 918)

    outcome = RunOutcome(text="done")
    _attach_usage(outcome, _Result())
    assert outcome.input_tokens == 4210
    assert outcome.cache_write_tokens is None
    assert outcome.cache_read_tokens is None


# ---------------------------------------------------------------------------
# A workforce: several crews, one run
# ---------------------------------------------------------------------------


def test_a_step_is_named_under_the_crew_it_came_from() -> None:
    crew = _Crew(usage_answers=[_Usage(10, 5)])
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="", prefix="review")
    crew.task_callback(_TaskOutput(name="summarise_contract"))
    crew2 = _Crew(usage_answers=[_Usage(10, 5)])
    _attach_task_callback(crew2, record, model_id="", prefix="review")
    crew2.task_callback(_TaskOutput(name=None, description=""))
    assert [name for name, _ in calls] == ["review.summarise_contract", "review.task"]


def test_the_run_spent_what_every_crew_spent() -> None:
    crews = [
        _Crew(usage_answers=[_CachedUsage(1600, 300, written=1000, read=500)]),
        _Crew(usage_answers=[_CachedUsage(3200, 600, written=1000, read=1500)]),
    ]
    total = _summed_counts(crews)
    assert total is not None
    assert (total.prompt_tokens, total.completion_tokens) == (4800, 900)
    assert (total.cache_write_tokens, total.cache_read_tokens) == (2000, 2000)


def test_a_crew_that_cannot_say_what_it_spent_leaves_the_run_unmeasured() -> None:
    """All or nothing: the sum of the others is a smaller number, not a known one."""
    crews = [_Crew(usage_answers=[_Usage(1600, 300)]), _Crew(raises=True)]
    assert _summed_counts(crews) is None


def test_a_crew_with_no_cache_fields_leaves_the_run_s_cache_split_unknown() -> None:
    crews = [
        _Crew(usage_answers=[_CachedUsage(1600, 300, written=1000, read=500)]),
        _Crew(usage_answers=[_Usage(10, 5)]),
    ]
    total = _summed_counts(crews)
    assert total is not None
    assert (total.prompt_tokens, total.completion_tokens) == (1610, 305)
    assert (total.cache_write_tokens, total.cache_read_tokens) == (None, None)


def test_a_workforce_that_built_no_crew_measured_nothing() -> None:
    assert _summed_counts([]) is None


def test_two_steps_finishing_at_once_are_handed_on_one_at_a_time() -> None:
    """A panel's units finish on threads of their own, and the worker writes each
    step as a frame: two written at once could interleave."""
    spans: list[tuple[float, float]] = []

    def slow_writer(_step: RunStep) -> None:
        began = time.monotonic()
        time.sleep(0.2)
        spans.append((began, time.monotonic()))

    recorder = _StepRecorder(slow_writer)
    together = threading.Barrier(2)

    def finish(name: str) -> None:
        together.wait()
        recorder(name, latency_ms=5, input_tokens=1, output_tokens=1)

    threads = [threading.Thread(target=finish, args=(name,)) for name in ("review.one", "build.one")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    first, second = sorted(spans)
    assert first[1] <= second[0], "two steps were written at once"
    assert sorted(step.name for step in recorder.steps) == ["build.one", "review.one"]


class _KickedOffCrew(_Crew):
    """A crew that says when its kickoff starts, as crewai's ``before_kickoff_callbacks`` do."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.before_kickoff_callbacks: list[Any] = []

    def start(self) -> None:
        for callback in self.before_kickoff_callbacks:
            callback({})


def test_a_step_is_timed_from_its_own_crew_s_start() -> None:
    """A pipeline builds its second unit with its first; timed from then, its first
    step would carry the whole first unit's run."""
    crew = _KickedOffCrew(usage_answers=[_Usage(10, 5)])
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="")
    time.sleep(0.3)  # the first unit runs
    crew.start()
    crew.task_callback(_TaskOutput(name="first"))
    assert calls[0][1]["latency_ms"] < 150


def test_a_crew_that_cannot_say_when_it_started_is_timed_from_its_callback() -> None:
    crew = _Crew(usage_answers=[_Usage(10, 5)])
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="")
    time.sleep(0.2)
    crew.task_callback(_TaskOutput(name="first"))
    assert calls[0][1]["latency_ms"] >= 150


def test_each_step_after_the_first_is_timed_from_its_crew_s_last() -> None:
    crew = _KickedOffCrew(usage_answers=[_Usage(10, 5), _Usage(20, 10)])
    calls, record = _recorder()
    _attach_task_callback(crew, record, model_id="")
    crew.start()
    crew.task_callback(_TaskOutput(name="first"))
    time.sleep(0.2)
    crew.task_callback(_TaskOutput(name="second"))
    assert calls[1][1]["latency_ms"] >= 150
