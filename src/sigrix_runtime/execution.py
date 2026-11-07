"""The one execution path every Sigrix listing bundle runs through.

``main.py`` is a command-line front end over this module and
``sigrix_runtime.postern`` is an HTTP one. Neither owns the run: they own
argv parsing and a socket respectively, and both call :func:`execute`.

That split is the whole point. A crew's kickoff inputs are a contract
spanning five files, and a second caller
that built its own ``crew.kickoff(...)`` would be a sixth place the
``{prompt}`` / ``{configuration}`` pair has to stay in step — the failure
this codebase has already paid for once. There is one kickoff site per run
shape here, and the front ends reach them through :func:`execute`.

What a front end still owns:

* how the result is rendered — printed, or serialised into a Postern
  ``run`` body;
* whether a run happens at all — an entitlement gate is the server's, and
  the command line has none;
* what to say about a failure. :func:`execute` raises, having recorded the
  traceback where ``doctor.py`` looks for it.

Nothing here imports crewai at module load. The loader defers it, the
workforce flow defers it, and this module has no import of its own — so a
process that only needs :func:`load_run_config` (the Postern server
deriving a ``describe`` for a bundle whose virtualenv is not built yet)
pays nothing for the runtime it is not going to start.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sigrix_runtime import configuration, loader, quiet, sandbox, workforce
from sigrix_runtime.runner_env import (
    RUNNER_ENV_PREFIXES,
    drop_runner_settings,
    is_runner_setting,
    without_runner_settings,
)

# Written on any failed run, read by ``doctor.py``'s ``check_last_error``.
# It lives here rather than in a front end so a buyer who only ever starts
# the Postern server still gets the trace doctor asks them for.
LAST_ERROR_FILENAME = ".last_error"

WORKSPACE_DIRNAME = "workspace"

# The reserved Postern input key (SPEC 4.1.1) and the run input this
# starter has always kicked off with are the same string, which is what
# lets a Postern ``run`` body reach an unmodified reviewed config.
PROMPT_INPUT_KEY = "prompt"

# The environment variable crewai reads its model from, and the one the
# generated ``.env.example`` documents as the optional override. Both run
# paths prefer the model crewai itself resolved over this — see
# ``_resolved_model_id``, read off the crew, or off each crew a workforce
# builds — so this is only the answer when no crew names one.
MODEL_ENV_KEY = "OPENAI_MODEL_NAME"

# Where a step carries what Postern's ``usage`` has no member for. Unknown
# members are ignored by every conforming client, and a reverse-domain name
# is what keeps this one from colliding with a member the specification may
# add later: the same namespace the run body's ``files_written`` rides under.
SIGRIX_MEMBER = "org.sigrix"


class MissingDependencyError(RuntimeError):
    """The bundle's requirements are not installed in this Python.

    Raised in place of a bare ``ModuleNotFoundError`` so a front end can
    say what to do about it — the most common first-run stumble is a venv
    that was never activated, and a raw crewai traceback names none of the
    three commands that fix it. ``module`` is what was missing.
    """

    def __init__(self, module: str) -> None:
        super().__init__(f"missing dependency: {module!r} — this Python can't see the bundle's packages.")
        self.module = module


@dataclass
class RunStep:
    """One completed unit of work, in Postern's ``usage.steps`` shape.

    ``input_tokens``/``output_tokens`` default to ``None`` rather than ``0``
    for the same reason ``model_id`` defaults to ``""``: absent means this
    run never determined a value, where ``0`` would claim the step spent
    nothing. A crew whose usage this module could read reports real
    (possibly zero) numbers; one it could not says nothing at all.

    ``cache_write_tokens``/``cache_read_tokens`` say how many of
    ``input_tokens`` the provider wrote to its prompt cache and read back from
    it. They are part of ``input_tokens``, not added to it, and they are priced
    apart from the rest: Anthropic charges a cache write at 1.25 times its
    input rate and a cache read at a tenth of it. Postern's ``usage`` has no
    member for either, so they ride under :data:`SIGRIX_MEMBER`, and only when
    crewai reported them.
    """

    name: str
    latency_ms: int
    model_id: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "latency_ms": self.latency_ms}
        if self.model_id:
            payload["model_id"] = self.model_id
        if self.input_tokens is not None:
            payload["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            payload["output_tokens"] = self.output_tokens
        cache = cache_split(self.cache_write_tokens, self.cache_read_tokens)
        if cache:
            payload[SIGRIX_MEMBER] = cache
        return payload


@dataclass(frozen=True)
class RunUsage:
    """What a run has spent so far, every model call it has made counted.

    crewai's running totals, summed over every crew the run has built, read
    after each model call rather than after each task. A step reports its
    tokens only once its task finishes, so a run stopped inside a task (its
    time limit, a cancel, a crash) reported nothing for that task; this is
    what it had spent when it stopped. The cache counts are inside
    ``input_tokens``, as a step's are, and ``None`` when crewai did not say.
    """

    input_tokens: int
    output_tokens: int
    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None
    model_id: str = ""

    def as_frame(self) -> dict[str, Any]:
        """The worker's frame: the cache split apart, for the engine to place as it places a run's."""
        payload: dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache": cache_split(self.cache_write_tokens, self.cache_read_tokens),
        }
        if self.model_id:
            payload["model_id"] = self.model_id
        return payload


@dataclass
class RunOutcome:
    """What a completed run produced, before any front end renders it.

    The cache counts are the run's totals, read as :class:`RunStep` reads them
    for one step, and ``None`` when crewai did not report them.
    """

    text: str
    files_written: list[str] = field(default_factory=list)
    steps: list[RunStep] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None
    model_id: str = ""
    duration_seconds: float = 0.0
    # True when the bundle carried a workforce.yaml manifest and it loaded.
    # A front end that wants to say "orchestrated N unit crews" needs to know
    # which of the two shapes ran; nothing else does.
    workforce: bool = False


@dataclass
class RunConfig:
    """The bundle's own declaration of what it is, without running it.

    Loaded by both :func:`execute` and by the Postern server's ``describe``
    fallback, which is why it carries the tool refs and the agent/task
    names rather than only the objects the loader builds from them.
    """

    tool_refs: list[str]
    task_names: list[str]
    agent_names: list[str]
    workforce: bool


def load_run_config(bundle_root: Path) -> RunConfig:
    """Parse the bundle's config without constructing anything runnable.

    Prefers the workforce manifest, exactly as :func:`execute` does, so the
    two cannot disagree about which shape a bundle is. A manifest that
    fails to load degrades to the flattened config here for the same reason
    it does there — the flattened form is the reviewed, run-captured
    fallback, and a buyer always has something runnable.
    """
    config_dir = bundle_root / "config"
    manifest = _load_workforce_manifest(config_dir)[0]
    if manifest is not None:
        # ``load_workforce`` has already run each unit through the single-crew
        # loader and hung the result on the unit, so there is nothing to parse
        # again here.
        tool_refs: list[str] = []
        task_names: list[str] = []
        agent_names: list[str] = []
        for unit in manifest.units:
            unit_config = unit.config
            if unit_config is None:
                continue
            tool_refs.extend(ref for ref in unit_config.tool_refs if ref not in tool_refs)
            task_names.extend(f"{unit.key}.{task.name}" for task in unit_config.tasks)
            agent_names.extend(f"{unit.key}.{agent.name}" for agent in unit_config.agents)
        return RunConfig(tool_refs=tool_refs, task_names=task_names, agent_names=agent_names, workforce=True)

    config = loader.load_config(config_dir)
    return RunConfig(
        tool_refs=list(config.tool_refs),
        task_names=[task.name for task in config.tasks],
        agent_names=[agent.name for agent in config.agents],
        workforce=False,
    )


def prepare_environment(bundle_root: Path) -> Path:
    """Load ``.env``, apply the quiet defaults, and ensure ``workspace/``.

    Ordering is load-bearing and is the ordering ``main.py`` has always
    used: the buyer's ``.env`` wins over the quiet defaults, and both land
    before anything imports crewai — so a run stays local and
    non-interactive unless the buyer opted in.

    **The runner's own settings come straight back out again**.
    ``Engine._spawn`` keeps them out of the subprocess, but ``.env.example``
    tells the buyer to copy itself to ``.env`` with ``SIGRIX_TOKEN`` in it —
    so on the documented happy path this ``load_dotenv`` puts the buyer's
    account-wide token back into a process running a seller's crew. Nothing
    below this line reads any of them; see :data:`RUNNER_ENV_PREFIXES`.
    """
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(exc.name or "python-dotenv") from exc
    load_dotenv(bundle_root / ".env")
    drop_runner_settings()
    quiet.apply_quiet_env_defaults()
    workspace = bundle_root / WORKSPACE_DIRNAME
    workspace.mkdir(exist_ok=True)
    return workspace


def execute(
    bundle_root: Path,
    prompt: str,
    *,
    variables: Mapping[str, Any] | None = None,
    on_step: Callable[[RunStep], None] | None = None,
    on_notice: Callable[[str], None] | None = None,
    on_usage: Callable[[RunUsage], None] | None = None,
) -> RunOutcome:
    """Run the bundle against ``prompt`` and return what it produced.

    ``variables`` are the buyer's setup answers. Omitted, they are read
    from ``variables.json`` — which is what the command line wants, since
    that file *is* how a buyer reconfigures a run there. A caller that has
    its own answers passes them instead: a Postern ``run`` declares those
    same keys as inputs (SPEC 4.1.1), so a client supplying one has to be
    able to override the file for that run without editing it.

    ``on_step`` is called as each task finishes, which is what a Postern
    ``stream`` relays as a ``step`` event. Only the *finished* edge is
    reported, and deliberately: a hierarchical crew's manager decides which
    member runs, so a ``started`` event emitted from the task list would be
    a guess about work that may never happen. The specification asks for at
    least a name and an edge, not for the pair.

    ``on_notice`` receives the run's own asides — the setup values in play,
    a workforce manifest that failed to load — as plain sentences. The
    command line prints them to stderr; the server logs them.

    ``on_usage`` is handed what the run has spent so far after every model
    call it makes (:class:`RunUsage`), read off crewai's event bus. A step
    says what its task spent only once the task finishes, so this is what a
    run stopped inside a task had spent. A crewai without the bus reports at
    each step alone, and the run is the same either way.

    Raises whatever the run raised, after writing the traceback to
    ``.last_error`` for ``doctor.py``.
    """
    started = time.monotonic()
    last_error = bundle_root / LAST_ERROR_FILENAME

    setup_values = configuration.load_variables(bundle_root) if variables is None else dict(variables)
    config_block = configuration.render_configuration_block(setup_values)
    if setup_values and on_notice is not None:
        on_notice(
            f"Using {len(setup_values)} setup value(s) from "
            f"{configuration.VARIABLES_FILENAME}: {', '.join(setup_values)}"
        )

    workspace = prepare_environment(bundle_root)
    before_run = _workspace_files(workspace)

    # The buyer's own explicit override, or nothing, until a crew is built:
    # both branches below upgrade it to what CrewAI actually resolved, which
    # is why this is reassigned rather than read again later.
    model_id = str(os.environ.get(MODEL_ENV_KEY) or "").strip()
    recorder = _StepRecorder(on_step)
    steps = recorder.steps
    # Every crew a workforce builds, units and its own, for the run's totals.
    workforce_crews: list[Any] = []
    usage = _UsageListener(on_usage, model_id=model_id) if on_usage is not None else None

    def _instrument(crew: Any, prefix: str) -> None:
        # A workforce's unit crews and its own, each reporting its tasks as a
        # single crew does, its steps named under its unit's key.
        workforce_crews.append(crew)
        if usage is not None:
            usage.watch(crew)
        _attach_task_callback(crew, recorder, model_id=_resolved_model_id(crew) or model_id, prefix=prefix)

    try:
        manifest, manifest_error = _load_workforce_manifest(bundle_root / "config")
        with _reporting_usage(usage):
            if manifest is not None:
                result = workforce.run_workforce(
                    manifest, prompt=prompt, workspace=workspace, configuration=config_block, on_crew=_instrument
                )
                model_id = next((m for m in map(_resolved_model_id, workforce_crews) if m), model_id)
            else:
                if manifest_error is not None and on_notice is not None:
                    on_notice(
                        f"workforce.yaml could not be loaded ({manifest_error}); falling back to the flattened crew "
                        "config."
                    )
                config = loader.load_config(bundle_root / "config")
                tools = sandbox.build_tools(config.tool_refs, workspace=workspace)
                crew = loader.build_crew(config, tools)
                model_id = _resolved_model_id(crew) or model_id
                if usage is not None:
                    usage.watch(crew)
                _attach_task_callback(crew, recorder, model_id=model_id)
                result = crew.kickoff(inputs={"prompt": prompt, "configuration": config_block})
    except ModuleNotFoundError as exc:
        # crewai and its tree are imported lazily, so a bundle whose
        # requirements were never installed fails here rather than at import.
        # Reported as the actionable error rather than as an agent failure.
        last_error.write_text(traceback.format_exc(), encoding="utf-8")
        raise MissingDependencyError(exc.name or "crewai") from exc
    except Exception:
        last_error.write_text(traceback.format_exc(), encoding="utf-8")
        raise

    if last_error.exists():
        last_error.unlink()

    after_run = _workspace_files(workspace)
    outcome = RunOutcome(
        text=str(result),
        files_written=sorted(after_run - before_run),
        steps=steps,
        duration_seconds=time.monotonic() - started,
        workforce=manifest is not None,
        model_id=model_id,
    )
    if manifest is not None:
        # A Flow's answer is text, with no usage on it: the run spent what
        # every crew it built spent.
        _attach_totals(outcome, _summed_counts(workforce_crews))
    else:
        _attach_usage(outcome, result)
    return outcome


# --- Internals -------------------------------------------------------


class _StepRecorder:
    """Every finished step of a run, kept and handed on one at a time.

    A workforce's panel runs its units on threads of their own, so two tasks
    can finish together. The lock keeps the list and the frames the worker
    writes from interleaving.
    """

    def __init__(self, on_step: Callable[[RunStep], None] | None) -> None:
        self.steps: list[RunStep] = []
        self._on_step = on_step
        self._lock = threading.Lock()

    def __call__(
        self,
        name: str,
        *,
        latency_ms: int = 0,
        model_id: str = "",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        cache_read_tokens: int | None = None,
    ) -> None:
        step = RunStep(
            name=name,
            latency_ms=max(0, latency_ms),
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
        )
        with self._lock:
            self.steps.append(step)
            if self._on_step is not None:
                # Raising here aborts the run, which is how a disconnected
                # stream client stops an agent: the relay's write fails, the
                # exception unwinds through crewai, and no further task starts.
                self._on_step(step)


class _UsageListener:
    """Hands ``on_usage`` what a run has spent so far, each time crewai finishes a model call.

    Read as a step's tokens are read (``_crew_usage_snapshot``), summed over
    every crew the run has built, all or nothing (``_summed_counts``), so the
    running totals and the steps' deltas come from one source and agree. A
    reading is handed on only when it moved: crewai fires one event per call,
    and the totals it counts are the same until the next call is counted.

    crewai calls the handler on a thread of its own, after the call it reports,
    so the lock keeps two readings from being handed on out of order. Once the
    run has returned, nothing more is handed on: a reading still in crewai's
    pool would otherwise reach the worker after its ``done``.
    """

    def __init__(self, on_usage: Callable[[RunUsage], None], *, model_id: str) -> None:
        self._on_usage = on_usage
        self._model_id = model_id
        self._crews: list[Any] = []
        self._last: _Counts | None = None
        self._closed = False
        self._lock = threading.Lock()

    def watch(self, crew: Any) -> None:
        """Count ``crew`` in the run's totals from now on."""
        with self._lock:
            self._crews.append(crew)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def report(self) -> None:
        """Read the run's totals and hand them on, if they moved since the last reading. Never raises."""
        with self._lock:
            if self._closed:
                return
            counts = _summed_counts(self._crews)
            if counts is None or counts == self._last:
                return
            self._last = counts
            model_id = next((m for m in map(_resolved_model_id, self._crews) if m), self._model_id)
            reading = RunUsage(
                input_tokens=counts.prompt_tokens,
                output_tokens=counts.completion_tokens,
                cache_write_tokens=counts.cache_write_tokens,
                cache_read_tokens=counts.cache_read_tokens,
                model_id=model_id,
            )
            try:
                self._on_usage(reading)
            except Exception:  # noqa: BLE001 - progress reporting is not the run
                pass


#: The longest a run waits, once it has returned, for crewai to call the handlers of its last events.
_USAGE_FLUSH_SECONDS = 5.0


@contextmanager
def _reporting_usage(listener: _UsageListener | None) -> Iterator[None]:
    """Listen for crewai's finished model calls for the length of the block, if anyone asked.

    Guarded rather than assumed, as the task callback is: a crewai whose event
    bus is missing or renamed costs a run its running usage, never the run. At
    the end the bus is flushed, a bounded wait for the handlers crewai has still
    to call, so the run's last call is reported before the listener closes; then
    the handler is taken off the bus, so a process that runs again (the command
    line, a test) does not hand one run's calls to another's listener.
    """
    if listener is None:
        yield
        return
    try:
        from crewai.events import LLMCallCompletedEvent, crewai_event_bus
    except Exception:  # noqa: BLE001 - no bus to listen on: the steps still report
        yield
        return

    def _finished_call(_source: Any, _event: Any) -> None:
        listener.report()

    try:
        crewai_event_bus.on(LLMCallCompletedEvent)(_finished_call)
    except Exception:  # noqa: BLE001 - as above
        yield
        return
    try:
        yield
    finally:
        try:
            crewai_event_bus.flush(timeout=_USAGE_FLUSH_SECONDS)
        except Exception:  # noqa: BLE001 - the run is over either way
            pass
        listener.close()
        try:
            crewai_event_bus.off(LLMCallCompletedEvent, _finished_call)
        except Exception:  # noqa: BLE001 - as above
            pass


def _load_workforce_manifest(config_dir: Path) -> tuple[Any | None, Exception | None]:
    """The manifest, or ``None`` plus the reason it could not be loaded.

    An AI Workforce bundle carries ``workforce.yaml`` next to the flattened
    single-crew config. The manifest is the primary form; a manifest that
    fails to load degrades to the flattened config rather than failing the
    run.
    """
    try:
        return workforce.load_workforce(config_dir), None
    except workforce.WorkforceConfigError as exc:
        return None, exc


def _attach_task_callback(crew: Any, record: Callable[..., None], *, model_id: str, prefix: str = "") -> None:
    """Report each finished task through crewai's own callback, if it has one.

    ``prefix`` names the crew a step came from when there is more than one: a
    workforce's step is ``<unit key>.<task>``, as its ``describe`` lists the
    task, and its own crews' steps ``workforce.<task>``.

    Guarded rather than assumed: ``task_callback`` is not part of anything
    this bundle pins, and a crewai release that drops or renames it should
    cost a run its progress reporting, never the run itself.

    **Per-step tokens are a delta, not a measurement crewai hands over.**
    Usage is tracked cumulatively per agent —
    ``agent.llm.get_token_usage_summary()`` / ``agent._token_process`` — and
    ``TaskOutput`` itself carries no usage field at all (re-checked against
    the pinned 1.15.20 at that bump: its fields are description, name,
    expected_output, summary, raw, pydantic, json_dict, agent, output_format,
    messages, tool_failures). So a step's tokens are read as (the crew's running total
    right after this task) minus (the total after the previous one) — the
    only per-task granularity the public API exposes, and exact for a
    sequential crew. For a fan-out one (``_process: parallel``'s
    ``async_execution`` tasks), ``task_callback`` can fire from more than one
    thread, so two tasks finishing close together can split a delta unevenly
    between them; the deltas still sum to the crew's real total, which is
    what ``cost_usd`` is computed from, so that number stays correct even
    when one step's does not. The lock keeps the running-total bookkeeping
    itself race-free — a plain read-then-write here would double-count or
    drop tokens under exactly that concurrency.

    The cache counts are deltas of crewai's own running totals in the same
    way, and stay ``None`` for a crewai whose totals do not carry them.

    **A step's ``latency_ms`` is its own crew's time.** SPEC 4.3 makes it the
    step's elapsed time, so each crew keeps its own clock, started when its
    kickoff starts (crewai's ``before_kickoff_callbacks``) and moved on at each
    of its steps. One clock for the run read a panel's units against each
    other: a unit finishing just after another reported nothing. A crewai
    without that list times the first step from here.
    """
    lock = threading.Lock()
    previous = _Counts(prompt_tokens=0, completion_tokens=0, cache_write_tokens=0, cache_read_tokens=0)
    clock = {"last": time.perf_counter()}

    def _kicked_off(inputs: Any) -> Any:
        with lock:
            clock["last"] = time.perf_counter()
        return inputs

    try:
        crew.before_kickoff_callbacks.append(_kicked_off)
    except Exception:  # noqa: BLE001 - progress reporting is not the run
        pass

    def _callback(task_output: Any) -> None:
        nonlocal previous
        # The task's own name is never truncated: it is the task's identity,
        # and a short YAML key, not prose. Only the
        # fallback needs a bound: a task with no ``name=`` reports none on
        # its ``TaskOutput`` either, and its ``description`` is the seller's
        # own prompt text, unbounded, which is what the cut-to-120 protects
        # against — see ``loader.build_crew``'s docstring for why a task
        # should carry ``name=`` at all.
        name = str(getattr(task_output, "name", "") or "").strip()
        if not name:
            description = str(getattr(task_output, "description", "") or "").strip()
            name = description.splitlines()[0][:120] if description else ""
        delta: _Counts | None = None
        with lock:
            now = time.perf_counter()
            latency_ms = int((now - clock["last"]) * 1000)
            clock["last"] = now
            snapshot = _crew_usage_snapshot(crew)
            if snapshot is not None:
                delta = snapshot.since(previous)
                previous = snapshot
        name = name or "task"
        record(
            f"{prefix}.{name}" if prefix else name,
            latency_ms=latency_ms,
            model_id=model_id,
            input_tokens=delta.prompt_tokens if delta is not None else None,
            output_tokens=delta.completion_tokens if delta is not None else None,
            cache_write_tokens=delta.cache_write_tokens if delta is not None else None,
            cache_read_tokens=delta.cache_read_tokens if delta is not None else None,
        )

    try:
        crew.task_callback = _callback
    except Exception:  # noqa: BLE001 - progress reporting is not the run
        pass


def _resolved_model_id(crew: Any) -> str:
    """The model crewai actually resolved for this crew, read off an agent.

    ``Agent.llm`` resolves to a real LLM object the moment the agent is
    constructed — crewai's own default when the buyer set nothing — so its
    ``.model`` is what actually produced the tokens being counted, unlike
    ``OPENAI_MODEL_NAME``, which most buyers never set and which then leaves
    every step, and ``cost_usd``, silently absent. Every agent this
    loader builds shares one model — nothing passes a per-agent ``llm=``
    override — so the first one found, worker or manager, is the run's.
    """
    for agent in [*(getattr(crew, "agents", None) or []), getattr(crew, "manager_agent", None)]:
        if agent is None:
            continue
        model = getattr(getattr(agent, "llm", None), "model", "")
        if model:
            return str(model)
    return ""


@dataclass(frozen=True)
class _Counts:
    """One reading of crewai's token totals, a crew's running ones or a run's.

    crewai counts every input token in ``prompt_tokens``, cached or not: its
    Anthropic provider adds the cache writes and the cache reads into the
    input it reports, and keeps each beside it, as ``cache_creation_tokens``
    and ``cached_prompt_tokens`` (both on ``UsageMetrics`` since this extra's
    floor, 1.15.20). A cache count is ``None`` when the reading has no such
    field, which says nothing, where a zero says no cache was used.
    """

    prompt_tokens: int
    completion_tokens: int
    cache_write_tokens: int | None
    cache_read_tokens: int | None

    def plus(self, other: _Counts) -> _Counts:
        """Two crews' totals together; a cache count either one lacks is unknown."""
        return _Counts(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_write_tokens=_count_plus(self.cache_write_tokens, other.cache_write_tokens),
            cache_read_tokens=_count_plus(self.cache_read_tokens, other.cache_read_tokens),
        )

    def since(self, earlier: _Counts) -> _Counts:
        """What was spent between ``earlier`` and this reading, never below zero."""
        return _Counts(
            prompt_tokens=max(0, self.prompt_tokens - earlier.prompt_tokens),
            completion_tokens=max(0, self.completion_tokens - earlier.completion_tokens),
            cache_write_tokens=_count_since(self.cache_write_tokens, earlier.cache_write_tokens),
            cache_read_tokens=_count_since(self.cache_read_tokens, earlier.cache_read_tokens),
        )


def _count_plus(one: int | None, other: int | None) -> int | None:
    if one is None or other is None:
        return None
    return one + other


def _count_since(now: int | None, earlier: int | None) -> int | None:
    if now is None:
        return None
    return max(0, now - (earlier or 0))


def _optional_count(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None)
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _read_counts(usage: Any) -> _Counts:
    return _Counts(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cache_write_tokens=_optional_count(usage, "cache_creation_tokens"),
        cache_read_tokens=_optional_count(usage, "cached_prompt_tokens"),
    )


def cache_split(cache_write_tokens: int | None, cache_read_tokens: int | None) -> dict[str, int]:
    """The members a step, or a run's ``usage``, carries under :data:`SIGRIX_MEMBER`.

    Each count that was reported, and none that was not. An empty answer
    leaves the namespace out, rather than writing a zero nobody measured.
    """
    split: dict[str, int] = {}
    if cache_write_tokens is not None:
        split["cache_write_tokens"] = cache_write_tokens
    if cache_read_tokens is not None:
        split["cache_read_tokens"] = cache_read_tokens
    return split


def _crew_usage_snapshot(crew: Any) -> _Counts | None:
    """The crew's token totals right now, cumulative, or ``None``.

    ``calculate_usage_metrics`` is the same method ``kickoff()`` calls at the
    end to populate its own result; calling it mid-run just re-reads each
    agent's running counters early, live, rather than a separate in-progress
    API.
    """
    try:
        usage = crew.calculate_usage_metrics()
    except Exception:  # noqa: BLE001 - progress reporting is not the run
        return None
    if usage is None:
        return None
    return _read_counts(usage)


def _summed_counts(crews: list[Any]) -> _Counts | None:
    """What every crew spent, together, or ``None`` if any could not say.

    All or nothing: a crew whose totals cannot be read spent something
    nobody knows, and a sum without it is a smaller number, not a known one.
    """
    if not crews:
        return None
    total = _Counts(prompt_tokens=0, completion_tokens=0, cache_write_tokens=0, cache_read_tokens=0)
    for crew in crews:
        counts = _crew_usage_snapshot(crew)
        if counts is None:
            return None
        total = total.plus(counts)
    return total


def _attach_usage(outcome: RunOutcome, result: Any) -> None:
    """Copy crewai's token totals onto the outcome, when it reports them.

    Absent rather than zero when it does not: Postern's ``usage`` is
    ``SHOULD`` be present "when the runner can determine it", and a zero
    token count is a determination rather than an absence.
    """
    usage = getattr(result, "token_usage", None)
    if usage is None:
        return
    _attach_totals(outcome, _read_counts(usage))


def _attach_totals(outcome: RunOutcome, counts: _Counts | None) -> None:
    if counts is None:
        return
    outcome.input_tokens = counts.prompt_tokens
    outcome.output_tokens = counts.completion_tokens
    outcome.cache_write_tokens = counts.cache_write_tokens
    outcome.cache_read_tokens = counts.cache_read_tokens


def _workspace_files(workspace: Path) -> set[str]:
    return {p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file()}


__all__ = [
    "LAST_ERROR_FILENAME",
    "MODEL_ENV_KEY",
    "MissingDependencyError",
    "PROMPT_INPUT_KEY",
    "WORKSPACE_DIRNAME",
    "RunConfig",
    "RunOutcome",
    "RUNNER_ENV_PREFIXES",
    "RunStep",
    "RunUsage",
    "SIGRIX_MEMBER",
    "cache_split",
    "drop_runner_settings",
    "execute",
    "is_runner_setting",
    "load_run_config",
    "prepare_environment",
    "without_runner_settings",
]
