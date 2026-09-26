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
from collections.abc import Callable, Mapping
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
# generated ``.env.example`` documents as the optional override. The single
# crew path (below) prefers the model crewai itself resolved over this — see
# ``_resolved_model_id`` — so this is now only the workforce path's answer,
# since ``execute()`` builds no crew there this module can introspect.
MODEL_ENV_KEY = "OPENAI_MODEL_NAME"


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
    """

    name: str
    latency_ms: int
    model_id: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "latency_ms": self.latency_ms}
        if self.model_id:
            payload["model_id"] = self.model_id
        if self.input_tokens is not None:
            payload["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            payload["output_tokens"] = self.output_tokens
        return payload


@dataclass
class RunOutcome:
    """What a completed run produced, before any front end renders it."""

    text: str
    files_written: list[str] = field(default_factory=list)
    steps: list[RunStep] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
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

    # The workforce path builds no crew this module can introspect, so this
    # is its whole answer: the buyer's own explicit override, or nothing.
    # The single-crew branch below upgrades it to what CrewAI actually
    # resolved, which is why this is reassigned rather than read again later.
    model_id = str(os.environ.get(MODEL_ENV_KEY) or "").strip()
    steps: list[RunStep] = []
    step_clock = {"last": time.perf_counter()}

    def _record(
        name: str, *, model_id: str = "", input_tokens: int | None = None, output_tokens: int | None = None
    ) -> None:
        now = time.perf_counter()
        step = RunStep(
            name=name,
            latency_ms=int((now - step_clock["last"]) * 1000),
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        step_clock["last"] = now
        steps.append(step)
        if on_step is not None:
            # Raising here aborts the run, which is how a disconnected
            # stream client stops an agent: the relay's write fails, the
            # exception unwinds through crewai, and no further task starts.
            on_step(step)

    try:
        manifest, manifest_error = _load_workforce_manifest(bundle_root / "config")
        if manifest is not None:
            result = workforce.run_workforce(manifest, prompt=prompt, workspace=workspace, configuration=config_block)
        else:
            if manifest_error is not None and on_notice is not None:
                on_notice(
                    f"workforce.yaml could not be loaded ({manifest_error}); falling back to the flattened crew config."
                )
            config = loader.load_config(bundle_root / "config")
            tools = sandbox.build_tools(config.tool_refs, workspace=workspace)
            crew = loader.build_crew(config, tools)
            model_id = _resolved_model_id(crew) or model_id
            _attach_task_callback(crew, _record, model_id=model_id)
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
    _attach_usage(outcome, result)
    return outcome


# --- Internals -------------------------------------------------------


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


def _attach_task_callback(crew: Any, record: Callable[..., None], *, model_id: str) -> None:
    """Report each finished task through crewai's own callback, if it has one.

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
    """
    lock = threading.Lock()
    previous = {"prompt_tokens": 0, "completion_tokens": 0}

    def _callback(task_output: Any) -> None:
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
        input_tokens: int | None = None
        output_tokens: int | None = None
        with lock:
            snapshot = _crew_usage_snapshot(crew)
            if snapshot is not None:
                prompt_total, completion_total = snapshot
                input_tokens = max(0, prompt_total - previous["prompt_tokens"])
                output_tokens = max(0, completion_total - previous["completion_tokens"])
                previous["prompt_tokens"] = prompt_total
                previous["completion_tokens"] = completion_total
        record(name or "task", model_id=model_id, input_tokens=input_tokens, output_tokens=output_tokens)

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


def _crew_usage_snapshot(crew: Any) -> tuple[int, int] | None:
    """``(prompt_tokens, completion_tokens)`` right now, cumulative, or ``None``.

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
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)


def _attach_usage(outcome: RunOutcome, result: Any) -> None:
    """Copy crewai's token totals onto the outcome, when it reports them.

    Absent rather than zero when it does not: Postern's ``usage`` is
    ``SHOULD`` be present "when the runner can determine it", and a zero
    token count is a determination rather than an absence.
    """
    usage = getattr(result, "token_usage", None)
    if usage is None:
        return
    outcome.input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    outcome.output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)


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
    "drop_runner_settings",
    "execute",
    "is_runner_setting",
    "load_run_config",
    "prepare_environment",
    "without_runner_settings",
]
