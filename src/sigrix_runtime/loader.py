"""YAML loader + validator for Sigrix listing bundles.

Parses ``agents.yaml`` and ``tasks.yaml`` into structured data, validates
references and tool support, and (when ``build_crew`` is invoked)
constructs CrewAI ``Agent`` / ``Task`` / ``Crew`` objects.

The pure parsing/validation half is testable without crewai installed.
``build_crew`` imports crewai lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_TOOLS = frozenset({"file_read", "file_write", "web_search", "serper_search"})
SUPPORTED_PROCESSES = frozenset({"sequential", "hierarchical", "parallel"})

PROCESS_KEY = "_process"
# Hierarchical crews name their manager agent here (top-level tasks.yaml key,
# next to ``_process``). CrewAI requires a hierarchical Crew to carry a
# manager and requires that manager NOT to be listed among its agents, so the
# loader needs to know which agent to lift out.
MANAGER_KEY = "_manager"


class BundleConfigError(ValueError):
    """Raised when agents.yaml or tasks.yaml fails validation."""


@dataclass
class AgentSpec:
    name: str
    role: str
    goal: str
    backstory: str
    tools: list[str] = field(default_factory=list)


@dataclass
class TaskSpec:
    name: str
    description: str
    expected_output: str
    # Empty means unassigned: allowed only under ``_process: hierarchical``,
    # where the manager delegates the task to the member it chooses.
    agent: str = ""
    context: list[str] = field(default_factory=list)
    # Fan-out: an async task is kicked off without being awaited;
    # a later task lists it in ``context`` to join on its result. This is how
    # ``_process: parallel`` panels run members concurrently.
    async_execution: bool = False


@dataclass
class BundleConfig:
    agents: list[AgentSpec]
    tasks: list[TaskSpec]
    process: str = "sequential"
    manager: str = ""  # agent name from ``_manager``; set iff hierarchical

    @property
    def tool_refs(self) -> list[str]:
        seen: list[str] = []
        for agent in self.agents:
            for tool_ref in agent.tools:
                if tool_ref not in seen:
                    seen.append(tool_ref)
        return seen


def load_config(config_dir: Path) -> BundleConfig:
    agents_path = config_dir / "agents.yaml"
    tasks_path = config_dir / "tasks.yaml"

    if not agents_path.exists():
        raise BundleConfigError(f"agents.yaml not found at {agents_path}")
    if not tasks_path.exists():
        raise BundleConfigError(f"tasks.yaml not found at {tasks_path}")

    raw_agents = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
    raw_tasks = yaml.safe_load(tasks_path.read_text(encoding="utf-8")) or {}

    if not isinstance(raw_agents, dict):
        raise BundleConfigError("agents.yaml must be a mapping of agent_name -> spec")
    if not isinstance(raw_tasks, dict):
        raise BundleConfigError("tasks.yaml must be a mapping of task_name -> spec")

    process = "sequential"
    if PROCESS_KEY in raw_tasks:
        candidate = raw_tasks.pop(PROCESS_KEY)
        if not isinstance(candidate, str):
            raise BundleConfigError(f"{PROCESS_KEY} must be a string")
        process = candidate

    manager = ""
    if MANAGER_KEY in raw_tasks:
        candidate = raw_tasks.pop(MANAGER_KEY)
        if not isinstance(candidate, str):
            raise BundleConfigError(f"{MANAGER_KEY} must be a string")
        manager = candidate

    agents = [_parse_agent(name, spec) for name, spec in raw_agents.items()]
    tasks = [_parse_task(name, spec) for name, spec in raw_tasks.items()]

    config = BundleConfig(agents=agents, tasks=tasks, process=process, manager=manager)
    _validate(config)
    return config


def _parse_agent(name: str, spec: Any) -> AgentSpec:
    if not isinstance(spec, dict):
        raise BundleConfigError(f"agent {name!r}: spec must be a mapping")
    required = {"role", "goal", "backstory"}
    missing = required - spec.keys()
    if missing:
        raise BundleConfigError(f"agent {name!r}: missing required fields: {sorted(missing)}")
    tools = spec.get("tools") or []
    if not isinstance(tools, list):
        raise BundleConfigError(f"agent {name!r}: tools must be a list")
    return AgentSpec(
        name=name,
        role=str(spec["role"]),
        goal=str(spec["goal"]),
        backstory=str(spec["backstory"]),
        tools=[str(t) for t in tools],
    )


def _parse_task(name: str, spec: Any) -> TaskSpec:
    if not isinstance(spec, dict):
        raise BundleConfigError(f"task {name!r}: spec must be a mapping")
    # ``agent`` is validated later: it may be omitted only under
    # ``_process: hierarchical`` (the manager delegates the task).
    required = {"description", "expected_output"}
    missing = required - spec.keys()
    if missing:
        raise BundleConfigError(f"task {name!r}: missing required fields: {sorted(missing)}")
    context = spec.get("context") or []
    if not isinstance(context, list):
        raise BundleConfigError(f"task {name!r}: context must be a list")
    async_execution = spec.get("async_execution", False)
    if not isinstance(async_execution, bool):
        raise BundleConfigError(f"task {name!r}: async_execution must be a boolean")
    return TaskSpec(
        name=name,
        description=str(spec["description"]),
        expected_output=str(spec["expected_output"]),
        agent=str(spec.get("agent") or ""),
        context=[str(c) for c in context],
        async_execution=async_execution,
    )


def _validate(config: BundleConfig) -> None:
    if not config.agents:
        raise BundleConfigError("agents.yaml must define at least one agent")

    agent_names = [a.name for a in config.agents]
    if len(set(agent_names)) != len(agent_names):
        raise BundleConfigError("agents.yaml has duplicate agent names")

    for agent in config.agents:
        for tool_ref in agent.tools:
            if tool_ref not in SUPPORTED_TOOLS:
                raise BundleConfigError(
                    f"agent {agent.name!r}: unsupported tool {tool_ref!r}. Supported tools: {sorted(SUPPORTED_TOOLS)}"
                )

    if not config.tasks:
        raise BundleConfigError("tasks.yaml must define at least one task")

    task_names = [t.name for t in config.tasks]
    if len(set(task_names)) != len(task_names):
        raise BundleConfigError("tasks.yaml has duplicate task names")

    if config.process not in SUPPORTED_PROCESSES:
        raise BundleConfigError(f"unsupported process {config.process!r}. Supported: {sorted(SUPPORTED_PROCESSES)}")

    hierarchical = config.process == "hierarchical"
    agent_name_set = set(agent_names)

    # CrewAI requires a hierarchical Crew to carry a manager, and requires
    # that manager NOT to be listed among its agents — so a hierarchical
    # bundle must name one via ``_manager``, and the loader lifts it out at
    # build time. Reject at load what would crash at runtime.
    if hierarchical and not config.manager:
        raise BundleConfigError(f"{PROCESS_KEY}: hierarchical requires {MANAGER_KEY} to name the manager agent")
    if config.manager and not hierarchical:
        raise BundleConfigError(f"{MANAGER_KEY} is only valid with {PROCESS_KEY}: hierarchical")
    if config.manager:
        if config.manager not in agent_name_set:
            raise BundleConfigError(f"{MANAGER_KEY}: references undefined agent {config.manager!r}")
        if len(agent_name_set) < 2:
            raise BundleConfigError(
                f"{MANAGER_KEY}: a hierarchical crew needs at least one member agent besides the manager"
            )
        manager_spec = next(a for a in config.agents if a.name == config.manager)
        if manager_spec.tools:
            # CrewAI rejects a manager agent that carries tools.
            raise BundleConfigError(f"{MANAGER_KEY}: manager agent {config.manager!r} must not have tools")

    task_name_set = set(task_names)
    for task in config.tasks:
        if not task.agent:
            if not hierarchical:
                raise BundleConfigError(
                    f"task {task.name!r}: 'agent' is required (only {PROCESS_KEY}: hierarchical tasks may omit it)"
                )
        elif task.agent not in agent_name_set:
            raise BundleConfigError(f"task {task.name!r}: references undefined agent {task.agent!r}")
        elif task.agent == config.manager:
            raise BundleConfigError(
                f"task {task.name!r}: must not be assigned to the manager agent {config.manager!r} "
                "(the manager delegates; leave the task unassigned instead)"
            )
        for ctx in task.context:
            if ctx not in task_name_set:
                raise BundleConfigError(f"task {task.name!r}: context references undefined task {ctx!r}")

    # Fan-out rules. An async task is launched without being
    # awaited, so a trailing one would finish after the crew returns (its
    # output silently lost), and one that declares ``context`` would have to
    # block on other tasks — defeating the point. Reject at load what would
    # misbehave at runtime.
    if config.tasks[-1].async_execution:
        raise BundleConfigError(
            f"task {config.tasks[-1].name!r}: the final task must not set async_execution "
            "(nothing would wait for its result)"
        )
    for task in config.tasks:
        if task.async_execution and task.context:
            raise BundleConfigError(
                f"task {task.name!r}: an async_execution task must not declare context "
                "(async tasks fan out independently; a later synchronous task joins them)"
            )
    if config.process == "parallel" and not any(t.async_execution for t in config.tasks):
        raise BundleConfigError(
            f"{PROCESS_KEY}: parallel requires at least one async_execution task "
            "(without one it would run as a plain sequential chain)"
        )

    _check_acyclic(config.tasks)


def _check_acyclic(tasks: list[TaskSpec]) -> None:
    deps = {t.name: set(t.context) for t in tasks}
    resolved: set[str] = set()
    while len(resolved) < len(deps):
        progress = False
        for name, requires in deps.items():
            if name in resolved:
                continue
            if requires <= resolved:
                resolved.add(name)
                progress = True
        if not progress:
            unresolved = sorted(set(deps) - resolved)
            raise BundleConfigError(f"task DAG has a cycle involving: {unresolved}")


def build_crew(config: BundleConfig, tools: list[Any]) -> Any:
    """Construct a CrewAI ``Crew`` object from the parsed config.

    The ``tools`` list is in the same order as ``config.tool_refs`` (this
    is the contract ``sandbox.build_tools`` honours).

    A hierarchical config (``_process: hierarchical`` + ``_manager``) builds
    the manager as CrewAI's ``manager_agent`` — excluded from ``agents=``
    (CrewAI rejects a manager listed there) and free to delegate an
    unassigned task to whichever member it chooses.

    Each task carries its own ``name=``: CrewAI's ``TaskOutput`` —
    what a ``task_callback`` receives — mirrors it verbatim, and
    ``execution.py``'s progress reporting reads that name for a Postern
    ``usage.steps[*].name``. Leave it unset and CrewAI's own ``TaskOutput``
    reports none either, so the step falls back to the task's *description* —
    the seller's prompt text, truncated mid-word at 120 characters.
    """
    # Every crew-construction path (buyer main.py, the server-side example
    # runner, tests) inherits the quiet posture; setdefault semantics keep
    # caller overrides intact.
    from sigrix_runtime.quiet import apply_quiet_env_defaults

    apply_quiet_env_defaults()

    from crewai import Agent, Crew, Process, Task

    tools_by_ref: dict[str, Any] = dict(zip(config.tool_refs, tools, strict=True))

    crewai_agents: dict[str, Any] = {
        spec.name: Agent(
            role=spec.role,
            goal=spec.goal,
            backstory=spec.backstory,
            tools=[tools_by_ref[t] for t in spec.tools],
            allow_delegation=(spec.name == config.manager),
            verbose=False,
        )
        for spec in config.agents
    }

    crewai_tasks: dict[str, Any] = {}
    for spec in config.tasks:
        crewai_tasks[spec.name] = Task(
            name=spec.name,
            description=spec.description,
            expected_output=spec.expected_output,
            agent=crewai_agents[spec.agent] if spec.agent else None,
            context=[crewai_tasks[c] for c in spec.context],
            async_execution=spec.async_execution,
        )

    process_map = {
        "sequential": Process.sequential,
        "hierarchical": Process.hierarchical,
        # CrewAI's public Process enum has no parallel mode; a parallel
        # bundle runs as a sequential pipeline whose ``async_execution``
        # member tasks fan out concurrently and whose final synchronous
        # task (``context`` = the members) joins them — real fan-out, not
        # a silent chain.
        "parallel": Process.sequential,
    }

    manager_agent = crewai_agents.pop(config.manager) if config.manager else None

    return Crew(
        agents=list(crewai_agents.values()),
        tasks=list(crewai_tasks.values()),
        process=process_map[config.process],
        manager_agent=manager_agent,
        verbose=False,
    )
