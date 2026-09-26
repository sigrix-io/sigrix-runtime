"""Workforce (crew-of-crews) manifest loader + Flow orchestration.

An AI Workforce bundle carries, next to the flattened single-crew config,
a ``config/workforce.yaml`` manifest and one *unflattened* crew config per
composed unit under ``config/units/<key>/``. This module loads that tree
and orchestrates the unit crews with CrewAI Flows — each unit runs as a
real ``Crew`` with its own process (a hierarchical unit actually
delegates internally), and the workforce's ``process`` decides how the
units work together:

- ``sequential`` — a pipeline: each unit builds on the previous unit's
  output.
- ``parallel`` — a panel: every unit answers the same request
  concurrently (``kickoff_async``); a synthesizer joins the answers.
- ``hierarchical`` — a router: a workforce-manager step classifies the
  request (org chart + handoff map) and dispatches it to the unit crew
  that owns it; no match falls back to the manager answering itself.

The trust model is unchanged: sellers author YAML, never code — the Flow
classes here are part of the fixed runtime, identical across listings,
driven entirely by the manifest. As in ``loader``, the parsing/validation
half is pure (testable without crewai); crewai imports happen lazily in
the run path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sigrix_runtime import loader, sandbox
from sigrix_runtime.loader import BundleConfig, BundleConfigError

MANIFEST_FILENAME = "workforce.yaml"

SUPPORTED_WORKFORCE_PROCESSES = frozenset({"sequential", "hierarchical", "parallel"})

DEFAULT_SYNTHESIZER_ROLE = "Workforce Synthesizer"
DEFAULT_MANAGER_ROLE = "Workforce Manager"
# The router's "no unit owns this" sentinel; kept lowercase-simple so the
# reply parse stays robust across models.
NO_UNIT_SENTINEL = "none"


class WorkforceConfigError(ValueError):
    """Raised when the workforce manifest or a unit config fails validation."""


@dataclass
class WorkforceUnit:
    key: str
    name: str
    role: str
    dir: str
    config: BundleConfig | None = None


@dataclass
class WorkforceConfig:
    name: str
    process: str
    units: list[WorkforceUnit]
    handoff_map: dict[str, list[str]] = field(default_factory=dict)
    synthesizer: str = ""
    fallback_instructions: str = ""


# ---------------------------------------------------------------------------
# Parsing / validation (pure — no crewai)
# ---------------------------------------------------------------------------


def parse_manifest(raw: Any) -> WorkforceConfig:
    """Parse + validate a workforce manifest mapping (unit configs unloaded).

    Split from :func:`load_workforce` so server-side validation can check
    the manifest the compiler emits without touching the filesystem.
    Rejects at parse time what would misbehave at runtime, matching the
    loader's posture.
    """
    if not isinstance(raw, dict):
        raise WorkforceConfigError(f"{MANIFEST_FILENAME} must be a mapping")

    process = raw.get("process")
    if not isinstance(process, str) or process not in SUPPORTED_WORKFORCE_PROCESSES:
        raise WorkforceConfigError(
            f"{MANIFEST_FILENAME}: process must be one of {sorted(SUPPORTED_WORKFORCE_PROCESSES)}, got {process!r}"
        )

    raw_units = raw.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise WorkforceConfigError(f"{MANIFEST_FILENAME}: units must be a non-empty list")
    units: list[WorkforceUnit] = []
    seen_keys: set[str] = set()
    for index, entry in enumerate(raw_units, 1):
        if not isinstance(entry, dict):
            raise WorkforceConfigError(f"{MANIFEST_FILENAME}: units[{index}] must be a mapping")
        key = str(entry.get("key") or "").strip()
        if not key or not all(ch.isascii() and (ch.isalnum() or ch == "_") for ch in key) or key[:1].isdigit():
            raise WorkforceConfigError(f"{MANIFEST_FILENAME}: units[{index}] needs an identifier-safe 'key'")
        if key in seen_keys:
            raise WorkforceConfigError(f"{MANIFEST_FILENAME}: duplicate unit key {key!r}")
        seen_keys.add(key)
        unit_dir = str(entry.get("dir") or f"units/{key}").strip().strip("/")
        if ".." in Path(unit_dir).parts or Path(unit_dir).is_absolute():
            raise WorkforceConfigError(f"{MANIFEST_FILENAME}: unit {key!r} has an unsafe dir {unit_dir!r}")
        units.append(
            WorkforceUnit(
                key=key,
                name=str(entry.get("name") or key).strip() or key,
                role=str(entry.get("role") or "").strip(),
                dir=unit_dir,
            )
        )

    synthesizer = str(raw.get("synthesizer") or "").strip()
    if synthesizer and process != "parallel":
        raise WorkforceConfigError(f"{MANIFEST_FILENAME}: synthesizer is only valid with process: parallel")

    fallback = str(raw.get("fallback_instructions") or "").strip()
    if fallback and process != "hierarchical":
        raise WorkforceConfigError(
            f"{MANIFEST_FILENAME}: fallback_instructions is only valid with process: hierarchical"
        )

    handoff_raw = raw.get("handoff_map") or {}
    if handoff_raw and process != "hierarchical":
        raise WorkforceConfigError(f"{MANIFEST_FILENAME}: handoff_map is only valid with process: hierarchical")
    if not isinstance(handoff_raw, dict):
        raise WorkforceConfigError(f"{MANIFEST_FILENAME}: handoff_map must be a mapping")
    handoff_map: dict[str, list[str]] = {}
    for key, value in handoff_raw.items():
        source = str(key).strip()
        if not source:
            continue
        if isinstance(value, (list, tuple)):
            targets = [str(v).strip() for v in value if str(v).strip()]
        else:
            targets = [str(value or "").strip()] if str(value or "").strip() else []
        if targets:
            handoff_map[source] = targets

    return WorkforceConfig(
        name=str(raw.get("name") or "AI Workforce").strip() or "AI Workforce",
        process=process,
        units=units,
        handoff_map=handoff_map,
        synthesizer=synthesizer,
        fallback_instructions=fallback,
    )


def load_workforce(config_dir: Path) -> WorkforceConfig | None:
    """Load the workforce tree, or ``None`` when this bundle is a plain crew.

    Each unit's config passes the full single-crew loader validation —
    the same rules a standalone crew bundle must satisfy.
    """
    manifest_path = config_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise WorkforceConfigError(f"{MANIFEST_FILENAME} is not valid YAML: {exc}") from exc

    config = parse_manifest(raw)
    for unit in config.units:
        unit_dir = config_dir / unit.dir
        try:
            unit.config = loader.load_config(unit_dir)
        except BundleConfigError as exc:
            raise WorkforceConfigError(f"unit {unit.key!r} ({unit.dir}): {exc}") from exc
    return config


# ---------------------------------------------------------------------------
# Orchestration (lazy crewai imports)
# ---------------------------------------------------------------------------


def run_workforce(config: WorkforceConfig, *, prompt: str, workspace: Path, configuration: str = "") -> str:
    """Run the workforce against *prompt* and return the final answer text."""
    flow = build_workforce_flow(config, prompt=prompt, workspace=workspace, configuration=configuration)
    return _output_text(flow.kickoff())


def build_workforce_flow(config: WorkforceConfig, *, prompt: str, workspace: Path, configuration: str = "") -> Any:
    """Construct the per-mode Flow over the unit crews (crewai imported here).

    ``configuration`` is the buyer's setup block, passed to every unit
    crew beside the brief so a unit task carrying the ``{configuration}`` slot
    resolves it. It defaults to ``""`` so a bundle predating the bridge — and
    every caller that has nothing to pass — runs exactly as before.
    """
    from sigrix_runtime.quiet import apply_quiet_env_defaults

    apply_quiet_env_defaults()

    unit_crews: list[tuple[WorkforceUnit, Any]] = []
    for unit in config.units:
        if unit.config is None:
            raise WorkforceConfigError(f"unit {unit.key!r} has no loaded config")
        tools = sandbox.build_tools(unit.config.tool_refs, workspace=workspace)
        unit_crews.append((unit, loader.build_crew(unit.config, tools)))

    flow_cls = {
        "sequential": PipelineWorkforceFlow,
        "parallel": PanelWorkforceFlow,
        "hierarchical": RouteWorkforceFlow,
    }[config.process]
    return flow_cls(config=config, unit_crews=unit_crews, prompt=prompt, configuration=configuration)


def _output_text(result: Any) -> str:
    raw = getattr(result, "raw", None)
    return raw if isinstance(raw, str) else str(result)


def _org_chart_lines(config: WorkforceConfig) -> str:
    lines = ["The workforce's units — each a crew with its own specialists:"]
    for unit in config.units:
        entry = f"- {unit.key}: {unit.name}"
        if unit.role:
            entry += f" — {unit.role}"
        if unit.config is not None:
            roles = [a.role for a in unit.config.agents if a.role]
            if roles:
                entry += f" (specialists: {', '.join(roles)})"
        lines.append(entry)
    return "\n".join(lines)


def _handoff_lines(config: WorkforceConfig) -> str:
    lines = [f"- {source} → {', '.join(targets)}" for source, targets in config.handoff_map.items()]
    return "\n".join(lines)


def _single_agent_answer(*, role: str, goal: str, backstory: str, description: str, expected_output: str) -> str:
    """One agent, one task, one answer — the runtime's helper-crew shape."""
    from crewai import Agent, Crew, Process, Task

    agent = Agent(role=role, goal=goal, backstory=backstory, verbose=False)
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return _output_text(crew.kickoff())


def parse_routed_unit_key(reply: str, unit_keys: list[str]) -> str:
    """The router reply → a unit key, or ``""`` for no-match/fallback.

    Tolerant of model chatter: an exact (stripped, lowercased) match wins;
    else the first manifest-ordered key that appears as a token in the
    reply. The ``none`` sentinel — or anything unparseable — means the
    manager keeps the request.
    """
    normalized = reply.strip().strip("`'\"").lower()
    if normalized == NO_UNIT_SENTINEL:
        return ""
    if normalized in unit_keys:
        return normalized
    tokens = set()
    for line in reply.lower().splitlines():
        for word in line.replace(",", " ").split():
            tokens.add(word.strip("`'\".:;!()[]"))
    for key in unit_keys:
        if key in tokens:
            return key
    return ""


class _WorkforceFlowFactory:
    """Late-bound Flow classes: crewai imports only when a flow is built."""

    _cache: dict[str, Any] = {}

    @classmethod
    def get(cls, mode: str) -> Any:
        if mode in cls._cache:
            return cls._cache[mode]
        from crewai.flow.flow import Flow, listen, start

        if mode == "pipeline":

            class PipelineFlow(Flow):
                """Units chain: each builds on the previous unit's output."""

                def __init__(
                    self, *, config: WorkforceConfig, unit_crews: list[Any], prompt: str, configuration: str = ""
                ) -> None:
                    super().__init__()
                    self._config = config
                    self._unit_crews = unit_crews
                    self._prompt = prompt
                    self._configuration = configuration

                @start()
                def run_pipeline(self) -> str:
                    previous_name = ""
                    previous_output = ""
                    for unit, crew in self._unit_crews:
                        unit_prompt = self._prompt
                        if previous_output:
                            unit_prompt = (
                                f"{self._prompt}\n\n"
                                f"## Previous unit's output ({previous_name})\n\n"
                                f"{previous_output}\n\n"
                                "Build on the previous unit's output above."
                            )
                        previous_output = _output_text(
                            crew.kickoff(inputs={"prompt": unit_prompt, "configuration": self._configuration})
                        )
                        previous_name = unit.name
                    return previous_output

            cls._cache[mode] = PipelineFlow
        elif mode == "panel":

            class PanelFlow(Flow):
                """Units answer concurrently; a synthesizer joins the panel."""

                def __init__(
                    self, *, config: WorkforceConfig, unit_crews: list[Any], prompt: str, configuration: str = ""
                ) -> None:
                    super().__init__()
                    self._config = config
                    self._unit_crews = unit_crews
                    self._prompt = prompt
                    self._configuration = configuration

                @start()
                async def fan_out(self) -> list[str]:
                    outputs = await asyncio.gather(
                        *(
                            crew.kickoff_async(inputs={"prompt": self._prompt, "configuration": self._configuration})
                            for _unit, crew in self._unit_crews
                        )
                    )
                    return [_output_text(output) for output in outputs]

                @listen(fan_out)
                def synthesize(self, answers: list[str]) -> str:
                    sections = []
                    for (unit, _crew), answer in zip(self._unit_crews, answers):
                        header = unit.name + (f" — {unit.role}" if unit.role else "")
                        sections.append(f"## {header}\n\n{answer}")
                    role = self._config.synthesizer or DEFAULT_SYNTHESIZER_ROLE
                    return _single_agent_answer(
                        role=role,
                        goal=(
                            "Chair a panel of unit crews: every unit has answered the user's request "
                            "independently and in parallel; weave their answers into one clear, "
                            "complete response."
                        ),
                        backstory=(
                            "You never answer for the units — you read every unit's answer, reconcile "
                            "agreements, surface disagreements honestly, and keep each unit's "
                            "perspective visible.\n\n" + _org_chart_lines(self._config)
                        ),
                        description=(
                            f"The user's request:\n{self._prompt}\n\n"
                            "The units' independent answers:\n\n" + "\n\n".join(sections)
                        ),
                        expected_output="One clear, complete response that keeps each unit's perspective visible.",
                    )

            cls._cache[mode] = PanelFlow
        else:

            class RouteFlow(Flow):
                """A manager step classifies the request and dispatches the owning unit crew."""

                def __init__(
                    self, *, config: WorkforceConfig, unit_crews: list[Any], prompt: str, configuration: str = ""
                ) -> None:
                    super().__init__()
                    self._config = config
                    self._unit_crews = unit_crews
                    self._prompt = prompt
                    self._configuration = configuration

                @start()
                def classify(self) -> str:
                    keys = [unit.key for unit, _crew in self._unit_crews]
                    handoff = _handoff_lines(self._config)
                    description = (
                        "Decide which unit of your workforce owns this request.\n\n"
                        f"The request:\n{self._prompt}\n\n"
                        + (f"How to route between units:\n{handoff}\n\n" if handoff else "")
                        + f"Reply with exactly one unit key from: {', '.join(keys)}. "
                        f"If no unit clearly owns it, reply exactly: {NO_UNIT_SENTINEL}"
                    )
                    reply = _single_agent_answer(
                        role=DEFAULT_MANAGER_ROLE,
                        goal=("Run an AI workforce of specialist crews: route each request to the unit that owns it."),
                        backstory=_org_chart_lines(self._config),
                        description=description,
                        expected_output="One unit key, nothing else.",
                    )
                    return parse_routed_unit_key(reply, keys)

                @listen(classify)
                def dispatch(self, unit_key: str) -> str:
                    for unit, crew in self._unit_crews:
                        if unit.key == unit_key:
                            return _output_text(
                                crew.kickoff(inputs={"prompt": self._prompt, "configuration": self._configuration})
                            )
                    # No unit owns it: the manager answers, following the
                    # workforce's fallback instructions when it has them.
                    guidance = self._config.fallback_instructions or (
                        "Answer helpfully and note which units of the workforce could help with follow-ups."
                    )
                    return _single_agent_answer(
                        role=DEFAULT_MANAGER_ROLE,
                        goal="Cover requests no unit clearly owns, in the workforce's voice.",
                        backstory=_org_chart_lines(self._config),
                        description=(
                            f"No unit of the workforce clearly owns this request — answer it yourself.\n\n"
                            f"Guidance:\n{guidance}\n\nThe request:\n{self._prompt}"
                        ),
                        expected_output="A clear, complete answer to the request.",
                    )

            cls._cache[mode] = RouteFlow
        return cls._cache[mode]


def PipelineWorkforceFlow(**kwargs: Any) -> Any:
    return _WorkforceFlowFactory.get("pipeline")(**kwargs)


def PanelWorkforceFlow(**kwargs: Any) -> Any:
    return _WorkforceFlowFactory.get("panel")(**kwargs)


def RouteWorkforceFlow(**kwargs: Any) -> Any:
    return _WorkforceFlowFactory.get("route")(**kwargs)
