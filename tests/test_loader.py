"""Tests for the crew configuration loader and validator."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from sigrix_runtime import loader  # noqa: E402
from tests.support import CREWS


def _write_config(config_dir: Path, agents_yaml: str, tasks_yaml: str) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "agents.yaml").write_text(textwrap.dedent(agents_yaml).strip() + "\n", encoding="utf-8")
    (config_dir / "tasks.yaml").write_text(textwrap.dedent(tasks_yaml).strip() + "\n", encoding="utf-8")


def test_loads_minimal_persona(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            strategist:
              role: 'Marketing Strategist'
              goal: 'Develop GTM plans.'
              backstory: 'Experienced marketer.'
        """,
        tasks_yaml="""
            answer:
              description: 'Answer the question {prompt}.'
              expected_output: 'A short answer.'
              agent: strategist
        """,
    )
    config = loader.load_config(tmp_path / "config")
    assert len(config.agents) == 1
    assert config.agents[0].name == "strategist"
    assert config.agents[0].role == "Marketing Strategist"
    assert config.tasks[0].agent == "strategist"
    assert config.process == "sequential"


def test_loads_crew_with_chained_tasks(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            researcher:
              role: 'Researcher'
              goal: 'Find facts.'
              backstory: 'A researcher.'
            writer:
              role: 'Writer'
              goal: 'Write summaries.'
              backstory: 'A writer.'
        """,
        tasks_yaml="""
            _process: sequential
            research:
              description: 'Research {prompt}.'
              expected_output: 'Notes.'
              agent: researcher
            summarise:
              description: 'Summarise the notes.'
              expected_output: 'A summary.'
              agent: writer
              context:
                - research
        """,
    )
    config = loader.load_config(tmp_path / "config")
    assert [a.name for a in config.agents] == ["researcher", "writer"]
    assert config.tasks[1].context == ["research"]


def test_tool_refs_are_deduplicated_and_ordered(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            a:
              role: 'A'
              goal: 'g'
              backstory: 'b'
              tools:
                - serper_search
                - file_read
            b:
              role: 'B'
              goal: 'g'
              backstory: 'b'
              tools:
                - file_read
                - file_write
        """,
        tasks_yaml="""
            t1:
              description: 'd'
              expected_output: 'o'
              agent: a
            t2:
              description: 'd'
              expected_output: 'o'
              agent: b
        """,
    )
    config = loader.load_config(tmp_path / "config")
    assert config.tool_refs == ["serper_search", "file_read", "file_write"]


def test_missing_agents_yaml_is_error(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "tasks.yaml").write_text("noop: {description: d, expected_output: o, agent: x}", encoding="utf-8")
    with pytest.raises(loader.BundleConfigError, match="agents.yaml not found"):
        loader.load_config(config_dir)


def test_agent_missing_required_field_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            a:
              role: 'A'
              goal: 'g'
        """,
        tasks_yaml="""
            t:
              description: 'd'
              expected_output: 'o'
              agent: a
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="missing required fields"):
        loader.load_config(tmp_path / "config")


def test_unsupported_tool_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            a:
              role: 'A'
              goal: 'g'
              backstory: 'b'
              tools:
                - shell_exec
        """,
        tasks_yaml="""
            t:
              description: 'd'
              expected_output: 'o'
              agent: a
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="unsupported tool"):
        loader.load_config(tmp_path / "config")


def test_task_references_undefined_agent_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            a:
              role: 'A'
              goal: 'g'
              backstory: 'b'
        """,
        tasks_yaml="""
            t:
              description: 'd'
              expected_output: 'o'
              agent: nonexistent
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="undefined agent"):
        loader.load_config(tmp_path / "config")


def test_task_context_references_undefined_task_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            a:
              role: 'A'
              goal: 'g'
              backstory: 'b'
        """,
        tasks_yaml="""
            t:
              description: 'd'
              expected_output: 'o'
              agent: a
              context:
                - phantom
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="undefined task"):
        loader.load_config(tmp_path / "config")


def test_cycle_in_tasks_is_detected(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            a:
              role: 'A'
              goal: 'g'
              backstory: 'b'
        """,
        tasks_yaml="""
            t1:
              description: 'd'
              expected_output: 'o'
              agent: a
              context: [t2]
            t2:
              description: 'd'
              expected_output: 'o'
              agent: a
              context: [t1]
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="cycle"):
        loader.load_config(tmp_path / "config")


def test_unsupported_process_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            a:
              role: 'A'
              goal: 'g'
              backstory: 'b'
        """,
        tasks_yaml="""
            _process: chaotic
            t:
              description: 'd'
              expected_output: 'o'
              agent: a
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="unsupported process"):
        loader.load_config(tmp_path / "config")


# --- Hierarchical crews (_manager) -----------------------------------------

_TEAM_AGENTS = """
    lead:
      role: 'Lead'
      goal: 'Route each question to its owner.'
      backstory: 'A routing lead.'
    qa:
      role: 'QA'
      goal: 'Test things.'
      backstory: 'A QA engineer.'
    architect:
      role: 'Architect'
      goal: 'Design things.'
      backstory: 'An architect.'
"""


def test_hierarchical_with_manager_loads(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_TEAM_AGENTS,
        tasks_yaml="""
            _process: hierarchical
            _manager: lead
            answer:
              description: 'Delegate and answer {prompt}.'
              expected_output: 'A routed answer.'
        """,
    )
    config = loader.load_config(tmp_path / "config")
    assert config.process == "hierarchical"
    assert config.manager == "lead"
    # The unassigned task is the one the manager delegates.
    assert config.tasks[0].agent == ""


def test_hierarchical_allows_a_preassigned_member_task(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_TEAM_AGENTS,
        tasks_yaml="""
            _process: hierarchical
            _manager: lead
            answer:
              description: 'Answer {prompt}.'
              expected_output: 'An answer.'
              agent: qa
        """,
    )
    config = loader.load_config(tmp_path / "config")
    assert config.tasks[0].agent == "qa"


def test_hierarchical_without_manager_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_TEAM_AGENTS,
        tasks_yaml="""
            _process: hierarchical
            answer:
              description: 'd'
              expected_output: 'o'
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="requires _manager"):
        loader.load_config(tmp_path / "config")


def test_manager_without_hierarchical_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_TEAM_AGENTS,
        tasks_yaml="""
            _manager: lead
            answer:
              description: 'd'
              expected_output: 'o'
              agent: qa
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="only valid with _process: hierarchical"):
        loader.load_config(tmp_path / "config")


def test_manager_referencing_undefined_agent_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_TEAM_AGENTS,
        tasks_yaml="""
            _process: hierarchical
            _manager: phantom
            answer:
              description: 'd'
              expected_output: 'o'
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="undefined agent 'phantom'"):
        loader.load_config(tmp_path / "config")


def test_task_assigned_to_the_manager_is_error(tmp_path: Path) -> None:
    # CrewAI lifts the manager out of agents=, so a task pinned to it could
    # never resolve; the loader rejects what would crash at runtime.
    _write_config(
        tmp_path / "config",
        agents_yaml=_TEAM_AGENTS,
        tasks_yaml="""
            _process: hierarchical
            _manager: lead
            answer:
              description: 'd'
              expected_output: 'o'
              agent: lead
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="must not be assigned to the manager"):
        loader.load_config(tmp_path / "config")


def test_sequential_task_without_agent_is_error(tmp_path: Path) -> None:
    # Back-compat: outside hierarchical, every task still requires an agent
    # (a sequential CrewAI crew rejects agentless tasks at construction).
    _write_config(
        tmp_path / "config",
        agents_yaml=_TEAM_AGENTS,
        tasks_yaml="""
            answer:
              description: 'd'
              expected_output: 'o'
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="'agent' is required"):
        loader.load_config(tmp_path / "config")


def test_manager_with_tools_is_error(tmp_path: Path) -> None:
    # CrewAI raises "Manager agent should not have tools" at kickoff; the
    # loader rejects it at load with a clearer message.
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            lead:
              role: 'Lead'
              goal: 'g'
              backstory: 'b'
              tools:
                - web_search
            qa:
              role: 'QA'
              goal: 'g'
              backstory: 'b'
        """,
        tasks_yaml="""
            _process: hierarchical
            _manager: lead
            answer:
              description: 'd'
              expected_output: 'o'
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="must not have tools"):
        loader.load_config(tmp_path / "config")


def test_hierarchical_needs_a_member_besides_the_manager(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml="""
            lead:
              role: 'Lead'
              goal: 'g'
              backstory: 'b'
        """,
        tasks_yaml="""
            _process: hierarchical
            _manager: lead
            answer:
              description: 'd'
              expected_output: 'o'
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="at least one member agent"):
        loader.load_config(tmp_path / "config")


def test_hierarchical_build_crew_constructs_with_manager_lifted_out(tmp_path: Path, monkeypatch) -> None:
    # The runtime contract, verified against the pinned crewai wheel: the
    # manager becomes Crew.manager_agent, is excluded from agents= (CrewAI
    # rejects it there), and the unassigned task is left for delegation.
    pytest.importorskip("crewai")
    # API drift in crewai 1.9.x: an Agent's default LLM constructs eagerly
    # and requires OPENAI_API_KEY at Crew construction (older pins deferred
    # the check to call time). No call is made — a placeholder suffices.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-construction-only")
    _write_config(
        tmp_path / "config",
        agents_yaml=_TEAM_AGENTS,
        tasks_yaml="""
            _process: hierarchical
            _manager: lead
            answer:
              description: 'Delegate and answer {prompt}.'
              expected_output: 'A routed answer.'
        """,
    )
    config = loader.load_config(tmp_path / "config")
    crew = loader.build_crew(config, tools=[])  # must not raise
    assert crew.manager_agent is not None
    assert crew.manager_agent.role == "Lead"
    assert {a.role for a in crew.agents} == {"QA", "Architect"}
    assert crew.tasks[0].agent is None
    # name= reaches the real Task, which mirrors it onto TaskOutput —
    # what a task_callback receives — rather than leaving it to fall back to
    # the task's own (seller-authored, unbounded) description.
    assert crew.tasks[0].name == "answer"


def test_demo_marketing_strategist_loads(tmp_path: Path) -> None:
    """End-to-end: the shipped demo persona must parse and validate."""
    demo_config = CREWS / "marketing-strategist" / "config"
    config = loader.load_config(demo_config)
    assert len(config.agents) == 1
    assert config.agents[0].name == "senior_marketing_strategist"
    assert config.tool_refs == ["serper_search"]
    assert config.process == "sequential"


_TEAM_DEMO_CONFIG = CREWS / "software-team" / "config"


def test_demo_software_team_crew_loads_as_hierarchical() -> None:
    """End-to-end: the shipped delegating demo crew must parse and validate."""
    config = loader.load_config(_TEAM_DEMO_CONFIG)
    assert config.process == "hierarchical"
    assert config.manager == "team_lead"
    assert {a.name for a in config.agents} == {"team_lead", "qa_engineer", "software_architect", "product_designer"}
    # One unassigned task: the manager delegates it to the member it chooses.
    assert [t.agent for t in config.tasks] == [""]


_LAUNCH_DEMO_CONFIG = CREWS / "product-launch" / "config"


def test_demo_product_launch_crew_is_the_tooled_sequential_flagship() -> None:
    """The S6.3 flagship shape: live research + sequential context
    chaining + deliverable files — output a pasted prompt cannot produce."""
    config = loader.load_config(_LAUNCH_DEMO_CONFIG)
    assert config.process == "sequential"
    assert config.tool_refs == ["serper_search", "file_write"]
    # Every downstream task consumes upstream output; the final assembly
    # task sees the whole chain.
    by_name = {t.name: t for t in config.tasks}
    assert by_name["write_positioning"].context == ["draft_launch_plan"]
    assert set(by_name["assemble_launch_brief"].context) == {
        "draft_launch_plan",
        "write_positioning",
        "pick_channels",
        "define_metrics",
    }
    # The coordinator owns the file deliverables.
    coordinator = next(a for a in config.agents if a.name == "launch_coordinator")
    assert coordinator.tools == ["file_write"]
    assert "launch_brief.md" in by_name["assemble_launch_brief"].description


def test_demo_product_launch_crew_constructs_with_real_tools(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("crewai")
    # crewai 1.9.x validates OPENAI_API_KEY at construction; see the drift
    # note on test_hierarchical_build_crew_constructs_with_manager_lifted_out.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-construction-only")
    from sigrix_runtime import sandbox  # noqa: PLC0415 — imported where the stub is in place

    config = loader.load_config(_LAUNCH_DEMO_CONFIG)
    tools = sandbox.build_tools(config.tool_refs, workspace=tmp_path / "ws")
    crew = loader.build_crew(config, tools)  # must not raise
    assert len(crew.agents) == 5
    assert len(crew.tasks) == 5


def test_demo_software_team_crew_constructs_a_real_hierarchical_crew(monkeypatch) -> None:
    pytest.importorskip("crewai")
    # crewai 1.9.x validates OPENAI_API_KEY at construction; see the drift
    # note on test_hierarchical_build_crew_constructs_with_manager_lifted_out.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-construction-only")
    config = loader.load_config(_TEAM_DEMO_CONFIG)
    crew = loader.build_crew(config, tools=[])  # must not raise
    assert crew.manager_agent is not None
    assert crew.manager_agent.role == "Engineering Team Lead"
    assert {a.role for a in crew.agents} == {"Senior QA Engineer", "Software Architect", "Product Designer"}
    assert crew.tasks[0].agent is None


# ---------------------------------------------------------------------------
# Fan-out / parallel panel rules
# ---------------------------------------------------------------------------

_PANEL_AGENTS = """
    qa:
      role: 'QA Specialist'
      goal: 'Find defects.'
      backstory: 'A QA engineer.'
    architect:
      role: 'Architect'
      goal: 'Judge the design.'
      backstory: 'A systems architect.'
    chair:
      role: 'Panel Chair'
      goal: 'Merge the answers.'
      backstory: 'The synthesizer.'
"""


def test_parallel_panel_loads(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_PANEL_AGENTS,
        tasks_yaml="""
            _process: parallel
            qa_answer:
              description: 'Review {prompt} for defects.'
              expected_output: 'A defect list.'
              agent: qa
              async_execution: true
            architect_answer:
              description: 'Review {prompt} for design.'
              expected_output: 'A design verdict.'
              agent: architect
              async_execution: true
            synthesize:
              description: 'Merge both answers to {prompt}.'
              expected_output: 'One combined review.'
              agent: chair
              context: [qa_answer, architect_answer]
        """,
    )
    config = loader.load_config(tmp_path / "config")
    assert config.process == "parallel"
    assert [t.async_execution for t in config.tasks] == [True, True, False]


def test_parallel_without_any_async_task_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_PANEL_AGENTS,
        tasks_yaml="""
            _process: parallel
            only:
              description: 'Do {prompt}.'
              expected_output: 'An answer.'
              agent: qa
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="parallel requires at least one async_execution"):
        loader.load_config(tmp_path / "config")


def test_trailing_async_task_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_PANEL_AGENTS,
        tasks_yaml="""
            first:
              description: 'Do {prompt}.'
              expected_output: 'An answer.'
              agent: qa
            last:
              description: 'Also do {prompt}.'
              expected_output: 'An answer.'
              agent: architect
              async_execution: true
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="final task must not set async_execution"):
        loader.load_config(tmp_path / "config")


def test_async_task_with_context_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_PANEL_AGENTS,
        tasks_yaml="""
            first:
              description: 'Do {prompt}.'
              expected_output: 'An answer.'
              agent: qa
            second:
              description: 'Continue {prompt}.'
              expected_output: 'An answer.'
              agent: architect
              context: [first]
              async_execution: true
            final:
              description: 'Wrap up {prompt}.'
              expected_output: 'An answer.'
              agent: chair
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="must not declare context"):
        loader.load_config(tmp_path / "config")


def test_non_boolean_async_execution_is_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "config",
        agents_yaml=_PANEL_AGENTS,
        tasks_yaml="""
            only:
              description: 'Do {prompt}.'
              expected_output: 'An answer.'
              agent: qa
              async_execution: 'yes please'
        """,
    )
    with pytest.raises(loader.BundleConfigError, match="async_execution must be a boolean"):
        loader.load_config(tmp_path / "config")
