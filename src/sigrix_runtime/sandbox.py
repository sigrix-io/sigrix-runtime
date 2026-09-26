"""File-IO sandbox and tool builders.

The default ``file_read`` and ``file_write`` tools are confined to a
single workspace directory passed in at construction time. Path
traversal (via ``..`` or absolute paths) is rejected with a
``SandboxViolationError`` naming the path the agent tried to use.

The sandbox functions ``safe_read_file`` and ``safe_write_file`` are
independently testable without crewai installed. The ``build_tools``
factory wires them into CrewAI ``@tool`` decorators at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SandboxViolationError(PermissionError):
    """Raised when a tool tries to access a path outside its workspace."""


def _resolve_inside(workspace: Path, requested: str) -> Path:
    workspace_root = workspace.resolve()
    candidate = (workspace / requested).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise SandboxViolationError(f"path {requested!r} resolves outside the workspace ({workspace_root})") from exc
    return candidate


def safe_read_file(workspace: Path, relative_path: str) -> str:
    target = _resolve_inside(workspace, relative_path)
    if not target.exists():
        raise FileNotFoundError(f"file not found: {relative_path}")
    if not target.is_file():
        raise IsADirectoryError(f"not a regular file: {relative_path}")
    return target.read_text(encoding="utf-8")


def safe_write_file(workspace: Path, relative_path: str, content: str) -> int:
    target = _resolve_inside(workspace, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return len(content)


def build_tools(tool_refs: list[str], workspace: Path) -> list[Any]:
    """Construct CrewAI tool objects for the given tool refs.

    Imports crewai lazily so that callers who only need ``safe_read_file``
    or ``safe_write_file`` (e.g. unit tests) do not pull in crewai.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    tools: list[Any] = []
    for ref in tool_refs:
        if ref == "file_read":
            tools.append(_make_file_read_tool(workspace))
        elif ref == "file_write":
            tools.append(_make_file_write_tool(workspace))
        elif ref == "serper_search":
            from crewai_tools import SerperDevTool

            tools.append(SerperDevTool())
        elif ref == "web_search":
            from crewai_tools import WebsiteSearchTool

            tools.append(WebsiteSearchTool())
        else:
            raise ValueError(f"unsupported tool: {ref!r}")
    return tools


def _make_file_read_tool(workspace: Path) -> Any:
    from crewai.tools import tool

    @tool("file_read")
    def file_read(path: str) -> str:
        """Read a UTF-8 text file from the bundle's workspace/ directory."""
        return safe_read_file(workspace, path)

    return file_read


def _make_file_write_tool(workspace: Path) -> Any:
    from crewai.tools import tool

    @tool("file_write")
    def file_write(path: str, content: str) -> str:
        """Write content to a file inside the bundle's workspace/ directory."""
        n = safe_write_file(workspace, path, content)
        return f"wrote {n} bytes to {path}"

    return file_write
