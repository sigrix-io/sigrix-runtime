"""Tests for the Sigrix starter file-IO sandbox.

The sandbox confines file_read/file_write tools to a single workspace
directory. These tests exercise the safe_* helpers directly without
requiring crewai to be installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sigrix_runtime import sandbox  # noqa: E402


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def test_safe_write_then_read_roundtrip(workspace: Path) -> None:
    sandbox.safe_write_file(workspace, "notes.txt", "hello")
    assert sandbox.safe_read_file(workspace, "notes.txt") == "hello"


def test_safe_write_creates_subdirectories(workspace: Path) -> None:
    sandbox.safe_write_file(workspace, "nested/deep/file.txt", "x")
    assert (workspace / "nested" / "deep" / "file.txt").read_text(encoding="utf-8") == "x"


def test_safe_write_reports_byte_count(workspace: Path) -> None:
    written = sandbox.safe_write_file(workspace, "a.txt", "abcd")
    assert written == 4


def test_read_missing_file_raises_filenotfound(workspace: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sandbox.safe_read_file(workspace, "missing.txt")


def test_read_directory_raises_isadirectoryerror(workspace: Path) -> None:
    (workspace / "subdir").mkdir()
    with pytest.raises(IsADirectoryError):
        sandbox.safe_read_file(workspace, "subdir")


def test_dotdot_traversal_is_rejected_on_read(workspace: Path) -> None:
    outside = workspace.parent / "secrets.txt"
    outside.write_text("sensitive", encoding="utf-8")
    with pytest.raises(sandbox.SandboxViolationError):
        sandbox.safe_read_file(workspace, "../secrets.txt")


def test_dotdot_traversal_is_rejected_on_write(workspace: Path) -> None:
    with pytest.raises(sandbox.SandboxViolationError):
        sandbox.safe_write_file(workspace, "../escape.txt", "x")


def test_absolute_path_is_rejected(workspace: Path) -> None:
    with pytest.raises(sandbox.SandboxViolationError):
        sandbox.safe_read_file(workspace, "/etc/passwd")


def test_symlink_to_outside_is_rejected(workspace: Path, tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_text("sensitive", encoding="utf-8")
    link = workspace / "shortcut.txt"
    link.symlink_to(target)
    with pytest.raises(sandbox.SandboxViolationError):
        sandbox.safe_read_file(workspace, "shortcut.txt")


def test_symlink_inside_workspace_is_allowed(workspace: Path) -> None:
    real = workspace / "real.txt"
    real.write_text("ok", encoding="utf-8")
    link = workspace / "alias.txt"
    link.symlink_to(real)
    assert sandbox.safe_read_file(workspace, "alias.txt") == "ok"


def test_violation_error_message_names_the_path(workspace: Path) -> None:
    with pytest.raises(sandbox.SandboxViolationError) as excinfo:
        sandbox.safe_read_file(workspace, "../something.txt")
    assert "../something.txt" in str(excinfo.value)


def test_build_tools_rejects_unsupported_tool(workspace: Path) -> None:
    with pytest.raises(ValueError, match="unsupported tool"):
        sandbox.build_tools(["shell_exec"], workspace=workspace)


def test_build_tools_creates_workspace_if_missing(tmp_path: Path) -> None:
    ws = tmp_path / "does-not-exist-yet"
    sandbox.build_tools([], workspace=ws)
    assert ws.exists()
