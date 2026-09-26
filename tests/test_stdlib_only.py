"""The runner's "zero dependencies beyond the standard library" claim, held to.

``sigrix_runtime/postern/__init__.py`` states it outright — *"Zero dependencies
beyond the standard library"* — and ``entitlement.py``, ``describe.py`` and three
docs repeat it. It was false: ``describe.py`` imported ``sigrix_runtime.execution``
at module scope, which reaches ``loader``/``workforce`` and ``import yaml``, so

    python3 -S -c "import sigrix_runtime.postern.server"

died on ``ModuleNotFoundError: No module named 'yaml'``. The import was then
made lazy. Nothing kept it that way, which is what this file is for.

**Why it is worth a guard rather than a comment.** The break is invisible where
anyone would notice it. The published container ships PyYAML, and so does every
developer machine with the test requirements installed, so a module-scope
``from sigrix_runtime import execution`` reintroduced tomorrow passes the whole
suite and every image smoke test. It bites only the bare-runner path — a buyer
running the bundle on a machine with nothing installed, which is exactly the
audience the claim is addressed to.

It is also load-bearing for this package, which is published on the strength
of the framework-neutral half being genuinely framework-neutral; a false claim
here is a public one.

Two checks, deliberately overlapping. The subprocess reproduces the issue's own
command and proves the whole import graph; the AST sweep names the offending line
instead of a traceback, and catches a boundary crossing in a module the subprocess
tests do not happen to import.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import RUNTIME_ROOT

POSTERN = RUNTIME_ROOT / "sigrix_runtime" / "postern"

# `-S` drops site-packages, which is what makes this an isolation rather than a
# preference. PYTHONPATH carries the one tree under test and nothing else, and the
# working directory is somewhere neutral so that an implicit `''` on sys.path
# cannot quietly re-add this repository.
ISOLATED = (sys.executable, "-S", "-c")


def _import_in_isolation(statement: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*ISOLATED, statement],
        cwd=cwd,
        env={"PYTHONPATH": str(RUNTIME_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_isolation_actually_isolates(tmp_path: Path) -> None:
    """The canary. Without it every other test here could pass vacuously.

    These tests assert that importing the runner under `-S` succeeds. That
    assertion is only evidence of anything if a third-party import under the same
    flags *fails* — otherwise `-S` has stopped removing site-packages, the
    subprocess is importing from the ambient environment, and a reintroduced
    `import yaml` would sail through looking exactly like this.

    PyYAML is the right probe because it is the dependency the claim once broke
    on, and the dev extra guarantees it is installed here — so a failure of this
    test means the isolation broke, never that the package is absent.
    """
    assert _import_in_isolation("import yaml", tmp_path).returncode != 0, (
        "`python -S` imported PyYAML, so site-packages is still on the path and "
        "every other test in this file proves nothing. Fix the isolation."
    )

    ambient = subprocess.run([sys.executable, "-c", "import yaml"], capture_output=True, text=True, timeout=60)
    assert ambient.returncode == 0, (
        "PyYAML is not installed at all, so the check above passed for the wrong reason. Install requirements-test.txt."
    )


@pytest.mark.parametrize(
    "module",
    [
        "sigrix_runtime.postern",
        "sigrix_runtime.postern.server",
        "sigrix_runtime.postern.describe",
        "sigrix_runtime.postern.entitlement",
        "sigrix_runtime.postern.__main__",
    ],
)
def test_the_runner_imports_with_nothing_installed(module: str, tmp_path: Path) -> None:
    """The original reproduction, one module per case.

    `describe` and `server` are the two that broke — they answer `describe`
    and `status`, the Level 1 verbs the claim is about. `__main__` is here because
    it is what a buyer actually runs, and importing it pulls the rest.
    """
    result = _import_in_isolation(f"import {module}", tmp_path)

    assert result.returncode == 0, (
        f"{module} does not import with site-packages removed, so the runner's "
        f'"zero dependencies beyond the standard library" claim is false again.\n'
        f"{result.stderr.strip()}"
    )


def test_no_module_scope_import_crosses_the_boundary() -> None:
    """Every module-scope import in `postern/` is stdlib or a `postern` sibling.

    This is the same rule the subprocess proves, read off the source instead of
    the interpreter, which buys two things: the failure names the file and line
    rather than handing over a traceback from a subprocess, and it holds for
    modules no test above happens to import.

    The allowance is narrow on purpose. `sigrix_runtime.postern.*` is the package
    itself and subject to this same rule; `sigrix_runtime.execution`,
    `.configuration`, `.loader` and `.workforce` are the framework half that pulls
    PyYAML, and they are reached lazily inside functions — `describe._derived_config`
    and `engine._spawn` are the pattern to copy.
    """
    stdlib = sys.stdlib_module_names
    offenders: list[str] = []
    imports_seen = 0

    modules = sorted(POSTERN.glob("*.py"))
    assert modules, f"found no modules under {POSTERN}; this sweep checked nothing"

    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module scope only — a nested import is the fix
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative: a sibling by construction
                    continue
                names = [node.module or ""]
            else:
                continue

            for name in names:
                imports_seen += 1
                root = name.split(".")[0]
                if root in stdlib or name.startswith("sigrix_runtime.postern"):
                    continue
                offenders.append(f"{path.name}:{node.lineno}  {name}")

    assert imports_seen, "parsed no module-scope imports; the sweep is not reading the tree"
    assert not offenders, (
        "module-scope imports in postern/ that are neither standard library nor a "
        "postern sibling — each one breaks the bare-runner path:\n  " + "\n  ".join(offenders)
    )
