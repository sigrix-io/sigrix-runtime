"""One run, in its own process. Not a public entry point.

Reads a JSON request on stdin, runs the bundle through
``sigrix_runtime.execution`` — the same function ``python main.py`` calls —
and writes JSON Lines to stdout as it goes::

    {"event": "notice", "text": "…"}
    {"event": "step",   "name": "…", "latency_ms": 1234}
    {"event": "done",   "text": "…", "files_written": [...], ...}
    {"event": "error",  "code": "agent_error", "message": "…"}

**Why a process rather than a thread.** Three properties the server owes
its clients are only obtainable by having something it can kill:

* ``limits.max_run_seconds`` (SPEC 4.4, 4.5) must not be "a bound longer
  than the shortest one it can actually enforce". A thread running inside
  a model call cannot be stopped from outside, so a threaded runner can
  declare no limit honestly at all.
* "A disconnected client is a cancelled run" (SPEC 4.5). The agent stops
  because the process stops, rather than at whatever checkpoint it happens
  to reach.
* ``describe`` and ``status`` **MUST** answer at Level 1 regardless (SPEC
  3, 4.1, 4.4). Keeping crewai's import out of the server process is what
  lets a bundle whose virtualenv is half-built still say what it is and
  what is wrong with it.

Everything this writes on stdout is a protocol frame, which is why the
run's own asides travel as ``notice`` events rather than being printed:
anything else on this stream would corrupt it.

A request carrying ``mcp`` is an MCP server's rather than the bundle's, and
:func:`sigrix_runtime.postern.mcp.work` answers it in the same frames.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from sigrix_runtime.postern.errors import requirements_not_installed


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
    except ValueError as exc:
        _emit({"event": "error", "code": "agent_error", "message": f"unreadable run request: {exc}"})
        return 2
    if not isinstance(request, dict):
        _emit({"event": "error", "code": "agent_error", "message": "run request was not an object"})
        return 2

    if isinstance(request.get("mcp"), dict):
        from sigrix_runtime.postern import mcp

        return mcp.work(request["mcp"], _emit)

    bundle_root = Path(str(request.get("bundle_root") or ".")).resolve()
    prompt = str(request.get("prompt") or "")
    variables = request.get("variables")
    variables = variables if isinstance(variables, dict) else {}

    try:
        from sigrix_runtime import execution
    except ModuleNotFoundError as exc:  # pragma: no cover - the engine imports it first
        _emit({"event": "error", "code": "unavailable", "message": requirements_not_installed(str(exc.name)).message})
        return 3

    try:
        outcome = execution.execute(
            bundle_root,
            prompt,
            variables=variables,
            on_step=lambda step: _emit({"event": "step", **step.as_dict()}),
            on_notice=lambda note: _emit({"event": "notice", "text": note}),
        )
    except execution.MissingDependencyError as exc:
        # Not an agent failure: the agent never ran (see `requirements_not_installed`).
        _emit({"event": "error", "code": "unavailable", "message": requirements_not_installed(exc.module).message})
        return 3
    except BaseException as exc:  # noqa: BLE001 - the agent's failure is the payload
        _emit(
            {
                "event": "error",
                "code": "agent_error",
                "message": f"{type(exc).__name__}: {exc}".strip(),
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    _emit(
        {
            "event": "done",
            "text": outcome.text,
            "files_written": outcome.files_written,
            "steps": [step.as_dict() for step in outcome.steps],
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "model_id": outcome.model_id,
            "duration_seconds": outcome.duration_seconds,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
