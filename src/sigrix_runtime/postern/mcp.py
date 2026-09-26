"""The runner's MCP mode: ``describe`` from ``tools/list``, ``run`` as ``tools/call``.

:class:`Toolbox` is the server process's half, :func:`work` the worker's: the
MCP server is started only in the worker, so it ends the way a crew's run
does.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sigrix_runtime.postern import POSTERN_VERSION
from sigrix_runtime.postern.describe import LOCAL_ID_OWNER
from sigrix_runtime.postern.errors import PosternError, bad_request

PROTOCOL_VERSION = "2025-06-18"
# Revisions whose tools/list and tools/call read the same; others are refused.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# The listing's own launch. Mirrors ``mcp_launcher_config``; a test holds the pair.
LAUNCHER_COMMAND = ("uvx", "sigrix-launcher", "run")

TOOL_INPUT_KEY = "tool"
INVALID_PARAMS = -32602
LIST_TIMEOUT_SECONDS = 600  # a first start installs the server
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_STDERR_LINES = 20
_KEY = re.compile(r"^[a-zA-Z0-9_.-]+$")  # SPEC 4.1.1's input key


class McpError(Exception):
    """The server could not be started, or did not answer as MCP."""


class McpRpcError(McpError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class McpServer:
    """How to start the server. ``launcher`` hands it this runner's distributor settings."""

    command: tuple[str, ...]
    launcher: bool = False


def launcher_command(agent_id: str) -> tuple[str, ...]:
    return (*LAUNCHER_COMMAND, agent_id)


def local_agent_id(command: Sequence[str]) -> str:
    """A server no distributor named: stable per command, and visibly local."""
    return f"{LOCAL_ID_OWNER}/{hashlib.sha256(json.dumps(list(command)).encode('utf-8')).hexdigest()[:12]}"


# --- The client ------------------------------------------------------------


class StdioMcpClient:
    """One session with one server, for a ``with``. No capabilities, and no clock of its own."""

    def __init__(self, command: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
        self._command = [str(part) for part in command]
        self._env = dict(env) if env is not None else None
        self._ids = itertools.count(1)
        self._inbox: queue.Queue[Any] = queue.Queue()
        self._process: subprocess.Popen[bytes] | None = None
        self._threads: list[threading.Thread] = []
        self.stderr: collections.deque[str] = collections.deque(maxlen=_STDERR_LINES)
        self.server_info: dict[str, Any] = {}
        self.protocol_version = ""

    def __enter__(self) -> StdioMcpClient:
        path = (self._env if self._env is not None else os.environ).get("PATH")
        executable = shutil.which(self._command[0], path=path) if self._command else None
        if executable is None:
            raise McpError(f"The MCP server command {(self._command or [''])[0]!r} was not found on PATH.")
        try:
            self._process = subprocess.Popen(  # noqa: S603 - the operator's own command, no shell
                [executable, *self._command[1:]],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env,
            )
        except OSError as exc:
            raise McpError(f"The MCP server could not be started: {exc}") from exc
        for target in (self._read_stdout, self._drain_stderr):
            thread = threading.Thread(target=target, args=(self._process,), daemon=True)
            thread.start()
            self._threads.append(thread)
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """The specification's shutdown: close its input, wait, then escalate."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        for stop in (None, process.terminate, process.kill):
            try:
                if stop is not None:
                    stop()
                process.wait(timeout=2)
                break
            except (OSError, subprocess.TimeoutExpired):
                continue
        for thread in self._threads:
            thread.join(timeout=1)

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            result = self._request("tools/list", {"cursor": cursor} if cursor else {})
            if not isinstance(result.get("tools"), list):
                raise McpError("The MCP server answered tools/list without a list of tools.")
            tools.extend(tool for tool in result["tools"] if isinstance(tool, dict))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
            if not isinstance(cursor, str) or cursor in seen:
                raise McpError("The MCP server repeated a tools/list page.")
            seen.add(cursor)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("tools/call", {"name": name, "arguments": dict(arguments)})

    def _initialize(self) -> None:
        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "postern-sigrix-runner", "version": POSTERN_VERSION},
            },
        )
        version = result.get("protocolVersion")
        if version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise McpError(f"The MCP server speaks protocol {version!r}, which this runner does not.")
        self.protocol_version = str(version)
        info = result.get("serverInfo")
        self.server_info = info if isinstance(info, dict) else {}
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, message: Mapping[str, Any]) -> None:
        process = self._process
        try:
            assert process is not None and process.stdin is not None
            process.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
            process.stdin.flush()
        except (AssertionError, OSError) as exc:
            raise self._exited("stopped reading its input") from exc

    def _request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        request_id = next(self._ids)
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)})
        while True:
            message = self._inbox.get()
            if message is None or isinstance(message, McpError):
                self._inbox.put(message)  # every later request meets it too
                if message is None:
                    raise self._exited(f"exited before it answered {method}")
                raise message
            if "method" in message:  # the server's own request, or a notification
                if "id" in message:
                    ping = message["method"] == "ping"
                    answer = {"result": {}} if ping else {"error": {"code": -32601, "message": "Method not found"}}
                    self._send({"jsonrpc": "2.0", "id": message["id"], **answer})
                continue
            if message.get("id") != request_id:
                continue
            error = message.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                raise McpRpcError(code if isinstance(code, int) else -32603, str(error.get("message") or "error"))
            result = message.get("result")
            if not isinstance(result, dict):
                raise McpError(f"The MCP server answered {method} without a result.")
            return result

    def _exited(self, what: str) -> McpError:
        process = self._process
        code = None
        if process is not None:
            try:
                code = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        for thread in self._threads[1:]:
            thread.join(timeout=1)  # the lines a dying server writes last say why
        said = next((line for line in reversed(self.stderr) if line.strip()), "")
        status = f" (exit status {code})" if code is not None else ""
        return McpError(f"The MCP server {what}{status}." + (f" It said: {said}" if said else ""))

    def _read_stdout(self, process: subprocess.Popen[bytes]) -> None:
        stream = process.stdout
        assert stream is not None
        try:
            while line := stream.readline(MAX_MESSAGE_BYTES + 1):
                if len(line) > MAX_MESSAGE_BYTES:
                    self._inbox.put(McpError("The MCP server sent a message larger than this runner reads."))
                    return
                try:
                    message = json.loads(line)
                except ValueError:
                    continue  # not MCP: the server printed to the wrong stream
                if isinstance(message, dict) and message.get("jsonrpc") == "2.0":
                    self._inbox.put(message)
        except (OSError, ValueError):
            pass
        finally:
            self._inbox.put(None)

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        stream = process.stderr
        assert stream is not None
        try:
            while line := stream.readline(64 * 1024):
                self.stderr.append(line.decode("utf-8", "replace").rstrip()[:400])
        except (OSError, ValueError):
            pass


# --- The worker's half -------------------------------------------------------


def work(request: Mapping[str, Any], emit: Callable[[dict[str, Any]], None]) -> int:
    """Answer one worker request: list the tools, or call one."""
    env = {**os.environ, **{str(k): str(v) for k, v in (request.get("env") or {}).items()}}
    client = StdioMcpClient(request.get("command") or [], env=env)
    listing = bool(request.get("list"))
    name = str(request.get("tool") or "")
    calling = False
    try:
        with client:
            if listing:
                answer = {
                    "tools": client.list_tools(),
                    "server": client.server_info,
                    "protocol": client.protocol_version,
                }
            else:
                calling = True
                started = time.monotonic()
                result = client.call_tool(name, request.get("arguments") or {})
                step = {"name": name, "latency_ms": int((time.monotonic() - started) * 1000)}
    except McpRpcError as exc:
        if not calling:
            emit(
                {"event": "error", "code": "unavailable", "message": f"The MCP server could not list its tools: {exc}"}
            )
        elif exc.code == INVALID_PARAMS:  # the arguments or the name: the request's fault
            emit({"event": "error", "code": "bad_request", "message": f"The MCP server refused {name}: {exc}"})
        else:
            emit({"event": "error", "code": "agent_error", "message": f"The MCP server could not run {name}: {exc}"})
        return 1
    except McpError as exc:
        emit(
            {
                "event": "error",
                "code": "agent_error" if calling else "unavailable",
                "message": str(exc),
                "traceback": "\n".join(client.stderr),
            }
        )
        return 3
    if listing:
        emit({"event": "done", **answer})
        return 0
    emit({"event": "step", **step})
    text = output_text(result)
    if result.get("isError") is True:
        emit({"event": "error", "code": "agent_error", "message": text or f"The tool {name} reported an error."})
        return 1
    emit({"event": "done", "text": text})
    return 0


def output_text(result: Mapping[str, Any]) -> str:
    """A ``tools/call`` result as text: what text cannot carry is named, not dropped."""
    parts: list[str] = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        raw_resource = block.get("resource")
        resource = raw_resource if isinstance(raw_resource, dict) else {}
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind == "resource" and isinstance(resource.get("text"), str):
            parts.append(resource["text"])
        elif kind in ("resource", "resource_link"):
            parts.append(f"[{block.get('name') or resource.get('uri') or block.get('uri') or 'a resource'}, not shown]")
        else:
            parts.append(f"[{block.get('mimeType') or kind or 'content'}, not shown]")
    if not any(parts) and "structuredContent" in result:
        parts = [json.dumps(result["structuredContent"], indent=2, ensure_ascii=False)]
    return "\n\n".join(part for part in parts if part)


# --- The server process's half ----------------------------------------------


@dataclass(frozen=True)
class _Param:
    key: str
    name: str
    label: str
    kind: str  # text, number, integer, boolean, enum or json
    required: bool
    choices: tuple[Any, ...] = ()
    json_types: frozenset[str] = frozenset()


class Toolbox:
    """A server's ``tools/list`` as a ``describe`` document, and a run's inputs back as a call."""

    def __init__(self, listing: Mapping[str, Any], *, agent_id: str) -> None:
        self.warnings: list[str] = []
        self._tools: dict[str, list[_Param]] = {}
        self._keys: set[str] = set()
        specs: dict[str, list[dict[str, Any]]] = {}
        described: list[dict[str, Any]] = []
        read_only: set[str] = set()
        for tool in listing.get("tools") or []:
            name = tool.get("name") if isinstance(tool, dict) else None
            if not isinstance(name, str) or not name or name in self._tools:
                continue
            pairs = self._params(name, tool.get("inputSchema"))
            if pairs is None:
                continue
            self._tools[name] = [param for param, _ in pairs]
            specs[name] = [spec for _, spec in pairs]
            annotations = tool.get("annotations")
            if isinstance(annotations, dict) and annotations.get("readOnlyHint") is True:
                read_only.add(name)
            described.append(
                {key: tool[key] for key in ("name", "title", "description") if isinstance(tool.get(key), str)}
            )
        self.names = names = list(self._tools)
        single = len(names) == 1
        inputs: list[dict[str, Any]] = [
            {
                "key": TOOL_INPUT_KEY,
                "label": "Tool",
                "type": "select",
                "required": True,
                "default": names[0] if names else None,
                "validation": {"options": names},
            }
        ]
        for tool_name, params in self._tools.items():
            inputs.extend(
                _declare(tool_name, p, spec, single) for p, spec in zip(params, specs[tool_name], strict=True)
            )
        raw_server = listing.get("server")
        server = raw_server if isinstance(raw_server, dict) else {}
        title, version = server.get("title") or server.get("name"), server.get("version")
        self.document: dict[str, Any] = {
            "postern": POSTERN_VERSION,
            "agent": {
                "id": agent_id,
                "name": title if isinstance(title, str) and title else "MCP server",
                "version": version if isinstance(version, str) and version else "0",
                "summary": f"An MCP server offering {len(names)} tool{'' if single else 's'}.",
            },
            "inputs": inputs,
            "output": {"type": "text"},
            # A write tool unless marked read-only: a hint the server may omit.
            "capabilities": {"tools": names, "write_tools": [n for n in names if n not in read_only]},
            "credentials": [],
            "org.sigrix": {
                "mcp": {
                    "protocol": str(listing.get("protocol") or ""),
                    "tool_input": TOOL_INPUT_KEY,
                    "tools": described,
                }
            },
        }

    def call_for(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """The ``tools/call`` a validated ``inputs`` map asks for (``400`` if it cannot)."""
        name = str(inputs.get(TOOL_INPUT_KEY) or "")
        if name not in self._tools:
            raise bad_request(f"Choose a tool: one of {', '.join(self.names)}.", {"key": TOOL_INPUT_KEY})
        arguments: dict[str, Any] = {}
        for param in self._tools[name]:
            value = inputs.get(param.key)
            if value is None or (isinstance(value, str) and not value.strip()):
                if param.required:
                    raise _refuse(param, "is required")
                continue
            arguments[param.name] = _convert(param, value)
        return {"tool": name, "arguments": arguments}

    def _params(self, tool: str, schema: Any) -> list[tuple[_Param, dict[str, Any]]] | None:
        schema = schema if isinstance(schema, dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = {n for n in schema.get("required") or [] if isinstance(n, str)}
        pairs: list[tuple[_Param, dict[str, Any]]] = []
        for name, spec in properties.items():
            key = f"{tool}.{name}"
            if not _KEY.fullmatch(key) or key in self._keys:  # ``a.b``'s ``c`` is ``a``'s ``b.c``
                if name in required:
                    self.warnings.append(
                        f"the MCP tool {tool!r} is left out: its argument {name!r} cannot be an input key."
                    )
                    return None
                self.warnings.append(
                    f"the MCP tool {tool!r} is offered without {name!r}, which cannot be an input key."
                )
                continue
            spec = spec if isinstance(spec, dict) else {}
            kind, json_types = argument_kind(spec)
            title = spec.get("title")
            label = title if isinstance(title, str) and title.strip() else _humanize(name)
            choices = tuple(spec["enum"]) if kind == "enum" else (True, False) if kind == "boolean" else ()
            pairs.append((_Param(key, name, label, kind, name in required, choices, json_types), spec))
        self._keys.update(param.key for param, _ in pairs)
        return pairs


def _declare(tool: str, param: _Param, spec: Mapping[str, Any], single: bool) -> dict[str, Any]:
    default = spec.get("default")
    declaration: dict[str, Any] = {
        "key": param.key,
        "label": param.label + (" (JSON)" if param.kind == "json" else ""),
        "type": {"number": "number", "integer": "number", "boolean": "select", "enum": "select"}.get(
            param.kind, "text"
        ),
        "required": param.required and single,
        "default": None,
    }
    validation: dict[str, Any] = {}
    if param.choices:
        validation["options"] = [_option(choice) for choice in param.choices]
        if default is not None and _option(default) in validation["options"]:
            declaration["default"] = _option(default)
    elif param.kind in ("number", "integer"):
        if _is_number(default):
            declaration["default"] = default
        for ours, theirs in (("min", "minimum"), ("max", "maximum")):
            if _is_number(spec.get(theirs)):
                validation[ours] = spec[theirs]
    elif param.kind == "text":
        declaration["default"] = default if isinstance(default, str) else None
        if isinstance(spec.get("maxLength"), int) and not isinstance(spec.get("maxLength"), bool):
            validation["max_length"] = spec["maxLength"]
    elif default is not None:
        declaration["default"] = json.dumps(default)
    if validation:
        declaration["validation"] = validation
    extension: dict[str, Any] = {"tool": tool, "argument": param.name, "required": param.required, "kind": param.kind}
    if isinstance(spec.get("description"), str):
        extension["description"] = spec["description"]
    declaration["org.sigrix"] = {"mcp": extension}
    return declaration


def argument_kind(spec: Mapping[str, Any]) -> tuple[str, frozenset[str]]:
    """``text``, ``number``, ``integer``, ``boolean``, ``enum`` or ``json``, and the declared types."""
    declared = spec.get("type")
    listed = declared if isinstance(declared, list) else [declared]
    types = frozenset(t for t in listed if isinstance(t, str) and t != "null")
    enum = spec.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(v, (str, int, float, bool)) for v in enum):
        return "enum", types
    if len(types) == 1 and next(iter(types)) in ("string", "number", "integer", "boolean"):
        return {"string": "text"}.get(next(iter(types)), next(iter(types))), types
    return "json", types


def _convert(param: _Param, value: Any) -> Any:
    if param.choices:
        for choice in param.choices:
            if _option(choice) == value:
                return choice
        raise _refuse(param, f"must be one of: {', '.join(_option(choice) for choice in param.choices)}")
    if param.kind in ("number", "integer") and not math.isfinite(value):
        raise _refuse(param, "must be a finite number")  # JSON has no NaN to send
    if param.kind == "integer":
        if isinstance(value, float) and not value.is_integer():
            raise _refuse(param, "must be a whole number")
        return int(value)
    if param.kind != "json":
        return value
    try:
        parsed = json.loads(value)
    except ValueError:
        if param.json_types and "string" not in param.json_types:
            raise _refuse(param, "must be JSON") from None
        return value
    containers = {"object": dict, "array": list}
    if param.json_types and param.json_types <= containers.keys():
        if not isinstance(parsed, tuple(containers[t] for t in param.json_types)):
            raise _refuse(param, f"must be a JSON {' or '.join(sorted(param.json_types))}")
    return parsed


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _refuse(param: _Param, complaint: str) -> PosternError:
    return bad_request(f"{param.label} {complaint} (input '{param.key}').", {"key": param.key})


def _option(value: Any) -> str:
    return json.dumps(value) if isinstance(value, bool) else str(value)


def _humanize(name: str) -> str:
    words = re.sub(r"[_\-.]+", " ", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)).strip()
    return words[:1].upper() + words[1:].lower() if words else name


__all__ = [
    "LAUNCHER_COMMAND",
    "PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "McpError",
    "McpRpcError",
    "McpServer",
    "StdioMcpClient",
    "Toolbox",
    "argument_kind",
    "launcher_command",
    "local_agent_id",
    "output_text",
    "work",
]
