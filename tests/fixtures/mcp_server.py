"""A stdio MCP server for the runner's MCP mode, standard library only.

It speaks what a client that lists and calls tools needs: ``initialize``,
``notifications/initialized``, ``tools/list`` (paginated when asked) and
``tools/call``. Its tools cover each shape the bridge maps -- a string,
numbers, an enum, a boolean, a nested object only a text box can take, a tool
error (``isError``), a JSON-RPC error, markdown, a non-text block -- plus
``environment``, which reports which runner settings reached it.

Misbehaviour is chosen on the command line, so a test names the failure it
is about: ``--exit-before-initialize``, ``--exit-on-call``, ``--refuse-calls``
(every call answered with invalid params), ``--noise`` (a
line on stdout that is not JSON before every answer), ``--ping`` (asks the
client a ``ping`` before answering ``tools/call``), ``--protocol VERSION``,
``--page-size N``, ``--hang-on-call``, ``--single`` (offers ``echo`` alone),
``--stderr-lines N`` and ``--pid-file PATH``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROTOCOL_VERSION = "2025-06-18"

#: The settings a runner holds, which only a launcher may be handed.
REPORTED_ENV = ("SIGRIX_TOKEN", "POSTERN_DISTRIBUTOR", "POSTERN_AGENT_ID", "POSTERN_INBOUND_TOKEN", "FIXTURE_MARK")

TOOLS = [
    {
        "name": "echo",
        "title": "Echo",
        "description": "Repeat the text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "What to repeat.", "maxLength": 200}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number", "minimum": -1000}, "b": {"type": "integer", "maximum": 1000}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "greet",
        "description": "Greet someone in a chosen style.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "style": {"type": "string", "enum": ["formal", "casual"], "default": "casual"},
                "shout": {"type": "boolean", "default": False},
            },
            "required": ["name"],
        },
    },
    {
        "name": "lookup",
        "description": "Look records up by a filter object.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filter": {"type": "object", "properties": {"city": {"type": "string"}}},
                "fields": {"type": "array", "items": {"type": "string"}},
                "note": {},
            },
            "required": ["filter"],
        },
    },
    {
        "name": "report",
        "description": "Return a markdown report.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "fail",
        "description": "Always report a tool error.",
        "inputSchema": {"type": "object", "properties": {"reason": {"type": "string"}}},
    },
    {
        "name": "crash",
        "description": "Always answer with a JSON-RPC error.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "picture",
        "description": "Return an image block beside the text.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "environment",
        "description": "Report which runner settings reached this server.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

REPORT = "# Report\n\n- one\n- two\n"


def _write(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def _call(name: str, arguments: dict) -> dict:
    if name == "echo":
        return {"content": [_text(str(arguments.get("text", "")))]}
    if name == "add":
        return {"content": [_text(json.dumps(arguments["a"] + arguments["b"]))]}
    if name == "greet":
        greeting = "Good day" if arguments.get("style") == "formal" else "Hi"
        line = f"{greeting}, {arguments.get('name', '')}"
        return {"content": [_text(line.upper() if arguments.get("shout") is True else line)]}
    if name == "lookup":
        return {"content": [_text(json.dumps(arguments, sort_keys=True))]}
    if name == "report":
        return {"content": [_text(REPORT)]}
    if name == "fail":
        return {"content": [_text(arguments.get("reason") or "the tool failed")], "isError": True}
    if name == "picture":
        return {
            "content": [_text("see the picture"), {"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"}]
        }
    if name == "environment":
        return {"content": [_text(json.dumps({key: os.environ.get(key) for key in REPORTED_ENV}))]}
    raise LookupError(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-before-initialize", action="store_true")
    parser.add_argument("--exit-on-call", action="store_true")
    parser.add_argument("--refuse-calls", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--ping", action="store_true")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--page-size", type=int, default=0)
    parser.add_argument("--hang-on-call", action="store_true")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--stderr-lines", type=int, default=0)
    parser.add_argument("--pid-file", default="")
    options = parser.parse_args(argv)
    tools = [tool for tool in TOOLS if tool["name"] == "echo"] if options.single else TOOLS

    if options.pid_file:
        Path(options.pid_file).write_text(str(os.getpid()), encoding="utf-8")
    for index in range(options.stderr_lines):
        sys.stderr.write(f"fixture log line {index}\n")
    sys.stderr.flush()
    if options.exit_before_initialize:
        sys.stderr.write("fixture: refusing to start\n")
        return 3

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        message = json.loads(raw)
        method = message.get("method")
        if "id" not in message or method is None:
            continue  # a notification, or the client's answer to a ping
        if options.noise:
            sys.stdout.write("this line is not JSON\n")
            sys.stdout.flush()
        request_id = message["id"]
        params = message.get("params") or {}
        if method == "initialize":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": options.protocol or params.get("protocolVersion") or PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "postern-fixture", "version": "1.0.0"},
                    },
                }
            )
        elif method == "tools/list":
            start = int(params.get("cursor") or 0)
            size = options.page_size or len(tools)
            result: dict = {"tools": tools[start : start + size]}
            if start + size < len(tools):
                result["nextCursor"] = str(start + size)
            _write({"jsonrpc": "2.0", "id": request_id, "result": result})
        elif method == "tools/call":
            if options.hang_on_call:
                continue
            if options.refuse_calls:
                error = {"code": -32602, "message": "Invalid params: rejected by the fixture"}
                _write({"jsonrpc": "2.0", "id": request_id, "error": error})
                continue
            if options.exit_on_call:
                sys.stderr.write("fixture: exiting mid-call\n")
                sys.stderr.flush()
                return 5
            if options.ping:
                _write({"jsonrpc": "2.0", "id": "fixture-ping", "method": "ping"})
                answer = json.loads(sys.stdin.readline())
                if answer.get("id") != "fixture-ping" or "result" not in answer:
                    _write({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": "no pong"}})
                    continue
            name = params.get("name")
            if name == "crash":
                _write({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": "the server broke"}})
                continue
            try:
                result = _call(name, params.get("arguments") or {})
            except LookupError:
                _write(
                    {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
                )
                continue
            _write({"jsonrpc": "2.0", "id": request_id, "result": result})
        else:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
