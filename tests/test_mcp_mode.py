"""The runner's MCP mode: a stdio MCP server's tools, served as Postern.

``python -m sigrix_runtime.postern --mcp`` starts a Sigrix-delivered MCP
server, generates ``describe`` from ``tools/list`` and turns ``run`` into
``tools/call``. Everything here runs against
``tests/fixtures/mcp_server.py``, a real server on a real pipe:

- ``describe`` validates against the Postern schemas this platform serves;
- ``run`` returns the tool's result as text, markdown included;
- tool errors map to Postern's codes, each envelope valid.

It also holds the promises the mode makes beyond the issue's three: the MCP
path needs no PyYAML (the runner is standard library only); only the launcher
is handed the runner's token; seller code is started in the worker, so a
departed client and a timeout end the server too; and the two spellings of
the launcher's command, this runner's and the platform's, agree.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest

from sigrix_runtime.postern import PATH_PREFIX, mcp  # noqa: E402
from sigrix_runtime.postern import __main__ as cli  # noqa: E402
from sigrix_runtime.postern import entitlement as ent  # noqa: E402
from sigrix_runtime.postern.engine import Limits  # noqa: E402
from sigrix_runtime.postern.errors import PosternError  # noqa: E402
from sigrix_runtime.postern.server import PosternServer, Runner, RunnerConfig, build_runner  # noqa: E402
from tests import postern_distributor as fake  # noqa: E402
from tests import postern_schema  # noqa: E402
from tests.support import RUNTIME_ROOT

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mcp_server.py"
distributor = fake.distributor

SERVER = (sys.executable, str(FIXTURE))
AGENT_ID = "acme/tools"


def _fixture_tools() -> list[dict[str, Any]]:
    """The fixture's own tool list, read rather than restated."""
    spec = importlib.util.spec_from_file_location("mcp_server", FIXTURE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TOOLS


def _toolbox(tools: list[dict[str, Any]], **listing: Any) -> mcp.Toolbox:
    return mcp.Toolbox({"tools": tools, "server": {"name": "fx", "version": "1.0.0"}, **listing}, agent_id=AGENT_ID)


def _inputs(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {declaration["key"]: declaration for declaration in document["inputs"]}


class _Client:
    def __init__(self, port: int) -> None:
        self.port = port

    def json(self, method: str, path: str, payload: Any = None) -> tuple[int, dict[str, Any]]:
        status, raw = self.raw(method, path, payload)
        return status, json.loads(raw)

    def raw(self, method: str, path: str, payload: Any = None) -> tuple[int, str]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            body = json.dumps(payload).encode("utf-8") if payload is not None else None
            headers = {"Content-Type": "application/json"} if body is not None else {}
            connection.request(method, PATH_PREFIX + path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read().decode("utf-8")
        finally:
            connection.close()

    def run(self, inputs: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return self.json("POST", "/run", {"inputs": inputs})


@contextmanager
def _serve(
    folder: Path,
    *args: str,
    launcher: bool = False,
    gate: ent.Entitlement | None = None,
    limits: Limits | None = None,
) -> Iterator[_Client]:
    config = RunnerConfig(
        bundle_root=folder,
        port=0,
        limits=limits or Limits(),
        mcp=mcp.McpServer(command=(*SERVER, *args), launcher=launcher),
    )
    runner = Runner(config, gate or ent.Entitlement(base_url="", token="", agent_id=""))
    runner.load_tools(timeout=60)
    server = PosternServer(config, runner)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield _Client(server.server_address[1])
    finally:
        runner.engine.stop_all()
        server.shutdown()
        server.server_close()


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[_Client]:
    with _serve(tmp_path) as served:
        yield served


# ---------------------------------------------------------------------------
# describe, generated from tools/list
# ---------------------------------------------------------------------------


def test_describe_validates_against_the_postern_schema(client: _Client) -> None:
    status, body = client.json("GET", "/describe")

    assert status == 200
    postern_schema.validate(body, postern_schema.load("describe"), name="describe")
    assert body["agent"]["name"] == "postern-fixture"
    assert body["agent"]["version"] == "1.0.0"
    assert body["agent"]["id"] == mcp.local_agent_id(SERVER)
    assert body["capabilities"]["tools"] == [tool["name"] for tool in _fixture_tools()]
    assert body["output"] == {"type": "text"}


def test_status_answers_for_an_mcp_runner(client: _Client) -> None:
    status, body = client.json("GET", "/status")

    assert status == 200
    postern_schema.validate(body, postern_schema.load("status"), name="status")
    assert body["state"] == "ready"
    assert body["credentials"] == {"satisfied": True, "missing": []}


def test_the_form_is_a_choice_of_tool_then_each_tools_inputs() -> None:
    document = _toolbox(_fixture_tools()).document
    tool = document["inputs"][0]

    assert tool["key"] == mcp.TOOL_INPUT_KEY
    # Named in the document, so a client draws the picker without a constant of its own.
    assert document["org.sigrix"]["mcp"]["tool_input"] == mcp.TOOL_INPUT_KEY
    assert (tool["type"], tool["required"], tool["default"]) == ("select", True, "echo")
    assert tool["validation"]["options"] == [t["name"] for t in _fixture_tools()]
    # Every argument keyed under its tool, in the tool's own order.
    keys = [declaration["key"] for declaration in document["inputs"][1:]]
    assert keys[:3] == ["echo.text", "add.a", "add.b"]
    assert all(key.split(".", 1)[0] in tool["validation"]["options"] for key in keys)


def test_simple_types_map_to_text_number_and_select_and_the_rest_to_a_text_box() -> None:
    inputs = _inputs(_toolbox(_fixture_tools()).document)

    assert (inputs["echo.text"]["type"], inputs["echo.text"]["validation"]) == ("text", {"max_length": 200})
    assert (inputs["add.a"]["type"], inputs["add.a"]["validation"]) == ("number", {"min": -1000})
    assert (inputs["add.b"]["type"], inputs["add.b"]["validation"]) == ("number", {"max": 1000})
    assert inputs["greet.style"]["type"] == "select"
    assert (inputs["greet.style"]["validation"]["options"], inputs["greet.style"]["default"]) == (
        ["formal", "casual"],
        "casual",
    )
    assert (inputs["greet.shout"]["type"], inputs["greet.shout"]["default"]) == ("select", "false")
    assert inputs["greet.shout"]["validation"]["options"] == ["true", "false"]
    for richer in ("lookup.filter", "lookup.fields", "lookup.note"):
        assert inputs[richer]["type"] == "text"
        assert inputs[richer]["label"].endswith("(JSON)")


def test_with_several_tools_the_chosen_tools_required_rides_in_the_extension() -> None:
    """Required per tool cannot be said per input when the form holds every tool."""
    inputs = _inputs(_toolbox(_fixture_tools()).document)

    assert inputs["echo.text"]["required"] is False
    assert inputs["echo.text"]["org.sigrix"] == {
        "mcp": {"tool": "echo", "argument": "text", "required": True, "kind": "text", "description": "What to repeat."}
    }
    assert inputs["lookup.filter"]["org.sigrix"]["mcp"]["kind"] == "json"
    assert inputs["greet.style"]["org.sigrix"]["mcp"]["required"] is False


def test_a_single_tools_required_is_the_inputs_own() -> None:
    echo = [tool for tool in _fixture_tools() if tool["name"] == "echo"]
    document = _toolbox(echo).document

    postern_schema.validate(document, postern_schema.load("describe"), name="describe")
    assert _inputs(document)["echo.text"]["required"] is True
    assert document["agent"]["summary"] == "An MCP server offering 1 tool."


def test_a_tool_is_a_write_tool_unless_its_server_marks_it_read_only() -> None:
    capabilities = _toolbox(_fixture_tools()).document["capabilities"]

    assert set(capabilities["tools"]) - set(capabilities["write_tools"]) == {"echo", "report"}


def test_the_input_key_grammar_is_the_schemas() -> None:
    """A mirrored constant: SPEC 4.1.1's grammar, as the served schema spells it."""
    schema = postern_schema.load("describe")

    assert mcp._KEY.pattern == schema["$defs"]["input"]["properties"]["key"]["pattern"]


def test_an_argument_that_cannot_be_keyed_is_left_out_and_said_so() -> None:
    tools = [
        {"name": "fine", "inputSchema": {"type": "object", "properties": {"ok": {"type": "string"}}}},
        {
            "name": "partly",
            "inputSchema": {"type": "object", "properties": {"ok": {"type": "string"}, "not ok": {"type": "string"}}},
        },
        {
            "name": "broken",
            "inputSchema": {"type": "object", "properties": {"not ok": {"type": "string"}}, "required": ["not ok"]},
        },
    ]
    toolbox = _toolbox(tools)

    assert toolbox.names == ["fine", "partly"]
    assert [d["key"] for d in toolbox.document["inputs"][1:]] == ["fine.ok", "partly.ok"]
    assert len(toolbox.warnings) == 2
    assert "'broken' is left out" in toolbox.warnings[1]
    postern_schema.validate(toolbox.document, postern_schema.load("describe"), name="describe")


def test_two_tools_that_would_share_an_input_key_do_not() -> None:
    """``a.b``'s ``c`` and ``a``'s ``b.c`` are both ``a.b.c``: the second is left out, not merged."""
    tools = [
        {"name": "a.b", "inputSchema": {"type": "object", "properties": {"c": {"type": "string"}}}},
        {
            "name": "a",
            "inputSchema": {"type": "object", "properties": {"b.c": {"type": "string"}}, "required": ["b.c"]},
        },
    ]
    toolbox = _toolbox(tools)

    assert toolbox.names == ["a.b"]
    keys = [d["key"] for d in toolbox.document["inputs"]]
    assert keys == ["tool", "a.b.c"]
    assert toolbox.warnings == ["the MCP tool 'a' is left out: its argument 'b.c' cannot be an input key."]


# ---------------------------------------------------------------------------
# run, as tools/call
# ---------------------------------------------------------------------------


def test_run_returns_the_tools_text(client: _Client) -> None:
    status, body = client.run({"tool": "echo", "echo.text": "hello"})

    assert status == 200
    postern_schema.validate(body, postern_schema.load("run-response"), name="run")
    assert body["output"] == {"type": "text", "value": "hello"}
    assert [step["name"] for step in body["usage"]["steps"]] == ["echo"]


def test_run_returns_markdown_as_the_text_it_is(client: _Client) -> None:
    status, body = client.run({"tool": "report"})

    assert status == 200
    assert body["output"]["value"] == "# Report\n\n- one\n- two\n"


def test_inputs_reach_the_tool_as_the_types_its_schema_asks_for(client: _Client) -> None:
    assert client.run({"tool": "add", "add.a": 1.5, "add.b": 2.0})[1]["output"]["value"] == "3.5"
    greeting = client.run({"tool": "greet", "greet.name": "Ada", "greet.style": "formal", "greet.shout": "true"})
    assert greeting[1]["output"]["value"] == "GOOD DAY, ADA"
    looked_up = client.run(
        {"tool": "lookup", "lookup.filter": '{"city": "Paris"}', "lookup.fields": '["a"]', "lookup.note": "plain"}
    )
    assert json.loads(looked_up[1]["output"]["value"]) == {
        "fields": ["a"],
        "filter": {"city": "Paris"},
        "note": "plain",
    }


def test_a_result_text_cannot_carry_is_named_rather_than_dropped(client: _Client) -> None:
    status, body = client.run({"tool": "picture"})

    assert status == 200
    assert body["output"]["value"] == "see the picture\n\n[image/png, not shown]"


def test_structured_content_stands_in_when_no_text_came_with_it() -> None:
    assert mcp.output_text({"content": [], "structuredContent": {"temperature": 21}}) == '{\n  "temperature": 21\n}'
    assert mcp.output_text({"content": [{"type": "text", "text": "21"}], "structuredContent": {"t": 21}}) == "21"


def test_stream_emits_start_a_step_and_done(client: _Client) -> None:
    status, raw = client.raw("POST", "/stream", {"inputs": {"tool": "echo", "echo.text": "streamed"}})

    assert status == 200
    events = [
        (frame.split("\n")[0].removeprefix("event: "), json.loads(frame.split("\n")[1].removeprefix("data: ")))
        for frame in raw.strip().split("\n\n")
    ]
    assert [name for name, _ in events] == ["start", "step", "done"]
    for name, payload in events[:2]:
        postern_schema.validate(payload, postern_schema.load("stream-event"), name=name)
    postern_schema.validate(events[2][1], postern_schema.load("run-response"), name="done")
    assert events[2][1]["output"]["value"] == "streamed"


# ---------------------------------------------------------------------------
# Tool errors, as Postern's codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "status", "code", "said"),
    [
        # The tool's own error report: the agent failed, in its words.
        ({"tool": "fail", "fail.reason": "no upstream"}, 500, "agent_error", "no upstream"),
        # A JSON-RPC error from the server.
        ({"tool": "crash"}, 500, "agent_error", "the server broke"),
        # The chosen tool's own required argument, checked before anything starts.
        ({"tool": "echo"}, 400, "bad_request", "(input 'echo.text')"),
        # A tool nobody offered.
        ({"tool": "nope"}, 400, "bad_request", "(input 'tool')"),
        # Types the schema asked for.
        ({"tool": "add", "add.a": 1, "add.b": 2.5}, 400, "bad_request", "whole number"),
        ({"tool": "lookup", "lookup.filter": "not json"}, 400, "bad_request", "must be JSON"),
        ({"tool": "lookup", "lookup.filter": "[1]"}, 400, "bad_request", "must be a JSON object"),
        ({"tool": "echo", "echo.text": "x" * 201}, 400, "bad_request", "200 characters"),
    ],
)
def test_each_failure_maps_to_a_postern_code(
    client: _Client, inputs: dict[str, Any], status: int, code: str, said: str
) -> None:
    answered, body = client.run(inputs)

    assert (answered, body["error"]["code"]) == (status, code)
    assert said in body["error"]["message"]
    postern_schema.validate(body, postern_schema.load("error"), name="error")


def test_the_server_refusing_the_arguments_is_the_requests_fault(tmp_path: Path) -> None:
    """JSON-RPC's invalid params (-32602) is ``400``: the arguments, or a tool it no longer offers."""
    with _serve(tmp_path, "--refuse-calls") as served:
        status, body = served.run({"tool": "echo", "echo.text": "hi"})

    assert (status, body["error"]["code"]) == (400, "bad_request")
    assert body["error"]["message"] == "The MCP server refused echo: Invalid params: rejected by the fixture"
    postern_schema.validate(body, postern_schema.load("error"), name="error")


def test_a_server_that_exits_mid_call_is_an_agent_error_naming_what_it_said(tmp_path: Path) -> None:
    with _serve(tmp_path, "--exit-on-call") as served:
        status, body = served.run({"tool": "echo", "echo.text": "hi"})

    assert (status, body["error"]["code"]) == (500, "agent_error")
    assert "exited before it answered tools/call" in body["error"]["message"]
    assert "fixture: exiting mid-call" in body["error"]["message"]


def test_a_call_past_the_runners_bound_is_a_timeout(tmp_path: Path) -> None:
    started = time.monotonic()
    with _serve(tmp_path, "--hang-on-call", limits=Limits(max_run_seconds=1)) as served:
        status, body = served.run({"tool": "echo", "echo.text": "hi"})

    assert (status, body["error"]["code"]) == (504, "run_timeout")
    assert time.monotonic() - started < 15


# ---------------------------------------------------------------------------
# Reading the tools at boot
# ---------------------------------------------------------------------------


def _runner(folder: Path, *args: str) -> Runner:
    config = RunnerConfig(bundle_root=folder, port=0, mcp=mcp.McpServer(command=(*SERVER, *args)))
    return Runner(config, ent.Entitlement(base_url="", token="", agent_id=""))


def test_a_server_that_will_not_start_says_why_at_boot(tmp_path: Path) -> None:
    with pytest.raises(PosternError) as refused:
        _runner(tmp_path, "--exit-before-initialize").load_tools(timeout=60)

    assert refused.value.code == "unavailable"
    assert "exited before it answered initialize (exit status 3)" in refused.value.message
    assert "fixture: refusing to start" in refused.value.message


def test_a_server_on_a_protocol_this_runner_does_not_speak_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PosternError) as refused:
        _runner(tmp_path, "--protocol", "1999-01-01").load_tools(timeout=60)

    assert refused.value.code == "unavailable"
    assert "'1999-01-01'" in refused.value.message


def test_describe_before_the_tools_are_read_is_unavailable_rather_than_empty(tmp_path: Path) -> None:
    with pytest.raises(PosternError) as refused:
        _runner(tmp_path).describe()

    assert refused.value.code == "unavailable"


def test_the_client_pages_answers_a_ping_and_skips_what_is_not_mcp() -> None:
    """A paginated list, a server asking ``ping`` mid-call, and noise on stdout."""
    with mcp.StdioMcpClient([*SERVER, "--page-size", "2", "--ping", "--noise", "--stderr-lines", "3"]) as client:
        tools = client.list_tools()
        result = client.call_tool("echo", {"text": "hi"})

    assert [tool["name"] for tool in tools] == [tool["name"] for tool in _fixture_tools()]
    assert result == {"content": [{"type": "text", "text": "hi"}]}
    assert client.protocol_version == mcp.PROTOCOL_VERSION
    assert list(client.stderr) == ["fixture log line 0", "fixture log line 1", "fixture log line 2"]


# ---------------------------------------------------------------------------
# Who is handed what
# ---------------------------------------------------------------------------


def _reached(served: _Client) -> dict[str, str | None]:
    status, body = served.run({"tool": "environment"})
    assert status == 200, body
    return json.loads(body["output"]["value"])


@pytest.fixture()
def runner_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "SIGRIX_TOKEN": "from-the-environment",
        "POSTERN_AGENT_ID": AGENT_ID,
        "POSTERN_INBOUND_TOKEN": "inbound",
        "FIXTURE_MARK": "ordinary",
    }.items():
        monkeypatch.setenv(name, value)


def _active(distributor) -> ent.Entitlement:
    served = distributor(lambda path: (200, {"Content-Type": "application/json"}, fake.check_body(agent_id=AGENT_ID)))
    return ent.Entitlement(base_url=served.base_url, token="runner-token", agent_id=AGENT_ID)


def test_the_launcher_alone_is_handed_the_runners_token_and_distributor(
    tmp_path: Path, runner_settings: None, distributor
) -> None:
    gate = _active(distributor)
    with _serve(tmp_path, launcher=True, gate=gate) as served:
        reached = _reached(served)

    assert reached == {
        "SIGRIX_TOKEN": "runner-token",
        "POSTERN_DISTRIBUTOR": gate.base_url,
        "POSTERN_AGENT_ID": None,
        "POSTERN_INBOUND_TOKEN": None,
        "FIXTURE_MARK": "ordinary",
    }


def test_a_server_named_by_its_command_is_handed_no_runner_setting(
    tmp_path: Path, runner_settings: None, distributor
) -> None:
    """The same runner, holding the same token, as the launcher test: only the command differs."""
    with _serve(tmp_path, gate=_active(distributor)) as served:
        reached = _reached(served)

    assert reached == {
        "SIGRIX_TOKEN": None,
        "POSTERN_DISTRIBUTOR": None,
        "POSTERN_AGENT_ID": None,
        "POSTERN_INBOUND_TOKEN": None,
        "FIXTURE_MARK": "ordinary",
    }


def test_a_run_still_passes_the_entitlement_gate_first(tmp_path: Path, distributor) -> None:
    revoked = distributor(
        lambda path: (200, {"Content-Type": "application/json"}, fake.check_body("revoked", agent_id=AGENT_ID))
    )
    gate = ent.Entitlement(base_url=revoked.base_url, token="runner-token", agent_id=AGENT_ID)
    with _serve(tmp_path, launcher=True, gate=gate) as served:
        status, body = served.run({"tool": "echo", "echo.text": "hi"})

    assert (status, body["error"]["code"]) == (403, "not_entitled")


def test_an_mcp_runner_names_itself_a_local_connector(tmp_path: Path) -> None:
    config = RunnerConfig(bundle_root=tmp_path, port=0, mcp=mcp.McpServer(command=SERVER))

    assert build_runner(config, {}).entitlement.delivery_mode == ent.DELIVERY_MODE_CONNECTOR_LOCAL
    operator = build_runner(
        RunnerConfig(bundle_root=tmp_path, port=0, mcp=mcp.McpServer(command=SERVER)),
        {"SIGRIX_DELIVERY_MODE": "container"},
    )
    assert operator.entitlement.delivery_mode == "container"
    bundle = build_runner(RunnerConfig(bundle_root=tmp_path, port=0), {})
    assert bundle.entitlement.delivery_mode == ent.DELIVERY_MODE_ZIP


# ---------------------------------------------------------------------------
# Seller code ends with its run
# ---------------------------------------------------------------------------


def _gone(pid: int, *, within: float) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
        except OSError:
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="reads process state from /proc")
def test_a_client_that_leaves_stops_the_mcp_server_too(tmp_path: Path) -> None:
    pid_file = tmp_path / "server.pid"
    with _serve(tmp_path, "--hang-on-call", "--pid-file", str(pid_file)) as served:
        # The listing started the server once already; this run starts it again.
        pid_file.unlink()
        body = json.dumps({"inputs": {"tool": "echo", "echo.text": "hi"}}).encode("utf-8")
        sock = socket.create_connection(("127.0.0.1", served.port), timeout=10)
        head = f"POST {PATH_PREFIX}/run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        sock.sendall(head.encode("ascii") + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        deadline = time.monotonic() + 20
        while not pid_file.exists() or not pid_file.read_text(encoding="utf-8"):
            assert time.monotonic() < deadline, "the run never started the server"
            time.sleep(0.05)
        pid = int(pid_file.read_text(encoding="utf-8"))
        assert not _gone(pid, within=0.3)
        sock.close()

        assert _gone(pid, within=5), "the MCP server ran on for a client that had left"


# ---------------------------------------------------------------------------
# Standard library only
# ---------------------------------------------------------------------------


def test_an_mcp_runner_needs_no_pyyaml(tmp_path: Path) -> None:
    """The bundle path imports PyYAML on its first run; this one must not at all.

    A ``yaml`` that raises on import sits first on every process's path -- the
    runner's, its worker's and the server's -- and the runner starts with
    site-packages removed. The canary shows the poison takes.
    """
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "yaml.py").write_text('raise ImportError("an MCP runner imported PyYAML")\n', encoding="utf-8")
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("POSTERN_", "SIGRIX_"))},
        "PYTHONPATH": os.pathsep.join((str(poison), str(RUNTIME_ROOT))),
        "NO_PROXY": "127.0.0.1,localhost",
    }
    canary = subprocess.run([sys.executable, "-c", "import yaml"], env=env, capture_output=True, text=True, timeout=60)
    assert canary.returncode != 0 and "an MCP runner imported PyYAML" in canary.stderr

    argv = [sys.executable, "-S", "-m", "sigrix_runtime.postern", "--mcp", "--port", "0", "--bundle", str(tmp_path)]
    process = subprocess.Popen([*argv, "--", *SERVER], env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout is not None
        announced = process.stdout.readline()
        assert announced.startswith("POSTERN_PORT="), announced
        served = _Client(int(announced.split("=", 1)[1]))
        assert served.json("GET", "/describe")[0] == 200
        status, body = served.run({"tool": "echo", "echo.text": "stdlib"})
        assert (status, body["output"]["value"]) == (200, "stdlib")
    finally:
        process.terminate()
        process.wait(timeout=10)


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_runner_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SIGRIX_TOKEN", "POSTERN_AGENT_ID", "POSTERN_DISTRIBUTOR", "POSTERN_INBOUND_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> tuple[int, RunnerConfig | None]:
    seen: dict[str, RunnerConfig] = {}
    monkeypatch.setattr(cli, "serve", lambda config: seen.__setitem__("config", config))
    return cli.main(argv), seen.get("config")


def test_a_named_command_is_served_as_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_runner_settings: None
) -> None:
    code, config = _main(monkeypatch, ["--mcp", "--bundle", str(tmp_path), "--", *SERVER, "--single"])

    assert code == cli.EXIT_OK and config is not None
    assert config.mcp == mcp.McpServer(command=(*SERVER, "--single"), launcher=False)
    assert config.update_check is None  # the launcher updates on every start; there is no bundle to compare


@pytest.mark.skipif(sys.platform == "win32", reason="stands a shell script in for uvx")
def test_a_listing_is_served_through_the_launcher_with_its_licence_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_runner_settings: None, distributor
) -> None:
    served = distributor(lambda path: (200, {"Content-Type": "application/json"}, fake.check_body(agent_id=AGENT_ID)))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uvx").write_text("#!/bin/sh\n", encoding="utf-8")
    (bin_dir / "uvx").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("SIGRIX_TOKEN", "runner-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)

    code, config = _main(monkeypatch, ["--mcp", AGENT_ID, "--bundle", str(tmp_path)])

    assert code == cli.EXIT_OK and config is not None
    assert config.mcp == mcp.McpServer(command=("uvx", "sigrix-launcher", "run", AGENT_ID), launcher=True)
    assert config.agent_id == AGENT_ID
    assert [path for path in served.paths if "/entitlements/" in path], "the boot never checked the licence"
    assert not [path for path in served.paths if "/bundles/" in path or "/versions/" in path]
    modes = {headers.get("x-sigrix-delivery-mode") for _, headers in served.exchanges}
    assert modes == {ent.DELIVERY_MODE_CONNECTOR_LOCAL}


@pytest.mark.skipif(sys.platform == "win32", reason="stands a shell script in for uvx")
def test_a_listing_refused_at_boot_is_not_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_runner_settings: None, distributor
) -> None:
    served = distributor(
        lambda path: (200, {"Content-Type": "application/json"}, fake.check_body("revoked", agent_id=AGENT_ID))
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uvx").write_text("#!/bin/sh\n", encoding="utf-8")
    (bin_dir / "uvx").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("SIGRIX_TOKEN", "runner-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)

    code, config = _main(monkeypatch, ["--mcp", AGENT_ID, "--bundle", str(tmp_path)])

    assert (code, config) == (cli.EXIT_REFUSED, None)


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        (["--mcp"], "--mcp serves a listing named as AGENT or POSTERN_AGENT_ID"),
        (["--", "python", "server.py"], "add --mcp"),
        (["--mcp", "--", "definitely-not-a-command-6219"], "'definitely-not-a-command-6219' was not found on PATH"),
    ],
)
def test_what_mcp_mode_cannot_start_is_a_configuration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_runner_settings: None,
    argv: list[str],
    said: str,
) -> None:
    code, config = _main(monkeypatch, ["--bundle", str(tmp_path), *argv])

    assert (code, config) == (cli.EXIT_NOT_CONFIGURED, None)
    assert said in capsys.readouterr().err


def test_a_listing_with_no_token_is_refused_before_the_launcher_is_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], no_runner_settings: None
) -> None:
    code, config = _main(monkeypatch, ["--mcp", AGENT_ID, "--bundle", str(tmp_path)])

    assert (code, config) == (cli.EXIT_NOT_CONFIGURED, None)
    assert "needs SIGRIX_TOKEN" in capsys.readouterr().err


def test_a_launcher_that_is_not_installed_says_where_uv_comes_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], no_runner_settings: None
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("SIGRIX_TOKEN", "runner-token")

    code, _ = _main(monkeypatch, ["--mcp", AGENT_ID, "--bundle", str(tmp_path)])

    assert code == cli.EXIT_NOT_CONFIGURED
    assert "'uvx' was not found on PATH. The launcher runs through uv" in capsys.readouterr().err


def test_a_server_that_fails_at_boot_exits_unavailable_with_its_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], no_runner_settings: None
) -> None:
    code = cli.main(["--mcp", "--port", "0", "--bundle", str(tmp_path), "--", *SERVER, "--exit-before-initialize"])

    assert code == cli.EXIT_UNAVAILABLE
    assert "fixture: refusing to start" in capsys.readouterr().err
