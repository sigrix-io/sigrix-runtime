"""The runner's four verbs over a real socket.

``python -m sigrix_runtime.postern`` inside a bundle folder is one of the
runner's two deployment shapes, and the ``sigrix/runner`` image is the other,
running the same module. What is asserted here is the
protocol: the shape of each payload against the specification's own
schemas, and the handful of rules that are easy to satisfy on the happy path
and get wrong everywhere else.

The bundle under test is real — the shipped starter tree plus a demo's
``config/`` — and so is the run: the engine spawns
``sigrix_runtime.postern._worker`` as a subprocess exactly as it does on a
buyer's machine, and the only thing standing in is crewai itself, whose
absence would otherwise make every run answer "install your requirements".
Stubbing the provider rather than the plumbing is what keeps this a test of
the runner instead of a test of a mock.
"""

from __future__ import annotations

import inspect
import json
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest

from sigrix_runtime.postern import CONFORMANCE_LEVEL, PATH_PREFIX  # noqa: E402
from sigrix_runtime.postern import describe as describe_module  # noqa: E402
from sigrix_runtime.postern import entitlement as ent  # noqa: E402
from sigrix_runtime.postern import server as server_module  # noqa: E402
from sigrix_runtime.postern.engine import Limits  # noqa: E402
from sigrix_runtime.postern.server import PosternServer, Runner, RunnerConfig  # noqa: E402
from tests import postern_schema  # noqa: E402
from tests.support import BUNDLE_VERSION, CREW_CONFIG, RUNTIME_ROOT

# A crew whose members never call a model. ``kickoff`` fires the crew's own
# task callback per task with a ``TaskOutput``-shaped object — the
# real crewai never hands the callback the raw ``Task`` — and returns an
# object carrying crewai's token totals. Each ``Agent`` resolves an ``llm``
# eagerly the way the real ``Agent.post_init_setup`` validator does, and
# ``Crew.calculate_usage_metrics()`` answers the running per-agent totals
# ``execution.py`` samples mid-run to build a step's token delta. The three
# per-task deltas below sum to exactly ``_Usage``'s totals, mirroring the
# real invariant: crewai's own final ``result.token_usage`` is the same
# running total ``calculate_usage_metrics()`` reports along the way.
_CREWAI_STUB = """
import os as _os

_TASK_TOKEN_DELTAS = [(2000, 400), (1500, 350), (710, 168)]


class Process:
    sequential = "sequential"
    hierarchical = "hierarchical"


class LLM:
    def __init__(self, model=None):
        self.model = model or _os.environ.get("STUB_AGENT_MODEL", "gpt-4o-mini")


class Agent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.llm = LLM()


class Task:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.name = kwargs.get("name")


class _Usage:
    prompt_tokens = 4210
    completion_tokens = 918


class _Output:
    token_usage = _Usage()

    def __str__(self):
        return "## Positioning brief\\n\\nThe mid-market segment..."


class _RunningUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _TaskOutput:
    def __init__(self, task):
        self.name = getattr(task, "name", None)
        self.description = getattr(task, "description", "") or ""
        agent = getattr(task, "agent", None)
        self.agent = getattr(agent, "role", "") if agent is not None else ""


class Crew:
    task_callback = None

    def __init__(self, **kwargs):
        self.tasks = kwargs.get("tasks") or []
        self.agents = kwargs.get("agents") or []
        self.manager_agent = kwargs.get("manager_agent")
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._task_index = 0

    def calculate_usage_metrics(self):
        return _RunningUsage(self._prompt_tokens, self._completion_tokens)

    def kickoff(self, inputs=None):
        import json as _json
        import pathlib as _pathlib

        _pathlib.Path("kickoff_inputs.json").write_text(_json.dumps(inputs or {}))
        for task in self.tasks:
            delta = _TASK_TOKEN_DELTAS[self._task_index % len(_TASK_TOKEN_DELTAS)]
            self._task_index += 1
            self._prompt_tokens += delta[0]
            self._completion_tokens += delta[1]
            if self.task_callback:
                self.task_callback(_TaskOutput(task))
        return _Output()
"""

_DOTENV_STUB = "def load_dotenv(*args, **kwargs):\n    return False\n"


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    shutil.copytree(RUNTIME_ROOT, root)
    shutil.copytree(CREW_CONFIG, root / "config", dirs_exist_ok=True)
    (root / "VERSION").write_text(BUNDLE_VERSION, encoding="utf-8")
    # Importable from the worker because its cwd is the bundle root.
    (root / "crewai.py").write_text(_CREWAI_STUB, encoding="utf-8")
    (root / "dotenv.py").write_text(_DOTENV_STUB, encoding="utf-8")
    return root


class _Client:
    """The smallest HTTP client that can see what this test needs to see."""

    def __init__(self, port: int) -> None:
        self.port = port

    def request(
        self, method: str, path: str, *, body: bytes | None = None, headers: dict[str, str] | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request(method, PATH_PREFIX + path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
        finally:
            connection.close()

    def json(self, method: str, path: str, payload: Any = None, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        headers = {"Content-Type": "application/json", **(kwargs.pop("headers", None) or {})}
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        if body is not None:
            headers["Content-Length"] = str(len(body))
        status, _, raw = self.request(method, path, body=body, headers=headers, **kwargs)
        return status, json.loads(raw.decode("utf-8")) if raw else {}


@contextmanager
def _serve(
    bundle_root: Path,
    *,
    entitlement: ent.Entitlement | None = None,
    origins: tuple[str, ...] = (),
    limits: Limits | None = None,
    environ: dict[str, str] | None = None,
) -> Iterator[_Client]:
    config = RunnerConfig(
        bundle_root=bundle_root,
        port=0,
        allowed_origins=origins,
        limits=limits or Limits(),
    )
    gate = entitlement or ent.Entitlement(base_url="", token="", agent_id="")
    server = PosternServer(config, Runner(config, gate))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Client(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def client(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Client]:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    with _serve(bundle) as served:
        yield served


# ---------------------------------------------------------------------------
# The payloads, against the schemas this platform serves
# ---------------------------------------------------------------------------


def test_describe_answers_the_contract_and_validates(client: _Client) -> None:
    """SPEC 4.1: side-effect free, no credentials, no entitlement."""
    status, body = client.json("GET", "/describe")
    assert status == 200
    postern_schema.validate(body, postern_schema.load("describe"), name="describe")
    assert body["postern"] == "0.1"
    assert body["inputs"][0]["key"] == "prompt"


def test_describe_and_status_name_the_same_agent(bundle: Path) -> None:
    """One runner, one agent (SPEC 2.2), so one identifier in both payloads.

    ``describe.agent.id``, ``status.agent.id`` and the id the entitlement
    check addresses are the same string in the specification — "one agent,
    one identifier, spelled one way in all four places" — so a runner whose
    ``describe`` names one listing while its check asks about another has
    already lost the property a client relies on.
    """
    licensed = ent.Entitlement(base_url="https://d.example", token="t", agent_id="acme/market-research-crew")
    with _serve(bundle, entitlement=licensed) as client:
        assert client.json("GET", "/describe")[1]["agent"]["id"] == "acme/market-research-crew"
        assert client.json("GET", "/status")[1]["agent"]["id"] == "acme/market-research-crew"


def test_status_answers_at_level_1_and_validates(client: _Client) -> None:
    status, body = client.json("GET", "/status")
    assert status == 200
    postern_schema.validate(body, postern_schema.load("status"), name="status")
    assert body["level"] == CONFORMANCE_LEVEL
    assert body["entitlement"] == {"state": "not_required"}
    assert body["credentials"] == {"satisfied": True, "missing": []}


def test_a_run_returns_the_response_body_and_validates(client: _Client) -> None:
    status, body = client.json("POST", "/run", {"inputs": {"prompt": "review this"}})
    assert status == 200
    postern_schema.validate(body, postern_schema.load("run-response"), name="run")
    assert body["output"]["type"] == "text"
    assert body["output"]["value"].startswith("## Positioning brief")
    assert body["usage"]["input_tokens"] == 4210
    # The example in SPEC 4.2 is this arithmetic at gpt-4o-mini's prices.
    assert body["usage"]["cost_usd"] == pytest.approx(0.001182)
    assert [step["name"] for step in body["usage"]["steps"]]


def test_a_step_carries_the_task_s_own_name_model_and_token_delta(client: _Client) -> None:
    """``usage.steps[*]`` names the task, not its truncated description.

    Each step's tokens are the delta between two ``calculate_usage_metrics()``
    reads — the only per-task granularity crewai's public API exposes — and
    the stub's own per-task increments are chosen so the three deltas sum to
    exactly the run's own ``usage.input_tokens``/``output_tokens``, which is
    the property the design actually claims: crewai hands over no per-task
    number, only a cumulative running total, so a step's share is only ever
    a partition of the whole rather than a measurement of its own.
    """
    status, body = client.json("POST", "/run", {"inputs": {"prompt": "review this"}})
    assert status == 200
    steps = body["usage"]["steps"]
    assert [step["name"] for step in steps] == [
        "summarise_contract",
        "identify_risks",
        "draft_negotiation_play",
    ]
    assert all(step["model_id"] == "gpt-4o-mini" for step in steps)
    assert [(step["input_tokens"], step["output_tokens"]) for step in steps] == [
        (2000, 400),
        (1500, 350),
        (710, 168),
    ]
    assert sum(step["input_tokens"] for step in steps) == body["usage"]["input_tokens"]
    assert sum(step["output_tokens"] for step in steps) == body["usage"]["output_tokens"]


def test_a_run_on_an_unpriced_model_omits_cost_usd_rather_than_zeroing_it(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cost is absent for a model this bundle has no price for, never a zero.

    ``gpt-4.1-mini`` is crewai's own real hardcoded default
    (``crewai.cli.constants.DEFAULT_LLM_MODEL``) — what a buyer who sets no
    override actually runs on — and it is not in ``MODEL_PRICING``, which
    only carries the models a seller can pick. Per-step ``model_id`` is still
    reported; only the cost estimate, which needs a price, is missing.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("STUB_AGENT_MODEL", "gpt-4.1-mini")
    with _serve(bundle) as client:
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "review this"}})
    assert status == 200
    postern_schema.validate(body, postern_schema.load("run-response"), name="run")
    assert body["usage"]["input_tokens"] == 4210
    assert "cost_usd" not in body["usage"]
    assert all(step["model_id"] == "gpt-4.1-mini" for step in body["usage"]["steps"])


def test_the_run_body_carries_no_status_member(client: _Client) -> None:
    """SPEC 4.2 removed it deliberately, and says why.

    Every failure routes through the error envelope on a non-2xx status, so
    a body exists only where the run succeeded — and a field whose one legal
    value is ``ok`` repeats what the status line already said.
    """
    _, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
    assert "status" not in body


def test_the_run_reaches_the_reviewed_config_through_the_kickoff_contract(bundle: Path, client: _Client) -> None:
    """A Postern ``run`` is the same kickoff ``python main.py`` performs.

    Both go through ``sigrix_runtime.execution``, so the crew receives the
    ``{prompt}`` / ``{configuration}`` pair the compiler emits rather than a
    second shape invented for HTTP.
    """
    client.json("POST", "/run", {"inputs": {"prompt": "review this"}})
    kickoff = json.loads((bundle / "kickoff_inputs.json").read_text(encoding="utf-8"))
    assert set(kickoff) == {"prompt", "configuration"}
    assert kickoff["prompt"] == "review this"


# ---------------------------------------------------------------------------
# stream (SPEC 4.3)
# ---------------------------------------------------------------------------


def _events(raw: bytes) -> list[tuple[str, dict[str, Any]]]:
    parsed: list[tuple[str, dict[str, Any]]] = []
    for block in raw.decode("utf-8").split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) != 2:
            continue
        parsed.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return parsed


def test_a_stream_starts_with_start_and_ends_with_exactly_one_done(client: _Client) -> None:
    body = json.dumps({"inputs": {"prompt": "go"}}).encode("utf-8")
    status, headers, raw = client.request(
        "POST",
        "/stream",
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    events = _events(raw)
    names = [name for name, _ in events]
    assert names[0] == "start"
    assert names[-1] == "done"
    assert names.count("done") + names.count("error") == 1

    schema = postern_schema.load("stream-event")
    for name, payload in events:
        if name in {"start", "step", "delta"}:
            postern_schema.validate(payload, schema, name=name)
    postern_schema.validate(dict(events)["done"], postern_schema.load("run-response"), name="done")


def test_a_step_reports_the_finished_edge_and_the_stream_emits_no_delta(client: _Client) -> None:
    """SPEC 4.3. Both are conformant and both are deliberate.

    Only ``finished`` is reported because a hierarchical crew's manager
    decides which member runs, so a ``started`` taken from the task list
    would be a guess about work that may never happen. No ``delta`` is
    emitted because crewai returns its result whole — and the specification
    says a runner that cannot produce incremental text emits none, which is
    what keeps the concatenation invariant true rather than approximated.
    """
    body = json.dumps({"inputs": {"prompt": "go"}}).encode("utf-8")
    _, _, raw = client.request(
        "POST",
        "/stream",
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    events = _events(raw)
    steps = [payload for name, payload in events if name == "step"]
    assert steps and all(step["status"] == "finished" for step in steps)
    assert not any(name == "delta" for name, _ in events)


def test_the_run_id_correlates_the_stream_with_its_own_done_payload(client: _Client) -> None:
    body = json.dumps({"inputs": {"prompt": "go"}}).encode("utf-8")
    _, _, raw = client.request(
        "POST",
        "/stream",
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    events = dict(_events(raw))
    assert events["start"]["run_id"] == events["done"]["run_id"]


# ---------------------------------------------------------------------------
# Errors (SPEC 2.1)
# ---------------------------------------------------------------------------


def test_every_failure_carries_the_envelope_and_nothing_beside_error(client: _Client) -> None:
    """SPEC 2.1: the root is closed — the only schema here that is."""
    schema = postern_schema.load("error")
    for method, path, payload in [
        ("GET", "/nope", None),
        ("GET", "/run", None),
        ("POST", "/run", {"inputs": {}}),
    ]:
        status, body = client.json(method, path, payload)
        assert status >= 400
        assert set(body) == {"error"}
        postern_schema.validate(body, schema, name=f"{method} {path}")


def test_a_verb_reached_by_the_wrong_method_names_the_right_one(client: _Client) -> None:
    """SPEC 2.1: a runner's ``404`` is a path it does not implement.

    Answering with a code outside the specification's table would hand a
    client something it has no rule for, so the diagnosis goes in the
    message where a client is told to show it.
    """
    status, body = client.json("GET", "/run")
    assert status == 404
    assert body["error"]["code"] == "not_found"
    assert "POST" in body["error"]["message"]


def test_a_missing_required_input_is_refused_by_name(client: _Client) -> None:
    """SPEC 4.2: reject with ``bad_request`` and SHOULD name the key."""
    status, body = client.json("POST", "/run", {"inputs": {}})
    assert status == 400
    assert body["error"]["code"] == "bad_request"
    assert body["error"]["detail"] == {"key": "prompt"}


def test_the_refusal_names_the_offending_key_in_the_message(client: _Client) -> None:
    """SPEC 4.2's SHOULD: name the offending key in ``message``.

    The key was already in ``error.detail``, which a client can read and a
    person cannot. The conformance checker reads the message, and reported
    this SHOULD as not followed against this runner --- visibly, once
    SPEC 4.6 let it reach the rule at all.

    Both halves are asserted. The label is what makes the sentence
    readable, and dropping it to satisfy a machine would trade one
    audience for the other; the runner's own acceptance criterion is
    "failure modes produce plain-language errors".
    """
    status, body = client.json("POST", "/run", {"inputs": {}})
    assert status == 400
    message = body["error"]["message"]
    assert "prompt" in message
    assert "Your question or brief" in message
    assert body["error"]["detail"] == {"key": "prompt"}


def test_a_key_that_is_its_own_label_is_named_once(client: _Client) -> None:
    """A declaration with no label has ``label == key``, so naming both stutters."""
    from sigrix_runtime.postern.describe import validate_run_inputs

    document = {"inputs": [{"key": "segment", "type": "text", "required": True}]}
    with pytest.raises(Exception) as caught:
        validate_run_inputs(document, {})
    message = str(getattr(caught.value, "message", caught.value))
    assert message.count("segment") == 1, message


def test_a_declared_credential_that_is_not_set_answers_424(bundle: Path, monkeypatch) -> None:
    """SPEC 2.1's ``missing_credential``, and the plain-language rule.

    "Failure modes produce plain-language errors" is this story's own
    acceptance criterion, so the message names the variable and the file to
    put it in rather than reporting that a check returned false.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with _serve(bundle) as client:
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
        assert status == 424
        assert body["error"]["code"] == "missing_credential"
        assert "OPENAI_API_KEY" in body["error"]["message"] and ".env" in body["error"]["message"]
        assert client.json("GET", "/status")[1]["state"] == "degraded"


def test_a_declared_credential_set_only_in_the_bundles_env_file_answers_ready(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the documented happy path is copying ``.env.example`` to ``.env``.

    ``load_dotenv(bundle_root / ".env")`` runs inside the *worker* subprocess
    (``sigrix_runtime.execution.prepare_environment``); this server process
    never imports it. A buyer who did exactly what ``.env.example`` says —
    copy it to ``.env``, fill in the key, export nothing — used to see
    ``status: degraded`` and every ``run`` refused with 424 naming the
    variable they had already set, because ``missing_credentials`` read only
    this process's own environment.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (bundle / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
    with _serve(bundle) as client:
        status, body = client.json("GET", "/status")
        assert status == 200
        assert body["state"] == "ready"
        assert body["credentials"] == {"satisfied": True, "missing": []}

        status, body = client.json("POST", "/run", {"inputs": {"prompt": "review this"}})
        assert status == 200
        assert body["output"]["value"].startswith("## Positioning brief")


def test_an_env_file_key_declared_with_no_value_still_answers_424(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``KEY=`` line is the un-filled-in template, not a set credential.

    ``.env.example`` itself ships ``OPENAI_API_KEY=`` with no value, so a
    buyer who copies it to ``.env`` without filling it in must still see the
    credential reported missing — the same "a key with a value is a default,
    a key without one is a requirement" line ``doctor.py`` and
    ``_derived_credentials`` already draw, now drawn a third time by the
    presence check this bundle's ``.env`` is read through.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (bundle / ".env").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    with _serve(bundle) as client:
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
        assert status == 424
        assert body["error"]["code"] == "missing_credential"
        assert client.json("GET", "/status")[1]["state"] == "degraded"


def test_a_malformed_request_is_refused_before_the_environment_is_inspected(bundle: Path, monkeypatch) -> None:
    """SPEC 4.6: the request check precedes the environment check.

    Both refusals describe this request --- a required input is missing
    *and* a declared credential is unset --- and SPEC 4.6 orders them, so
    the answer is ``bad_request`` rather than ``missing_credential``.

    Neither neighbouring test covers this. The 4.2 test above sends a
    malformed request with the credential set; the 424 test sends a valid
    request with it unset. Each exercises one rule with the other
    satisfied, so both stayed green while this runner answered 424 here,
    which is what SPEC 4.6 was written to settle.

    The runner is still degraded, and ``status`` still says so --- the
    ordering decides which refusal a *request* earns, not whether the
    environment is reported as incomplete.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with _serve(bundle) as client:
        status, body = client.json("POST", "/run", {"inputs": {}})
        assert status == 400
        assert body["error"]["code"] == "bad_request"
        assert body["error"]["detail"] == {"key": "prompt"}

        assert client.json("GET", "/status")[1]["state"] == "degraded"


def test_the_entitlement_gate_still_precedes_the_request_check() -> None:
    """SPEC 4.6's steps 2 to 5, pinned in the order the source states them.

    This used to read "4.6 orders the request against the environment, not
    entitlement", which was true when written and is not now: 4.6 places the
    entitlement refusals in its sequence, as step 2 — ahead of the media
    type, the inputs and the environment.

    Kept as a source assertion *alongside* the served request in
    ``test_a_lapsed_entitlement_outranks_a_mistyped_body``, because the two
    catch different edits. The served request is the one that matters and is
    the only thing that can see the 2-before-3 seam, which spans two classes.
    This one catches a tidy-up that shuffles these four lines as a unit while
    every served answer still happens to look right — the gate sits directly
    above the block that moved for 4 before 5, so it is the line most likely
    to be carried along.
    """
    source = inspect.getsource(server_module.Runner.prepare_run)
    gate = source.index("self.entitlement.gate()")
    read = source.index("read_body()")
    validate = source.index("validate_run_inputs")
    credentials = source.index("missing_credentials(document, bundle_root=self.config.bundle_root)")
    assert gate < read < validate < credentials


def test_a_body_larger_than_the_cap_is_refused_without_being_read(client: _Client) -> None:
    from sigrix_runtime.postern.server import MAX_REQUEST_BYTES

    status, _, raw = client.request(
        "POST",
        "/run",
        body=b"",
        headers={"Content-Type": "application/json", "Content-Length": str(MAX_REQUEST_BYTES + 1)},
    )
    assert status == 400
    assert json.loads(raw)["error"]["code"] == "bad_request"


# ---------------------------------------------------------------------------
# The Content-Type gate (SPEC 2.3, 7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["/run", "/stream"])
@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", ""])
def test_a_body_that_is_not_json_is_refused_before_the_agent_runs(
    bundle: Path, client: _Client, verb: str, content_type: str
) -> None:
    """The rule that makes the preflight real (SPEC 2.3).

    ``text/plain`` is on the browser's safelist, so a cross-origin ``POST``
    carrying it is sent with no preflight at all — and a runner that parses
    whatever it is handed has therefore no origin check on the two verbs
    that spend money. The refusal has to happen before the agent runs, which
    is why this asserts on the side effect rather than only on the status.
    """
    body = json.dumps({"inputs": {"prompt": "go"}}).encode("utf-8")
    headers = {"Content-Length": str(len(body))}
    if content_type:
        headers["Content-Type"] = content_type
    status, _, raw = client.request("POST", verb, body=body, headers=headers)
    assert status == 400
    assert json.loads(raw)["error"]["code"] == "bad_request"
    assert not (bundle / "kickoff_inputs.json").exists()


@pytest.mark.parametrize("content_type", ["application/json", "application/json; charset=utf-8"])
def test_a_charset_parameter_does_not_make_a_conforming_client_nonconformant(
    client: _Client, content_type: str
) -> None:
    """SPEC 2 requires a client to send the second; SPEC 2.3 says both pass."""
    body = json.dumps({"inputs": {"prompt": "go"}}).encode("utf-8")
    status, _, _ = client.request(
        "POST", "/run", body=body, headers={"Content-Type": content_type, "Content-Length": str(len(body))}
    )
    assert status == 200


# ---------------------------------------------------------------------------
# CORS (SPEC 2.3)
# ---------------------------------------------------------------------------


def test_no_origin_is_allowed_by_default(client: _Client) -> None:
    """A runner defines no authentication, so the origin check is all of it."""
    _, headers, _ = client.request("GET", "/describe", headers={"Origin": "https://app.example"})
    assert "access-control-allow-origin" not in headers
    assert headers.get("vary") == "Origin"


def test_a_configured_origin_is_echoed_on_the_preflight_and_on_the_answer(bundle: Path) -> None:
    """Both, because the two are refused separately (SPEC 2.3).

    A ``run`` whose response arrives without the header is discarded by the
    browser exactly as an unpermitted one would be — the agent having run.
    """
    with _serve(bundle, origins=("https://app.example",)) as client:
        status, headers, _ = client.request(
            "OPTIONS",
            "/run",
            headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST"},
        )
        assert status == 204
        assert headers["access-control-allow-origin"] == "https://app.example"
        assert headers["access-control-allow-methods"] == "POST, OPTIONS"
        assert "Content-Type" in headers["access-control-allow-headers"]
        assert headers["vary"] == "Origin"

        _, answer_headers, _ = client.request("GET", "/describe", headers={"Origin": "https://app.example"})
        assert answer_headers["access-control-allow-origin"] == "https://app.example"
        assert answer_headers["vary"] == "Origin"


def test_a_refused_origin_gets_a_bare_204_rather_than_an_error(bundle: Path) -> None:
    """A ``403`` would send whoever reads the log after an entitlement bug."""
    with _serve(bundle, origins=("https://app.example",)) as client:
        status, headers, raw = client.request(
            "OPTIONS",
            "/run",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert status == 204
        assert raw == b""
        assert "access-control-allow-origin" not in headers


def test_origin_null_is_never_an_origin_a_configuration_can_name(bundle: Path) -> None:
    """It is a wildcard wearing the shape of one specific origin (SPEC 2.3).

    Sandboxed documents, ``file://`` pages and several redirect chains all
    send it, so allowing ``null`` allows all of them at once.
    """
    with _serve(bundle, origins=("null",)) as client:
        _, headers, _ = client.request("GET", "/describe", headers={"Origin": "null"})
        assert "access-control-allow-origin" not in headers


def test_credentials_are_never_invited(bundle: Path) -> None:
    """SPEC 2.3: the header can only admit ambient credentials nobody asked for."""
    with _serve(bundle, origins=("https://app.example",)) as client:
        _, headers, _ = client.request(
            "OPTIONS",
            "/run",
            headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-credentials" not in headers


def test_a_preflight_needs_no_entitlement(bundle: Path) -> None:
    """SPEC 2.3: it is not the verb behind it.

    A runner whose licence has been revoked still answers the preflight, so
    the ``POST`` arrives and can be refused something a client can read.
    """
    revoked = ent.Entitlement(base_url="https://d.example", token="t", agent_id="acme/x")
    revoked._store(
        ent.CheckAnswer(
            state=ent.STATE_REVOKED,
            checked_at=ent.datetime.now(ent.UTC),
            stale_after_seconds=60,
            grace_seconds=0,
        )
    )
    with _serve(bundle, entitlement=revoked, origins=("https://app.example",)) as client:
        status, headers, _ = client.request(
            "OPTIONS",
            "/run",
            headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST"},
        )
        assert status == 204
        assert headers["access-control-allow-origin"] == "https://app.example"
        assert client.json("POST", "/run", {"inputs": {"prompt": "go"}})[0] == 403


# ---------------------------------------------------------------------------
# Entitlement, from the client's side of the socket (SPEC 5.7.4)
# ---------------------------------------------------------------------------


def test_a_revoked_entitlement_stops_run_and_stream_and_leaves_the_rest(bundle: Path) -> None:
    """SPEC 5.7.4: "Only ``run`` and ``stream`` stop."

    ``describe`` answers without an entitlement and ``status`` answers at
    Level 1, so a runner that cannot run its agent still says what it is and
    what is wrong.
    """
    revoked = ent.Entitlement(base_url="https://d.example", token="t", agent_id="acme/x")
    revoked._store(
        ent.CheckAnswer(
            state=ent.STATE_REVOKED,
            checked_at=ent.datetime.now(ent.UTC),
            stale_after_seconds=60,
            grace_seconds=0,
        )
    )
    with _serve(bundle, entitlement=revoked) as client:
        assert client.json("GET", "/describe")[0] == 200
        status, body = client.json("GET", "/status")
        assert status == 200
        assert body["entitlement"]["state"] == "revoked"
        postern_schema.validate(body, postern_schema.load("status"), name="status")

        for verb in ("/run", "/stream"):
            code, error = client.json("POST", verb, {"inputs": {"prompt": "go"}})
            assert (code, error["error"]["code"]) == (403, "not_entitled")
        assert not (bundle / "kickoff_inputs.json").exists()


def test_a_runner_that_has_never_reached_its_distributor_answers_503(bundle: Path) -> None:
    """SPEC 5.7.3: ``not_entitled`` would assert something no distributor said."""
    unreachable = ent.Entitlement(base_url="http://127.0.0.1:9", token="t", agent_id="acme/x", timeout=0.05)
    with _serve(bundle, entitlement=unreachable) as client:
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
        assert (status, body["error"]["code"]) == (503, "unavailable")
        state = client.json("GET", "/status")[1]["entitlement"]
        assert state["state"] == "unknown" and "checked_at" not in state


def _manifest_naming(url: str) -> str:
    return json.dumps(
        {
            "$schema": "https://example.invalid/plugin.schema.json",
            "name": "crew",
            "extensions": {"org.sigrix": {"listing_url": url}},
        }
    )


def test_build_runner_reads_the_listing_page_from_the_bundles_manifest(bundle: Path) -> None:
    """``plugin.json`` is where a bundle knows its own listing page."""
    url = "https://sigrix.io/crew/contract-review-crew"
    (bundle / "plugin.json").write_text(_manifest_naming(url), encoding="utf-8")
    runner = server_module.build_runner(RunnerConfig(bundle_root=bundle), environ={})
    assert runner.entitlement.listing_url == url


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"name": "crew"}),
        json.dumps({"extensions": {"org.sigrix": {"listing_url": "javascript:alert(1)"}}}),
        json.dumps({"extensions": {"org.sigrix": {"listing_url": "https://sigrix.io/crew/x y"}}}),
        json.dumps({"extensions": {"org.sigrix": {"listing_url": 7}}}),
    ],
)
def test_a_manifest_that_names_no_usable_page_yields_no_link(bundle: Path, payload: str) -> None:
    """read defensively — a link a client would render has to be one."""
    (bundle / "plugin.json").write_text(payload, encoding="utf-8")
    assert describe_module.listing_url(bundle) == ""
    assert describe_module.listing_url(bundle / "nowhere") == ""


def test_a_refused_run_points_at_the_listing_page_and_status_names_it_too(bundle: Path) -> None:
    """the 403 and ``status`` carry the one address a refused buyer can act on.

    Both stay inside the schemas this platform serves: ``error.detail`` and
    the ``entitlement`` block admit members beyond the ones the specification
    names, and the sentence itself still names no cause.
    """
    url = "https://sigrix.io/crew/contract-review-crew"
    (bundle / "plugin.json").write_text(_manifest_naming(url), encoding="utf-8")
    revoked = ent.Entitlement(
        base_url="https://d.example", token="t", agent_id="acme/x", listing_url=describe_module.listing_url(bundle)
    )
    revoked._store(
        ent.CheckAnswer(
            state=ent.STATE_REVOKED,
            checked_at=ent.datetime.now(ent.UTC),
            stale_after_seconds=60,
            grace_seconds=0,
        )
    )
    with _serve(bundle, entitlement=revoked) as client:
        code, error = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
        assert (code, error["error"]["code"]) == (403, "not_entitled")
        assert url in error["error"]["message"]
        assert error["error"]["detail"]["listing_url"] == url
        assert error["error"]["detail"]["state"] == "revoked"
        postern_schema.validate(error, postern_schema.load("error"), name="error")

        status, body = client.json("GET", "/status")
        assert status == 200
        assert body["entitlement"]["listing_url"] == url
        postern_schema.validate(body, postern_schema.load("status"), name="status")


# ---------------------------------------------------------------------------
# The life of a run (SPEC 4.5)
# ---------------------------------------------------------------------------


def test_an_overlapping_run_is_refused_with_unavailable(bundle: Path, monkeypatch) -> None:
    """SPEC 4.5: it fits without a new code — retrying genuinely may help."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    slow = (
        (bundle / "crewai.py")
        .read_text(encoding="utf-8")
        .replace(
            "        for task in self.tasks:",
            "        import time as _time\n        _time.sleep(1.5)\n        for task in self.tasks:",
        )
    )
    (bundle / "crewai.py").write_text(slow, encoding="utf-8")

    with _serve(bundle, limits=Limits(max_concurrent_runs=1)) as client:
        outcomes: list[int] = []

        def _run() -> None:
            outcomes.append(client.json("POST", "/run", {"inputs": {"prompt": "go"}})[0])

        first = threading.Thread(target=_run)
        first.start()
        threading.Event().wait(0.6)
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
        assert (status, body["error"]["code"]) == (503, "unavailable")
        assert client.json("GET", "/status")[1]["state"] == "running"
        first.join(timeout=30)
        assert outcomes == [200]


def test_limits_are_declared_only_where_the_runner_imposes_them(bundle: Path) -> None:
    """SPEC 4.4: absent means no limit of its own, and a bound must be real.

    "A runner MUST NOT declare a bound longer than the shortest one it can
    actually enforce" — which is why the run happens in a process this
    runner can kill rather than a thread it cannot.
    """
    with _serve(bundle) as client:
        assert "max_run_seconds" not in client.json("GET", "/status")[1]["limits"]
    with _serve(bundle, limits=Limits(max_run_seconds=900)) as client:
        assert client.json("GET", "/status")[1]["limits"]["max_run_seconds"] == 900


def test_a_run_past_the_declared_bound_is_stopped_and_says_which_bound(bundle: Path, monkeypatch) -> None:
    """SPEC 4.5: ``504`` ``run_timeout``, with the bound inside ``detail``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    slow = (
        (bundle / "crewai.py")
        .read_text(encoding="utf-8")
        .replace(
            "        for task in self.tasks:",
            "        import time as _time\n        _time.sleep(30)\n        for task in self.tasks:",
        )
    )
    (bundle / "crewai.py").write_text(slow, encoding="utf-8")

    with _serve(bundle, limits=Limits(max_run_seconds=1)) as client:
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
        assert (status, body["error"]["code"]) == (504, "run_timeout")
        assert body["error"]["detail"] == {"max_run_seconds": 1}


def test_a_lapsed_entitlement_outranks_a_mistyped_body(bundle: Path) -> None:
    """SPEC 4.6 step 2 before step 3, which is what that section added.

    A request can earn both refusals at once, and until this was fixed the
    media-type gate won: ``_handle_run`` passed ``self._read_json_body()``
    as an *argument*, and Python evaluates arguments before the call, so
    the ``400`` was raised before ``prepare_run`` reached the gate.

    The ordering is not cosmetic. SPEC 5.7.4 says a runner told no "does
    not pretend the answer might change on the next request", and a ``400``
    is exactly that pretence — it names something the caller can fix, so it
    invites a retry that cannot succeed. The revoked runner has to say the
    one thing true of every request it will ever be sent.

    Driven over the socket rather than asserted against the source: the
    ordering spans two functions in different classes, which is precisely
    what the existing source scan over ``prepare_run`` alone cannot see.
    """
    revoked = ent.Entitlement(base_url="https://d.example", token="t", agent_id="acme/x")
    revoked._store(
        ent.CheckAnswer(
            state=ent.STATE_REVOKED,
            checked_at=ent.datetime.now(ent.UTC),
            stale_after_seconds=60,
            grace_seconds=0,
        )
    )
    with _serve(bundle, entitlement=revoked) as client:
        for verb in ("/run", "/stream"):
            for content_type in ("text/plain", ""):
                body = b'{"inputs": {"prompt": "go"}}'
                status, _, raw = client.request(
                    "POST",
                    verb,
                    body=body,
                    headers={"Content-Type": content_type, "Content-Length": str(len(body))},
                )
                assert (status, json.loads(raw)["error"]["code"]) == (403, "not_entitled"), (
                    f"{verb} with Content-Type {content_type!r} answered the media type before the entitlement"
                )
