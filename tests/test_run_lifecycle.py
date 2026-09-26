"""The life of one run on the Postern runner: its slot, its worker, its end.

``Engine.stream`` holds a concurrency slot for the length of a run and a worker
subprocess for the length of the slot. Everything here is about the edges of
that span -- a run that never starts, a client that leaves, a worker that
finishes on the deadline or leaves a grandchild behind -- because the happy
path is covered by ``tests/test_postern_runner_server.py`` and none of these
show on it.

The bundle is the real starter tree with crewai stubbed, the same fixture the
server tests use, so the worker that runs is the one a buyer's machine runs.
"""

from __future__ import annotations

import errno
import json
import logging
import shutil
import socket
import struct
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from sigrix_runtime.postern import engine as engine_module  # noqa: E402
from sigrix_runtime.postern.engine import Limits  # noqa: E402
from sigrix_runtime.postern.errors import PosternError  # noqa: E402
from tests.support import RUNTIME_ROOT
from tests.test_server import (  # noqa: E402
    _CREWAI_STUB,
    _DOTENV_STUB,
    BUNDLE_VERSION,
    CREW_CONFIG,
    _Client,
    _serve,
)

_BODY = b'{"inputs": {"prompt": "go"}}'


def _slow(bundle: Path, seconds: float) -> None:
    """Hold the stub crew ``seconds`` before its first step, then leave a mark.

    The pause stands in for a long model call: the worker writes nothing
    while it lasts. The mark is written only by a worker that sat it out.
    """
    stub = (bundle / "crewai.py").read_text(encoding="utf-8")
    slow = stub.replace(
        "        for task in self.tasks:",
        f"        import time as _time\n        _time.sleep({seconds})\n"
        "        _pathlib.Path('slept_through').write_text('1')\n        for task in self.tasks:",
    )
    assert slow != stub, "the stub crew's shape moved; this helper no longer slows it"
    (bundle / "crewai.py").write_text(slow, encoding="utf-8")


def _start(client: _Client, verb: str) -> socket.socket:
    """Open ``verb`` on a raw connection, which the test then abandons."""
    sock = socket.create_connection(("127.0.0.1", client.port), timeout=10)
    head = f"POST /postern/v0{verb} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
    sock.sendall(head.encode("ascii") + f"Content-Length: {len(_BODY)}\r\n\r\n".encode("ascii") + _BODY)
    return sock


def _read_until_start_event(sock: socket.socket) -> None:
    received = b""
    while b"event: start" not in received or not received.endswith(b"\n\n"):
        chunk = sock.recv(4096)
        assert chunk, f"the stream ended before its start event: {received!r}"
        received += chunk


def _state_becomes(client: _Client, state: str, *, within: float) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if client.json("GET", "/status")[1]["state"] == state:
            return True
        time.sleep(0.05)
    return False


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    shutil.copytree(RUNTIME_ROOT, root)
    shutil.copytree(CREW_CONFIG, root / "config", dirs_exist_ok=True)
    (root / "VERSION").write_text(BUNDLE_VERSION, encoding="utf-8")
    (root / "crewai.py").write_text(_CREWAI_STUB, encoding="utf-8")
    (root / "dotenv.py").write_text(_DOTENV_STUB, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# A run that never starts gives its slot back
# ---------------------------------------------------------------------------


def test_a_worker_that_fails_to_spawn_gives_its_slot_back(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a failed ``fork`` must not wedge the runner until a restart.

    The spawn sat outside the ``try`` whose ``finally`` releases the slot, so
    one ``EAGAIN`` held it for good: every later ``run`` answered 503 "already
    running an agent" and ``status`` said ``running`` about a run that never
    began. Driven over the socket because that is where a buyer meets it.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    real_popen = engine_module.subprocess.Popen
    attempts: list[object] = []

    def fork_fails_once(*args: object, **kwargs: object) -> object:
        attempts.append(args)
        if len(attempts) == 1:
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(engine_module.subprocess, "Popen", fork_fails_once)

    with _serve(bundle, limits=Limits(max_concurrent_runs=1)) as client:
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
        assert (status, body["error"]["code"]) == (503, "unavailable")
        assert "could not start the agent" in body["error"]["message"]

        assert client.json("GET", "/status")[1]["state"] == "ready"
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
        assert status == 200, body
        assert len(attempts) == 2


def test_a_bundle_whose_requirements_are_not_installed_is_told_so_and_is_not_wedged(tmp_path: Path) -> None:
    """The commonest way a spawn fails: the first run on a bundle nobody ``pip install``-ed.

    The engine imports ``sigrix_runtime.execution`` to build the worker's
    environment, and that reaches ``import yaml``. With PyYAML absent -- the
    ordinary state of a bundle whose virtualenv is not built yet -- the
    ``ModuleNotFoundError`` escaped with the slot held: the first run was an
    "internal error", and every run after it, including the one made once the
    buyer had installed the requirements, was refused as overlapping.

    Run under ``python -S`` with only the starter tree importable, which is
    how ``tests/test_postern_runner_stdlib_only.py`` proves the same absence;
    PyYAML is genuinely missing here rather than patched out.
    """
    script = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from sigrix_runtime.postern.engine import Engine
        from sigrix_runtime.postern.errors import PosternError

        engine = Engine(Path(sys.argv[1]))
        outcomes = []
        for attempt in ("first", "second"):
            try:
                engine.run(prompt="go", variables={}, run_id=attempt)
                outcomes.append(["ran", ""])
            except PosternError as exc:
                outcomes.append([exc.code, exc.message])
            except Exception as exc:
                outcomes.append([type(exc).__name__, str(exc)])
        print(json.dumps({"outcomes": outcomes, "running": engine.running}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(tmp_path)],
        cwd=tmp_path,
        env={"PYTHONPATH": str(RUNTIME_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["running"] is False
    for code, message in report["outcomes"]:
        assert code == "unavailable", report
        assert "'yaml' is missing" in message
        assert "pip install -r requirements.txt" in message


# ---------------------------------------------------------------------------
# A client that leaves stops its run
# ---------------------------------------------------------------------------


def test_a_client_that_leaves_a_run_stops_its_worker(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SPEC 4.5: a disconnected client is a cancelled run, on ``run`` as on ``stream``.

    ``run`` writes nothing until it ends, so a departure was noticed only at
    the final write: the review's probe closed at 0.3 s and the worker ran its
    full 3 s for nobody. ``status`` reads ``ready`` only once the engine has
    terminated the worker and freed the slot, so reaching it well inside the
    crew's 5 s pause is the worker being stopped, not merely abandoned.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _slow(bundle, 5)
    with _serve(bundle) as client:
        sock = _start(client, "/run")
        assert _state_becomes(client, "running", within=5)
        sock.close()
        assert _state_becomes(client, "ready", within=2), "the worker ran on for a client that had left"
    assert not (bundle / "slept_through").exists()


def test_a_client_that_leaves_a_stream_mid_step_stops_it_without_waiting_for_the_step(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stream`` noticed only at its next write, which a long model call postpones.

    The client reads ``start`` and leaves inside the crew's first, 5 s step;
    the run is stopped then rather than when that step would have ended.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _slow(bundle, 5)
    with _serve(bundle) as client:
        sock = _start(client, "/stream")
        _read_until_start_event(sock)
        sock.close()
        assert _state_becomes(client, "ready", within=2), "the step ran to its end for a client that had left"
    assert not (bundle / "slept_through").exists()


def test_a_client_reset_is_a_departure_rather_than_an_error(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A reset is ``ConnectionResetError``, a sibling of ``BrokenPipeError`` rather than one.

    Only ``BrokenPipeError`` was caught, so every tab-close that sent a reset
    fell through to ``logger.exception("postern: stream failed")``: an ERROR
    and a traceback in the buyer's log for the most ordinary event there is.
    The pause is short so that, unfixed, the next write meets the reset.
    """
    caplog.set_level(logging.INFO, logger="sigrix_runtime.postern")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _slow(bundle, 0.5)
    with _serve(bundle) as client:
        sock = _start(client, "/stream")
        _read_until_start_event(sock)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        sock.close()  # SO_LINGER 0: a reset, not a close
        assert _state_becomes(client, "ready", within=5)
    postern = [record for record in caplog.records if record.name.startswith("sigrix_runtime.postern")]
    assert [record.getMessage() for record in postern if record.levelno >= logging.ERROR] == []
    assert any("client left" in record.getMessage() or "disconnected" in record.getMessage() for record in postern)


# ---------------------------------------------------------------------------
# The worker's end: the timeout verdict, the group, the diagnostic
# ---------------------------------------------------------------------------


class _ClockPastTheDeadline:
    """The engine's clock: zero while a run is set up, then far past any bound.

    The two readings the engine takes before the worker's first frame --
    the start and the watchdog's interval -- see zero, so the watchdog is
    armed for the full bound in real time and never fires. Any reading after
    that is a clock consulted once the run is over.
    """

    def __init__(self) -> None:
        self.readings = 0

    def monotonic(self) -> float:
        self.readings += 1
        return 0.0 if self.readings <= 2 else 1e9


def test_a_worker_that_finished_inside_its_bound_is_not_called_a_timeout(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verdict was the clock read after the loop, not the watchdog's.

    A worker that wrote ``done`` a moment before its deadline, read a moment
    after, was answered ``run_timeout`` and its answer thrown away. Here the
    clock reads past the bound the moment the run is over, as it would in
    that race, and the answer stands.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(engine_module, "time", _ClockPastTheDeadline())
    engine = engine_module.Engine(bundle, limits=Limits(max_run_seconds=60))
    body = engine.run(prompt="go", variables={}, run_id="finished-in-time")
    assert body["run_id"] == "finished-in-time"
    assert body["output"]["value"].startswith("## Positioning brief")


def test_a_timeout_stops_what_the_worker_started_too(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool's subprocess holding the worker's stdout kept the relay open past the bound.

    Killing the worker alone left the grandchild -- here a 20 s sleeper that
    inherited the pipe -- and the 1 s bound was really 20. The worker now
    leads its own process group, and the group goes.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    stub = (bundle / "crewai.py").read_text(encoding="utf-8")
    holds_stdout = stub.replace(
        "        for task in self.tasks:",
        "        import subprocess as _subprocess, sys as _sys, time as _time\n"
        "        _subprocess.Popen([_sys.executable, '-c', 'import time; time.sleep(20)'])\n"
        "        _time.sleep(20)\n"
        "        for task in self.tasks:",
    )
    assert holds_stdout != stub
    (bundle / "crewai.py").write_text(holds_stdout, encoding="utf-8")

    started = time.monotonic()
    with _serve(bundle, limits=Limits(max_run_seconds=1)) as client:
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
    assert (status, body["error"]["code"]) == (504, "run_timeout")
    assert time.monotonic() - started < 10, "the bound held only once the grandchild let go of the pipe"


def test_an_agent_failure_leaves_its_stack_in_the_runners_log(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The worker's stderr goes nowhere and its ``traceback`` field was dropped: no diagnostic anywhere."""
    caplog.set_level(logging.WARNING, logger="sigrix_runtime.postern")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    stub = (bundle / "crewai.py").read_text(encoding="utf-8")
    failing = stub.replace(
        "        for task in self.tasks:",
        "        raise RuntimeError('the model provider refused the request')\n        for task in self.tasks:",
    )
    assert failing != stub
    (bundle / "crewai.py").write_text(failing, encoding="utf-8")

    with _serve(bundle) as client:
        status, body = client.json("POST", "/run", {"inputs": {"prompt": "go"}})
    assert (status, body["error"]["code"]) == (500, "agent_error")
    logged = [record.getMessage() for record in caplog.records if "the agent failed" in record.getMessage()]
    assert len(logged) == 1
    assert "Traceback" in logged[0]
    assert "the model provider refused the request" in logged[0]
    assert "in kickoff" in logged[0]


def test_a_runner_that_is_stopped_stops_its_runs(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker in its own group no longer hears the Ctrl+C that stops the runner.

    Unstopped, it would run on -- spending money -- with nothing left to
    read its output, so ``serve()`` stops every run in flight on its way out.
    """
    from sigrix_runtime.postern import server as server_module
    from sigrix_runtime.postern.server import RunnerConfig

    stopped: list[object] = []

    def interrupted(self: object) -> None:
        raise KeyboardInterrupt

    # Raised from inside the real loop, as Ctrl+C would be: replacing the loop
    # instead leaves the event ``shutdown()`` waits on unset, and it waits forever.
    monkeypatch.setattr(server_module.PosternServer, "service_actions", interrupted)
    monkeypatch.setattr(engine_module.Engine, "stop_all", lambda self: stopped.append(self))
    server_module.serve(RunnerConfig(bundle_root=bundle, port=0), environ={})
    assert len(stopped) == 1


def test_stopping_every_run_stops_one_in_flight(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _slow(bundle, 5)
    engine = engine_module.Engine(bundle)
    outcome: list[str] = []

    def _run() -> None:
        try:
            engine.run(prompt="go", variables={}, run_id="in-flight")
            outcome.append("finished")
        except PosternError as exc:
            outcome.append(exc.code)

    worker = threading.Thread(target=_run)
    worker.start()
    deadline = time.monotonic() + 5
    while not engine._workers and time.monotonic() < deadline:
        time.sleep(0.05)
    assert engine._workers, "the run never reached its worker"

    engine.stop_all()
    worker.join(timeout=3)
    assert not worker.is_alive(), "the run outlived stop_all"
    assert outcome == ["agent_error"]
    assert not engine.running
    assert not (bundle / "slept_through").exists()
