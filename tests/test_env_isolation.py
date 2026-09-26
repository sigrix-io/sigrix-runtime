"""The buyer's credentials must not reach the seller's agent subprocess.

``Engine._spawn`` built the run's environment as ``dict(os.environ)``, so every
variable the *runner* is configured with was copied into a subprocess that runs
a **seller's** code. ``SIGRIX_TOKEN`` is the one that makes this a defect rather
than untidiness: it is the buyer's account-wide plugin-feed token — one per
account, not one per listing — so it answers every entitlement check and every
bundle pull that account is entitled to. ``POSTERN_INBOUND_TOKEN`` is the same
shape pointed the other way, the bearer this runner demands of its own callers.
Nothing downstream has ever read either: ``_worker`` reads only its stdin
request, ``execution`` reads ``OPENAI_MODEL_NAME``, and crewai reads provider
keys. So the copy bought nothing and spent a credential.

**Two legs, and a strip at the spawn closes only one.** The variables arrive by
``export`` and by the container's ``docker run -e``, which ``Engine._spawn``
covers — and also through the bundle's own ``.env``, which
``execution.prepare_environment`` loads *inside* the run. That second leg is the
documented happy path rather than a corner: ``.env.example``'s first line tells
the buyer to copy it to ``.env``, and ``entitlement.from_environment`` reads the
token from there precisely because buyers do. So ``load_dotenv`` puts back what
the spawn took out, and both enforcement points are driven here.
``sigrix_runtime.execution`` owns the one rule they share.

**This is driven, not read.** The leak is in what ``Popen`` is handed and in what
``load_dotenv`` then re-adds, neither of which an assertion about source can see
and neither of which a stub of ``Popen`` can either — a double shaped like the
caller's intent would agree with the engine about an environment that was never
passed. So the bundle here is real (the shipped starter tree plus a demo's
``config/``), the run is real, and the crew standing in for crewai reports
``os.environ`` as the worker actually received it. The provider key and ``HOME``
are asserted **present** in the same payload, because a strip that took those
would close the leak and break every run.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path

import pytest

from sigrix_runtime import execution as execution_module  # noqa: E402
from sigrix_runtime.postern import PATH_PREFIX  # noqa: E402
from sigrix_runtime.postern import entitlement as ent  # noqa: E402
from sigrix_runtime.postern.engine import Limits  # noqa: E402
from sigrix_runtime.postern.server import PosternServer, Runner, RunnerConfig  # noqa: E402
from tests.support import BUNDLE_VERSION, CREW_CONFIG, RUNTIME_ROOT

#: What a buyer's machine really has set. ``pull.py``'s own first paragraph
#: documents the shape that produces most of it:
#: ``docker run -e SIGRIX_TOKEN=… sigrix/runner acme/my-crew``.
RUNNER_ENVIRONMENT = {
    "SIGRIX_TOKEN": "sgx-buyer-account-wide-feed-token",
    "SIGRIX_DELIVERY_MODE": "container",
    "POSTERN_AGENT_ID": "acme/listing-5520",
    "POSTERN_DISTRIBUTOR": "https://distributor.example",
    "POSTERN_INBOUND_TOKEN": "inbound-bearer-this-runner-demands",
    "POSTERN_MAX_RUN_SECONDS": "240",
}

#: Variables the run legitimately inherits, each for a reason that is not
#: "it happened to be in the parent". A strip wide enough to take one of
#: these is a broken runner, not a tighter one.
INHERITED_BY_DESIGN = {
    # crewai reads this out of the environment itself; SPEC 4.1.3 gives a
    # provider key nowhere else to travel.
    "OPENAI_API_KEY": "sk-test",
    # ``execution.MODEL_ENV_KEY``.
    "OPENAI_MODEL_NAME": "gpt-4o-mini",
    # ``__main__._ensure_writable_home`` fixes this in the *server* process
    # precisely so the subprocess inherits it.
    "HOME": "",  # filled per-test from tmp_path
}

#: Variables a *deployment* depends on the run inheriting, asserted against the
#: rule rather than against a run. They cannot be exercised here: the bundle
#: this fixture builds is importable only because the worker's cwd is on
#: ``sys.path``, which is the very thing ``PYTHONSAFEPATH`` removes — the image
#: supplies its runtime on ``PYTHONPATH`` instead, and reproducing that shape
#: would make this a test of the image rather than of the strip.
#: The image's own suite is where its shape is
#: driven. What is worth pinning is the widening direction: a strip that grew a
#: ``PYTHON`` prefix would break every container run, silently and only there.
LOAD_BEARING_OUTSIDE_THE_NAMESPACES = (
    # Without it the server process runs the image's runtime while every run
    # subprocess imports the bundle's copy — the two-implementations bug
    # ``container/Dockerfile`` explains at length.
    "PYTHONSAFEPATH",
    "PYTHONPATH",
    # The engine sets this itself, so a rebuilt mapping has to leave room for it.
    "PYTHONUNBUFFERED",
    # The run is a subprocess of this interpreter; it needs to find one.
    "PATH",
)

# A crew that calls no model and writes down the environment it was handed.
# ``kickoff`` is the first seller-reachable code in the run, which is the
# point the question is asked at.
_CREWAI_STUB = """
import json as _json
import os as _os
import pathlib as _pathlib


class Process:
    sequential = "sequential"
    hierarchical = "hierarchical"


class LLM:
    def __init__(self, model=None):
        self.model = model or "gpt-4o-mini"


class Agent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.llm = LLM()


class Task:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.name = kwargs.get("name")


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class _Output:
    token_usage = _Usage()

    def __str__(self):
        return "the answer"


class Crew:
    task_callback = None

    def __init__(self, **kwargs):
        self.tasks = kwargs.get("tasks") or []

    def calculate_usage_metrics(self):
        return _Usage()

    def kickoff(self, inputs=None):
        _pathlib.Path("worker_env.json").write_text(_json.dumps(dict(_os.environ), sort_keys=True))
        return _Output()
"""

#: The ``.env`` the bundle's own ``.env.example`` tells the buyer to create,
#: with the token in it because that is where the Postern server looks for it
#: (``entitlement.from_environment``), and one seller-declared key beside it to
#: show the file is still read for everything else.
DOTENV_FILE = (
    "SIGRIX_TOKEN=sgx-token-out-of-the-dotenv-file\n"
    "POSTERN_AGENT_ID=acme/listing-5520\n"
    "SOME_SELLER_KEY=the-agent-may-have-this\n"
)

# Stands in for ``python-dotenv`` rather than neutering it: the question on the
# second leg is what that function puts *back* into the run's environment, so
# this copies its contract — set what is absent, never override what is set.
_DOTENV_STUB = """
import json
import os
import pathlib


def load_dotenv(dotenv_path=None, *args, **kwargs):
    # First thing `prepare_environment` calls, so what is in `os.environ`
    # here is what `Popen` handed this process — before the run path has
    # dropped anything. That is the only window in which the spawn's own
    # strip is observable, and without this report a worker-side drop
    # masks a missing one entirely.
    pathlib.Path('spawned_env.json').write_text(json.dumps(dict(os.environ), sort_keys=True))
    path = pathlib.Path(str(dotenv_path))
    if not path.exists():
        return False
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        os.environ.setdefault(key.strip(), value.strip().strip('\"').strip("'"))
    return True
"""

#: What the crew — the first seller-reachable code in the run — was handed.
AGENT_ENV_REPORT = "worker_env.json"

#: What ``Popen`` handed the process, written before the run path drops
#: anything. Pins the spawn's own strip, which the worker-side drop would
#: otherwise cover for: measured, removing the spawn strip alone left all 32
#: assertions green until this report existed.
SPAWNED_ENV_REPORT = "spawned_env.json"


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


@contextmanager
def _serve(bundle_root: Path) -> Iterator[int]:
    config = RunnerConfig(bundle_root=bundle_root, port=0, allowed_origins=(), limits=Limits())
    gate = ent.Entitlement(base_url="", token="", agent_id="")
    server = PosternServer(config, Runner(config, gate))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


def _run_and_report(bundle_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Drive one real ``/run``; return (what Popen handed it, what the crew saw)."""
    body = json.dumps({"inputs": {"prompt": "review this"}}).encode("utf-8")
    with _serve(bundle_root) as port:
        connection = HTTPConnection("127.0.0.1", port, timeout=60)
        try:
            connection.request(
                "POST",
                PATH_PREFIX + "/run",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            response = connection.getresponse()
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()
    assert status == 200, payload
    assert payload["output"]["value"] == "the answer"
    spawned_report = bundle_root / SPAWNED_ENV_REPORT
    agent_report = bundle_root / AGENT_ENV_REPORT
    assert spawned_report.exists(), "the run path never loaded the bundle's .env, so the spawn is unobserved"
    assert agent_report.exists(), "the stub crew never ran, so this proves nothing about its environment"
    spawned = json.loads(spawned_report.read_text(encoding="utf-8"))
    seen = json.loads(agent_report.read_text(encoding="utf-8"))
    assert isinstance(spawned, dict) and isinstance(seen, dict)
    return spawned, seen


@pytest.fixture(params=["exported", "dotenv"])
def worker_environment(
    request: pytest.FixtureRequest, bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    """The worker's environment, with the runner configured each way it can be.

    ``exported`` is an ``export`` or the container's ``docker run -e``, which
    ``Engine._spawn`` is the enforcement point for. ``dotenv`` is the bundle's
    own ``.env``, which the run loads itself — so nothing the *spawn* does can
    reach it, and every assertion below has to hold on both.

    The provider key is exported in both, because the credential gate (SPEC 4.6
    step 5) reads the process environment rather than the bundle's ``.env``, so
    a key that lives only in the file refuses the run before it starts.
    """
    for name, value in {**INHERITED_BY_DESIGN, "HOME": str(tmp_path)}.items():
        monkeypatch.setenv(name, value)

    if request.param == "exported":
        for name, value in RUNNER_ENVIRONMENT.items():
            monkeypatch.setenv(name, value)
    else:
        # Nothing exported: the file is the only source, which is what the
        # bundle's own `.env.example` instructs and what the Postern server
        # reads the token from.
        for name in RUNNER_ENVIRONMENT:
            monkeypatch.delenv(name, raising=False)
        (bundle / ".env").write_text(DOTENV_FILE, encoding="utf-8")

    spawned, seen = _run_and_report(bundle)
    if request.param == "dotenv":
        # Otherwise a `.env` nothing read would satisfy every assertion here.
        assert seen.get("SOME_SELLER_KEY") == "the-agent-may-have-this", (
            "the bundle's .env never reached the run, so this leg proves nothing"
        )
    request.node.stash[_SPAWNED] = spawned
    request.node.stash[_LEG] = str(request.param)
    return seen


#: Where the fixture parks the spawn-time report, so the two observation points
#: can be asserted separately without running the bundle twice.
_SPAWNED: pytest.StashKey[dict[str, str]] = pytest.StashKey()

#: Which arrival path this case ran. A test that needs only one of them keys on
#: this rather than on the value it is about to check — skipping because the
#: value is missing would skip exactly the failure the test exists to catch.
_LEG: pytest.StashKey[str] = pytest.StashKey()


@pytest.fixture()
def spawned_environment(request: pytest.FixtureRequest, worker_environment: dict[str, str]) -> dict[str, str]:
    """What ``Popen`` handed the run process, before the run path dropped any."""
    return request.node.stash[_SPAWNED]


@pytest.fixture()
def leg(request: pytest.FixtureRequest, worker_environment: dict[str, str]) -> str:
    """Which arrival path this case ran: ``exported`` or ``dotenv``."""
    return request.node.stash[_LEG]


# ---------------------------------------------------------------------------
# What must not cross
# ---------------------------------------------------------------------------


def test_the_buyers_account_wide_token_does_not_reach_the_agent(worker_environment: dict[str, str]) -> None:
    """The defect itself: one token, every listing that account has bought."""
    assert "SIGRIX_TOKEN" not in worker_environment
    rendered = json.dumps(worker_environment)
    assert RUNNER_ENVIRONMENT["SIGRIX_TOKEN"] not in rendered
    assert "sgx-token-out-of-the-dotenv-file" not in rendered


def test_the_bearer_this_runner_demands_does_not_reach_the_agent(worker_environment: dict[str, str]) -> None:
    """``POSTERN_INBOUND_TOKEN`` would let a crew drive the runner hosting it."""
    assert "POSTERN_INBOUND_TOKEN" not in worker_environment
    assert RUNNER_ENVIRONMENT["POSTERN_INBOUND_TOKEN"] not in json.dumps(worker_environment)


@pytest.mark.parametrize("name", sorted(RUNNER_ENVIRONMENT))
def test_no_runner_setting_reaches_the_agent(name: str, worker_environment: dict[str, str]) -> None:
    assert name not in worker_environment


def test_every_variable_this_runner_reads_is_stripped(worker_environment: dict[str, str]) -> None:
    """Derived from the runner's own vocabulary rather than restated here.

    ``entitlement._ENV_FILE_KEYS`` is what the runner will read out of a
    bundle's ``.env``; the strip is by namespace prefix. The two agree today,
    and a fifth credential added to that tuple under a name outside both
    prefixes would reach the agent — so ask the question of the tuple instead
    of trusting the prefixes to keep covering it.
    """
    for name in ent._ENV_FILE_KEYS:
        assert name not in worker_environment, f"{name} is read by the runner and must not reach a run"


def test_nothing_in_either_runner_namespace_survives(worker_environment: dict[str, str]) -> None:
    """A variable added to either namespace later inherits the strip.

    This is the property a prefix buys over a list of four names, and it is
    the direction that matters: a new runner setting is invisible to the agent
    by default rather than reaching it until somebody notices.
    """
    crossed = sorted(name for name in worker_environment if name.startswith(execution_module.RUNNER_ENV_PREFIXES))
    assert crossed == []


# ---------------------------------------------------------------------------
# The two enforcement points, asserted separately
# ---------------------------------------------------------------------------


def test_the_spawn_does_not_hand_the_runners_settings_to_the_process_at_all(
    spawned_environment: dict[str, str],
) -> None:
    """``Engine._spawn``'s own half, observed before the run path can cover it.

    The worker-side drop removes these a moment later whatever the spawn did,
    so without this the spawn strip is unpinned: measured, reverting it alone
    left every other assertion in this file green. What the strip buys over the
    drop is that the value is never in the process — a library that snapshots
    the environment when it is imported cannot capture what was never there.
    """
    crossed = sorted(name for name in spawned_environment if execution_module.is_runner_setting(name))
    assert crossed == []


def test_the_run_path_drops_what_the_bundles_own_dotenv_puts_back(
    spawned_environment: dict[str, str], worker_environment: dict[str, str]
) -> None:
    """``prepare_environment``'s half, which no spawn can reach.

    ``load_dotenv`` runs inside the run and sets whatever the spawn left
    absent, so on the documented happy path — ``.env.example`` copied to
    ``.env`` with the token in it — the file puts the credential back into the
    process running a seller's crew. The spawn-time report is clean either way,
    which is exactly why this needs asking of the crew's own view.
    """
    assert sorted(name for name in spawned_environment if execution_module.is_runner_setting(name)) == []
    assert sorted(name for name in worker_environment if execution_module.is_runner_setting(name)) == []


def test_the_dotenv_file_is_still_read_for_everything_else(leg: str, worker_environment: dict[str, str]) -> None:
    """The drop is the runner's namespaces, not a refusal to read the file.

    A seller's own declared key in the same ``.env`` has to arrive, or the drop
    has been implemented as "ignore the bundle's environment file", which would
    break every bundle that documents one.

    Keyed on the leg rather than on the key's presence: skipping because the
    value is absent would skip precisely the failure this exists to catch.
    """
    if leg != "dotenv":
        pytest.skip("the exported leg writes no .env; this asks about the file")
    assert worker_environment["SOME_SELLER_KEY"] == "the-agent-may-have-this"


# ---------------------------------------------------------------------------
# What must still cross — a strip that takes these is a broken runner
# ---------------------------------------------------------------------------


def test_the_provider_key_still_reaches_the_agent(worker_environment: dict[str, str]) -> None:
    """SPEC 4.1.3 gives it nowhere else to travel, and the agent spends it."""
    assert worker_environment.get("OPENAI_API_KEY") == INHERITED_BY_DESIGN["OPENAI_API_KEY"]
    assert worker_environment.get("OPENAI_MODEL_NAME") == INHERITED_BY_DESIGN["OPENAI_MODEL_NAME"]


def test_the_writable_home_fix_still_reaches_the_agent(worker_environment: dict[str, str], tmp_path: Path) -> None:
    """The writable ``HOME`` is fixed in the server process and relies on this inheritance."""
    assert worker_environment.get("HOME") == str(tmp_path)


@pytest.mark.parametrize("name", LOAD_BEARING_OUTSIDE_THE_NAMESPACES)
def test_the_strip_does_not_reach_a_variable_a_deployment_depends_on(name: str) -> None:
    """The widening direction, asked of the rule rather than of a run.

    See :data:`LOAD_BEARING_OUTSIDE_THE_NAMESPACES` for why these are not
    driven: the fixture's bundle is importable only because the worker's cwd
    is on ``sys.path``, which is what ``PYTHONSAFEPATH`` exists to remove.
    """
    assert not name.startswith(execution_module.RUNNER_ENV_PREFIXES)


def test_the_worker_is_still_told_to_run_unbuffered(worker_environment: dict[str, str]) -> None:
    """The strip rebuilds the mapping, so the one value the engine *adds*
    has to survive being rebuilt. Without it a step event reaches a
    streaming client when the pipe fills rather than when it happens."""
    assert worker_environment.get("PYTHONUNBUFFERED") == "1"


def test_the_agent_still_inherits_the_rest_of_the_environment(worker_environment: dict[str, str]) -> None:
    """The strip is two namespaces, not an allow-list of its own.

    A bundle may legitimately need any number of variables nobody here has
    thought of — a proxy setting, a CA bundle, a seller's own declared key —
    so the shape to keep is "everything except the runner's own settings".
    """
    assert "PATH" in worker_environment
    assert len(worker_environment) > 10
