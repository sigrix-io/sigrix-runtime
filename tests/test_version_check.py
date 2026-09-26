"""The boot-time version check (SPEC 8): fetch, compare, and the wiring.

Three layers, three groups of tests below:

* :func:`fetch_latest_version` / :func:`check_for_update` — the client
  module in isolation, against a real loopback distributor (the same
  ``_Distributor`` shape ``test_postern_runner_entitlement.py`` uses for
  SPEC 5.3's check — repeated here rather than imported, since that one is a
  private test helper of that file, not a shared fixture).
* ``Runner.status()``'s ``update`` block — ``server.py``'s rendering of a
  held :class:`~sigrix_runtime.postern.version_check.UpdateCheck`.
* ``__main__.main()``'s boot wiring — ``_check_for_update`` end to end,
  including the exact log line a buyer sees when a re-pull would help.

**Unauthenticated is the point being tested, not an incidental fact.** SPEC
8's endpoint is safe to call with no token (a distributor answers it only for
a listing a stranger can already see), so :func:`fetch_latest_version` is the one
Postern client call in this tree that carries no ``Authorization`` header —
asserted directly here, not merely implied by a passing request.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from sigrix_runtime.postern import __main__ as cli  # noqa: E402
from sigrix_runtime.postern import entitlement as ent  # noqa: E402
from sigrix_runtime.postern import version_check as vc  # noqa: E402
from sigrix_runtime.postern.engine import Limits  # noqa: E402
from sigrix_runtime.postern.server import Runner, RunnerConfig  # noqa: E402
from tests import postern_schema  # noqa: E402

AGENT_ID = "acme/market-research-crew"


# ---------------------------------------------------------------------------
# A stand-in for SPEC 8's endpoint, on loopback
# ---------------------------------------------------------------------------


class _Distributor:
    def __init__(self, handler) -> None:
        self.requests: list[tuple[str, str]] = []
        outer = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                outer.requests.append((self.path, self.headers.get("Authorization") or ""))
                status, payload = handler(self.path)
                body = json.dumps(payload).encode("utf-8") if payload is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except ConnectionError:
                    pass  # a runner that stopped reading at its bound, and hung up
                # One request per connection, as the runner makes them: a handler
                # left waiting for a second meets that hang-up as a reset.
                self.close_connection = True

            def log_message(self, *args) -> None:  # noqa: A002
                pass

        self._server = HTTPServer(("127.0.0.1", 0), _H)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def distributor() -> Iterator[Any]:
    made: list[_Distributor] = []

    def _make(handler) -> _Distributor:
        instance = _Distributor(handler)
        made.append(instance)
        return instance

    yield _make
    for instance in made:
        instance.close()


def _good_body(**overrides: Any) -> dict[str, Any]:
    body = {"postern": "0.1", "agent_id": AGENT_ID, "version": "1.3.0"}
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# fetch_latest_version
# ---------------------------------------------------------------------------


def test_a_good_answer_is_read_and_carries_no_bearer_token(distributor) -> None:
    served = distributor(lambda path: (200, _good_body()))
    version = vc.fetch_latest_version(served.base_url, AGENT_ID)
    assert version == "1.3.0"
    path, authorization = served.requests[-1]
    assert path == "/postern/v0/versions/acme/market-research-crew"
    assert authorization == ""


def test_an_answer_for_a_different_agent_is_not_an_answer(distributor) -> None:
    """Same rule ``entitlement.py``'s ``_parse_check_body`` follows: an echo
    that names somebody else's listing is not an answer to this question."""
    served = distributor(lambda path: (200, _good_body(agent_id="someone/else")))
    assert vc.fetch_latest_version(served.base_url, AGENT_ID) is None


@pytest.mark.parametrize("status", [404, 429, 500, 503])
def test_a_non_200_is_none_not_an_exception(distributor, status: int) -> None:
    served = distributor(lambda path: (status, {"error": {"code": "not_found"}}))
    assert vc.fetch_latest_version(served.base_url, AGENT_ID) is None


def test_an_empty_body_is_none(distributor) -> None:
    served = distributor(lambda path: (200, None))
    assert vc.fetch_latest_version(served.base_url, AGENT_ID) is None


def test_an_answer_past_the_read_bound_is_none_rather_than_read_whole(distributor) -> None:
    """this endpoint is unauthenticated, so an unbounded read reached every runner.

    A well-formed answer padded past the bound is refused, not believed.
    """
    from sigrix_runtime.postern import transport

    # Built here rather than in the handler: a failure raised on the server's
    # thread ends the exchange with no answer, which is also ``None``.
    padded = _good_body(padding="x" * (2 * transport.MAX_ANSWER_BYTES))
    served = distributor(lambda path: (200, padded))
    assert vc.fetch_latest_version(served.base_url, AGENT_ID) is None
    assert len(served.requests) == 1


def test_plaintext_to_a_distant_peer_is_allowed_when_no_token_travels(distributor, monkeypatch) -> None:
    """SPEC 7's loopback rule protects the token, and this request carries none.

    Refusing it anyway made an ``http://`` distributor on another machine
    report every version check unreachable, for nothing kept safe.
    """
    from sigrix_runtime.postern import transport

    monkeypatch.setattr(transport, "_peer_is_loopback", lambda sock: False)
    served = distributor(lambda path: (200, _good_body()))
    assert vc.fetch_latest_version(served.base_url, AGENT_ID) == "1.3.0"


def test_a_body_that_is_not_an_object_is_none(distributor) -> None:
    served = distributor(lambda path: (200, ["not", "a", "dict"]))
    assert vc.fetch_latest_version(served.base_url, AGENT_ID) is None


@pytest.mark.parametrize("bad_version", ["", "   ", None, 42])
def test_a_missing_or_non_string_version_is_none(distributor, bad_version: Any) -> None:
    served = distributor(lambda path: (200, _good_body(version=bad_version)))
    assert vc.fetch_latest_version(served.base_url, AGENT_ID) is None


def test_an_unreachable_distributor_is_none_rather_than_raising() -> None:
    assert vc.fetch_latest_version("http://127.0.0.1:9", AGENT_ID, timeout=0.05) is None


# ---------------------------------------------------------------------------
# check_for_update
# ---------------------------------------------------------------------------


def test_no_base_url_is_not_required() -> None:
    check = vc.check_for_update(base_url="", agent_id=AGENT_ID, current_version="1.0.0")
    assert check.state == vc.STATE_NOT_REQUIRED
    assert check.as_dict() == {"state": vc.STATE_NOT_REQUIRED}


def test_no_agent_id_is_not_required() -> None:
    check = vc.check_for_update(base_url="https://d.example", agent_id="", current_version="1.0.0")
    assert check.state == vc.STATE_NOT_REQUIRED


def test_an_unreachable_check_is_unreachable_and_still_names_the_current_version() -> None:
    check = vc.check_for_update(base_url="http://127.0.0.1:9", agent_id=AGENT_ID, current_version="1.0.0")
    assert check.state == vc.STATE_UNREACHABLE
    assert check.as_dict() == {"state": vc.STATE_UNREACHABLE, "current": "1.0.0"}


def test_the_same_version_is_current(distributor) -> None:
    served = distributor(lambda path: (200, _good_body(version="1.0.0")))
    check = vc.check_for_update(base_url=served.base_url, agent_id=AGENT_ID, current_version="1.0.0")
    assert check.state == vc.STATE_CURRENT
    assert check.as_dict() == {"state": vc.STATE_CURRENT, "current": "1.0.0", "latest": "1.0.0"}


def test_a_different_version_is_available(distributor) -> None:
    served = distributor(lambda path: (200, _good_body(version="1.3.0")))
    check = vc.check_for_update(base_url=served.base_url, agent_id=AGENT_ID, current_version="1.0.0")
    assert check.state == vc.STATE_UPDATE_AVAILABLE
    assert check.as_dict() == {"state": vc.STATE_UPDATE_AVAILABLE, "current": "1.0.0", "latest": "1.3.0"}


def test_an_unversioned_bundle_still_compares_cleanly(distributor) -> None:
    """A composition-fingerprint version (``0.0.0+<hex>``) or a bare
    ``0.0.0`` is still just a string to this comparison — nothing here
    parses or orders it (a versioning model is planned, not built;
    this only needs octet-for-octet equality)."""
    served = distributor(lambda path: (200, _good_body(version="0.0.0+8f3a2c1d0e0f")))
    check = vc.check_for_update(base_url=served.base_url, agent_id=AGENT_ID, current_version="0.0.0")
    assert check.state == vc.STATE_UPDATE_AVAILABLE


# ---------------------------------------------------------------------------
# Runner.status()'s ``update`` block
# ---------------------------------------------------------------------------


def _runner_config(tmp_path: Path, *, update_check: vc.UpdateCheck | None = None) -> RunnerConfig:
    return RunnerConfig(bundle_root=tmp_path, port=0, limits=Limits(), update_check=update_check)


def _bare_runner(config: RunnerConfig) -> Runner:
    return Runner(config, ent.Entitlement(base_url="", token="", agent_id=""))


def test_status_carries_no_update_block_when_none_was_ever_checked(tmp_path: Path) -> None:
    """``None`` — the config's own default — means "checked-on-start never
    ran" (a config built by hand, as every other test in this tree does),
    distinct from an ``UpdateCheck(state=not_required)``, which means it ran
    and found no distributor configured."""
    runner = _bare_runner(_runner_config(tmp_path))
    assert "update" not in runner.status()


@pytest.mark.parametrize(
    "check",
    [
        vc.UpdateCheck(state=vc.STATE_NOT_REQUIRED),
        vc.UpdateCheck(state=vc.STATE_UNREACHABLE, current="1.0.0"),
        vc.UpdateCheck(state=vc.STATE_CURRENT, current="1.0.0", latest="1.0.0"),
        vc.UpdateCheck(state=vc.STATE_UPDATE_AVAILABLE, current="1.0.0", latest="1.3.0"),
    ],
    ids=lambda case: case.state,
)
def test_status_renders_a_held_check_verbatim(tmp_path: Path, check: vc.UpdateCheck) -> None:
    runner = _bare_runner(_runner_config(tmp_path, update_check=check))
    assert runner.status()["update"] == check.as_dict()


def test_a_status_carrying_an_update_block_still_validates_against_the_served_schema(tmp_path: Path) -> None:
    check = vc.UpdateCheck(state=vc.STATE_UPDATE_AVAILABLE, current="1.0.0", latest="1.3.0")
    runner = _bare_runner(_runner_config(tmp_path, update_check=check))
    postern_schema.validate(runner.status(), postern_schema.load("status"), name="status")


# ---------------------------------------------------------------------------
# __main__.main()'s boot wiring
# ---------------------------------------------------------------------------


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    (root / "config").mkdir(parents=True)
    (root / "config" / "agents.yaml").write_text("agents: []\n", encoding="utf-8")
    (root / "VERSION").write_text("1.0.0", encoding="utf-8")
    return root


def _boot(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> RunnerConfig:
    """Run ``main()`` with ``serve`` stubbed and return the config it built.

    Same shape as ``test_postern_runner_home_fallback.py``'s
    ``test_main_fixes_home_before_a_bundle_ever_gets_a_chance_to_run``:
    ``--no-pull`` against a bundle that already has files on disk keeps the
    entitlement check and the pull step both off the network, so only the
    version check's own request (if any) can reach a socket.
    """
    seen: dict[str, RunnerConfig] = {}
    monkeypatch.setattr(cli, "serve", lambda config: seen.__setitem__("config", config))
    exit_code = cli.main(["--bundle", str(bundle), "--no-pull", "--port", "0"])
    assert exit_code == cli.EXIT_OK
    return seen["config"]


def test_main_skips_the_check_with_no_distributor_or_agent_id(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIGRIX_TOKEN", raising=False)
    monkeypatch.delenv("POSTERN_AGENT_ID", raising=False)
    monkeypatch.delenv("POSTERN_DISTRIBUTOR", raising=False)

    config = _boot(bundle, monkeypatch)

    assert config.update_check == vc.UpdateCheck(state=vc.STATE_NOT_REQUIRED)


def test_main_reports_current_when_the_distributor_echoes_the_running_version(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, distributor
) -> None:
    served = distributor(lambda path: (200, _good_body(agent_id=AGENT_ID, version="1.0.0")))
    monkeypatch.delenv("SIGRIX_TOKEN", raising=False)
    monkeypatch.setenv("POSTERN_AGENT_ID", AGENT_ID)
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)

    config = _boot(bundle, monkeypatch)

    assert config.update_check == vc.UpdateCheck(state=vc.STATE_CURRENT, current="1.0.0", latest="1.0.0")


def test_main_warns_and_reports_update_available_for_a_newer_version(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, distributor, caplog: pytest.LogCaptureFixture
) -> None:
    served = distributor(lambda path: (200, _good_body(agent_id=AGENT_ID, version="1.3.0")))
    monkeypatch.delenv("SIGRIX_TOKEN", raising=False)
    monkeypatch.setenv("POSTERN_AGENT_ID", AGENT_ID)
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)

    with caplog.at_level(logging.WARNING, logger="sigrix_runtime.postern"):
        config = _boot(bundle, monkeypatch)

    assert config.update_check == vc.UpdateCheck(state=vc.STATE_UPDATE_AVAILABLE, current="1.0.0", latest="1.3.0")
    assert AGENT_ID in caplog.text
    assert "1.0.0 -> 1.3.0" in caplog.text
    # The procedure the image's documentation gives for updating, not "re-pull": a
    # pull never writes over a bundle, so re-pulling into the same folder serves
    # the old one.
    assert "pull it into an empty folder" in caplog.text


def test_main_degrades_to_unreachable_without_failing_the_boot(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The container has to work offline after the initial pull.
    An unreachable version-check endpoint is therefore not a boot failure —
    ``exit_code == EXIT_OK`` (asserted inside ``_boot``) is the point."""
    monkeypatch.delenv("SIGRIX_TOKEN", raising=False)
    monkeypatch.setenv("POSTERN_AGENT_ID", AGENT_ID)
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", "http://127.0.0.1:9")

    config = _boot(bundle, monkeypatch)

    assert config.update_check == vc.UpdateCheck(state=vc.STATE_UNREACHABLE, current="1.0.0")
