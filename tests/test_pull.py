"""The runner's half of bundle retrieval, and the image's boot sequence.

The distributor's half lives with the distributor. This is
the other end of the same wire: what a runner that holds *nothing but a
token* does with it, which is what makes `docker run -e SIGRIX_TOKEN=…
sigrix/runner acme/my-crew` on a clean machine end at a running agent.

Four properties here are the ones that pass a happy-path test and are still
wrong:

* **Nothing reaches disk before the digest agrees.** A wrong checksum, an
  entry that escapes the folder, a zip that unpacks to a hundred times its
  download — each has to leave the folder as it was, because a
  half-installed bundle looks servable.
* **Refused and unavailable are opposite answers**, exactly as they are for
  the check (SPEC 5.7). A `404` stops the boot for good; a `5xx` or a dead
  socket invites the retry that a restart already is.
* **A `404` after an *active* check is not the buyer's fault.** The two
  endpoints are allowed to disagree — this distributor licenses every
  purchasable type and packages only some for download (contract doc §12,
  decision 14) — so a runner that was just told *active* must not then tell
  its buyer their purchase may have been refunded.
* **A folder that already holds a bundle is served, never written over.**
  It holds the buyer's `.env` and their `workspace/`.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from sigrix_runtime.postern import PATH_PREFIX, pull, transport  # noqa: E402
from sigrix_runtime.postern import __main__ as cli  # noqa: E402
from sigrix_runtime.postern import entitlement as ent  # noqa: E402
from sigrix_runtime.postern.engine import Limits  # noqa: E402
from sigrix_runtime.postern.errors import WITHDRAWN_MESSAGE, access_ended  # noqa: E402
from sigrix_runtime.postern.server import RunnerConfig, build_runner  # noqa: E402
from tests import postern_distributor as fake  # noqa: E402
from tests.support import CREW_CONFIG, RUNTIME_ROOT

AGENT_ID = fake.AGENT_ID
BUNDLE_PATH = f"{PATH_PREFIX}/bundles/acme/market-research-crew"

#: The fake distributor's own fixture, re-exported so this file can ask for it
#: by name. One stand-in for SPEC 5.6, shared with the image's tests.
distributor = fake.distributor


def _pull(served: fake.Distributor, root: Path, **kwargs) -> pull.PullOutcome:
    return pull.ensure_bundle(
        root,
        base_url=served.base_url,
        token="feed-token",
        agent_id=AGENT_ID,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The happy path, and what it leaves behind
# ---------------------------------------------------------------------------


def test_a_pull_lands_a_bundle_the_runner_can_serve(distributor, tmp_path: Path) -> None:
    """The AC, minus Docker: a token and an identifier become a running agent."""
    served = distributor(fake.serving(fake.real_bundle_zip()))
    root = tmp_path / "bundle"

    outcome = _pull(served, root)

    assert outcome.pulled and outcome.verified
    assert (root / "config" / "agents.yaml").is_file()
    assert (root / "sigrix_runtime" / "postern" / "server.py").is_file()

    runner = build_runner(
        RunnerConfig(bundle_root=root, port=0, limits=Limits(), agent_id=AGENT_ID),
        {"POSTERN_AGENT_ID": AGENT_ID},
    )
    assert runner.describe()["postern"]
    assert runner.status()["agent"]["id"] == AGENT_ID


def test_the_request_is_two_path_segments_carrying_the_bearer(distributor, tmp_path: Path) -> None:
    """SPEC 5.3.1's addressing, which 5.6 adopts verbatim. Never ``%2F``."""
    served = distributor(fake.serving(fake.real_bundle_zip()))
    _pull(served, tmp_path / "bundle")

    path, authorization = served.requests[0]
    assert path == BUNDLE_PATH
    assert "%2F" not in path
    assert authorization == "Bearer feed-token"


def test_the_folder_remembers_which_agent_it_holds(distributor, tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _pull(distributor(fake.serving(fake.real_bundle_zip())), root)

    stamp = json.loads((root / pull.STAMP_FILENAME).read_text(encoding="utf-8"))
    assert stamp["agent_id"] == AGENT_ID
    assert stamp["verified"] is True
    assert pull.stamped_agent_id(root) == AGENT_ID


# ---------------------------------------------------------------------------
# The digest (SPEC 5.6), which is the only thing standing between a buyer
# and a bundle somebody else wrote
# ---------------------------------------------------------------------------


def test_a_digest_that_disagrees_leaves_the_folder_empty(distributor, tmp_path: Path) -> None:
    payload = fake.real_bundle_zip()
    served = distributor(fake.serving(payload, digest=fake.digest_header(b"a different bundle")))
    root = tmp_path / "bundle"

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, root)

    assert "checksum" in str(caught.value)
    assert not root.exists() or list(root.iterdir()) == []


def test_a_missing_digest_is_a_warning_rather_than_a_refusal(distributor, tmp_path: Path, caplog) -> None:
    """SPEC 5.6 makes ``Repr-Digest`` a SHOULD, so refusing would be stricter
    than the protocol — but the bundle is marked unverified rather than
    silently treated as checked."""
    served = distributor(fake.serving(fake.real_bundle_zip(), digest=""))
    root = tmp_path / "bundle"

    with caplog.at_level("WARNING", logger="sigrix_runtime.postern.pull"):
        outcome = _pull(served, root)

    assert outcome.pulled and not outcome.verified
    assert (root / "config").is_dir()
    assert any("no_digest" in record.message for record in caplog.records)
    assert json.loads((root / pull.STAMP_FILENAME).read_text(encoding="utf-8"))["verified"] is False


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("sha-256=:abc=:", "abc="),
        ("sha-512=:zzz:, sha-256=:abc=:", "abc="),
        ("SHA-256=:abc=:", "abc="),
        ("sha-512=:zzz:", ""),
        ("sha-256=abc=", ""),
        ("", ""),
        ("nonsense", ""),
    ],
)
def test_the_digest_header_is_read_as_the_structured_field_it_is(header: str, expected: str) -> None:
    assert pull.sha256_from_repr_digest(header) == expected


# ---------------------------------------------------------------------------
# Failure modes the story names: bad token, no entitlement, refunded, withdrawn
# ---------------------------------------------------------------------------


def test_a_404_names_the_three_causes_and_picks_none_of_them(distributor, tmp_path: Path) -> None:
    """SPEC 5.5 makes them indistinguishable at the distributor, so the
    sentence here covers all three — and it is the same sentence ``run``
    gives a client, so a buyer never sees the boot and the runner disagree."""
    served = distributor(fake.serving(fake.envelope("not_found", "Not found."), status=404))

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, tmp_path / "bundle")

    message = str(caught.value)
    assert "refunded" in message
    assert "SIGRIX_TOKEN" in message and "POSTERN_AGENT_ID" in message
    assert not isinstance(caught.value, pull.PullUnavailable)


def test_a_404_after_an_active_check_does_not_blame_the_purchase(distributor, tmp_path: Path) -> None:
    """Decision 14's reality, in plain language.

    The check answers for every purchasable type while retrieval serves only
    the types whose publish path stores a bundle, so an entitled buyer of a
    marketplace listing can be told *active* and then handed a ``404``.
    Telling them their purchase may have been refunded would send a paying
    buyer to support over a listing that is simply delivered another way.
    """
    served = distributor(fake.serving(fake.envelope("not_found", "Not found."), status=404))

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, tmp_path / "bundle", licensed=True)

    message = str(caught.value)
    assert "licensed" in message and "no bundle to hand over" in message
    assert "refunded" not in message


def test_a_410_says_when_access_ended(distributor, tmp_path: Path) -> None:
    """§5.6's SHOULD: the date rides in ``detail`` because the envelope's
    root is closed (§2.1)."""
    body = fake.envelope("withdrawn", "Gone.", {"access_ends_at": "2027-08-15T00:00:00Z"})
    served = distributor(fake.serving(body, status=410))

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, tmp_path / "bundle")

    assert "withdrawn" in str(caught.value)
    assert "2027-08-15" in str(caught.value)


def test_a_withdrawal_without_a_date_still_says_something_true(distributor, tmp_path: Path) -> None:
    served = distributor(fake.serving(fake.envelope("withdrawn", "Gone."), status=410))

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, tmp_path / "bundle")

    assert "access has ended" in str(caught.value)


@pytest.mark.parametrize("status", [500, 502, 503])
def test_a_server_error_is_unavailable_and_invites_a_retry(distributor, tmp_path: Path, status: int) -> None:
    served = distributor(fake.serving(fake.envelope("unavailable", "Later."), status=status))

    with pytest.raises(pull.PullUnavailable) as caught:
        _pull(served, tmp_path / "bundle")

    assert "Try again" in str(caught.value)


def test_a_distributor_that_never_answers_is_unavailable_not_refused(tmp_path: Path) -> None:
    """The line SPEC 5.7 draws for the check, drawn the same way here: a
    refusal is something a distributor said, not something a socket did."""
    with pytest.raises(pull.PullUnavailable):
        pull.ensure_bundle(
            tmp_path / "bundle",
            base_url="http://127.0.0.1:9",
            token="t",
            agent_id=AGENT_ID,
            timeout=0.25,
        )


def test_a_malformed_identifier_is_refused_before_any_request(tmp_path: Path) -> None:
    with pytest.raises(pull.PullRefused) as caught:
        pull.ensure_bundle(
            tmp_path / "bundle",
            base_url="https://sigrix.io",
            token="t",
            agent_id="not-an-identifier",
        )
    assert "two parts" in str(caught.value)


# ---------------------------------------------------------------------------
# The bounds — SPEC 5.6 sets none, so the runner owns them
# ---------------------------------------------------------------------------


def test_a_declared_length_over_the_bound_is_refused_before_the_body_is_read(distributor, tmp_path: Path) -> None:
    payload = fake.zip_bytes({"config/agents.yaml": b"x" * 4096})
    served = distributor(fake.serving(payload))

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, tmp_path / "bundle", max_bytes=16)

    assert "POSTERN_MAX_BUNDLE_BYTES" in str(caught.value)


def test_a_body_over_the_bound_stops_mid_read_when_no_length_was_declared(distributor, tmp_path: Path) -> None:
    """A distributor that declares nothing must not be able to send forever."""
    payload = fake.zip_bytes({"config/agents.yaml": b"x" * 200_000}, stored=True)
    assert len(payload) > 100_000, "the archive has to be big on the wire, not just once unpacked"
    served = distributor(fake.serving(payload), send_length=False, http_1_1=False)

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, tmp_path / "bundle", max_bytes=1024)

    assert "over" in str(caught.value)


def test_a_zip_that_unpacks_far_beyond_its_download_is_refused(distributor, tmp_path: Path) -> None:
    """The download bound bounds memory; this one bounds the disk. 64 MB of
    well-chosen zeroes unpack to gigabytes."""
    unpacked = pull.MIN_EXTRACT_BYTES + 1024
    payload = fake.zip_bytes({"config/agents.yaml": b"\0" * unpacked})
    assert pull.extract_budget(len(payload)) < unpacked
    served = distributor(fake.serving(payload))
    root = tmp_path / "bundle"

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, root)

    assert "unpacks to" in str(caught.value)
    assert list(root.iterdir()) == []


def test_a_small_text_bundle_is_not_refused_for_being_compressible(distributor, tmp_path: Path) -> None:
    """The floor under the ratio. A few kilobytes of YAML deflates far past
    twentyfold, and refusing a real listing for that would arrive as "my
    agent will not start"."""
    payload = fake.zip_bytes({"config/agents.yaml": b"role: researcher\n" * 4000})
    assert len(payload) * pull.EXTRACT_RATIO_LIMIT < 4000 * len(b"role: researcher\n")

    assert _pull(distributor(fake.serving(payload)), tmp_path / "bundle").pulled


# ---------------------------------------------------------------------------
# What an archive may contain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "escaping",
    ["../evil.py", "config/../../evil.py", "/etc/cron.d/evil", "..\\evil.py"],
)
def test_an_entry_that_would_write_outside_the_folder_is_refused(distributor, tmp_path: Path, escaping: str) -> None:
    """Refused rather than sanitised: ``zipfile`` would strip the ``..`` and
    write it somewhere harmless, which keeps a runner running on an artifact
    nobody should trust."""
    served = distributor(fake.serving(fake.zip_bytes({"config/agents.yaml": b"ok\n", escaping: b"pwned\n"})))
    root = tmp_path / "bundle"

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, root)

    assert "outside the folder" in str(caught.value)
    assert list(root.iterdir()) == []
    assert not (tmp_path / "evil.py").exists()


def test_a_symlink_entry_is_refused(distributor, tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("config/agents.yaml", b"ok\n")
        info = zipfile.ZipInfo("config/link")
        info.external_attr = (0o120777 << 16) | 0o20
        archive.writestr(info, "/etc/passwd")
    served = distributor(fake.serving(buffer.getvalue()))
    root = tmp_path / "bundle"

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, root)

    assert "symbolic link" in str(caught.value)
    assert list(root.iterdir()) == []


def test_something_that_is_not_a_zip_is_refused_by_name(distributor, tmp_path: Path) -> None:
    payload = b"<html>your wifi wants you to sign in</html>"
    served = distributor(fake.serving(payload))

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, tmp_path / "bundle")

    assert "not a readable zip" in str(caught.value)


# ---------------------------------------------------------------------------
# A folder that already holds something
# ---------------------------------------------------------------------------


def test_a_folder_that_already_holds_a_bundle_is_served_as_it_stands(distributor, tmp_path: Path) -> None:
    """No pull, and nothing written over: the folder holds the buyer's
    ``.env`` and their ``workspace/``."""
    root = tmp_path / "bundle"
    shutil.copytree(CREW_CONFIG, root / "config")
    (root / ".env").write_text("OPENAI_API_KEY=theirs\n", encoding="utf-8")
    served = distributor(fake.serving(fake.real_bundle_zip()))

    outcome = _pull(served, root)

    assert outcome.action == "already_present"
    assert served.requests == []
    assert (root / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=theirs\n"


def test_a_folder_holding_something_else_is_refused_before_anything_moves(distributor, tmp_path: Path) -> None:
    """Not a bundle (no ``config/``), but not empty either. Every collision
    is found before the first move, so the refusal leaves the folder exactly
    as it was rather than holding half a bundle."""
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "README.md").write_text("notes of my own\n", encoding="utf-8")
    served = distributor(fake.serving(fake.real_bundle_zip()))

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, root)

    assert "README.md" in str(caught.value)
    assert [entry.name for entry in root.iterdir()] == ["README.md"]
    assert (root / "README.md").read_text(encoding="utf-8") == "notes of my own\n"


def test_a_move_that_fails_midway_takes_back_what_it_moved(distributor, tmp_path: Path, monkeypatch) -> None:
    """The docstring promised no half-bundle; a rename failing on the third entry left two.

    Among them ``config/`` -- the marker that makes a folder look servable,
    so the next boot would have served a bundle missing the rest of itself.
    """
    root = tmp_path / "bundle"
    served = distributor(fake.serving(fake.real_bundle_zip()))
    real_rename = Path.rename
    moved: list[Path] = []

    def fails_on_the_third(self: Path, target):
        if len(moved) == 2:
            raise OSError(28, "No space left on device")
        moved.append(Path(target))
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", fails_on_the_third)
    with pytest.raises(pull.PullUnavailable):
        _pull(served, root)
    monkeypatch.undo()

    assert len(moved) == 2, "the failure came before anything had moved, which proves nothing"
    assert list(root.iterdir()) == []
    assert _pull(served, root).pulled


def test_a_zip_of_more_entries_than_a_bundle_has_is_refused(distributor, tmp_path: Path, monkeypatch) -> None:
    """Byte budgets bound what a zip unpacks to, not how many files: empty entries weigh nothing."""
    monkeypatch.setattr(pull, "MAX_BUNDLE_ENTRIES", 50)
    served = distributor(fake.serving(fake.zip_bytes({f"config/empty-{n}.txt": b"" for n in range(51)})))
    root = tmp_path / "bundle"
    with pytest.raises(pull.PullRefused, match="entries"):
        _pull(served, root)
    assert not pull.is_bundle(root)


def test_a_real_bundle_is_far_inside_the_entry_bound() -> None:
    real = zipfile.ZipFile(io.BytesIO(fake.real_bundle_zip()))
    assert len(real.infolist()) * 10 < pull.MAX_BUNDLE_ENTRIES


def test_a_volume_stamped_for_another_agent_is_refused_by_name(distributor, tmp_path: Path) -> None:
    """The same volume with a different POSTERN_AGENT_ID would otherwise
    serve the previous buyer's agent, which looks exactly like a working
    runner."""
    root = tmp_path / "bundle"
    shutil.copytree(CREW_CONFIG, root / "config")
    (root / pull.STAMP_FILENAME).write_text(json.dumps({"agent_id": "acme/another-crew"}), encoding="utf-8")
    served = distributor(fake.serving(fake.real_bundle_zip()))

    with pytest.raises(pull.PullNotConfigured) as caught:
        _pull(served, root)

    assert "acme/another-crew" in str(caught.value) and AGENT_ID in str(caught.value)
    assert served.requests == []


def test_an_unstamped_folder_is_no_claim_rather_than_a_mismatch(distributor, tmp_path: Path) -> None:
    """A buyer who unzipped their own bundle stamped nothing, and that must
    not read as *some other agent*."""
    root = tmp_path / "bundle"
    shutil.copytree(CREW_CONFIG, root / "config")

    assert _pull(distributor(fake.serving(b"")), root).action == "already_present"


def test_pulling_can_be_turned_off_entirely(distributor, tmp_path: Path) -> None:
    served = distributor(fake.serving(fake.real_bundle_zip()))

    with pytest.raises(pull.PullNotConfigured) as caught:
        _pull(served, tmp_path / "bundle", allow_pull=False)

    assert "--no-pull" in str(caught.value)
    assert served.requests == []


# ---------------------------------------------------------------------------
# What leaves the machine (the story's second AC)
# ---------------------------------------------------------------------------


def test_nothing_but_the_token_leaves_the_machine(distributor, tmp_path: Path, served_main, monkeypatch) -> None:
    """A bundle's ``.env`` is the buyer's provider keys, and a whole boot —
    check, pull, and the version check — must carry none of them
    anywhere.

    SPEC 4.1.3 gives a credential nowhere to travel, so the way this stops
    being true is not a request that sends one deliberately: it is a resolver
    that lifts the whole file into the environment of a process that makes
    requests, and then something ordinary that reads that environment. Hence
    the second assertion.
    """
    secret = "sk-provider-key-that-must-stay-here"  # noqa: S105 - a fake, and the point
    root = tmp_path / "bundle"
    root.mkdir()
    (root / ".env").write_text(
        f"SIGRIX_TOKEN=feed-token\nPOSTERN_AGENT_ID={AGENT_ID}\nOPENAI_API_KEY={secret}\n",
        encoding="utf-8",
    )
    served = distributor(fake.licensed_bundle())
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)

    assert cli.main(["--bundle", str(root)]) == cli.EXIT_OK

    assert len(served.exchanges) == 3, "a boot is one check, one pull, and one version check"
    for path, headers in served.exchanges:
        assert secret not in path
        assert secret not in json.dumps(headers)
        if "/versions/" in path:
            # SPEC 8's endpoint is deliberately unauthenticated — no
            # bearer token to carry a secret in, and none sent.
            assert "authorization" not in headers
        else:
            assert headers["authorization"] == "Bearer feed-token"
    assert secret not in json.dumps(dict(os.environ))


# ---------------------------------------------------------------------------
# One transport, one rule about tokens (SPEC 7)
# ---------------------------------------------------------------------------


def test_the_token_crosses_plaintext_only_to_a_loopback_peer(distributor, tmp_path: Path, monkeypatch) -> None:
    """The pull carries the same token as the check, so it inherits the same
    rule — and inherits it by sharing the code rather than by agreeing."""
    served = distributor(fake.serving(fake.real_bundle_zip()))
    monkeypatch.setattr(transport, "_peer_is_loopback", lambda sock: False)

    with pytest.raises(pull.PullUnavailable) as caught:
        _pull(served, tmp_path / "bundle")

    assert "plaintext" in str(caught.value)


def test_neither_caller_opens_a_connection_of_its_own() -> None:
    """A source assertion, because the behavioural one above can be satisfied
    by a second copy of the rule that happens to agree today."""
    for module in ("entitlement.py", "pull.py"):
        source = (RUNTIME_ROOT / "sigrix_runtime" / "postern" / module).read_text(encoding="utf-8")
        assert "open_response(" in source
        assert "HTTPSConnection" not in source
        assert "HTTPConnection" not in source


# ---------------------------------------------------------------------------
# The boot sequence — the image's, and the same one a local buyer meets
# ---------------------------------------------------------------------------


@pytest.fixture()
def served_main(monkeypatch):
    """``cli.main`` with the serve step recorded rather than run."""
    recorded: dict[str, Any] = {}

    def _record(config, *args, **kwargs) -> None:
        recorded["config"] = config

    monkeypatch.setattr(cli, "serve", _record)
    for name in ("SIGRIX_TOKEN", "POSTERN_AGENT_ID", "POSTERN_DISTRIBUTOR", "POSTERN_PULL"):
        monkeypatch.delenv(name, raising=False)
    return recorded


def test_the_boots_answer_is_kept_once_a_pull_has_made_the_folder(
    distributor, tmp_path: Path, served_main, monkeypatch
) -> None:
    """Pulled into a folder that did not exist, the boot's answer had nowhere to go.

    It was dropped, so a runner restarted before its first run -- or one
    whose distributor blinked -- held nothing to count grace from, and
    refused every run as never checked.
    """
    served = distributor(fake.licensed_bundle())
    monkeypatch.setenv("SIGRIX_TOKEN", "feed-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)
    root = tmp_path / "new-folder"

    assert cli.main([AGENT_ID, "--bundle", str(root)]) == cli.EXIT_OK
    kept = json.loads((root / ent.CACHE_FILENAME).read_text(encoding="utf-8"))
    assert (kept["state"], kept["agent_id"]) == ("active", AGENT_ID)
    environ = {"SIGRIX_TOKEN": "feed-token", "POSTERN_DISTRIBUTOR": served.base_url}
    assert build_runner(served_main["config"], environ).entitlement._answer is not None


def test_a_runner_named_only_by_its_argument_starts_from_its_cache(tmp_path: Path) -> None:
    """``docker run … sigrix/runner acme/my-crew`` names the listing in the config, not the environment.

    ``build_runner`` built the client from the environment and set the
    identifier afterwards -- after construction, which is when the persisted
    answer is read. So that shape never read its cache, and a container
    restarted offline was never checked rather than inside its grace.
    """
    cache = tmp_path / ent.CACHE_FILENAME
    ent.Entitlement(base_url="https://d.example", token="feed-token", agent_id=AGENT_ID, cache_path=cache)._store(
        ent.CheckAnswer(
            state=ent.STATE_ACTIVE, checked_at=ent.datetime.now(ent.UTC), stale_after_seconds=60, grace_seconds=86400
        )
    )
    environ = {"SIGRIX_TOKEN": "feed-token", "POSTERN_DISTRIBUTOR": "https://d.example"}
    gate = build_runner(RunnerConfig(bundle_root=tmp_path, port=0, agent_id=AGENT_ID), environ).entitlement
    assert gate._answer is not None
    assert gate.snapshot()["state"] == ent.STATE_ACTIVE


def test_the_listing_can_be_the_one_positional_argument(distributor, tmp_path: Path, served_main, monkeypatch) -> None:
    """``docker run … sigrix/runner acme/my-crew``, which is the shape the
    story's AC is written in."""
    served = distributor(fake.serving(fake.real_bundle_zip()))
    monkeypatch.setenv("SIGRIX_TOKEN", "feed-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)
    root = tmp_path / "bundle"

    assert cli.main([AGENT_ID, "--bundle", str(root), "--host", "0.0.0.0"]) == cli.EXIT_OK
    assert (root / "config" / "agents.yaml").is_file()
    assert served_main["config"].agent_id == AGENT_ID
    assert served_main["config"].host == "0.0.0.0"


def test_the_argument_beats_the_flag_and_the_flag_beats_the_environment(
    distributor, tmp_path: Path, served_main, monkeypatch
) -> None:
    """And what wins reaches the gate, not just the config — a check made
    about a different listing than the one being served is the disagreement
    nothing downstream could notice."""
    served = distributor(fake.serving(fake.real_bundle_zip()))
    monkeypatch.setenv("SIGRIX_TOKEN", "feed-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)
    monkeypatch.setenv("POSTERN_AGENT_ID", "acme/from-the-environment")
    root = tmp_path / "bundle"

    assert cli.main([AGENT_ID, "--agent-id", "acme/from-the-flag", "--bundle", str(root)]) == cli.EXIT_OK
    assert served_main["config"].agent_id == AGENT_ID
    gate = build_runner(served_main["config"], {"POSTERN_AGENT_ID": "acme/from-the-environment"}).entitlement
    assert gate.agent_id == AGENT_ID


def test_the_check_runs_before_the_pull(distributor, tmp_path: Path, served_main, monkeypatch) -> None:
    """§6's boot sequence in that order, and the order is what makes a 404
    legible: the check is the only thing that can tell an unlicensed buyer
    from a listing this distributor does not package.

    The version check runs last, after there is a bundle in place to
    compare — see ``_check_for_update``'s own docstring.
    """
    answers: list[str] = []

    def _handler(path: str):
        answers.append(path)
        if "/entitlements/" in path:
            return 200, {"Content-Type": "application/json"}, fake.check_body("active")
        if "/versions/" in path:
            return 200, {"Content-Type": "application/json"}, fake.version_body()
        payload = fake.real_bundle_zip()
        return (
            200,
            {"Content-Type": pull.BUNDLE_MEDIA_TYPE, pull.REPR_DIGEST_HEADER: fake.digest_header(payload)},
            payload,
        )

    served = distributor(_handler)
    monkeypatch.setenv("SIGRIX_TOKEN", "feed-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)

    assert cli.main([AGENT_ID, "--bundle", str(tmp_path / "bundle")]) == cli.EXIT_OK
    assert answers == [
        f"{PATH_PREFIX}/entitlements/acme/market-research-crew",
        BUNDLE_PATH,
        f"{PATH_PREFIX}/versions/acme/market-research-crew",
    ]


def test_what_the_check_said_reaches_the_sentence_the_pull_prints(
    distributor, tmp_path: Path, served_main, monkeypatch, capsys
) -> None:
    """Decision 14 through the real boot path, which is the only place the
    two halves are wired together.

    ``_obtain_bundle`` could pass the check's answer nowhere and every other
    test here would still pass: the check's own test only watches the order
    of the two requests, and the pull's only calls ``licensed=`` directly.
    """
    payload = fake.envelope("not_found", "Not found.")

    def _handler(path: str):
        if "/entitlements/" in path:
            return 200, {"Content-Type": "application/json"}, fake.check_body("active")
        return 404, {"Content-Type": "application/json"}, payload

    served = distributor(_handler)
    monkeypatch.setenv("SIGRIX_TOKEN", "feed-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)

    assert cli.main([AGENT_ID, "--bundle", str(tmp_path / "bundle")]) == cli.EXIT_REFUSED

    printed = capsys.readouterr().err
    assert "no bundle to hand over" in printed
    assert "refunded" not in printed


def test_a_revoked_licence_still_boots_because_status_is_how_a_buyer_finds_out(
    distributor, tmp_path: Path, served_main, monkeypatch, caplog
) -> None:
    """The check's ``404`` is a revocation (SPEC 5.7.4), and a container that
    exited on it would leave a restart loop where an answer should be. The
    bundle is already there, so there is something to serve; ``run`` refuses
    on its own."""
    root = tmp_path / "bundle"
    shutil.copytree(CREW_CONFIG, root / "config")
    served = distributor(lambda path: (404, {"Content-Type": "application/json"}, fake.envelope("not_found", "No.")))
    monkeypatch.setenv("SIGRIX_TOKEN", "feed-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)

    with caplog.at_level("WARNING", logger="sigrix_runtime.postern"):
        assert cli.main([AGENT_ID, "--bundle", str(root)]) == cli.EXIT_OK

    assert any("not licensed" in record.message for record in caplog.records)
    assert served_main["config"].bundle_root == root


@pytest.mark.parametrize(
    ("state", "ends_at", "said"),
    [
        ("revoked", "2026-01-01T00:00:00Z", "ended on 2026-01-01"),
        ("active", "2099-01-01T00:00:00Z", "ends on 2099-01-01"),
    ],
)
def test_the_boot_says_when_a_withdrawn_listings_access_ends(
    distributor, tmp_path: Path, served_main, monkeypatch, caplog, state: str, ends_at: str, said: str
) -> None:
    """SPEC 5.3's date, said at boot whichever side of it the runner is.

    After it the boot still does not exit, for the reason the test above
    gives, and it no longer reads as the refund-or-token refusal either.
    """
    root = tmp_path / "bundle"
    shutil.copytree(CREW_CONFIG, root / "config")

    def _handler(path: str):
        if "/entitlements/" in path:
            return 200, {"Content-Type": "application/json"}, fake.check_body(state, access_ends_at=ends_at)
        return 404, {"Content-Type": "application/json"}, fake.envelope("not_found", "No.")

    served = distributor(_handler)
    monkeypatch.setenv("SIGRIX_TOKEN", "feed-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", served.base_url)

    with caplog.at_level("WARNING", logger="sigrix_runtime.postern"):
        assert cli.main([AGENT_ID, "--bundle", str(root)]) == cli.EXIT_OK

    said_at_boot = [record.getMessage() for record in caplog.records]
    assert any(WITHDRAWN_MESSAGE in line and said in line for line in said_at_boot), said_at_boot
    assert not any("not licensed" in line for line in said_at_boot)


def test_the_pulls_410_and_the_runs_403_open_with_one_sentence(distributor, tmp_path: Path) -> None:
    """the boot and the run meet the same withdrawal, so they name it alike."""
    body = fake.envelope("withdrawn", "Gone.", {"access_ends_at": "2027-08-15T00:00:00Z"})
    served = distributor(fake.serving(body, status=410))

    with pytest.raises(pull.PullRefused) as caught:
        _pull(served, tmp_path / "bundle")

    assert str(caught.value).startswith(WITHDRAWN_MESSAGE)
    assert access_ended("2027-08-15").message.startswith(WITHDRAWN_MESSAGE)


def test_the_exit_code_says_which_kind_of_problem_it_is(distributor, tmp_path: Path, served_main, monkeypatch) -> None:
    """An operator staring at a container that will not stay up has the exit
    code and nothing else."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["--bundle", str(empty)]) == cli.EXIT_NOT_CONFIGURED

    refusing = distributor(lambda path: (404, {"Content-Type": "application/json"}, fake.envelope("not_found", "No.")))
    monkeypatch.setenv("SIGRIX_TOKEN", "feed-token")
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", refusing.base_url)
    assert cli.main([AGENT_ID, "--bundle", str(tmp_path / "refused")]) == cli.EXIT_REFUSED

    monkeypatch.setenv("POSTERN_DISTRIBUTOR", "http://127.0.0.1:9")
    assert cli.main([AGENT_ID, "--bundle", str(tmp_path / "unreachable")]) == cli.EXIT_UNAVAILABLE

    assert "config" not in served_main


def test_the_message_for_an_empty_folder_still_tells_a_local_buyer_what_to_do(
    tmp_path: Path, served_main, capsys
) -> None:
    """The commonest reader of it is somebody who ran the module in the wrong
    folder, not a container missing a variable."""
    empty = tmp_path / "empty"
    empty.mkdir()

    assert cli.main(["--bundle", str(empty)]) == cli.EXIT_NOT_CONFIGURED

    printed = capsys.readouterr().err
    assert "does not look like a Sigrix bundle" in printed
    assert "--bundle path/to/bundle" in printed
    assert "SIGRIX_TOKEN" in printed


def test_pulling_off_is_a_flag_and_an_environment_variable(tmp_path: Path, served_main, monkeypatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("SIGRIX_TOKEN", "feed-token")
    monkeypatch.setenv("POSTERN_AGENT_ID", AGENT_ID)
    monkeypatch.setenv("POSTERN_DISTRIBUTOR", "http://127.0.0.1:9")

    assert cli.build_parser().parse_args(["--no-pull"]).no_pull is True
    monkeypatch.setenv("POSTERN_PULL", "0")
    assert cli.build_parser().parse_args([]).no_pull is True
    assert cli.main(["--bundle", str(empty)]) == cli.EXIT_NOT_CONFIGURED


def test_the_bound_is_a_flag_whose_default_is_the_environment(monkeypatch) -> None:
    """One reader of POSTERN_MAX_BUNDLE_BYTES, and it is this flag's default."""
    monkeypatch.delenv("POSTERN_MAX_BUNDLE_BYTES", raising=False)
    assert cli.build_parser().parse_args([]).max_bundle_bytes == pull.MAX_BUNDLE_BYTES
    monkeypatch.setenv("POSTERN_MAX_BUNDLE_BYTES", "1024")
    assert cli.build_parser().parse_args([]).max_bundle_bytes == 1024
    assert cli.build_parser().parse_args(["--max-bundle-bytes", "77"]).max_bundle_bytes == 77


def test_the_three_settings_resolve_once_for_the_check_and_the_pull(tmp_path: Path, monkeypatch) -> None:
    """``settings_from_environment`` is what both read, so a bundle's ``.env``
    and an exported variable cannot mean one thing at boot and another at run."""
    root = tmp_path / "bundle"
    root.mkdir()
    (root / ".env").write_text(
        "SIGRIX_TOKEN=from-the-file\nPOSTERN_AGENT_ID=acme/from-the-file\nOPENAI_API_KEY=not-this-runners-business\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SIGRIX_TOKEN", raising=False)
    monkeypatch.delenv("POSTERN_AGENT_ID", raising=False)
    monkeypatch.delenv("POSTERN_DISTRIBUTOR", raising=False)

    settings = ent.settings_from_environment(root)
    assert settings.token == "from-the-file"
    assert settings.agent_id == "acme/from-the-file"
    assert settings.base_url == ent.DEFAULT_DISTRIBUTOR

    monkeypatch.setenv("SIGRIX_TOKEN", "exported")
    assert ent.settings_from_environment(root).token == "exported"
    assert ent.from_environment(root).token == "exported"
