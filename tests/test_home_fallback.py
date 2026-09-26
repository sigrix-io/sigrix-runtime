"""``HOME`` has to be writable before anything gets a chance to write into it.

The first real buyer-shaped run of ``sigrix/runner`` followed the image's
own documented advice for a bind mount that keeps its host ownership — ``--user
"$(id -u):$(id -g)"`` — and died on the first ``run``:

    PermissionError: [Errno 13] Permission denied: '/.local'

An arbitrary uid has no entry in the container's ``/etc/passwd``, so the
container runtime resolves ``HOME`` to ``/`` — owned by root, unwritable to
anyone else — and CrewAI writes into ``~/.local/…`` on first use with no
``try/except`` of its own.

``sigrix_runtime.postern.__main__._ensure_writable_home`` is the fix, run once
at boot before anything else gets a chance to write: a plain ``os.environ``
mutation, so a real container adds nothing this file cannot already see.
The image's own suite is where the same fix is proven inside a real image,
under a real arbitrary uid.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest

from sigrix_runtime.postern import __main__ as cli  # noqa: E402


def test_a_writable_home_is_left_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    cli._ensure_writable_home()

    assert os.environ["HOME"] == str(tmp_path)


def test_an_unset_home_falls_back_to_a_writable_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOME", raising=False)

    cli._ensure_writable_home()

    assert os.environ["HOME"] == tempfile.gettempdir()
    assert os.access(os.environ["HOME"], os.W_OK)


def test_a_home_pointed_at_nothing_falls_back_to_a_writable_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not the production shape (that is ``/``, which exists) but the other
    way a bogus ``HOME`` reaches this function — an operator's own ``-e
    HOME=…`` typo, or a chroot with no home tree at all."""
    monkeypatch.setenv("HOME", "/this/path/does/not/exist/on/any/machine")

    cli._ensure_writable_home()

    assert os.environ["HOME"] == tempfile.gettempdir()


def test_the_5327_shape_home_exists_but_this_uid_cannot_write_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact reported condition: ``HOME`` resolves to a real, existing
    directory (``/``, in production — the container runtime's own answer for
    an arbitrary uid with no ``/etc/passwd`` entry) that this process cannot
    write to.

    A CI sandbox commonly runs as root, which can write to ``/`` regardless
    of its mode — verified directly against this one (``os.access("/",
    os.W_OK)`` answers ``True`` here) — so the permission *check* is patched
    directly rather than relying on filesystem bits a root test process would
    bypass either way. The real, unprivileged uid this container ships for
    (``useradd --uid 10001``) and the arbitrary host uid ``--user`` supplies
    are both refused by the kernel for real; only this sandbox's own root
    shell is not.
    """
    monkeypatch.setenv("HOME", "/")
    monkeypatch.setattr(cli.os, "access", lambda *_args, **_kwargs: False)

    cli._ensure_writable_home()

    assert os.environ["HOME"] == tempfile.gettempdir()


def test_an_unwritable_home_says_so_and_names_where_it_moved(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("HOME", raising=False)

    with caplog.at_level(logging.WARNING, logger="sigrix_runtime.postern"):
        cli._ensure_writable_home()

    assert "HOME" in caplog.text
    assert tempfile.gettempdir() in caplog.text


def test_main_fixes_home_before_a_bundle_ever_gets_a_chance_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run subprocess inherits ``os.environ`` verbatim
    (``Engine._spawn`` copies it wholesale), so the fix only has to land in
    *this* process's environment before ``serve()`` hands control over — proven
    here by stopping ``main()`` at the door of ``serve()`` and reading what it
    would have handed a run.

    A bind mount that already holds a bundle (the image's documented
    second recipe, and the one ``--user`` is documented for) never touches
    the network on boot, which is what keeps this a fast, offline test of the
    same code path a real container runs.
    """
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("SIGRIX_TOKEN", raising=False)
    monkeypatch.delenv("POSTERN_AGENT_ID", raising=False)
    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "agents.yaml").write_text("agents: []\n", encoding="utf-8")

    seen: dict[str, str | None] = {}

    def _fake_serve(_config: object) -> None:
        seen["home"] = os.environ.get("HOME")

    monkeypatch.setattr(cli, "serve", _fake_serve)

    exit_code = cli.main(["--bundle", str(bundle), "--no-pull", "--port", "0"])

    assert exit_code == cli.EXIT_OK
    assert seen["home"] == tempfile.gettempdir()
