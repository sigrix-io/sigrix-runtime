"""Tests for the runtime's quiet-by-default posture.

A purchased bundle must not prompt interactively, must not ship buyer
content to a third party by default, and must not flip persistent
preferences on its own. ``sigrix_runtime.quiet`` owns the env knobs;
``main.py`` applies them after ``.env`` (buyer wins) and
``loader.build_crew`` applies them for every other construction path
(the server-side example runner, tests).
"""

from __future__ import annotations

from sigrix_runtime import quiet  # noqa: E402
from tests.support import RUNTIME_ROOT


def test_quiet_defaults_cover_tracing_first_run_and_telemetry() -> None:
    # The three behaviors from the first real buyer run: run tracing
    # (upload), the first-run interactive consent flow, and OTel telemetry —
    # plus chromadb's own product telemetry.
    assert quiet.QUIET_ENV_DEFAULTS["CREWAI_TRACING_ENABLED"] == "false"
    assert quiet.QUIET_ENV_DEFAULTS["CREWAI_TESTING"] == "true"
    assert quiet.QUIET_ENV_DEFAULTS["CREWAI_DISABLE_TELEMETRY"] == "true"
    assert quiet.QUIET_ENV_DEFAULTS["OTEL_SDK_DISABLED"] == "true"
    assert quiet.QUIET_ENV_DEFAULTS["ANONYMIZED_TELEMETRY"] == "false"


def test_apply_sets_every_default_when_unset(monkeypatch) -> None:
    for key in quiet.QUIET_ENV_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    quiet.apply_quiet_env_defaults()
    import os

    for key, value in quiet.QUIET_ENV_DEFAULTS.items():
        assert os.environ[key] == value


def test_apply_never_overrides_a_buyer_opt_in(monkeypatch) -> None:
    # The documented opt-in path: CREWAI_TRACING_ENABLED=true in .env is
    # loaded before the defaults run, and must win.
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    quiet.apply_quiet_env_defaults()
    import os

    assert os.environ["CREWAI_TRACING_ENABLED"] == "true"


def test_the_run_path_applies_quiet_defaults_after_dotenv_before_crew_construction() -> None:
    # The run path needs crewai installed to execute, so pin the ordering
    # contract at source level: .env first (buyer values win), the quiet
    # defaults next, crew construction last.
    #
    # Read from execution.py rather than a bundle's main.py: the command line
    # and the Postern server are two front ends over one run path, so this is
    # where the ordering lives — and it now covers the server too, which is
    # the point of there being one.
    source = (RUNTIME_ROOT / "sigrix_runtime" / "execution.py").read_text(encoding="utf-8")
    load_env = source.index("load_dotenv(")
    apply_quiet = source.index("quiet.apply_quiet_env_defaults()")
    build = source.index("loader.build_crew(")
    assert load_env < apply_quiet < build


def test_no_front_end_can_start_a_run_without_going_through_that_ordering() -> None:
    """The ordering is only a contract while every entry point shares it.

    A front end that loaded ``.env`` itself, or built a crew of its own, would
    be free to get this wrong — and a source assertion about ``execution.py``
    could not see it. The server's two modules are checked by absence here; the
    ``main.py`` a bundle carries is checked where bundles are built.
    """
    for name in ("sigrix_runtime/postern/server.py", "sigrix_runtime/postern/engine.py"):
        source = (RUNTIME_ROOT / name).read_text(encoding="utf-8")
        assert "load_dotenv(" not in source, name
        assert "build_crew(" not in source, name


def test_build_crew_is_a_quiet_chokepoint() -> None:
    # Every crew-construction path (buyer main.py, the server-side example
    # runner) must inherit the posture even if the caller forgot: build_crew
    # applies the defaults before importing crewai.
    source = (RUNTIME_ROOT / "sigrix_runtime" / "loader.py").read_text(encoding="utf-8")
    build_crew_body = source[source.index("def build_crew(") :]
    apply_quiet = build_crew_body.index("apply_quiet_env_defaults()")
    crewai_import = build_crew_body.index("from crewai import")
    assert apply_quiet < crewai_import
