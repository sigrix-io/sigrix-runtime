"""Quiet-by-default runtime posture for a purchased bundle.

A purchased bundle must not prompt interactively mid-run, must not ship
the buyer's questions or the crew's internals to a third party by
default, and must not flip persistent preferences on its own. On a
fresh machine CrewAI does all three unless told otherwise:

- first run: an interactive "view your execution traces?" prompt
  (20-second timeout) followed by a trace upload to CrewAI's servers;
- every run: anonymous OpenTelemetry usage telemetry;
- afterwards: tracing consent recorded machine-wide without being asked.

``apply_quiet_env_defaults`` sets the environment knobs the pinned
crewai version honours. It uses ``os.environ.setdefault``, so anything
the buyer exports or puts in ``.env`` (loaded before this runs) wins —
opting in deliberately stays a one-line edit: set
``CREWAI_TRACING_ENABLED=true`` in ``.env``.
"""

from __future__ import annotations

import os

QUIET_ENV_DEFAULTS: dict[str, str] = {
    # Run tracing (event upload to CrewAI's servers + the "view traces?"
    # flow): off unless the buyer opts in. In the pinned crewai only
    # "true"/"1" enables; this is the one buyer-facing knob, documented
    # in .env.example and the README.
    "CREWAI_TRACING_ENABLED": "false",
    # The first-run trace auto-collection and its interactive consent
    # prompt are NOT governed by CREWAI_TRACING_ENABLED=false — in the
    # pinned crewai the only reliable off-switch is the test-environment
    # flag (crewai.events.listeners.tracing.utils._is_test_environment).
    # It has no other effect in the pinned version, and an explicit
    # CREWAI_TRACING_ENABLED=true still traces with it set.
    "CREWAI_TESTING": "true",
    # Anonymous usage telemetry (OpenTelemetry spans to CrewAI's
    # collector). Two switches, one honoured by crewai and one by the
    # OTel SDK itself; deliberate tracing opt-in uses neither (it posts
    # through CrewAI's own API client).
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    # chromadb (pulled in by crewai for its RAG features; the web_search
    # tool constructs a chroma client) posts product telemetry of its own.
    "ANONYMIZED_TELEMETRY": "false",
}


def apply_quiet_env_defaults() -> None:
    """Apply the quiet defaults without overriding buyer-set values."""
    for key, value in QUIET_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
