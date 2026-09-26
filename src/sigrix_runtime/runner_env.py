"""The variables that configure the runner rather than the agent, which no run may see.

``SIGRIX_TOKEN`` above all. Prefixes, so a setting added later is
covered by being added. ``execution.prepare_environment`` drops them after
``load_dotenv`` and ``postern.engine.Engine._spawn`` keeps them out of the
worker. Apart from ``execution`` so an MCP run strips them without
PyYAML.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

RUNNER_ENV_PREFIXES: tuple[str, ...] = ("POSTERN_", "SIGRIX_")


def is_runner_setting(name: str) -> bool:
    """Is *name* the runner's own configuration rather than the agent's?"""
    return name.startswith(RUNNER_ENV_PREFIXES)


def without_runner_settings(environ: Mapping[str, str]) -> dict[str, str]:
    """*environ* without the runner's own settings, for building a child's."""
    return {name: value for name, value in environ.items() if not is_runner_setting(name)}


def drop_runner_settings() -> tuple[str, ...]:
    """Remove the runner's own settings from this process, naming them."""
    dropped = tuple(sorted(name for name in list(os.environ) if is_runner_setting(name)))
    for name in dropped:
        del os.environ[name]
    return dropped


__all__ = ["RUNNER_ENV_PREFIXES", "drop_runner_settings", "is_runner_setting", "without_runner_settings"]
