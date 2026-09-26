"""Sigrix runtime helpers shared by every listing bundle.

Submodules:
    execution     - the one run path; main.py and postern both call it
    sandbox       - file-IO sandbox + CrewAI tool wrappers
    loader        - YAML config loader and CrewAI crew builder
    workforce     - crew-of-crews manifest loader + Flow orchestration
    quiet         - local/non-interactive run defaults
    configuration - the buyer's setup answers as a kickoff input
    runner_env    - the runner's own settings, kept out of what it runs
    postern       - the Postern v0 server (describe/run/stream/status)
"""

from __future__ import annotations

#: The package's one version, which its build reads. A bundle states its own
#: version in its ``VERSION`` file.
__version__ = "0.1.0"
