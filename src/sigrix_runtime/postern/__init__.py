"""Postern v0 for a Sigrix listing bundle — the reference implementation.

Postern (``sigrix-io/postern``) is the open execution and entitlement
protocol for packaged AI agents: four verbs over local HTTP —
``describe`` / ``run`` / ``stream`` / ``status`` — and one entitlement flow.
This package serves them for the bundle it is started inside::

    python -m sigrix_runtime.postern

It is also what the ``sigrix/runner`` image runs as its entrypoint. Those
are two deployment shapes of one codebase rather than two implementations;
if they ever diverge, that is the bug.

**Nothing here is a second execution path.** ``run`` and ``stream`` reach
the agent through ``sigrix_runtime.execution``, which is the same function
``python main.py "…"`` calls. A crew's kickoff inputs are a contract
spanning several files, and a server that built its own ``crew.kickoff``
would be one more place for it to drift.

**Zero dependencies beyond the standard library.** The bundle's
``requirements.txt`` is something a buyer installs on their own machine,
and four routes on loopback do not justify adding a web framework and its
tree to it. ``http.server`` serves them; ``http.client`` makes the one
outbound call this protocol defines.

Layout:

``errors``       the specification's error envelope, as exceptions
``describe``     the agent's contract: shipped by the distributor, or derived
``transport``    the one outbound connection, and SPEC 7's rule on it
``entitlement``  the distributor client and the state machine of SPEC 5.7
``pull``         bundle retrieval (SPEC 5.6), which is the image's boot step
``execution``    the run engine — a subprocess per run, so a timeout is real
``server``       the HTTP surface: routing, CORS, and the SSE framing
"""

from __future__ import annotations

# The specification version every payload carries. The ``/postern/v0/``
# path prefix is deliberately coarser than this (SPEC 5.3, VERSIONING.md):
# the prefix moves only on a breaking revision, and this field is the only
# sanctioned way to know exactly what you are talking to.
POSTERN_VERSION = "0.1"

# SPEC 3. This runner implements all four verbs, so it reports Level 3.
# ``stream`` emits no ``delta`` — see ``server`` for why that is conformant
# rather than a shortfall.
CONFORMANCE_LEVEL = 3

PATH_PREFIX = "/postern/v0"

DEFAULT_PORT = 8787
DEFAULT_HOST = "127.0.0.1"

__all__ = [
    "CONFORMANCE_LEVEL",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "PATH_PREFIX",
    "POSTERN_VERSION",
]
