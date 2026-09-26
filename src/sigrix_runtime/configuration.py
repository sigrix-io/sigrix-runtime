"""The buyer's setup answers as a kickoff input.

A crew's Step 5 variables are its **one canonical configuration**. The
seller's values are baked
into the reviewed YAML when it is generated; the *buyer's* answers ride the
run-input channel instead — never a per-buyer YAML rewrite, so *reviewed bytes
= shipped bytes = run-captured bytes* still holds and ``run_fingerprint`` keeps
meaning what it meant.

``main.py`` reads the bundle's ``variables.json`` and passes this block beside
the brief::

    crew.kickoff(inputs={"prompt": brief, "configuration": block})

and the compiled task descriptions carry a ``{configuration}`` slot next to
``{prompt}`` (``crew_config_compiler.RUN_INPUT_LINE``).

**The block carries its own separator.** It is ``""`` when nothing is filled
in, and otherwise *opens with a blank line* — the slot sits directly after the
brief (``The user's request: {prompt}{configuration}``), so a buyer who has
answered nothing gets a task description byte-for-byte identical to what the
same YAML rendered before this input existed. A missing ``variables.json``, an
empty one, and one whose values are all blank are therefore indistinguishable
at run time, which is the compatibility promise the decision record makes.

Only the wizard's ``variables`` reach the run. The sibling ``platform`` key
names the chat platform ``crew.md`` was compiled for — it says nothing about a
local CrewAI run, so it is deliberately left out of the block.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VARIABLES_FILENAME = "variables.json"

# Opens the block. Phrased as an instruction because the crew reads it: the
# seller's defaults are already baked into the YAML around it, so these are
# the values that should win where they overlap.
CONFIGURATION_HEADING = "The buyer's setup values for this run — prefer these where they apply:"


def load_variables(bundle_root: Path) -> dict[str, str]:
    """The buyer's answers from ``variables.json``, or ``{}``.

    Looks next to ``main.py`` first (the crew bundle's runnable-first layout,
    where the runtime and ``variables.json`` share the bundle root), then one
    directory up (the legacy mega-prompt layout, where the runtime lives under
    ``runnable/`` and ``variables.json`` stays at the bundle root).

    Never raises: a missing, unreadable or malformed file is the same answer as
    an unanswered wizard — no configuration — because a bad file must not cost
    the buyer their run.
    """
    for candidate in (bundle_root / VARIABLES_FILENAME, bundle_root.parent / VARIABLES_FILENAME):
        try:
            if not candidate.is_file():
                continue
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        variables = raw.get("variables")
        return _coerce_variables(variables if isinstance(variables, dict) else {})
    return {}


def _coerce_variables(raw: Mapping[str, Any]) -> dict[str, str]:
    """Scalar ``key -> value`` pairs, blanks dropped, file order preserved."""
    values: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name or isinstance(value, (dict, list)) or value is None or isinstance(value, bool):
            continue
        text = str(value).strip()
        if text:
            values[name] = text
    return values


def render_configuration_block(variables: Mapping[str, Any]) -> str:
    """Heading + ``- key: value`` lines, or ``""`` when nothing is filled in.

    A non-empty block opens with a blank line so the ``{configuration}`` slot
    can sit flush against ``{prompt}`` — see the module docstring for why that
    is what makes the empty case degrade byte-for-byte.
    """
    values = _coerce_variables(variables or {})
    if not values:
        return ""
    lines = [CONFIGURATION_HEADING]
    for key, value in values.items():
        head, _, rest = value.partition("\n")
        lines.append(f"- {key}: {head}")
        # Continuation lines are indented under their bullet so a multi-line
        # answer cannot be misread as the start of another variable.
        lines.extend(f"  {line}" for line in rest.splitlines())
    return "\n\n" + "\n".join(lines)


__all__ = [
    "CONFIGURATION_HEADING",
    "VARIABLES_FILENAME",
    "load_variables",
    "render_configuration_block",
]
