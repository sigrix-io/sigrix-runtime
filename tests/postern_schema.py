"""The specification's own JSON Schemas, so a test asserts the contract itself.

The schemas are the ones ``postern-conformance`` ships, which are copied from
the specification's repository when that checker is built. Asserting a payload
against them beats restating their rules in a test: a schema that moves
upstream moves the assertion with it, at the next pin of the checker, instead
of leaving a test quietly checking last month's contract.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from postern_conformance import schemas


class SchemaViolation(AssertionError):
    """A payload that does not satisfy the schema, with the path that failed."""


def load(name: str) -> dict[str, Any]:
    """The schema named ``name``, e.g. ``"describe"`` for ``describe.schema.json``."""
    return schemas.load(f"{name}.schema.json")


def validate(payload: Any, schema: dict[str, Any], *, name: str = "") -> None:
    """Raise :class:`SchemaViolation` unless ``payload`` satisfies ``schema``."""
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(part) for part in first.absolute_path)
        raise SchemaViolation(f"{name or '$'}{'/' + where if where else ''}: {first.message}")
