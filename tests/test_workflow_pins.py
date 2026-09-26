"""Every action this repository runs is pinned to a commit. This is what notices if one stops being.

A `uses:` line names code that runs with this repository's checkout, and in one
case with upload rights to PyPI. A tag -- `@v7`, or `@release/v1`, which is not
even a tag but a *branch* -- is moved by whoever owns the action, so the code
that ran on a release could change with nothing here recording it, and the diff
of the run that shipped would read identically to the one before.

Pinning the commit closes that. The cost is that a pin only stays correct if
something bumps it, which is what the `github-actions` entry in
`.github/dependabot.yml` is for -- and Dependabot reads the `# vX.Y.Z` comment
beside each SHA to know what it is bumping from, so a pin written without one
goes quietly stale. Both halves are asserted below.

Read as text rather than parsed: the assertions are about *how the reference is
written*, which a YAML loader normalises away.

This checks the repository rather than the package, so the sdist, which carries
no workflows, leaves it out (`pyproject.toml`).
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = sorted((Path(__file__).resolve().parents[1] / ".github" / "workflows").glob("*.yml"))

#: `uses: owner/action@<ref>`, with whatever trailing comment.
_USES = re.compile(r"uses:\s*(\S+)@(\S+)(?:\s*#\s*(\S+))?")

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _references() -> list[tuple[Path, str, str, str]]:
    found = []
    for workflow in WORKFLOWS:
        for action, ref, comment in _USES.findall(workflow.read_text(encoding="utf-8")):
            found.append((workflow, action, ref, comment or ""))
    return found


def test_the_pattern_matches_the_way_these_workflows_write_it() -> None:
    """Guards the guard: a regex matching nothing would pass every assertion below."""

    assert _USES.findall("      - uses: actions/checkout@v7\n") == [("actions/checkout", "v7", "")]
    assert _USES.findall("- uses: a/b@abc # v1.2.3\n") == [("a/b", "abc", "v1.2.3")]
    assert WORKFLOWS, "no workflows found; the assertions below would be vacuous"
    assert _references(), "no `uses:` found; the assertions below would be vacuous"


def test_every_action_is_pinned_to_a_commit() -> None:
    for workflow, action, ref, _ in _references():
        assert _COMMIT_SHA.match(ref), (
            f"{workflow.name} uses {action} at {ref!r}. A tag is mutable and a branch moves "
            "on its own; pin the 40-character commit."
        )


def test_every_pin_says_which_release_it_is() -> None:
    """A bare SHA is safe and unreadable. The trailing comment is what a reviewer
    reads, and what Dependabot rewrites when it bumps the pin."""

    for workflow, action, ref, comment in _references():
        assert re.fullmatch(r"v\d+\.\d+(\.\d+)?", comment), (
            f"{workflow.name} pins {action} at {ref[:12]}… with the comment {comment!r}; "
            "write the release it is, as `# v1.14.2`."
        )


def test_the_release_stores_no_pypi_credential() -> None:
    """The property the publish pin exists to protect.

    Trusted publishing is why there is no token here to leak, and a `secrets.`
    reference appearing in this file would mean that decision was quietly
    reversed -- at which point the third-party publish step is handling a
    long-lived credential rather than a short-lived identity.
    """

    release = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
    source = release.read_text(encoding="utf-8")
    offenders = [
        line.strip() for line in source.splitlines() if "secrets." in line and not line.lstrip().startswith("#")
    ]

    assert offenders == [], (
        f"{release.name} reads a secret: {offenders}. Releases publish through OIDC; see VERSIONING.md."
    )
    assert "id-token: write" in source, (
        "the OIDC permission is gone, so the claim above is no longer what makes this work"
    )
