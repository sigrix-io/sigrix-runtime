## What this changes

<!-- One or two sentences. The diff shows what changed; say why. -->

## Does a client or a buyer see it?

<!--
Delete the rows that do not apply.

- [ ] No: refactor, docs, tests or tooling only.
- [ ] Yes, and it is a fix: the runner disagreed with the specification. Name
      the section, so the changelog entry can.
- [ ] Yes, and it changes what a verb answers, or an option or variable a
      buyer sets. CONTRIBUTING.md calls that a breaking change for somebody's
      client or container; the changelog says so.
-->

## Have you watched the new test fail?

<!--
One change per pull request, with a test that fails without it. Break the
implementation on purpose, confirm the test that names the claim goes red,
then put it back. Say what you broke and which test caught it.
-->

## Checks

- [ ] `python -m pytest`
- [ ] `python -m ruff check src/ tests/`
- [ ] `python -m ruff format --check src/ tests/`
- [ ] `python -m mypy`
- [ ] The server still imports nothing outside the standard library
- [ ] `CHANGELOG.md` has a line under *Unreleased*, if a client or a buyer would notice
