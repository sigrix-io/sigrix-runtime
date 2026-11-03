# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versioning is
described in `VERSIONING.md`.

## [Unreleased]

### Added

- A run's `usage`, and each of its steps, reports how much of its input the
  provider's prompt cache wrote and read: `cache_write_tokens` and
  `cache_read_tokens`, under `org.sigrix`. A provider that caches prices the
  three kinds of input apart (Anthropic charges a cache write at 1.25 times
  its input rate and a cache read at a tenth of it), so a client pricing a run
  needs the split. Both are counts inside `input_tokens`, which keeps its
  meaning, and each is present only when CrewAI reported it. Postern's `usage`
  has no member for them, so they ride under the namespace `files_written`
  does, which a client that does not know it ignores. `cost_usd` is unchanged:
  it still prices every input token at the model's input rate.
- A workforce run reports what it spent. It reported no step and no usage
  before, so a client pricing runs priced every workforce run at nothing.
  Each task of each unit crew is a step named `<unit key>.<task>`, as
  `describe` lists it, and each of the workforce's own crews is one too:
  `workforce-manager.route`, `workforce-manager.answer` and
  `workforce-synthesizer.synthesize`, a hyphen keeping them apart from any
  unit's. Every step carries its tokens and its cache split, is timed on its
  own crew's clock (a panel runs its units at once), and streams as it
  finishes. The run's `usage` is every crew's totals summed, and is left out,
  never short, when a crew cannot say what it spent. Its `model_id`, and so its
  `cost_usd`, is the model the crews resolved, as a single crew's is.

### Fixed

- A crew's first step reported, as its `latency_ms`, the time the runner
  spent loading the bundle's configuration, importing CrewAI and building the
  crew as well as the task's own. SPEC 4.3 defines a step's latency as its own
  elapsed time, so each crew is now timed from its own kickoff.

### Changed

- The README opens with badges for the PyPI release, the Python versions, CI
  and the licence, and a new *Where it fits* section names the projects that
  meet at this runner: Postern, the checker CI holds it to, Gatehouse and
  sigrix-launcher. The README is the PyPI page too, so the next release
  carries it there. Nothing a client or a buyer sees changes.
- The README carries the project's logo above its title, and readers in
  GitHub's dark mode are shown a version drawn for it. The README is the PyPI
  page too, so the next release carries the logo there. Nothing in the package
  changes.
- Dependabot opens one pull request per ecosystem instead of one per
  dependency. The branch ruleset only merges a pull request that is up to date
  with `main`, so each separate update merged put every other one behind.
  Nothing in the package changes.

## [0.1.0] - 2026-09-26

First release as a package of its own. The same code has shipped inside every
Sigrix bundle before this; nothing it answers has changed.

### Added

- `sigrix_runtime.postern`: a Postern 0.1 runner at Level 3. `describe`,
  `status`, `run` and `stream` on loopback; the entitlement check of the
  specification's §5 with its cache and grace period; the verified bundle
  pull; the version check of §8.
- `--mcp`: an MCP server's tools served as one agent, `describe` from
  `tools/list` and each run a `tools/call`.
- The execution layer a bundle's `main.py` and the server share, and the
  `crewai` extra it needs to run a CrewAI crew.
- `sigrix-runtime`, a console script for the same entry point as
  `python -m sigrix_runtime.postern`.
- Type information (`py.typed`), checked under `mypy --strict`.
