# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versioning is
described in `VERSIONING.md`.

## [Unreleased]

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
