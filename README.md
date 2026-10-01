# sigrix-runtime

[![PyPI](https://img.shields.io/pypi/v/sigrix-runtime)](https://pypi.org/project/sigrix-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/sigrix-runtime)](https://pypi.org/project/sigrix-runtime/)
[![CI](https://github.com/sigrix-io/sigrix-runtime/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sigrix-io/sigrix-runtime/actions/workflows/ci.yml)
[![Licence](https://img.shields.io/github/license/sigrix-io/sigrix-runtime)](https://github.com/sigrix-io/sigrix-runtime/blob/main/LICENSE)

The [Postern](https://github.com/sigrix-io/postern) runner inside every bundle
[Sigrix](https://sigrix.io) delivers. It serves one agent on your own machine
over Postern's four verbs, checks that you are entitled to run it, and runs it
with the keys in your own environment.

This is the code that already ships in every Sigrix bundle, published here so it
can be read, tested and depended on. A bundle carries its own copy, so running
one needs nothing installed from here:

```console
$ cd my-bundle
$ python -m sigrix_runtime.postern
```

then point a client at `http://127.0.0.1:8787/postern/v0/describe`. Install it
to serve an MCP server's tools (below), or to build on it:

```console
$ pip install sigrix-runtime
```

Python 3.11 or newer. The server is standard library only.

## What it serves

| Verb | Route | Answers |
|---|---|---|
| `describe` | `GET /postern/v0/describe` | what the agent takes and returns, which credentials it needs, which tools cost money |
| `status` | `GET /postern/v0/status` | whether it is ready, and whether you may still run it |
| `run` | `POST /postern/v0/run` | the result |
| `stream` | `POST /postern/v0/stream` | the run as it goes, as server-sent events |

It meets [Level 3](https://github.com/sigrix-io/postern/blob/main/SPEC.md#3-conformance)
of the specification: all four verbs, the entitlement check of §5 and the version
check of §8. CI holds it to that with
[`postern-conformance`](https://pypi.org/project/postern-conformance/), run against
the runner itself rather than a stand-in.

One case falls short, and is said rather than hidden. An MCP server with
several tools is served as one agent whose `tool` input picks among them, and
`describe` has no way to declare a tool's own inputs required only when that
tool is chosen, so a request that satisfies `describe` can still be refused with
`bad_request` naming the missing input. Served with one tool, which is what CI
checks, the runner is conformant.

## What it does at start

    resolve the token → check the entitlement → pull the bundle → verify its digest → serve

A bundle already unzipped skips the middle two steps, because there is nothing
to fetch. The container image, `sigrix/runner`, runs this same module as its
entrypoint, so the two cannot drift.

The entitlement is checked with the distributor at start and again once the
answer is older than the bound the distributor declared. A distributor that
cannot be reached is not a refusal: a held answer lasts through the grace period
the distributor declared, a restart does not start a new one, and past it the
runner answers `unavailable`, which invites a retry. A refusal it has been given
keeps refusing, reachable or not.

## Configuration

Read from the process environment first and from the bundle's `.env` second, so
a container can export them and a buyer can fill in the `.env` their bundle came
with. An exported value always wins.

| Variable | Meaning |
|---|---|
| `SIGRIX_TOKEN` | a runner token for this listing |
| `POSTERN_AGENT_ID` | which listing the bundle is, as `{seller}/{listing-id}` |
| `POSTERN_DISTRIBUTOR` | the distributor's base URL; `https://sigrix.io` if unset |
| `POSTERN_INBOUND_TOKEN` | when set, every request must carry it as a bearer token |

The listing can also be named as the one positional argument, which beats both:
`python -m sigrix_runtime.postern acme/my-crew`.

| Option | Default |
|---|---|
| `--bundle PATH` | the current directory |
| `--host HOST`, `--port PORT` | `127.0.0.1`, `8787` |
| `--allow-origin ORIGIN` | none: a browser page may not call the runner until you allow its origin |
| `--max-run-seconds N` | no limit (`POSTERN_MAX_RUN_SECONDS`) |
| `--max-concurrent-runs N` | 1 (`POSTERN_MAX_CONCURRENT_RUNS`) |
| `--no-pull` | pulling on; an empty folder is then an error rather than a download (`POSTERN_PULL=0`) |
| `--max-bundle-bytes N` | 64 MiB (`POSTERN_MAX_BUNDLE_BYTES`) |
| `--mcp` | serve an MCP server's tools instead of a bundle (below) |
| `--quiet` | log warnings and errors only |

Provider keys are none of the runner's business. The run reads them from the
bundle's `.env`, the runner reads only the names above out of that file, and
there is nowhere in the protocol for a key to travel. The runner's own settings
are removed from the environment of everything it runs.

**Exit codes**, because a container's is the only thing an operator sees when it
does not stay up: `0` interrupted after serving; `2` not configured; `3` refused,
by the distributor or by the checks on the bundle it served; `4` no usable
answer, or the bundle could not be written, which is worth a retry.

## Serving an MCP server's tools

```console
$ sigrix-runtime --mcp -- python my_server.py
```

`--mcp` starts an MCP server over stdio and serves its tools as one agent:
`describe` comes from the server's `tools/list`, and a `run` is a `tools/call`,
with the same bounds and cancellation as a crew's run. Nothing is pulled.

## Running a CrewAI bundle

A crew bundle runs through the execution layer, which needs CrewAI. A bundle's
own `requirements.txt` pins the exact versions it ships with; outside one, the
`crewai` extra installs a supported range:

```console
$ pip install "sigrix-runtime[crewai]"
```

## Layout

- `sigrix_runtime.postern`: the server, the entitlement check, the verified
  pull, the version check and the MCP mode. Standard library only.
- `sigrix_runtime.execution`, `loader`, `workforce`, `sandbox`, `configuration`,
  `quiet`, `runner_env`: the one run path a bundle's `main.py` and the server
  share, and what it needs to build and run a crew.

## Where it fits

sigrix-runtime is one of the open-source projects [Sigrix](https://sigrix.io)
publishes. The others it meets:

- **[Postern](https://github.com/sigrix-io/postern)**, the specification this
  runner serves.
- **[postern-conformance](https://pypi.org/project/postern-conformance/)**
  (`pip install postern-conformance`), the checker CI holds this runner to.
  Point it at yours too.
- **[Gatehouse](https://github.com/sigrix-io/gatehouse)**
  (`npm install @sigrix-io/gatehouse`), the page a person runs an agent from
  in the browser. Give it this runner's address, and start the runner with
  `--allow-origin` set to the page's origin.
- **[sigrix-launcher](https://github.com/sigrix-io/sigrix-launcher)**, which
  starts an MCP server a buyer bought on Sigrix: it checks the purchase and
  downloads the seller's package with this package's client code.

Every project Sigrix publishes, and a map of how they connect:
[sigrix.io/open-source](https://sigrix.io/open-source).

## Versioning and licence

Semantic versioning, pre-1.0; `VERSIONING.md` says what may change and how a
release is made. Licensed under Apache-2.0; `NOTICE` covers the Sigrix name and
logo, which the licence does not grant.
