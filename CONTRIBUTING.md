# Contributing

Thanks for looking. This is the runner inside every Sigrix bundle, so a change
here reaches every buyer's machine at the next release. The bar is that it stays
a faithful implementation of the [Postern specification](https://github.com/sigrix-io/postern)
and nothing more.

## What fits

- A runner that answers differently from what the specification says. The
  specification is what anyone pinned to; where this code and the document
  disagree, the code is the one to change.
- A case the entitlement check, the verified pull or the MCP mode gets wrong.
- A clearer sentence when it refuses. Say what happened and what the buyer can
  do about it; never guess at a cause the distributor did not give.

## What does not

- Behaviour the specification does not define. Propose it at
  `sigrix-io/postern` first; a runner that grows its own dialect is one every
  client has to special-case.
- A third-party import in `sigrix_runtime.postern`. The server is standard
  library only, a test holds it to that, and the reason is practical: it has to
  start before anything is installed.
- Anything that gives a credential value a way to travel over the protocol.

## Working on it

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
```

The tests run a distributor and an MCP server on loopback, so they need no
network. Three need CrewAI and skip without it; `pip install -e ".[dev,crewai]"`
runs them too.

To check conformance the way CI does, start the runner in MCP mode against the
test server and point the checker at it:

```sh
sigrix-runtime --mcp --port 8791 -- python tests/fixtures/mcp_server.py &
postern-conformance http://127.0.0.1:8791 --execute
```

## Pull requests

- One change per pull request, with a test that fails without it.
- `CHANGELOG.md` gets a line under *Unreleased*.
- A change to what a verb answers, or to an option or variable a buyer sets, is
  a breaking change for somebody's client or container; say so in the
  changelog.
