"""``python -m sigrix_runtime.postern`` — serve this bundle over Postern.

Run it from the bundle folder::

    python -m sigrix_runtime.postern

and point a client at ``http://127.0.0.1:8787/postern/v0/describe``.

The same module is the ``sigrix/runner`` image's entrypoint. That is the
one thing worth keeping true about this file: the container is a
deployment shape of this runner, not a second one, so anything it needs is
a flag or an environment variable here rather than a wrapper of its own.
The image's boot sequence is therefore this file's :func:`main` —

    resolve the token → check the entitlement → pull the bundle →
    verify its digest → serve

— and a buyer who already has their bundle unzipped meets the same
sequence with the middle two steps skipped, because there is nothing to
fetch. Neither shape can drift from the other; there is one of them.

Configuration, in the order a buyer meets it:

``SIGRIX_TOKEN``          a runner token for this listing (the plugin feed token works too)
``POSTERN_AGENT_ID``      which listing this bundle is, as ``{seller}/{listing-id}``
``POSTERN_DISTRIBUTOR``   the distributor base URL (defaults to ``https://sigrix.io``)

Each is read from the process environment first and from the bundle's
``.env`` second, so both shapes work: a container exports them, while a
buyer on their own machine fills in the ``.env`` the bundle already told
them to create — ``.env.example`` ships with ``POSTERN_AGENT_ID`` filled in
and says where to copy the token from. An exported value always wins, so a
stale ``.env`` cannot override an operator. The listing may also be named
as the one positional argument, which is what ``docker run … sigrix/runner
acme/my-crew`` does, and an argument beats both.

Provider keys are never any of this runner's business: they are read from
the bundle's ``.env`` by the run itself, and there is nowhere in the
protocol for one to travel (SPEC 4.1.3). This process reads only the three
names above out of that file and leaves the rest of it alone.

``--mcp`` serves an MCP server's tools instead of a bundle: the
listing's, through ``uvx sigrix-launcher run`` (the one command handed
``SIGRIX_TOKEN``), or any command given after ``--``. Nothing is pulled.

**Exit codes**, because a container's is the only thing an operator sees
when it does not stay up:

``0``  interrupted after serving
``2``  not configured: a variable, a folder, an identifier that is not one
``3``  refused, by the distributor or by the checks on the bundle it served
``4``  no usable answer, or the bundle could not be written; worth a retry
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from sigrix_runtime.postern import DEFAULT_HOST, DEFAULT_PORT
from sigrix_runtime.postern import describe as describe_module
from sigrix_runtime.postern import entitlement as entitlement_module
from sigrix_runtime.postern import mcp as mcp_module
from sigrix_runtime.postern import pull as pull_module
from sigrix_runtime.postern import version_check as version_check_module
from sigrix_runtime.postern.engine import Limits
from sigrix_runtime.postern.errors import WITHDRAWN_MESSAGE, PosternError
from sigrix_runtime.postern.server import RunnerConfig, serve

logger = logging.getLogger("sigrix_runtime.postern")

EXIT_OK = 0
EXIT_NOT_CONFIGURED = 2
EXIT_REFUSED = 3
EXIT_UNAVAILABLE = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sigrix_runtime.postern",
        description="Serve this Sigrix bundle over the Postern protocol (describe/run/stream/status).",
    )
    parser.add_argument(
        "agent",
        nargs="?",
        default="",
        metavar="AGENT",
        help=(
            "the listing to serve, as '{seller-handle}/{listing-id}'. "
            "Given here it also names the bundle to fetch when the folder is empty, "
            "which is the container's shape: docker run … sigrix/runner acme/my-crew"
        ),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="the bundle folder to serve (default: the current directory)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"interface to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to bind (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help=(
            "allow a browser origin to call this runner; repeatable. "
            "'*' allows any, which grants every page on the web a surface with no authentication — "
            "an operator may choose it, and this runner never chooses it for you."
        ),
    )
    parser.add_argument(
        "--max-run-seconds",
        type=int,
        default=_int_env("POSTERN_MAX_RUN_SECONDS", 0),
        help="stop a run that passes this many seconds (default: no limit)",
    )
    parser.add_argument(
        "--max-concurrent-runs",
        type=int,
        default=_int_env("POSTERN_MAX_CONCURRENT_RUNS", 1),
        help="how many runs may overlap (default: 1)",
    )
    parser.add_argument(
        "--agent-id",
        default="",
        help="this listing's Postern identifier, if it is not in POSTERN_AGENT_ID",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        default=_flag_env("POSTERN_PULL", True) is False,
        help=(
            "never fetch a bundle; serve only what is already in the folder. "
            "An empty folder is then an error rather than a download."
        ),
    )
    parser.add_argument(
        "--max-bundle-bytes",
        type=int,
        default=_int_env("POSTERN_MAX_BUNDLE_BYTES", pull_module.MAX_BUNDLE_BYTES),
        metavar="BYTES",
        help=f"the largest bundle this runner will accept (default: {pull_module.MAX_BUNDLE_BYTES:,})",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help=(
            "serve an MCP server's tools instead of a bundle: AGENT's, started through sigrix-launcher, "
            "or the server command given after --"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="log warnings and errors only")
    return parser


def _ensure_writable_home() -> None:
    """A container run under an arbitrary ``--user`` — the image's documented
    answer for a bind mount that keeps its host ownership — has no
    ``/etc/passwd`` entry for that uid, so the container runtime resolves
    ``HOME`` to ``/``: owned by root, unwritable to anyone else. CrewAI (and
    anything else with a dot-directory habit) writes there on first use with
    no ``try/except`` of its own, so the first such write died with
    ``PermissionError: [Errno 13] Permission denied: '/.local'``.

    Run once, here, before anything gets a chance to write. That is enough
    for the run subprocess too: :meth:`Engine._spawn` hands the agent this
    process's environment less its own ``POSTERN_*``/``SIGRIX_*`` settings
    (``execution.RUNNER_ENV_PREFIXES``), and ``HOME`` is in neither
    namespace — so a value fixed here is fixed for every bundle it spawns.
    """
    home = os.environ.get("HOME", "")
    if home and os.path.isdir(home) and os.access(home, os.W_OK):
        return
    fallback = tempfile.gettempdir()
    logger.warning("Postern: HOME (%s) is not writable; using %s for this run.", home or "<unset>", fallback)
    os.environ["HOME"] = fallback


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in argv:  # an MCP server's own command, which is not this parser's to read
        argv, command = argv[: argv.index("--")], argv[argv.index("--") + 1 :]
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        # Logs go to stderr because stdout carries the POSTERN_PORT line a
        # parent process reads (SPEC 2), and one stream cannot be both.
        stream=sys.stderr,
    )
    _ensure_writable_home()

    bundle_root = (args.bundle or Path.cwd()).resolve()
    settings = entitlement_module.settings_from_environment(bundle_root)
    agent_id = (args.agent or args.agent_id or settings.agent_id).strip()
    if agent_id and not describe_module.is_agent_id(agent_id):
        # Refused once, here, for all four sources. Served, it reached
        # ``describe`` and ``status`` verbatim and every check as a network
        # outage, or, with no slash, as a refund.
        source = "The AGENT argument" if args.agent else "--agent-id" if args.agent_id else "POSTERN_AGENT_ID"
        print(
            f"{source} is {agent_id!r}, which is not a Postern agent identifier: an identifier is "
            f"{describe_module.AGENT_ID_SHAPE}. Copy it from the listing's page, or from the top of "
            "the bundle's .env.example.",
            file=sys.stderr,
        )
        return EXIT_NOT_CONFIGURED
    if args.mcp or command:
        return _serve_mcp(args, bundle_root=bundle_root, settings=settings, agent_id=agent_id, command=command)

    try:
        outcome = _obtain_bundle(bundle_root, settings=settings, agent_id=agent_id, args=args)
    except pull_module.PullNotConfigured as exc:
        print(exc, file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    except pull_module.PullRefused as exc:
        print(exc, file=sys.stderr)
        return EXIT_REFUSED
    except pull_module.PullUnavailable as exc:
        print(exc, file=sys.stderr)
        return EXIT_UNAVAILABLE

    if outcome.pulled:
        logger.info(
            "Postern: fetched %s into %s (%s)",
            agent_id,
            bundle_root,
            "digest verified" if outcome.verified else "no digest offered — unverified",
        )

    update_check = _check_for_update(bundle_root, settings=settings, agent_id=agent_id)

    serve(_config(args, bundle_root=bundle_root, agent_id=agent_id, update_check=update_check))
    return EXIT_OK


def _config(
    args: argparse.Namespace,
    *,
    bundle_root: Path,
    agent_id: str,
    update_check: version_check_module.UpdateCheck | None = None,
    mcp: mcp_module.McpServer | None = None,
) -> RunnerConfig:
    return RunnerConfig(
        bundle_root=bundle_root,
        host=args.host,
        port=args.port,
        allowed_origins=tuple(origin for origin in args.allow_origin if origin and origin != "*"),
        allow_any_origin="*" in args.allow_origin,
        limits=Limits(
            max_run_seconds=max(0, args.max_run_seconds),
            max_concurrent_runs=max(1, args.max_concurrent_runs),
        ),
        agent_id=agent_id,
        update_check=update_check,
        mcp=mcp,
    )


def _serve_mcp(
    args: argparse.Namespace,
    *,
    bundle_root: Path,
    settings: entitlement_module.Settings,
    agent_id: str,
    command: list[str],
) -> int:
    """MCP mode's boot: resolve the command, check the licence, read the tools, serve."""
    if not args.mcp:
        print("A command after -- names an MCP server to serve; add --mcp.", file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    launcher = not command
    if launcher and not agent_id:
        print(
            "--mcp serves a listing named as AGENT or POSTERN_AGENT_ID, or the MCP server command given after --.",
            file=sys.stderr,
        )
        return EXIT_NOT_CONFIGURED
    if launcher and not settings.token:
        print(
            f"Serving {agent_id} through sigrix-launcher needs SIGRIX_TOKEN, the token it checks the purchase "
            "with: set it in the environment or in this folder's .env.",
            file=sys.stderr,
        )
        return EXIT_NOT_CONFIGURED
    server = mcp_module.McpServer(command=tuple(command) or mcp_module.launcher_command(agent_id), launcher=launcher)
    if shutil.which(server.command[0]) is None:
        hint = " The launcher runs through uv: https://docs.astral.sh/uv/" if launcher else ""
        print(f"The MCP server command {server.command[0]!r} was not found on PATH.{hint}", file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    mode = os.environ.get("SIGRIX_DELIVERY_MODE") or entitlement_module.DELIVERY_MODE_CONNECTOR_LOCAL
    licensed, _ = _boot_check(bundle_root, settings=settings, agent_id=agent_id, delivery_mode=mode)
    if licensed is False and launcher:
        # The launcher would refuse the same licence, and without it there are
        # no tools to describe: the one boot a refusal ends (the log says why).
        return EXIT_REFUSED
    try:
        serve(_config(args, bundle_root=bundle_root, agent_id=agent_id, mcp=server))
    except PosternError as exc:
        print(f"Could not read the MCP server's tools: {exc.message}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    return EXIT_OK


def _obtain_bundle(
    bundle_root: Path,
    *,
    settings: entitlement_module.Settings,
    agent_id: str,
    args: argparse.Namespace,
) -> pull_module.PullOutcome:
    """SPEC 5.3's check, then SPEC 5.6's pull.

    The check runs first because it is what makes the pull's ``404``
    legible: the two endpoints are allowed to disagree — a distributor may
    license a listing it does not package for download — and a runner that
    was just told *active* must not then tell its buyer their purchase may
    have been refunded. It is best-effort in every other respect: an
    unreachable distributor is a condition SPEC 5.7 declares grace for, and
    a boot that died on it would take ``status`` down exactly when somebody
    needs to read it.
    """
    licensed, gate = _boot_check(bundle_root, settings=settings, agent_id=agent_id)
    outcome = pull_module.ensure_bundle(
        bundle_root,
        base_url=settings.base_url,
        token=settings.token,
        agent_id=agent_id,
        allow_pull=not args.no_pull,
        licensed=licensed,
        max_bytes=max(1, args.max_bundle_bytes),
    )
    if outcome.pulled and gate is not None and gate.cache_path is None:
        # Kept now that there is a folder to keep it in. Pulled into a new
        # one, the boot's answer was dropped, and a restart before the first
        # run found nothing to count grace from.
        gate.keep_in(bundle_root / entitlement_module.CACHE_FILENAME)
    return outcome


def _boot_check(
    bundle_root: Path,
    *,
    settings: entitlement_module.Settings,
    agent_id: str,
    delivery_mode: str = entitlement_module.DELIVERY_MODE_ZIP,
) -> tuple[bool | None, entitlement_module.Entitlement | None]:
    """What the distributor says about this licence right now, if anything.

    ``None`` means nothing was learned — half a configuration, or an
    endpoint that did not answer — and no caller may read that as *no*.
    The client that asked comes back beside it, holding the answer.
    """
    if not (settings.base_url and settings.token and agent_id):
        # SPEC 5.1: half a configuration is no configuration, and a runner
        # with no distributor is not an unlicensed one.
        return None, None
    gate = entitlement_module.Entitlement(
        base_url=settings.base_url,
        token=settings.token,
        agent_id=agent_id,
        cache_path=bundle_root / entitlement_module.CACHE_FILENAME if bundle_root.is_dir() else None,
        delivery_mode=delivery_mode,
    )
    try:
        verdict = gate.refresh()
    except Exception as exc:  # noqa: BLE001 - the boot must survive its own diagnostics
        # `refresh` already swallows an unreachable distributor. Anything
        # left is this runner failing at a step whose entire purpose is to
        # word a later message, and it must not be the reason a bundle
        # already on disk goes unserved.
        logger.warning("Postern: the licence check did not complete (%s); carrying on.", exc)
        return None, gate
    if gate.misaddressed:
        logger.warning("Postern: %s", gate.misaddressed)
        return None, gate
    if verdict.state == entitlement_module.STATE_REVOKED:
        # Said once, at boot, and not fatal: `run` refuses on its own and
        # `status` is how a client finds out. A container that exited here
        # would leave a buyer with a restart loop instead of an answer.
        ended_on = gate.ended_on()
        if ended_on:
            logger.warning("Postern: %s Your access to %s ended on %s.", WITHDRAWN_MESSAGE, agent_id, ended_on)
        else:
            logger.warning("Postern: %s is not licensed to run here.", agent_id)
        return False, gate
    if verdict.state == entitlement_module.STATE_UNKNOWN:
        logger.warning("Postern: could not confirm the licence for %s; carrying on.", agent_id)
        return None, gate
    ends_on = gate.ends_on()
    if ends_on:
        # SPEC 5.3's date, ahead of time: the buyer can plan by it.
        logger.warning("Postern: %s Your access to %s ends on %s.", WITHDRAWN_MESSAGE, agent_id, ends_on)
    return verdict.state == entitlement_module.STATE_ACTIVE, gate


def _check_for_update(
    bundle_root: Path,
    *,
    settings: entitlement_module.Settings,
    agent_id: str,
) -> version_check_module.UpdateCheck:
    """SPEC 8's check-on-start. Best-effort, once, never fatal.

    Runs after the bundle is in place, since there is nothing to compare
    before that — and it must never be why a boot fails: this runner has to
    work offline after the initial pull, so a network
    failure here degrades to ``unreachable`` exactly the way an unreachable
    entitlement check degrades to ``unknown`` in :func:`_boot_check`.
    """
    if not (settings.base_url and agent_id):
        return version_check_module.UpdateCheck(state=version_check_module.STATE_NOT_REQUIRED)
    try:
        document = describe_module.load_describe(bundle_root, agent_id=agent_id)
    except Exception as exc:  # noqa: BLE001 - a version check must survive its own diagnostics
        logger.warning("Postern: could not read this bundle's own version (%s); skipping the update check.", exc)
        return version_check_module.UpdateCheck(state=version_check_module.STATE_UNREACHABLE)
    current_version = str((document.get("agent") or {}).get("version") or "")
    check = version_check_module.check_for_update(
        base_url=settings.base_url,
        agent_id=agent_id,
        current_version=current_version,
    )
    if check.state == version_check_module.STATE_UPDATE_AVAILABLE:
        logger.warning(
            "Postern: an update is available for %s (%s -> %s). A bundle is never updated in place: "
            "pull it into an empty folder, then copy workspace/, variables.json and .env across.",
            agent_id,
            check.current,
            check.latest,
        )
    return check


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name) or default))
    except ValueError:
        return default


def _flag_env(name: str, default: bool) -> bool:
    """A yes/no environment variable, for the container's ``-e`` shape."""
    raw = str(os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


if __name__ == "__main__":
    sys.exit(main())
