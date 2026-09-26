"""Bundle retrieval, the runner's half (SPEC 5.6).

The distributor serves the bundle; this
is what a runner that has *nothing but a token* does with it — the middle
of the image's boot sequence, between the check and the serve. It exists so
``docker run -e SIGRIX_TOKEN=… sigrix/runner acme/my-crew`` on a clean
machine can end at a running agent, with no unzipping step a person has to
perform first.

Four rules shape it, and each is a way an implementation that "works" is
still wrong:

**Nothing reaches disk before the digest agrees.** The response is held in
memory, hashed, compared against ``Repr-Digest``, and only then extracted —
so a truncated download or a tampered mirror leaves a buyer with no bundle
rather than most of one. That is also why there must be a **size bound**:
holding the representation is what makes verification possible, and an
unbounded hold is a distributor's stream deciding how much of a buyer's
memory to use. SPEC 5.6 sets no limit deliberately (it is a protocol, not a
deployment), so the bound belongs here — see :data:`MAX_BUNDLE_BYTES`.

**A missing digest is a warning; a wrong one is fatal.** RFC 9530's header
is a SHOULD in SPEC 5.6, so refusing a distributor that omits it would make
this runner less interoperable than the protocol asks for. A header that is
present and disagrees is the case the header exists for, and there is
nothing to weigh.

**The pull never writes over a bundle that is already there.** A buyer's
bundle folder holds their ``.env`` and their ``workspace/``; a runner that
re-pulled over it would destroy both to fix a problem nobody had. So the
pull is what happens when there is *no* bundle, and a folder that already
holds one is served as it stands (an update is the version check's, and
it does not self-update either).

**A folder remembers which agent it holds.** :data:`STAMP_FILENAME` records
the identifier and the digest, so a second boot against the same volume
with a different ``POSTERN_AGENT_ID`` is refused by name rather than
silently serving the previous buyer's agent — which looks exactly like a
working runner right up until someone reads the output.

Failures split in two, and the split is the same one SPEC 5.7 draws for the
check: :class:`PullRefused` is something the distributor said, so retrying
changes nothing; :class:`PullUnavailable` is something the network did, so
retrying is the whole remedy. A boot sequence that conflates them either
loops forever on a refund or gives up on a flaky minute.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sigrix_runtime.postern import PATH_PREFIX
from sigrix_runtime.postern.errors import NOT_ENTITLED_MESSAGE, WITHDRAWN_MESSAGE
from sigrix_runtime.postern.transport import TransportError, open_response

logger = logging.getLogger(__name__)

BUNDLE_MEDIA_TYPE = "application/zip"

#: RFC 9530's field, as SPEC 5.6 spells it. ``Repr-Digest`` rather than
#: ``Content-Digest`` because what is verified is the bundle kept, not the
#: bytes of one hop — a content-coding changes the second and leaves the
#: first alone.
REPR_DIGEST_HEADER = "Repr-Digest"

#: The largest response this runner will hold: the pull side's own answer
#: to how large a bundle a runner should accept.
#:
#: **Not the distributor's cap.** Sigrix refuses to *build* a bundle over
#: 5 MB, but that is a fact about one
#: distributor's publish path, and this runner ships inside every bundle and
#: can be pointed at any distributor. So the number here is a buyer-side
#: memory bound with room above every bundle that exists — twelve times the
#: platform's own cap — chosen so that raising that cap is not silently also
#: a runner change. ``POSTERN_MAX_BUNDLE_BYTES`` moves it for a buyer who
#: knows their distributor serves something larger.
MAX_BUNDLE_BYTES = 64 * 1024 * 1024

#: How far a bundle may expand once unzipped, as a multiple of the bytes
#: received. A zip is compressed, so a bound on the download bounds nothing
#: on disk: 64 MB of well-chosen zeroes unpack to gigabytes.
#:
#: Twenty rather than something tighter because the ratio is a *bomb*
#: detector, not a size limit — a bundle is code and YAML, which deflate
#: happily compresses tenfold, and refusing a real listing for being
#: compressible would be a bug reported as "my agent will not start".
EXTRACT_RATIO_LIMIT = 20

#: …and a floor under it, because the ratio alone punishes small bundles: a
#: 4 KB zip of text would otherwise be held to 80 KB unpacked. Sixteen
#: megabytes of disk is nothing to a machine that just downloaded a bundle
#: and is about to install CrewAI, and it is far below anything an accident
#: or a malformed archive would produce.
MIN_EXTRACT_BYTES = 16 * 1024 * 1024

#: …and a bound on how many files, which bytes cannot give: a million empty
#: entries weigh nothing and cost a million files. A bundle is a few
#: hundred at most.
MAX_BUNDLE_ENTRIES = 10_000

#: Longer than the check's ten seconds, because this is megabytes rather
#: than a sentence, and it happens once at boot rather than before every
#: run. A check that hangs delays an agent that could be running; a pull
#: that hangs is the only thing happening.
PULL_TIMEOUT_SECONDS = 60

#: What makes a folder a bundle, and the same thing ``__main__`` looks for
#: before it will serve one.
BUNDLE_MARKER = "config"

#: Written beside the entitlement cache, and for the same reason: it is the
#: runner's own state about this folder, not the agent's.
STAMP_FILENAME = ".postern_bundle.json"

_STAGING_PREFIX = ".postern-incoming-"
_READ_CHUNK_BYTES = 64 * 1024
#: An error body is one sentence in an envelope (SPEC 2.1). Bounded anyway,
#: because the reason to bound the success path applies to a distributor
#: answering 500 with a megabyte of HTML.
_MAX_ERROR_BODY_BYTES = 64 * 1024


class PullError(RuntimeError):
    """Base for both halves. ``str()`` is what a buyer is shown."""


class PullRefused(PullError):
    """The distributor answered, and the answer was no. Retrying will not help."""


class PullNotConfigured(PullRefused):
    """There is no bundle and not enough to fetch one — the operator's to fix.

    Separated from :class:`PullRefused` because the two want different
    exits and different advice: a refusal is about this buyer's licence,
    and this is about a variable nobody set.
    """


class PullUnavailable(PullError):
    """No usable answer was obtained. Retrying is the remedy."""


@dataclass(frozen=True)
class PulledBundle:
    """The bytes a distributor served, and what is known about them."""

    content: bytes
    digest: str
    verified: bool

    @property
    def size(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class PullOutcome:
    """What :func:`ensure_bundle` did, in a shape a caller can log."""

    action: str
    bundle_root: Path
    digest: str = ""
    verified: bool = False

    @property
    def pulled(self) -> bool:
        return self.action == "pulled"


def is_bundle(bundle_root: Path) -> bool:
    """Whether this folder already holds a bundle to serve."""
    return (bundle_root / BUNDLE_MARKER).is_dir()


def stamped_agent_id(bundle_root: Path) -> str:
    """The identifier this folder was pulled for, or ``""`` if unstamped.

    Unstamped is the ordinary case for a buyer who unzipped a bundle
    themselves, so it means *no claim* rather than *no*, and no caller may
    read it as a mismatch.
    """
    try:
        raw = json.loads((bundle_root / STAMP_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(raw.get("agent_id") or "") if isinstance(raw, dict) else ""


def ensure_bundle(
    bundle_root: Path,
    *,
    base_url: str,
    token: str,
    agent_id: str,
    allow_pull: bool = True,
    licensed: bool | None = None,
    max_bytes: int | None = None,
    timeout: float = PULL_TIMEOUT_SECONDS,
) -> PullOutcome:
    """The boot step: a folder with a bundle in it, or an error saying why not.

    Serves what is already there when there is anything there — see the
    module docstring for why that is the rule rather than a shortcut.
    """
    if is_bundle(bundle_root):
        stamped = stamped_agent_id(bundle_root)
        if stamped and agent_id and stamped != agent_id:
            raise PullNotConfigured(
                f"{bundle_root} already holds {stamped}, and this runner was asked for {agent_id}. "
                "Point it at an empty folder (a fresh volume, for a container) so the two do not "
                "share one, or start it with the identifier this folder was pulled for."
            )
        return PullOutcome(action="already_present", bundle_root=bundle_root)

    if not allow_pull:
        raise PullNotConfigured(f"{_no_bundle_here(bundle_root)} This runner was started with --no-pull.")
    missing = [
        name
        for name, value in (("POSTERN_AGENT_ID", agent_id), ("SIGRIX_TOKEN", token), ("POSTERN_DISTRIBUTOR", base_url))
        if not str(value or "").strip()
    ]
    if missing:
        raise PullNotConfigured(
            f"{_no_bundle_here(bundle_root)} There is not enough here to fetch one either: "
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set. "
            "Create a runner token for this agent under Runner tokens on your account's Plugins "
            "page. The listing's identifier is on its own page (for an entitled buyer) and in your "
            "Sigrix library, and is also filled in for you at the top of the bundle's .env.example."
        )

    bundle = fetch_bundle(
        base_url=base_url,
        token=token,
        agent_id=agent_id,
        licensed=licensed,
        max_bytes=max_bytes if max_bytes is not None else MAX_BUNDLE_BYTES,
        timeout=timeout,
    )
    install_bundle(bundle, bundle_root, agent_id=agent_id, source=base_url)
    return PullOutcome(
        action="pulled",
        bundle_root=bundle_root,
        digest=bundle.digest,
        verified=bundle.verified,
    )


def extract_budget(downloaded: int) -> int:
    """How many bytes a download of ``downloaded`` bytes may unpack to."""
    return max(downloaded * EXTRACT_RATIO_LIMIT, MIN_EXTRACT_BYTES)


def _no_bundle_here(bundle_root: Path) -> str:
    """The sentence both not-configured paths open with.

    Kept as the wording ``__main__`` printed before there was a pull at all,
    because the commonest reader of it is still someone who ran the module
    in the wrong folder rather than a container missing a variable.
    """
    return (
        f"{bundle_root} does not look like a Sigrix bundle — it has no {BUNDLE_MARKER}/ folder. "
        "Run this from the folder you unzipped, or pass --bundle path/to/bundle."
    )


def fetch_bundle(
    *,
    base_url: str,
    token: str,
    agent_id: str,
    licensed: bool | None = None,
    max_bytes: int = MAX_BUNDLE_BYTES,
    timeout: float = PULL_TIMEOUT_SECONDS,
) -> PulledBundle:
    """One ``GET`` at SPEC 5.6's endpoint, verified before it is returned.

    ``licensed`` is what the check said a moment earlier, and it is only
    ever used to word a ``404``. The two endpoints can disagree — a
    distributor may license a listing it does not package for download —
    and a runner that has just been told *active* should not then tell its
    buyer their purchase may have been refunded.
    """
    owner, _, name = agent_id.partition("/")
    if not owner or not name:
        raise PullRefused(
            f"{agent_id!r} is not a Postern agent identifier. It is two parts, "
            "'{seller-handle}/{listing-id}', and the bundle's .env.example carries yours."
        )

    path = f"{PATH_PREFIX}/bundles/{owner}/{name}"
    try:
        with open_response(base_url, path, token=token, accept=BUNDLE_MEDIA_TYPE, timeout=timeout) as response:
            status = int(response.status)
            if status == 200:
                _refuse_declared_length(response.getheader("Content-Length"), max_bytes=max_bytes)
                content = _read_bounded(response, max_bytes=max_bytes)
                digest_header = response.getheader(REPR_DIGEST_HEADER) or ""
                error_body = b""
            else:
                content, digest_header = b"", ""
                error_body = response.read(_MAX_ERROR_BODY_BYTES)
    except TransportError as exc:
        raise PullUnavailable(
            f"Could not reach {base_url} to fetch this agent's bundle: {exc}. "
            "Check the machine's network connection and try again."
        ) from exc

    if status != 200:
        raise _refusal_for(status, error_body, agent_id=agent_id, base_url=base_url, licensed=licensed)
    return _verified(content, digest_header, agent_id=agent_id)


def install_bundle(bundle: PulledBundle, bundle_root: Path, *, agent_id: str, source: str) -> None:
    """Unpack into ``bundle_root``, or leave it exactly as it was.

    Extraction runs in a staging folder *inside* the destination — one
    filesystem, so the move into place is a rename, and a failure removes
    the staging folder rather than leaving a half-bundle that looks
    servable.
    """
    if is_bundle(bundle_root):  # pragma: no cover - ensure_bundle checks first
        raise PullRefused(f"{bundle_root} already holds a bundle; this runner does not write over one.")

    try:
        bundle_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=bundle_root))
    except OSError as exc:
        raise PullUnavailable(f"Could not write to {bundle_root}: {exc}") from exc

    try:
        _extract(bundle.content, staging, budget=extract_budget(len(bundle.content)))
        staged = sorted(staging.iterdir())
        # Every collision is found before the first move. Refusing halfway
        # through would leave the folder holding part of a bundle, which is
        # the one outcome this whole function is arranged to avoid.
        clashing = [entry.name for entry in staged if (bundle_root / entry.name).exists()]
        if clashing:
            raise PullRefused(
                f"{bundle_root} already contains {', '.join(clashing)}, and the bundle carries its own. "
                "Point this runner at an empty folder."
            )
        moved: list[Path] = []
        try:
            for entry in staged:
                entry.rename(bundle_root / entry.name)
                moved.append(bundle_root / entry.name)
        except OSError:
            # Every name was free a moment ago, so all of these are the
            # bundle's: taken back, or a rename failing midway left part of
            # one looking servable.
            for target in moved:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise PullUnavailable(f"Could not unpack the bundle into {bundle_root}: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    _write_stamp(bundle_root, bundle=bundle, agent_id=agent_id, source=source)


# --- Internals -------------------------------------------------------------


def _verified(content: bytes, digest_header: str, *, agent_id: str) -> PulledBundle:
    """SPEC 5.6's digest, checked before anything is written."""
    stated = sha256_from_repr_digest(digest_header)
    computed = base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")
    if not stated:
        logger.warning(
            "postern.pull.no_digest: %s served %s without a %s header; the bundle cannot be verified",
            agent_id,
            f"{len(content)} bytes",
            REPR_DIGEST_HEADER,
        )
        return PulledBundle(content=content, digest=computed, verified=False)
    if stated != computed:
        raise PullRefused(
            "The bundle that arrived is not the one the distributor described — its checksum does not "
            "match. Nothing was written. Try again; if it keeps happening, the download is being "
            "altered in transit and you should not run it."
        )
    return PulledBundle(content=content, digest=computed, verified=True)


def sha256_from_repr_digest(header: str) -> str:
    """The base64 SHA-256 out of RFC 9530's structured field, or ``""``.

    The field is a dictionary of algorithm to byte-sequence, so several may
    be offered and the colons are syntax rather than decoration::

        Repr-Digest: sha-512=:…:, sha-256=:IAtqOIW3wX…:

    Anything unparseable answers ``""`` — *no digest offered* — rather than
    raising: a malformed header is the distributor failing to make a promise,
    which is the case a warning covers, not the case a refusal covers.
    """
    for member in str(header or "").split(","):
        key, _, value = member.strip().partition("=")
        if key.strip().lower() != "sha-256":
            continue
        raw = value.strip()
        if len(raw) > 2 and raw.startswith(":") and raw.endswith(":"):
            return raw[1:-1]
    return ""


def _refuse_declared_length(header: str | None, *, max_bytes: int) -> None:
    """Refuse before reading when the distributor says how big it is."""
    try:
        declared = int(str(header or "").strip())
    except ValueError:
        return
    if declared > max_bytes:
        raise PullRefused(_too_large_message(declared, max_bytes))


def _read_bounded(response: Any, *, max_bytes: int) -> bytes:
    """Read the body, stopping the moment it passes the bound."""
    buffer = io.BytesIO()
    read = 0
    while True:
        chunk = response.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        read += len(chunk)
        if read > max_bytes:
            raise PullRefused(_too_large_message(read, max_bytes, at_least=True))
        buffer.write(chunk)
    return buffer.getvalue()


def _too_large_message(size: int, max_bytes: int, *, at_least: bool = False) -> str:
    seen = f"over {size:,}" if at_least else f"{size:,}"
    return (
        f"This bundle is {seen} bytes and this runner accepts {max_bytes:,}. Nothing was written. "
        "Set POSTERN_MAX_BUNDLE_BYTES higher if you know this distributor serves bundles this large."
    )


def _refusal_for(status: int, body: bytes, *, agent_id: str, base_url: str, licensed: bool | None = None) -> PullError:
    """The distributor's answer, as the sentence a buyer is shown."""
    code, detail = _envelope(body)

    if status == 404 and licensed:
        # The check said this buyer is licensed, so the missing bundle is
        # not about them. Saying "your purchase may have been refunded"
        # here would send a paying buyer to support over a listing their
        # distributor simply does not hand to runners.
        return PullRefused(
            f"You are licensed for {agent_id}, but {base_url} has no bundle to hand over for it — "
            "some listings are delivered as a download from your account rather than fetched by a "
            "runner. Nothing was downloaded."
        )
    if status == 404:
        # SPEC 5.5: the distributor cannot say which of the three it is, so
        # neither can this. `NOT_ENTITLED_MESSAGE` is the same sentence the
        # runner gives a client for the same three causes — one wording, so
        # the boot and the run never seem to disagree about the licence.
        return PullRefused(
            f"{NOT_ENTITLED_MESSAGE} Nothing was downloaded for {agent_id}. Check SIGRIX_TOKEN against "
            "the runner tokens on your account's Plugins page and "
            "POSTERN_AGENT_ID against the listing's own page."
        )
    if status == 410 or code == "withdrawn":
        ends_at = str(detail.get("access_ends_at") or "")[:10]
        when = f" Your access ended on {ends_at}." if ends_at else " Your access has ended."
        # Composed here rather than relayed: `message` is the distributor's
        # own copy, and a runner that prints it verbatim has made a remote
        # string part of its own interface.
        return PullRefused(f"{WITHDRAWN_MESSAGE}{when} Nothing was downloaded.")
    if status == 400:
        return PullRefused(
            f"{base_url} did not recognise {agent_id!r} as an agent identifier. It is two parts, "
            "'{seller-handle}/{listing-id}', and the bundle's .env.example carries yours."
        )
    if 400 <= status < 500:
        return PullRefused(f"{base_url} refused to hand over {agent_id} (HTTP {status}).")
    return PullUnavailable(
        f"{base_url} could not hand over {agent_id} just now (HTTP {status}). Try again in a few minutes."
    )


def _envelope(body: bytes) -> tuple[str, dict[str, Any]]:
    """SPEC 2.1's ``error`` object, as far as it can be read."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return "", {}
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return "", {}
    detail = error.get("detail")
    return str(error.get("code") or ""), detail if isinstance(detail, dict) else {}


def _extract(content: bytes, destination: Path, *, budget: int) -> None:
    """Unzip into ``destination``, refusing anything that reaches outside it."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise PullRefused(f"The bundle that arrived is not a readable zip file: {exc}") from exc

    with archive:
        if len(archive.infolist()) > MAX_BUNDLE_ENTRIES:
            raise PullRefused(
                f"The bundle carries {len(archive.infolist()):,} entries; this runner unpacks "
                f"{MAX_BUNDLE_ENTRIES:,} at most. Nothing was kept."
            )
        declared = sum(max(0, info.file_size) for info in archive.infolist())
        if declared > budget:
            raise PullRefused(_unpacks_to_message(declared, budget))
        written = 0
        for info in archive.infolist():
            target = _safe_target(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                # zipfile writes a symlink entry as a regular file holding
                # its target, so this is refused for what it says about the
                # archive rather than for what extracting it would do.
                raise PullRefused(f"The bundle carries a symbolic link ({info.filename!r}); this runner refuses it.")
            target.parent.mkdir(parents=True, exist_ok=True)
            # binary both ends: a zip member has no encoding to declare, and
            # a bundle carries fonts and images as readily as it carries YAML.
            with archive.open(info) as source, target.open("wb") as out:
                while True:
                    chunk = source.read(_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > budget:
                        # The declared sizes above are the archive's own
                        # claim about itself, so they are checked again
                        # against what actually arrives.
                        raise PullRefused(_unpacks_to_message(written, budget, at_least=True))
                    out.write(chunk)


def _unpacks_to_message(size: int, budget: int, *, at_least: bool = False) -> str:
    seen = f"over {size:,}" if at_least else f"{size:,}"
    return (
        f"This bundle unpacks to {seen} bytes, more than the {budget:,} this runner allows for its "
        "download size. Nothing was kept."
    )


def _safe_target(destination: Path, name: str) -> Path:
    """The path an entry may be written to, or a refusal.

    Refused rather than sanitised. ``zipfile`` would strip a leading slash
    and drop ``..`` for us, and that is the wrong remedy: an archive naming
    ``../../etc/cron.d/x`` is not a bundle with a typo, and quietly writing
    it somewhere else keeps a runner running on an artifact nobody should
    trust.
    """
    parts = [part for part in name.replace("\\", "/").split("/") if part not in ("", ".")]
    unsafe = not parts or name.startswith("/") or ".." in parts or ":" in parts[0]
    if unsafe:
        raise PullRefused(f"The bundle carries an entry that would write outside the folder ({name!r}).")
    target = destination.joinpath(*parts)
    try:
        target.resolve().relative_to(destination.resolve())
    except ValueError as exc:
        raise PullRefused(f"The bundle carries an entry that would write outside the folder ({name!r}).") from exc
    return target


def _write_stamp(bundle_root: Path, *, bundle: PulledBundle, agent_id: str, source: str) -> None:
    payload = {
        "agent_id": agent_id,
        "digest": f"sha-256=:{bundle.digest}:",
        "verified": bundle.verified,
        "bytes": bundle.size,
        "pulled_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
    }
    try:
        (bundle_root / STAMP_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 - a runner that cannot stamp still runs
        logger.info("postern.pull.stamp_write_failed: %s", exc)


__all__ = [
    "BUNDLE_MARKER",
    "BUNDLE_MEDIA_TYPE",
    "EXTRACT_RATIO_LIMIT",
    "MAX_BUNDLE_ENTRIES",
    "MIN_EXTRACT_BYTES",
    "MAX_BUNDLE_BYTES",
    "PULL_TIMEOUT_SECONDS",
    "REPR_DIGEST_HEADER",
    "STAMP_FILENAME",
    "PullError",
    "PullNotConfigured",
    "PullOutcome",
    "PullRefused",
    "PullUnavailable",
    "PulledBundle",
    "ensure_bundle",
    "extract_budget",
    "fetch_bundle",
    "install_bundle",
    "is_bundle",
    "sha256_from_repr_digest",
    "stamped_agent_id",
]
