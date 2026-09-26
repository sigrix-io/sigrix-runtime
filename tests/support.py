"""What the suite runs against: the package under test, laid out the way a bundle
carries it, and the crews it serves.

A Sigrix bundle carries this package as a ``sigrix_runtime/`` folder beside its
configuration, and a runner started in the bundle imports that copy. The tests
build bundles the same way, so :data:`RUNTIME_ROOT` is a folder holding one
``sigrix_runtime/``, copied from whichever package this suite
imported: the checkout under ``src/`` in development, the installed wheel when
the release workflow tests the artifact it is about to publish. Copying the
directory the import resolved to, rather than ``src/``, is what keeps the second
case honest.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

import sigrix_runtime

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: A bundle's static root: ``sigrix_runtime/``, and the ``.env.example`` every
#: bundle carries beside it, which is where ``describe`` reads the names of the
#: credentials a crew needs. Copied into a bundle, or put on a subprocess's
#: ``PYTHONPATH``, it is the code under test.
RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="sigrix-runtime-under-test-"))
shutil.copytree(
    Path(sigrix_runtime.__file__).resolve().parent,
    RUNTIME_ROOT / "sigrix_runtime",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)
shutil.copyfile(FIXTURES / "env.example", RUNTIME_ROOT / ".env.example")
atexit.register(shutil.rmtree, RUNTIME_ROOT, ignore_errors=True)

#: The crews a bundle can hold. ``CREW_CONFIG`` is the one most tests serve.
CREWS = FIXTURES / "crews"
CREW_CONFIG = CREWS / "contract-review" / "config"

#: What a bundle's ``VERSION`` file says, which ``describe`` reports as the
#: agent's version. A bundle's version, never this package's.
BUNDLE_VERSION = "fixture-crew-1.0.0"
