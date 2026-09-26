# Versioning

Semantic versioning, `MAJOR.MINOR.PATCH`, released from a `vX.Y.Z` tag. The
version lives in one place, `src/sigrix_runtime/__init__.py`, and the build
reads it from there.

- **PATCH**: a fix that changes no verb's answer, no option and no variable.
- **MINOR**: a new option or variable; a new field in an answer, which
  Postern's own rules say a client must ignore when it does not know it.
- **MAJOR**: a removed or renamed option or variable, or an answer a client
  already relies on that changes shape.

**Before 1.0** a MINOR bump may carry what would later be MAJOR, and the
changelog says so when it does. Pin the minor (`sigrix-runtime>=0.1,<0.2`)
if you build on it.

**Two versions are not this one.** The Postern version a runner speaks is the
specification's (`POSTERN_VERSION`, `"0.1"`), and moves only when the
specification does. A bundle states its own version in its `VERSION` file,
and the version check of the specification's §8 compares that, never this.

## Releasing

A release is a tag. `release.yml` builds the distribution and publishes it on
any `v*` tag pushed to this repository; nothing is uploaded by hand and no API
token exists to leak.

That works because PyPI is configured to trust this repository rather than a
credential, which takes two things that must both be in place before the first
tag, and neither fails loudly if it is missing:

1. On PyPI, a **trusted publisher** for `sigrix-io/sigrix-runtime`, workflow
   `release.yml`, environment `pypi`.
2. In this repository's settings, an **environment named `pypi`**. The
   publisher's claim names it, so a workflow running outside it is refused.

Then move the version in `src/sigrix_runtime/__init__.py`, file the
*Unreleased* lines under its heading in `CHANGELOG.md`, and:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

The build refuses a tag that does not name the version it built, and a
changelog that still files anything under *Unreleased*. Allow about ten minutes
after the upload before expecting `pip install sigrix-runtime` to resolve it:
that is the index's cache, not a failed publish.
