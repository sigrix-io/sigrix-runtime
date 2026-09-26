# Security

## Reporting

Email security@sigrix.io with what you found and how to reproduce it. Do not
open a public issue for a vulnerability. You will get an acknowledgement
within three working days.

A flaw in the Postern specification itself, rather than in this runner, is
worth the same email: every runner that follows the document faithfully
inherits it.

## What the runner holds

One secret of its own: the buyer's runner token, in `SIGRIX_TOKEN`. It is sent
only in the `Authorization` header of the entitlement check and the bundle
download, to the configured distributor, and in plain text only when the
connection's peer is a loopback address, which is read off the socket rather
than trusted from a name. The answer it caches in the bundle folder
(`.postern_entitlement.json`) carries a fingerprint of the token, never the
token.

Provider keys are not the runner's. The run reads them from the bundle's
`.env`; the runner reads only its own variables out of that file, and removes
every `POSTERN_` and `SIGRIX_` variable from the environment of what it runs.
There is nowhere in the protocol for a key to travel.

## What it exposes

It binds `127.0.0.1` unless told otherwise, and checks the `Host` header of
every request, which is the whole of its defence against a page re-resolving
its own name to loopback. A browser page may call it only from an origin the
operator allowed with `--allow-origin`; nothing is allowed by default. Bound to
anything but loopback with no `POSTERN_INBOUND_TOKEN` to require, it serves
anyway and says so in its log: the operator may have a gateway of their own in
front of it.

A pulled bundle is verified against the distributor's `Repr-Digest` before
anything is written, and read under a size bound.

If a token leaks, regenerate it on your Sigrix account. The old one stops
working at the next entitlement check.
