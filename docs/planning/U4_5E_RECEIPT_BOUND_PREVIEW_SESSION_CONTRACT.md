# U4.5E — Receipt-bound three-stock preview session

## Question

Can the existing private U4.5C warm preview session bind the exact cache-index
bytes supplied by its producer, so that later mutation of cache metadata or of
the expected preview hashes fails closed before a new session is admitted,
without changing U7.3H or U4.5C historical behavior?

U7.3H validates the current input, profile and three preview files against the
hashes stored inside `preview-cache.json`. U4.5C then retains independently
hashed immutable preview bytes. The cache index itself is not bound by a
caller-held identity. Consequently, a modified index can supply a different
parent contract, dimensions, pixel count, look amount, claim text or expected
output hash and still become the authority used by a new admission. This is a
cache metadata/provenance integrity gap, not a pixel, colour, stock or browser
question.

## Frozen mechanism

- Preserve the existing unbound U7.3H/U4.5C functions and evidence unchanged.
- Add a separately named receipt-bound cache inspection function. It requires
  one lowercase 64-hex SHA-256 for `preview-cache.json`, reads that file once,
  verifies those exact bytes, parses those same bytes, and then applies the
  unchanged complete input/profile/output validation.
- Add a separately named receipt-bound session admission function. It uses only
  the bound inspection result, reads each preview into owned immutable bytes,
  independently verifies each payload against its bound row hash, and rechecks
  the complete input and profile identities after payload admission.
- Expose the accepted cache-index SHA through a new frozen wrapper around the
  unchanged `VerifiedThreeStockPreviewSnapshot`; do not add fields to or alter
  the old snapshot/session types.
- A warm lookup returns only the wrapper retained in memory and performs zero
  filesystem reads, pixel decodes, renders or writes.
- The formal run uses the exact U4.5C source/profile/preview generator and
  computes the receipt immediately after the existing atomic cache-index
  publication. No preview pixels, render parameters, stock operators or
  thresholds change.

## Formal gates

1. four fresh receipt-bound admissions and eight warm lookups per admission
   reproduce the exact U4.5C ordered Velvia 50, Portra 400 and Ektar 100 bytes;
2. the accepted receipt equals the SHA-256 of the exact cache-index bytes used
   for parsing and is present in every warm snapshot;
3. maximum individual warm lookup wall time remains at most `0.300` seconds;
4. warm lookup filesystem reads, pixel decodes, renders and writes are zero;
5. a wrong, uppercase, short, non-hex or non-string receipt rejects before
   input/profile/output hashing;
6. independent mutations of parent contract, preview width, pixel count, look
   amount, claim text and one row output hash each reject before a new bound
   admission, even when all media files remain unchanged;
7. replacement of the index after admission leaves the current immutable
   session exact and causes a new admission with the original receipt to reject;
8. preview-file, input and profile mutations retain the existing fail-closed
   behavior after the index receipt passes;
9. forward/reverse fresh-process scientific payloads are exact after excluding
   timing, all owned scratch is removed, and source/profile files remain exact;
10. existing U7.3H/U4.5C functional and evidence tests remain unchanged and
    pass.

## Claim ceiling and stop rule

A pass establishes only private Windows/Python receipt-bound admission and
session-resident warm lookup for three already-rendered deterministic
`film-inspired / Look Approximation` previews. It is not render acceleration,
pixel authenticity, calibrated or physical stock response, stock
distinguishability, browser/default integration, installer, public API, package
or release evidence.

Do not rescue a failure by accepting an optional/missing receipt, normalizing
receipt case, rereading the index after verification, changing preview bytes,
changing the three operators, weakening complete-file hashes, increasing the
latency ceiling or rewriting U7.3H/U4.5C evidence.
