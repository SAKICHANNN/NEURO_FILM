# P88 pinned libultrahdr decoder consumption contract

## Question

Can the private P87 absolute-Rec.2020 MatchView ingress consume HDR RGBA16F
bytes emitted by one exact official libultrahdr v2.0.0 command-line build on
the two already pinned CC-BY-4.0 gain-map fixtures, without changing the
production file loader or claiming arbitrary Ultra HDR support?

## Frozen inputs

- Official source: Google libultrahdr tag `v2.0.0`, commit
  `b2aacb366e1542cfc29605cb0d8a0ebd06bb07f8`.
- Exact external decoder executable SHA-256:
  `cffdfcae90b2e999260877b646c36c1d04b87d81745c12417c089ca7099d5df0`.
  The path is supplied at execution time and never stored as a drive-specific
  project contract.
- Mandatory fixtures: the exact unmodified U1.5C
  `apple_gainmap_old.jpg` and `apple_gainmap_new.jpg`, each 384x512, with
  their existing source hashes and CC-BY-4.0 source manifest.
- Exact decoder invocation per fixture:
  `ultrahdr_app.exe -m 1 -j INPUT -o 0 -O 4 -z OUTPUT`.
- P87 conversion/profile/version rules remain unchanged.

Both fixtures are already consumed by U1.5C/U1.5D. P88 is a retrospective
private mechanics/compatibility check, not fresh HDR-quality confirmation.

## Frozen gates

1. executable, producer R1BL evidence, P87 evidence and both fixture hashes
   match before decoder invocation;
2. decoder exit code is zero and emits exactly `384*512*4*2` bytes per row;
3. decoded RGBA16F passes P87 finite/range/opaque-alpha rules unchanged;
4. output MatchView profile/reference-white/render-bridge facts are exact;
5. source and decoded payload identities bind each row independently;
6. fixture bytes remain unchanged;
7. truncated input exits nonzero and publishes neither decoded payload nor
   MatchView;
8. fixture order reversal and two fresh-process normalized reports are exact;
9. the existing production loader continues to reject both gain-map fixtures;
10. no public package/schema/capability/default/product declaration changes.

Any gate failure closes P88 without changing the command, decoder binary,
fixture set, output format, range, alpha rule or P87 arithmetic.

## Claim ceiling

Private Windows execution compatibility for one exact official libultrahdr
v2.0.0 executable, two exact consumed CC-BY-4.0 fixtures and the retained P87
decoded-payload ingress only. No arbitrary Ultra HDR/ISO 21496-1 support, HDR
quality, captured truth, display validation, public decoder/package/schema,
capability, product or delivery admission.
