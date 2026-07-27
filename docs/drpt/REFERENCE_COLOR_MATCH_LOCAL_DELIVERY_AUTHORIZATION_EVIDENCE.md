# Reference Color Match Local Delivery Authorization Evidence

Date: 2026-07-28

Status: **no-write local authorization complete; delivery commit remains
closed**.

## Decision

P38 separates verified staging from permission to perform a product action.
It does not infer delivery authority from P37 alone. It reruns P37 live and
cross-binds the exact P35 composition, P34 core verification and P30 product
staging authorization before issuing a narrowly scoped capability.

The fixed scope is `local-user-export`, state is
`authorized-for-local-delivery`, and claim ceiling is
`authorized-local-delivery-not-committed`. The object contains no destination
and performs no file operation.

## Contract

`ExternalLocalDeliveryAuthorizationV1` binds:

- exact P37 verification and P36 run IDs;
- exact P35 composition plan ID;
- exact P34 core staging verification ID;
- exact P30 product staging authorization ID;
- shared reference intent and ordered source count;
- output label `reference-look+film-effects`;
- fixed local-only scope, state and claim ceiling.

The schema SHA-256 is
`cd1b97aa1a48e0be37f66aa455aee31337efdd72004cf612ae41f65c803b067b`.

Before authorization, P37 is reconstructed from its own report path, report
hash and run ID. P30 must be atomically `authorized-for-staging`, and every
source must retain `accepted_for_product_staging=true` with action
`authorized-for-staging`.

## Adversarial evidence

Ten dedicated tests cover:

- exact full-chain authorization and schema/JSON roundtrip;
- byte-for-byte proof that successful authorization writes no file;
- live P36 output tamper;
- valid but foreign P35 composition;
- valid but foreign P34 verification;
- valid but foreign P30 authorization;
- public-scope, delivered-state, stock-label, committed-claim and canonical-ID
  escalation.

## Verification

- dedicated P38: 10 passed;
- combined color-match and FilmFX: 457 passed;
- compileall and diff check: passed;
- complete CPU suite: 1297 passed, one skipped and the unchanged 36
  isolated-worktree failures caused by missing ignored outputs or historical
  Windows checkout asset hashes;
- no color-match or FilmFX test failed.

Implementation commit: `18eb813`.

## Latest-main and producer propagation

- common base: `c03c321`;
- main snapshot: `5819b48`;
- consumer implementation: `18eb813`;
- consumer changed paths: 180;
- main changed paths: 119;
- exact changed-path overlap: zero;
- conflict-free merge tree:
  `523b4e0d7ebb65624f210db885cf9e3f0cc7ba97`;
- a fresh detached synthetic merge passed 127 P30-P38 and FilmFX tests and
  was removed.

D-PCT `d3e41bc` is clean and records BMKL as the strongest local PST50
development candidate, but explicitly not a promoted candidate. It changes
no producer schema, ABI, receipt or HDR rail, so P38 does not consume it.

## Remaining boundary

P38 authorizes a future local file transaction only. It does not commit,
deliver, apply or share pixels. Atomic export mechanics can be implemented as
a separate consumer leaf, but a synthetic fixture must never be used to claim
real product readiness. Genuine delivery still requires a fixed real producer
invocation and A1/A4/A5 plus P27-P30 passage.
