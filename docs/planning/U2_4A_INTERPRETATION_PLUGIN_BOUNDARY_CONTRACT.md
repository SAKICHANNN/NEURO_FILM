# U2.4A interpretation plugin boundary contract

Date: 2026-07-18

Node: `ULT > U2 > U2.4 > U2.4A`

Status: **frozen before implementation**

## Purpose and epistemic boundary

The v1 profile ontology already names `color_negative_neutral_scan`,
`color_negative_print`, `slide_direct_scan` and `bw_developer_scan`, but the
current renderer has no explicit interpretation-plugin boundary. U2.4A creates
that boundary without claiming that any real negative, slide, developer,
scanner or print response has been identified.

The current safe-rich profile remains `look_approximation`. Non-look
interpretations must remain production-ineligible until a separately gated
operator and evidence bundle exist.

## DoR

- U2.1A strict profile/replay validation passes;
- U2.5 current-profile adapter and U2.6 evidence visibility pass;
- current stock learning/operator fitting remains forbidden;
- no calibrated or S2/S3 interpretation profile exists;
- the worktree is clean at the parent commit.

## Allowed implementation

1. Add one pure `src/inference/interpretation.py` module containing a frozen
   interpretation request/result contract and registry validation.
2. Separate an interpretation's semantic ID, operator ID, input/output colour
   domains, evidence/claim ceiling and production eligibility.
3. Require float32 finite HxWx3 input, deterministic shape-preserving output,
   input immutability and JSON-safe metadata.
4. Unknown, unregistered, domain-mismatched or production-ineligible requests
   fail closed before returning output.
5. Provide three synthetic test-only plugins/fixtures for:
   `color_negative_neutral_scan`, `slide_direct_scan` and
   `bw_developer_scan`. Their operators may be minimal deterministic witnesses
   but must be labeled `synthetic_test_only` and never exported as production
   profiles.
6. Preserve current renderer/profile/recipe behavior and hashes; integration
   is a separate leaf.

## Required negative controls

- an interpretation name without a registered operator;
- a registered operator under the wrong interpretation;
- display-domain input supplied to a negative-scan-only contract;
- nonfinite/wrong-dtype/wrong-shape data;
- output mutation of the input or an aliased return;
- nonfinite/out-of-bounds/wrong-shape output;
- ndarray or mutable containers in public metadata;
- any synthetic plugin requested with `production=true`.

## Frozen gates

- each of three synthetic witnesses is deterministic, finite, bounded,
  shape-preserving and non-mutating;
- B&W witness output is exactly neutral-axis;
- slide witness is exact identity;
- negative witness is explicitly synthetic and invertible within a frozen
  numerical tolerance;
- all negative controls fail closed;
- no import/reference from renderer, profile schema, recipe schema or tracked
  production profile;
- focused/full/compile/diff checks pass.

## Forbidden fallback and claim ceiling

No real-film fitting, visual-to-physical inference, new stock profile,
calibrated claim, renderer integration, schema migration, default change,
scanner/process label invention, or use of the synthetic operators as style
teachers.

Passing proves only a safe software extension boundary using synthetic
witnesses. It does not prove a useful or authentic negative/slide/B&W
interpretation, a digital-to-film operator, or any real-film evidence.

