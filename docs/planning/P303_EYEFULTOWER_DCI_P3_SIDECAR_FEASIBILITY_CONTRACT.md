# P303 EyefulTower DCI-P3 authoritative-sidecar feasibility contract

## Question

Can the exact P292/P293 EyefulTower EXR be assigned an unambiguous linear
P3 color identity by a prospectively bound authoritative sidecar, without
rewriting P293 or choosing a white point after inspecting pixels?

## Frozen source boundary

- Image source is exactly P293 `40_DSC0001.exr`, 9,399,313 bytes, SHA-256
  `c7b65fe69568b4d04bbf162381fd6d74c9b9682057a0f9dd8b931ec6b038f174`.
- Dataset authority is exact EyefulTower commit
  `06a01a4915afc872b893c20a025a0e14598c8478`, README blob
  `ce06d91887cab8522dae3ceefe3941cd2821c675`, and README SHA-256
  `bf4fe411c0c0efb756cf18f70dd0eb01bf9679926e1fa91649602fae4bba1689`.
- The dataset declaration is only `Color space: DCI-P3 (linear)`; it does not
  contain a white-point name or chromaticity tuple.
- Color-space authority is restricted to the ICC color-encoding registry's
  DCI P3 entry and the official SMPTE ST 2113 publication. Both authority
  bodies must be acquired into the repo-relative source root under bounded
  size limits and hashed before formal evaluation.
- EXR channel and pixel reads are forbidden unless the exact dataset
  declaration plus the authoritative sources select exactly one complete
  RGB-primary and white-point tuple.

## Frozen gates

All gates must pass for intake:

1. P292/P293 evidence, exact image bytes and dataset authority identities bind;
2. both color-space authority objects are exact, bounded and independently
   agree on the P3 primary chromaticities;
3. the literal dataset declaration maps to exactly one authoritative white
   point, with no filename, JPEG, display practice or user-agent inference;
4. a canonical sidecar binds the source SHA-256, dataset declaration authority,
   complete chromaticity tuple and linear transfer identity before pixel read;
5. only after gates 1-4, the unchanged P296 private chromaticity arithmetic may
   decode the EXR and emit an owned linear-Rec.2020/D65 WorkingImage;
6. two fresh forward/reverse reports are byte-identical; source bytes remain
   unchanged and JPEG/network/pixel reads outside the frozen roles are zero.

## Stop rule and claim ceiling

If `DCI-P3` admits more than one authoritative white point, P303 must return
`FAIL_CLOSED_AMBIGUOUS_DCI_P3_WHITE_POINT_BEFORE_SIDECAR` with zero EXR pixel
reads and zero sidecar publication. Do not select P3D65, P3DCI, D60 or any
other white after the result; do not use modern HDR display practice to
reinterpret the 2023 dataset; do not inspect JPEGs or pixels to choose.

A pass would prove only one private exact-file authoritative-sidecar ingress.
A failure is an identity/source-contract result, not HDR quality evidence.
Neither outcome changes candidate count, P293, arbitrary EXR support, package,
schema, capability, product or stock claims.
