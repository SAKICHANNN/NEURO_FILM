# U6.2B — Boolean Grain Existing-Image Crop Frontier Contract

**Status:** frozen before candidate rendering or metric inspection

**DRPT level:** L2

**Parent:** U6.2A synthetic Boolean/Poisson representation pass

**Primary writer:** current Codex Goal session

## Question

Can either frozen U6.2A radius policy be composed as a bounded explicit
luminance-grain residual on difficult existing-image crops, remain visibly
grain-like, and avoid severe artifacts relative to both the source and the
accepted U1.6F legacy heuristic?

This is a B0 development visual frontier. It is not a measured-film grain
study and cannot complete U6.2.

## Frozen inputs

The five `512x512` crops are pinned before any new grain output exists:

- ID 11 centre: saturated red and deep-shadow regression;
- ID 14 centre: inherited high-ISO noise and soft detail;
- ID 37 top-left: smooth sky gradient and cloud texture;
- `FS_FACE_01`: face, skin and hair;
- ID 21: face, uniform textile and fine insignia.

Every file path, source hash, decoded size, crop box and per-sample seed is
frozen in config. The source set is the existing U41 evaluation manifest.
The FilmSet face is content stress only and is not a film-grain reference.

## Frozen Boolean composition

1. Decode the crop as sRGB8 and convert to D65 linear sRGB.
2. Compute linear luminance and BOX-downsample it to `128x128`.
3. Cap only the Boolean-model intensity input at `255/256`.
4. Render the unchanged U6.2A Boolean coverage at zoom 4.
5. Bilinear-upsample the `128x128` reference luminance to `512x512`.
6. Form `d = grain - reference`.
7. For `d >= 0`, use `rgb + .12*d*(1-rgb)`.
8. For `d < 0`, use `rgb + .12*d*rgb`.

The signed headroom equation is an explicit range-preserving composition, not
a fitted operator. No output clamp is allowed. Both radius `.22` and `.38`
use the same `.12` strength. The already accepted legacy residual at `.018`
is a comparator, not truth.

## Frozen automatic gates

Each Boolean candidate must pass on all five crops:

- exact source hash/size and exact crop;
- finite linear output structurally inside `[0,1]`;
- absolute mean linear-luma drift at most `.015`;
- linear-luma residual RMS in `[.004,.04]`;
- sigma-4 low-pass linear-luma RMSE at most `.02`;
- new sRGB8 hard-endpoint fraction at most `.5%`;
- exact repeated float output;
- byte-identical formal reports.

Only automatic survivors receive visual review. Visual promotion requires zero
confirmed severe failures across all five full-resolution crops. Face/object
corruption, red/chroma speckle, posterization/banding, blocks/seams, large
unintended clipping or objectionable noise domination veto a candidate.

## Branches

- **One or two Boolean survivors:** retain only severe-clean policies as
  research candidates and open a separately frozen broader-image/full-frame
  resource frontier.
- **Automatic failure:** close the failing policy without changing strength,
  radius, crop, metric or threshold.
- **Visual severe failure:** veto regardless of numerical or physical appeal.
- **Only legacy looks acceptable:** keep the legacy heuristic; U6.2A remains
  representation evidence but does not integrate.
- **Both Boolean policies are bland:** do not raise strength after inspection;
  record the negative visual result.

## DoD

- isolated explicit composition helper and property tests;
- formal deterministic runner, report and full-resolution comparison pages;
- two byte-identical formal runs;
- autonomous full-resolution severe adjudication;
- targeted and full CPU tests;
- evidence propagation, scoped commits and push.

## Claim ceiling

B0 autonomous existing-image crop evidence for two frozen clean-room Boolean
grain policies versus the accepted legacy heuristic. No measured NPS,
stock/process/scanner calibration, physical realism, preference, full-frame,
production or named-stock claim.
