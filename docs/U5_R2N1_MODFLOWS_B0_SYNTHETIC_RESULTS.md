# U5.R2N1 ModFlows B0 Synthetic Audit Results

**Date:** 2026-07-26

**Node:** `ULT > U5 > U5.R2 > U5.R2N1`

**Decision:** **closed before real images**. The pinned B0 checkpoint is
architecture-compatible and its clean-room RGB flow is repeatable, smooth,
positive-orientation and near-identity when content/style embeddings match.
However, the encoder fails palette/content separation and cross-palette raw
flows leave the RGB cube.

## Asset and reproducibility

- B0 checkpoint: `18,970,914` bytes;
- checkpoint SHA-256: `124f7b42...d3eacb`;
- tensor-only `weights_only=True` load: pass;
- exact EfficientNet-B0 classifier shape: `515x1280 + 515`;
- clean-room implementation commit: `d2cc79b...25533b`;
- two formal reports byte-identical at `0e657a32...34596`;
- embedding and transfer repeats: exact;
- four focused tests and all **886** CPU tests pass.

No official source was copied or imported. B6, training data, stock-labelled
pixels and real reference images were not accessed.

## What passed

The independently frozen `4-64-3` tanh velocity interpretation is compatible
with the checkpoint. Under fixed eight-step RK4:

- same-palette geometry cosine reaches at least `.99628`;
- same-embedding forward/reverse maximum error is `5.48e-5`;
- minimum sampled Jacobian determinant is `.63683`;
- maximum sampled Jacobian norm is `1.60684`;
- all values and repeats are finite and deterministic.

This is meaningful: the checkpoint can parameterize coherent, invertible and
moderate-gain explicit colour flows without an image decoder.

## Decisive failures

### Palette-only embedding fails

An exact pixel permutation preserves every RGB value and the complete colour
histogram, but changes spatial arrangement. The minimum embedding cosine falls
to `.72546`, below the frozen `.98` gate.

Different warm/cool palettes have mean normalized embedding distance
`.06833`, while the same palette under geometry change has `.08510`. The
palette/geometry ratio is only `.80286`, below the required `2.0`; geometry
changes influence the embedding more than the deliberately large palette
change.

The embedding may still be useful for generic image similarity, but it is not
identified as a palette-only case key and cannot route stock/mode experts
without reintroducing content shortcuts.

### Raw Style-safe range fails

Warm-to-cool/cool-to-warm RGB-cube probes span
`[-.07296, 1.11860]` without clamping. Positive Jacobians do not guarantee
gamut preservation. A post-hoc clamp, strength reduction or projection would
be a new policy and is explicitly forbidden as an N1 rescue.

## Branch decision

- no A0 real-image reference pilot;
- no stock-labelled pixel access or film/operator claim;
- no encoder fine-tuning, B6 download, preprocessing/solver/order change,
  clamp, projection or strength rescue;
- retain N1 as negative evidence: an invertible parameter flow can be
  numerically regular while its selector is content-sensitive and its raw map
  is not Style-safe.

The broader project should retain separate content and mode/style spaces and
continue seeking data-eligible hard case retrieval or bounded explicit
operators. Ultimate remains active.

## Claim ceiling

Clean-room synthetic evidence that one external ModFlows B0 checkpoint is
architecture-compatible and numerically smooth but fails palette/content
separation and raw RGB bounds. No unlicensed code reuse, real-image safety,
stock identity, film operator, calibration, preference or production claim.
