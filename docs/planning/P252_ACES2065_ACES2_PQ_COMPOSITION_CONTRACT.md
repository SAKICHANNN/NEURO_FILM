# P252 ACES2065-1 to ACES 2 PQ composition contract

## Role

This is the frozen execution contract for one private HDR interoperability
leaf. It is not a stock, matching-quality, display-quality, or product
admission plan. P251 already accepts one strictly identified ACES2065-1/AP0
OpenEXR and returns an ACEScg/AP1 scene-linear `WorkingImage`. The retained
official ACES 2 adapter and canonical Rec.2100-PQ RGB16 PNG writer remain
separate because the adapter currently rejects the exact
`acescg_ap1_d60` working-space identity produced by P251.

## Question and hypothesis

Can the exact P251 ingress compose with the retained official ACES 2
Rec.2020 1000-nit output and canonical PQ PNG writer by adding only a strict
ACEScg identity branch, while preserving every frozen legacy WorkingImage
output hash?

The hypothesis is that `acescg_ap1_d60` plus `scene_linear` requires no source
colour conversion before the existing official display/view processor. The
identity branch must return an owned C-contiguous copy and must be numerically
identical to the P251 pixels. Any other new working-space spelling remains
unsupported.

## Frozen inputs and dependencies

- exact P249/P251 synthetic 7x9 ACES2065-1 container:
  `1a25181406270142c98f2169897d3a4c468ec50c979fff9077d37f956c888726`;
- exact P251 AP0-to-AP1 ingress and its OpenEXR 3.4.15 wheel;
- exact built-in OCIO 2.5.2 ACES 2 CG config and
  `Rec.2100-PQ - Display / ACES 2.0 - HDR 1000 nits (Rec.2020)`;
- exact existing canonical partition-invariant Rec.2100-PQ RGB16 PNG writer;
- the untouched U1.4E 986-row fixture and its four frozen legacy adapter
  hashes for `linear_srgb` / `linear_rec2020` and SDR / HDR targets.

No natural image, film scan, target reference, paired payload, network source,
or candidate-3 role is opened.

## Implementation boundary

The only permitted shared-core semantic change is recognizing exact
`working_space == "acescg_ap1_d60"` inside the official ACES 2 WorkingImage
adapter and returning an owned contiguous copy as ACEScg. Existing
`linear_srgb` and `linear_rec2020` branches remain byte-for-byte behaviorally
compatible. Generic raster/RAW/EXR dispatch, default renderer behavior,
profiles, recipes, schemas, and product capability maps remain unchanged.

One new private composition callable may:

1. call the exact P251 loader;
2. apply the official Rec.2020 1000-nit ACES 2 view;
3. publish through the existing canonical RGB16 PQ PNG writer;
4. return a bounded receipt without retaining mutable input aliases.

## Formal execution and controls

Two fresh processes execute forward and reverse row-partition order. Both
start from the same exact AP0 container generated from the frozen P249 lattice.
Before accepting the composition, formal execution must prove:

- exact source, wheel, config, module, and runner identities;
- P251 WorkingImage identity, ownership, metadata, and AP1 roundtrip gate;
- identity conversion max error exactly zero and no input mutation;
- official composed float output exactly equals a direct ACEScg display/view
  processor invocation;
- canonical RGB16 samples and complete PNG bytes are identical across row
  partitions and fresh processes;
- strict Rec.2100-PQ CICP, range, finite, create-only, and failure-atomic rules;
- all four frozen U1.4E legacy adapter hashes remain unchanged;
- malformed/wrong-space EXR and foreign destination controls reject before a
  usable or overwritten output;
- zero network access and zero owned temporary residue.

## Decision and stop rule

Pass retains only one private exact-file interoperability composition. Any
identity, legacy-regression, direct-parity, range, publication, atomicity,
replay, or cleanup failure closes P252. Do not rescue with clipping, exposure,
tone changes, alternate ACES view/config, approximate metadata, half/uint
ingress, tolerance relaxation, different source rows, or generic EXR routing.

## Claim ceiling

Private Windows/Python exact P249/P251 ACES2065-1 OpenEXR to official ACES 2
Rec.2020-PQ canonical RGB16 PNG mechanics only. No arbitrary EXR or HDR image,
photographic/display quality, HDR10 mastering metadata, SMPTE certification,
public dependency/package/schema/capability, stock evidence, single-reference
matching, calibrated reference, or product admission.

## Rollback and propagation

The contract, implementation, formal evidence, and tracker propagation are
separate local commits. Reverting the implementation commit restores the old
explicit rejection of ACEScg WorkingImages without changing P251 or any
legacy output. A pass or fail does not alter the bounded candidate counter.
