# SF3.A0T RGB2RAW capture-metadata explicit ISP preregistration

## Question

Does capture white balance provide enough genuinely new physical information
to improve a fixed, portable, explicit RAW-to-phone-ISP colour operator on
unseen capture groups, beyond equal-capacity camera-static, daylight-WB and
source-only gray-world controls?

This is candidate 1 of the bounded final three-candidate natural cycle, but the
counter increments only when calibration target scoring begins. It is not an
after-only reference inversion and it does not predict RGB with a neural model.

## Frozen mechanism

The archive's standardized RGGB planes are divided by the metadata white
level and reduced to `R, mean(G1,G2), B`. The candidate applies the exact
capture white-balance diagonal, then one fixed 12-parameter logit-affine
operator per camera. Both operations are serialized explicit colour math. The
operator is fitted only on capture-group-disjoint fit rows. At application it
reads RAW pixels and the bound capture metadata, never a target or reference.

The three legitimate controls have the same 12 learned parameters per camera
and differ only in observation: no WB, camera daylight WB, or source-only
gray-world WB. A within-camera cyclic metadata permutation tests attribution.

## Roles, order and stop

For each camera, SHA-ranked numeric-normalized capture groups allocate 40 fit,
12 calibration and 12 sealed confirmation groups; all remaining groups stay
reserve-unread. One pair per group is independently SHA-ranked. Fit data are
read first, every operator is frozen and serialized, then calibration targets
are read once. Sealed targets remain unread unless every calibration gate
passes. A second fresh process reverses row enumeration.

The exact gates and hashes are authoritative in
`configs/sf3_a0t_rgb2raw_metadata_explicit_isp_d0_v1.json`. Any failure closes
candidate 1 without changing roles, metadata fields, capacity, ridge, sampling
or thresholds. A pass remains a private two-camera RAW ISP mechanism result;
it cannot map A1/A4/A5 or promote a package, schema, capability or product.

## Paper boundary

The current physical-parameter direction is informed by CVPR 2026 PPISP and
the camera-parameter conditioning precedent in CVPR 2024 ParamISP. No paper
code, weights, model architecture or data are copied. PPISP's released JPEG
source already failed the direct-pair geometry contract and is not rescued or
mixed into this experiment.
