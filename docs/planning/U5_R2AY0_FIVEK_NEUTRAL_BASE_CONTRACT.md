# U5.R2AY0 FiveK bounded neutral-base contract

This leaf tests one distinct hypothesis: a content-safe canonicalization layer
may reduce input exposure and white-balance variation before a fixed film-look
operator. It does not learn film style.

The retained source is exactly 64 FiveK RAW/Expert-C pairs. Source feasibility
is decided before fitting by hashes, pair identity, EXIF make/model groups and
the frozen 25% maximum group share. All evaluation holds out complete camera
model groups.

The first operator family is deliberately small and explicit:

```text
source statistics
  -> bounded parameter predictor
  -> logit exposure/contrast + channel WB shifts
  -> bounded luma-centred saturation
  -> deterministic RGB
```

Only the parameters may be learned. The renderer remains deterministic and
cube-safe. Identity, one global parameter vector, hard nearest neighbour and
ridge prediction are required controls. A per-pair fit is an evaluator upper
bound, not an inference method.

Expert C is digital-retouch supervision for a neutral photographic base. It is
not film, stock, scanner or calibrated truth. AO6 and all other film-look
parameters remain frozen and outside training. Only a neutral-base predictor
that beats the global control on held camera models, retains a positive paired
interval, introduces no boundary pixels and passes an independent fixed-look
ablation may continue.

Authority:
`configs/u5_r2ay0_fivek_neutral_base_source_v1.json`.
