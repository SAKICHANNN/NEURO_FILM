# U5.R2AD0 — spektrafilm spatial-DIR source and non-duplication audit

## Decision

**One isolated external ablation is feasible; direct reuse remains blocked.**

The pinned spektrafilm source at
`3bb2c2d2801ff68b92019cf1dbcbb133d60832bc` contains two separable DIR
coupler behaviours:

1. a per-pixel same-layer and inter-layer inhibitor correction; and
2. an optional image-space diffusion of that inhibitor field.

This distinction matters because RF2.C0 retained the first behaviour but
disabled the second. Its runner set `debug.deactivate_spatial_effects=true`;
the pinned `digest_params()` consequently set
`film_render.dir_couplers.diffusion_size_um=0` while leaving the non-spatial
coupler correction active. The previously retained Ektar100/fixed-e0 result
therefore does not answer whether spatial inhibitor diffusion adds useful
local film-like behaviour.

## Exact source evidence

The ignored audit checkout is pinned to the same external revision as RF2.C0.
The decision artifact records exact hashes for the repository licences,
README, coupler implementation, parameter digester and parameter schema.
Relevant observed defaults at that revision are:

- inhibitor diffusion core: 20 micrometres;
- long diffusion tail: 200 micrometres;
- tail weight: 0.06.

The source describes diffusion as filtering the inhibitor correction before
the corrected exposure is mapped back through the density curves. This can
change local contrast and colour near spatial transitions; it is neither
grain nor halation, and it cannot be represented by RF2.C0's per-pixel
operator alone.

## Rights and implementation boundary

The code is GPL-3.0-or-later and the bundled profiles/LUTs are CC-BY-SA-4.0.
K-MCFM still has no root licence decision. Therefore:

- no source, profile, LUT or parameter implementation is copied into tracked
  project files;
- no linking, port, compatibility or production integration is authorized;
- the external runtime remains an ignored, isolated research comparator;
- outputs cannot become teachers, training targets or stock truth.

The numerical defaults are recorded as observed external-source facts, not as
calibrated film chemistry.

## Non-duplicate next leaf

AD1 may run exactly one source-bound ablation:

- same pinned external revision and isolated runtime as RF2.C0;
- same Ektar100 negative-to-Portra-Endura, fixed-e0 spectral path;
- same frozen nine-image gold set and display-sRGB proxy interpretation;
- all grain, halation, glare, lens blur, diffusion-filter and scanner-sharpen
  effects disabled in both arms;
- only DIR inhibitor spatial diffusion differs between arms;
- external defaults are frozen before output metrics are inspected.

AD1 must compare spatial-on against the already established spatial-off
control, a matched simple local-contrast control, and artifact diagnostics.
It must require exact replay and full automatic gates before any visual
review. A weak effect, basic/local-contrast equivalence, instability,
clipping, ringing, colour speckles or other severe artifacts closes the leaf
without parameter rescue.

Even a pass means only that this external spatial mechanism deserves a
future comparison slot. It does not identify a digital-to-film operator,
prove Ektar behaviour, authorize stock learning or open production.
