# P158 Halo-Aware Safe-Lab Style Memory Contract

## Purpose

P157 removes the reference-fit peak and exposes the unchanged full-frame
safe-Lab style application at about 0.99--1.01 GB. P158 tests whether the exact
finite vertical support of its Gaussian luma-detail term can bound temporary
memory without changing the style operator.

The functional parent is
`6e1387b0f98a18ba7e89f9c68e47bc143205ac5b`; the measurement baseline is
`c715b5ad2a7834f8af95532c2c82e225d8c6fd29`.

## Frozen implementation

The candidate may change only `src/color_match/render.py` and its focused
tests. It must:

1. compute one unchanged full-image `SafeLabSourceContext`;
2. divide output into 128-row cores;
3. call the authoritative `apply_safe_lab_transform` on each core expanded by
   five real source rows above and below, clipped only at image boundaries;
4. crop the expanded result back to the core and write one preallocated
   float32 output;
5. derive gamut-adjustment diagnostics before releasing obsolete source/styled
   Lab arrays ahead of final output conversion.

Five rows exceed the current Gaussian radius implied by sigma `1.1` and the
unchanged default filter settings. A pre-contract 257-by-389 probe with the
default nonzero luma-detail policy reproduced the full-frame float32 result
exactly with zero differing scalars. The formal test must cover non-divisible
heights, image edges, both working spaces, zero/nonzero luma-detail and source/
chroma gamut policies.

Changing Gaussian sigma, boundary mode, safe-Lab equations, recipe/policy
identity, gamut math, producer contracts or the main renderer is forbidden.

## Frozen gates

The P157 6 MP inputs and baseline/candidate/candidate/baseline order remain
fixed. A pass requires exact output, recipe and normalized-report identities;
identity fallback; zero residue; focused full-versus-halo bit exactness; at
least 256 MiB median RSS reduction; median RSS ratio no higher than `0.70`; and
median worker-wall ratio no higher than `1.10`.

The thresholds are frozen before implementation and formal measurement.

## Claim ceiling

A pass proves only the local Windows/Python safe-Lab style stage has an exact
halo-row implementation. It does not prove total high-resolution readiness,
generic safe-Lab tiling, main/native/device parity, RAW/HDR/video, producer
compatibility, A1/A4/A5, visual quality or product admission.
