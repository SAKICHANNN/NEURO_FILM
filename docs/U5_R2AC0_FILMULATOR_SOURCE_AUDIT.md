# U5.R2AC0 — Filmulator mechanism and source audit

## Decision

**Retain the mechanism prior; block direct source reuse.**

The [official Filmulator site](https://filmulator.org/) describes a RAW
pipeline that simulates film development and claims four relevant behaviours:
large bright-region compression, small-region local-contrast enhancement,
highlight saturation retention and saturated-highlight detail retention.
Those are materially different from a global LUT and directly relevant to the
project's “obvious style without severe artifact” objective.

The official repository was inspected read-only at commit
`57fbaec57555432d86d3aa632990cd8fa09114ad` (2026-04-04). Its high-level
development loop contains:

- bounded exposure activation;
- three colour layers sharing one spatial developer field;
- developer and per-layer resource consumption during density formation;
- spatial developer diffusion;
- reservoir exchange and periodic agitation;
- density readout.

This is a generic development simulation, not a bank of measured stock
profiles. No film/digital paired target or stock-identification evidence is
provided by the mechanism itself.

## Licence boundary

The audited repository states GNU GPL version 3 or later. Exact retained
source hashes include:

- `LICENSE`: `1919ab24...74602`
- `README.md`: `884ebf3b...92750`
- core orchestration: `3f43150f...ee22`
- reaction: `e5a549c7...e5b6d`
- diffusion: `a8a330ae...980e0`

K-MCFM still has no root `LICENSE`; choosing project licensing or GPL
compatibility is an owner decision outside this Goal. Therefore:

- do not copy, translate, vendor, link or port Filmulator source;
- do not copy its constants, defaults, discretization or parameter semantics;
- do not call a new implementation Filmulator-compatible;
- do not integrate an external Filmulator binary into production.

The ignored shallow checkout is audit cache only and is never committed.

## Non-duplicate value

The useful idea is **shared-resource spatial coupling**. It differs from:

- global 1D/3D LUTs, which have no neighbourhood state;
- existing halation, which redistributes exposure around highlights;
- generic bilateral residuals, which directly predict/add a local correction;
- procedural grain/dust, which add optical/material texture.

A shared developer-like resource can reduce growth in broad bright regions
while diffusion replenishes boundaries and small regions differently. It may
therefore produce local tone and colour behaviour that a global transform
cannot.

## Allowed next leaf

Only a separately frozen, synthetic-only, first-principles
shared-resource-diffusion representation may open. It must be written from a
new mathematical contract, use no Filmulator code or constants, and test:

- finite bounded state;
- identity at zero strength;
- uniform-field reduction;
- translation/rotation symmetry;
- bounded local adjacency response;
- no ringing, halo reversal, seams or channel instability;
- exact replay and explicit finite support/resource cost.

Even a pass is a generic `film-development-inspired` mechanism result. It
cannot identify a stock, process chemistry or calibrated response, and it
cannot enter production without a separate licence and product gate.
