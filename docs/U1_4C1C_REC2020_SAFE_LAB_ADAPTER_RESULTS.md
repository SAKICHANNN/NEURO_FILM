# U1.4C1C Rec.2020 Safe-Lab Adapter Results

**Status:** pass

**Parent:** `ULT > U1.4 > U1.4C > U1.4C1`

**Contract:** `docs/planning/U1_4C1C_REC2020_SAFE_LAB_ADAPTER_CONTRACT.md`

**Contract SHA-256:** `3172ac64acb388b26f549a49481cab2aff7abf569c86fb0953955add3222f5e2`

**Implementation commit:** `5791bf9a65933f88194fe0501eddd16f2d1ec703`

## Implementation

- `src/color_engine/gamut.py` adds explicit destination-working-space gamut
  tests plus fixed-24-iteration source-segment and same-L/hue chroma
  compression.
- `src/color_engine/rec2020_safe_lab.py` adds the isolated display-linear
  Rec.2020 `WorkingImage` adapter for exactly the six frozen colour styles.
- The adapter consumes the C1A Lab conversion and C1B pure Lab kernel. It does
  not load profiles, infer stock identity, convert through sRGB, or call the
  production renderer/FilmFX.
- Provenance is copied without nested metadata or warning-list aliasing, and a
  research limitation warning is appended.

## Evidence

| Gate | Result |
|---|---|
| Six allowed styles x source/chroma modes | 12/12 finite and in Rec.2020 gamut |
| Preregistered Velvia witness linear-sRGB excursion | `0.479715288` >= `0.02` |
| Witness Rec.2020 output range | `0.023341233..0.975545287` |
| Gamut source endpoint / neutral endpoint failures | fail closed |
| Chroma policy fixed-L / hue direction | pass |
| Repeat determinism | byte-identical |
| Input mutation | none |
| Provenance and nested metadata aliasing | pass |
| Incompatible space/state/style/mode/input | fail closed |
| Frozen legacy seeded hashes | exact |
| Eight legacy full/tiled styles | unchanged at `1e-6` gate |
| Core focused tests | 55 passed |
| Wider colour/renderer-boundary tests | 101 passed |
| Complete CPU suite | 625 passed |
| Compile and diff checks | pass |

One development assertion initially used an absolute unnormalised Lab `a,b`
cross product and failed at `0.00390625` because magnitude scaled float32
rounding. It was replaced with the dimensionless angular-direction error that
actually encodes the preregistered hue-direction property. No implementation,
product threshold or frozen gate was widened.

## Decision

Retain U1.4C1C as an isolated research API. C1A-C now satisfy the frozen
working-space-aware Lab contract, including legacy parity and a mathematical
wide-colour witness. This does not demonstrate visual appeal or artifact safety
on real photographs.

Before any renderer integration, a separate U1.4C2 visual/OOD audit must compare
the six colour styles on bounded rights-cleared local inputs, inspect full-
resolution severe artifacts and wide-gamut failure behavior, and preserve the
same no-effects/no-production boundary.

## Claim ceiling

Isolated colour-only Rec.2020 safe-Lab research adapter with frozen legacy
parity and a wide-colour mathematical witness. No film-stock authenticity,
calibration, preference, HDR, ACES/OCIO, FilmFX, production renderer, profile,
schema or user-facing support claim.
