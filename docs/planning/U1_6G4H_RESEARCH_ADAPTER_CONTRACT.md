# U1.6G4H Research Adapter Parity and Lifetime Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4H`

**Status:** frozen / implementation ready

**Config SHA-256:** `f2b1829360767ee0bccbbca37ee608e5d2fda97f164937a415c5732413f48ea7`

## Purpose

G4G shows that the fixed staged-density executor plus a row composite fits the
local 24MP resource gates. G4H freezes the smallest reusable orchestration
boundary without integrating it into production:

```text
finite float32 display-sRGB base
  -> existing staged-density executor
  -> row-bounded calls to the existing composite_layers implementation
  -> one final float32 RGB array + scalar/tuple metadata
```

The adapter must not reproduce the density operator or screen equation. It must
not enter `scripts/render_film.py`, inference profiles, recipe schemas, CLI
arguments or package-level exports.

## Placement and API boundary

New code belongs at `src/filmfx/staged_density_adapter.py`, beside the existing
effect-specific tiled/staged adapters. It may import the existing staged
executor and compositor. Lower-level operator/compositor modules may not import
the adapter.

The only callable under test is:

```python
render_staged_density_halation_research(
    base_rgb,
    *,
    tile_size,
    source_row_chunk=64,
    coarse_row_chunk=7,
    composite_row_chunk=64,
    output_margin=0,
) -> tuple[np.ndarray, StagedDensityAdapterMetadata]
```

Version: `staged-density-halation-research-adapter-v1`.

The adapter remains a direct-module research API and is deliberately absent
from `src.filmfx.__all__`.

## Ownership and lifetime contract

- caller owns and retains `base_rgb`; the adapter never mutates it;
- staged executor owns temporary contexts and returns one internal RGB+alpha
  `FilmLayer`;
- adapter allocates exactly one returned full RGB composite;
- each composite row delegates to `composite_layers` using read-only slices;
- layer RGB/alpha is not returned, stored globally or embedded in metadata;
- metadata is a frozen dataclass containing only scalars, tuples and the frozen
  executor metadata; recursive inspection must find no ndarray or mutable
  container;
- successful public return contains exactly one ndarray: the final RGB;
- exceptions before/during executor or composite publish no tuple/output;
- scratch bytes remain zero.

This is a Python ownership/API invariant, not direct allocator telemetry. G4G
remains the measured process-RSS evidence.

## Frozen evidence

The config fixes two seeded, non-divisible synthetic shapes plus already-viewed
`u41-12` (hash and decoded shape frozen). Every case crosses:

- both listed executor tile sizes;
- composite row chunks 17, 64 and 113;
- output margins 0 and 4.

For each policy, compare against:

```python
layer, executor_metadata = execute_staged_density_halation_default(...)
reference = composite_layers(base, [layer], output_margin=margin)
```

The reference is test-only and may materialize the same layer. It does not
become a second production path.

## Frozen gates

- adapter float32 RGB bytes exactly equal the reference for every policy;
- rounded sRGB8 bytes exactly equal the reference;
- input bytes and writeability state are unchanged;
- repeated output and metadata are byte/value identical;
- recursive metadata contains zero ndarray or mutable list/dict/set;
- one and only one ndarray is publicly returned;
- scratch bytes equal zero;
- injected executor failure and injected second-row compositor failure return
  no partial public result;
- static source audit finds zero adapter imports in renderer, inference,
  profile/recipe or CLI modules and zero corresponding schema/argument changes;
- focused and complete CPU suites pass.

## Failure and collateral controls

The trigger class is partial orchestration: an internal layer may already exist
when a later composite row fails. A surface-only `try/except` could hide the
error while retaining partial output. The adapter instead propagates failure
and has no callback/output sink, so no partial result can escape.

Equivalent triggers include nonfinite compositor output, invalid row chunks,
invalid margins and injected executor/provider errors. Tests must cover invalid
geometry and both pre-layer and post-layer failures. Existing legacy renderer,
simple/physical halation and direct compositor tests remain regression controls.

## DoR / DoD

DoR: G4G is frozen; current clean head is
`60adc953aef4db4862a80f0443e939b607d50068`; the existing compositor and
staged executor are authoritative and production renderer files are untouched.

DoD: committed module/tests/audit; all frozen policies and failure/static gates
pass or branch closed; complete CPU suite; evidence propagation; scoped
commit/push.

## Branches

- **Pass:** retain the isolated adapter and open a separately frozen full-path
  adapter/encoding or 100MP audit; production wiring still requires its own
  gate.
- **Parity failure:** repair adapter orchestration only; do not change the
  executor or compositor gates.
- **Lifetime/failure escape:** reject adapter until ownership is explicit and
  tests pass.
- **Production import/schema drift:** remove the wiring and keep the adapter
  isolated.
- **Real-case severe difference:** impossible under byte parity; any such
  observation indicates evidence or file mismatch and fails closed.

## Claim ceiling

A pass proves an isolated Python research adapter reproduces current compositor
output while keeping layer ownership private. It does not prove renderer/CLI or
recipe integration, streaming decode/encode, complete-path memory, 100MP,
cross-platform parity, default promotion, physical calibration or stock style.
