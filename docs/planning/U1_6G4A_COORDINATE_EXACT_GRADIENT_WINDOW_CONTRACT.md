# U1.6G4A Coordinate-Exact Gradient Window Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4A`

**Status:** frozen / implementation ready

## Purpose

Both physical and density weighted-source paths use `np.gradient(y)` before
local and global blurs. Calling `np.gradient` independently on row chunks would
turn every chunk boundary into a false image boundary. U1.6G4A freezes a bounded
original-coordinate window primitive that reproduces current full-frame
float32 gradient bytes without materialising a full gradient field.

## Frozen semantics

- input is one finite float32 scalar field with declared `(H, W)`, both axes at
  least two pixels;
- the caller supplies a window reader accepting ordered global
  `(y0, y1, x0, x1)` bounds and returning exactly that finite float32 array;
- the primitive expands the requested core by one pixel on every available
  side, calls the current `np.gradient` default (`edge_order=1`, unit spacing),
  and crops `gy/gx` back to the core;
- one-sided differences occur only at the declared original-image boundary;
  interior chunk/window boundaries always have both neighbours;
- outputs are contiguous, read-only float32 arrays with immutable metadata.

This deliberately matches the current effect, not a new Sobel, Scharr or
colour-gradient definition.

## Placement and scope

Add an independent module under `src/filmfx/`, public exports and focused tests.
An array-backed reader helper is allowed for parity tests and current callers.
No current effect, G1/G2/G3 result, renderer, profile, recipe or CLI changes.

Metadata records source/core/expanded shapes and bounds, expanded input bytes,
two-output bytes and maximum one-pixel halo. It does not claim library-internal
temporary allocation or total renderer memory.

## DoR

- G3 passes and names `coordinate_exact_gradient_window` as a hard missing
  capability;
- current effects use NumPy default gradient on a 2-D float32 luma field;
- clean pre-contract HEAD is `f5be8a8`;
- no renderer, download or training process is active.

## DoD and gates

1. full-window output is byte-identical to full-frame `np.gradient`;
2. irregular interior, edge, corner, single-row-core and single-column-core
   windows are byte-identical to slices of the full result;
3. row-chunk and 2-D tile assemblies are byte-identical to full gradients for
   multiple non-divisor sizes;
4. expanded bounds never exceed one pixel beyond the core or source bounds;
5. outputs and metadata repeat byte-identically and arrays are read-only;
6. invalid source shape, bounds, reader result shape/dtype/finiteness and reader
   failures fail closed;
7. focused tests and complete CPU suite pass;
8. G3 may remove only this exact missing capability after evidence propagation;
   row-chunked global staging remains missing and integration remains false.

## Branches

- **Pass:** retain the primitive and open row-chunked U1.6G1 stage construction.
- **Boundary mismatch:** repair coordinate expansion/cropping; do not loosen
  byte parity or change current gradient semantics.
- **Unbounded/full-gradient allocation:** reject the implementation.
- **Reader-contract failure:** fail closed rather than infer shapes or cast.

## Claim ceiling

A pass proves exact current-gradient windows with a one-pixel declared input
halo. It does not prove effect parity, complete field execution, physical
accuracy, renderer integration, stock/calibrated response, total-memory bounds
or 100MP performance.
