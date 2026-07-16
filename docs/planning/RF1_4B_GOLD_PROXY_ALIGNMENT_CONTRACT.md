# RF1.4B0 Gold100 display-proxy alignment contract

Date: 2026-07-16

Status: frozen before loading `transformations.pkl`

## Objective

Determine whether all 47 Gold100 preview/proxy siblings can be paired through
BlueNeg's official transformation metadata with sufficient whole-roll support
to preregister a later colour-transform consistency test.

This child performs no colour fitting and inherits no positive signal from the
closed RF1.4A preview route.

## Alignment evidence

BlueNeg documents `transformations.pkl` as a filename-keyed dictionary with a
perspective matrix and `(x0, y0, x1, y1)` bbox. The documented contract is that
the preview bbox has exactly the pseudo-ground-truth dimensions. The local
82,690-byte file is pinned to SHA-256
`d552b5c4606099234ccf02cfd252dc5652c5aa0a89fa3e002d8cba5252429601`.

Because pickle can execute code, ordinary `pickle.load` is forbidden. The
loader must inspect opcodes, reject persistent IDs/extensions/object
constructors, and allow only NumPy ndarray reconstruction globals.

## Pass gates

- exactly 47 Gold100 paired frames;
- at least five physical rolls and at least two pairs in every roll;
- every frame key exists exactly once;
- every matrix is finite, 3x3 and nonsingular;
- every bbox is integer, positive-area and inside its preview;
- every bbox width/height exactly equals its proxy width/height;
- all evidence hashes are checked before loading the pickle or image headers.

Any failure closes the current proxy lane. Homography guessing, resize-to-fit,
dropping hard pairs or fitting colour before the contract passes are forbidden
fallbacks.

## Next branch

A pass authorizes only a new frozen RF1.4B1 contract: whole-roll-held-out
comparison of identity, RGB mean/std or per-channel affine controls, a bounded
explicit colour operator, wrong-roll operators and paired targets. Promotion
will require improvement beyond simple global colour plus consistency across
held-out rolls. The output claim remains an archive print/scan display chain,
not isolated Kodak Gold stock response.
