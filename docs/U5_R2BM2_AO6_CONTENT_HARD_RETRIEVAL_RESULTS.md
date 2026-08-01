# U5.R2BM2 colour-blind content hard retrieval

BM2 fails its frozen selector gates in two byte-identical runs. The report
SHA-256 is `374e4d5acb65a2d6b163127a3e9c70733577d4bffa8bab8fb50bfdf767379663`
and the stable evidence ID is
`4d4737c554fcf97f48ddc9730e5f751358d4062767d253aa16c09ef5f9e2b867`.

The selector uses only a fixed DINOv2-S embedding of source luma replicated to
RGB. It hard-selects the nearest development photo inside each held-camera
fold, with a development-only OOD threshold and BM1 medoid fallback. No source
colour statistics, target pixel, operator signature, output diagnostic,
training or dense blend enters routing.

The result is a routing failure rather than a renderer artifact. Retrieval
reduces median accuracy by 12.98% relative to the global medoid, recovers
`-45.0%` of the BM1 Oracle gain, improves only 4/17 sources and reaches an
18.18x worst-source error ratio. It exactly matches the Oracle case for only
2/17 sources. Median style retention remains 0.983 and new boundary fraction
is zero: plausible-looking strength does not make the selected adjustment
correct.

This closes the fixed global-semantic Top-1 selector without changing the
embedding, pooling, k, OOD threshold or case blending. BM1 still proves that a
case bank has evaluator-Oracle value; BM2 proves that global semantic
similarity does not recover it. One separately frozen classical grayscale
tone/layout nearest-case control remains justified because it tests
photometric structure rather than semantic identity. Failure there will close
simple source-only case retrieval on this population.
