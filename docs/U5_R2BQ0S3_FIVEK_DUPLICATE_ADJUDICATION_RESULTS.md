# U5.R2BQ0S3 source-only duplicate adjudication

BQ0S3 passes twice with byte-identical reports at SHA-256
`f6838274b3ce73c827c0c9435947723d0cf9dc589733585109dc3641b8c697a6`.
It is explicitly an adaptive successor after BQ0S2 failed; the frozen BQ0S2
result remains failed.

The prior-pool candidate has dHash/pHash distances `4/28`, normalized luma
correlation `.4634`, and zero mutual ORB ratio matches. The internal candidate
has distances `4/32`, luma correlation `-.2061`, and zero mutual matches.
Neither is a confirmed duplicate under the separately frozen DCT-layout or
ORB-homography branches. Synthetic tone, crop and low-information collision
controls pass.

No Expert or filtered target pixel was opened. This permits only a separately
versioned off-diagonal evaluator Oracle bound to the unchanged target-blind
381/128 manifest. It does not authorize router training or any film, stock,
calibration, preference, product, or BQ0S2-reinterpretation claim.
