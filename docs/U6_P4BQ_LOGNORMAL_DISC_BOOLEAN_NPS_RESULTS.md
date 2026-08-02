# U6.P4BQ lognormal-disc Boolean NPS result

U6.P4BQ is a formal negative and closes the independent Poisson-disc Boolean
NPS family tested here. It replaced P4BP's fixed radius with one bounded
lognormal radius distribution shared by the density states of each material,
while retaining the exact held frequency bands and controls.

Two full optimizations are byte-identical at report SHA-256
`efc4f154659922bf0a40972b170bc52dc274145cbf74dd4324b59c3ed4617ee9`
and stable ID
`968cb5a14ebbf19877ab8a07322c6b3dab3389719b56349f908da7f1194b4ffc`.
Confirmation median log10 RMSE is `.81444`, versus `.34271` for the Gaussian
control and `.81202` for fixed discs. This is `-137.65%` versus Gaussian and
`-0.30%` versus fixed discs; the worst-error ratio to Gaussian is `1.39013`.
Both fitted log-radius sigmas approach the lower bound, and the Medical X-ray
coverage sequence decreases to the `.02` lower bound as density rises.

The added radius distribution therefore supplies no useful held-band
explanation and numerically collapses toward the already rejected fixed-radius
case. No radius bound, truncation, quadrature, frequency split or gate rescue
is allowed. A new leaf must change the spatial process itself, such as an
explicit clustered germ process. Effective fitted radii are not microscopic
grain-size measurements; no image render or product path opens.
