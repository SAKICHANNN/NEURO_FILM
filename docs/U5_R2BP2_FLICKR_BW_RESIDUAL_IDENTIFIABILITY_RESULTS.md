# U5.R2BP2 B&W residual identifiability result

BP2 passes its frozen exploratory gates. A bounded public-source audit found
nine CC BY 2.0 Flickr composites by one author; seven explicitly contain the
same scene as film on top and digital on the bottom. Two incomplete rows were
excluded before pixel acquisition. The seven selected source JPEGs total
1,449,449 bytes and split deterministically into fourteen role-labelled PNGs.

All seven pairs pass the unchanged SIFT/RANSAC registration contract. Two
registration reports are exact at SHA-256 `c22b13fb...b391`; the autonomous
overlay review found no obvious split, role or warp failure. After per-scene
33-quantile monotone tone normalization, two BP2 reports are byte-identical at
`46009c9c...2b47`.

The remaining B&W high-frequency direction is positive across all seven
scenes: median film/basic luma high-pass energy is `1.2085x`, all six pooled
radial PSD bins are positive, and six of seven leave-one-scene-out residual
shapes meet the frozen cosine gate. Correctly aligned residual RMS is `.6168x`
the shifted control; median edge correlation is `.2876` and the JPEG block
ratio is `1.0015`, so neither edge leakage nor JPEG blocking explains the
aggregate result under this pilot.

This opens exactly one leave-one-scene-out, mean-preserving density-domain B&W
stochastic-structure challenger against basic, white-noise and wrong-shape
controls. It does not identify a stock, emulsion, developer, process or
scanner response. The data are seven display composites from one author, and
the per-scene tone normalization uses the target; the evidence is therefore a
generic display-chain mechanism pilot, not a product operator or calibration.

Sources: [Flickr tag surface](https://www.flickr.com/photos/tags/filmvsdigital/)
and [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
