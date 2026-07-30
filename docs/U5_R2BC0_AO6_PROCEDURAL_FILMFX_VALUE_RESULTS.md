# U5.R2BC0 AO6 Procedural FilmFX Value Result

BC0 held fixed AO6 colour and added two existing deterministic effect arms:
monochrome encoded-RGB grain at `.012`, and the same grain followed by simple
halation at `.14`. The exact 16-image AO7 population was reused only as a new
effect-development ablation; AO7 colour promotion was not reopened.

Two complete renders are byte-identical at SHA-256
`e01081a6ec2a452da0c12c492f5d2e4f31e67803907045cda8f643e0dd45d783`.
Grain changes a median 84.04% of pixels, all 16 images support the halation
mask, and no new raw clipping appears. The automatic gate nevertheless fails:
the worst grain high-frequency chroma P99.9 is `.01302`, above the frozen
`.004` ceiling.

The mechanism is explicit. The legacy layer adds equal encoded-RGB noise, but
the compositor clips channels independently after the layer. AO6 pixels near
different channel boundaries therefore acquire chromatic high-frequency
residuals even from nominally monochrome noise. This is the same artifact
class that makes coloured speckles unacceptable.

Visual review is forbidden. Both legacy arms close without strength or gate
rescue. The next distinct algorithm is shared optical-density noise in linear
RGB with analytical bright-end safety and chromaticity preservation.
