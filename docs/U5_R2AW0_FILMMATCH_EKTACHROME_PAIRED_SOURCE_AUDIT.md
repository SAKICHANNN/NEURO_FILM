# U5.R2AW0 — FilmMatch Ektachrome/Sony paired-chart source audit

FilmMatch publishes a bounded chart set for practising film profiling. Its
shooting protocol is materially stronger than the previously closed
CineStill lead: the film and digital cameras observe the same chart setup,
the lens is moved between cameras, exposures span -5 EV through +5 EV, and
reflective plus emissive targets are photographed under multiple illuminants.

The public Drive contains 136 chart TIFFs arranged as 33 reflective and 35
emissive candidates for each of Kodak Ektachrome and Sony S-Gamut3.Cine, plus
two TIFF validation frames and two Resolve DRX files. The frozen expected
payload is 140 files / 4,255,132,151 bytes. A bounded pilot confirms a genuine
same-chart pair: both files are uint16 RGB TIFFs, but the film scan is
2160x3840 while the Sony file is 1080x1920, and neither embeds an ICC profile.

This passes source-interest screening, not fitting readiness. Exact download
integrity, condition-based pair mapping, explicit Sony and scan colour states,
and a group split that does not leak exposure or illuminant conditions must
all pass before fitting.

No explicit reusable dataset licence was observed. FilmMatch's author-hosted
practice download supports internal, non-redistributed research, but source
pixels and fitted source-derived assets must not be redistributed or promoted
to commercial/product use. A single unknown roll/process/scan session cannot
establish cross-roll Ektachrome calibration even if paired chart fitting later
succeeds.

The exact acquisition and claim limits are frozen in
`configs/u5_r2aw0_filmmatch_ektachrome_paired_source_v1.json`.
