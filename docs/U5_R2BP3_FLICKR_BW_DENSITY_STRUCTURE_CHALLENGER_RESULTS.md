# U5.R2BP3 B&W density-structure challenger result

BP3 is closed. Two formal reports are byte-identical at SHA-256
`990a567a...f259` with stable evidence ID `ebbd8346...2732`.

For each held scene, the candidate radial profile came only from the other six
scenes. It generated a deterministic random-phase log-density field, converted
that field to a mean-zero linear-luma residual, and applied one analytical
global scale that forbade hard clipping. Equal-energy white and reversed-shape
controls used the same bounded construction.

The safety mechanism worked: maximum mean-luma drift is numerical noise,
new boundary fraction is zero and median added-structure/edge correlation is
`.0012`. But the no-clipping scale collapses to `.0070` in the worst scene.
Consequently the median PSD-error gain over basic is only `.118%`, the gain
over white noise is `.067%`, reversed shape is slightly better, and high-pass
energy error improves only `.036%`. Five of seven scenes improve nominally,
but none of the frozen magnitude or shape-specific gates pass.

Do not lower the gates, replace the global safety rule with a per-pixel rescue,
or run visual review. BP2 remains useful descriptive evidence, while stochastic
structure fitting on this source is closed. A future leaf may test a genuinely
different deterministic density-domain acutance/spatial-response mechanism;
stock, emulsion, scanner, developer, calibration and product claims remain
closed.
