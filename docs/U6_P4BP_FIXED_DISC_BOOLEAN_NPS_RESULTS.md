# U6.P4BP fixed-disc Boolean NPS result

U6.P4BP is a formal negative. A clean-room implementation of the fixed-radius
Poisson-disc Boolean covariance from Newson et al. was fit only on the frozen
build frequency bands of the five P4BL historical Fuji B&W spectra. Each
material shared one radius; density states had independent coverage and
analytic amplitude. A separately fit one-Gaussian covariance per state was the
simple control.

Two fresh processes produced the same report SHA-256
`4f289048b384aef3ad83ff9279d89682c84099101cdcaa10cecf7b2647b77232`
and stable evidence ID
`98288cc78212b7629f57b549a59806d122df5918ede8abc063bb9633d44aafd2`.
The Boolean candidate confirmation median log10 RMSE was `.81202` versus
`.34271` for the Gaussian control, a `-136.94%` relative change. Its worst
error ratio was `1.33348`. Neopan improved only `6.14%`; Medical X-ray
regressed `139.37%`, and its fitted coverage decreased across increasing
density. Four frozen gates therefore fail.

This closes only the fixed-radius equation family on these measurements. The
roughly 3.2--3.4um fits are effective correlation parameters, not identified
grain radii. Variable-radius or clustered-particle geometry is materially
different and may be tested prospectively; no fixed-radius frequency split,
bounds, optimizer or gate rescue is allowed. P4BL/P4BM remain the measured
second-order references, and no photographic render or product profile opens.

Primary source: [Newson et al., Realistic Film Grain Rendering](https://doi.org/10.5201/ipol.2017.192).
