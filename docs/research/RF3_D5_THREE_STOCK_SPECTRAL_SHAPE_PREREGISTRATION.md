# RF3.D5 three-stock spectral-shape preregistration

RF3.D5 tests the last unscored common physical-prior domain retained by the
current first-party Velvia 50, Portra 400 and Ektar 100 documents. It compares
the published layer spectral-sensitivity shapes only after subtracting each
layer peak and applying one common `-1.25` log-sensitivity floor. Absolute film
speed, exposure scale and chart floor therefore cannot decide the result.

The exact page rasters, wavelength and sensitivity axes, common wavelength
grid, curve-completion rule, interpolation, three-pixel digitization
uncertainty penalty and all materiality gates are frozen before writing the
curve observations. Every pair must retain both mean-absolute and RMSE
separation after the conservative uncertainty subtraction, and at least one
layer peak must differ by 10 nm. Overlays make the manual centerline trace
auditable.

A pass retains only a non-renderable K=1 manufacturer prior. A failure closes
spectral curves as a three-stock discriminator. Neither outcome supplies the
controlled pixels, scanner interpretation, target closeness or blind stock
distinguishability required by SF3.A3.
