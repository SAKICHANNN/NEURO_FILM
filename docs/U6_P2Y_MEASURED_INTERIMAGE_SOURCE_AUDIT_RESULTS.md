# U6.P2Y measured interimage source audit results

The exact primary-source audit closes without an operator fit. The official RIT
record for Helen Shin's 1996 thesis describes measured digital-drive to
processed positive-35mm-film spectral response, but its 171-page PDF was only
available through the browser viewer during this audit; command-line retrieval
returned an HTTP 202 challenge and no bypass or local thesis PDF was retained.

The independently accessible [official CIC 1994 companion paper](https://library.imaging.org/cic/articles/2/1/art00020)
was retained and parsed at SHA-256 `920f74ea...e3bbf9`. It reports Kodak
Ektachrome 100 Plus Professional, a Solitaire 8xp CRT recorder, 60 spectral
absorptivity exposures, 11-step single-channel ramps, 21 modelling exposures
and a 36-colour RGB-to-CMY dataset. It also describes the fitted family: three
one-dimensional LUTs followed by a row-sum-one 3x3 matrix.

The paper does not publish the complete numeric RGB-to-CMY rows. The frozen
minimum was 24 complete machine-readable or explicitly tabulated rows; the
observed count is zero. Figure OCR, manual digitization, heuristic
reconstruction, fitting and rendering were forbidden and were not performed.
Two reports are byte-identical at `11a0fb05...a5498`; the stable evidence ID is
`d5163e03...32384`.

This source family therefore remains mechanism literature, not reusable
measured operator evidence. U6.P2Z may audit the latest Emulating Emulsion
author package for exact code, fitted parameters or paired patch rows. Nothing
here identifies a camera-to-film, scanner, process or named-stock response.
