# U6.P6L — University of Chicago repeat-scan source result

## Result

The bounded source acquisition is mechanically valid but fails the frozen
repeat-scan correlation gate, so residual scanner-noise analysis does not
open.

- exact acquired subset: 88,475,062 bytes;
- three single-frame `uint16` TIFFs, each `3835 x 3841`;
- official MD5 and local SHA-256 checks pass;
- both estimated integer shifts are `[0, 0]`;
- registered correlations are `.9951791` and `.9931007`;
- frozen minimum required correlation: `.995`;
- two reports are byte-identical at
  `5aa10c16...c5b557`, with stable evidence
  `51bb644e...4523f`.

The third repeat misses the source gate. Per the preregistration, no affine
photometric normalization, alternate crop, registration expansion, threshold
change, or selective-repeat removal is allowed after observing this result.
The NPS/ACF/row-column residual analysis and comparison with P6A therefore
close.

The exact CC BY 4.0 files and hashes remain useful source provenance and
negative evidence. This result does not characterize scanner noise or film
grain and makes no stock, process, scanner-calibration, colour, product or
population claim.
