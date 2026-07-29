# Reference-match output metadata policy

Status: P168 implements and enforces
`neuro-film.reference-file-output-metadata-policy.v1`.

The policy is a separate immutable layer over
`reference-file-output-capabilities.v1`; it does not change the existing
capability payload or encoder bytes. Before a staged image can enter the file
transaction, the consumer:

- hashes the staged file before and after inspection;
- validates PNG structure, CRCs, exact ICC/cICP binding and rejects all other
  ancillary chunks;
- permits JPEG only with one exact thumbnail-free JFIF APP0 and the complete
  exact sRGB ICC APP2 sequence, rejecting EXIF, XMP, IPTC, comments and other
  application segments;
- permits TIFF only with the encoder's structural tag allowlist, identity or
  absent orientation, exact sRGB ICC and the fixed generated Software value;
- fails the whole transaction before publication if any check fails.

The nine advertised output tuples pass. Product-path tests also start from a
JPEG carrying source description, artist, user comment and non-identity
orientation and prove PNG/JPEG/TIFF outputs contain none of that source
metadata. Injected metadata regressions leave no output, recipe or staging
residue.

Claim ceiling: this proves the current Neuro-Film reference-match encoders and
transaction boundary do not copy source metadata. It is not a sanitizer for
arbitrary third-party files or future formats.
