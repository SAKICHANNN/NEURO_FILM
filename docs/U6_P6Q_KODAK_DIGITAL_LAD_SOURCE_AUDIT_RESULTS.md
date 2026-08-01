# U6.P6Q Kodak Digital LAD source-audit results

The official [Kodak laboratory page](https://www.kodak.com/en/motion/page/laboratory-tools-and-techniques/),
H-387 guide, Cineon format description and both LAD archives pass exact
source/hash/page/path and expansion checks. H-387 defines the CINEON negative
aim as `printing density = 0.002 * 10-bit code value`, keeps printing density,
Status M and D-min distinct, separates negative from interpositive operation,
and recommends an RGB LAD code of `445/445/445`.

The frozen archive gate nevertheless fails before decompression: each ZIP has
a `51,019,264`-byte member, above the preregistered `50,000,000`-byte member
limit. The limit was not changed after inventory. Two reports are byte-identical
at `85cf7b6a...b7c6a`; stable evidence ID `a47f3934...c8294`. No archive member
was decompressed and no image header, pixel, operator or render was consumed.

The machine-readable DPX/Cineon image route therefore closes without rescue.
U6.P6R may use only the exact H-387 guide tables to implement and test a typed
neutral code-to-density primitive. This evidence is recorder/print calibration
documentation, not a stock colour response or measured local process profile.
