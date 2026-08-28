# P308 DNG LinearizationTable explicit-delegation contract

## Question

Can the private exact-five P98 DNG ForwardMatrix ingress make every
`LinearizationTable(50712)` delegation explicit before camera decode, reject
ambiguous or structurally invalid declarations, and preserve all five frozen
WorkingImage pixels and prior ingress guards?

## Frozen authority and cohort

- Adobe DNG SDK 1.7.1 build 2652 fixes tag code `50712`, TIFF type SHORT and
  count range 2 through 65,536.
- The exact P98 five-source cohort and P244 frozen output hashes are reused
  without replacement or new pixels.
- Four sources contain no `LinearizationTable`. The exact Blackmagic Pocket
  Cinema Camera 4K row contains one primary-IFD SHORT table with 4,096 entries,
  uint16 little-endian SHA-256
  `37ff0c95bd8e810277ee2e0f34c33c613638473e28802af758c7b7317aeca9a0`.
- The existing LibRaw camera-raster decoder remains unchanged. This leaf does
  not independently execute or validate its table arithmetic.

## Frozen behavior and gates

Before `_decode_camera_linear_dng`:

1. no table returns no delegation receipt;
2. exactly one valid SHORT table returns a source-bound receipt containing
   numeric tag, IFD path, entry count and exact uint16 payload SHA;
3. multiple tables, wrong type, count outside 2..65,536 or non-uint16 payload
   reject with zero camera-decode calls;
4. the public loader appends one explicit warning only for the Blackmagic row;
5. every exact P98 WorkingImage float32 hash, source hash, working contract and
   all previous profile/opcode/version guards remain unchanged;
6. two fresh forward/reverse reports must be byte-identical.

Any drift closes the leaf without table implementation, warning suppression,
row replacement or relaxation. Passing establishes only an explicit private
delegation receipt for the exact P98 cohort.

## Claim ceiling

Private exact-five P98 `LinearizationTable` structural validation and explicit
LibRaw-delegation receipt only. No independent table arithmetic, Adobe/vendor
parity, full or arbitrary DNG renderer, image quality, default-loader,
package/schema/capability, stock evidence, candidate3 or product admission.
