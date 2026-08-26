# P247 ACEScg OpenEXR 24MP resource contract

## Question

Does the unchanged exact R1DO writer consumed by P246 materialize one
deterministic 24-megapixel procedural ACEScg float32 master twice with exact
container/pixel identity while remaining within bounded local Windows memory,
wall-time, file-size and cleanup gates?

## Parent and non-goals

- Parent is P246 exact private consumer intake. P247 changes only probe size and
  measures local resources.
- Use the same exact producer writer Git object and exact OpenEXR 3.4.15 wheel;
  do not copy the writer or add OpenEXR to neuro_film dependencies.
- No external/project pixels, RAW, target, reference, network, GPU, display
  transform or image-quality metric may be used.
- This is not a product performance gate, streaming implementation, AP0/ST
  2065-4 ACES container, arbitrary metadata or public API result.

## Frozen probe

- Shape is exactly `4000x6000x3` float32 ACEScg (24,000,000 pixels;
  288,000,000 input bytes).
- Allocate the destination once, then fill bounded 64-row blocks from fixed
  integer x/y coordinates. The three channels are deterministic periodic
  scene-linear functions containing negative values and values above one.
- Input construction, hashing and writer input ownership are part of the worker
  peak. No full-size float64 coordinate mesh is allowed.
- Two fresh worker processes use labels `outer-a` and `outer-b`; labels and
  execution order are excluded from the scientific identity.

## Frozen execution and gates

1. Bind P246 evidence SHA-256, config SHA-256, writer source/blob/SHA and wheel
   size/SHA before implementation.
2. Install the wheel offline into one repo-relative `tmp/` workspace per formal
   controller. Never place durable data on C or D and never hardcode P.
3. Launch two fresh worker processes sequentially. Sample worker plus child RSS
   at no slower than 100 ms; worker wall time includes source construction,
   hashing, write and readback inspection.
4. Each worker must produce the same input SHA, OpenEXR SHA/bytes, decoded pixel
   SHA and standard AP1/D60 metadata. Decoded maximum absolute error is zero;
   input remains unchanged; negative and above-one values remain present.
5. Peak process-tree RSS for each worker is at most 2,147,483,648 bytes.
6. Worker wall time is at most 120 seconds and output is at most 1,073,741,824
   bytes.
7. Temporary OpenEXR, wheel target, worker reports and complete workspace are
   removed after both workers exit; retained formal output is only one
   canonical JSON report under repo-relative `outputs/`.

Any identity, equality, resource or cleanup failure closes this exact 24MP
runtime leaf. Do not change compression, resolution, data pattern, wheel,
metadata, thresholds or process topology after observing results.

## Claim ceiling

At most P247 can establish private local Windows CPython 24MP runtime/resource
feasibility for the exact R1DO/P246 ACEScg writer on one procedural probe. It
does not establish cross-machine performance, natural-image quality,
large-image streaming, arbitrary EXR support, an installed dependency, public
package/schema/capability, renderer integration or product admission.
