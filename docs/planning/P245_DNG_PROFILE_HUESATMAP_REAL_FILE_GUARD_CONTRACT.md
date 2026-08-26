# P245 real-file DNG ProfileHueSatMap guard contract

## Question

Does the unchanged P244 no-silent-drop guard reject two exact rights-cleared
real DNGs known from P242 metadata to contain distinct 3D and 2.5D
`ProfileHueSatMap` tables, before any LibRaw camera-raster decode?

## Frozen sources and roles

- Use only raw.pixls.us repository IDs `1052` (DJI FC220) and `2644`
  (Fujifilm X100S), both bound CC0/Public Domain rows in the exact P240
  repository snapshot.
- Bind their already-published SHA-256 identities from P240/P242.
- Download through only the exact `nice` URLs present in that snapshot, with a
  combined 60 MiB network/body ceiling and no redirect to another host.
- Publish create-only under repo-relative
  `data/reference_color_match/p245_dng_profile_guard_v1/raw/`.
- P242 EXIF remains the expected tag-structure oracle; it is not a pixel or
  rendered target.

## Frozen execution

1. Acquisition verifies URL host, response length ceiling, exact SHA-256,
   `.dng` suffix and create-only publication before a source is eligible.
2. Each formal process hashes and opens both DNGs with TIFF metadata only.
3. It records the actual guarded tag family and calls unchanged
   `load_dng_forward_working_image` with the exact source identity.
4. The call must reject with `DngForwardRasterError` naming the real tags.
5. `_decode_camera_linear_dng` is instrumented and must remain at zero calls.
6. No image sample, target, reference, preview, operator output or encoded
   image may be decoded or written.

## Frozen gates

- both exact source hashes and CC0 snapshot rows pass;
- DJI exposes dimensions/Data1/Data2 and a three-dimensional value axis;
- Fujifilm exposes dimensions/Data1/Data2 and a one-slice 2.5D value axis;
- every actual guarded tag is reported by the P244 rejection;
- rejection occurs before camera decode for both rows;
- source bytes are unchanged;
- forward/reverse scientific payloads and two fresh-process reports are byte
  exact;
- acquisition scratch and partial files are zero after publication.

Any failure closes this exact confirmation. Do not replace a row, redownload
from a mirror, decode pixels, apply P243, infer missing tags, relax diagnostics
or continue from a partial file.

## Claim ceiling

At most P245 can confirm P244's private predecode refusal on two exact CC0 real
DNG files containing structurally distinct HueSatMap profiles. It cannot
establish profile application, full DNG conformance, arbitrary RAW support,
image quality, vendor parity, default-loader integration, package/schema/
capability, film-stock evidence or product admission.
