# P259 real DNG to ACES2065-1 OpenEXR master contract

## Question

Can the five already-consumed P98 real DNG WorkingImages traverse the existing
official `Linear Rec.2020 -> ACEScg` transform, the exact P249 AP1-to-AP0
writer, and the strict P251 AP0-to-AP1 reader without silent clipping, metadata
drift, source mutation or retained temporary media?

This closes one mature composition gap. It does not add a DNG decoder, IDT,
tone map, stock operator, display transform or default product route.

## Frozen inputs and components

- Reuse exactly the five P98 DNG rows and their frozen source/WorkingImage
  hashes. No new source, target, metadata or network request is allowed.
- Use the current P257-guarded P98 loader unchanged. Every selected file is
  already confirmed free of P244/P257 guarded profile families.
- Convert each scene-linear `linear_rec2020` WorkingImage to ACEScg with the
  pinned OCIO 2.5.2 ACES 2 CG config and its exact `Linear Rec.2020` to
  `ACEScg` processor.
- Load the exact P249/R1DT writer from its frozen producer Git object into an
  isolated temporary module. Install only the exact offline OpenEXR 3.4.15
  wheel into the same owned temporary root; do not copy producer source or a
  dependency into this repository.
- Read each emitted AP0/D60 file through the unchanged strict P251 ingress.
  Delete the EXR after its receipt/readback facts are captured.

## Frozen roles and order

- The five P98 rows are evaluation-only, previously consumed real files.
- Forward and reverse processes differ only in row enumeration; canonical
  scientific rows are sorted by `source_id`.
- No target pixels, target score, fitting, parameter selection or quality
  comparison exists in P259.

## Frozen gates

1. P98/P249/P251/OCIO sources, configs, evidence, producer Git object and
   offline wheel identities are exact;
2. all five source files and frozen P98 WorkingImage hashes are exact before
   composition, and source bytes remain unchanged afterward;
3. every OCIO result is owned, contiguous, finite float32 ACEScg and direct
   official processor bytes are exact;
4. every EXR is a single non-deep scanline float32 RGB ZIP file with exact
   AP0/D60 chromaticities, adopted neutral, `acesImageContainerFlag=1` and
   `colorInteropID=lin_ap0_scene`;
5. decoded AP0 bytes equal the independently calculated frozen AP1-to-AP0
   float32 matrix result exactly;
6. strict P251 readback returns owned contiguous writable ACEScg with maximum
   absolute AP1 error at most `3.814697265625e-6` (`2^-18`);
7. all input samples below `-1e-5` remain negative and all input samples above
   `1.00001` remain above one after readback; new exact-zero/one samples are
   zero;
8. each temporary EXR is at most 512 MiB; network, target and external pixel
   reads are zero; producer source is temporary only;
9. all EXR, installed-wheel and module residues are zero after each process;
10. forward/reverse scientific payloads and two complete committed-head
    reports are byte exact.

Any failure closes P259 without changing transforms, matrices, source rows,
precision, tolerance, writer, reader, dependency, output compression or gate.

## Claim ceiling

At most P259 can establish private exact-five real-DNG composition mechanics
from the existing scene-linear Rec.2020 ingress to temporary strict
ACES2065-1/AP0 OpenEXR masters and back to ACEScg. It cannot establish
arbitrary DNG/EXR/ACES or SMPTE conformance, calibrated sensor/IDT or Adobe
rendering, photographic/colorimetric/display quality, persistent media,
public dependency/API/package/schema/capability, film-stock evidence,
single-reference matching, product admission or candidate 3.
