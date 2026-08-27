# P277 MakeHDR ISO 21496-1 decoder compatibility contract

## Node

`ULT > U1 > U1.5 > P277`

## Question

Can the unchanged pinned libultrahdr v2.0.0 Windows decoder and unchanged P87
absolute-Rec.2020 MatchView ingress consume three independently encoded
MakeHDR ISO 21496-1 JPEG samples, repeatably and without weakening production
fail-closed behavior?

P277 extends format-source coverage beyond the two Apple fixtures consumed by
P88. It is not an HDR-quality experiment: the MakeHDR source scenes are
synthetic and no independent HDR target is available.

## Frozen sources

- MakeHDR press page: `https://makehdr.com/press`.
- Press-kit archive: `https://makehdr.com/press/makehdr-press-kit.zip`.
- Pre-download HTTP facts observed on 2026-08-27:
  - content length: `27,089,268` bytes;
  - content type: `application/zip`;
  - ETag: `"576a49a45f3b3757d88e6edeeb301daf"`;
  - Last-Modified: `Thu, 27 Aug 2026 09:46:52 GMT`.
- The source page and bundled README state that the three `*-hdr.jpg` files
  carry ISO 21496-1 gain maps, that their paired `*-sdr.jpg` files share the
  same base image, and that the supplied synthetic sample files may be
  published without clearance.
- Existing decoder and ingress identities remain those frozen by P88/P87. No
  decoder rebuild, formula change or replacement executable is permitted.

Only the three sample-pair HDR JPEGs and their three SDR counterparts plus the
bundled README may become persistent P277 inputs. Screenshots, logo and other
press assets are excluded and must not be retained after source extraction.

## Execution

1. Download the exact bounded archive through the repository-relative
   P-backed data entry and bind archive/member hashes before decoder execution.
2. Verify the required six JPEG members and README, reject duplicate names,
   unsafe paths, links, unexpected required-member cardinality or source drift.
3. For each HDR JPEG, run the unchanged P88 decoder command to RGBA16F in a
   fresh temporary directory, then pass bytes through P87 unchanged.
4. Decode the paired SDR JPEG only with the existing SDR raster path for a
   base-image compatibility diagnostic. It is never an HDR target.
5. Run two fresh processes with opposite sample order. Canonical reports must
   be byte-exact after sorting records by scene.
6. Re-run the production loader rejection for every HDR input and a truncated
   input atomicity control. No production loader or core file may change.
7. Delete all formal decoded payloads and temporary outputs. Persistent source
   is limited to the selected seven files and a source manifest.

## Frozen gates

All gates are conjunctive:

1. press-kit HTTP/archive/member/source identities are exact;
2. exactly three required HDR/SDR pairs and one README are retained;
3. all three HDR files decode successfully to exact-length RGBA16F;
4. all decoded payloads pass P87 finite/range/opaque-alpha/profile rules;
5. each HDR result is materially non-SDR: maximum luminance exceeds
   `203.0 cd/m2` and decoded HDR pixels are not byte-equal to the SDR base;
6. each paired SDR base is compatible: equal dimensions and mean absolute
   RGB8 difference no greater than `1.0` code value after ordinary SDR decode;
7. sources remain immutable, production ingress still rejects all HDR files,
   and truncated input fails atomically;
8. forward/reverse process reports and stable scientific identities are exact;
9. network writes outside the one bounded archive request, hidden targets,
   fitting, training, and retained decoded payloads are zero.

Any failed gate closes P277 without replacing samples, changing the decoder,
altering headroom, relaxing thresholds, or using the MakeHDR web application.

## Claim ceiling

At most `PASS_PRIVATE_INDEPENDENT_ISO21496_JPEG_DECODER_COMPATIBILITY`: private
Windows compatibility for three exact independently encoded synthetic JPEG
gain-map samples through the pinned libultrahdr-v2-to-P87 chain. No natural or
captured HDR truth, grade correctness, complete ISO 21496-1 conformance,
HEIF/AVIF support, arbitrary-media support, display validation, public
package/schema/capability, default-loader change, product admission, film-stock
claim or candidate3 change is allowed.

