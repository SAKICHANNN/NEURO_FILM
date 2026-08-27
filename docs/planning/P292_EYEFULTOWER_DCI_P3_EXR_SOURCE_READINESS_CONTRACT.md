# P292 EyefulTower linear DCI-P3 EXR source-readiness contract

## Question

Does the current official EyefulTower release provide a rights-compatible,
exactly inventoried and bounded real-capture HDR EXR source that can open a
separate strict linear DCI-P3 OpenEXR intake leaf?

## Frozen official sources

- Repository: `facebookresearch/EyefulTower`, commit
  `06a01a4915afc872b893c20a025a0e14598c8478`, tree
  `845efc2cb2c8b6ebefda8df91bf940990ae2aa49`.
- Commit-pinned recursive Git tree, `README.md` and `LICENSE` only.
- Anonymous S3 prefix:
  `EyefulTower/playroom_small/images-1k/` in the official bucket
  `fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15`.
- Exact prefix listing and its `md5sums.txt` are metadata. The deterministic
  future payload candidate is the lexicographically first EXR object. P292 may
  issue a HEAD request for that object but may not request its body.

The official README states that these HDR images are merged from nine-photo
RAW exposure brackets, are linear DCI-P3, and are stored as uncompressed
float32 EXR. The same README states that JPEG derivatives are white-balanced
and tone-mapped from the HDR EXRs. Therefore JPEGs are not independent targets
and are forbidden as quality truth in this or any child admitted by P292.

Only bounded GitHub API/raw responses, one S3 ListObjectsV2 response, the
official checksum text and one selected-object HEAD response may be requested.
Repository cloning, EXR/JPEG bodies, pixels, decoding, fitting, training,
inference, rendering and scoring are forbidden.

## Frozen readiness gates

All gates must pass simultaneously:

1. repository commit/tree, README and MIT licence identities are exact;
2. the official README identifies the source as real-capture nine-bracket HDR,
   linear DCI-P3 and uncompressed float32 EXR;
3. the current release grants MIT rights to all content;
4. the bounded S3 listing is complete and contains exactly 126 EXRs from 14
   cameras and nine synchronized filename suffixes;
5. the official checksum file contains exactly one MD5 entry for every listed
   EXR and no unlisted EXR;
6. the selected lexicographically first EXR has the frozen key, byte length,
   multipart ETag and official MD5, and its HEAD request succeeds without a
   response body;
7. synchronized filename suffixes provide explicit capture-group identities
   suitable for a future group-disjoint intake protocol;
8. total response bytes remain within the frozen metadata-only budget and all
   EXR/JPEG payload requests, pixel reads and scientific executions remain zero;
9. two fresh forward/reverse reports are byte-identical.

## Stop rule and claim ceiling

Failure of any gate yields
`FAIL_CLOSED_EYEFULTOWER_DCI_P3_EXR_SOURCE_READINESS`. No source, checksum,
format, grouping or threshold rescue is allowed in this leaf.

A complete pass opens only a separately preregistered download-integrity and
strict DCI-P3-float32-EXR-to-WorkingImage intake leaf for the one selected
object. It establishes no decoded-pixel conformance, HDR quality, bracket
quality, independent target, arbitrary EXR support, package/schema/capability,
product admission, stock evidence or candidate-3 change.
