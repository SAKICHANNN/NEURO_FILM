# P268 HDR+ Burst Source Feasibility Contract

## Scope and parent

P268 is a DRPT L2, zero-pixel source-admission leaf under the Goal's
`rights-cleared paired/capture-time physical observation` lane. It tests whether
the official Google HDR+ Burst Photography Dataset exposes one bounded,
independently addressable RAW burst with the capture-time side information and
pipeline outputs needed for a later prospective experiment. It is not an
after-only candidate, HDR-quality test, model run, product integration or
candidate-3 admission.

## Frozen official sources

- Dataset page: `http://www.hdrplusdata.org/dataset.html`
- Google Research paper page:
  `https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/`
- Public bucket root: `gs://hdrplusdata/20171106_subset/`
- Licence named by the official page: Creative Commons Attribution-ShareAlike
  (`CC-BY-SA`); commercial product compatibility and share-alike obligations
  remain unresolved and are not inferred by this leaf.

The official page states that the curated subset contains 153 bursts / 37 GiB
and the full release 3,640 bursts / 765 GiB. Both bulk downloads are forbidden.

## Frozen enumeration and selection

1. List only immediate prefixes under
   `gs://hdrplusdata/20171106_subset/bursts/` using the installed Google Cloud
   CLI against the anonymously readable public bucket.
2. Normalize each prefix to its terminal burst ID. Require exactly 153 unique,
   path-safe IDs.
3. Select exactly one burst by the minimum pair
   `(SHA-256(UTF-8 burst_id), burst_id)`. No gallery, image, result or member
   content may influence selection.
4. After selection only, recursively list metadata for the selected burst and
   the matching two official result roots (`results_20161014` and
   `results_20171023`). Record object URL, size, generation, metageneration,
   CRC32C, MD5 where exposed, content type and update time.
5. Do not download object bodies, decode pixels, fit, train, infer or score.

## Admission gates

The source-feasibility leaf passes only if:

- official page and paper identities are repeat-fetch stable;
- the page explicitly describes 2--10 RAW DNG frames, capture metadata,
  lens-shading maps, `rgb2rgb.txt`, merged DNG, final JPG and CC-BY-SA;
- the curated burst-prefix inventory is repeat exact and contains 153 unique
  safe IDs;
- the selected burst is identical in forward/reverse enumeration;
- the selected burst contains at least two `payload_N.dng` members plus the
  side information needed to interpret the RAW burst;
- at least one matching result root exposes `merged.dng`, `final.jpg` and
  `reference_frame.txt`;
- selected object metadata exposes fixed sizes and immutable generations;
- the sum of a separately admissible one-burst acquisition is at most 1 GiB;
- network body downloads, object payload reads, DNG/JPEG/TIFF decodes, pixels,
  fit, training, inference and target scores are all zero.

Any missing licence statement, ambiguous object identity, missing RAW/sidecar/
result role, inventory drift or over-budget selected burst closes this exact
source leaf. Do not replace the burst, browse galleries to choose content,
download the 37 GiB subset, accept account terms, use mirrors or relax the
budget.

## Evidence and claim ceiling

Run two complete audits with reversed enumeration presentation, canonicalize
all scientific fields and require byte-exact reports. A PASS opens only a new
prospective contract for one exact burst's bounded member acquisition and
source-only capture-time/RAW D0. It does not authorize the download itself,
declare HDR truth or quality, establish product rights, consume candidate 3,
or create a package/schema/capability/product mapping.
