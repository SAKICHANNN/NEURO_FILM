# P307 Cambridge HDR-deghost source-feasibility contract

## Question

Do the untouched official `exposure_stacks_part2.zip` and
`exposure_stacks_part3.zip` objects for Cambridge DOI
`10.17863/CAM.6881` expose an exactly inventoried, rights-clear and
group-isolatable physical multi-exposure observation with both motion-bearing
test stacks and independently captured motion-free reference stacks?

## Frozen source and role boundary

- The authoritative item is Cambridge Apollo record
  `b65f07c4-a4f5-4310-b5ec-a1c27fd52d6a`, licensed CC BY 4.0.
- Official item metadata, ORIGINAL/LICENSE bundle inventories, `README.txt`,
  `license.txt`, and bounded HTTP byte ranges from only the two named exposure
  archives are allowed.
- The README was inspected before this freeze. No archive byte, member body,
  image, thumbnail, RAW sample or pixel was requested before the contract.
- R1DN consumed only 12 `ground_truth/raw` groups from part1:
  `complex`, `handheld` and `lolm`, image sets 1--4. P307 forbids part1 and
  every one of those groups.
- P307 freezes the remaining six README categories
  `losm`, `multiview`, `nrm`, `occlusion`, `solm`, `sosm`, four image sets per
  category. Each eligible group must contain both `ghosted` and
  `ground_truth`, with five Canon CR2 and five JPEG members in each role.
- The README accidentally lists `exposure_stacks_part1.zip` twice. P307 may
  not infer part3 from prose; the two official bitstream identities and their
  central directories must independently establish the actual inventory.

## Frozen transport and execution

1. A discovery pass may read at most the final 131,072 bytes of each archive,
   locate a single-disk ZIP EOCD/ZIP64 EOCD, then read exactly its central
   directory. No local header or member payload is allowed.
2. Before formal replay, archive size, UUID, MD5, tail range, central offset,
   central size, entry count, central SHA-256 and canonical member-inventory
   SHA-256 must be committed into the config.
3. Each formal process re-fetches exact official metadata, README, licence,
   tails and central directories. Forward and reverse archive enumeration must
   produce byte-identical canonical reports.
4. Total network body bytes per formal process may not exceed 4 MiB. Archive
   body bytes are limited to the frozen tails plus central directories.

## Frozen gates

All gates must pass:

1. official item, bundle, bitstream, size and MD5 identities are exact;
2. README and licence bytes are exact and establish CC BY plus the paired
   motion-bearing/reference role semantics;
3. both archives honor exact HTTP Range with matching `Content-Range`;
4. each archive is a safe single-disk ZIP/ZIP64 with no unsafe name,
   encryption, symlink, duplicate name or unsupported compression method;
5. the six frozen categories form exactly 24 unique category/image-set groups;
6. every group has complete `ghosted` and `ground_truth` RAW/JPEG roles, five
   files per format and role, with one copied middle `gt3` exposure allowed in
   the ghosted stack but no other role overlap;
7. no part1 or R1DN-consumed category appears;
8. member-body, local-header, image, RAW and pixel reads are all zero;
9. two committed-head reports are byte-identical.

Any failure yields
`FAIL_CLOSED_CAMBRIDGE_HDR_DEGHOST_SOURCE_STRUCTURE_GAP_NOT_SCIENTIFIC_RESULT`.
It authorizes no archive materialization, member acquisition, model, fusion,
deghosting, metric, candidate slot or product action.

## Claim ceiling

A complete pass is only a private zero-pixel source and role lock for a future
separately preregistered physical multi-bracket/deghosting experiment. It is
not HDR quality, algorithm evidence, arbitrary camera support, A1/A4/A5,
package/schema/capability, candidate 3 or product admission.
