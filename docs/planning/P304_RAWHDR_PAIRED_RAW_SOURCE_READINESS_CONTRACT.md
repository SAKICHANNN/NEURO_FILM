# P304 RawHDR paired-RAW source-readiness contract

## Question

Does the official ICCV 2023 RawHDR release provide a rights-clear, exactly
inventoried and group-isolatable physical Canon RAW-to-HDR observation that can
open the final fresh candidate slot without requesting image, archive or model
payloads?

## Frozen source and scope

- Official repository: `https://github.com/yunhao-zou/RawHDR`, frozen at commit
  `c49e8b2d6ae83ee156aeb957a4c328d1df68e9bd` and recursive tree
  `eacc5b8c8b3a287cb5a954360608adc880e6467a`.
- Commit-pinned README and repository `LICENSE` are the only content sources.
- P304 may observe the exact OneDrive/Baidu locators printed in the README, but
  may not follow them, enumerate a folder, request an archive, image,
  thumbnail, model or checkpoint, or decode any pixel.
- Repository code rights and dataset-payload rights are independent. A root MIT
  licence passes only the code-rights gate unless official text explicitly
  grants the same rights to the photographed RAW/HDR payloads.

## Frozen facts and gates

Two fresh processes must independently bind the official Git commit, tree,
README and licence bytes and evaluate all gates:

1. **official identity** — commit/tree and repository title are exact;
2. **materially distinct physical observation** — official text states 324 real
   Canon EOS 5D Mark IV scenes, tripod bracket capture at -3/0/+3 EV, 0 EV RAW
   input and merged HDR target;
3. **anonymous payload locator** — official text exposes a current dataset
   locator without a contact or enrollment step;
4. **commercial-compatible code rights** — an explicit repository licence
   covers the executable source;
5. **commercial-compatible data rights** — official text explicitly licenses
   every RAW input and HDR target for this project's intended use;
6. **exact public inventory** — filenames, sizes and cryptographic hashes (or an
   equivalently exact manifest) bind the train and test payloads before access;
7. **group isolation** — public scene identities and a fixed split manifest can
   support scene-disjoint development and confirmation roles;
8. **exact replay** — both canonical reports are byte-identical.

Any failed gate yields
`FAIL_CLOSED_RAWHDR_SOURCE_RIGHTS_OR_MANIFEST_GAP_NOT_SCIENTIFIC_RESULT`.
The failure consumes zero candidate slots and authorizes no data/model request,
mirror use, licence inference, pixel read, training, inference or product map.

## Claim ceiling

P304 is a zero-pixel current official-source readiness audit. A complete pass
could open only a separately preregistered exact-object acquisition and
physical-observation preflight. It cannot establish HDR quality, arbitrary RAW
support, a shared operator, A1/A4/A5, package/schema/capability or product
admission.
