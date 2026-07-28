# U5.R2AJ0A — RawTherapee HaldCLUT source, rights and semantics audit

Date: 2026-07-28  
Node: `ULT > U5 > U5.R2 > U5.R2AJ0A`  
Decision: `licensed_bounded_external_operator_bank_acquisition_feasible`

## Question

Can the RawTherapee Film Simulation Collection supply a fixed, licensed and
explicit RGB-to-RGB operator universe for an independent Look Approximation
frontier after R2AI1 closed, without treating preset names as real-film,
stock-response or unpaired-operator truth?

## Decision

Yes, for one bounded archive acquisition and structural-only audit. The
collection is a global HaldCLUT bank, the archive version and size are stable,
and the archive's own README explicitly licenses the collection under CC
BY-SA 4.0. The primary AJ0 universe is frozen to the 194 non-Creative colour
CLUT file entries already visible in the remote ZIP central directory.

This is not a stock-data pass. The collection documents only that its names
describe looks designed to approximate film stocks. It does not provide the
film/digital pairs, measurements, scanner/process records or per-preset
construction lineage needed to validate those names. Every retained item
therefore remains an `external-look/<id>` / `film-look CLUT` control with a
`Look Approximation` ceiling.

## Exact upstream evidence

- Official documentation:
  `https://rawpedia.rawtherapee.com/Film_Emulation`
  (retrieved old revision marker `6151`).
- Official archive:
  `https://rawtherapee.com/shared/HaldCLUT.zip`.
- Archive version in its README: `2015-09-20`.
- HTTP HEAD on 2026-07-28:
  - `Content-Length: 421602289`;
  - `Content-Type: application/zip`;
  - `ETag: "192123f1-52031888d4800"`;
  - `Last-Modified: Sun, 20 Sep 2015 18:00:00 GMT`.
- Independent SlackBuilds 15.0 package record:
  - version `20150920`;
  - upstream archive MD5 `4742e362a70c1a1c0fb9042a17d285e1`.
- RawTherapee implementation semantics were read at repository commit
  `123b4d7b52a7f023712281a7320b3fa643d8f03f`, file
  `rtengine/clutstore.cc` blob
  `74f1e5fc067a35bfc08dc74ff031210412186b3a`.

No PNG or TIFF body was requested or decoded in AJ0A. Bounded HTTP Range reads
were limited to the ZIP tail, central directory, local README header and
compressed README body.

## Remote ZIP inventory

The remote EOCD and central directory are internally consistent:

- central-directory offset: `421565794`;
- central-directory bytes: `36473`;
- central-directory SHA-256:
  `dfaa92b708442c63bebd6c21151ec052c3d1d9588610cc694120e944f50719a8`;
- 311 entries: 296 files and 15 directories;
- 294 PNG files, one TIFF identity and one README;
- 227 files below `HaldCLUT/Color/`;
- 66 files below `HaldCLUT/Black-and-White/`;
- 33 of the colour files are below `CreativePack-1`;
- one root `Negative.png`;
- 194 non-Creative colour entries form the frozen AJ0 primary universe.

The README is 2,357 bytes, CRC32 `410697ab`, and SHA-256
`363cffb318e25dc4f880e070915aceddfe842e1fdb0a68216bef04719185c69c`.
It states `CC BY-SA 4.0` and credits Pat David, Pavlov Dmitry and Michael
Ezra. It also says that stock names are informational approximations and that
the authors/publisher are not affiliated with the trademark owners.

## Rights boundary

CC BY-SA 4.0 permits copying, redistribution and adaptation subject to
attribution, licence linking, change indication and ShareAlike. That is enough
for local bounded research acquisition.

It does not authorize an undocumented asset copy inside K-MCFM. This project
still has no owner-selected root licence. Therefore:

- the archive and decoded assets stay under ignored `data/external/`;
- no CLUT image is committed, packaged, released or embedded;
- any future distribution needs a separate asset manifest, attribution,
  licence text, change record and project-licence decision;
- trademarked filenames remain informational and must not imply endorsement.

## Frozen colour semantics

RawPedia describes HaldCLUT as a global colour transform. Local contrast,
denoising, sharpening, geometry and spatial editing cannot be represented.
The collection README states sRGB, 8-bit PNG unless a filename says otherwise.

The pinned RawTherapee implementation:

- accepts a square image whose side is an integer cube;
- derives a 3D cube side as the square of that Hald level;
- defaults the working CLUT profile to sRGB unless the filename explicitly
  names another supported profile;
- indexes red fastest, then green, then blue;
- performs trilinear interpolation;
- linearly blends full CLUT output with input for optional strength.

AJ0 will implement only a clean-room reader/evaluator of these public file
semantics. It will not copy RawTherapee GPL source. Formal AJ0 structural
analysis uses full strength and encoded sRGB. Any future production-strength
control would require a new contract.

## Epistemic and product boundary

The collection cannot establish:

- a measured or calibrated response for any named stock;
- a true Push/Pull, exposure, EI, illuminant, process or scanner condition;
- a digital-to-film operator;
- a stock or latent-mode label;
- a training target, pseudo-teacher or authenticity claim.

Filename suffixes such as `+`, `-`, `faded` or `expired` are weak descriptive
metadata only. AJ0 must group exact duplicates and near-collinear
strength-like directions before counting diversity. The 53/55/56 project
anchor family remains the mandatory negative-control principle: different
strengths of one direction are not multiple modes.

## Allowed next leaf

`U5.R2AJ0B` may:

1. download exactly the one 421,602,289-byte archive to ignored local data;
2. require the published MD5 before atomic promotion;
3. compute and freeze SHA-256;
4. verify ZIP paths, CRCs, exact inventory, README and all image decodes twice;
5. report image dimensions, Hald levels, bit depths and profile metadata;
6. open a separately committed structural contract only if both audits are
   byte-identical and every integrity gate passes.

AJ0B may not render a photograph, inspect a preset aesthetically, modify the
candidate universe, extract assets into tracked paths or infer stock truth.

## Failure branches

- Header, byte-count or MD5 mismatch: close the source without mirror search.
- ZIP path, CRC, inventory or decode failure: close without removing entries.
- README/licence mismatch: quarantine and close.
- Integrity pass: freeze archive SHA and open only AJ0C synthetic structural
  analysis.

G'MIC remains an optional later conformance executor, not an additional
source dependency for AJ0B. NamedCurves-style clean-room analytic colour
partitions remain a distinct later representation hypothesis and are not
mixed into this source leaf.

