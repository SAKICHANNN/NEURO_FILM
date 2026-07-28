# U5.R2AJ0B — RawTherapee HaldCLUT bounded acquisition contract

## Parent and question

Parent: `U5.R2AJ0A`.

Can the exact licensed RawTherapee Film Simulation Collection archive be
acquired, hash-bound and decoded reproducibly without changing the frozen
194-entry primary colour-operator universe?

This is an integrity leaf. It does not decide whether any CLUT is structurally
safe, visually useful or film-authentic.

## Fixed acquisition

- URL: `https://rawtherapee.com/shared/HaldCLUT.zip`;
- expected bytes: exactly `421602289`;
- published MD5: `4742e362a70c1a1c0fb9042a17d285e1`;
- maximum retained archive payload: one exact file;
- destination:
  `data/external/rawtherapee_haldclut/20150920/HaldCLUT.zip`;
- partial download:
  `data/external/rawtherapee_haldclut/20150920/HaldCLUT.zip.part`;
- assets remain ignored and are not extracted into tracked paths;
- no mirror fallback, new asset, pixel replacement or open-ended retry.

The downloader may resume only when the server returns a valid matching byte
range. It must enforce the byte ceiling while streaming and atomically rename
only after exact size and MD5 pass. SHA-256 is computed and recorded as a new
observed fact, never guessed before download.

## ZIP and decode audit

Each of two independent complete audits must:

1. recheck archive bytes, MD5 and SHA-256;
2. reject encrypted, absolute, parent-traversal, duplicate-normalized or
   symlink-like entries;
3. verify every ZIP CRC;
4. reproduce the exact 311-entry / 296-file inventory and extension counts;
5. reproduce the remote central-directory and README hashes;
6. decode all 294 PNGs and the identity TIFF from in-memory ZIP members;
7. require RGB image payloads, square dimensions and a valid Hald side whose
   integer cube root is greater than one;
8. record dimensions, mode, bit depth, ICC presence, Hald level and derived
   cube side for every image;
9. reproduce the exact 194 non-Creative colour primary members in lexical
   path order;
10. write only strict JSON manifest/report evidence under ignored outputs.

The second run may reuse the verified local archive but must reopen, rehash
and re-decode it in a new process.

## Automatic pass

AJ0B passes only if:

- both runs complete without exception;
- every corresponding manifest/report byte is identical;
- archive size and MD5 equal the frozen values;
- both runs report one identical SHA-256;
- all paths, CRCs, counts and decodes pass;
- README licence/hash and primary membership match AJ0A;
- no image has been rendered through any CLUT.

## Branches

- Network interruption: retain only the bounded `.part` and resume later.
- Range mismatch or size overflow: delete no user data; quarantine the part
  inside the AJ0 data directory and close the attempt.
- Size/MD5 mismatch: close without mirror search.
- ZIP/path/CRC/decode/inventory mismatch: close without dropping entries.
- Repeat mismatch: close without thread, decoder or ordering rescue.
- Full pass: commit the exact SHA/inventory decision and open only a separately
  frozen AJ0C synthetic structural audit.

## Forbidden

No photograph render, contact sheet, aesthetic preview, candidate selection,
stock claim, training, fitting, LSM, tracked asset copy, production import,
release, deployment or licence decision is allowed in AJ0B.

