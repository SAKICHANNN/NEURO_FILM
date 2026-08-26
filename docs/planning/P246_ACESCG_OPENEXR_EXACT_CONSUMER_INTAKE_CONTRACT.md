# P246 exact ACEScg OpenEXR consumer-intake contract

## Question

Can this repository independently consume the exact producer R1DO writer from
its frozen Git object, install the exact pinned OpenEXR wheel offline, and
reproduce the frozen synthetic ACEScg OpenEXR bytes and semantics without
copying the writer implementation into this repository?

## Frozen identities

- Producer repository is supplied explicitly to the audit runner; no physical
  drive path is stored in the contract or implementation.
- R1DO writer implementation commit:
  `e4023509f1a1ca21606468312bf710972d9e1059`.
- Writer path: `src/zhuise/acescg_openexr.py`.
- Writer Git blob: `398f4de2cfb04a887bb13632df324fea1624f5bc`.
- Writer SHA-256:
  `3707b55b2cb7dd7fb102b4237e8ee4317035e917bd71441eb06dd9cc0a2757c3`.
- R1DO evidence commit:
  `e1a22951e6aae752b73ae9f407e6b0d7a223597c`.
- R1DP evidence commit:
  `3ab2a6fcad8457dd1b53cf841d657fdd3948d071`.
- Exact Windows CPython 3.12 wheel:
  `data/vendor/openexr-3.4.15/openexr-3.4.15-cp312-cp312-win_amd64.whl`,
  740,197 bytes, SHA-256
  `7aa145813acfe10a83d6e89340349379828190b321608e69f2aee426e10c890b`.

## Frozen execution

1. Read the writer and both evidence files with `git show` from their exact
   commits. Do not read the producer worktree versions of those files.
2. Verify all frozen Git blobs, byte counts, SHA-256 identities, evidence
   statuses, writer ID, and OpenEXR version before execution.
3. Create a fresh system-temporary target per process. Install the exact wheel
   into that target with pip `--no-index --no-deps`; network use is forbidden.
4. Compile the exact writer bytes as an ephemeral module. Do not copy or stage
   the writer source in this repository.
5. Materialize only the frozen R1DO 7x9 synthetic float32 lattice. No producer
   real-photo row, project image, target, reference, RAW, HDR media, or pixel
   dataset may be read.
6. Inspect the result through a fresh pinned `OpenEXR.File` read, then remove
   the complete temporary target. The only retained output is the canonical
   JSON audit report.
7. Run two fresh Python processes in forward/reverse control order. Reports and
   OpenEXR bytes must be exact.

## Frozen gates

- exact producer source commit/path/blob/SHA and R1DO/R1DP evidence bindings;
- exact wheel path/size/SHA, offline install, Python 3.12 and OpenEXR 3.4.15;
- writer ID `acescg-ap1-d60-float32-zip-openexr-master-v1`;
- exact synthetic input SHA
  `3b14e75aa87a09794e0c4cb6339a4fc09862657d9aea190a9e3311c155bdb727`;
- exact 696-byte OpenEXR SHA
  `14a122a7d9f674cc5d6fae50a6a8a5e3e62e33217d38bed48fdf0df2194e48ce`;
- decoded float32 pixels bit exact with maximum error zero;
- scanline, ZIP, coalesced RGB float32, exact AP1/D60 chromaticities and
  adopted neutral, absent white luminance, preserved negative and above-one
  values;
- input immutability, invalid-input pre-publication rejection, injected
  publication failure atomicity, and zero temporary residue;
- zero network and zero external/project pixel reads.

Any mismatch closes this exact intake. Do not fetch another wheel, use the
producer worktree source, relax equality, copy the writer into this repository,
substitute another OpenEXR version, or tune encoding/metadata after results.

## Claim ceiling

At most P246 can establish private Windows CPython consumer-side reproducibility
of the exact producer R1DO writer/source/wheel handoff on one synthetic probe.
It does not create a public or versioned neuro_film writer API, dependency,
package, schema, capability, AP0/ST 2065-4 ACES container, large-image result,
image-quality result, arbitrary metadata support, Linux parity claim beyond
the bound producer R1DP evidence, or product admission.
