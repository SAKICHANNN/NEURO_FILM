# P249 exact ACES2065-1 OpenEXR consumer-intake contract

## Question

Can this repository independently consume the exact producer R1DT AP1-to-AP0
writer from its frozen Git object, install the exact pinned OpenEXR wheel
offline, and reproduce the frozen synthetic ACES2065-1 OpenEXR bytes and
semantics without copying the producer implementation into this repository?

## Frozen identities

- Producer repository is supplied explicitly to the audit runner; no physical
  drive path is stored in the contract, config or implementation.
- R1DT writer implementation commit:
  `a7ec64513c4fa097bc49b9bdbdfd8c3c7c2eb901`.
- Writer path: `src/zhuise/acescg_openexr.py`.
- Writer Git blob: `8c3a1452cc13b29d4ad5f07dcd53cc6c626dd927`.
- Writer bytes/SHA-256: 7,699 /
  `63995dc36cf41708118f32e90687b8bb5c9b2738100b3db557371bbaf086d84d`.
- R1DT evidence commit:
  `9e16b14145415c668ebf7c76ca88ec8b3e3618f9`.
- R1DT evidence bytes/SHA-256: 5,175 /
  `5d037d2f98f90d6f7d872c6a1b009637423ed2994189f94d5e9ec7f0375a68ff`.
- Exact producer Windows CPython 3.12 OpenEXR 3.4.15 wheel: 740,197 bytes,
  SHA-256
  `7aa145813acfe10a83d6e89340349379828190b321608e69f2aee426e10c890b`.

## Frozen execution

1. Read the writer and R1DT evidence with `git show` from their exact commits;
   never read the corresponding producer worktree source or evidence files.
2. Verify commit, blob, byte count, SHA-256, evidence status, writer ID, matrix,
   metadata and wheel identity before execution.
3. Install the exact producer wheel with pip `--no-index --no-deps` into one
   fresh system-temporary target per process; network use is forbidden.
4. Compile the verified writer bytes as an ephemeral module. Do not copy or
   stage producer source in this repository.
5. Materialize only P246's exact 7x9 synthetic float32 ACEScg lattice. No
   producer photo row, project image, target, reference, RAW, HDR media or
   pixel dataset may be read.
6. Call only `write_aces2065_1_openexr`, inspect the result with a fresh pinned
   `OpenEXR.File`, and remove the complete temporary target. Retain only JSON
   reports.
7. Run two fresh controller processes in forward/reverse control order.
   Scientific reports and OpenEXR bytes must be exact.

## Frozen gates

- exact producer writer and evidence Git object, byte and SHA identities;
- exact offline wheel identity, Python 3.12 and OpenEXR 3.4.15;
- writer ID `aces2065-ap0-d60-float32-zip-openexr-container-v1`;
- exact AP1-to-AP0 matrix bound in R1DT;
- exact synthetic input SHA
  `3b14e75aa87a09794e0c4cb6339a4fc09862657d9aea190a9e3311c155bdb727`;
- exact AP0 pixel SHA
  `2b3417e344aa7c26962a109f55d14237a0926d0e03eb46767477b274c4179cb4`;
- exact 1,144-byte OpenEXR SHA
  `1a25181406270142c98f2169897d3a4c468ec50c979fff9077d37f956c888726`;
- decoded float32 pixels bit exact with maximum error zero;
- scanline, ZIP, coalesced RGB float32, exact AP0/D60 chromaticities and
  adopted neutral, integer `acesImageContainerFlag=1`, exact
  `colorInteropID=lin_ap0_scene`, and absent white luminance;
- preserved negative and above-one values, input immutability, invalid-input
  pre-publication rejection and injected publication failure atomicity;
- zero network, external/project pixel reads and temporary residue;
- two fresh scientific reports exact.

Any mismatch closes this exact intake. Do not fetch another wheel, read the
producer worktree source, copy the writer, refit the matrix, rewrite metadata,
relax equality, change the probe or add a new runtime after results.

## Claim ceiling

At most P249 can establish private Windows CPython consumer-side
reproducibility of the exact producer R1DT AP1-to-AP0 writer/source/wheel
handoff on one synthetic probe. It does not create a neuro_film public writer
API or dependency, certify SMPTE ST 2065-4, establish arbitrary metadata or
image support, large-image resources, cross-platform parity, display or image
quality, package/schema/capability, renderer integration or product admission.
