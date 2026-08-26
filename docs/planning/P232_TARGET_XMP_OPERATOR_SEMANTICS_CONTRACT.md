# P232 Target-XMP Operator Semantics Contract

Date: 2026-08-26

Status: frozen before operator compilation or image-pixel access

Parent: ultimate reference matching, paired/capture-time metadata lane

## Question

Can the exact public MMArt-PPR10K XMP sample already retained by the project
define a deterministic, source-independent explicit colour operator, or does
it only describe work performed by an Adobe renderer?

This is a target-side metadata question. It is not another output-only image
encoder, a Lightroom-compatible renderer, or a request to acquire the gated
INRetouch pixels.

## Competing hypotheses

1. **Explicit operator:** multiple independent, locally authorized recipes
   contain equation-complete global colour fields, their exact processing
   order, and every referenced profile byte.
2. **Renderer recipe:** fields identify edit intent, but exact rendering still
   depends on Adobe process-version, profile, ordering, or private semantics.
3. **Descriptor only:** the fields may later condition an abstention policy or
   paired renderer fit, but do not themselves define an executable operator.

The audit reports all three possibilities. It does not assume that a numeric
XMP value is an implementation specification.

## Frozen source and rights boundary

- The only retained recipe is
  `outputs/source_recon/mmart_ppr10k_public_metadata/config.xmp`, 9,648 bytes,
  SHA-256 `78ec5860...07d75`.
- The earlier read-only repository audit listed 4,055 XMP recipes across 1,412
  content IDs, but did not acquire them.
- The derivative card says Apache-2.0 while upstream PPR10K restricts images
  and derived data to non-commercial research. Existing project adjudication
  therefore forbids bulk recipe acquisition, training, fitting and
  redistribution.
- Formal execution is offline and metadata-only: zero image pixels, operator
  compilations, renders, scores or network reads.

## Admission gates

Direct compilation requires all of the following:

- at least eight independent, locally authorized recipes;
- equations and processing order for every active non-identity colour field;
- exact bytes for every referenced camera/look profile;
- no active spatial or geometry edit in a purported global operator;
- no unknown or private renderer dependency;
- exact forward/reverse fresh-process replay.

Any missing gate closes before pixels. Missing or untrusted metadata always
abstains. A curve point list may be observed exactly while still lacking an
identified input domain or place in the renderer pipeline.

## Stop rule and claim ceiling

Failure closes only direct XMP-to-operator compilation. It does not deny that
the metadata contains useful edit intent, nor does it evaluate Adobe output
quality. A PASS would open only a new paired-renderer semantic-parity leaf.

No outcome establishes Lightroom/Camera Raw compatibility, image quality,
arbitrary XMP handling, film/stock calibration, public package/schema/
capability, or product admission.
