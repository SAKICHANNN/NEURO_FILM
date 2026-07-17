# U5.R1A FilmStyleSafe ontology and schema results

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1A`

Decision: **complete local evaluation-contract tooling; open R1B design only**

## Delivered

U5.R1A now has a strict, versioned contract and pure validation module for
future strong-colour evaluation:

- `configs/filmstylesafe_r1a_contract_v1.json` freezes ontology, rater rule,
  A0/A1/B0--B4 split semantics, current-evidence assignment and planning
  values;
- `configs/schemas/filmstylesafe_annotation_v1.schema.json` defines strict
  severity and deployed-output style records;
- `src/eval/filmstylesafe.py` validates records, aggregates the frozen 3+3
  scene label, audits split leakage and computes a non-binding zero-event
  planning worksheet;
- `tests/test_filmstylesafe.py` covers claim escalation, missingness, leakage,
  traceability and power semantics.

The implementation reads metadata only. It does not inspect images, infer a
label, train SCIS, modify rendering or contact participants.

## Epistemic safeguards

The validator keeps three concepts separate:

1. local chromatic severe artifacts such as speckle, posterization, gamut hard
   boundaries, seams/halos and perturbation instability;
2. global exposure/tone/saturation/palette choices and style mismatch, which
   are quality diagnostics rather than automatic glitches;
3. face/text/object/texture corruption, which is a separate ontology relevant
   only to generative RGB controls.

A severe record needs a known category plus region evidence. Unknown fields or
labels fail closed. Autonomous VLM evidence cannot be marked as primary human
evidence. A film-stock look ID cannot be used when the reference-board film
origin is untraceable.

## Scene aggregation

Three initial blind human ratings all in `safe/minor` yield a non-severe scene.
Any severe, uncertain or missing initial result requires three senior blind
ratings. A senior majority decides; incomplete or unresolved adjudication is
conservatively severe. Crops, candidates, perturbations and VLM votes never
inflate the independent scene count.

## Leakage and current evidence

The split validator rejects cross-split parent-scene, source, creator, camera,
roll, exact-hash or perceptual-hash reuse. A1 must also hold out complete
transform and failure families from A0. The current U4 provisional set, owner
anchors, RF2.C0 outputs and ID11 red-speckle evidence are enforced as A0-only;
they cannot later become hidden confirmation data.

## Power worksheet

For the non-binding planning example of a 1% one-sided risk target at 95%
confidence, zero observed events require 299 independent accepted scenes. At
50% expected coverage this is 598 total independent scenes before clustering,
label error, failures, tie-model power or replication. The API always reports
`binding_sample_size=null`; a binding count remains unknown until those pilots
exist.

## Verification and branch

- 7/7 focused tests pass;
- 363/363 full CPU tests pass in 27.18 seconds;
- compile, JSON syntax and diff checks pass.

U5.R1A opens only U5.R1B design for a leakage-safe synthetic and real failure
suite. It does not authorize image acquisition, participant recruitment, paid
annotation, metric/model training, hidden-test creation, adaptive routing,
public release or any human-population/risk claim.
