# U2.1A versioned render profile and recipe contract

Date: 2026-07-17

Node: `ULT > U2.1 > U2.1A`

Status: frozen before implementation

## Question

Can the current deterministic renderer expose a strict, versioned and
replayable profile/recipe contract without changing a single rendered pixel or
overstating the evidence behind existing looks?

This is a product/reproducibility foundation. It does not create a new film
operator, promote a stock, open LSM, or convert current heuristic safe-Lab
looks into calibrated profiles.

## Parent evidence and DoR

- U1.2 already resolves every current render to `film-inspired` and
  `look-approximation`; calibrated Reference is false;
- U1.3B provides one float32 core and deterministic output boundary;
- U1.2A/U1.5A fail closed on invalid ICC and unsupported HDR/gain-map ingress;
- existing `color_rendering_profiles.yaml`, stats and guardrails are runtime
  inputs but do not form one immutable profile identity or replay recipe;
- future stock-global, `Mode A/B/C`, retrieval and fallback policies need the
  same evidence/identity envelope even though none is currently authorized.

## Frozen profile v1 contract

Add a strict JSON schema and dependency-free validator for
`kmcfm.render-profile.v1`. A profile records:

- stable profile ID/version/display name and engine ID;
- optional evidence-backed `film_stock_id` and optional latent mode ID;
- interpretation and explicit claim ceiling;
- data/expert evidence grades and `heuristic/measured/paired/held-out` method
  label;
- immutable asset references with repository-relative path and SHA-256;
- fully resolved colour parameters, effect defaults and output contract;
- fallback profile ID and `calibrated_reference_allowed`.

Unknown keys, unknown enum values, malformed hashes, non-finite numbers and
out-of-range parameters fail closed. Loading data never imports code or follows
arbitrary external paths.

The first tracked profile is a migration of current `safe-rich` only. It must
be labelled heuristic look approximation, have no stock or latent-mode truth,
and retain calibrated Reference as false. The migration must reproduce the
current per-style resolved values exactly; it does not replace the legacy YAML
runtime path in this leaf.

## Frozen recipe v1 contract

Add a strict schema and validator for `kmcfm.render-recipe.v1`. A resolved
recipe records:

- profile ID/version/hash and all referenced asset hashes;
- input content hash, transfer/color state, profile fingerprint, bit depth and
  decode warnings;
- renderer commit, deterministic seed and fully resolved colour/effect values;
- output format, bit depth, transfer, ICC fingerprint and output content hash;
- frozen output claim and evidence ceiling.

Recipe creation is opt-in through `--write-recipe`; the default CLI and metrics
remain compatible. The recipe is written only after a successful output encode
so its output hash is real. Paths may be recorded for diagnostics, but replay
identity relies on hashes and schema IDs rather than host-specific paths.

## Compatibility and safety gates

- current default rendered bytes must remain exact with and without recipe
  writing;
- recipe output must validate and its recorded input/output/config hashes must
  match disk;
- current safe-rich migration must match every style's resolved legacy values;
- mutated schema IDs, unknown keys, invalid grades, hash mismatch, NaN/Inf,
  unbounded strengths/effects and calibrated-claim escalation must reject;
- no new dependency, network access, model, dataset, renderer branch or output
  format;
- targeted tests precede the complete CPU suite.

## DoD, propagation and claim ceiling

Commit/push this contract before code. Implement schemas, validator, migrated
profile and optional recipe writer in a coherent inference/reproducibility
module. Document structural placement, run parity/security/property tests and
the full suite, then update AGENTS, tracker, board, implementation pointer and
agent log in a scoped commit.

At most U2.1A establishes a strict v1 identity/provenance envelope and an exact
migration of the existing heuristic renderer configuration. It makes no stock
response, paired evidence, calibrated Reference, authenticity, preference,
artifact-safety or latent-mode claim.
