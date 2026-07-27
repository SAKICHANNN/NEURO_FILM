# U5.R2AE1 — spectral_film_lut structural-bank contract

## Question

Does the source-bound subset of the pinned external simulator produce a
replay-stable, structurally safe and non-basic bank of explicit colour
operators on a fixed synthetic RGB cube?

This is deliberately narrower than asking whether its outputs resemble real
film. AE1 has no real-film targets, no digital/film pairs and no authority to
validate the embedded named profiles.

## Frozen bank

The eight chains are frozen before formal metrics:

- Ektar 100 and Portra 400 through the same Endura Premier output;
- VISION3 50D, 250D, 200T and 500T through the same 2383 output;
- Ektachrome 100D and Kodachrome 64 as direct reversal paths.

Every material has an identifiable historical datasheet Git blob in the
external repository history. That association is a lineage filter only, not a
licence or accuracy claim. UltraMax 400, Velvia 50 and all other executable
profiles are excluded because AE0 did not find equivalent historical document
association for them. Candidate membership cannot change after metrics.

## Fixed execution

- external revision: `02ecafd78c4a97bd0708d69a9f5ec39caf492d2e`;
- CPython 3.14.4 and exact package versions in the config;
- encoded sRGB 17-cube input;
- full pipeline, zero exposure/push-pull/printer-light adjustment;
- 6500 K exposure and projection;
- Rec.709 / sRGB encoded output;
- external auto white clipping remains on because it is a published default
  under test, not an optimization;
- two independent complete process runs;
- raw float32 arrays and strict manifests are the only formal inputs to the
  project evaluator.

The project-owned runner may call the public headless API but must not copy or
modify external source or profile arrays.

## Controls and gates

Each chain must pass finite/range, interior clipping, neutral-luma
monotonicity, local Jacobian folding and derivative-amplification gates.
Identity distance and residual after a bounded joint EV/WB/contrast/saturation
fit measure whether the output is more than a simple basic adjustment.

Pairwise diversity is evaluated only within a shared output-chain family:
photo negatives share Endura, cine negatives share 2383, and reversals are
direct. This prevents print/direct-chain differences from masquerading as
stock separation. At least two of the three families must contain a pair whose
residual after the same bounded basic control reaches the frozen floor.

Two negative controls are mandatory:

1. an exact duplicate invocation of Ektar+Endura must be byte-identical; and
2. a synthetic 0.75 identity-to-Ektar strength path must be recovered as one
   direction, with effectively zero residual and at least
   `0.999999999999` explained energy.

Thus the evaluator must not treat duplicate objects or one operator direction
at different strengths as distinct modes.

## Branches

- Runtime, source hash or exact replay failure closes without version, thread
  or precision rescue.
- Range, folding, monotonicity or derivative failure closes without clamping,
  smoothing or changing gates.
- Basic-only behaviour or insufficient within-family diversity closes without
  adding profiles or model capacity.
- A full pass retains only an external structural comparison bank. Any later
  real-image or visual leaf needs a new frozen contract.

AE1 never opens current-pixel fitting, training, a teacher bank, stock
authenticity, calibration, LSM or product integration.
