# U5.R2BK24 — FiveK semantic pseudo-pair flow results

BK24 closes with a repeat-exact `K=1/identity` outcome. Both complete GPU runs
have SHA-256 `b819057d7bc39562e69a4ea7f5a0924b7b990f49c19e5e87d35c439f8ef7a1bb`.

The fixed colour-blind DINOv2 matcher excludes every same-photo target, uses
all 48 target photographs, and has only 3.75% maximum target share. Its
semantic operator is structurally valid and introduces zero new code-boundary
pixels. It also beats the fixed target-colour-permuted control by 12.90
percentage points at the median, so the retrieval is not merely random.

That signal is not enough. On 16 target-hidden photographs, semantic
pseudo-pairing has `-89.57%` median improvement, `-823.23%` worst improvement
and only `1/16` improving rows. The shuffled control is worse at `-102.46%`
median, but both lose decisively to identity.

The fixed configuration therefore closes without tuning DINO, top-k,
Sinkhorn, flow capacity, split or gates. Cross-photo semantic similarity is a
useful retrieval cue but not reliable supervision for the colour response
that should be applied to another photograph. The result is a digital-retouch
method control, not film, stock, calibration, product or unpaired-operator
identification evidence.
