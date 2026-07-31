# U5.R2BK24 — FiveK Semantic Pseudo-Pair Flow

BK23 showed that pooled unpaired appearance fitting can improve by 92.89% while
failing catastrophically on individual photographs. BK24 tests one distinct
mechanism: frozen DINOv2-S/14 features retrieve semantically similar patches
across *different* photographs, then the existing cube-safe flow fits those
pseudo-paired RGB samples.

The experiment uses only the 64-row FiveK development freeze: 48 rows build
pseudo-pairs and 16 rows are target-hidden development evaluation. DINO sees
replicated luma, never colour; matching a photograph to its own styled target
is forbidden. A fixed control permutes target colours after semantic transport
while preserving the exact colour multiset. The model is never fine-tuned and
does not render RGB; only the existing explicit flow does.

This adapts the pseudo-pairing idea from
[Cho et al. 2026](https://arxiv.org/abs/2605.07495), whose
[MIT-licensed code](https://github.com/nuniniyujin/Unpaired-ISP) uses DINOv2
and an FGW-inspired Sinkhorn matcher. DINOv2 code and standard weights are
[Apache-2.0](https://github.com/facebookresearch/dinov2).

Pass requires useful target diversity, a held-out median gain of at least 10%,
no worse than -5% on any photograph, at least 75% improving photographs, a
five-point advantage over the permuted control, and the existing structural
and boundary safety gates. Failure returns identity and closes this exact
configuration. No threshold or representation tuning is allowed after formal
output. Even a pass opens only a newly frozen disjoint confirmation population.
