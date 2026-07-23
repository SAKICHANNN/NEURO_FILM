# U5.R2D2 canonicalizer and hard-retrieval sensitivity results

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2D2`  
Decision: **canonicalizer-sensitive/unidentified; hard retrieval closes**

## Result

The 72-item independent neutral bank, parent hashes, group splits and exact
oracle all reproduce. Every predicted explicit operator remains numerically
valid. The result fails on information stability, not rendering mechanics.

| Practical canonicalizer / selected policy | Confirmation RGB RMSE | Improvement vs global | Captured style | Worst family degradation | Useful |
|---|---:|---:|---:|---:|---|
| query-as-neutral / ridge | 0.16164 | -134.4% | -0.928 | 280.3% | no |
| nearest raw-Lab / ridge | 0.05706 | 17.3% | 0.309 | 22.3% | no |
| nearest normalized-quantile / ridge | 0.06208 | 10.0% | 0.201 | 15.7% | no |
| Gaussian neutralization / ridge | 0.14368 | -108.3% | -0.802 | 274.8% | no |

The best practical method is promising in aggregate but violates the frozen
per-family safety gate. None of four practical hypotheses passes
independently, versus the required three.

Practical canonicalizers disagree strongly: median pairwise rendered-operator
RGB RMSE is 0.15667 and the maximum is 0.20436, against the frozen 0.03
stability gate.

## What the neutral retrieval did learn

On confirmation:

- raw-Lab neutral retrieval selects the correct synthetic palette family
  95.31% of the time;
- normalized-quantile retrieval selects it 89.58% of the time;
- the palette-oracle lane is 100% by construction;
- raw-Lab concentrates 40.63% of selections on one bank item;
- normalized quantiles concentrate 29.17%.

The palette-oracle ridge reaches RGB RMSE 0.05345 and 32.2% captured style,
slightly ahead of the reproduced exact-raw oracle at 0.05428 and 31.8%.
However, both still lose to the unusually strong global baseline on at least
the matrix-only family. They are useful research controls, not complete
policies.

This supports a data/connectivity interpretation: a correct content/palette
cell greatly improves the neutral-control hypothesis, but neither raw
appearance retrieval nor a known coarse palette label uniquely supplies a
stable operator across families.

## Hard case retrieval

Hard Top-1 improves over ridge only for the two canonicalizers that are already
grossly invalid as recovery policies (query-as-neutral and Gaussian). For the
two plausible nearest-neutral methods, Top-1 is 20.7% to 33.3% worse than
ridge and has large family degradation. No independently useful
canonicalizer supports a Top-1 win; sparse Top-3 also fails.

Therefore, this benchmark does not validate “pick the most similar photo and
copy its operator.” The idea remains plausible only after a future dataset
provides a stable, evidence-backed operator signature or same-content neutral
connection. Raw image similarity is not that signature.

## Visual and reproducibility evidence

The non-binding worst-case Hald sheet for nearest-raw-Lab/ridge shows bounded,
smooth global transforms but visible tone/hue direction errors. There is no
geometry rewrite, banding or numerical clipping. This agrees with the
statistical conclusion that the method is safe in form but wrong in inferred
operator.

- two reports are byte-identical at
  `d36204914a91ed8e75f353b376260e56e69a8d93ef661b16b681fddd90f99856`;
- config SHA-256:
  `23b087133c934694f63c15da6c167f6e404daec77cfb0783bc834e079c0b34bd`;
- neutral-bank manifest SHA-256:
  `bae7cb2e866956b8dc6ff989f0ec3930d256d649031dee816f9455537b001713`;
- implementation commit:
  `01a2a077d7f733aaf24d0e320d8723e2788a4544`;
- visual sheet SHA-256:
  `bd89c42255bed1eb3d9ef0e763cfa13ba7b1dab32ff1fad4a7d6dcac3c1228bc`;
- 17 focused and 743 complete CPU tests pass.

## Branch

Close the current statistics-to-LUT, imperfect-canonicalizer and raw hard-case
retrieval path. Do not rescue it with a larger neural network.

The next autonomous algorithm leaf should move to an explicit
density/sensitometry-domain film-inspired operator family. That family can
provide strong, stable, bounded colour hypotheses without pretending to infer
a real operator from one unpaired scan. It must remain a Look Approximation
until independently eligible stock evidence exists, and it must challenge
safe-rich, margin-4 anchor56 and the isolated RF2.C0 spectral control without
using any as teacher truth.

Real-film operator fitting, stock learning, LSM and production integration
remain closed. The Ultimate Goal remains ACTIVE.
