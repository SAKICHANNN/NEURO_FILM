# U5.R2F1 CanonCGT reference-condition results

Date: 2026-07-23

Node: `ULT > U5 > U5.R2 > U5.R2F1`

Decision: **reference-sensitive and stylized, but all raw E2E policies fail
the frozen clipping/range safety screen**

## Result

The fixed public CanonCGT E2E checkpoint was evaluated without training on
the exact nine-reference by nine-gold bank. Both complete passes contain 81
RGB outputs and 162 predicted LUT arrays. Their manifests are byte-identical.
Explicit application of the predicted canonicalizer and restyler 17-cube
LUTs reproduces every model output exactly.

The reference bank is not averaging references into one bland result:
matched-input pairwise output Delta E76 has median `8.5322`, range
`0.4599--25.5040`, above the frozen `2.0` sensitivity floor.

## Automatic frontier

| Reference | Style Delta E76 | Non-basic residual | Worst new clipping | Worst raw-final OOR | Result |
|---|---:|---:|---:|---:|---|
| a0_ref_01 | 11.6392 | 6.1098 | 8.4326% | 12.1329% | safety fail |
| a0_ref_02 | 11.0478 | 6.0475 | 1.5816% | 0.4626% | clipping fail |
| a0_ref_03 | 6.1068 | 3.6110 | 8.9329% | 8.8520% | style/basic/safety fail |
| a0_ref_04 | 11.4658 | 5.8738 | 8.1707% | 12.2397% | safety fail |
| a0_ref_05 | 5.1877 | 3.5944 | 8.2410% | 6.8714% | style/basic/safety fail |
| a0_ref_06 | 11.5600 | 5.9652 | 8.5503% | 12.1844% | safety fail |
| a0_ref_07 | 5.3827 | 3.3387 | 8.7454% | 8.5615% | style/basic/safety fail |
| a0_ref_08 | 8.0777 | 5.1596 | 0.5702% | 0.5060% | near-boundary safety fail |
| a0_ref_09 | 10.6562 | 5.9124 | 8.5070% | 10.2074% | safety fail |

Six of nine references pass both the style and non-basic floors. None passes
all four automatic gates. The protocol therefore produces no shortlist and
forbids blind or full-resolution candidate promotion.

The closest policy, `a0_ref_08`, misses both safety limits narrowly because
ID11 reaches 0.5702% new endpoint clipping and 0.5060% raw-final
out-of-range values. `a0_ref_02` stays within the raw range limit but reaches
1.5816% new clipping on sample 29.

## Unbounded-LUT evidence

Across all 81 pairs:

- canonicalizer LUT range: `-0.41491` to `1.38847`;
- restyler LUT range: `-0.06239` to `1.15702`;
- median canonicalizer LUT-node out-of-range fraction: `4.9054%`;
- median restyler LUT-node out-of-range fraction: `8.0060%`;
- maximum raw-final out-of-range fraction: `12.2397%`;
- maximum explicit replay error: exactly `0`.

The failures are therefore not neural spatial glitches. They are a concrete
parameter-safety failure in otherwise explicit, reference-sensitive colour
operators.

## Reproducibility

- frozen configuration SHA-256:
  `58d2a2b349763f7b29d08ded37e028f394585250f1ded9452a5beb41a4693d8b`;
- both formal manifests:
  `77c25ad0d75c9ec5d0eca66a60a133b6e6a0d48f719e6121d3540fc53bfc3f7c`;
- automatic report:
  `f5b510282cef4e8ce9ac780d5389c6c4a8eb1d26e3611e7eae13fa1b4b8002a8`;
- evaluation software commit:
  `3a975bb261022c75a73c207c0d421cd2d1ebbcc7`;
- 760 complete project tests pass.

No paid compute, training, fine-tuning or new pixel acquisition was used.

## Interpretation and boundary

F1 rejects the raw public E2E policies under the unchanged severe-first
screen. It does not show that reference-conditioned ML is hopeless:
conditioning is materially different across references, and several policies
are strongly non-basic. It shows that the public model's unconstrained LUT
parameterization is unsafe for this product contract.

The references remain A0 runtime conditions, not pairs, targets, teacher
truth or named-stock evidence. No stock response, calibration, latent mode,
independent-human preference or production claim opens.

## Branch

Open U5.R2F2 as a new preregistered development leaf: reuse the immutable F1
LUT evidence without model inference and compare a very small bank of
data-independent explicit safety projections. Candidate operations may clip
LUT nodes to gamut, contract the projected residual toward the identity LUT
and audit finite-difference monotonicity/Jacobian signs. They must retain the
unchanged style/non-basic/clipping gates and may not select a reference or
strength after observing F2 results.

If bounded projection removes style or remains unsafe, close CanonCGT as an
external control. Do not increase model capacity or train on current pixels.

The Ultimate Goal remains ACTIVE.
