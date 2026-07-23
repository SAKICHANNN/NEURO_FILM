# U5.R2H1 safe-bank content-routing results

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2H1`  
Decision: **retain hard 1-NN for an A0 visual diagnostic only**

## Integrity and boundary

- config SHA-256:
  `f9db09f8986c1e0e053191b261991bc6da29d57ad0696c9fba481f15a9483f04`
- software commit:
  `ea5db6640a61ebeece23eb7177c5905f7e28562a`
- report SHA-256:
  `ae4e160aa4e622591d4a709a4e0e22d354e5f3bebbc6d47237bf776660740464`
- descriptor dimension: 71
- all source, candidate-output, manifest and report hashes pass
- stderr is empty
- all 772 repository tests passed before the formal audit

Every row is previously exposed A0 development evidence. The labels are
autonomous metrics, not owner or population preference.

## Oracle evidence

| View | Assigned | Anchor / density | Oracle median gain |
|---|---:|---:|---:|
| non-basic residual | 37/41 | 14 / 23 | 14.20% |
| style Delta E76 | 38/41 | 16 / 22 | 16.13% |
| style + non-basic | 37/41 | 13 / 24 | 18.76% |

The formal composite gain is larger than the earlier nine-gold disclosure
because H1 evaluates all assigned A0 rows. Both operator directions retain
substantial support; this is not a one-class collapse.

## Router result

Primary composite view:

| Router | Balanced accuracy | Permutation p | Mean regret closed |
|---|---:|---:|---:|
| majority | 50.00% | 1.000 | 0.00% |
| hard 1-NN | **70.67%** | **0.010** | **45.61%** |
| weighted 3-NN | 62.98% | 0.063 | 29.34% |
| L2 logistic | 68.27% | 0.049 | 39.26% |

Hard 1-NN is retained because it is simpler and stronger than logistic on all
three primary gates.

Component views qualify the result:

- style hard 1-NN: balanced accuracy 65.91%, p=0.040, regret closure 43.16%;
- non-basic hard 1-NN: balanced accuracy 45.50%, p=0.628, despite descriptive
  regret closure 29.23%.

Thus content resemblance predicts the **style-oriented** winner in this small
bank, but does not reliably predict the operator with the largest
basic-adjustment residual. This is a real limitation, not a reason to average
the two views or tune features.

## Interpretation

The user's hard case-retrieval intuition has credible local evidence: a source
image can retrieve a similar A0 case and select one of two already safe,
clearly different explicit looks better than a constant global choice under
the disclosed style-composite metric. The successful model is 1-NN, not a
neural network, and its output is a hard existing operator.

This does not solve stock learning or real-film operator identification. It
does not establish human preference, generalization, or a production router.
It only justifies a separately frozen visual/full-resolution diagnostic of the
exact hard 1-NN policy. No larger model, embedding, soft blend, or feature
search opens.

Ultimate remains active.
