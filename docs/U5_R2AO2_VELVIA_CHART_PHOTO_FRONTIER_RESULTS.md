# U5.R2AO2 real-photo frontier

Two 164-output manifests are byte-identical at `d5477f56...2b37`. No frozen
automatic survivor exists:

| Strength | Gold style | Gold non-basic | Worst new gold clipping |
|---:|---:|---:|---:|
| 0.50 | 7.6480 | 3.0946 | 0% |
| 0.60 | 9.5223 | 3.8305 | 1.3362% |
| 0.70 | 11.8201 | 4.8367 | 1.3362% |
| 0.75 | 13.2390 | 5.3537 | 1.3362% |

The fixed display-proxy operator is visibly strong, but one blend strength
cannot satisfy both the non-basic style and clipping gates. No visual review
opens and no threshold changes. AO3 may test a separately frozen tone/chroma
factorization with deterministic boundary protection.
