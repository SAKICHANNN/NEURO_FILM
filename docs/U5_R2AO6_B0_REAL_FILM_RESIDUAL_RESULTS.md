# U5.R2AO6 B0 real-film residual result

The frozen four-candidate bank applies the fixed AO5 71-pair Velvia
display-proxy operator only as a small, factorized residual after the fixed
vivid B0 look. It performs no per-image adjustment, refit, dense expert blend,
or hard clipping.

Two 164-output runs are byte-identical at `87b2c50a...b6224`; two automatic
reports are byte-identical at `582f3cd7...8c9ab`. Three candidates pass:

| candidate | gold style | gold non-basic | delta from B0 | worst gold/stress clipping |
|---|---:|---:|---:|---:|
| `t15/c35` | 13.2353 | 9.8742 | 2.0468 | 0 / 0 |
| `t20/c50` | 12.7368 | 9.4141 | 2.7969 | 0 / 0 |
| `t25/c65` | 13.1178 | 9.1212 | 3.6700 | 0 / 0 |

The weaker `t10/c25` misses only the preregistered 1.5 Delta E76 residual
floor at 1.4866. No candidate invokes the boundary guard on any gold pixel.

Before reveal, the three blind layouts were ranked `A>C>B`, `B>A>C`, and
`C>A>B`; the frozen pre-reveal record hashes to `95434e98...7fb0`.
After reveal:

- `t15/c35` beats fixed B0 in 2/3 layouts;
- `t20/c50` beats fixed B0 in 1/3;
- `t15/c35` beats `t20/c50` in 3/3.

Full-resolution inspection of both shortlisted candidates on all nine gold
images finds zero confirmed severe artifacts. ID11 has no red speckle or
posterization; both face rows preserve anatomy and texture; smooth walls,
highlights, text and fine structures show no new banding, blocks, seams or
boundary pileup.

Retain `b0_plus_film_t15_c35` as the new B0 development champion and advance
it only to an independently frozen OOD confirmation. This is evidence that a
small explicit real-film display-proxy residual can improve a strong project
look without averaging away its style. It is not a Velvia stock response,
calibration, independent human preference, or product promotion.
