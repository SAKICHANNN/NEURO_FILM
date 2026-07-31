# U5.R2BK19 Synthetic Hard-Mode Identifiability Results

Two byte-identical runs produced report SHA-256
`7260c3aa065fd0a889c038ad20c47fad30f230d133110cad6d12dbb539dd2d6a`.
All frozen gates pass.

- Two genuinely different explicit residual directions are recovered with
  `1.0` leave-one-group-out accuracy across 12 independent groups per mode.
- A hard two-medoid bank reduces mean directional error by `0.815424`
  absolutely versus K=1.
- One operator direction at strengths `0.6`, `0.8` and `1.0` has minimum
  cross-strength cosine `0.999909`; K=2 improves absolute directional error by
  only `0.00000294`, so it must remain K=1 plus continuous strength.
- Every synthetic output remains inside the RGB cube.

This validates only a known-operator synthetic method prerequisite. It does
not show that any film stock has multiple modes and does not solve unpaired
digital-to-film operator identification. The next legal leaf may test whether
the same conclusion survives multiple unpaired canonicalizers and matched
control pools; disagreement must return `unidentified`.
