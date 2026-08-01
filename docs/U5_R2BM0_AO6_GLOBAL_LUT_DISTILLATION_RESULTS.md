# U5.R2BM0 fixed global AO6 LUT distillation

BM0 asks whether AO6 t15/c35 can be reduced to one context-free monotone
channel shaper plus a bounded tetrahedral 17-cube LUT.  Seventeen camera makes
are split by source ID into three held groups; every image contributes the
same number of aligned samples.  The target is the already frozen AO6 Look
Approximation, not film or stock truth.

The two complete 21-file runs are byte-identical (report SHA-256
`77a6ff57...ddb0`, stable ID `cafeee05...e93ac`).  The median held-camera error
falls 55.97% versus identity and median style retention is 82.77%, so a real
global component exists.  However, the bounded 3D residual has zero admissible
strength in every fold.  The remaining shaper-only approximation does not
improve over the shaper baseline, has median target Delta E76 8.52, reaches a
1.430x worst-camera error ratio, and varies from under- to over-application.
The frozen automatic gate fails; no visual review is allowed.

This is direct evidence for the user's averaging concern: one global mapping
captures central tendency but does not reproduce AO6's image-specific action
reliably.  It does not prove that every fixed LUT or every film style must
fail.  BM0 closes same-population grid enlargement, projection and threshold
rescue.  AO6 remains the global fallback; the next useful experiment must
separate operator-signature selection from content similarity and use hard
case/medoid routing rather than dense style averaging.
