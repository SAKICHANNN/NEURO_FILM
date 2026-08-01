# U5.R2BO7 safe smooth-LUT baseline result

BO7 is closed. Two reports are byte-identical at SHA-256
`854771695bccfdcf1fe09c92e458693d8b57a72e383b8b242fa29e672557aee9`.
The degree-2 Bernstein residual is materially active (minimum safe strength
`0.28`), stays inside the cube with zero new boundary pixels, and retains a
minimum combined Jacobian determinant of `0.001517`.

Held-scene error improves on 33/40 scenes and all three family medians are
positive. The effect is nevertheless too small: mean and median improvements
over the BO3 control are only `2.09%` and `1.65%` versus frozen `5%` gates;
P95 ratio `0.9532` narrowly misses `0.95`. Worst error improves to `0.9595x`.

This is useful fixed-global LUT baseline evidence, not an algorithm survivor.
Degree, ridge, strength search, and confirmation access are closed. A next
candidate must change the information structure rather than merely add LUT
capacity.
