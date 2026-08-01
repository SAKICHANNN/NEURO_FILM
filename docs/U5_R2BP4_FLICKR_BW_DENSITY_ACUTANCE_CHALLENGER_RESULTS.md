# U5.R2BP4 B&W density-acutance challenger result

BP4 is closed. Two reports are byte-identical at SHA-256
`e12eb431...78ef` with stable evidence ID `8f30701f...83b`.

Each held scene excluded its film target while a 27-candidate density-domain
difference-of-Gaussians grid was selected on the other six scenes. All seven
folds independently selected the weakest candidate, `sigma 0.6/1.8 px` at
strength `.1`. It improves held PSD error in 7/7 scenes, but the median gain is
only `.948%`, far below the frozen 10% effect gate. High-pass energy error
improves only `1.61%`.

The result is not density-domain specific. The same parameter triplet applied
as a bounded linear-luma DoG is better by `.440%` at the median, and the
wrong-radius control is within `.102%`. The analytical no-clipping scale falls
to `.0267`, limiting maximum visible luma change to `.00501`. Mean preservation
and zero-boundary safety pass, but all magnitude, mechanism-specificity and
nontrivial-scale gates fail.

No visual review, grid extension, stronger candidate or safety-rule rescue is
allowed. BP2 remains descriptive one-author B&W display-chain evidence; fitted
spatial-response use of this source is closed. Continue only with independent
controlled physical evidence or an unrelated explicit mechanism.
