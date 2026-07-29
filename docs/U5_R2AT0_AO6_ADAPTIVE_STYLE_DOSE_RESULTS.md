# U5.R2AT0 AO6 adaptive style-dose result

AT0 tested whether a deterministic per-image choice along the already frozen
AO6 strength path could improve the fixed global `t15/c35` policy. It did not
use scene labels, camera identity, semantic embeddings, retrieval, refitting or
pixel blending. For each image it selected the existing AO6 output whose
median residual from the exact B0 image was nearest a development-only target
of 2.5 Delta E76.

Two reports are byte-identical at `6eb288df...85b7`; stable evidence is
`8593fce3...6cdf`. All automatic gates pass:

- the four strength levels are selected `13/11/16/1` times;
- residual MAD falls from `.42877` to `.15986` Delta E76, a `.37284` ratio;
- median and p90 target error are `.19210/.40967`;
- gold style/non-basic/residual are `13.0142/9.5171/2.6497`;
- gold and stress new hard clipping remain zero;
- every selected output hash is inherited exactly from the frozen AO6
  manifest.

The visual gate fails decisively. Three blinded column permutations of the
same nine AO6 gold images all reveal the same ordering:

1. fixed B0;
2. fixed AO6 `t15/c35`;
3. adaptive dose.

The adaptive policy therefore wins `0/3` against fixed AO6, below the frozen
`2/3` gate. Full-resolution review confirms no severe artifact and no ID11 red
speckle/posterization, so the failure is aesthetic rather than technical.

Close this exact target policy without retuning on the same images. The result
is useful negative evidence: equalizing an explicit residual's numeric
Delta-E dose can make outputs statistically more uniform while making the
look less appealing. Retain fixed AO6 only at its prior development Look
Approximation ceiling and seek a genuinely different explicit colour
direction or new controlled evidence.
