# U5.R2BM3 tone/layout hard retrieval

BM3 fails in two byte-identical runs. The report SHA-256 is
`edc49c27990443a90e84371d77f796d7c1ea5a8b26b5729066f75e5c599428ba`
and the stable evidence ID is
`74f155544f1cd7b1a942e5400153bfe0864dd7d79e5153d30d493a9ef066b326`.

The fixed 74-dimensional descriptor contains only grayscale tone quantiles,
4x4 regional means/standard deviations and gradient-magnitude quantiles. It
uses no fitted weights, chroma, target, operator signature or output
diagnostic. The hard Top-1 and development-only OOD fallback are otherwise
identical to BM2.

Median error regresses 45.72% versus the BM1 global medoid. Only 4/17 sources
improve, only 2/17 selected cases equal the Oracle, and the worst source is
18.18x global. Style retention is 1.061 and new boundary remains zero, so this
again isolates wrong routing rather than insufficient effect or gamut failure.

Together BM2 and BM3 close unlearned source-only nearest-neighbour retrieval
on this population. BM1 remains valuable: it shows that the case bank has a
large post-hoc ceiling beyond continuous strength. The next justified method
is not another fixed embedding; it is a small group-crossfit pairwise
compatibility ranker trained only on development queries and case errors,
whose sole output is one hard case ID. Held queries, final RGB, stock/mode
claims and dense case blending remain forbidden.
