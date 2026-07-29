# U6.P1C ColorReference density-basis result

Date: 2026-07-28

Decision: retain family structure, reject the family basis bank, prefer one
shared rank-6 control for the next generic reference-simulator test.

The input is the 100-charge SF2.9B target measurement set converted to bounded
optical density. Four deterministic NMF models were fit only on years ending
0--7 and evaluated on the 13 frozen years-ending-8/9 charges.

| Held-charge comparison | Wins | Median relative improvement |
|---|---:|---:|
| correct family rank-3 vs shared rank-3 | 12/13 | +1.45% |
| correct family rank-3 vs wrong family rank-3 | 13/13 | +5.90% |
| correct family rank-3 vs shared rank-6 | 0/13 | -815.56% |

All four fits converge and two reports are byte-identical at
`f4fe8b46...5b4c85b` (stable evidence `f04990a6...a95635`).

The first two rows confirm descriptive family structure. They do not justify
hard routing: the two family models contain six total basis components, and
one unrouted six-component basis reconstructs every held charge far better.
No family expert bank, mode router or product profile opens.

The next leaf may evaluate the shared rank-6 basis as a generic offline
reference-simulator primitive. Its components remain non-unique target-density
factors, not identified dye layers or a scene-to-film response.
