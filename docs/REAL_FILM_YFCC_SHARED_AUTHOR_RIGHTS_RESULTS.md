# SF1.2 YFCC shared-author live-rights results

Date: 2026-07-16

Node: `ULT > RF0.4 > SF1.2`

Decision: `pass_live_rights_feasibility`

## Frozen inputs and execution

- SF1.1 audit SHA-256: `19fd20b30b6724c1033f90ad2c3dafaabfeaf1dad528df12e2de77deaa3723c2`;
- SF1.1 decision SHA-256: `34254d047c56820286aec03dd08b178f9a30afa1dc2e479776ce3bf167549e85`;
- SF1.2 config SHA-256: `2dea85054ade237fddc36cbdd23d96bf2da59766c56a01579d4e225c9fc5b760`;
- software commit: `583dae8`;
- report SHA-256: `a16391c068aa4a5275aa74bbaf4bd4e3462d3020f15490ad789f6c1a413a938a`.

The verifier made 61 sequential, bounded HTML page requests. It requested no
image/download URL, retained no HTML body and recorded every page status,
bounded body hash, final URL and retrieval timestamp.

## Result

Eighteen UID/stock arms find a current CC BY 2.0 page. Eight UIDs pass both the
Ektar100 and Velvia50 arms:

```text
27760134@N03
32676622@N05
36084044@N07
36521968871@N01
37089490@N06
50893125@N03
77971723@N00
79259760@N00
```

The frozen gate requires five usable shared authors, so SF1.2 passes. Eight of
16 metadata-connected authors remain jointly usable at the page-rights level.

## Claim boundary and next branch

This is live-rights feasibility, not pixel quality, content balance, stock
identifiability or operator evidence. The result does not open training,
operator fitting, LSM, `S1/S2`, calibration, authenticity or release claims.

It opens only `SF1.3A`, a separately frozen maximum-38-image pilot covering
the eight passing UIDs and two stocks, still capped at four candidates per
UID/stock and 512 MiB total. Every page licence is reverified during download.
`SF1.3A` must retain at least five authors with a valid pixel in each stock and
then pass hash/decode/duplicate/content/full-resolution visual audit before a
new stock-identifiability contract can be considered.
