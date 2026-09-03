# U7.22B — 100MP product render resource contract

Status: frozen before the first 100MP product render under this node.

## Question

Can the unchanged public Look Approximation CLI render the existing exact
10000x10000 sRGB-identified U6.P8CH artifact twice in fresh processes, with
byte-identical image and normalized strict-recipe semantics, while staying
inside an ordinary 64GB Windows workstation resource envelope?

## Frozen candidate

- Input: `outputs/eval/u6_p8ch_100mp_native_thomas_png_v1/performance-1.png`,
  exact 545168196-byte / SHA-256 `17fcc4f...b5f9` U6.P8CH artifact.
- Product route: unchanged `scripts/render_film.py --product-look ektar_100
  --look-amount 0.65 --output-bit-depth 8 --write-recipe --tile-size 512
  --tile-workers 8`.
- No effect, Look, renderer, source, tiling, worker, encoding or threshold
  change is allowed after execution begins.
- Forward and reverse labels change only the formal orchestration order; each
  accepted report must contain two fresh worker processes.

## Frozen gates

1. Source bytes, dimensions and embedded sRGB ICC identity are exact.
2. Both workers exit successfully, publish one image plus one strict recipe,
   and leave no owned process or scratch residue.
3. Image bytes and decoded RGB8 samples are exact across workers.
4. Recipe semantics are exact after normalizing only absolute input/output
   paths; each recipe remains `film-inspired / look-approximation`, selects
   Ektar 100 at amount 0.65 and records the committed worker source revision.
5. Each worker peak process-tree RSS is at most 16 GiB and wall time is at most
   600 seconds. The second/first wall ratio is at most 1.25.
6. Input bytes remain immutable; output and recipe publication are create-only.
7. Forward/reverse accepted reports must be byte-identical.

Any failed gate closes this exact 100MP product path without parameter,
algorithm, source, scheduler or threshold rescue. A failure may only motivate
a separately frozen materially different streaming mechanism.

## Claim ceiling

Pass means one exact 100MP sRGB source completes deterministically through the
private Windows/Python Ektar 100 Look Approximation product path under the
frozen local resource envelope. It does not establish arbitrary 100MP support,
image quality, calibrated Ektar response, physical film reproduction,
cross-platform parity, public release or redistribution rights.
