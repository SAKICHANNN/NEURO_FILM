# P314 — Canon sRAW/mRAW Product-Chain Compatibility Contract

## Question

Can the six exact CC0 Canon sRAW/mRAW files admitted by P313 traverse the
unchanged public local product chain — generic RAW ingress, explicit available
Ektar 100 Look Approximation, deterministic PNG8 publication, strict recipe
creation, and exact recipe replay — without source mutation or persistent
scratch residue?

## Parent and novelty boundary

- P313 proves only generic `WorkingImage` ingress for the six exact files.
- U7.2F/U7.2H/U7.2J prove product-look CLI and recipe enforcement on other
  fixtures, not this six-model Canon cohort.
- Earlier U6 native-standard evidence contains the EOS 7D file only and does
  not establish this six-file public CLI plus recipe-replay chain.
- P314 changes no production module, profile, recipe schema, decoder, look
  parameters, or output encoder. It consumes the existing public interfaces.
- Producer R1FX/R1FY reconstruction code and reports are not copied, imported,
  or used as an oracle.

## Frozen execution

- Sources: the six rows and file identities in the exact P313 contract.
- Profile: `safe-rich-product-v1`.
- Explicit look: `ektar_100`, amount `1.0`.
- Effects: grain, halation, and dust all zero.
- Output: deterministic 8-bit sRGB PNG.
- No tile argument is supplied because current v1 recipe metadata does not bind
  a tile size; CLI and replay therefore use the same full-frame execution.
- Each row is rendered through `scripts/render_film.py --use-render-profile
  --write-recipe`, then replayed by
  `replay_style_safe_recipe_to_file` to a distinct create-only target.
- Formal execution consists of two fresh committed-head processes, one forward
  and one reverse source order. Canonical row order is `source_id`.
- All generated PNGs and recipes live below the repo-relative, P-backed
  `outputs/eval/p314_canon_sraw_product_chain_v1/` root and are deleted after
  their hashes and metadata are recorded.

## Required gates

1. Every frozen source identity and every bound production artifact is exact.
2. All six CLI renders return zero and publish one PNG plus one strict recipe.
3. Each recipe binds the exact input, explicit Ektar look, product profile,
   output hash, Look Approximation claim, and committed software identity.
4. Each replay publishes byte-identical PNG bytes at a new target.
5. Forward and reverse fresh-process canonical reports are byte-identical.
6. All source files remain immutable.
7. No network access, media outside the owned scratch root, or final scratch
   residue is permitted.
8. The tracked worktree is clean at formal execution.

## Stop rule

Any failed gate closes this exact six-file product-chain compatibility leaf.
Do not replace a source, change the look, change output depth/format, add a tile
parameter, relax identity, substitute a decoder, copy producer code, or rescue
the result after observation.

## Claim ceiling

At most: private Windows/Python end-to-end product-chain compatibility for six
exact CC0 Canon sRAW/mRAW files through the existing Ektar 100 Look
Approximation and strict recipe replay. This is not vendor-exact colour,
general Canon/sRAW support, camera calibration, scene-to-display quality,
Ektar stock evidence, product promotion, a new decoder/interface, or candidate
3.
