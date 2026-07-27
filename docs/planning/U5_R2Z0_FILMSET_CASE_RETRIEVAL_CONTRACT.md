# U5.R2Z0 FilmSet Within-Recipe Case-Retrieval Contract

Date: 2026-07-27

Status: **implementation and gates frozen; formal repeat pending**

## Parent and distinct hypothesis

U5.R2W2F0 repeats exactly. `ClassNeg` and `Velvia` are not coherent single
global operators, while their aligned per-image O0 fits are useful and
structurally safe. W2F1 output-only reference recovery remains closed.

Z0 is a new paired-control hypothesis:

> Inside one already-known recipe domain, does a fixed bank of development
> case operators have held-out Oracle value, and can input-only photometric
> similarity recover a useful fraction of that value?

This directly tests the proposed “this input resembles a previous case, so
use that case's colour operator” mechanism. It does not infer the recipe or a
stock. `ClassNeg` and `Velvia` are Capture One software controls, not film.

The 2026 NTIRE Photography Retouching Transfer challenge and INRetouch use an
explicit before/after reference pair. Z0 deliberately preserves that
information boundary; it does not pretend a styled output alone identifies
its edit.

## Frozen data boundary

Reuse the W2F partition exactly:

- first 24 internal development identities;
- first 16 internal confirmatory identities;
- input plus aligned `ClassNeg` and `Velvia` outputs only;
- 120 hash-verified decoded payloads;
- disjoint fit/evaluation coordinates, 32 pixels per `4 x 4` cell;
- all other internal identities and final 628 remain unread.

Development targets may fit the case bank. Confirmatory targets may only score
the fixed bank and define the evaluator Oracle; retrieval never sees them.

## Development-only bank

For each recipe:

1. fit one shared bounded O0 on all development fit pixels;
2. fit one bounded O0 per development aligned pair;
3. evaluate each case on its own disjoint development evaluation pixels;
4. retain only cases improving identity by at least 25%;
5. require at least 12 eligible cases;
6. freeze the bank before confirmatory scoring.

All flows keep the W2F0 axis, integration, coefficient and structural bounds.
There is no spatial model and no direct neural RGB output.

## Frozen selectors and controls

The primary selector is hard Top-1 nearest neighbour in a development-z-scored
input descriptor:

- global RGB mean, standard deviation and quantiles;
- global luma and chroma mean, standard deviation and quantiles;
- per-cell RGB mean plus luma mean/standard deviation over the `4 x 4` grid.

A global-only descriptor is secondary. Both use input pixels only.

Compare:

1. shared O0;
2. best eligible case chosen post hoc by target RMSE: evaluator Oracle;
3. primary and secondary input-only nearest neighbour;
4. exact expected random-bank RMSE;
5. deterministic shuffled feature-to-operator assignment;
6. best operator from the other recipe bank, post hoc, as a domain negative
   control.

Content representation is used only after the recipe is known. It is not a
mode-space feature and cannot establish stock, process or latent mode.

## Frozen branches

Oracle value requires:

- at least 10% mean RMSE improvement over shared O0;
- wins on at least 75% of confirmatory identities;
- paired-bootstrap 95% lower bound above zero;
- every retained operator passes structural gates.

The primary retrieval passes only if it:

- closes at least 25% of the shared-to-Oracle mean gap;
- beats exact random expectation by at least 2%;
- beats the shuffled assignment by at least 2%.

Branches:

- `case_bank_oracle_and_retrieval_pass`: a simple within-recipe hard retrieval
  mechanism survives; only a separately frozen full-raster/product gate may
  open.
- `case_bank_oracle_only`: bank diversity matters, but the simple selector
  fails; preserve the gap and do not train a router without a distinct
  preregistered hypothesis.
- `case_bank_no_oracle_value`: close FilmCase for these recipe controls and
  retain the shared champion.
- `case_bank_invalid`: insufficient development cases or unsafe operators.

No branch reopens W2F1, permits descriptor/gate selection after results, reads
the final 628, accesses current real-film pixels, trains a stock/mode model or
creates a product visual shortlist.
