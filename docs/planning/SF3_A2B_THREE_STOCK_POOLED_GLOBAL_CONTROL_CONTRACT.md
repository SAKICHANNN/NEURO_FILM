# SF3.A2B — pooled-global control for three stock-specific K=1 operators

## Question

Do the independently selected Velvia 50, Portra 400 and Ektar 100 global
operators improve sealed confirmation targets beyond a single fixed operator
fit to the pooled development observations from all three stocks?

## Frozen comparison

- Reuse the exact SF3.A2 candidate family, bounds, development folds and
  confirmation samples.
- Select each stock-specific operator from only that stock's development
  rolls, exactly as SF3.A2 does.
- Select one pooled operator from all development rolls, with each
  `(stock, roll)` treated as an independent fold.
- Freeze both selections before any confirmation target is evaluated.
- On each confirmation frame, compare the stock-specific and pooled operators
  against the same target samples.

The pooled operator is the required fixed-global baseline. It represents
colour changes shared across the acquisition, process and scanner population;
beating identity or a wrong-stock operator is not sufficient if the
stock-specific operator does not also beat this pooled control.

## Gates

For every stock, require at least 75% confirmation-frame wins over the pooled
operator, at least 5% median relative error reduction and no worse than a 10%
relative loss on the worst frame. All outputs and metrics must be finite and
the complete report must replay exactly.

## Stop rules

- No confirmation-informed family selection, threshold change or rescue.
- No adaptive LUT, retrieval, router, latent mode or source/content feature.
- No change to the frozen SF3.A2 v1 result or candidate family.
- A failure keeps the pooled/global or identity baseline and closes the failed
  stock without adding capacity.

## Claim ceiling

Passing can establish only that three controlled stock-labelled K=1 operators
add held-out predictive value beyond one pooled global explicit operator. It is
not calibrated stock response, scanner/process generalisation, population
preference, product promotion, K>1 or multi-stock completion.
