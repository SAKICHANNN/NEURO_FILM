# U5.R2BV0 semantic-gated explicit curves

BV0 tested the mechanism-distinct part of the CVPR 2026 PrismNet paper without
claiming an official reproduction: fixed local DINOv2-S CLS attention gated a
63-parameter bank of nine explicit residual curves. Global, source-luminance
and half-frame-shifted gates used the same curves, fit pixels and analytical
no-clipping dose. Twenty-four development camera groups, two target variants
and two held-block folds were fixed before scoring; confirmation stayed unread.

Two formal reports are byte-identical at SHA-256
`7eca2e9e01bb2380c615cd65be1f644f835fc98cdeba7e5f885180c24c25a4df`
with stable evidence ID
`aa327ccd7f659aaa383533731cb5773a08cc96f41b958df1a2e10fb9f9a85941`.

The fixed semantic gate fails decisively. It wins zero of 48 held comparisons
against the global gate for each target variant. Mean error worsens by 37.87%
on aligned Expert C and 39.86% on the filtered target; P95 error ratios are
1.3701 and 1.3537. It also loses to luminance and shifted-gate controls, while
new-boundary fractions reach 15.97% and 3.38%. Exact replay, gate variation,
style movement, fit-to-held stability, finite output and cube bounds pass.

Close this exact source-only fixed-attention gate without model, layer,
normalization, curve or threshold rescue. Do not train its predictor or read
confirmation pixels. The result does not reproduce or falsify full PrismNet;
it shows that this low-cost frozen semantic signal is not a useful replacement
for the current global explicit colour path on this controlled population.
