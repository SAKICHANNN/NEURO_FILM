# U7.15B Desktop Visible Strength Quantization Contract

## Purpose

The native Tk desktop currently displays an integer percentage while passing the
unrounded `DoubleVar` value to preview rendering. A programmatic or pointer value
of `0.654321` is therefore shown as `65%` but rendered and recorded as
`0.654321`. This leaf makes the visible control authoritative without narrowing
the continuous `[0, 1]` workflow or CLI APIs.

This is a bounded product-correctness leaf under the Look Approximation goal. It
does not change a renderer, look, profile, output transform, recipe schema,
receipt schema, batch order, preview representative, or evidence label.

## Frozen parent and defect

- Parent commit: `ea805347a1082737fc0efa3be47b1275ac6ff754`.
- Parent UI Git blob: `6928bc18cb0b81367eaeb7b82813a2f195c6613e`.
- Parent UI Git-object SHA-256:
  `ef381c2f9a3dcac103acd90435d9fec644b50b9b4db36376c14867b7c81fb60e`.
- A read-only real-Tk reproduction before this contract set the desktop amount
  to `0.654321`: `_amount_changed()` displayed `65%`, while
  `render_previews()` forwarded `0.654321` to the workflow.
- U7.15A is the immediate parent. Its representative-selection behavior and
  evidence remain immutable.

## Frozen implementation

1. Only `src/inference/product_desktop_ui.py` may change in production code.
2. Quantize to one visible percent using deterministic half-up arithmetic:
   `percent = floor(value * 100 + 0.5)`, clamped to `[0, 100]`, then
   `amount = percent / 100`.
3. Reject non-finite values before rendering.
4. The scale callback writes the snapped amount back to the `DoubleVar`, updates
   the label from the same integer, and preserves existing preview invalidation.
5. `render_previews()` repeats the same snap before starting work so direct
   `DoubleVar.set()` cannot bypass the visible contract.
6. `ProductDesktopWorkflow` and public CLI calls retain continuous `[0, 1]`
   amounts. Direct core calls at `0.625` remain `0.625`; only the native UI turns
   a visible `63%` selection into `0.63`.

## Frozen cases

- `0.654321 -> 65% / 0.65`
- `0.625 -> 63% / 0.63`
- `0.624999 -> 62% / 0.62`
- `0.005 -> 1% / 0.01`
- `0.004999 -> 0% / 0.0`
- values below zero and above one clamp to `0%` and `100%`
- NaN and infinities reject before workflow execution
- existing integral-percent anchors `0`, `0.5`, `0.65`, and `1.0` remain exact

## Gates

1. The committed parent reproduces the visible/forwarded mismatch.
2. Current slider callback label, `DoubleVar`, workflow amount, preview state,
   exported receipt, and strict recipe all agree on the snapped value.
3. A programmatic non-integral `DoubleVar` write immediately before render is
   snapped again and cannot bypass the UI contract.
4. Non-finite values reject before workflow execution.
5. Parent anchors `0`, `0.5`, `0.65`, and `1.0` keep exact rendered preview,
   output, recipe, and receipt bytes.
6. U7.15A explicit representative selection, canonical batch ordering, and
   omission behavior remain unchanged.
7. The product desktop core, renderer, catalog, profile, recipe schema, receipt
   schema, source bytes, and output-format implementation remain unchanged.
8. Forward/reverse committed-head audits are byte-identical and leave zero
   owned scratch or published residue.

## Stop rules

- Do not change percent precision, rounding mode, clamp, look, source, renderer,
  profile, recipe, receipt, output format, representative, threshold, fixture,
  or gate after observing candidate output.
- A failure closes this exact native-UI quantization. Do not rescue it with a
  second slider, editable text field, per-look strength, presets, strength
  history, automatic strength, or adjacent widget work.
- Do not generalize a pass to GUI quality, installer, public release,
  cross-platform UI, calibrated stock response, physical-film reproduction,
  stock distinguishability, HDR/wide-gamut support, or product-value evidence.

## Claim ceiling

Private Windows/Python native-Tk visible-strength consistency for deterministic
`film-inspired / Look Approximation` output only.
