# U1.4C0 Wide-Gamut Operator Compatibility Audit

**Date:** 2026-07-18

**Decision:** current colour/effect operators are sRGB-bound; direct Rec.2020
integration is forbidden

## Question

After U1.4A/B, can a display-linear Rec.2020 WorkingImage be passed through the
current safe-Lab colour engine and FilmFX stack without losing or
misinterpreting its colour space?

No. The current interfaces correctly fail closed, and that behavior must remain
until a working-space-aware operator backend exists.

## Colour-engine findings

The portable part is narrower than the full engine:

- Lab mean/std transfer, L/chroma interpolation, tone rolloff, Lab guardrails
  and local L-detail are conceptually colour-space independent once the source
  has been converted to the same D65 Lab convention;
- `rgb2lab` input is explicitly encoded sRGB;
- `lab_to_linear_srgb`, `lab2rgb`, `in_srgb_gamut`, source/chroma gamut
  compression and `project_to_neutral_srgb` are explicitly sRGB;
- `_validate_style_rgb` clips before Lab conversion;
- colour-core grain, dither and output margin are encoded-sRGB/8-bit semantics;
- the current public renderer first calls `working_image_to_srgb_float`, which
  intentionally rejects `linear_rec2020`.

The Lab kernel is therefore a candidate for extraction, not an already
wide-gamut operator.

## Effect findings

Current FilmFX cannot be relabelled as Rec.2020-compatible:

- `luminance` uses Rec.709/sRGB coefficients `0.2126/0.7152/0.0722`;
- physical/staged halation uses an sRGB inverse transfer helper;
- grain envelopes and highlight masks consume those sRGB-bound luminance
  values;
- screen, additive, alpha and soft-light compositing operate directly on
  bounded encoded RGB components;
- fixed warm layer colours are authored in the current sRGB component space.

Each effect family requires its own later colour-space contract. A colour-only
wide-gamut pass must not imply effect compatibility.

## Rejected shortcuts

1. Passing BT.2020-encoded samples to `style_transfer_rgb` would make
   `rgb2lab` interpret them as sRGB.
2. Converting Rec.2020 to sRGB, clipping, running the legacy engine and
   converting back irreversibly collapses out-of-sRGB colour. It may be an
   explicitly labelled compatibility render, but it is not wide-gamut
   preservation.
3. Tagging an sRGB result as BT.2020 without a colourimetric conversion is a
   metadata error.
4. Applying existing FilmFX component constants to Rec.2020 without new
   luminance/transfer/compositing definitions is unidentified.

## Authorized next child

U1.4C1 may design and implement the smallest working-space-aware colour-only
path:

- extract a pure D65 Lab transformation kernel from encoding/gamut/output;
- prove byte or frozen-tolerance parity for every existing sRGB style;
- add Rec.2020 linear-to-Lab and Lab-to-linear-Rec.2020 conversion;
- use a Rec.2020-specific bounded gamut policy;
- initially forbid grain, dither, output-margin and all FilmFX;
- return a display-linear Rec.2020 WorkingImage for U1.4B output;
- do not change renderer defaults or schemas.

If sRGB parity cannot be preserved, the extraction stops. If a Rec.2020 result
only survives by sRGB clipping, it is rejected as a wide-gamut operator.

## Claim ceiling

This is a source-code compatibility audit and negative integration decision. It
does not prove a Rec.2020 style operator, visual advantage or product support.
