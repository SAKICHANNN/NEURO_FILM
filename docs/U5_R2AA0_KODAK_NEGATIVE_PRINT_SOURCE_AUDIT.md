# U5.R2AA0 Kodak VISION3 250D to 2383 Source Audit

Date: 2026-07-28

Node: `ULT > U5 > U5.R2 > U5.R2AA0`

Decision: **a bounded numerical nuisance audit is feasible; an identified
negative-to-print operator is not**

## Question

Can current, first-party Kodak material support a reproducible clean-room
candidate that is more film-specific than an arbitrary matrix or saturation
increase, while preserving the project's strict claim boundary?

The answer is narrowly positive. The March 2026 VISION3 250D and VISION Color
Print Film 2383 sheets each publish:

1. three channel characteristic curves;
2. three layer spectral-sensitivity curves;
3. separated cyan, magenta and yellow spectral dye-density curves.

The 250D panel additionally publishes a midscale-neutral spectral-density
curve and D-min. The 2383 panel states that its separated dyes are normalized
to form visual-neutral density `1.0` for a xenon-arc viewing illuminant.
Kodak's 2383 LAD sheet supplies the current neutral Status-A aims
`1.09/1.06/1.03` and the local relation that one `0.025` log-exposure printer
light changes print density by about `.07` near aim.

This is materially stronger source support than the previously closed
Portra/Ektar datasheet paths, which do not publish separated negative CMY dye
bases. It still does not identify the missing scene, printer, process or
viewing variables.

## Reproducible source inventory

The PDFs are retained locally under ignored `data/physics`; only source
metadata, hashes and reproducible download jobs enter Git.

| Source | Local file | Bytes / SHA-256 | Direct evidence | Claim-limiting gap |
|---|---|---|---|---|
| [VISION3 250D 5207/7207 technical data](https://www.kodak.com/content/products-brochures/motion-picture/KODAK-VISION3-250D-5207-7207-technical-information.pdf) | `data/physics/kodak_vision3_250d/technical_data.pdf` | 684,722 / `70adb298...2e16` | ECN-2 characteristic, sensitivity, separated peak-normalized CMY dyes, midscale neutral and D-min | raster plots; relative exposure; Status-M density is not analytical dye amount |
| [VISION Color Print Film 2383/3383 technical data](https://www.kodak.com/content/products-brochures/motion-picture/KODAK-VISION-Color-Print-Film-2383-3383-technical-information.pdf) | `data/physics/kodak_vision_print_2383/technical_data.pdf` | 631,142 / `210d5e8e...7565` | print characteristic, sensitivity and separated visual-neutral CMY dyes | raster plots; test filtration is not a complete printer SPD; projector/view transform absent |
| [LAD for 2383/3383](https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-VISION-Color-Print-Film-H-61b.pdf) | `data/physics/kodak_vision_print_2383/lad_print_h61b.pdf` | 120,558 / `56eab4bb...b229` | visual-neutral and Status-A print aims; local printer-light slope | setup/aim control only, not a full scene colour transform |
| [LAD for Kodak Color Negative Film](https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-Color-Negative-Film-H-61a.pdf) | `data/physics/kodak_vision_print_2383/lad_negative_h61a.pdf` | 107,171 / `2f096952...d053f` | historical generic LAD timing procedure | not specific to 250D; its numeric aims are forbidden as 250D truth |
| [Additive and Subtractive Printing on Motion Picture Film](https://www.kodak.com/content/products-brochures/Film/Additive-and-Subtractive-Printing-on-Motion-Picture-Film-tib5207.pdf) | `data/physics/kodak_vision_print_2383/additive_subtractive_printing_tib5207.pdf` | 4,567 / `54d459ba...0017` | tungsten-halogen/dichroic additive-printer topology | no dichroic transmission or primary spectral distributions |

All five files parse as unencrypted PDFs. The two 2026 technical sheets have
four and five pages respectively. The exact 250D file was already present in
the scripted physics scope; the four print-chain documents are added to the
same restore command.

## Graph evidence

The PDFs contain losslessly extractable embedded raster graphs. The four
primary images needed by a first numerical pilot are:

| Graph | Embedded dimensions | Extracted PNG SHA-256 |
|---|---:|---|
| 250D characteristic/granularity page 3 | 587 x 557 | `8d4ba7ac...ca0d` |
| 250D spectral sensitivity page 4 | 737 x 749 | `e51e2e29...6e77` |
| 250D spectral dye density page 4 | 693 x 754 | `a2627cee...7291` |
| 2383 characteristic page 4 | 490 x 496 | `d0c48f6d...fa0b` |
| 2383 spectral sensitivity page 5 | 428 x 397 | `5c3afcae...829c` |
| 2383 spectral dye density page 5 | 401 x 390 | `e8fefcba...a56e` |

Human-guided semantic traces can be snapped to exact source ink and validated
with overlays, as already established by U5.R2H0A. Pixel proximity would prove
only faithful graph transcription, never calibration.

## What the sources do not identify

The complete digital-input-to-projected-print chain is still underdetermined.

1. **Digital input spectrum.** Display/scene RGB has infinitely many spectral
   metamers. It does not recover capture-camera sensitivities, illuminant or
   scene reflectance.
2. **Negative density interpretation.** The 250D characteristic graph reports
   Status-M diffuse density. Its peak-normalized CMY spectra and midscale curve
   constrain a descriptive decomposition but do not publish absolute layer
   amounts for every exposure.
3. **Printer light.** Kodak describes tungsten-halogen additive printers with
   dichroic red/green/blue regions, but publishes no exact primary SPDs,
   filtration or printer-specific cross talk.
4. **Colour timing.** Kodak explicitly states that normal film and scene
   variation is handled by scene-to-scene timing. LAD is a setup point, not a
   rule that every scene must match.
5. **Print/view chain.** The 2383 graph names a xenon-arc visual-neutral
   condition but does not publish a numerical xenon/projector SPD, lens,
   screen, flare or display encoding.
6. **Process/batch state.** Both technical sheets state that typical curves are
   representative rather than specifications for a particular roll. The print
   body references current process guidance while graph captions retain their
   stated test process; neither supplies a measured joint 250D/2383 batch.

No single choice for these missing quantities may be silently promoted as the
physical Kodak chain.

## Allowed next experiment

`U5.R2AA1` may test a deterministic **datasheet-constrained negative-to-print
Look Approximation** on a synthetic RGB grid. It must be a nuisance-sensitivity
audit, not a visual grading exercise.

Required independent axes:

- the existing smooth constrained RGB-to-reflectance canonicalizer plus
  colour-matched null-space metamers;
- at least three preregistered additive-printer primary families derived from
  the published topology/sensitivity support, each neutrally rebalanced at the
  2383 LAD point and explicitly labelled `printer_hypothesis`;
- D50, D55 and D65 viewing controls, with no claim that any is the missing
  xenon projector SPD;
- at least two legal mappings between Status density and normalized dye amount;
- neutral exposure placement varied prospectively rather than selected from
  project photographs.

The pilot must compare its effect with identity and best-basic
exposure/WB/contrast/saturation controls. It may continue only if a stable
non-basic residual remains after those controls while canonical-spectrum,
printer and viewing nuisance spread stays below frozen fractions of the
effect.

## Stop conditions

Close before any real-image render if:

- graph axes or semantic traces cannot be reproduced;
- neutral luminance is non-monotone or neutral chroma exceeds the frozen gate;
- outputs become non-finite or unbounded;
- nuisance/canonicalizer spread dominates the apparent film effect;
- the apparent distinction collapses to basic exposure, contrast, WB or
  saturation;
- the result requires tuning against current film pixels, owner anchors,
  product images or an external LUT/profile.

No neural model, stock pixels, paired-target fitting, current output anchor or
external film-emulation LUT is allowed as a rescue.

## Claim ceiling

At best AA1 can establish that a bounded, reproducible, Kodak-datasheet-
constrained **film-inspired Look Approximation hypothesis** retains a
non-basic colour direction under a declared nuisance ensemble. It cannot
establish the real 250D response, the real 250D-to-2383 printing operator, a
scanner/projector transform, calibrated reference, stock authenticity,
original scene spectrum, latent stock mode or product preference.

Ultimate remains stock-first and ACTIVE regardless of this branch result.
