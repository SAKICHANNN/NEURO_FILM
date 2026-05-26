# SDXL LoRA Validation Results

> Last updated: 2026-05-26 on the Windows RTX machine.

## Summary

The current SDXL LoRA img2img/SDEdit path is not yet a usable film translation deliverable.

It is better than the SD1.5-IP2P run in one narrow sense: low-strength SDEdit does not create the severe painterly smearing seen in IP2P. But it has a different failure mode:

- Low strength preserves content but produces almost no visible film change.
- Medium strength produces visible change but starts rewriting semantic content.
- The HP5/Tri-X black-and-white prompts do not reliably force black-and-white output before content drift appears.
- Community Portra 400 LoRA failed to load in the current diffusers environment with `IndexError: list index out of range`, while the self-trained LoRAs load.

## Evidence

Low-strength probe on pseudo-digital film-derived input:

- `outputs/sdxl_lora_grid/portra800_low_strength_probe/contact_sheets/portra_800_contact_sheet.jpg`
- `outputs/sdxl_lora_grid/hp5_low_strength_probe/contact_sheets/hp5_contact_sheet.jpg`
- `outputs/sdxl_lora_grid/tri_x_400_low_strength_probe/contact_sheets/tri_x_400_contact_sheet.jpg`
- `outputs/sdxl_lora_grid/velvia_50_low_strength_probe/contact_sheets/velvia_50_contact_sheet.jpg`

Result: content is preserved, but the outputs are nearly unchanged.

Medium-strength probe on a standard digital photo sample:

- `outputs/sdxl_lora_grid/portra_800_astronaut_medium_probe/contact_sheets/portra_800_contact_sheet.jpg`
- `outputs/sdxl_lora_grid/hp5_astronaut_medium_probe/contact_sheets/hp5_contact_sheet.jpg`

Result: style effect becomes visible, but subject identity, face, clothing, and local details begin to change.

## Decision

Do not promote the current SDXL LoRA + SDEdit pipeline as the default solution.

Keep the trained LoRAs because they are useful artifacts, but the next implementation direction should add a non-generative or tightly constrained color pipeline before returning to diffusion:

1. Establish a deterministic color-transform baseline that cannot alter geometry.
2. Use film-domain statistics or curated references to drive tone/color changes.
3. Add optional grain/halation after the color transform.
4. Reintroduce diffusion only as an optional, low-strength embellishment after a content-safe baseline is acceptable.

This is a plan correction: the V3 diffusion-first assumption is too optimistic for the current data and hardware.
