# Data License Boundaries

This file is the project-level warning label for dataset licensing, publication, and model release decisions.

## Hard Rules

- Do not redistribute third-party datasets, extracted samples, processed copies, or private access links.
- Keep restricted datasets out of public archives, demo bundles, checkpoints, and model cards unless the license explicitly allows that use.
- Do not train a publicly released or commercial model on datasets whose license is research-only, non-commercial, or unclear.
- Record every manifest row with `source`, `license`, `split`, `task`, `style`, and `redistributable`.
- Treat scanner output, lab correction, ICC profiles, and negative conversion settings as dataset metadata, not incidental details.

## Current Dataset Boundaries

| Dataset | Current Role | Boundary |
|---------|--------------|----------|
| MIT-Adobe FiveK | RAW/TIFF pretraining and VAE reconstruction | Research-use dataset; verify publication terms before releasing trained weights. |
| FilmSet | Film style supervision and LoRA warmup | Check Kaggle/license terms before publishing weights or examples. |
| FilmGrainStyle740k | Grain/NPS prior, if access is granted | Research/evaluation only; no redistribution; keep isolated from public artifacts. |
| DPED | Optional auxiliary enhancement pretraining | Academic dataset; not film ground truth. |
| Cinestill800T | Future sanity-check validation if released | Small validation data only; verify license before use. |
| Self-built L5 pairs | Final physical validation | Release only if subject, lab, scanner, and photographer rights are cleared. |
| PBR synthetic data | Generated physical-pair training | Prefer for redistributable training data when all assets and renderer licenses allow it. |

## Release Checklist

- Confirm licenses for all manifest sources used in the checkpoint.
- Export a training-source summary with dataset names, weights, and license classes.
- Exclude restricted validation images from examples and test fixtures.
- Separate research checkpoints from redistributable checkpoints.
- Include citations for FiveK, FilmSet, FilmGrainStyle740k, DPED, Cinestill800T, and any film manufacturer technical PDFs used for parameter extraction.
- If any uncertainty remains, release code and configs only, not weights or data-derived assets.

## Current Human Decisions

- Disk policy: continuing to use the current project disk for data and checkpoints is allowed.
- Fuji Reala is no longer a default target because independent public technical data is too weak.
- Replacement and expansion targets: Fujifilm Velvia 50, Kodak Ektar 100, Kodak Portra 800, Kodak Vision3 250D, Kodak Tri-X 400.
- L5 self-built real digital/film pairs should be pursued for final validation.
