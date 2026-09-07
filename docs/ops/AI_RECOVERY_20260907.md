# Early model recovery: descriptive smoke, not promotion

The current user asks for AI-learned film character without supplied paired
captures. Manual creative-v6 changes remain interrupted and untouched.

## Live evidence

- `scripts/train_neural_lut.py` constructs supervised targets by calling
  `pipeline_color_baseline.style_transfer`; checkpoint
  `outputs/neural_lut/challenge_color6_s800_b12_init/model.pt` exists. Its
  800-step training loss is teacher approximation, not film authenticity.
- `outputs/neural_film_lut_v2/scheme_a_distilled_v1/metrics.json` records
  20 images and teacher `outputs/eval/color_engine_challenge/film_response_v1_s1p0`.
  `fit_distilled_film_lut.py` fits curves/residual LUT and context gain from that
  teacher. This is data fitting, but not independently learned real-film style.
- The numpy model SHA256 is
  `1be4dce9b99bb9565997e22b6265c87fdbfe11f85ac1d6155f7359f45e2059a0`.

## Executed recovery

Unchanged `scripts/evaluate_distilled_film_lut.py`, existing `.venv`, first three
historical manifest rows, Ektar/Portra/Velvia configuration IDs, strength 1.0,
max-side 768, historical output margin 4. Output directory was checked absent
before execution: `outputs/ai_recovery_20260907` (47 files, 20,561,301 bytes).
No training, new downloads, assessment-set reads, model changes or product changes.
Summary SHA256 `36b3bce76a6ac144c58ed4ba863b5201b385e25c4a45987a26585595f63ab809`.

Ektar and Velvia contact sheets visually inspected: snow scene, fruit/chart and
flowers. Changes mainly appear as contrast/chroma differences. This inspection
does not establish compelling film character, full-size safety or user appeal.
Portra executed but was not separately visually adjudicated. Same historical
training-era images are intentionally reused for recovery, never held-out claims.
Legacy clipping margin/SSIM are reported by the old runner, not current promotion
criteria. This run does not establish which early algorithm was the user's best.

## Next useful action

Recover neural checkpoint inference and trace the genuinely preferred early
outputs separately. Compare on the same development photographs before deciding
what is reusable. Keep diffusion-era output evidence, but do not restart final-RGB
generation. A new learned route must change the supervision, not merely distill
another hand-designed palette. Independent assessment remains sealed.
