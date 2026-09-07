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

### Three-checkpoint diagnostic closure

`scripts/audit_early_lut_conditioning.py` is a read-only CPU probe on one fixed
64x64 synthetic RGB ramp. No training or photographs are used by the diagnostic.
Results: smoke style-LUT span .0090916; s300 and s800 span exactly0. Smoke logits
range -1.65..2.84; s300 -20.21..23.24; s800 -73.12..154.51. Softmax saturation is
observed; causal attribution to learning rate, target diversity or preprocessing
is not established by comparing distinct runs.

The training script minimizes only L1 against safe-rich targets, reports loss on
the same examples and has no condition-effect validation. Training uses padded
128px squares while the old evaluation consumes unpadded 1600px images: another
known mismatch to avoid in new experiments, not a proven sole collapse cause.

Unchanged smoke checkpoint inference also completed on the same three inputs.
Its metrics record only Portra/Velvia, six examples and60 steps. Ektar was
additionally executed but is an **untrained ID**, excluded from quality claims.
Velvia sheet inspected: still visually near-input. Summary SHA256
`fc58c54d5b01fd67bf36a638f98dc69d0f87f824d9abc5b0adcd02546ed404c2`, under
`outputs/ai_recovery_20260907/neural_smoke`. No safety/full-size promotion.

This closes the bounded old-LUT recovery pass: do not expand old-weight tests or
repair pseudo-teacher fitting as the new film-learning method. Next audit legal
non-paired style supervision and official learned operator implementations;
retain these recovered outputs as bland learned baselines.

### Neural s800 recovery and conditioning failure

Executed unchanged `evaluate_neural_lut.py` with
`outputs/neural_lut/challenge_color6_s800_b12_init/model.pt`, first same three
historical inputs at their 1600px source size, Ektar/Portra/Velvia IDs and legacy
output margin4. CUDA inference completed nine outputs; 28 artifacts / 31,297,190
bytes under `outputs/ai_recovery_20260907/neural_s800`. This differs in resolution
from the preceding 768px numpy recovery; do not compare their metrics directly.

Checkpoint SHA256:
`83c21d2121ab9bb2a3e99ffa49653a947ae16c1ccddb0222a7a152f098919899`.
Summary SHA256:
`7a9e69c9a71605fe7c7de9c112ca8c40adefafe22a5afb360f40b96f707d0dd9`.
For each input, all three style output PNG hashes are identical:

- 01: `77b815776025e5311df93c42a182661749e36d1c404f97a6e3d21fc786c40c83`
- 02: `703dad793922e41b7476f0b52bf464b9d8704280b68a44b54e0aed8e95c64366`
- 03: `a47ad16647b119fdfab1bdfffb2b00edb22d72a534f628b719bab537c47cedfd`

Read-only CPU probe on input01 through the exact loaded encoder: every one of
the eight checkpoint style IDs returns one-hot weight at zero-based basis7
of12, all other weights zero; maximum cross-style LUT difference is exactly0.
Thus this checkpoint exhibits collapsed conditioning on this probe. This is not
proof of the training cause, nor a universal statement about all images/models.
Ektar sheet inspected: very small visual change. SSIM .99902-.99968 is not a
style-success measure. No promotion, retraining or checkpoint modification.

The documented owner anchors 01/09/53/55/56 are recovered in
`render_filmcase_anchor_set.py` as deterministic pipeline recipes, explicitly
anchor-inspired rather than exact historical replay. They do not identify a
preferred AI checkpoint. Keep that distinction instead of retroactively labeling
those preferences as neural-model success.

Recover neural checkpoint inference and trace the genuinely preferred early
outputs separately. Compare on the same development photographs before deciding
what is reusable. Keep diffusion-era output evidence, but do not restart final-RGB
generation. A new learned route must change the supervision, not merely distill
another hand-designed palette. Independent assessment remains sealed.
