# U5.R2W2F0 FilmSet Recipe Global-Explainability Results

Date: 2026-07-27

Decision: **no coherent global recipe champion**

## Reproducibility and boundary

- software commit:
  `7f799b043ca18158a4a44035cea1f3a58b650cef`;
- config SHA-256:
  `41cb4554a11a6660c9d28fa1e89ce23f7429b7fd9f059139789e36f745befaa6`;
- two complete reports are byte-identical at
  `27db75f2e74fe15a2fdfef49ed82f92ac313e44719fbc6220fcba195e51c5b3e`;
- both stderr logs are empty;
- the local CUDA device was the NVIDIA GeForce RTX 5070 Ti Laptop GPU with
  PyTorch `2.11.0+cu128`;
- 40 identities and exactly 160 aligned payload images were read;
- all selected payloads were hash checked through `WorkingImage`;
- no unselected payload was read, the final 628 manifest contributed zero
  parsed payload rows, and no full-raster output or visual shortlist was
  generated.

The formal runs retain the frozen 24-development/16-confirmatory split,
disjoint fit/evaluation coordinates and three recipe domains. They perform
51 recipe-operator fits each.

## Domain results

| Recipe domain | Identity median RMSE | Basic median RMSE | Shared O0 median RMSE | Per-pair O0 median RMSE | Shared gain over identity / basic | Shared regret | Grid dispersion | Spatial residual | Branch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cinema | `.03271` | `.01234` | `.01204` | `.00756` | `63.20% / 2.44%` | `.10370` | `.16406` | `.23081` | `basic_only` |
| ClassNeg | `.05630` | `.03003` | `.01499` | `.00806` | `73.38% / 50.09%` | `.09203` | `.16291` | `.26069` | `adaptive_or_spatial_recipe` |
| Velvia | `.02921` | `.02314` | `.01552` | `.01002` | `46.89% / 32.96%` | `.16511` | `.20832` | `.20937` | `adaptive_or_spatial_recipe` |

No domain passes `global_operator_coherent`.

Cinema's basic control improves identity by `62.29%`, while the shared O0
adds only `2.44%` over basic against the frozen 5% boundary. It is therefore
`basic_only`, even though its shared regret and spatial residual pass.

ClassNeg has a strong shared nonlinear gain and acceptable shared regret, but
its per-pair grid dispersion is `.16291` against `.08` and its spatial
residual is `.26069` against `.25`. It is not one coherent global operator.

Velvia also has a useful shared nonlinear gain, but basic improves identity
by only `20.78%`, shared regret is `.16511` against `.15`, and grid
dispersion is `.20832` against `.08`. It is likewise adaptive or spatial
under this evaluator.

## Structural evidence

All shared and per-pair O0 flows pass every frozen structural gate:

| Domain | Minimum determinant | Maximum Jacobian norm | Maximum inverse error | Maximum coefficient norm | Replay error |
|---|---:|---:|---:|---:|---:|
| Cinema | `.27897` | `2.06792` | `6.66e-8` | `1.28411` | `0` |
| ClassNeg | `.18529` | `2.03971` | `2.42e-8` | `1.28679` | `0` |
| Velvia | `.35107` | `1.70220` | `2.73e-8` | `1.12475` | `0` |

All sampled outputs remain inside the RGB cube. The negative result is
therefore about cross-content recipe coherence, not an unsafe operator,
optimizer collapse or insufficient bounded O0 capacity for individual pairs.

## Interpretation

FilmSet supplies repeated Capture One recipe outputs, not physical film
measurements. The result shows:

1. one named recipe domain is adequately explained by basic exposure,
   white-balance, contrast and saturation adjustment;
2. two recipe domains contain meaningful nonlinear change, but a single
   shared global operator does not explain their held-out content behavior;
3. a per-image fit can improve the pair without identifying what adaptive or
   spatial rule Capture One used.

This is useful mechanism evidence for the Ultimate architecture: a strong
look may require bounded conditional parameters or an explicit spatial
residual, but adding either requires a new preregistered hypothesis and
separate shortcut controls. The present result does not authorize such a
rescue.

## Branch consequences

- W2F1 output-only reference recovery stays closed because W1 already failed
  its practical information regime.
- Cinema retains only the basic control.
- ClassNeg and Velvia retain `adaptive_or_spatial_recipe` as a research
  diagnosis; no local or neural model is opened.
- No visual candidate is generated because the frozen contract forbids one.
- The final 628, current real-film pixels, stock learning, latent modes,
  calibration and production integration remain closed.

`Cinema`, `ClassNeg` and `Velvia` here are recipe-domain names. They are not
evidence of physical stocks, processes, scanners or real digital-to-film
operators.
