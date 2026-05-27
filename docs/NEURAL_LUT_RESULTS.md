# Neural LUT Results

> Created: 2026-05-27 on the Windows RTX machine.

## Scope

Part 2A starts with a constrained AI color-rendering path: a model may predict
LUT weights, but the final image is still produced by a deterministic LUT
operator and must pass the Part 1 safety evaluator.

## Scaffold

Implemented:

- `src/models/color_lut/lut.py`
  - differentiable trilinear 3D LUT apply op,
  - identity LUT helper,
  - basis LUT module,
  - tiny image/style encoder for basis weights.
- `configs/model/neural_lut.yaml`
- `scripts/smoke_neural_lut.py`

Smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_neural_lut.py
```

Smoke result:

```text
identity_max_error=5.96e-08
weights_shape=[2, 3]
batch_lut_shape=[2, 17, 17, 17, 3]
output_shape=[2, 3, 24, 32]
```

Decision:

- Status is scaffold only.
- Next step is training the Neural LUT to imitate the Part 1 `safe-rich`
  renderer before adding any unpaired film-stat losses.
