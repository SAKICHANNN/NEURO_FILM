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

## MVP Imitation Smoke

Command:

```powershell
.\.venv\Scripts\python.exe scripts\train_neural_lut.py --limit 3 --styles portra_400,velvia_50 --steps 60 --image-size 96 --lut-size 17 --num-basis 4 --output-dir outputs\neural_lut\mvp_smoke
```

Result:

```text
device=cuda
styles=portra_400,velvia_50
example_count=6
initial_l1=0.010428
final_l1=0.003276
improvement_ratio=3.18x
```

Ignored outputs:

```text
outputs/neural_lut/mvp_smoke/metrics.json
outputs/neural_lut/mvp_smoke/sample_source.png
outputs/neural_lut/mvp_smoke/sample_target_safe_rich.png
outputs/neural_lut/mvp_smoke/sample_pred_neural_lut.png
outputs/neural_lut/mvp_smoke/model.pt
```

MVP decision:

- The Neural LUT can learn a small imitation target from the Part 1 renderer.
- This is still research, not a replacement for `safe_lab`: the smoke test only
  covers two styles, six examples, and low resolution.
- Next useful step is a larger imitation run over all six color stocks, followed
  by Part 1 safety evaluation on Neural LUT outputs.
