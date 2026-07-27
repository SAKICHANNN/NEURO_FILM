# U5.R2AF1 — AceTone tokenizer topology results

Date: 2026-07-28
Decision: `close_topology_failure`

## Outcome

The exact released AceTone LUT tokenizer does not preserve the topology or
fidelity of the frozen analytic safe-LUT population.

All harness controls pass:

- all eight source LUTs are finite, range-safe and have zero negative
  Jacobian cells;
- the weakest source minimum determinant is above the frozen `0.25` floor;
- the axis-swap negative control is detected at `100%`;
- two independent CPU runs reproduce every source, code tensor and decoded
  LUT exactly;
- the duplicate input produces exact codes and output in both runs.

The model result nevertheless fails decisively:

- every one of eight decoded LUTs fails the `0.5%` negative-Jacobian ceiling;
- negative-cell fractions span `7.5358–10.5972%`;
- minimum local determinants span `-0.9259` to `-2.3935`;
- all eight RGB RMSE values fail at `0.05123–0.06607` versus `0.03`;
- all eight median Delta E76 values fail at `8.287–11.139` versus `2.0`;
- all eight p95 Delta E76 values fail at `21.763–27.570` versus `5.0`;
- all eight remain finite/in-range, introduce zero measured interior hard
  clipping, and pass the local spectral-norm ceiling.

Bounded output is therefore not a topology or fidelity guarantee. Even the
identity LUT reconstructs with `10.5972%` negative cells, minimum determinant
`-2.3935`, RGB RMSE `0.06607` and median/p95 Delta E76
`11.139/27.570`.

## Exact evidence

- Evaluator commit:
  `c4c37e6ab9196f85df14075a4f439c8c72aca5c9`
- External revision:
  `916393b3f26bdf89c3d939cc5f2a9a3c115ccbc5`
- Checkpoint SHA-256:
  `115e4e8c147655fe0ae0c7fd494ec6cd3a8ff5a441ce536a886edb1ea556ef92`
- Config file SHA-256:
  `09ad3e993d9dee0f4d4a04cf2744a3bffe99033e0ee36160f3f200bff08b75bf`
- Canonical config SHA-256:
  `2388fde699e92045b621bb08f2dbc5fcb67c287ef8fb4579dd57efd11fdec156`
- Run A manifest SHA-256:
  `ca93ef5fe9ac58b69aee6376ce141fc50123cc78f3824572e51cc19e091b5fd0`
- Run B manifest SHA-256:
  `3d5daf6889da4637816476e207c03b4849f8c57c5e869671f5324022b4d9f023`
- Formal report SHA-256:
  `83aea6148af7f8660062d779edb9ccc2fdad1091611bcb2149e2fd53dc46422f`
- A second evaluation report is byte-identical.

All arrays and reports remain ignored under
`outputs/external_controls/acetone_tokenizer_topology_v1/`.

## Decision and boundary

AF1 closes the AceTone tokenizer as a safe explicit-operator representation
for this project. Per the pre-registered branch:

- do not change the decoder, loss, codebook, checkpoint or threshold;
- do not project, smooth, clamp or otherwise repair decoded LUTs;
- do not run the Qwen/GRPO selector;
- do not download the rights-unidentified benchmark or use photographs;
- do not open visual review, training, FilmCase, LSM, stock claims or product
  integration.

This is not evidence that all learned parameterizations fail. It specifically
shows that an unconstrained voxel-reconstruction LUT autoencoder can be both
range-bounded and severely orientation-unsafe. Future learned candidates must
make orientation preservation true by construction or emit a simpler
explicit parameterization with an independently verified topology guarantee.
