# P254 Stage B — R1EC callable source lock

## State at freeze

This additive source lock was written before the consumer deserialized the
canonical fixture or imported the producer callable.  The fixture was treated
only as an opaque Git blob for byte count and SHA-256 verification.  P254 may
execute only after this file and the matching configuration are committed.

The producer repository was clean at
`6fd669de5e50a54233d788c3f31226989c6901de`.  The callable implementation is
fixed at `89d0d71df4d10949b7e5a148ab399021d74dc09c`, its execution lock at
`9e1d7a33c92ecd06bb0e52faee6b782f655c0b9f`, and formal execution at
`134b2e9a82c905fc6c013155d6486410c3036def`.

## Exact Git objects

All SHA-256 values below were independently recomputed from the corresponding
Git blob without importing the producer package.

| Role | Path | Git blob | Bytes | SHA-256 |
|---|---|---|---:|---|
| contract | `docs/research/R1DZ_R1DR_DNG_PROFILE_HUESAT_CALLABLE_V1_CONTRACT.json` | `8a4654c26b19ad89181d7f18142810eadab5af0b` | 2,985 | `dded21b4cc3ca868e6ef2d16c56485f10e1676f21b243cfc306b40925d77f970` |
| payload schema | `schemas/zhuise_dng_profile_huesat_callable_v1.schema.json` | `e2d38afc51ab342079f72a03b6b1349ba857ca0b` | 1,260 | `8e64fedabc17fdb54ddfad5c2f6f74b9d3c4a7e3b5f33a4542be9a0250affb89` |
| callable | `src/zhuise/dng_profile_huesat_callable.py` | `49a405060f86e3b3f6bb5196d9a5d057343fc8b9` | 4,507 | `0bd850a4869d8956d1a56498baec472b2baeaf680a8c9252bfd8aef4c7fcecbc` |
| arithmetic dependency | `src/zhuise/dng_profile.py` | `63b1b617879bead1cd542915301075856764e916` | 36,685 | `f9df747091754c9020c24e0bd53f2db1c02287e9a57be25593ea89f23af11b43` |
| canonical fixture | `tests/fixtures/r1dz_dng_profile_huesat_callable_v1_fixture.json` | `daca22389488eff392fe83aef0b982a32d77a07c` | 1,393 | `73a24e296e1d0a65232fbc9dc976369b5769ae8e6be5bd5181432dd7e953d26b` |
| execution lock | `docs/research/R1EC_DNG_PROFILE_HUESAT_CALLABLE_HANDOFF_EXECUTION_LOCK.json` | `64cc01d7a1f0175d9e6db25e513278370b62bcc1` | 2,304 | `691913b40bf79e603ef80b34b0233f222b8b98fcae6b86aada27f49fecb6fb3d` |
| evidence | `docs/evidence/R1EC_DNG_PROFILE_HUESAT_CALLABLE_HANDOFF_RESULT.json` | `f3a56ae0eef81b77fbee6f29561cff48e1f09766` | 4,027 | `d93ab2f24800225f4b224044b9c6597508850c6141fe63c9dc8563dcfe08f0d4` |

The consumer must extract the callable and arithmetic dependency from the
implementation commit into an isolated temporary `zhuise` package.  It must
never import the producer worktree or copy either implementation into consumer
`src/`.

## Callable and data contract

- Callable ID: `zhuise.dng-profile-huesat-callable.v1`.
- Function: `zhuise.dng_profile_huesat_callable.apply_dng_profile_huesat_callable_v1`.
- Signature: `(xyz_d50: NDArray[floating], payload: Mapping[str, Any]) -> NDArray[float64]`.
- Input: finite `HxWx3` linear XYZ D50.
- Internal domain: linear ProPhoto/ROMM RGB after camera white balance and the
  camera-profile matrix.
- Stage order: ProfileHueSatMap, then ProfileGainTableMap, exposure curve,
  optional ProfileLookTable, and tone curve.
- Output: owned, C-contiguous, writable float64 `HxWx3` linear XYZ D50.
- Table dimensions are `[hue, saturation, value]`; flat table entries are
  Adobe/DNG `[hue_shift, saturation_scale, value_scale]` triplets reshaped by
  the locked core to `[value, hue, saturation, 3]`.
- `calibration_1_weight` is the Data1 weight.  Two tables resolve in the exact
  float32 order `float32(weight * Data1 + float32(1-weight) * Data2)`; a null
  Data2 resolves to an owned Data1 copy.  Weight must be finite and in `[0,1]`.
- Encoding must be integer DNG linear value `0`.
- `support_overrange=false` preserves the R1DR/R1DQ SDR behavior and rejects
  ROMM inputs outside its bounded SDR tolerance.  `true` selects the exact
  R1DZ Adobe-reference encode/apply/decode path; negative ROMM components use
  the frozen floor behavior.
- Caller arrays and payload remain owned and unmodified.  Contract failures
  raise `ValueError` before any output is returned; there is no partial output,
  filesystem access, or network access.

## Canonical fixture and controls

The canonical fixture binds input f64le SHA-256
`6dbb8ce614ccadb5dd9d61cba5de03a117103e4f67ba2e551fe0de721854f369`,
canonical payload SHA-256
`ae8912a42bd8cba0eecd76feb0536261364fb15b37140512acb32a525e907c23`,
and output f64le SHA-256
`558fd7ba79339c668e3cf09585a3160244a1ec6f9d45d424e3659b17c3a34536`.
The producer's two 2,670-byte reports are byte-exact at SHA-256
`ec456d50a751554b919a138c587acdf9eef09f560d54d4dfd06273905746771e`;
their stable identity is
`d4cf3997830a840b29a2debf6fb0e9e0205f3f215ed0b4e9e2c2a2d0e84c43dd`.

P254 independently requires rejection of nonfinite/wrong-rank RGB, nonfinite
or wrong-length tables, malformed dimensions, negative saturation/value scale,
non-unit zero-saturation value scale, invalid interpolation weights, encoding
other than `0`, and non-boolean `support_overrange`.  The callable has no
block, workspace, or capacity argument, so capacity controls are explicitly
not applicable.  Failure must leave caller inputs and payload unchanged.

Default-SDR parity is limited to exact equality between this wrapper and the
locked core on the safe interior control.  R1DQ/R1DR science, table hashes, and
consumer P244/P245 predecode guards remain unchanged.

## Rights and claim ceiling

The callable contains producer-authored Python arithmetic and depends only on
NumPy at execution.  The fixed DJI metadata used by consumer verification is
the already rights-screened local metadata-only source; P254 downloads no RAW,
DNG, pixel, target, or reference asset.

At most a pass proves private, exact consumer replay of this one R1EC callable
and fixture.  It does not admit a full DNG renderer, real-pixel quality,
arbitrary profiles/cameras, public API/package/schema/capability, stock
evidence, automatic single-reference matching, or product behavior.  Candidate
3 remains closed at `2/3`.
