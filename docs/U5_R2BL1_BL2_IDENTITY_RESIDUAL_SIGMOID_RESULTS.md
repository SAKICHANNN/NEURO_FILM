# U5.R2BL1-BL2 Identity-Residual Sigmoid Results

BL1 retained the useful capacity of FilmMatch's two-matrix sigmoid family while
removing its near-singular cube response. Two formal runs are byte-identical at
SHA-256 `8e10c665...a709784` and stable evidence ID
`6cb29367...3893d6`.

- Reflective held-illuminant RGB RMSE is `0.04694`, 9.31% below the original
  two-matrix sigmoid and 64.78% below identity.
- Emissive held-sector RGB RMSE is `0.09618`, 27.08% below identity.
- Combined mean RMSE is `0.14312`, below the frozen safe-family control
  `0.20527`.
- All fitted operators remain in cube; the minimum analytic 33-cube Jacobian
  determinant is `0.01071` rather than the original family's approximately
  singular `8.18e-11` response.

BL2 applies the exact fixed BL1 operator to the already-consumed, fit-forbidden
real scene. Its two reports are byte-identical at SHA-256
`db8e589b...1ebe37c`. Style displacement from source is `0.15398` RGB RMSE,
with zero new boundary pixels. Full-resolution autonomous review finds strong
contrast/chroma salience and no severe corruption, red speckle, posterization,
banding, text damage, face damage or geometry change.

This is a promising internal **Look Approximation equation family**, not a
calibrated Ektachrome response. The validation scene was already consumed and
has different film framing, so it cannot serve as independent confirmation or
support target-pixel accuracy. No product integration opens. The next useful
step is genuinely independent paired evidence or a new content-diverse
fit-forbidden population; BL2 must not be tuned on the consumed scene.

## BL3 ordinary-photo OOD result

BL3 then applied the unchanged operator to 12 CC0 RAW photographs from 12
camera makes and compared its boundary behavior with fixed AO6. Both complete
24-output runs are exact at report SHA-256 `b1c43707...3dbb75`. Median BL1
style displacement remains strong at `0.12196`, but the Olympus E-450 output
places `3.09543%` of pixels on the output code boundary where AO6 does not,
above the frozen `0.5%` gate. Automatic failure correctly prevents blind or
visual preference review.

Direct BL1 ordinary-photo application is closed without refitting, threshold
changes or post-fit clipping. A separately frozen experiment may use AO6 as a
safe base and treat fixed BL1 only as an analytically bounded residual
direction; that is a new composition test, not a rescue or promotion of BL1.

## BL4 analytical residual composition

BL4 uses AO6 as the base and fixed BL1 only as a residual direction, with the
existing source-inclusive analytical maximum-safe-scale policy. Two runs are
exact at report SHA-256 `0de27873...0d391`. It removes all output/new boundary
pixels, retains `99.9976%` median residual L2 energy, and keeps `0.08714` median
RGB style displacement from AO6. However, the P95 per-source fraction of
safety-limited pixels is `17.996%`, above the frozen `10%` gate. Visual review
therefore remains closed.

The analytical executor is effective, but this particular composition policy
is not promoted or retuned. A next family must make strict-interior behavior
part of the fitted equation rather than relying on a large tail of per-pixel
safety intervention.

## BL5 strict-interior equation

BL5 makes the `0.5/255` output margin part of the fitted equation instead of a
per-pixel post-execution guard. Two reports are exact at SHA-256
`805f4850...6d151`. Reflective RMSE improves slightly from BL1 to `0.04669`;
emissive RMSE changes only to `0.09633`; combined mean RMSE is `0.14302`.
The 33-cube has zero quantized endpoint samples and minimum Jacobian
`0.01064`. All frozen gates pass, opening only fixed ordinary-photo OOD
execution with no refit or retuning.
