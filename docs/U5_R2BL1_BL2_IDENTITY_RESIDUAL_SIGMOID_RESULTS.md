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
