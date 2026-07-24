# U2.2B sensitometry-to-print composition results

Date: 2026-07-24  
Decision: **numerical composition pass; U2.2 representation closes**

## Outcome

The U2.2A exposure-to-layer-density primitive composes successfully with an
explicit dye/print/paper interpretation without repeating a capture matrix or
negative characteristic curve. All frozen boundedness, determinism,
non-affinity and local-orientation gates pass.

Formal report SHA-256:
`97a7a2f3d92e914b88cbc12449dfcd6d0e1b1ab3fd7794bf172723287bee8886`.

## Results

- architecture audit: one sensitometry stage; zero forbidden print fields;
- exact black/white endpoint error: `0.0`;
- complete 17^3 output range: `[0,1]`, without clipping;
- full/11-part execution, replay and source preservation: exact;
- minimum directional derivative: `0.003792`;
- minimum finite-difference Jacobian determinant: `0.04701`;
- identity RGB RMSE: `0.15958` against `0.05` floor;
- best affine residual RGB RMSE: `0.10146` against `0.02` floor;
- RGB and layer-density out-of-domain guards: pass;
- two complete runs: hash-identical.

The large affine residual establishes that toe/shoulder, layer-density and
print interaction produce a meaningful nonlinear colour operator. It is not a
claim that the descriptive witness resembles a real stock or is visually safe.

## Branch

U2.2's numerical representation is complete: explicit exposure encoding,
anchored layer sensitometry and non-duplicative print interpretation now exist.
Open U2.3 only for a separately frozen explicit smooth 3D residual/identity
audit. Production integration, named-stock fitting, calibrated claims and
visual promotion remain closed.

## Reproducibility

- config SHA-256:
  `48093062cb8601c365717ba399a64559382233bdb9d3b05b1980dc81c243e137`;
- U2.2A config SHA-256:
  `8f43edcbc7e168af830081dce16ecb5f268a294d2f0cd777b942aec4b95c8af2`;
- print config SHA-256:
  `fffb390eb7bb01789b770964da6cea081393ce09e0019dcd7d4a43098d3bf337`;
- software commit: `ab81c79a518c8a7c04b65b7b76ba74df949726a5`;
- full CPU suite: 800 passed in 65.92 seconds.

## Claim ceiling

Uncalibrated clean-room numerical composition of exposure-to-density and
density-to-print primitives.

