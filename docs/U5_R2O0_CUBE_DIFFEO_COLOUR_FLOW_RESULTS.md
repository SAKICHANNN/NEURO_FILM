# U5.R2O0 Cube-Preserving Diffeomorphic Colour-Flow Results

Date: 2026-07-26  
Node: `ULT > U5 > U5.R2 > U5.R2O0`  
Decision: **retain as a synthetic explicit-operator representation**

## Outcome

The fixed `4×4×4×3` stationary velocity grid is the first frozen K/O-series
candidate to pass the complete conjunction of nonlinear fidelity, cube range,
positive orientation, derivative norm, inverse and reproducibility gates on
both existing explicit synthetic controls.

| Target | Confirm RMSE | Affine improvement | Min det(J) | Max ‖J‖₂ | Max inverse error |
|---|---:|---:|---:|---:|---:|
| density cyan s0.50 | 0.010908 | 84.45% | 0.1174 | 3.9926 | 8.40e-7 |
| positive warm s0.35 | 0.005436 | 80.27% | 0.2752 | 2.7806 | 3.75e-7 |

Identity is exact. All eight RGB-cube corners are exact fixed points. Every
confirmation output remains in `[0,1]` without an output clamp. The maximum
fitted coefficient is `3.636`, below the frozen `6.0` ceiling. Serialization
replay and arbitrary row partition are bit-exact.

## What changed relative to prior representations

- K0 had sufficient local fitting capacity but folded colour space.
- K1 preserved orientation and had an analytic inverse but was too weak.
- K2 improved fidelity but failed confirmation and Jacobian-norm gates.
- K3 was expressive and analytic but one fixed witness exceeded the norm gate.
- N1's external flow was smooth but its encoder followed geometry/content and
  its raw map escaped the RGB cube.

R2O0 instead integrates
`v_i(x)=x_i(1-x_i)f_i(x)` with fixed float64 RK4. The boundary-vanishing
factor and continuous flow supply a structurally bounded, orientation-preserving
operator, while the trilinear velocity grid supplies enough cross-channel
capacity for both targets.

## Reproducibility

- frozen config:
  `configs/u5_r2o0_cube_diffeomorphic_colour_flow_v1.json`;
- implementation software commit:
  `281004ebb16229d748e4159d8ee67f83852234cb`;
- two independent full optimizations are byte-identical;
- report SHA-256:
  `c8fcd0c19be421ed7d1c5f9adddd6be31ba775254b7ddd70c24d937e67e87bf0`;
- no image, film scan, external code, checkpoint or GPU was used.

Ignored reports remain under
`outputs/u5_r2o0_cube_diffeomorphic_colour_flow_v1/`.

## Boundary and next decision

This result validates an explicit parameter family, not a new aesthetic
candidate. Rendering it on existing photographs would merely approximate the
already evaluated R2E1 and J1 target looks, so no redundant real-image
frontier opens. Preserve the representation for future evidence-eligible
paired controls or stock-specific work where an ML component may predict only
the bounded velocity parameters.

The result does not identify an unpaired digital-to-film operator, authorize
current real-film fitting, establish a stock or latent mode, or open production
integration.
