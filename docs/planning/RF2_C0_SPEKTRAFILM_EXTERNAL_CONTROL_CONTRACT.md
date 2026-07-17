# RF2.C0 spektrafilm external spectral-control contract

Date: 2026-07-17

Node: `ULT > RF2.C > RF2.C0`

Status: frozen before runtime installation or rendering

## Question

Can a current, explicit spectral negative/print/scan simulator produce visibly
stylised, non-basic and artifact-safe stock distinctions on the project's
existing provisional gold inputs, without being integrated into the codebase
or misrepresented as real-film evidence?

This is a control experiment, not a new Ultimate truth source. It cannot repair
the failed Commons/YFCC identifiability gates.

## Source, licence and epistemic boundary

The external source is `spektrafilm` revision
`3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`, audited in an isolated shallow
checkout of 35,142,687 bytes. The code is GPL-3.0-or-later. Its 28 profiles are
CC BY-SA 4.0; 20 are capture-film profiles, six are print papers and two are
cine print stocks. The official releases contain no prebuilt assets.

The profiles state that data were processed from manufacturer data sheets and
scientific papers, but do not provide a per-profile independent measurement
ledger. The runtime also contains generic or stock-specific hand-modelled
coupler parameters; comments explicitly describe some as eyeballed. Therefore
the simulator may be physically informed and aesthetically strong, but its
output is not a measured stock response or a digital-to-film identification.

No external code, profile or LUT may be copied into tracked project files.
The root project still lacks a licence decision, so this node cannot make an
integration or redistribution decision. Generated comparison images remain
ignored internal evidence and may not become training data or teacher truth.

## Frozen runtime and inputs

Use official CPython 3.13.14 in a temporary isolated install, with the official
64-bit installer SHA-256 pinned in the config. Do not add PATH, launcher,
shortcuts or file associations. Total download is capped at 1 GiB, temporary
storage at 3 GiB, and GPU/paid resources are forbidden.

The input is the exact nine-sample `u41_provisional_eval_set` gold split at
SHA-256 `de78bb8e...26cc15`. These JPEGs are display-sRGB proxies, not calibrated
scene-linear values. Decode their sRGB transfer function and label every output
`Look Approximation`. Phase A is capped at a 1024-pixel long edge.

Freeze six profile/interpretation chains: Ektar100, UltraMax400 and Portra400
through Portra Endura; Pro400H through Fuji Crystal Archive Type II; Velvia100
and Provia100F as direct positive scans. Velvia100 is not Velvia50.

For each profile compare fixed exposure zero against the simulator's
center-weighted auto exposure. This separates the stock/prior transform from
one form of per-image adaptation.

## Frozen renderer and gates

Use the direct spectral path. Disable LUT approximations, grain, halation,
glare, diffusion, every lens blur and scanner sharpening. Keep the profile's
DIR couplers and stock-specific defaults because they are part of the external
colour hypothesis; document their heuristic status.

A candidate reaches visual review only if all are true:

1. every output is finite and in `[0,1]`;
2. worst new hard clipping is at most 0.5%;
3. gold median style Delta E76 is at least 7.0;
4. median residual after EV/WB/contrast/saturation/global-luma matching is at
   least 4.9 Delta E76.

Automatic colour metrics use at most 65,536 deterministic uniform pixels per
image. A newly hard-clipped channel is one whose output is within 0.5/255 of an
endpoint while the corresponding source channel was not already within that
same endpoint. The reported fraction is over all sampled channels, and the
candidate gate uses the worst of its nine images. These definitions are frozen
before the first formal Phase A render and may not be retuned after inspection.

These style/non-basic thresholds reuse the frozen RF2.S0 screen; they do not
validate authenticity. Pairwise profile distinctiveness is diagnostic because
no confirmatory threshold has been established.

At most three automatic survivors may enter three blind autonomous visual
rounds on all nine samples beside input, safe-rich and owner-anchor controls.
Any confirmed severe artifact vetoes the candidate. This visual evidence is
not owner or population preference.

## Branches and DoD

- **Stylised and artifact-safe:** retain only as an external future comparison
  control. No code/profile/LUT or product integration opens.
- **Bland/basic-only:** close as unable to answer the user's style problem.
- **Clipping/severe artifact:** close despite strong style.
- **Runtime/contract failure:** close this revision/environment; do not silently
  change interpreter, source revision, input interpretation or profiles.

Record external revision and licence hashes, official interpreter and package
provenance, exact input/output hashes, commands, parameters, automatic metrics,
blind mappings/reviews and the claim ceiling. Propagate the decision and run
the project CPU suite before closure.

Every branch keeps training, operator fitting, latent modes, stock truth,
calibration, authenticity, product integration and root-licence decisions
closed.
