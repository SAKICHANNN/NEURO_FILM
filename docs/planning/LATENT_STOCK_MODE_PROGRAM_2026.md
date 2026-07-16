# Latent stock mode research programme

Date: 2026-07-16

Node: `ULT > RF stock-first > LSM`

Status: hypothesis programme; `LSM0` complete when this contract is accepted,
`LSM1` data-gated; no mode discovery, clustering, operator fitting or router
training is currently authorised.

## 1. Fact-safe summary

Ultimate remains stock-first. It no longer assumes that every stock must be
represented by only one average model, but it also does not claim that every
stock necessarily contains multiple modes. Within-stock multimodality is a
strictly data-gated research hypothesis: only after stock/source/content
identifiability passes may the project test whether stable latent
appearance/operator modes remain across independent groups after controlling
basic adjustment, strength, content and scanner/source nuisance. Without
independent metadata, modes may be called only `Mode A/B/C`, not exposure,
illuminant or push/pull modes. Unpaired operator estimation remains an
unresolved identifiability problem; every candidate can provide only
film-inspired evidence. If evidence supports only `K=1`, the Oracle has no
gain, or a severe artifact appears, use the stock global champion or the
safe-rich fallback.

## 2. Epistemic contract and hypothesis

### H-LSM-1

Within one evidence-backed `film_stock_id`, two or more latent
appearance/operator modes **may** be stable across independent groups. This is
a falsifiable hypothesis, not current project truth.

Potential unobserved causes include exposure, metered EI, illuminant spectrum,
filtration, normal/push/pull processing, process session, scanner/lab/print
interpretation, long exposure, emulsion generation and their interactions.
Unless an independent source supplies trustworthy structured metadata, these
causes remain `unknown` or `hypothesis_only`. Appearance must never be used to
backfill an observed physical label.

Current evidence establishes none of the following:

- no stock has two or more demonstrated transferable modes;
- no unpaired digital-to-film operator has been identified;
- no cluster has been shown to be an exposure, illuminant, push/pull, process
  or emulsion mode;
- no current community pixel pool is eligible for stock learning or latent
  stock-mode discovery.

Legal identifiers are `<stock>/Mode A`, `<stock>/Mode B`, `<stock>/Mode C`, or
an explicitly visual description such as `warm-soft-like` or
`cyan-shadow/warm-highlight-like`. Physical interpretation fields default to
`unknown`; a proposed interpretation must be marked `hypothesis_only` and
stored separately from observed evidence.

## 3. Stock-first architecture and formal K=1 branch

```text
evidence-graded film_stock_id
  -> K=1 stock global / unknown-mode champion
  -> optional Mode A/B/C only after LSM gates pass
  -> within-mode content retrieval
  -> hard sparse routing
  -> OOD fallback
  -> bounded explicit operator
  -> deterministic final RGB renderer
```

The user selects the target stock. A mode system never guesses the stock of the
input digital photo. `K=1` is a formal result, not a failure to be hidden: close
latent FilmCase, do not train a router, and use the stock global champion plus
bounded strength. If that champion is absent or colour state is unknown, fall
back to safe-rich Look Approximation.

ML may predict curve knots, positive matrices, LUT weights, bounded grids,
mode IDs, sparse retrieval weights or optical-effect parameters. It may not
directly generate the final Style-safe RGB image. Dense mode blending is not a
fallback because it can average away the style the programme is testing.

## 4. Entry gates

All four gates are conjunctive. Failure keeps the stock in data/source work and
forbids added model capacity.

1. **Stock evidence:** sufficient label confidence, source eligibility and
   allowed use. `S0 claimed` data does not automatically enter mode research.
2. **Connectivity:** enough same-source/different-stock and
   same-stock/different-source structure, independent uploaders/groups and
   non-bound content cells to distinguish stock, source and content.
3. **Stock identifiability:** held-out-group stock signal survives scene
   colour, content, geometry, source, scanner and shuffled controls. An
   unidentified stock layer cannot contain identified stock modes.
4. **Pixels and rights:** live rights, exact manifest, duplicate/sibling and
   uploader/source groups, allowed-use scope, bounded acquisition and zero
   cross-split exact/perceptual leakage. Metadata support alone never opens
   pixels.

`SF1.1` is a valid upstream connectivity test. Its frozen 65,644,027,904-byte
metadata-only contract and pass/fail branches are unchanged. A pass opens only
the already defined live-rights preflight; a failure closes that expansion but
does not prove that latent modes do not exist.

## 5. Prohibited shortcuts and separated feature spaces

Do not cluster uncontrolled raw RGB means/histograms, thumbnails, CLIP or scene
embeddings, content labels, time/place, geometry, border, resolution, aspect
ratio, uploader, scanner/source or uncontrolled scan appearance and call the
result a film mode. Current community evidence already shows 4x4 scene colour
can predict stock labels better than global RGB and same-stock source geometry
can reach 92.86% balanced accuracy.

Two representations are mandatory:

- **mode space:** basic-normalised residuals, strength-aligned responses,
  hue-by-luma responses, monotone-curve or positive-matrix parameters, Hald
  residuals, and low-dimensional smooth-LUT signatures;
- **content space:** scene class, illumination proxy, luma/dynamic-range,
  face/skin, sky/foliage and texture/structure, used only for retrieval after
  a stock and eligible mode have been selected.

One embedding must not simultaneously decide mode identity and content
similarity.

## 6. DRPT nodes

Every experiment records config hash, software commit, seed, exact manifest,
group split, source/rights, report and claim ceiling. Normalisation,
PCA/whitening and thresholds are fit only on development groups and frozen
before confirmatory access.

### LSM0 — ontology and epistemic contract

- **Parent evidence:** stock-first hierarchy; RF2.S0 and SF1.0B negative
  results; external autonomous 53/55/56 and ID 11 evidence.
- **DoR:** current authorities and frozen gates refreshed.
- **Allowed:** define fact/inference/interpretation ledgers, legal names, K=1,
  claims and conditional tree.
- **Forbidden fallback:** describe H-LSM-1 as discovered truth or rewrite a
  completed experiment.
- **DoD/evidence:** this plan, the separate registry and authority propagation
  agree; terminology/links/diff checks pass.
- **Branch/stop:** complete the ontology regardless of later K; no experiment
  opens by LSM0 alone.

### LSM1 — data/connectivity feasibility

- **Parent evidence:** a candidate stock's observed registry rows and RF/SF
  source audits.
- **DoR:** label grade, rights, rows/pixels, groups, source/content cells,
  connectivity edges and duplicate/leakage state are all available.
- **Allowed:** build support/connectivity matrices and issue only
  `mode-study eligible`, `insufficient connectivity`,
  `source/content confounded`, `rights blocked`, `metadata only` or
  `unidentified`.
- **Forbidden fallback:** training, clustering, feature fitting or pixel access
  beyond an independently frozen source contract.
- **DoD/evidence:** a per-stock feasibility row with exact supporting artifact
  hashes and a branch decision.
- **Branch/stop:** any failed gate remains a data gap; only a full pass opens
  LSM2.

### LSM2 — residual appearance identifiability

- **Parent evidence/DoR:** LSM1 pass for one stock with development and sealed
  confirmatory groups.
- **Allowed:** handle known borders/date/orientation/scan artifacts; fit or
  match EV, WB, contrast, saturation and global luma on development only;
  align legal strength paths; test residual multimodality with group
  bootstrap and source/content/scanner controls.
- **Forbidden fallback:** call appearance residuals digital-to-film operators
  or tune gates after confirmatory access.
- **DoD:** preregistered residual evidence survives controls and sensitivity,
  or closes as unidentified/K=1.
- **Stop:** content/source/basic/strength explanation closes mode discovery for
  that pool.

### LSM3 — explicit operator identifiability candidates

- **Parent evidence/DoR:** LSM2 pass plus a separately eligible neutral digital
  control pool when required.
- **Candidates:** appearance-only residual signature; distribution-matched
  explicit bounded operators; multiple-canonicalizer sensitivity.
- **Allowed claim:** `film-inspired/unpaired-evidence` hypothesis only.
- **Forbidden fallback:** call distribution matching a pair, or claim a true
  digital-to-film/stock response. Partial FiveK, FilmSet or any other neutral
  pool needs its own lineage/right/claim gate.
- **DoD:** signatures are stable across legal matcher, canonicalizer and
  neutral-pool variants.
- **Stop:** materially different modes across assumptions means
  `unidentified`.

### LSM4 — latent mode existence audit

- **Parent evidence/DoR:** one signature definition passes LSM3 and the
  complete development protocol is frozen.
- **Candidates:** `K=1`, HDBSCAN, GMM, mixture of factor analysers and other
  preregistered group-aware mixtures.
- **Controls:** group support/effective count; balanced content; source/scanner
  and geometry; basic adjustments; group bootstrap; leave-source/group-out;
  initialisation, feature, canonicalizer, neutral-pool and matcher sensitivity;
  shuffled/random/source/content/scanner/strength-only negative controls.
- **Forbidden fallback:** promote by silhouette, BIC or a contact sheet alone;
  force every sample into a mode; refit development transforms on confirmatory
  groups.
- **DoD:** stable membership, medoids and operator directions survive all
  preregistered controls.
- **Stop:** accept `K=1`, merge strength-only splits, or close nuisance/content
  clusters.

The 53/55/56 anchors are a required strength-path negative control. The method
should prefer one mode plus continuous strength; stable separation into three
modes is evidence against the method. ID 11's 09/53/55/56 red
speckle/posterisation is a required worst-case regression, not a population
preference result.

### LSM5 — fixed mode-bank evaluation

- **Parent evidence/DoR:** LSM4 supports more than one mode.
- **Capacity ladder:** identity, best basic, per-channel affine, bounded
  positive 3x3, monotone curves plus 3x3, SepLUT, smooth 3D LUT, case residual.
- **Controls:** held-out groups; correct/wrong/shuffled mode; stock global;
  best basic; matched saturation/contrast/style; OOD and severe artifacts.
- **DoD:** a fixed explicit mode bank adds held-out value beyond the simplest
  controls without severe artifacts.
- **Stop:** promote the simplest adequate model or return to the global
  champion.

### LSM6 — Evaluator Oracle gate

- **Parent evidence/DoR:** frozen LSM5 bank and evaluator.
- **Question:** does post-hoc best mode/case selection significantly beat the
  single stock global champion?
- **DoD:** preregistered group-aware Oracle gain with no severe-veto breach.
- **Stop:** no significant gain closes product routing even if clusters remain
  descriptively interesting.

### LSM7 — simplest hard routing

- **Parent evidence/DoR:** LSM6 passes and the mode bank is frozen.
- **Order:** hard Top-1 medoid, sparse Top-K retrieval, then a small bounded
  parameter router only if the simple alternatives leave a measured gap.
- **Forbidden fallback:** dense averaging, cross-stock routing or direct RGB
  prediction.
- **DoD:** route stability, confidence/OOD calibration and independent gain
  over global/random/shuffled/source/content controls.
- **Stop:** low confidence or OOD falls back to stock global, then safe-rich.

### LSM8 — full-resolution product/OOD validation

- **Parent evidence/DoR:** frozen LSM7 policy, provenance and replay contract.
- **DoD:** unseen source/group evidence, full-resolution severe-artifact veto,
  stress confidence interval, deterministic replay and bounded target-hardware
  resources.
- **Stop:** any confirmed severe artifact rejects the candidate/policy without
  relaxing the veto; research descriptions may remain without product routing.

## 7. Formal decision branches

| Branch | Evidence | Required action |
|---|---|---|
| A | stock/data gate fails | prohibit mode discovery; continue rights/connectivity/source work |
| B | only `K=1` is supported | close latent FilmCase; global champion plus bounded strength |
| C | clusters are a strength path | merge to one mode and retain continuous strength |
| D | content explains clusters | label content shortcut; close; do not train router |
| E | source/scanner explains clusters | label nuisance/output profile; remove from stock modes |
| F | stable modes, no Oracle gain | retain research description; product uses global champion |
| G | stable modes and Oracle gain | freeze bank; hard medoid, then sparse retrieval, then bounded router only if needed |
| H | severe artifact | reject candidate/policy; never relax the severe veto for style |

No branch failure completes or blocks the Ultimate Goal. It returns execution
to the global champion/safe-rich product path or another evidence-authorised
stock/data leaf.

## 8. Current negative evidence and frozen boundaries

- `RF2.S0` remains closed. The Gold archive display-chain matrix is not a
  digital-to-Gold operator, a mode teacher or a capacity-escalation target.
- `SF1.0B` closes the 104-file community pool for training/operator fitting;
  content, low-frequency scene colour and source geometry dominate.
- 53/55/56 are near-collinear external autonomous evidence and a strength-path
  negative control, not three demonstrated modes.
- Exposure, EI, illuminant, push/pull, process and scanner fields remain
  `unknown` unless independently observed.
- Every completed experiment, data stop, threshold and historical conclusion
  remains frozen.
