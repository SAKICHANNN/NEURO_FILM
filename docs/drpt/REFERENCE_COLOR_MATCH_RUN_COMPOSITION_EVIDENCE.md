# Reference Match Run-Level Composition Evidence

## Decision

P18 closes the gap between a requested composition plan and the colour action
actually delivered by the guarded file run. Static intent is insufficient:
film effects may be attached only after the complete transactional report and
all delivered outputs are verified.

## Contract

`ReferenceRunCompositionBinding` binds:

- the exact fit or stored-recipe replay report schema and SHA-256;
- the exact recipe-backed `ReferenceCompositionPlan`;
- every source index, source SHA-256, output SHA-256 and safety action;
- the uniform batch safety state;
- the explicit research-baseline request;
- one canonical binding identity suitable for independent ports.

The builder re-reads the transaction-bound report and delivered output files.
It verifies the report hash, report/result fields, recipe identity, output
paths, output hashes and safety decisions. The original reference and source
files are not required after a successful run because their pixel/file
identities are already committed by the recipe and report.

## Failure-closed branches

- missing or changed transaction report: reject;
- changed or missing delivered output: reject;
- composition recipe differs from run recipe: reject;
- report/result safety decision differs: reject;
- mixed `applied` and `identity-fallback` outputs: reject one batch plan;
- fallback batch requests reference colour or film effects: reject;
- applied batch lacks explicit research-baseline composition: reject;
- serialized research request, plan override or canonical identity drifts:
  reject.

The current algorithm remains unpromoted. Therefore ordinary product delivery
still falls back to identity and cannot silently add film effects. This leaf
does not promote the photographic algorithm or claim a film stock identity.

## Verification

- dedicated: `13 passed`;
- colour-match/preprocess focused: `174 passed, 909 deselected`;
- complete CPU collection: `1046 passed, 1 skipped, 36 failed`;
- compileall: pass;
- `git diff --check`: pass.

The 36 complete-collection failures are the unchanged isolated-worktree
baseline: ignored `outputs/` evidence is absent and several frozen byte hashes
see CRLF checkout bytes. No `src/color_match` test failed and no new failure
family appeared.

Implementation commit: `b44fc96` (`feat: bind composition to delivered matcher
runs`).

## External dependency boundary

- Main neuro-film remains active at stable HEAD `ed1dbb5`; U1-B is running and
  has no stable repeated W1 report. No main-task mutable file or report was
  consumed.
- Standalone D-PCT is idle and clean at `bd3ff70`. Its completed DNG/LibRaw
  evidence remains a media-ingress dependency only; no decoder or media code
  was copied.
