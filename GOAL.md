# K-MCFM Ultimate — Stock-First Real-Film Learning

## Objective

Autonomously advance K-MCFM Ultimate to its genuine parent Definition of Done:
an excellent, modular, deterministic, content-safe film imaging system that
maximizes obvious and appealing film-style salience subject to a hard
severe-artifact veto, with as many evidence-backed and distinguishable
stock-specific real-film-derived experts as obtainable evidence supports.
Historical or unknown-stock film remains a separate auxiliary lane and never
substitutes for a named stock.

The complete live scientific, engineering, data, rights and product contract is
defined by `AGENTS.md`, `docs/ULTIMATE_EXECUTION_TRACKER.md`,
`docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md`,
`docs/planning/STOCK_FIRST_REAL_FILM_PROGRAM_2026.md`,
`docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md`,
`docs/planning/LATENT_STOCK_MODE_PROGRAM_2026.md` and
`docs/drpt/CURSOR_GOAL_STATE.json`.

## Completion authority

The implementation worker does not decide completion. The Goal remains active
until all of the following are externally evidenced:

1. `docs/drpt/CURSOR_GOAL_STATE.json` has `status: COMPLETE` and
   `ultimate_parent_dod_proven: true`.
2. The Ultimate parent DoD in `docs/ULTIMATE_EXECUTION_TRACKER.md` is fully
   satisfied, with evidence paths and no remaining required leaf.
3. Required stock/data/rights/identifiability, Style-safe severe-artifact,
   product/runtime and release-boundary evidence is present at its honest claim
   ceiling.
4. The full repository verification suite passes and protected Goal assets
   remain intact.

Green tests alone do not complete this research Goal. One finished or failed
leaf, one commit, a closed data source, K=1, no Oracle gain, or an algorithm
family closing is not completion.

## Immutable constraints

- No generative image AI and no learned direct final-RGB renderer.
- ML may predict only bounded explicit parameters such as curves, positive
  matrices, LUT weights/grids, hard mode IDs, sparse retrieval weights or
  physical-effect parameters.
- `film_stock_id` is the highest real-film expert class. Content, scene colour,
  source, uploader, geometry, scanner, borders, resolution, era and filenames
  cannot be used as stock shortcuts.
- Unpaired digital-to-film operator identification remains unresolved. Do not
  upgrade appearance matching, distribution matching or pseudo-pairs to
  calibrated stock response.
- Latent within-stock modes are a data-gated hypothesis; K=1 is a formal
  outcome. Without independent metadata, use Mode A/B/C and keep physical
  interpretations unknown or hypothesis-only.
- Apply the severe-artifact veto before maximizing style/appeal. Never weaken a
  frozen gate after viewing confirmatory results.
- Preserve all existing code, data, ignored outputs, reports, manifests,
  commits and user changes. Never delete or move unrelated files.
- No paid cloud/GPU, 290GB archive, purchase, recruiting, external messaging,
  secrets/IAM/billing, merge, release, deploy, public publication or license
  decision without new explicit authority.
- Local CPU and RTX 5070 Ti Laptop 12GB research computation are allowed.
- Scoped commits and pushes to the active research branch are authorized by
  the owner. Merge, release and deploy are not.

## Autonomous loop

Use the personal Cursor `goal-engine` skill as supervisor and
`autonomous-engineering` as the worker discipline. Prefer Goal Engine CLI only
when its live `doctor` passes. Otherwise use its agent-native
inspect -> verify -> implement -> re-verify loop together with the repository
stop hook.

Each verified leaf must:

1. refresh live repository/process/evidence state;
2. select a DoR-complete legal leaf;
3. freeze its contract before confirmatory evidence;
4. implement the smallest modular change;
5. run focused and proportional full verification;
6. adjudicate facts, inference, hypotheses and claim ceiling;
7. propagate to authorities and `docs/drpt/AGENT_LOG.md`;
8. make a scoped commit and push under the owner's project-specific authority;
9. update Goal/progress state and immediately continue to the next ready leaf.

The worker must update `.goal-engine/AGENT_PROGRESS.md` every round. The
authoritative lifecycle state remains `docs/drpt/CURSOR_GOAL_STATE.json`.

## External verification

Run these without weakening or deleting them:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
.\.venv\Scripts\python.exe -c "import json, pathlib, sys; p=json.loads(pathlib.Path('docs/drpt/CURSOR_GOAL_STATE.json').read_text(encoding='utf-8')); sys.exit(0 if p.get('status') == 'COMPLETE' and p.get('ultimate_parent_dod_proven') is True else 1)"
```

The first two commands establish repository health. The third is intentionally
nonzero while the long-term Goal is active and is the completion gate, not a
failure to be bypassed.
