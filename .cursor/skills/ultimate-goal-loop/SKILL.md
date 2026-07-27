---
name: ultimate-goal-loop
description: >-
  Run the K-MCFM Ultimate stock-first Goal loop: refresh authorities and git,
  select one ready leaf, freeze contract, execute, verify, adjudicate, propagate,
  commit/push, update CURSOR_GOAL_STATE, then continue. Use whenever Goal status
  is ACTIVE, after stop-hook followups, or when resuming Ultimate research.
---

# Ultimate Goal Loop

## When to use

- `docs/drpt/CURSOR_GOAL_STATE.json` has `"status": "ACTIVE"`
- User asks to continue Ultimate / stock-first / FilmStyleSafe work
- A `stop` hook followup re-enters the agent

Use this project loop under the personal `goal-engine` lifecycle supervisor
and apply the personal `autonomous-engineering` skill during each worker round.
If the Goal Engine CLI doctor fails, use its agent-native loop; do not stop the
project to repair the supervisor.

## State machine (mandatory order)

1. **S0 REFRESH** — Re-read `AGENTS.md`, Goal state, tracker, `git status`/`log`, processes, ignored outputs. Detect parallel writers.
2. **S1 SELECT** — Choose one DoR-complete ready leaf; record why not the other ready leaves.
3. **S2 FREEZE** — Contract: objective, non-goals, inputs, outputs, splits, seeds, metrics, stop rules, claim ceiling. Commit/push contract before confirmatory results when the leaf is experimental.
4. **S3 EXECUTE** — Minimal modular change; one primary writer; no opportunistic refactors.
5. **S4 VERIFY** — Targeted tests, hashes/determinism/leakage, proportional full `.\.venv\Scripts\python.exe -m pytest -q`.
6. **S5 ADJUDICATE** — Separate facts / inferences / hypotheses; write results + decision + negative evidence + claim ceiling; never lower gates after seeing results.
7. **S6 PROPAGATE** — Parent/children/siblings/interfaces/docs/tracker/board/AGENT_LOG/Goal state.
8. **S7 COMMIT** — Scoped stage; narrow commit; push active branch; refresh Goal state from live git.
9. **S8 CONTINUE** — Pick next ready leaf; keep Goal `ACTIVE`; return to S0.

## Hard continue rule

Do not stop for: finished plan, finished harness, one commit, one push, one experiment, failed data source, algorithm failure, rejected candidate, K=1, no Oracle gain, ML path closed, temporarily no trainable pixels, negative leaf, or green tests.

## Visual protocol

If images are read, report autonomous visual evidence only. If not read, mark `pending vision adjudication`. Never invent population preference.

## Local vs cloud

Prefer This Computer when ignored outputs / local data matter. Cloud Agents do not see gitignored local evidence unless explicitly mirrored.

## Handoff

Update `docs/drpt/CURSOR_GOAL_STATE.json` every significant phase. On session cap, leave Goal `ACTIVE` and emit `GOAL_ACTIVE_REQUIRES_NEXT_INVOCATION`.
