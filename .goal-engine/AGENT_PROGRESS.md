# K-MCFM Ultimate Goal Engine Progress

This is durable worker context for Cursor Goal Engine rounds. Lifecycle truth
lives in `docs/drpt/CURSOR_GOAL_STATE.json`; scientific and engineering truth
lives in the active project authorities and `docs/drpt/AGENT_LOG.md`.

## Completed

- Goal Engine project at `C:\Users\hhvrf\Documents\goalmode` was independently
  verified on 2026-07-27: formatting, lint, typecheck, 12 unit tests,
  11 integration tests, one end-to-end test, build and package boundaries pass.
- Personal Cursor skills `goal-engine` and `autonomous-engineering` are
  installed under `C:\Users\hhvrf\.cursor\skills`.
- The project already has an always-on Cursor rule, an Ultimate Goal loop skill,
  a stop hook and validated dynamic Goal state.
- Codex completed and pushed U5.R2Z0 result propagation through commit
  `ab24589`; live Git and the latest AGENT_LOG supersede this checkpoint.

## Evidence

- `GOAL.md`
- `AGENTS.md`
- `docs/drpt/CURSOR_GOAL_STATE.json`
- `docs/drpt/CURSOR_GOAL_PROTOCOL.md`
- `.cursor/skills/ultimate-goal-loop/SKILL.md`
- `docs/CURSOR_GROK45_ULTIMATE_AUTONOMOUS_HANDOFF_20260727.md`

## Remaining failures

- The Ultimate parent completion verifier must remain nonzero while Goal state
  is ACTIVE.
- U5.R2Z0 is closed: ClassNeg has evaluator Oracle value but its frozen
  photometric selector fails, while Velvia has no qualifying Oracle. U5.R2Z1
  is ready only for contract freeze; refresh live state before acting.
- Native Windows had no `agent`/`cursor-agent` executable on PATH. Official
  Cursor CLI support is via macOS, Linux or Windows WSL, so Goal Engine CLI
  `doctor` cannot yet pass natively. Use the personal goal-engine skill's
  agent-native loop unless live detection proves this changed.

## Decisions

- Run Cursor Agent on This Computer with Grok 4.5 High because local ignored
  FilmSet/BlueNeg/YFCC evidence and the local GPU are required.
- Do not use Cursor Background/Cloud Agents or Automations for this Goal.
- Use `dev-research-reliability` semantics as the single writing workflow and
  the other Codex skill contracts as read-only disciplines.
- Project-specific scoped pushes are owner-authorized supervisor integration
  actions; Goal Engine itself must not auto-merge, auto-push or rewrite history.

## Failed approaches

- Do not treat a green pytest suite as Ultimate completion.
- Do not invoke the Goal Engine CLI against neuro_film until `doctor` passes.
- Do not duplicate a live GPU/download process after a crash without proving
  the prior process is dead or invalid.
