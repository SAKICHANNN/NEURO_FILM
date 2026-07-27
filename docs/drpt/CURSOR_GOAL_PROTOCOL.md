# Cursor Ultimate Goal Protocol

Date: 2026-07-17

Goal ID: `KMCFM_ULTIMATE_STOCK_FIRST_CONTINUOUS_2026_07_17`

## Component roles

| Component | Path | Role |
|---|---|---|
| Persistent rule | `.cursor/rules/ultimate-goal.mdc` | Always-on invariants; does not store dynamic progress |
| Agent skill | `.cursor/skills/ultimate-goal-loop/SKILL.md` | On-demand Goal loop procedure |
| Stop hook | `.cursor/hooks.json` + `.cursor/hooks/ultimate_goal_stop.py` | Auto follow-up while Goal is ACTIVE |
| Goal state | `docs/drpt/CURSOR_GOAL_STATE.json` | Cross-chat checkpoint; validator in `src/drpt/goal_state.py` |
| Goal Engine objective | `GOAL.md` | Protected long-term objective and external completion rule |
| Worker progress | `.goal-engine/AGENT_PROGRESS.md` | Durable per-round context; not lifecycle authority |
| Tracker / board / AGENT_LOG | project docs | Scientific and engineering truth |
| Commits / push | git | Durable evidence and resume points |

## Personal Goal Engine

The personal Cursor skills are installed at:

- `C:\Users\hhvrf\.cursor\skills\goal-engine\SKILL.md`
- `C:\Users\hhvrf\.cursor\skills\autonomous-engineering\SKILL.md`

Use `goal-engine` as lifecycle supervisor and `autonomous-engineering` as the
worker discipline. Prefer the CLI implementation at
`C:\Users\hhvrf\Documents\goalmode` only when live `doctor` passes. On
2026-07-27 its complete repository verification passed, but native Windows had
no `agent`/`cursor-agent` executable on PATH; official CLI support is through
Windows WSL. Until a usable authenticated backend exists, run the skill's
agent-native inspect/verify/implement/re-verify loop with this repository's stop
hook. Do not pause Ultimate merely to bootstrap the supervisor.

Goal Engine itself never auto-pushes or merges. Scoped pushes on the active
research branch remain a separately owner-authorized project integration
action; merge/release/deploy remain forbidden.

## Status semantics

| Status | Meaning |
|---|---|
| `ACTIVE` | Continue; requires non-empty `next_action` |
| `PAUSED` | User requested pause; stop hook must not follow up |
| `BLOCKED` | Only when no legal ready leaf can proceed without new authority |
| `COMPLETE` | Entire Ultimate parent DoD proven (`ultimate_parent_dod_proven=true`) |

Ordinary leaf failure, K=1, no Oracle gain, ML close, or one finished commit must keep `ACTIVE` (or close that branch and select another leaf).

## Start / resume

1. Open Agent on **This Computer** with Grok 4.5 when local ignored data matter.
2. Read `AGENTS.md`, Goal state, tracker, `git status`.
3. Follow skill `ultimate-goal-loop`.
4. Use the personal `goal-engine` skill. If its CLI doctor fails, use its
   agent-native loop.
5. If hooks are unavailable, manually paste the Goal state's `next_action` or
   resume the local Cursor task.

`head` and `last_verified_commit` record the repository HEAD observed before the
state checkpoint is edited. A commit that contains the checkpoint necessarily
advances HEAD, so equality with the containing commit is neither possible nor
required. The explicit `head_semantics` field prevents a future agent from
mistaking this one-commit checkpoint relation for stale evidence.

## Stop hook behaviour

Continue only when:

- hook `status == completed`
- Goal `status == ACTIVE`
- `needs_human_authority == false`
- `next_action` non-empty
- `loop_count < max_session_stop_loops` (current maximum 30; hooks.json
  `loop_limit` 30)
- the state passes the complete project validator
- live Git has no merge/rebase/cherry-pick marker or unmerged index entry

Otherwise emit `{}`. Reaching the loop cap does **not** complete the Goal; the agent must report `GOAL_ACTIVE_REQUIRES_NEXT_INVOCATION`.

Hook unit tests live in `tests/test_cursor_goal_harness.py` and must not trigger a live infinite Agent loop.

## Cloud vs local

Cloud/Background Agents clone GitHub into an isolated environment and do **not** see local gitignored outputs (BlueNeg pilots, YFCC metadata, contact sheets, etc.). Prefer local Agent for data/experiment leaves. Cloud is acceptable for pure code/docs/tests when the branch is pushed and the leaf needs no ignored artifacts.

## Parallel writers

Never run two primary writers on the same files. If dirty files belong to another agent/user, stop auto-continue for those paths until ownership is clear.

## Handoff to Codex / next chat

1. Update Goal state from live git.
2. Ensure commits are pushed.
3. Leave `status=ACTIVE` with precise `next_action`.
4. Optionally refresh `docs/CURSOR_STOCK_FIRST_HANDOFF.md` for data-heavy leaves.
