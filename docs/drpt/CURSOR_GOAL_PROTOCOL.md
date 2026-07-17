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
| Tracker / board / AGENT_LOG | project docs | Scientific and engineering truth |
| Commits / push | git | Durable evidence and resume points |

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
4. If hooks are unavailable, manually paste the Goal state's `next_action` or use CLI `--resume`.

## Stop hook behaviour

Continue only when:

- hook `status == completed`
- Goal `status == ACTIVE`
- `needs_human_authority == false`
- `next_action` non-empty
- `loop_count < max_session_stop_loops` (default 10; hooks.json `loop_limit` 10)

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
