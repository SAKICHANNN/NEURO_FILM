#!/usr/bin/env python3
"""Cursor stop hook: continue Ultimate Goal when still ACTIVE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "docs" / "drpt" / "CURSOR_GOAL_STATE.json"
DEFAULT_MAX_LOOPS = 10
FOLLOWUP = (
    "Ultimate Goal remains ACTIVE. Re-read AGENTS.md, "
    "docs/drpt/CURSOR_GOAL_STATE.json, the tracker and current git/process state. "
    "Do not merely report. Continue the current legal ready leaf from next_action; "
    "if that leaf closed, perform change propagation, commit/push, select the next "
    "legal ready leaf and continue. Do not stop at a commit, negative result, K=1, "
    "no Oracle gain or one failed data source."
)


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))
    sys.stdout.flush()


def _empty() -> None:
    _emit({})


def decide(hook_input: dict, state: dict | None) -> dict:
    """Return stop-hook stdout payload. Pure function for unit tests."""
    status = str(hook_input.get("status") or "")
    if status != "completed":
        return {}
    if not isinstance(state, dict):
        return {}
    if state.get("status") != "ACTIVE":
        return {}
    if state.get("needs_human_authority") is True:
        return {}
    next_action = state.get("next_action")
    if not isinstance(next_action, str) or not next_action.strip():
        return {}
    try:
        loop_count = int(hook_input.get("loop_count", 0))
    except (TypeError, ValueError):
        return {}
    if loop_count < 0:
        return {}
    max_loops = state.get("max_session_stop_loops", DEFAULT_MAX_LOOPS)
    try:
        max_loops_i = int(max_loops)
    except (TypeError, ValueError):
        return {}
    if loop_count >= max_loops_i:
        return {}
    dirty = state.get("dirty_files")
    if isinstance(dirty, list) and any(
        str(item).endswith(".REBASE_HEAD") or "CONFLICT" in str(item) for item in dirty
    ):
        return {}
    return {"followup_message": FOLLOWUP}


def main() -> int:
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
        if not isinstance(hook_input, dict):
            _empty()
            return 0
        if not STATE_PATH.is_file():
            _empty()
            return 0
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        _emit(decide(hook_input, state if isinstance(state, dict) else None))
        return 0
    except Exception:
        # Fail open for hook runtime errors: never force an infinite loop.
        _empty()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
