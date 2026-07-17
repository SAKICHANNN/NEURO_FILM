from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.drpt.goal_state import (
    GoalStateError,
    load_and_validate_goal_state,
    objective_hash,
    validate_goal_state,
)

ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = ROOT / ".cursor" / "hooks" / "ultimate_goal_stop.py"
STATE_PATH = ROOT / "docs" / "drpt" / "CURSOR_GOAL_STATE.json"
HOOKS_JSON = ROOT / ".cursor" / "hooks.json"


def _load_hook_module():
    spec = importlib.util.spec_from_file_location("ultimate_goal_stop", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_state(**overrides):
    objective = (
        "Advance K-MCFM Ultimate stock-first real-film learning, FilmStyleSafe "
        "evaluation, and the deterministic non-generative product path under a "
        "hard severe-artifact veto while protecting existing code, data, commits "
        "and ignored outputs."
    )
    state = {
        "schema_version": 1,
        "goal_id": "KMCFM_ULTIMATE_STOCK_FIRST_CONTINUOUS_2026_07_17",
        "status": "ACTIVE",
        "objective": objective,
        "objective_hash": objective_hash(objective),
        "branch": "research/fivek-auto-optimize-cache",
        "head": "abc",
        "iteration": 0,
        "current_node": "ULT",
        "current_leaf": "test",
        "phase": "CONTINUE",
        "last_verified_commit": "abc",
        "last_full_test": {
            "command": "pytest",
            "passed": 1,
            "failed": 0,
            "timestamp": "2026-07-17T00:00:00Z",
        },
        "ready_leaves": ["U5.R1B"],
        "next_action": "continue U5.R1B",
        "running_processes": [],
        "waiting_until": null_fix(),
        "needs_human_authority": False,
        "authority_reason": None,
        "dirty_files": [],
        "evidence_paths": [],
        "max_session_stop_loops": 10,
        "last_update_utc": "2026-07-17T00:00:00Z",
    }
    state.update(overrides)
    return state


def null_fix():
    return None


def test_repo_goal_state_validates() -> None:
    payload = load_and_validate_goal_state(STATE_PATH)
    assert payload["status"] == "ACTIVE"
    assert payload["goal_id"] == "KMCFM_ULTIMATE_STOCK_FIRST_CONTINUOUS_2026_07_17"


def test_hooks_json_keeps_stop_entry() -> None:
    data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    assert data["version"] == 1
    stop = data["hooks"]["stop"]
    assert isinstance(stop, list) and stop
    assert "ultimate_goal_stop.py" in stop[0]["command"]
    assert stop[0]["loop_limit"] == 20


def test_objective_hash_mismatch_fails() -> None:
    state = _base_state(objective_hash="0" * 64)
    with pytest.raises(GoalStateError, match="objective_hash"):
        validate_goal_state(state)


def test_active_requires_next_action() -> None:
    state = _base_state(next_action=" ")
    with pytest.raises(GoalStateError, match="next_action"):
        validate_goal_state(state)


def test_complete_requires_parent_dod_proof() -> None:
    state = _base_state(status="COMPLETE", next_action="")
    with pytest.raises(GoalStateError, match="ultimate_parent_dod_proven"):
        validate_goal_state(state)


def test_stop_hook_followup_when_active_completed() -> None:
    hook = _load_hook_module()
    out = hook.decide({"status": "completed", "loop_count": 0}, _base_state())
    assert "followup_message" in out
    assert "Ultimate Goal remains ACTIVE" in out["followup_message"]


@pytest.mark.parametrize(
    "hook_input,state_overrides",
    [
        ({"status": "aborted", "loop_count": 0}, {}),
        ({"status": "error", "loop_count": 0}, {}),
        ({"status": "completed", "loop_count": 0}, {"status": "PAUSED"}),
        ({"status": "completed", "loop_count": 0}, {"status": "COMPLETE", "ultimate_parent_dod_proven": True, "next_action": ""}),
        ({"status": "completed", "loop_count": 0}, {"needs_human_authority": True, "authority_reason": "license"}),
        ({"status": "completed", "loop_count": 0}, {"next_action": ""}),
        ({"status": "completed", "loop_count": 10}, {}),
        ({"status": "completed", "loop_count": 0}, None),
    ],
)
def test_stop_hook_no_followup_cases(hook_input, state_overrides) -> None:
    hook = _load_hook_module()
    if state_overrides is None:
        state = None
    else:
        state = _base_state(**state_overrides)
        if state["status"] == "COMPLETE":
            # Bypass validator path; decide() only needs dict fields.
            pass
    assert hook.decide(hook_input, state) == {}


def test_stop_hook_malformed_state_fail_closed() -> None:
    hook = _load_hook_module()
    assert hook.decide({"status": "completed", "loop_count": 0}, ["not-a-dict"]) == {}


def test_rule_and_skill_exist() -> None:
    assert (ROOT / ".cursor" / "rules" / "ultimate-goal.mdc").is_file()
    assert (ROOT / ".cursor" / "skills" / "ultimate-goal-loop" / "SKILL.md").is_file()
    assert (ROOT / "docs" / "drpt" / "CURSOR_GOAL_PROTOCOL.md").is_file()
