"""Validate and hash Cursor Ultimate Goal checkpoint state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ALLOWED_STATUS = frozenset({"ACTIVE", "PAUSED", "COMPLETE", "BLOCKED"})
ALLOWED_PHASE = frozenset(
    {
        "RECON",
        "SELECT",
        "FREEZE",
        "EXECUTE",
        "VERIFY",
        "ADJUDICATE",
        "PROPAGATE",
        "COMMIT",
        "CONTINUE",
        "WAITING",
        "HANDOFF",
    }
)


class GoalStateError(ValueError):
    """Raised when Goal state schema or invariants fail closed."""


def objective_hash(objective: str) -> str:
    return hashlib.sha256(objective.encode("utf-8")).hexdigest()


def validate_goal_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed validation for docs/drpt/CURSOR_GOAL_STATE.json."""
    required = {
        "schema_version",
        "goal_id",
        "status",
        "objective",
        "objective_hash",
        "branch",
        "head",
        "iteration",
        "current_node",
        "current_leaf",
        "phase",
        "last_verified_commit",
        "last_full_test",
        "ready_leaves",
        "next_action",
        "running_processes",
        "waiting_until",
        "needs_human_authority",
        "authority_reason",
        "dirty_files",
        "evidence_paths",
        "max_session_stop_loops",
        "last_update_utc",
    }
    missing = sorted(required - set(state))
    if missing:
        raise GoalStateError(f"missing fields: {missing}")
    if int(state["schema_version"]) != 1:
        raise GoalStateError("unsupported schema_version")
    if state["status"] not in ALLOWED_STATUS:
        raise GoalStateError(f"invalid status: {state['status']!r}")
    if state["phase"] not in ALLOWED_PHASE:
        raise GoalStateError(f"invalid phase: {state['phase']!r}")
    if not isinstance(state["goal_id"], str) or not state["goal_id"]:
        raise GoalStateError("goal_id must be a non-empty string")
    if not isinstance(state["objective"], str) or len(state["objective"]) < 32:
        raise GoalStateError("objective must be a substantial string")
    expected = objective_hash(str(state["objective"]))
    if state["objective_hash"] != expected:
        raise GoalStateError("objective_hash mismatch")
    if not isinstance(state["iteration"], int) or state["iteration"] < 0:
        raise GoalStateError("iteration must be a non-negative int")
    if not isinstance(state["max_session_stop_loops"], int) or not (
        1 <= state["max_session_stop_loops"] <= 20
    ):
        raise GoalStateError("max_session_stop_loops must be in 1..20")
    if not isinstance(state["needs_human_authority"], bool):
        raise GoalStateError("needs_human_authority must be bool")
    if state["needs_human_authority"] and not state.get("authority_reason"):
        raise GoalStateError("authority_reason required when needs_human_authority")
    if state["status"] == "ACTIVE":
        if not isinstance(state["next_action"], str) or not state["next_action"].strip():
            raise GoalStateError("ACTIVE Goal requires non-empty next_action")
    for key in ("ready_leaves", "running_processes", "dirty_files", "evidence_paths"):
        if not isinstance(state[key], list):
            raise GoalStateError(f"{key} must be a list")
    test = state["last_full_test"]
    if not isinstance(test, Mapping):
        raise GoalStateError("last_full_test must be an object")
    for field in ("command", "passed", "failed", "timestamp"):
        if field not in test:
            raise GoalStateError(f"last_full_test missing {field}")
    if state["status"] == "COMPLETE" and state.get("ultimate_parent_dod_proven") is not True:
        raise GoalStateError(
            "COMPLETE requires ultimate_parent_dod_proven=true covering the full parent DoD"
        )
    if state["status"] == "BLOCKED" and not state.get("authority_reason"):
        raise GoalStateError("BLOCKED requires authority_reason")
    return {
        "passed": True,
        "goal_id": state["goal_id"],
        "status": state["status"],
        "objective_hash": expected,
    }


def load_and_validate_goal_state(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GoalStateError("goal state root must be an object")
    validate_goal_state(payload)
    return payload
