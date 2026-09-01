#!/usr/bin/env python3
"""Committed-head formal audit for U7.15B visible desktop strength."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/u7_15b_desktop_visible_strength_quantization_v1.json"
CONTRACT_PATH = "docs/planning/U7_15B_DESKTOP_VISIBLE_STRENGTH_QUANTIZATION_CONTRACT.md"
UI_PATH = "src/inference/product_desktop_ui.py"
CORE_PATH = "src/inference/product_desktop.py"
TEST_PATH = "tests/test_u7_15b_desktop_visible_strength_quantization.py"
SCRIPT_PATH = "scripts/audit_u7_15b_desktop_visible_strength_quantization.py"
BOUND_PATHS = (
    CONFIG_PATH,
    CONTRACT_PATH,
    UI_PATH,
    CORE_PATH,
    TEST_PATH,
    SCRIPT_PATH,
)
CASES = {
    "visible-strength-focused": TEST_PATH,
    "u7-15a-representative-parent": (
        "tests/test_u7_15a_desktop_batch_representative_selection.py"
    ),
    "u7-14a-input-basis-clear-parent": (
        "tests/test_u7_14a_desktop_input_basis_preview.py::"
        "test_native_ui_shows_truthful_nonselectable_input_basis_and_clears_it"
    ),
    "u7-10a-native-parent": (
        "tests/test_u7_10a_product_desktop_input_workflow.py::"
        "test_native_window_covers_bounded_states_without_path_disclosure"
    ),
}


def _run(*args: str) -> bytes:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _git_blob(path: str, commit: str = "HEAD") -> bytes:
    return _run("git", "show", f"{commit}:{path}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _load_parent_ui(payload: bytes) -> types.ModuleType:
    module = types.ModuleType("src.inference._u7_15b_parent_ui")
    module.__package__ = "src.inference"
    module.__file__ = "<u7-15b-parent-ui>"
    exec(compile(payload, module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module


class _WorkflowProbe:
    def __init__(self) -> None:
        self.preview_state: Any | None = None
        self.calls: list[float] = []

    def render_batch_previews(
        self,
        _sources: object,
        amount: float,
        *,
        representative_path: Path | None = None,
    ) -> object:
        if representative_path is not None:
            raise AssertionError("single-input probe cannot choose a representative")
        self.calls.append(amount)
        return object()


def _ui_probe(parent_payload: bytes) -> dict[str, Any]:
    import tkinter as tk

    from src.inference import product_desktop_ui as current_ui

    parent_ui = _load_parent_ui(parent_payload)
    scratch_parent = ROOT / "tmp"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="u7_15b_probe_", dir=scratch_parent) as raw:
        source = Path(raw) / "source.png"
        source.write_bytes(b"u7-15b source identity only")
        root = tk.Tk()
        root.withdraw()
        parent_root = tk.Toplevel(root)
        current_root = tk.Toplevel(root)
        parent_root.withdraw()
        current_root.withdraw()
        parent_workflow = _WorkflowProbe()
        current_workflow = _WorkflowProbe()
        parent = parent_ui.build_product_desktop_app(
            parent_root, parent_workflow, initial_input=source
        )
        current = current_ui.build_product_desktop_app(
            current_root, current_workflow, initial_input=source
        )
        parent._background = lambda action, _success: action()
        current._background = lambda action, _success: action()
        try:
            parent.amount.set(0.654321)
            parent._amount_changed()
            parent_visible = str(parent.amount_label.cget("text"))
            parent_variable = parent.amount.get()
            parent.render_previews()

            current.amount.set(0.625)
            current._amount_changed()
            callback_visible = str(current.amount_label.cget("text"))
            callback_variable = current.amount.get()

            current.busy = False
            current.amount.set(0.654321)
            current.render_previews()
            current_visible = str(current.amount_label.cget("text"))
            current_variable = current.amount.get()
        finally:
            root.destroy()
    return {
        "parent": {
            "visible": parent_visible,
            "variable": parent_variable,
            "workflow_amount": parent_workflow.calls[-1],
        },
        "current_callback": {
            "visible": callback_visible,
            "variable": callback_variable,
        },
        "current_render_entry": {
            "visible": current_visible,
            "variable": current_variable,
            "workflow_amount": current_workflow.calls[-1],
        },
    }


def _case_order(order: str) -> tuple[str, ...]:
    names = tuple(sorted(CASES))
    return tuple(reversed(names)) if order == "reverse" else names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    commit = _run("git", "rev-parse", "HEAD").decode("ascii").strip()
    config = json.loads(_git_blob(CONFIG_PATH))
    parent_commit = str(config["parent_head"])
    parent_ui = _git_blob(UI_PATH, parent_commit)
    bindings = {path: _sha256(_git_blob(path)) for path in BOUND_PATHS}
    tracked_diff_clean = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        == 0
    )
    probe = _ui_probe(parent_ui)
    results: dict[str, dict[str, Any]] = {}
    for name in _case_order(args.order):
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", CASES[name]],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        results[name] = {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
        }

    parent_binding = config["parent_bindings"]
    gates = {
        "committed_source_bound": all(len(value) == 64 for value in bindings.values()),
        "tracked_diff_clean": tracked_diff_clean,
        "parent_ui_identity_exact": (
            _sha256(parent_ui) == parent_binding["product_desktop_ui"]["git_sha256"]
        ),
        "parent_mismatch_reproduced": probe["parent"]
        == {
            "visible": "65%",
            "variable": 0.654321,
            "workflow_amount": 0.654321,
        },
        "callback_label_variable_exact": probe["current_callback"]
        == {"visible": "63%", "variable": 0.63},
        "render_entry_rebind_exact": probe["current_render_entry"]
        == {"visible": "65%", "variable": 0.65, "workflow_amount": 0.65},
        "visible_strength_focused_pass": results["visible-strength-focused"]["passed"],
        "u7_15a_representative_behavior_exact": results["u7-15a-representative-parent"][
            "passed"
        ],
        "focused_parent_behavior_pass": all(
            result["passed"]
            for name, result in results.items()
            if name.startswith(("u7-10a", "u7-14a"))
        ),
        "product_desktop_core_unchanged": (
            bindings[CORE_PATH] == parent_binding["product_desktop"]["git_sha256"]
        ),
    }
    scientific = {
        "schema": "kmcfm.u7-15b-desktop-visible-strength-quantization-report.v1",
        "node_id": "U7.15B",
        "source_commit": commit,
        "bindings": bindings,
        "ui_probe": probe,
        "case_results": {key: results[key] for key in sorted(results)},
        "gates": gates,
        "claim": {
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
            "private_windows_tk_visible_strength_only": True,
            "workflow_continuous_api_changed": False,
            "renderer_changed": False,
            "look_or_profile_changed": False,
            "recipe_or_receipt_schema_changed": False,
            "representative_selection_changed": False,
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
            "public_release": False,
        },
    }
    report = {
        **scientific,
        "scientific_identity": _canonical_sha256(scientific),
        "status": (
            "PASS_PRIVATE_U7_15B_DESKTOP_VISIBLE_STRENGTH_QUANTIZATION"
            if all(gates.values())
            else "FAIL_CLOSED_U7_15B_DESKTOP_VISIBLE_STRENGTH_QUANTIZATION"
        ),
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = args.report.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        raise SystemExit("report destination must be absent") from None
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
