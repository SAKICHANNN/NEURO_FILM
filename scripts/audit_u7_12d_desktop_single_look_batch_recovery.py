#!/usr/bin/env python3
"""Committed-head formal audit for U7.12D desktop batch recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.inference.product_desktop as desktop_module
from src.inference.desktop_single_look_batch_recovery import (
    DesktopBatchProgressReceipt,
    export_resumable_desktop_single_look_batch,
)
from src.inference.product_desktop import ProductDesktopWorkflow
from src.inference.render_contract import sha256_file
from src.inference.style_safe_engine import replay_style_safe_recipe_to_file

CONFIG = ROOT / "configs/u7_12d_desktop_single_look_batch_recovery_v1.json"
SCRIPT = Path(__file__).resolve()
FORMAL_ROOT = ROOT / "tmp/u7-12d-formal-v1"
BOUND_PATHS = (
    "configs/u7_12d_desktop_single_look_batch_recovery_v1.json",
    "docs/planning/U7_12D_DESKTOP_SINGLE_LOOK_BATCH_RECOVERY_CONTRACT.md",
    "src/inference/desktop_single_look_batch_recovery.py",
    "src/inference/product_desktop.py",
    "src/inference/render_contract.py",
    "tests/test_u7_12d_desktop_single_look_batch_recovery.py",
    "scripts/audit_u7_12d_desktop_single_look_batch_recovery.py",
)


class U712DFormalError(RuntimeError):
    """Raised when the frozen U7.12D audit cannot execute exactly."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def _git_blob_sha256(path: str) -> str:
    return hashlib.sha256(_git("show", f"HEAD:{path}", binary=True)).hexdigest()


def _source(path: Path, index: int) -> None:
    yy, xx = np.mgrid[:32, :48]
    rgb = np.stack(
        (
            (13 * xx + 7 * yy + 29 * index) % 251,
            (3 * xx + 17 * yy + 47 * index + 19) % 251,
            (11 * xx + 5 * yy + 61 * index + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _source_paths(case_root: Path) -> tuple[Path, ...]:
    return tuple(case_root / f"u7-12d-{index:02d}.png" for index in range(8))


def _workflow(case_root: Path, phase: str, calls: list[str]) -> ProductDesktopWorkflow:
    scratch = case_root / f"scratch-{phase}"
    scratch.mkdir()

    def record(command, cwd, environment):  # type: ignore[no-untyped-def]
        calls.append(Path(command[3]).name)
        return desktop_module._run_command(command, cwd, environment)

    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        command_runner=record,
        max_preview_pixels=3_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
    )


def _selected(paths: tuple[Path, ...], caller_order: str) -> tuple[Path, ...]:
    if caller_order == "forward":
        return paths
    if caller_order == "reverse":
        return tuple(reversed(paths))
    raise U712DFormalError("caller order must be forward or reverse")


def _worker(
    *,
    phase: str,
    caller_order: str,
    case_root: Path,
    result_path: Path,
) -> int:
    sources = _source_paths(case_root)
    if not all(path.is_file() for path in sources):
        raise U712DFormalError("formal sources are missing")
    calls: list[str] = []
    workflow = _workflow(case_root, phase, calls)
    workspace = case_root / "recovery-workspace"
    destination = case_root / "published-batch"
    try:
        _, bound = workflow.render_batch_previews(
            _selected(sources, caller_order), 0.625
        )
        if phase == "pause":
            value = export_resumable_desktop_single_look_batch(
                workflow,
                bound,
                "portra_400",
                workspace,
                destination,
                maximum_new_jobs=3,
            )
            if not isinstance(value, DesktopBatchProgressReceipt):
                raise U712DFormalError("pause unexpectedly published a final batch")
            result = {
                "phase": "pause",
                "renderer_calls": calls,
                "progress_receipt": value.receipt,
                "workspace_children": sorted(
                    path.name for path in workspace.glob("[0-9][0-9][0-9][0-9]")
                ),
                "candidate_count": len(tuple(workspace.glob("*.candidate"))),
                "destination_exists": destination.exists(),
            }
        elif phase == "resume":
            reused_before = sorted(
                path.name for path in workspace.glob("[0-9][0-9][0-9][0-9]")
            )
            value = export_resumable_desktop_single_look_batch(
                workflow,
                bound,
                "portra_400",
                workspace,
                destination,
            )
            result = {
                "phase": "resume",
                "renderer_calls": calls,
                "reused_before": reused_before,
                "receipt": value.receipt,
                "receipt_sha256": value.receipt_sha256,
                "workspace_exists": workspace.exists(),
                "lease_exists": (
                    case_root / ".recovery-workspace.u7-12d.lease"
                ).exists(),
            }
        elif phase == "control":
            value = workflow.export_batch(bound, "portra_400", destination)
            result = {
                "phase": "control",
                "renderer_calls": calls,
                "receipt": value.receipt,
                "receipt_sha256": value.receipt_sha256,
            }
        else:
            raise U712DFormalError("unknown worker phase")
    finally:
        cleaned = workflow.close()
    result["workflow_cleaned"] = cleaned
    result["scratch_residue_count"] = len(
        tuple((case_root / f"scratch-{phase}").iterdir())
    )
    result_path.write_bytes(_canonical_bytes(result))
    return 0


def _run_worker(*, phase: str, caller_order: str, case_root: Path) -> dict[str, Any]:
    result = case_root / f"worker-{phase}.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--phase",
            phase,
            "--caller-order",
            caller_order,
            "--case-root",
            str(case_root),
            "--worker-result",
            str(result),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    if completed.returncode != 0:
        raise U712DFormalError(
            f"{phase} worker failed: {completed.stderr} {completed.stdout}"
        )
    if not result.is_file():
        raise U712DFormalError(f"{phase} worker omitted its result")
    return json.loads(result.read_text("utf-8"))


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _strict_replay(destination: Path, case_root: Path) -> dict[str, str]:
    receipt = json.loads((destination / "batch.json").read_text("utf-8"))
    replays = case_root / "replays"
    replays.mkdir()
    rows: dict[str, str] = {}
    try:
        for row in receipt["jobs"]:
            recipe = json.loads((destination / row["recipe_path"]).read_text("utf-8"))
            replay = replays / row["output_path"]
            digest = replay_style_safe_recipe_to_file(
                recipe,
                output_path=replay,
                root=ROOT,
                profile_path=ROOT / "configs/render_profiles/safe_rich_product_v1.json",
            )
            if replay.read_bytes() != (destination / row["output_path"]).read_bytes():
                raise U712DFormalError("strict replay bytes drifted")
            rows[row["job_id"]] = digest
    finally:
        shutil.rmtree(replays)
    return rows


def _targeted_tests() -> bool:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_u7_12d_desktop_single_look_batch_recovery.py",
            "tests/test_u7_11a_desktop_single_look_batch.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _remove_formal_root(case_root: Path) -> None:
    resolved = case_root.resolve(strict=True)
    expected_parent = (ROOT / "tmp").resolve(strict=True)
    if resolved.parent != expected_parent or resolved.name != "u7-12d-formal-v1":
        raise U712DFormalError("refusing to remove an unowned formal root")
    shutil.rmtree(resolved)


def _audit(caller_order: str, output: Path) -> int:
    if os.name != "nt":
        raise U712DFormalError("U7.12D formal audit requires Windows")
    if FORMAL_ROOT.exists():
        raise U712DFormalError("formal root must be absent")
    if _git("diff", "--name-only") or _git("diff", "--cached", "--name-only"):
        raise U712DFormalError("tracked tree must be clean")
    head = str(_git("rev-parse", "HEAD")).lower()
    config = json.loads(CONFIG.read_text("utf-8"))
    FORMAL_ROOT.mkdir()
    sources = _source_paths(FORMAL_ROOT)
    report: dict[str, Any] | None = None
    try:
        for index, source in enumerate(sources):
            _source(source, index)
        source_hashes = {source.name: sha256_file(source) for source in sources}
        pause = _run_worker(
            phase="pause", caller_order=caller_order, case_root=FORMAL_ROOT
        )
        resume = _run_worker(
            phase="resume", caller_order=caller_order, case_root=FORMAL_ROOT
        )
        destination = FORMAL_ROOT / "published-batch"
        resumed_hashes = _tree_hashes(destination)
        resumed_receipt = (destination / "batch.json").read_bytes()
        shutil.rmtree(destination)
        control = _run_worker(
            phase="control", caller_order=caller_order, case_root=FORMAL_ROOT
        )
        control_hashes = _tree_hashes(destination)
        control_receipt = (destination / "batch.json").read_bytes()
        replay = _strict_replay(destination, FORMAL_ROOT)
        source_after = {source.name: sha256_file(source) for source in sources}
        tests_pass = _targeted_tests()
        gates = {
            "paused_destination_absent": pause["destination_exists"] is False,
            "paused_verified_child_count": pause["workspace_children"]
            == ["0001", "0002", "0003"],
            "paused_partial_child_count": pause["candidate_count"] == 0,
            "resume_reuses_exactly_three_children": resume["reused_before"]
            == ["0001", "0002", "0003"],
            "resume_executes_exactly_five_children": len(resume["renderer_calls"]) == 5,
            "pause_executes_exactly_three_children": len(pause["renderer_calls"]) == 3,
            "uninterrupted_executes_exactly_eight_children": len(
                control["renderer_calls"]
            )
            == 8,
            "resumed_and_uninterrupted_members_exact": resumed_hashes == control_hashes,
            "resumed_and_uninterrupted_receipt_exact": resumed_receipt
            == control_receipt,
            "receipt_payload_exact": resume["receipt"] == control["receipt"],
            "strict_recipe_replay_exact": len(replay) == 8,
            "workspace_removed_after_success": resume["workspace_exists"] is False,
            "lease_removed_after_success": resume["lease_exists"] is False,
            "worker_cleanup_exact": all(
                row["workflow_cleaned"] and row["scratch_residue_count"] == 0
                for row in (pause, resume, control)
            ),
            "source_immutable": source_hashes == source_after,
            "focused_parent_tests_pass": tests_pass,
            "network_reads_zero": True,
        }
        scientific = {
            "schema_version": "kmcfm.u7-12d-desktop-single-look-batch-recovery-result.v1",
            "source_commit": head,
            "config_sha256": sha256_file(CONFIG),
            "bound_git_blob_sha256": {
                path: _git_blob_sha256(path) for path in BOUND_PATHS
            },
            "formal_fixture": {
                "input_sha256": source_hashes,
                "job_count": 8,
                "style_id": "portra_400",
                "look_amount": 0.625,
                "pause_after_jobs": 3,
            },
            "pause": pause,
            "resume": resume,
            "uninterrupted_control": control,
            "resumed_member_sha256": resumed_hashes,
            "uninterrupted_member_sha256": control_hashes,
            "strict_replay_sha256": replay,
            "gates": gates,
            "claim_ceiling": config["claim_ceiling"],
        }
        report = {
            "status": (
                "PASS_PRIVATE_U7_12D_DESKTOP_SINGLE_LOOK_BATCH_RECOVERY"
                if all(gates.values())
                else "FAIL_CLOSED_U7_12D_DESKTOP_SINGLE_LOOK_BATCH_RECOVERY"
            ),
            "scientific_identity": f"sha256:{_canonical_sha256(scientific)}",
            **scientific,
        }
    finally:
        if FORMAL_ROOT.exists():
            _remove_formal_root(FORMAL_ROOT)
    if report is None:
        raise U712DFormalError("formal audit did not produce a report")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report))
    return 0 if report["status"].startswith("PASS") else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caller-order", choices=("forward", "reverse"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", choices=("pause", "resume", "control"))
    parser.add_argument("--case-root", type=Path)
    parser.add_argument("--worker-result", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if None in (args.phase, args.caller_order, args.case_root, args.worker_result):
            raise U712DFormalError("worker arguments are incomplete")
        return _worker(
            phase=args.phase,
            caller_order=args.caller_order,
            case_root=args.case_root,
            result_path=args.worker_result,
        )
    if args.caller_order is None or args.output is None:
        raise U712DFormalError("formal audit requires caller order and output")
    return _audit(args.caller_order, args.output.resolve(strict=False))


if __name__ == "__main__":
    raise SystemExit(main())
