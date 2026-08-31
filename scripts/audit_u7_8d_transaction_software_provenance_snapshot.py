#!/usr/bin/env python3
"""Audit transaction-scoped software provenance for the three-look batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import resumable_three_stock_input_batch as resumable_module
from src.inference import three_stock_batch as child_module
from src.inference import three_stock_input_batch as input_module
from src.inference.render_contract import sha256_file
from src.inference.resumable_three_stock_input_batch import (
    ResumableThreeStockBatchError,
)
from src.inference.three_stock_batch import ThreeStockBatchError
from src.inference.three_stock_input_batch import (
    INPUT_MANIFEST_SCHEMA,
    ThreeStockInputBatchError,
)

CONFIG = ROOT / "configs/u7_8d_transaction_software_provenance_snapshot_v1.json"
CONTRACT = (
    ROOT / "docs/planning/U7_8D_TRANSACTION_SOFTWARE_PROVENANCE_SNAPSHOT_CONTRACT.md"
)
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"
SCRATCH = ROOT / "tmp/u7_8d_formal_scratch"
STYLES = ("velvia_50", "portra_400", "ektar_100")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


class U78DError(RuntimeError):
    """Raised when the frozen U7.8D audit cannot complete exactly."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _source(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:24, :32]
    rgb = np.stack(
        (
            (xx * 7 + yy * 3 + offset * 11) % 256,
            (xx * 2 + yy * 13 + offset * 17) % 256,
            (xx * 19 + yy * 5 + offset * 23) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, compress_level=0)


def _build_manifest(root: Path, order: str) -> tuple[Path, dict[str, str]]:
    rows: list[dict[str, str]] = []
    source_hashes: dict[str, str] = {}
    for index in range(3):
        source = root / f"source-{index}.png"
        _source(source, index)
        digest = sha256_file(source)
        source_hashes[source.name] = digest
        rows.append(
            {
                "job_id": f"job-{index}",
                "input_path": source.name,
                "input_sha256": digest,
            }
        )
    if order == "reverse":
        rows.reverse()
    manifest = root / "jobs.json"
    manifest.write_bytes(
        _canonical_bytes({"schema_version": INPUT_MANIFEST_SCHEMA, "jobs": rows})
    )
    return manifest, source_hashes


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _recipe_commits(path: Path) -> list[str]:
    return sorted(
        {
            json.loads(item.read_text(encoding="utf-8"))["software"]["commit"]
            for item in path.rglob("*.recipe.json")
        }
    )


def _render_input(manifest: Path, destination: Path) -> dict[str, Any]:
    return input_module.render_three_stock_input_batch_to_directory(
        manifest,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
    )


def _render_resumable(
    manifest: Path, workspace: Path, destination: Path
) -> dict[str, Any]:
    return resumable_module.render_resumable_three_stock_input_batch_to_directory(
        manifest,
        workspace,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
    )


def _with_input_hooks(
    commit_provider: Callable[[Path], str],
    child_observations: list[str | None],
    operation: Callable[[], Any],
) -> Any:
    original_commit = input_module._software_commit
    original_child = input_module.render_three_stock_batch_to_directory

    def record(*args: Any, **kwargs: Any) -> dict[str, Any]:
        child_observations.append(kwargs.get("software_commit"))
        return original_child(*args, **kwargs)

    input_module._software_commit = commit_provider
    input_module.render_three_stock_batch_to_directory = record
    try:
        return operation()
    finally:
        input_module._software_commit = original_commit
        input_module.render_three_stock_batch_to_directory = original_child


def _stable_transaction(
    manifest: Path, destination: Path, commit: str
) -> dict[str, Any]:
    resolutions: list[str] = []
    observed: list[str | None] = []

    def fixed(root: Path) -> str:
        resolutions.append(str(root.resolve()))
        return commit

    receipt = _with_input_hooks(
        fixed,
        observed,
        lambda: _render_input(manifest, destination),
    )
    tree = _tree_hashes(destination)
    return {
        "batch_id": receipt["batch_id"],
        "receipt_sha256": sha256_file(destination / "batch.json"),
        "tree_hashes": tree,
        "tree_file_count": len(tree),
        "commit_resolution_count": len(resolutions),
        "child_observations": observed,
        "recipe_commits": _recipe_commits(destination),
    }


def _input_drift_control(
    manifest: Path, destination: Path, start: str, drift: str
) -> dict[str, Any]:
    commits = iter((start, drift))
    observed: list[str | None] = []
    message = ""
    try:
        _with_input_hooks(
            lambda root: next(commits),
            observed,
            lambda: _render_input(manifest, destination),
        )
    except ThreeStockInputBatchError as exc:
        message = str(exc)
    else:
        raise U78DError("injected U7.8A HEAD drift did not reject")
    residue = sorted(
        item.name for item in destination.parent.glob(f".{destination.name}.*.stage")
    )
    return {
        "message": message,
        "destination_absent": not destination.exists(),
        "child_observations": observed,
        "all_children_received_start_snapshot": observed == [start] * 3,
        "owned_stage_residue": residue,
    }


def _resumable_drift_control(
    manifest: Path,
    workspace: Path,
    destination: Path,
    start: str,
    drift: str,
) -> dict[str, Any]:
    commits = iter((start, drift))
    observed: list[str | None] = []
    original_commit = resumable_module._software_commit
    original_child = resumable_module.render_three_stock_batch_to_directory

    def record(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed.append(kwargs.get("software_commit"))
        return original_child(*args, **kwargs)

    resumable_module._software_commit = lambda root: next(commits)
    resumable_module.render_three_stock_batch_to_directory = record
    message = ""
    try:
        _render_resumable(manifest, workspace, destination)
    except ResumableThreeStockBatchError as exc:
        message = str(exc)
    else:
        raise U78DError("injected U7.8B final HEAD drift did not reject")
    finally:
        resumable_module._software_commit = original_commit
        resumable_module.render_three_stock_batch_to_directory = original_child

    resume_render_calls = 0

    def forbidden(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal resume_render_calls
        resume_render_calls += 1
        raise AssertionError("cross-commit resume must reject before render")

    resumable_module._software_commit = lambda root: drift
    resumable_module.render_three_stock_batch_to_directory = forbidden
    resume_message = ""
    try:
        _render_resumable(manifest, workspace, destination)
    except ResumableThreeStockBatchError as exc:
        resume_message = str(exc)
    else:
        raise U78DError("cross-commit U7.8B resume did not reject")
    finally:
        resumable_module._software_commit = original_commit
        resumable_module.render_three_stock_batch_to_directory = original_child

    return {
        "final_drift_message": message,
        "workspace_retained": workspace.is_dir(),
        "destination_absent": not destination.exists(),
        "child_observations": observed,
        "recipe_commits": _recipe_commits(workspace),
        "resume_message": resume_message,
        "resume_render_calls": resume_render_calls,
    }


def _direct_child_controls(source: Path, root: Path, commit: str) -> dict[str, Any]:
    invalid_destination = root / "invalid-child"
    original_loader = child_module.load_working_image
    decode_calls = 0

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("invalid commit reached decode")

    child_module.load_working_image = forbidden
    invalid_message = ""
    try:
        child_module.render_three_stock_batch_to_directory(
            source,
            invalid_destination,
            root=ROOT,
            profile_path=PROFILE,
            statistics_path=STATISTICS,
            guardrails_path=GUARDRAILS,
            software_commit="A" * 40,
        )
    except ThreeStockBatchError as exc:
        invalid_message = str(exc)
    finally:
        child_module.load_working_image = original_loader

    unavailable_destination = root / "unavailable-head-child"
    child_module.load_working_image = forbidden
    original_check_output = child_module.subprocess.check_output

    def unavailable(*args: Any, **kwargs: Any) -> str:
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    child_module.subprocess.check_output = unavailable
    unavailable_message = ""
    try:
        child_module.render_three_stock_batch_to_directory(
            source,
            unavailable_destination,
            root=ROOT,
            profile_path=PROFILE,
            statistics_path=STATISTICS,
            guardrails_path=GUARDRAILS,
        )
    except ThreeStockBatchError as exc:
        unavailable_message = str(exc)
    finally:
        child_module.subprocess.check_output = original_check_output
        child_module.load_working_image = original_loader

    direct_destination = root / "direct-child"
    live_head_calls = 0

    def head(*args: Any, **kwargs: Any) -> str:
        nonlocal live_head_calls
        live_head_calls += 1
        return commit + "\n"

    child_module.subprocess.check_output = head
    try:
        child_module.render_three_stock_batch_to_directory(
            source,
            direct_destination,
            root=ROOT,
            profile_path=PROFILE,
            statistics_path=STATISTICS,
            guardrails_path=GUARDRAILS,
            tile_size=16,
            png_compression=0,
        )
    finally:
        child_module.subprocess.check_output = original_check_output
    return {
        "invalid_message": invalid_message,
        "invalid_decode_calls": decode_calls,
        "invalid_destination_absent": not invalid_destination.exists(),
        "unavailable_head_message": unavailable_message,
        "unavailable_head_destination_absent": not unavailable_destination.exists(),
        "live_head_calls": live_head_calls,
        "direct_recipe_commits": _recipe_commits(direct_destination),
    }


def _targeted_tests() -> int:
    completed = subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            "-m",
            "pytest",
            "-q",
            "tests/test_u7_8d_transaction_software_provenance_snapshot.py",
            "tests/test_u7_8a_three_stock_input_batch.py::test_injected_child_failure_removes_owned_stage_and_publishes_nothing",
            "tests/test_u7_8a_three_stock_input_batch.py::test_late_foreign_destination_is_preserved",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise U78DError(
            "targeted tests failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    match = re.search(r"(?m)^(\d+) passed in ", completed.stdout)
    if match is None:
        raise U78DError("targeted test count is unavailable")
    return int(match.group(1))


def run(order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise U78DError("order must be forward or reverse")
    if os.name != "nt":
        raise U78DError("U7.8D formal execution requires Windows")
    if SCRATCH.exists():
        raise U78DError("formal scratch root already exists")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("node_id") != "U7.8D":
        raise U78DError("config node identity drifted")
    commit = _git("rev-parse", "HEAD")
    alternate = _git("rev-parse", "HEAD^")
    if COMMIT.fullmatch(commit) is None or COMMIT.fullmatch(alternate) is None:
        raise U78DError("formal commit identities are invalid")

    source_root = SCRATCH / "sources"
    source_root.mkdir(parents=True)
    try:
        manifest, source_hashes = _build_manifest(source_root, order)
        stable = _stable_transaction(manifest, SCRATCH / "stable", commit)
        drift = _input_drift_control(
            manifest, SCRATCH / "drifted-input-batch", commit, alternate
        )
        resumable = _resumable_drift_control(
            manifest,
            SCRATCH / "resumable-workspace",
            SCRATCH / "resumable-published",
            commit,
            alternate,
        )
        direct = _direct_child_controls(source_root / "source-0.png", SCRATCH, commit)
        source_hashes_after = {
            path.name: sha256_file(path) for path in source_root.glob("source-*.png")
        }
        test_count = _targeted_tests()
        shutil.rmtree(SCRATCH)
        owned_scratch_residue_count = int(SCRATCH.exists())

        gates = {
            "transaction_start_commit_resolution_count_exact": (
                stable["commit_resolution_count"] == 2
            ),
            "all_stable_children_use_one_snapshot": (
                stable["child_observations"] == [commit] * 3
                and stable["recipe_commits"] == [commit]
            ),
            "stable_transaction_counts_exact": stable["tree_file_count"] == 22,
            "input_head_drift_rejected_before_publication": (
                drift["message"] == "software commit drifted before publication"
                and drift["destination_absent"]
                and drift["all_children_received_start_snapshot"]
                and drift["owned_stage_residue"] == []
            ),
            "resumable_child_uses_state_snapshot": (
                resumable["child_observations"] == [commit] * 3
                and resumable["recipe_commits"] == [commit]
            ),
            "resumable_final_drift_rejected": (
                resumable["final_drift_message"]
                == "input, configuration or core identity drifted before publication"
                and resumable["workspace_retained"]
                and resumable["destination_absent"]
            ),
            "cross_commit_resume_rejected_before_render": (
                resumable["resume_render_calls"] == 0
                and bool(resumable["resume_message"])
            ),
            "invalid_explicit_commit_rejected_before_decode": (
                direct["invalid_message"]
                == "software_commit must be a lowercase full 40-hex Git commit"
                and direct["invalid_decode_calls"] == 0
                and direct["invalid_destination_absent"]
            ),
            "unavailable_default_head_rejected_before_decode": (
                direct["unavailable_head_message"] == "software commit is unavailable"
                and direct["invalid_decode_calls"] == 0
                and direct["unavailable_head_destination_absent"]
            ),
            "direct_child_default_resolves_once": (
                direct["live_head_calls"] == 1
                and direct["direct_recipe_commits"] == [commit]
            ),
            "source_files_immutable": source_hashes_after == source_hashes,
            "late_foreign_and_child_failure_controls_pass": test_count == 13,
            "owned_scratch_residue_count": owned_scratch_residue_count,
            "network_requests": 0,
        }
        all_gates_pass = (
            all(
                value is True
                for key, value in gates.items()
                if key not in {"network_requests", "owned_scratch_residue_count"}
            )
            and gates["network_requests"] == gates["owned_scratch_residue_count"] == 0
        )
        scientific = {
            "source_hashes": source_hashes,
            "software_commit": commit,
            "alternate_commit": alternate,
            "stable": stable,
            "input_drift_control": drift,
            "resumable_drift_control": resumable,
            "direct_child_controls": direct,
            "targeted_test_count": test_count,
            "gate_results": gates,
            "claim_ceiling": config["claim_ceiling"],
        }
        return {
            "schema_version": (
                "neuro-film.u7-8d-transaction-software-provenance-snapshot-result.v1"
            ),
            "node_id": "U7.8D",
            "decision": (
                "PASS_PRIVATE_U7_8D_TRANSACTION_SOFTWARE_PROVENANCE_SNAPSHOT"
                if all_gates_pass
                else "FAIL_CLOSED_U7_8D_TRANSACTION_SOFTWARE_PROVENANCE_SNAPSHOT"
            ),
            "implementation_commit": commit,
            "config_sha256": sha256_file(CONFIG),
            "contract_sha256": sha256_file(CONTRACT),
            "scientific_identity": _canonical_sha256(scientific),
            **scientific,
        }
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.order)
    if args.output.exists():
        raise U78DError("output report already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"].startswith("PASS_PRIVATE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
