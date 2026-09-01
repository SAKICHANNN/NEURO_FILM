#!/usr/bin/env python3
"""Committed-head formal audit for the U7.18A desktop export policy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.open_product_desktop import (
    DESKTOP_EXPORT_TILE_SIZE,
    DESKTOP_EXPORT_TILE_WORKERS,
)
from src.inference.product_desktop import (
    DesktopFinishingEffects,
    ProductDesktopWorkflow,
    sha256_file,
)
from src.inference.style_safe_engine import replay_style_safe_recipe_to_file
from src.inference.three_stock_preview import render_three_stock_previews_to_directory

CONFIG_PATH = "configs/u7_18a_desktop_fast_export_policy_v1.json"
CONTRACT_PATH = "docs/planning/U7_18A_DESKTOP_FAST_EXPORT_POLICY_CONTRACT.md"
CORE_PATH = "src/inference/product_desktop.py"
LAUNCHER_PATH = "scripts/open_product_desktop.py"
FOCUSED_TEST_PATH = "tests/test_u7_18a_desktop_fast_export_policy.py"
SCRIPT_PATH = "scripts/audit_u7_18a_desktop_fast_export_policy.py"
BOUND_PATHS = (
    CONFIG_PATH,
    CONTRACT_PATH,
    CORE_PATH,
    LAUNCHER_PATH,
    FOCUSED_TEST_PATH,
    SCRIPT_PATH,
)
BEHAVIOR_CASES = {
    "u7-10a-input": "tests/test_u7_10a_product_desktop_input_workflow.py",
    "u7-11a-batch": "tests/test_u7_11a_desktop_single_look_batch.py",
    "u7-12b-format": "tests/test_u7_12b_desktop_single_photo_output_format.py",
    "u7-12c-preview": "tests/test_u7_12c_desktop_display_native_preview.py",
    "u7-12g-batch-format": "tests/test_u7_12g_desktop_batch_output_format.py",
    "u7-14a-input-basis": "tests/test_u7_14a_desktop_input_basis_preview.py",
    "u7-16a-detail": "tests/test_u7_16a_desktop_exact_detail_inspection.py",
    "u7-17a-effects": "tests/test_u7_17a_desktop_bounded_finishing_effects.py",
    "u7-18a-policy": FOCUSED_TEST_PATH,
}
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"


def _run(*args: str) -> bytes:
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True).stdout


def _git_blob(path: str, commit: str = "HEAD") -> bytes:
    return _run("git", "show", f"{commit}:{path}")


def _git_oid(path: str, commit: str = "HEAD") -> str:
    return _run("git", "rev-parse", f"{commit}:{path}").decode("ascii").strip()


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


def _tree_rss(process: psutil.Process) -> int:
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


class MeasuringRunner:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(
        self,
        command: tuple[str, ...],
        cwd: Path,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        started = time.perf_counter()
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        process = psutil.Process(child.pid)
        peak = 0
        while child.poll() is None:
            peak = max(peak, _tree_rss(process))
            time.sleep(0.02)
        _, stderr = child.communicate()
        wall = time.perf_counter() - started
        self.rows.append(
            {
                "command": list(command),
                "wall_seconds": wall,
                "peak_process_tree_rss_bytes": peak,
                "returncode": child.returncode,
            }
        )
        return subprocess.CompletedProcess(
            command,
            int(child.returncode),
            stdout="",
            stderr=stderr,
        )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _icc_sha256(path: Path) -> str:
    with Image.open(path) as image:
        return _sha256(image.info.get("icc_profile", b""))


def _decoded_rgb16_sha256(path: Path) -> str:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype.name != "uint16" or decoded.ndim != 3:
        raise RuntimeError("formal output is not an RGB16 PNG")
    return _sha256(decoded[..., ::-1].tobytes())


def _normalized_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(recipe)
    normalized["output"]["path"] = "<OUTPUT>"
    normalized["output"]["sha256"] = "<OUTPUT_SHA256>"
    return normalized


def _preview_capture(target: list[dict[str, Any]]):
    def render(*args: Any, **kwargs: Any) -> dict[str, Any]:
        target.append(
            {
                "tile_size": kwargs["tile_size"],
                "tile_workers": kwargs["tile_workers"],
                "png_compression": kwargs["png_compression"],
            }
        )
        return render_three_stock_previews_to_directory(*args, **kwargs)

    return render


def _run_arm(
    *,
    config: dict[str, Any],
    case_root: Path,
    index: int,
    arm: str,
) -> dict[str, Any]:
    source = (ROOT / config["primary"]["input_relative_path"]).resolve(strict=True)
    policy = config["policies"][f"{arm}_export"]
    workflow_root = case_root / f"workflow-{index:02d}-{arm}"
    workflow_root.mkdir()
    preview_calls: list[dict[str, Any]] = []
    runner = MeasuringRunner()
    workflow = ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=workflow_root,
        python_executable=Path(sys.executable),
        command_runner=runner,
        preview_renderer=_preview_capture(preview_calls),
        max_preview_pixels=100_000,
        tile_size=int(config["policies"]["preview"]["tile_size"]),
        tile_workers=int(config["policies"]["preview"]["tile_workers"]),
        export_tile_size=int(policy["tile_size"]),
        export_tile_workers=int(policy["tile_workers"]),
        png_compression=int(config["policies"]["png_compression"]),
    )
    output = case_root / f"{index:02d}-{arm}.png"
    replay = case_root / f"{index:02d}-{arm}.replay.png"
    effects = DesktopFinishingEffects(**config["primary"]["effects"])
    try:
        workflow.render_previews(source, float(config["primary"]["look_amount"]))
        receipt = workflow.export(
            str(config["primary"]["style_id"]),
            output,
            output_format_id=str(config["primary"]["output_format_id"]),
            effects=effects,
        )
        if len(runner.rows) != 1:
            raise RuntimeError("formal export command count drifted")
        command_row = runner.rows[0]
        command = command_row["command"]
        if not isinstance(command, list):
            raise TypeError("formal command recording drifted")
        output_sha = sha256_file(output)
        recipe = json.loads(receipt.recipe_path.read_text(encoding="utf-8"))
        replay_sha = replay_style_safe_recipe_to_file(
            recipe,
            profile_path=PROFILE,
            output_path=replay,
            root=ROOT,
        )
        row = {
            "arm": arm,
            "policy": {
                "tile_size": int(_option(command, "--tile-size")),
                "tile_workers": int(_option(command, "--tile-workers")),
                "png_compression": int(_option(command, "--png-compression")),
            },
            "preview_calls": preview_calls,
            "wall_seconds": command_row["wall_seconds"],
            "peak_process_tree_rss_bytes": command_row["peak_process_tree_rss_bytes"],
            "output_sha256": output_sha,
            "decoded_rgb16_sha256": _decoded_rgb16_sha256(output),
            "icc_sha256": _icc_sha256(output),
            "normalized_recipe_sha256": _canonical_sha256(_normalized_recipe(recipe)),
            "recipe_effects": recipe["render"]["effects"],
            "replay_sha256": replay_sha,
            "replay_output_sha256": sha256_file(replay),
        }
        replay.unlink()
        receipt.output_path.unlink()
        receipt.recipe_path.unlink()
    finally:
        closed = workflow.close()
    if not closed:
        raise RuntimeError("formal preview workspace ownership drifted")
    workflow_root.rmdir()
    return row


def _case_order(order: str) -> tuple[str, ...]:
    names = tuple(sorted(BEHAVIOR_CASES))
    return tuple(reversed(names)) if order == "reverse" else names


def _run_behavior_cases(order: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name in _case_order(order):
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", BEHAVIOR_CASES[name]],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        failures = sorted(
            line.strip()
            for line in (completed.stdout + "\n" + completed.stderr).splitlines()
            if line.startswith(("FAILED ", "ERROR "))
        )
        results[name] = {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "failure_lines": failures,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    commit = _run("git", "rev-parse", "HEAD").decode("ascii").strip()
    config = json.loads(_git_blob(CONFIG_PATH))
    source = (ROOT / config["primary"]["input_relative_path"]).resolve(strict=True)
    source_before = sha256_file(source)
    bindings = {
        path: {
            "git_blob": _git_oid(path),
            "bytes": len(_git_blob(path)),
            "sha256": _sha256(_git_blob(path)),
        }
        for path in BOUND_PATHS
    }
    tracked_clean = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        == 0
    )
    parent_exact = all(
        _git_oid(row["path"], str(config["parent_head"])) == row["git_blob"]
        and _sha256(_git_blob(row["path"], str(config["parent_head"]))) == row["sha256"]
        for row in config["parent_bindings"].values()
    )
    sequence = tuple(config["formal"][f"{args.order}_sequence"])
    scratch_parent = (ROOT / "tmp").resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix=f"u7_18a_{args.order}_", dir=scratch_parent
    ) as raw:
        case_root = Path(raw)
        rows = [
            _run_arm(
                config=config,
                case_root=case_root,
                index=index,
                arm=str(arm),
            )
            for index, arm in enumerate(sequence)
        ]
        owned_residue_zero = not any(case_root.iterdir())
    cases = _run_behavior_cases(args.order)
    source_after = sha256_file(source)

    baseline_rows = [row for row in rows if row["arm"] == "baseline"]
    candidate_rows = [row for row in rows if row["arm"] == "candidate"]
    baseline_median = statistics.median(
        float(row["wall_seconds"]) for row in baseline_rows
    )
    candidate_median = statistics.median(
        float(row["wall_seconds"]) for row in candidate_rows
    )
    baseline_peak = max(
        int(row["peak_process_tree_rss_bytes"]) for row in baseline_rows
    )
    candidate_peak = max(
        int(row["peak_process_tree_rss_bytes"]) for row in candidate_rows
    )
    wall_ratio = candidate_median / baseline_median
    output_hashes = {str(row["output_sha256"]) for row in rows}
    decoded_hashes = {str(row["decoded_rgb16_sha256"]) for row in rows}
    icc_hashes = {str(row["icc_sha256"]) for row in rows}
    recipe_hashes = {str(row["normalized_recipe_sha256"]) for row in rows}
    replay_exact = all(
        row["replay_sha256"] == row["replay_output_sha256"] == row["output_sha256"]
        for row in rows
    )
    expected_effects = {
        "grain": {"strength": 0.05, "seed": 7, "color": True},
        "halation": {
            "strength": 0.15,
            "model": "simple",
            "preset": None,
            "control_mode": "locked",
            "resolved_parameters": None,
        },
        "dust": {"strength": 0.02, "seed": 24},
    }
    policies = config["policies"]
    gates = {
        "committed_source_bound": all(
            len(row["git_blob"]) == 40 and len(row["sha256"]) == 64
            for row in bindings.values()
        ),
        "tracked_diff_clean": tracked_clean,
        "all_parent_git_objects_exact": parent_exact,
        "source_immutable": (
            source_before == config["primary"]["input_sha256"] == source_after
        ),
        "launcher_policy_exact": (
            DESKTOP_EXPORT_TILE_SIZE == policies["candidate_export"]["tile_size"]
            and DESKTOP_EXPORT_TILE_WORKERS
            == policies["candidate_export"]["tile_workers"]
        ),
        "preview_policy_unchanged": all(
            row["preview_calls"]
            == [
                {
                    **policies["preview"],
                    "png_compression": policies["png_compression"],
                }
            ]
            for row in rows
        ),
        "export_policies_exact": all(
            row["policy"]
            == {
                **policies[f"{row['arm']}_export"],
                "png_compression": policies["png_compression"],
            }
            for row in rows
        ),
        "output_png16_bytes_exact": len(output_hashes) == 1,
        "decoded_rgb16_exact": len(decoded_hashes) == 1,
        "icc_exact": len(icc_hashes) == 1,
        "normalized_recipe_semantics_exact": len(recipe_hashes) == 1,
        "maximum_effect_recipe_exact": all(
            row["recipe_effects"] == expected_effects for row in rows
        ),
        "strict_replay_exact": replay_exact,
        "candidate_repeat_exact": len(
            {str(row["output_sha256"]) for row in candidate_rows}
        )
        == 1,
        "wall_ratio": wall_ratio
        <= config["gates"]["maximum_candidate_to_baseline_median_wall_ratio"],
        "rss_increase_bounded": candidate_peak
        <= baseline_peak + config["gates"]["maximum_candidate_rss_increase_bytes"],
        "absolute_rss_bounded": candidate_peak
        <= config["gates"]["maximum_candidate_peak_process_tree_rss_bytes"],
        "all_isolated_behavior_suites_pass": all(
            row["passed"] for row in cases.values()
        ),
        "owned_residue_zero": owned_residue_zero,
    }
    scientific = {
        "schema": "kmcfm.u7-18a-desktop-fast-export-policy-report.v1",
        "node_id": "U7.18A",
        "source_commit": commit,
        "bindings": bindings,
        "source_sha256": source_before,
        "policies": policies,
        "output_sha256": sorted(output_hashes),
        "decoded_rgb16_sha256": sorted(decoded_hashes),
        "icc_sha256": sorted(icc_hashes),
        "normalized_recipe_sha256": sorted(recipe_hashes),
        "case_results": {key: cases[key] for key in sorted(cases)},
        "gates": gates,
        "claim": config["claims"],
    }
    report = {
        "scientific": scientific,
        "scientific_sha256": _canonical_sha256(scientific),
        "observations": {
            "order": args.order,
            "rows": rows,
            "baseline_median_seconds": baseline_median,
            "candidate_median_seconds": candidate_median,
            "candidate_to_baseline_wall_ratio": wall_ratio,
            "baseline_peak_process_tree_rss_bytes": baseline_peak,
            "candidate_peak_process_tree_rss_bytes": candidate_peak,
            "candidate_rss_increase_bytes": candidate_peak - baseline_peak,
        },
        "status": (
            "PASS_PRIVATE_U7_18A_DESKTOP_FAST_EXPORT_POLICY"
            if all(gates.values())
            else "FAIL_CLOSED_U7_18A_DESKTOP_FAST_EXPORT_POLICY"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
