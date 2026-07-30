"""Compare P161 encoded-render lifetime against the frozen P160 baseline."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

if __package__:
    from scripts.audit_reference_match_file_memory_v1 import (
        _add_worktree,
        _atomic_write_json,
        _remove_worktree,
        _sha256_file,
        preflight,
    )
    from scripts.audit_reference_match_ordered_batch_scale_memory_v1 import (
        generate_inputs,
        launch_worker,
    )
else:
    from audit_reference_match_file_memory_v1 import (
        _add_worktree,
        _atomic_write_json,
        _remove_worktree,
        _sha256_file,
        preflight,
    )
    from audit_reference_match_ordered_batch_scale_memory_v1 import (
        generate_inputs,
        launch_worker,
    )


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "reference_match_batch_render_lifetime_v1.json"
)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("node") != "P161":
        raise ValueError("config must be the P161 lifetime contract")
    for field in ("baseline_commit", "candidate_commit"):
        value = payload.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"{field} must be a lowercase full Git commit")
    if payload.get("execution_order") != [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]:
        raise ValueError("execution order differs from the frozen contract")
    image = payload.get("image")
    if (
        not isinstance(image, dict)
        or image.get("width") != 6000
        or image.get("height") != 4000
        or len(image.get("source_seeds", [])) != 3
    ):
        raise ValueError("image contract must contain three 24 MP sources")
    return payload


def evaluate_runs(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    variants = {
        variant: [row for row in runs if row["variant"] == variant]
        for variant in ("baseline", "candidate")
    }
    complete = bool(
        len(variants["baseline"]) == 2
        and len(variants["candidate"]) == 2
        and all(row["run_pass"] for row in runs)
    )
    artifact_fields = (
        "source_file_sha256",
        "output_sha256",
        "recipe_sha256",
        "normalized_report_sha256",
    )
    parity = {
        field: bool(
            complete
            and len(
                {
                    json.dumps(
                        row["worker_result"][field],
                        separators=(",", ":"),
                    )
                    for row in runs
                }
            )
            == 1
        )
        for field in artifact_fields
    }
    ordered_binding_pass = bool(
        complete
        and all(
            row["worker_result"]["source_file_sha256"]
            == config["_input_sha256"]["sources"]
            and [
                output["source_path"]
                for output in row["worker_result"]["normalized_report"]["outputs"]
            ]
            == [f"<INPUT>/source-{index}.png" for index in range(3)]
            for row in runs
        )
    )
    identity_pass = bool(
        complete
        and all(
            row["worker_result"]["safety_actions"]
            == ["identity-fallback"] * 3
            for row in runs
        )
    )
    baseline_rss = (
        statistics.median(
            row["monitor"]["peak_process_tree_rss_bytes"]
            for row in variants["baseline"]
        )
        if complete
        else None
    )
    candidate_rss = (
        statistics.median(
            row["monitor"]["peak_process_tree_rss_bytes"]
            for row in variants["candidate"]
        )
        if complete
        else None
    )
    baseline_wall = (
        statistics.median(
            row["worker_result"]["worker_wall_seconds"]
            for row in variants["baseline"]
        )
        if complete
        else None
    )
    candidate_wall = (
        statistics.median(
            row["worker_result"]["worker_wall_seconds"]
            for row in variants["candidate"]
        )
        if complete
        else None
    )
    rss_reduction = (
        baseline_rss - candidate_rss
        if baseline_rss is not None and candidate_rss is not None
        else None
    )
    rss_ratio = (
        candidate_rss / baseline_rss
        if baseline_rss not in (None, 0) and candidate_rss is not None
        else None
    )
    wall_ratio = (
        candidate_wall / baseline_wall
        if baseline_wall not in (None, 0) and candidate_wall is not None
        else None
    )
    gates = config["gates"]
    performance_pass = bool(
        rss_reduction is not None
        and rss_ratio is not None
        and wall_ratio is not None
        and rss_reduction
        >= int(gates["minimum_median_rss_reduction_bytes"])
        and rss_ratio
        <= float(gates["maximum_candidate_to_baseline_median_rss_ratio"])
        and wall_ratio
        <= float(gates["maximum_candidate_to_baseline_median_wall_ratio"])
    )
    return {
        "complete_run_matrix": complete,
        "parity": parity,
        "ordered_binding_pass": ordered_binding_pass,
        "all_identity_fallback_pass": identity_pass,
        "baseline_median_peak_process_tree_rss_bytes": baseline_rss,
        "candidate_median_peak_process_tree_rss_bytes": candidate_rss,
        "median_peak_rss_reduction_bytes": rss_reduction,
        "candidate_to_baseline_median_peak_rss_ratio": rss_ratio,
        "baseline_median_worker_wall_seconds": baseline_wall,
        "candidate_median_worker_wall_seconds": candidate_wall,
        "candidate_to_baseline_median_worker_wall_ratio": wall_ratio,
        "artifact_parity_pass": all(parity.values()),
        "performance_gate_pass": performance_pass,
        "automatic_pass": bool(
            complete
            and all(parity.values())
            and ordered_binding_pass
            and identity_pass
            and performance_pass
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def run_parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if output_dir.exists():
        raise FileExistsError("output_dir must be create-only")
    output_dir.mkdir(parents=True)
    preflight_result = preflight(config, output_dir)
    report: dict[str, Any] = {
        "schema_version": 1,
        "node": "P161",
        "config_sha256": _sha256_file(config_path),
        "baseline_commit": config["baseline_commit"],
        "candidate_commit": config["candidate_commit"],
        "preflight": preflight_result,
        "input_sha256": None,
        "runs": [],
        "gate_result": None,
        "worktree_cleanup": [],
        "claim_ceiling": config["claim_ceiling"],
    }
    if not preflight_result["passed"]:
        report["gate_result"] = {
            "automatic_pass": False,
            "run_deferred_by_preflight": True,
            "claim_ceiling": config["claim_ceiling"],
        }
        _atomic_write_json(output_dir / "report.json", report)
        return report

    input_dir = output_dir / "inputs"
    report["input_sha256"] = generate_inputs(config, input_dir)
    config["_input_sha256"] = report["input_sha256"]
    worktree_root = output_dir / "worktrees"
    worktree_root.mkdir()
    worktrees = {
        "baseline": worktree_root / "baseline",
        "candidate": worktree_root / "candidate",
    }
    added: list[Path] = []
    try:
        for variant, field in (
            ("baseline", "baseline_commit"),
            ("candidate", "candidate_commit"),
        ):
            _add_worktree(worktrees[variant], config[field])
            added.append(worktrees[variant])
        for index, variant in enumerate(config["execution_order"], start=1):
            run_dir = output_dir / "runs" / f"{index:02d}-{variant}"
            result_path = output_dir / "worker-results" / f"{index:02d}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            monitor = launch_worker(
                repo_root=worktrees[variant],
                input_dir=input_dir,
                run_dir=run_dir,
                result_path=result_path,
                interval=float(config["rss_sample_interval_seconds"]),
                timeout=float(config["worker_timeout_seconds"]),
                output_bit_depth=int(config["file_match"]["output_bit_depth"]),
            )
            worker_result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if monitor["result_exists"]
                else None
            )
            staging_count = (
                len(worker_result["staging_temporaries"])
                if worker_result is not None
                else None
            )
            expected_commit = config[f"{variant}_commit"]
            run_pass = bool(
                monitor["exit_code"] == 0
                and not monitor["timed_out"]
                and monitor["orphan_worker_count"]
                == int(config["gates"]["orphan_worker_count"])
                and not monitor["result_temporary_exists"]
                and worker_result is not None
                and worker_result["repo_head"] == expected_commit
                and staging_count
                == int(config["gates"]["staging_temporary_count"])
                and worker_result["worker_wall_seconds"]
                <= float(config["worker_timeout_seconds"])
            )
            report["runs"].append(
                {
                    "index": index,
                    "variant": variant,
                    "expected_commit": expected_commit,
                    "monitor": monitor,
                    "worker_result": worker_result,
                    "run_pass": run_pass,
                }
            )
        report["gate_result"] = evaluate_runs(config, report["runs"])
    finally:
        for path in reversed(added):
            report["worktree_cleanup"].append(_remove_worktree(path))
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "prune"],
            check=False,
            capture_output=True,
            text=True,
        )
        cleanup_pass = bool(
            len(report["worktree_cleanup"]) == len(added)
            and all(
                row["exit_code"] == 0 and not row["path_exists_after"]
                for row in report["worktree_cleanup"]
            )
        )
        if report.get("gate_result") is not None:
            report["gate_result"]["worktree_cleanup_pass"] = cleanup_pass
            report["gate_result"]["automatic_pass"] = bool(
                report["gate_result"]["automatic_pass"] and cleanup_pass
            )
        _atomic_write_json(output_dir / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_parent(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(report["gate_result"], indent=2))
    return 0 if report["gate_result"]["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
