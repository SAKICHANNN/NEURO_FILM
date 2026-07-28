"""Run the frozen P159 24 MP reference-match file-path scale audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from scripts.audit_reference_match_file_memory_v1 import (
    _add_worktree,
    _atomic_write_json,
    _remove_worktree,
    _sha256_file,
    generate_inputs,
    launch_worker,
    preflight,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "reference_match_target_scale_memory_v1.json"


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("node") != "P159":
        raise ValueError("config must be the P159 target-scale contract")
    commit = payload.get("candidate_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError("candidate_commit must be a lowercase full Git commit")
    image = payload.get("image")
    if (
        not isinstance(image, dict)
        or image.get("width") != 6000
        or image.get("height") != 4000
    ):
        raise ValueError("image geometry differs from the frozen 24 MP contract")
    if payload.get("repeat_count") != 2:
        raise ValueError("repeat_count differs from the frozen contract")
    return payload


def evaluate_runs(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    complete = bool(len(runs) == 2 and all(row["run_pass"] for row in runs))
    fields = (
        "output_sha256",
        "recipe_sha256",
        "normalized_report_sha256",
    )
    parity = {
        field: bool(
            complete
            and len({row["worker_result"][field] for row in runs}) == 1
        )
        for field in fields
    }
    peaks = [row["monitor"]["peak_process_tree_rss_bytes"] for row in runs]
    peak_max = max(peaks) if complete else None
    peak_min = min(peaks) if complete else None
    repeat_ratio = (
        peak_max / peak_min
        if peak_max is not None and peak_min not in (None, 0)
        else None
    )
    actions = {
        row["worker_result"]["safety_action"]
        for row in runs
        if row.get("worker_result") is not None
    }
    gates = config["gates"]
    resource_pass = bool(
        peak_max is not None
        and repeat_ratio is not None
        and peak_max <= int(gates["maximum_peak_process_tree_rss_bytes"])
        and repeat_ratio <= float(gates["maximum_peak_rss_repeat_ratio"])
    )
    artifact_pass = all(parity.values())
    identity_pass = actions == {"identity-fallback"}
    return {
        "complete_run_matrix": complete,
        "parity": parity,
        "safety_actions": sorted(actions),
        "peak_process_tree_rss_bytes": peaks,
        "maximum_peak_process_tree_rss_bytes": peak_max,
        "minimum_peak_process_tree_rss_bytes": peak_min,
        "peak_rss_repeat_ratio": repeat_ratio,
        "artifact_parity_pass": artifact_pass,
        "identity_fallback_pass": identity_pass,
        "resource_gate_pass": resource_pass,
        "automatic_pass": bool(
            complete and artifact_pass and identity_pass and resource_pass
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
        "node": "P159",
        "config_sha256": _sha256_file(config_path),
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
    worktree = output_dir / "worktree"
    added = False
    try:
        _add_worktree(worktree, config["candidate_commit"])
        added = True
        for index in range(1, int(config["repeat_count"]) + 1):
            run_dir = output_dir / "runs" / f"{index:02d}"
            result_path = output_dir / "worker-results" / f"{index:02d}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            monitor = launch_worker(
                config_path=config_path,
                repo_root=worktree,
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
            run_pass = bool(
                monitor["exit_code"] == 0
                and not monitor["timed_out"]
                and monitor["orphan_worker_count"]
                == int(config["gates"]["orphan_worker_count"])
                and not monitor["result_temporary_exists"]
                and worker_result is not None
                and worker_result["repo_head"] == config["candidate_commit"]
                and staging_count
                == int(config["gates"]["staging_temporary_count"])
                and worker_result["worker_wall_seconds"]
                <= float(config["gates"]["worker_wall_seconds_max"])
            )
            report["runs"].append(
                {
                    "index": index,
                    "expected_commit": config["candidate_commit"],
                    "monitor": monitor,
                    "worker_result": worker_result,
                    "run_pass": run_pass,
                }
            )
        report["gate_result"] = evaluate_runs(config, report["runs"])
    finally:
        if added:
            report["worktree_cleanup"].append(_remove_worktree(worktree))
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "prune"],
            check=False,
            capture_output=True,
            text=True,
        )
        cleanup_pass = bool(
            len(report["worktree_cleanup"]) == int(added)
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
