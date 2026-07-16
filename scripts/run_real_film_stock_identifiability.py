"""Run the frozen RF1.4A stock-preview shortcut identifiability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.stock_identifiability import (  # noqa: E402
    evaluate_structural_support,
    extract_descriptor,
    permutation_test,
    roll_balanced_loo,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_stock_identifiability_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "stock_pilots_v1" / "rf1_4" / "report.json",
    )
    args = parser.parse_args()
    config = _load(args.config)
    for path_key, hash_key in (
        ("integrity_decision", "integrity_decision_sha256"),
        ("visual_decision", "visual_decision_sha256"),
        ("integrity_report", "integrity_report_sha256"),
    ):
        path = ROOT / config[path_key]
        if _sha(path) != config[hash_key]:
            raise ValueError(f"pinned evidence hash mismatch: {path}")
    integrity = _load(ROOT / config["integrity_report"])
    structural = evaluate_structural_support(integrity["file_records"], config["stage_zero"])
    clique = set(structural["largest_comparable_clique"])
    rows = [
        row for row in integrity["file_records"]
        if row["lane"] == "negative_preview" and row["film_stock_id"] in clique
    ]
    categories = {
        key: sorted({str(row[key]) for row in rows})
        for key in ("partition", "content_cell")
    }
    descriptor_names = [
        config["descriptors"]["primary"],
        *config["descriptors"]["crop_sensitivity"],
        *config["descriptors"]["ablation"],
        config["descriptors"]["simple_global"],
        config["descriptors"]["shortcut"],
    ]
    results: dict[str, dict] = {}
    if structural["passed"]:
        for descriptor in descriptor_names:
            features = np.stack([
                extract_descriptor(
                    ROOT / config["download_root"] / Path(*Path(row["path"]).parts),
                    descriptor,
                    row,
                    categories,
                )
                for row in rows
            ])
            roll_ids = [str(row["roll_id"]) for row in rows]
            labels = [str(row["film_stock_id"]) for row in rows]
            result = roll_balanced_loo(features, roll_ids, labels)
            if descriptor == config["descriptors"]["primary"]:
                result["permutation_test"] = permutation_test(
                    features,
                    roll_ids,
                    labels,
                    permutations=int(config["permutations"]),
                    seed=int(config["permutation_seed"]),
                )
            results[descriptor] = result
    gates = config["gates"]
    primary_name = config["descriptors"]["primary"]
    shortcut_name = config["descriptors"]["shortcut"]
    simple_global_name = config["descriptors"]["simple_global"]
    checks: dict[str, bool] = {}
    if structural["passed"]:
        primary = results[primary_name]
        full = results["rgb_quantiles_full"]
        center60 = results["rgb_quantiles_center60"]
        shortcut = results[shortcut_name]
        simple_global = results[simple_global_name]
        checks = {
            "permutation": primary["permutation_test"]["p_value_greater_equal"] <= float(gates["maximum_roll_permutation_p_value"]),
            "beats_simple_global": primary["accuracy"] - simple_global["accuracy"] >= float(gates["minimum_primary_over_simple_global_accuracy"]),
            "beats_shortcut": primary["accuracy"] - shortcut["accuracy"] >= float(gates["minimum_primary_over_shortcut_accuracy"]),
            "full_crop_stable": abs(primary["accuracy"] - full["accuracy"]) <= float(gates["maximum_full_vs_center80_accuracy_delta"]),
            "center60_crop_stable": abs(primary["accuracy"] - center60["accuracy"]) <= float(gates["maximum_center60_vs_center80_accuracy_delta"]),
            "minimum_stock_recall": min(primary["per_stock_roll_recall"].values()) >= float(gates["minimum_per_stock_roll_recall"]),
        }
    passed = structural["passed"] and all(checks.values())
    if not structural["passed"]:
        decision = "structurally_unidentified"
    elif not checks["permutation"]:
        decision = "permutation_null_not_rejected"
    elif not checks["beats_shortcut"]:
        decision = "nuisance_confounded"
    elif not checks["beats_simple_global"]:
        decision = "simple_global_colour_confounded"
    elif not checks["full_crop_stable"] or not checks["center60_crop_stable"]:
        decision = "crop_border_shortcut"
    elif not checks["minimum_stock_recall"]:
        decision = "stock_recall_gate_failed"
    else:
        decision = "preview_label_signal_retained"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": commit,
        "config_sha256": _sha(args.config),
        "input_hashes_verified_before_feature_decode": True,
        "domain": config["domain"],
        "structural_gate": structural,
        "evaluated_stocks": sorted(clique),
        "evaluated_frames": len(rows),
        "descriptor_results": results,
        "gate_checks": checks,
        "passed": passed,
        "decision": decision,
        "display_proxy_operator_fit_performed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    digest = _atomic_json(args.output, report)
    print(json.dumps({
        "report": str(args.output),
        "sha256": digest,
        "decision": decision,
        "passed": passed,
        "structural_clique": structural["largest_comparable_clique"],
        "gate_checks": checks,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
