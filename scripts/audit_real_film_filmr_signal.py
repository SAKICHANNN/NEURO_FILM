"""Run the metadata-first RF1.1 FILM-R signal identifiability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.filmr_signal_audit import (  # noqa: E402
    build_support_matrix,
    content_totals,
    evaluate_support_gate,
    matrix_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_filmr_signal_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "filmr_v2" / "signal_audit" / "report.json",
    )
    return parser.parse_args()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    decision_path = ROOT / "configs" / "real_film_filmr_decision.json"
    integrity_path = (
        ROOT / "outputs" / "real_film" / "filmr_v2" / "integrity_audit" / "report.json"
    )
    if _sha(decision_path) != config["dataset_decision_sha256"]:
        raise ValueError("FILM-R dataset decision hash mismatch")
    if _sha(integrity_path) != config["integrity_report_sha256"]:
        raise ValueError("FILM-R integrity report hash mismatch")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    expected_sheets = {
        Path(path).name: digest for path, digest in integrity["review_sheet_sha256"].items()
    }
    if expected_sheets != config["review_sheet_sha256"]:
        raise ValueError("FILM-R review sheet hashes do not match frozen labels")
    labels = {str(key): str(value) for key, value in config["manual_content_labels"].items()}
    families, contents, matrix = build_support_matrix(integrity["per_pair"], labels)
    gates = config["gates"]
    gate = evaluate_support_gate(
        families,
        contents,
        matrix,
        minimum_family_samples=int(gates["minimum_family_samples"]),
        minimum_cell_samples=int(gates["minimum_samples_per_family_content_cell"]),
        minimum_supported_cells=int(gates["minimum_supported_content_cells_per_family"]),
        minimum_shared_cells=int(gates["minimum_pairwise_shared_content_cells"]),
        minimum_multiclass_families=int(gates["minimum_families_for_multiclass_test"]),
    )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _commit(),
        "config_sha256": _sha(args.config),
        "evidence_hashes_verified_before_feature_decode": True,
        "new_image_feature_pixels_decoded": False,
        "pairs": len(integrity["per_pair"]),
        "families": list(families),
        "content_categories": list(contents),
        "content_totals": content_totals(labels),
        "support_matrix": matrix_records(families, contents, matrix),
        "gate": gate,
        "feature_audit_executed": False,
        "feature_audit_stop_reason": None if gate["passed"] else "stage_zero_structural_overlap_failed",
        "claim_ceiling": config["claim_ceiling"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(encoded)
    print(json.dumps({
        "report": str(args.output),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "decision": gate["decision"],
        "cross_content_families": gate["cross_content_families"],
        "largest_pairwise_comparable_clique": gate["largest_pairwise_comparable_clique"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
