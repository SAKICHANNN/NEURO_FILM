"""Run the frozen RF2.S0 Gold archive-display matrix transplant gates."""

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

from src.real_film.gold_matrix_transplant import (  # noqa: E402
    DigitalSample,
    GoldMatrixTransplantError,
    evaluate_transplant,
    sample_rgb_image,
    style_and_basic_residual,
)
from src.real_film.gold_transform_consistency import load_paired_frame_samples  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(path: Path, expected: str) -> None:
    if _sha(path) != expected:
        raise GoldMatrixTransplantError(f"pinned evidence hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "real_film_gold_matrix_transplant_v1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "real_film" / "stock_pilots_v1" / "rf2_s0" / "report.json",
    )
    args = parser.parse_args()
    config = _load(args.config)
    decision_path = ROOT / config["rf1_4b1_decision"]
    alignment_path = ROOT / config["alignment_report"]
    frozen_path = ROOT / config["digital_frozen_set"]
    anchors_path = ROOT / config["normalized_anchor_manifest"]
    for path, expected in (
        (decision_path, config["rf1_4b1_decision_sha256"]),
        (alignment_path, config["alignment_report_sha256"]),
        (frozen_path, config["digital_frozen_set_sha256"]),
        (anchors_path, config["normalized_anchor_manifest_sha256"]),
    ):
        _verify(path, expected)
    if _load(decision_path)["decision"] != "metric_and_visual_pass_simple_matrix_preferred":
        raise GoldMatrixTransplantError("RF1.4B1 did not select the required matrix class")

    alignment = _load(alignment_path)
    archive_frames = load_paired_frame_samples(
        download_root=ROOT / config["download_root"],
        pair_records=alignment["pair_records"],
        maximum_pixels_per_frame=int(config["all_roll_fit"]["maximum_pixels_per_frame"]),
    )
    frozen_payload = _load(frozen_path)
    frozen = frozen_payload.get("frozen_set", frozen_payload)
    digital_samples: list[DigitalSample] = []
    sample_records: dict[str, dict] = {}
    budget = int(config["digital_sampling"]["maximum_pixels_per_image"])
    for row in frozen["samples"]:
        if row.get("availability") != "available" or row.get("split") not in {"gold", "stress"}:
            continue
        source_path = ROOT / row["source_path"]
        _verify(source_path, row["source_sha256"])
        sample_id = str(row["id"])
        sample_records[sample_id] = row
        digital_samples.append(DigitalSample(
            sample_id, str(row["split"]), sample_rgb_image(source_path, budget), row["source_path"]
        ))
    if sum(sample.split == "gold" for sample in digital_samples) != 9 or sum(sample.split == "stress" for sample in digital_samples) != 32:
        raise GoldMatrixTransplantError("frozen digital set must contain 9 gold and 32 stress images")

    anchors = _load(anchors_path)
    anchor_metrics: dict[str, list[tuple[float, float]]] = {
        "anchor55_color_only": [], "anchor56_color_only": [], "bland_safe_rich_control": []
    }
    digital_by_id = {sample.sample_id: sample for sample in digital_samples}
    for row in anchors["records"]:
        candidate = row["candidate_id"]
        sample_id = str(row["sample_id"])
        if candidate not in anchor_metrics or sample_records.get(sample_id, {}).get("split") != "gold":
            continue
        output_path = ROOT / row["output"]
        _verify(output_path, row["output_sha256"])
        output = sample_rgb_image(output_path, budget)
        source = digital_by_id[sample_id].pixels
        if output.shape != source.shape:
            raise GoldMatrixTransplantError("anchor/source sample grids differ")
        anchor_metrics[candidate].append(style_and_basic_residual(source, output))
    anchor_summary = {
        key: {
            "gold_median_style_delta_e76": float(np.median([row[0] for row in values])),
            "gold_median_residual_delta_e76_after_matched_basic": float(np.median([row[1] for row in values])),
        }
        for key, values in anchor_metrics.items()
    }
    expected = config["anchor_derived_floors_frozen_before_candidate_render"]
    for candidate, style_key, residual_key in (
        ("anchor55_color_only", "anchor55_gold_median_style_delta_e76", "anchor55_gold_median_residual_delta_e76_after_matched_basic"),
        ("anchor56_color_only", "anchor56_gold_median_style_delta_e76", "anchor56_gold_median_residual_delta_e76_after_matched_basic"),
        ("bland_safe_rich_control", "safe_rich_gold_median_style_delta_e76", "safe_rich_gold_median_residual_delta_e76_after_matched_basic"),
    ):
        if abs(anchor_summary[candidate]["gold_median_style_delta_e76"] - float(expected[style_key])) > 1e-6:
            raise GoldMatrixTransplantError("frozen anchor style floor does not reproduce")
        if abs(anchor_summary[candidate]["gold_median_residual_delta_e76_after_matched_basic"] - float(expected[residual_key])) > 1e-6:
            raise GoldMatrixTransplantError("frozen anchor residual floor does not reproduce")

    result = evaluate_transplant(archive_frames=archive_frames, digital_samples=digital_samples, config=config)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": commit,
        "config_sha256": _sha(args.config),
        "all_pinned_hashes_and_source_hashes_verified_before_decode": True,
        "archive_fit_only_no_digital_fit": True,
        "anchor_summary_reproduced_before_candidate_evaluation": anchor_summary,
        **result,
        "visual_status": "pending_required_blind_adjudication" if result["automatic_passed"] else "not_run_automatic_gate_failed",
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(json.dumps({
        "report": str(args.output),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "decision": report["decision"],
        "automatic_passed": report["automatic_passed"],
        "gold_ood_eligible_images": report["gold_ood_eligible_images"],
        "stress_ood_eligible_fraction": report["stress_ood_eligible_fraction"],
        "selected_strength": report["selected_strength"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
