from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

import scripts.run_u5_r2repid4_shared_logit_affine_d0 as module


def _jpeg(path: Path, offset: int) -> dict[str, object]:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    y, x = np.indices((18, 24))
    values = np.stack(
        ((x * 9 + offset) % 256, (y * 13 + offset) % 256, ((x + y) * 7 + offset) % 256),
        axis=-1,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values, mode="RGB").save(
        path, format="JPEG", quality=95, icc_profile=profile
    )
    payload = path.read_bytes()
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "icc_sha256": hashlib.sha256(profile).hexdigest(),
    }


def test_repid_d0_is_enumeration_exact_and_sealed_blind(
    tmp_path: Path, monkeypatch
) -> None:
    records = []
    for index, role in enumerate(("fit", "fit", "calibration")):
        scene = f"scene-{index}.jpeg"
        loser = _jpeg(tmp_path / f"loser-{index}.jpeg", index)
        winner = _jpeg(tmp_path / f"winner-{index}.jpeg", index + 12)
        loser.update({"endpoint": "loser", "expert_role": "original"})
        winner.update({"endpoint": "winner", "expert_role": "tiff16_a"})
        records.append(
            {
                "scene_id": scene,
                "role": role,
                "loser": "original",
                "winner": "tiff16_a",
                "members": [loser, winner],
            }
        )
    acquisition = {
        "decision": "open-development",
        "sealed_member_requests": 0,
        "selected_identity_sha256": "selected",
        "records": records,
    }
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    acquisition_sha = hashlib.sha256(acquisition_path.read_bytes()).hexdigest()
    contract = {
        "experiment_id": "test",
        "parent": {
            "acquisition_sha256": acquisition_sha,
            "required_decision": "open-development",
            "selected_identity_sha256": "selected",
        },
        "fit": {
            "pixels_per_scene": 16,
            "ridge_alpha": 0.0001,
            "dose_grid": [1.0],
            "matrix_gates": {
                "determinant_min": -100.0,
                "condition_number_max": 1.0e9,
                "minimum_singular_value": 0.0,
            },
        },
        "calibration_metrics": {
            "candidate_improvement_rate_min": 0.0,
            "candidate_improvement_median_min": -100.0,
            "candidate_improvement_worst_min": -100.0,
            "beat_diagonal_rate_min": 0.0,
            "diagonal_relative_gain_median_min": -100.0,
            "beat_label_permuted_rate_min": 0.0,
            "label_permuted_relative_gain_median_min": -100.0,
            "reverse_direction_improvement_rate_max": 1.0,
            "median_output_delta_e_oklab_from_identity_min": 0.0,
            "new_exact_boundary_fraction_max": 1.0,
            "p999_gradient_ratio_max": 1.0e9,
        },
        "decision_if_pass": "pass",
        "decision_if_fail": "fail",
        "claim_ceiling": "synthetic-test-only",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    first = module.run(
        contract_path, acquisition_path, tmp_path / "first.json", reverse_enumeration=False
    )
    second = module.run(
        contract_path, acquisition_path, tmp_path / "second.json", reverse_enumeration=True
    )
    assert first["stable_evidence_id"] == second["stable_evidence_id"]
    assert first["sealed_member_requests"] == 0
    assert first["fit_sample_count"] == 32
