from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.csd_cross_content_operator_identification import (
    FEATURE_KEYS,
    audit_score_locks,
    create_score_lock,
    fixed_descriptors,
    sha256_file,
)
from src.eval.csd_cross_content_operator_identification import (
    _metrics_for_feature as metrics_for_feature,
)
from src.eval.csd_cross_content_operator_identification import (
    _verify_official_source as verify_official_source,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=True), "utf-8")


def _audit_contract(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    path = tmp_path / "contract.json"
    contract: dict[str, object] = {
        "resource_execution": {"fresh_process_count": 2},
        "external_asset": {
            "model_sha256": "model",
            "openai_clip_sha256": "clip",
        },
        "mechanics_gates": {
            "fresh_process_embedding_max_abs_error_max": 1e-5,
            "all_embedding_norm_abs_error_max": 1e-5,
        },
    }
    _write_json(path, contract)
    return path, contract


def _score_lock(contract_sha256: str, order: str) -> dict[str, object]:
    features = {key: [1.0, 0.0] for key in FEATURE_KEYS}
    return {
        "status": "BLIND_SCORE_LOCK_COMPLETE",
        "order": order,
        "contract_sha256": contract_sha256,
        "model_sha256": "model",
        "clip_sha256": "clip",
        "inventory": {"private_mapping_reads": 0, "direct_result_reads": 0},
        "load": {"missing_keys": [], "unexpected_keys": []},
        "rows": [
            {
                "row_id": "p:a",
                "asset_sha256": "asset",
                "geometry": [8, 8],
                "relative_path": "a.png",
                "role": "role",
                "label": "a",
                "features": features,
            }
        ],
    }


def test_fixed_descriptors_are_finite_and_unit_normalized() -> None:
    image = Image.fromarray(
        np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3), "RGB"
    )
    descriptors = fixed_descriptors(image)
    assert set(descriptors) == {"oklab_moments", "rgb_histogram"}
    for values in descriptors.values():
        vector = np.asarray(values)
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-12)


def test_cross_content_metric_recovers_shared_variant() -> None:
    variants = {
        f"source-{index}": {
            "candidate": {
                "role": "role",
                "features": {"feature": [1.0, 0.0]},
            },
            "identity": {
                "role": "role",
                "features": {"feature": [0.0, 1.0]},
            },
        }
        for index in range(3)
    }
    metrics = metrics_for_feature(variants, "feature", 1e-8)
    assert metrics["combined_correct"] == 12
    assert metrics["combined_total"] == 12
    assert metrics["directions"]["candidate"]["correct"] == 6
    assert metrics["directions"]["identity"]["correct"] == 6


def test_score_lock_audit_accepts_exact_reverse_replay(tmp_path: Path) -> None:
    contract_path, _ = _audit_contract(tmp_path)
    contract_sha256 = sha256_file(contract_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_json(first, _score_lock(contract_sha256, "canonical"))
    _write_json(second, _score_lock(contract_sha256, "reverse"))
    result = audit_score_locks(contract_path, [first, second])
    assert result["status"] == "PASS_MECHANICS_READY_FOR_PRIVATE_AGGREGATION"
    assert result["maximum_embedding_error"] == 0.0


def test_score_lock_audit_rejects_nonfinite_feature(tmp_path: Path) -> None:
    contract_path, _ = _audit_contract(tmp_path)
    contract_sha256 = sha256_file(contract_path)
    first_value = _score_lock(contract_sha256, "canonical")
    second_value = _score_lock(contract_sha256, "reverse")
    second_value["rows"][0]["features"]["csd_style"][0] = float("nan")  # type: ignore[index]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_json(first, first_value)
    _write_json(second, second_value)
    with pytest.raises(ValueError, match="non-finite embedding"):
        audit_score_locks(contract_path, [first, second])


def test_p_backed_official_source_verifies_without_global_safe_directory() -> None:
    contract = json.loads(
        (ROOT / "configs/u5_r2csd0_cross_content_operator_identification_v1.json").read_text(
            "utf-8"
        )
    )
    source_root = (
        ROOT / "outputs/external/u5_r2csd0/source/CSD_3a9df326"
    ).resolve()
    verify_official_source(contract, source_root)


def test_invalid_score_order_fails_before_model_load(tmp_path: Path) -> None:
    _write_json(tmp_path / "contract.json", {})
    with pytest.raises(ValueError, match="order must be canonical or reverse"):
        create_score_lock(
            tmp_path / "contract.json",
            tmp_path,
            tmp_path,
            tmp_path / "checkpoint.bin",
            tmp_path,
            tmp_path / "output.json",
            "invalid",
            "cpu",
        )
