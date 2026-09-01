from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_u7_19a_srw_arq_generic_working_image import _stratum_result

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_19a_srw_arq_generic_working_image_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_19a_freeze_binds_contract_and_two_independent_strata() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract = config["bindings"]["contract"]
    contract_path = ROOT / contract["path"]
    assert contract_path.stat().st_size == contract["bytes"]
    assert _sha256(contract_path) == contract["sha256"]
    assert config["status"] == "FROZEN_PREDECODE"
    assert config["predecode_fact"]["u7_19a_pixel_decodes"] == 0
    assert config["independent_admission_policy"] == {
        "failing_stratum_must_remain_absent_from_raw_suffixes": True,
        "frozen_before_pixel_decode": True,
        "passing_stratum_may_be_admitted_if_other_stratum_fails": True,
        "post_result_source_or_member_selection_forbidden": True,
    }
    strata = {row["extension"]: row for row in config["strata"]}
    assert set(strata) == {".arq", ".srw"}
    assert len(strata[".srw"]["sources"]) == strata[".srw"]["required_members"] == 4
    assert len(strata[".arq"]["sources"]) == strata[".arq"]["required_members"] == 1


def test_u7_19a_stratum_decision_is_all_members_and_all_gates(monkeypatch) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stratum = config["strata"][0]
    required_warnings = config["per_stratum_preflight_gates"]["required_warning_codes"]

    def record(_producer_repo: Path, row: dict[str, object]) -> dict[str, object]:
        return {
            "alpha_policy": "absent",
            "bit_depth_in": 16,
            "inspection": {"warning_codes": []},
            "orientation_applied": True,
            "pixels": {
                "c_contiguous": True,
                "dtype": "float32",
                "finite": True,
                "maximum": 1.0,
                "minimum": 0.0,
                "nonconstant": True,
                "owned": True,
                "shape": [4, 6, 3],
                "writeable": True,
            },
            "source_id": row["source_id"],
            "source_transfer_state": "scene_linear",
            "source_unchanged": True,
            "transfer_state": "scene_linear",
            "warning_codes": sorted(required_warnings),
            "warning_messages": {
                "generic_raw_render": "exact vendor/Adobe rendering is not promised",
                "generic_raw_display_mapping": (
                    "without a calibrated scene-to-display tone map"
                ),
            },
            "working_space": "linear_srgb",
        }

    monkeypatch.setattr(
        "scripts.audit_u7_19a_srw_arq_generic_working_image._row_record", record
    )
    result = _stratum_result(config, Path("unused"), stratum, reverse=True)
    assert result["admission"] == "PASS_PREFLIGHT"
    assert all(result["gates"].values())


def test_u7_19a_claim_ceiling_remains_generic_and_uncalibrated() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    ceiling = config["claim_ceiling"]
    assert "exact admitted source-locked files" in ceiling
    assert "No general SRW/ARQ" in ceiling
    assert "calibration" in ceiling
    assert config["product_chain"]["require_claim_output_label"] == "film-inspired"
    assert (
        config["product_chain"]["require_claim_evidence_grade"] == "look-approximation"
    )
    assert config["product_chain"]["require_calibrated_reference_allowed_false"] is True
