from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_u7_19a_srw_arq_generic_working_image import _stratum_result
from src.preprocess import pipeline, raw_decode

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_19a_srw_arq_generic_working_image_v1.json"
EXECUTION_LOCK = ROOT / "configs/u7_19a_srw_arq_generic_working_image_execution_v1.json"


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


def test_u7_19a_admitted_extensions_are_case_insensitive_raw_paths() -> None:
    assert raw_decode.is_raw_path(Path("capture.srw"))
    assert raw_decode.is_raw_path(Path("capture.SRW"))
    assert raw_decode.is_raw_path(Path("capture.arq"))
    assert raw_decode.is_raw_path(Path("capture.ARQ"))


def test_u7_19a_public_pipeline_routes_both_extensions_to_generic_raw(
    monkeypatch, tmp_path: Path
) -> None:
    inspections: list[Path] = []
    decodes: list[Path] = []

    def inspect(path: Path) -> object:
        inspections.append(path)
        return SimpleNamespace(source_kind="raw", format_name=path.suffix.lower()[1:])

    def load(path: Path) -> object:
        decodes.append(path)
        return SimpleNamespace(
            working_space="linear_srgb", transfer_state="scene_linear"
        )

    monkeypatch.setattr(pipeline, "inspect_raw", inspect)
    monkeypatch.setattr(pipeline, "load_raw_working_image", load)
    for suffix in (".SRW", ".ARQ"):
        path = tmp_path / f"source{suffix}"
        path.write_bytes(b"source")
        assert pipeline.inspect_input(path).source_kind == "raw"
        assert pipeline.load_working_image(path).working_space == "linear_srgb"
    assert inspections == [tmp_path / "source.SRW", tmp_path / "source.ARQ"]
    assert decodes == [tmp_path / "source.SRW", tmp_path / "source.ARQ"]


def test_u7_19a_execution_lock_admits_only_byte_exact_preflight_strata() -> None:
    lock = json.loads(EXECUTION_LOCK.read_text(encoding="utf-8"))
    assert lock["status"] == (
        "LOCKED_AFTER_BYTE_EXACT_PREFLIGHT_BEFORE_FORMAL_PRODUCT_EXECUTION"
    )
    assert lock["admitted_extensions"] == [".arq", ".srw"]
    assert lock["preflight_status"] == "PASS_BOTH_INDEPENDENT_EXTENSION_STRATA"
    reports = lock["preflight_reports"]
    assert reports["forward"]["bytes"] == reports["reverse"]["bytes"] == 16738
    assert reports["forward"]["sha256"] == reports["reverse"]["sha256"]
    for binding in lock["bindings"].values():
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
    assert all(len(value) == 40 for value in lock["commits"].values())
