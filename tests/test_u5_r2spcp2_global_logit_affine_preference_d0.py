from __future__ import annotations

import json
from pathlib import Path

from scripts.acquire_u5_r2spcp2_selected_pairs import PNG_SIGNATURE, _validate_payload
from scripts.build_u5_r2spcp2_preference_roles import _scene_sort_key

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2spcp2_global_logit_affine_preference_d0_v1.json"


def test_spcp2_contract_freezes_scene_disjoint_roles_before_pixels() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "U5.R2SPCP2"
    assert payload["selection"]["fit_scenes"] == 96
    assert payload["selection"]["calibration_scenes"] == 32
    assert payload["selection"]["sealed_scenes"] == 32
    assert payload["selection"]["replacement_forbidden"] is True
    assert payload["acquisition"]["initial_member_count_exact"] == 256
    assert payload["acquisition"][
        "sealed_image_payload_read_before_development_pass_forbidden"
    ] is True


def test_spcp2_contract_is_one_source_free_explicit_operator() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fit = payload["fit"]
    assert "one shared 3x3 M" in fit["operator"]
    assert fit["application_source_used_for_fit_or_selection"] is False
    assert fit["output_is_intrinsically_bounded"] is True
    assert "per-scene routing" in payload["stop_rules"][3]


def test_spcp2_contract_freezes_discriminating_controls_and_tails() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert len(payload["controls"]) == 4
    metrics = payload["calibration_metrics"]
    assert metrics["candidate_improvement_rate_min"] == 0.75
    assert metrics["candidate_improvement_worst_min"] == -0.10
    assert metrics["reverse_direction_improvement_rate_max"] == 0.35
    assert metrics["new_exact_boundary_fraction_max"] == 0.0


def test_spcp2_scene_sort_is_stable_and_scene_specific() -> None:
    assert _scene_sort_key("I0001") == _scene_sort_key("I0001")
    assert _scene_sort_key("I0001") != _scene_sort_key("I0002")
    assert len(_scene_sort_key("I0001")) == 64


def test_spcp2_acquisition_payload_validation_is_fail_closed() -> None:
    data = PNG_SIGNATURE + b"fixture"
    import zlib

    member = {
        "name": "SPCP_dataset/images/I0001_01_01.png",
        "uncompressed_size": len(data),
        "crc32_hex": f"{zlib.crc32(data) & 0xFFFFFFFF:08x}",
    }
    _validate_payload(data, member)
    member["crc32_hex"] = "00000000"
    try:
        _validate_payload(data, member)
    except ValueError as error:
        assert "CRC drift" in str(error)
    else:
        raise AssertionError("CRC drift was accepted")


def test_jpeg_magic_does_not_satisfy_frozen_png_gate() -> None:
    assert not b"\xff\xd8\xff\xe0JFIF".startswith(PNG_SIGNATURE)
