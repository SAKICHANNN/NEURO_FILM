from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.digital_lad_print_compatibility import (
    load_contract,
    run_audit,
)
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.digital_lad import aim_from_config
from src.film_physics.digital_lad_interpretation import (
    DIGITAL_LAD_PRINT_ANCHOR_SCHEMA,
    apply_digital_lad_aim_to_print,
    apply_digital_lad_codes_to_print,
)
from src.film_physics.interpretation_medium import compile_print_interpretation

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6s_digital_lad_print_compatibility_v1.json"


def _operator():
    sensitometry_config = json.loads(
        (ROOT / "configs/u2_2a_sensitometry_primitive_v1.json").read_text(
            encoding="utf-8"
        )
    )
    sensitometry = build_operator(sensitometry_config)
    source = json.loads(
        (ROOT / "configs/u5_r2e0_density_domain_operator_v1.json").read_text(
            encoding="utf-8"
        )
    )
    witness = source["witnesses"]["neutral_density_reference"]
    return compile_print_interpretation(
        sensitometry,
        dye_absorption_matrix=np.asarray(witness["dye_absorption_matrix"]),
        print_matrix=np.asarray(witness["print_matrix"]),
        paper_midpoints=np.asarray(witness["paper_midpoints"]),
        paper_slopes=np.asarray(witness["paper_slopes"]),
        paper_maximum_densities=np.asarray(witness["paper_maximum_densities"]),
        maximum_relative_layer_exposure=16.0,
        density_dtype=np.float64,
    )


def test_aim_adapter_is_neutral_typed_and_repeat_exact() -> None:
    payload = json.loads(
        (ROOT / "configs/u6_p6r_kodak_digital_lad_density_primitive_v1.json").read_text(
            encoding="utf-8"
        )
    )["lad_aims"][0]
    aim = aim_from_config(payload)
    first = apply_digital_lad_aim_to_print(
        aim,
        route="color_negative_print",
        print_interpretation=_operator(),
    )
    second = apply_digital_lad_aim_to_print(
        aim,
        route="color_negative_print",
        print_interpretation=_operator(),
    )
    assert np.array_equal(first.neutral_print_reflectance, second.neutral_print_reflectance)
    assert np.isfinite(first.neutral_print_reflectance).all()
    assert np.min(first.neutral_print_reflectance) >= 0.0
    assert np.max(first.neutral_print_reflectance) <= 1.0
    assert first.descriptor()["schema"] == DIGITAL_LAD_PRINT_ANCHOR_SCHEMA
    assert first.descriptor()["developed_exposure_result_fabricated"] is False


def test_negative_raw_ip_tail_and_non_print_routes_fail_closed() -> None:
    operator = _operator()
    with pytest.raises(ValueError, match="negative raw printing density"):
        apply_digital_lad_codes_to_print(
            [966, 1023],
            mode="interpositive",
            route="color_negative_print",
            print_interpretation=operator,
        )
    for route in (
        "color_negative_neutral_scan",
        "slide_direct_scan",
        "bw_developer_scan",
    ):
        with pytest.raises(ValueError, match="requires color_negative_print"):
            apply_digital_lad_codes_to_print(
                445,
                mode="negative",
                route=route,
                print_interpretation=operator,
            )


def test_code_sweep_partitions_exactly() -> None:
    operator = _operator()
    lower = float(np.max(operator.black_reference_density))
    upper = float(np.min(operator.white_reference_density))
    all_codes = np.arange(1024, dtype=np.int64)
    densities = 0.002 * all_codes
    codes = all_codes[(densities >= lower) & (densities <= upper)]
    full = apply_digital_lad_codes_to_print(
        codes,
        mode="negative",
        route="color_negative_print",
        print_interpretation=operator,
    )
    parts = np.concatenate(
        [
            apply_digital_lad_codes_to_print(
                codes[start:stop],
                mode="negative",
                route="color_negative_print",
                print_interpretation=operator,
            ).neutral_print_reflectance
            for start, stop in (
                (0, min(127, len(codes))),
                (min(127, len(codes)), min(509, len(codes))),
                (min(509, len(codes)), len(codes)),
            )
            if start < stop
        ]
    )
    assert np.array_equal(parts, full.neutral_print_reflectance)


def test_formal_audit_passes_without_rgb_image_transform() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"]
    assert report["decision"] == "retain_typed_neutral_recorder_print_anchor"
    assert report["gate_results"]["rgb_image_transform_count_zero"]
    assert report["forbidden_status_m_dmin_function_reads"] == 0
