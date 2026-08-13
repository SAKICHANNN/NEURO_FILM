from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval import p4hu_ao6_value as p7h
from src.film_physics.spatial_response import apply_scanner_mtf


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(root: Path, relative: str, payload: Any) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "sha256": _write_json(path, payload),
    }


class StubNativeStage:
    def __init__(self) -> None:
        self.native_toolchains = {"stub": {"toolchain": "synthetic-test"}}
        self.calls: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    def apply_source(
        self,
        source: np.ndarray,
        *,
        seeds: tuple[int, int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.calls.append((source.copy(), seeds))
        physical = np.ascontiguousarray(
            source * np.float32(0.92) + np.float32(0.01),
            dtype=np.float32,
        )
        rms = float(
            np.sqrt(np.mean((physical - source) ** 2, dtype=np.float64))
        )
        return physical, {
            "receipt_ids": ["synthetic-r", "synthetic-g", "synthetic-b"],
            "bounded_residual_rms": rms,
            "minimum_target_sigma_d": 0.0,
            "maximum_target_sigma_d": 0.01,
            "support_degenerate_fraction": 0.0,
            "minimum_developed_density": 0.01,
            "limited_fraction": 0.0,
            "hard_clipping_used": 0.0,
            "rank_bins": 65536,
            "pass_count": 4,
            "canonical_row_block_height": 128,
            "peak_live_temporary_bytes": int(physical.nbytes * 3),
            "full_frame_intermediate_count_excluding_input_output": 2,
        }


def _stub_context_builder(
    captures: list[np.ndarray],
):
    def build(
        payload: dict[str, Any], source: np.ndarray
    ) -> tuple[Any, Any]:
        assert payload == {"synthetic": "ao6"}
        captures.append(source.copy())

        def apply_base(values: np.ndarray) -> np.ndarray:
            return values * np.float32(0.8) + np.float32(0.04)

        def apply_residual(values: np.ndarray) -> np.ndarray:
            return values * np.float32(0.96) + np.float32(0.015)

        return apply_base, apply_residual

    return build


def _make_contract(
    root: Path,
    *,
    transform: str | None = "rotate_180",
) -> tuple[dict[str, Any], np.ndarray]:
    profile = {
        "bundle_sha256": "synthetic-profile-bundle",
        "execution": {
            "layer_field_seeds": [101, 202, 303],
            "field_seed_stride_per_source": 1009,
        },
    }
    profile_binding = _binding(root, "parents/profile.json", profile)
    profile_binding["bundle_sha256"] = profile["bundle_sha256"]
    p4hu_contract = {
        "schema": (
            "neuro-film.u6-p4hu-native-spatial-density-photographic-"
            "integration-contract.v1"
        ),
        "parents": {"profile": profile_binding},
        "candidate": {
            "native_backend": "msvc-x64-c11",
            "spatial_backend": "native-thomas-field-f32-v1",
            "gamma_backend": "fast-hybrid-v1",
            "cohort_refit_allowed": False,
            "hard_clipping_allowed": False,
        },
    }
    p4hu_contract_binding = _binding(
        root, "parents/p4hu_contract.json", p4hu_contract
    )
    p4hu_evidence = _binding(
        root,
        "parents/p4hu_evidence.json",
        {"decision": "retain-p4hu"},
    )
    p4hu_evidence["required_decision"] = "retain-p4hu"
    p4hv_evidence = _binding(
        root,
        "parents/p4hv_evidence.json",
        {"result": {"decision": "retain-windows-host-only"}},
    )
    p4hv_evidence["required_decision"] = "retain-windows-host-only"
    ao6_artifact = {
        "artifact": {
            "bundle_sha256": "synthetic-ao6-bundle",
            "component_payloads": {
                "ao6-source-context-display-look": {"synthetic": "ao6"}
            },
        }
    }
    ao6_binding = _binding(root, "parents/ao6_artifact.json", ao6_artifact)
    ao6_binding["required_bundle_sha256"] = "synthetic-ao6-bundle"

    height, width = 9, 11
    yy, xx = np.mgrid[:height, :width]
    rgb8 = np.stack(
        (
            35 + xx * 8 + yy * 2,
            55 + xx * 4 + yy * 7,
            75 + xx * 5 + yy * 3,
        ),
        axis=-1,
    ).astype(np.uint8)
    source_path = root / "inputs/source_one.png"
    source_path.parent.mkdir(parents=True)
    Image.fromarray(rgb8, "RGB").save(source_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = [
        {
            "id": "source_one",
            "source_id": "synthetic:source_one",
            "make": "Synthetic Camera",
            "decoded_path": "inputs/source_one.png",
            "decoded_sha256": source_sha256,
            "width": width,
            "height": height,
            "allowed_use": "internal_fixed_evaluation",
            "rights_scope": "synthetic_test",
            "decoded_color_state": "relative_display_srgb_approximation",
        }
    ]
    manifest_path = root / "inputs/manifest.json"
    manifest_sha256 = _write_json(manifest_path, manifest)
    transforms = {"source_one": transform} if transform else {}
    contract = {
        "schema": p7h.CONTRACT_SCHEMA,
        "experiment_id": "u6.p7h-synthetic-test-v1",
        "parents": {
            "p4hu_contract": p4hu_contract_binding,
            "p4hu_evidence": p4hu_evidence,
            "p4hv_evidence": p4hv_evidence,
            "ao6_artifact": ao6_binding,
        },
        "source": {
            "manifest": "inputs/manifest.json",
            "manifest_sha256": manifest_sha256,
            "expected_manifest_rows": 1,
            "included_ids": ["source_one"],
            "expected_evaluation_rows": 1,
            "expected_camera_makes": 1,
            "required_allowed_use": "internal_fixed_evaluation",
            "required_rights_scope": "synthetic_test",
            "required_color_state": "relative_display_srgb_approximation",
            "transforms": transforms,
        },
        "candidate": {
            "layer_field_seeds": [101, 202, 303],
            "field_seed_stride_per_source": 1009,
            "scanner_mtf_sigma_pixels_rgb": [0.7, 0.7, 0.7],
            "gaussian_truncate": 3.0,
            "cohort_fitting_allowed": False,
            "hard_clipping_allowed": False,
            "posthoc_limiting_allowed": False,
            "ao6_source_context": "original-encoded-source-only",
            "physical_only_promotion_eligible": False,
        },
        "comparison": {"arms": list(p7h.ARMS)},
        "automatic_gates": {
            "require_all_inputs_and_outputs_finite": True,
            "require_no_hard_clipping_or_posthoc_limiting": True,
            "minimum_per_row_physical_residual_rms": 1e-9,
            "minimum_population_p95_combined_vs_matched_scanner_ao6_abs": 1e-9,
            "maximum_population_p99_combined_vs_matched_scanner_ao6_abs": 1.0,
            "maximum_flat_region_p99_combined_vs_matched_scanner_ao6_abs": 1.0,
            "maximum_high_frequency_chroma_p999": 1.0,
            "isolated_excursion_threshold": 1.0,
            "isolated_support_radius_pixels": 2,
            "minimum_isolated_support_count": 3,
            "maximum_isolated_excursion_count": 0,
            "maximum_new_boundary_fraction_vs_matched_scanner_ao6": 1.0,
            "maximum_output_code_boundary_fraction": 1.0,
        },
        "decision_if_pass": "open-blind-review",
        "decision_if_fail": "close-without-rescue",
        "claim_ceiling": "synthetic evaluator semantics only",
    }
    return contract, rgb8


def _apply_stub_ao6(values: np.ndarray) -> np.ndarray:
    base = values * np.float32(0.8) + np.float32(0.04)
    return np.ascontiguousarray(
        base * np.float32(0.96) + np.float32(0.015),
        dtype=np.float32,
    )


def _read_rgb16(path: Path) -> np.ndarray:
    values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert values is not None
    assert values.dtype == np.uint16
    return np.ascontiguousarray(values[..., ::-1])


def _install_runtime_stub(
    monkeypatch: Any,
    runtime: StubNativeStage,
) -> None:
    monkeypatch.setattr(
        p7h,
        "reconstruct_bounded_photographic_profile",
        lambda payload: {"profile_bundle": payload["bundle_sha256"]},
    )

    def build(**kwargs: Any) -> StubNativeStage:
        assert kwargs["candidate"]["native_backend"] == "msvc-x64-c11"
        assert kwargs["profile_payload"]["bundle_sha256"]
        return runtime

    monkeypatch.setattr(p7h, "build_native_density_stage_runtime", build)


def test_four_arm_semantics_use_one_original_source_context_and_png16(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract, original_rgb8 = _make_contract(tmp_path)
    runtime = StubNativeStage()
    _install_runtime_stub(monkeypatch, runtime)
    context_sources: list[np.ndarray] = []
    monkeypatch.setattr(
        p7h,
        "build_source_context_display_look_stages",
        _stub_context_builder(context_sources),
    )
    original_open = p7h.Image.open
    source_open_count = 0

    def counted_open(*args: Any, **kwargs: Any) -> Any:
        nonlocal source_open_count
        source_open_count += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(p7h.Image, "open", counted_open)
    output_dir = tmp_path / "run"
    report = p7h.evaluate(
        contract,
        root=tmp_path,
        output_dir=output_dir,
        build_dir=tmp_path / "build-unused",
    )

    transformed_rgb8 = np.ascontiguousarray(original_rgb8[::-1, ::-1])
    encoded = np.ascontiguousarray(transformed_rgb8, dtype=np.float32) / 255.0
    linear = np.ascontiguousarray(
        encoded_srgb_to_linear(encoded.astype(np.float64)), dtype=np.float32
    )
    physical = np.ascontiguousarray(
        linear * np.float32(0.92) + np.float32(0.01), dtype=np.float32
    )
    scanner = p7h._scanner_profile(contract["candidate"])
    encoded_scanner_source = np.ascontiguousarray(
        linear_srgb_to_encoded(
            apply_scanner_mtf(linear, scanner).astype(np.float64)
        ),
        dtype=np.float32,
    )
    encoded_scanner_physical = np.ascontiguousarray(
        linear_srgb_to_encoded(
            apply_scanner_mtf(physical, scanner).astype(np.float64)
        ),
        dtype=np.float32,
    )
    expected = {
        p7h.CURRENT_AO6: _apply_stub_ao6(encoded),
        p7h.MATCHED_AO6: _apply_stub_ao6(encoded_scanner_source),
        p7h.PHYSICAL_ONLY: encoded_scanner_physical,
        p7h.COMBINED: _apply_stub_ao6(encoded_scanner_physical),
    }

    assert source_open_count == 1
    assert len(context_sources) == 1
    np.testing.assert_array_equal(context_sources[0], encoded)
    assert len(runtime.calls) == 1
    np.testing.assert_array_equal(runtime.calls[0][0], linear)
    assert runtime.calls[0][1] == (101, 202, 303)
    row = report["rows"][0]
    assert row["source"]["transform"] == "rotate_180"
    assert row["ao6_source_context_encoded_sha256"] == hashlib.sha256(
        encoded.tobytes()
    ).hexdigest()
    for arm_id, expected_values in expected.items():
        actual = _read_rgb16(
            output_dir / "renders" / arm_id / "source_one.png"
        )
        expected_rgb16 = np.rint(expected_values * 65535.0).astype(np.uint16)
        np.testing.assert_array_equal(actual, expected_rgb16)
    assert report["automatic_pass"] is True
    assert report["blind_review_allowed"] is True
    assert report["incremental_gate_reference_arm"] == p7h.MATCHED_AO6
    assert report["physical_only_promotion_eligible"] is False
    assert report["arm_roles"][p7h.PHYSICAL_ONLY].startswith("diagnostic_only")
    assert all(
        output["exact_uint16_readback"]
        for output in report["rows"][0]["outputs"]
    )


def test_basic_automatic_gate_accepts_and_fails_closed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract, _ = _make_contract(tmp_path, transform=None)
    accepted_runtime = StubNativeStage()
    _install_runtime_stub(monkeypatch, accepted_runtime)
    monkeypatch.setattr(
        p7h,
        "build_source_context_display_look_stages",
        _stub_context_builder([]),
    )
    accepted = p7h.evaluate(
        contract,
        root=tmp_path,
        output_dir=tmp_path / "accepted",
        build_dir=tmp_path / "build-unused-a",
    )
    assert accepted["automatic_pass"] is True
    assert accepted["decision"] == "open-blind-review"

    failing_contract = copy.deepcopy(contract)
    failing_contract["automatic_gates"][
        "minimum_per_row_physical_residual_rms"
    ] = 1.0
    _install_runtime_stub(monkeypatch, StubNativeStage())
    rejected = p7h.evaluate(
        failing_contract,
        root=tmp_path,
        output_dir=tmp_path / "rejected",
        build_dir=tmp_path / "build-unused-b",
    )
    assert rejected["automatic_pass"] is False
    assert rejected["gates"]["physical_residual"] is False
    assert rejected["blind_review_allowed"] is False
    assert rejected["decision"] == "close-without-rescue"
    assert rejected["production_default_changed"] is False


@pytest.mark.parametrize(
    ("gate", "value"),
    [
        ("require_all_inputs_and_outputs_finite", None),
        ("maximum_high_frequency_chroma_p999", float("nan")),
        ("maximum_output_code_boundary_fraction", float("inf")),
        ("minimum_per_row_physical_residual_rms", -1.0),
    ],
)
def test_contract_rejects_unsafe_gate_values(
    tmp_path: Path,
    gate: str,
    value: Any,
) -> None:
    contract, _ = _make_contract(tmp_path)
    if value is None:
        del contract["automatic_gates"][gate]
    else:
        contract["automatic_gates"][gate] = value
    with pytest.raises(ValueError, match="automatic"):
        p7h._validate_contract(contract)


def test_contract_rejects_field_realization_drift(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract, _ = _make_contract(tmp_path)
    contract["candidate"]["layer_field_seeds"][0] += 1
    monkeypatch.setattr(
        p7h,
        "reconstruct_bounded_photographic_profile",
        lambda payload: payload,
    )
    with pytest.raises(ValueError, match="field realization drift"):
        p7h.evaluate(
            contract,
            root=tmp_path,
            output_dir=tmp_path / "must-not-render",
            build_dir=tmp_path / "must-not-build",
        )
