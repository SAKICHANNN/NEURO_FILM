from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.eval.native_density_photographic_integration as integration
from src.eval.native_density_photographic_integration import (
    NativeDensityStageRuntime,
    build_native_density_stage_runtime,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p4hu_reuses_retained_native_spatial_and_fast_density_mechanisms() -> None:
    contract = load_contract(
        ROOT / "configs/u6_p4hu_native_spatial_density_photographic_integration_v1.json"
    )
    assert contract["candidate"]["spatial_backend"] == "native-thomas-field-f32-v1"
    assert contract["candidate"]["gamma_backend"] == "fast-hybrid-v1"
    assert (
        contract["parents"]["p8bs_evidence"]["required_decision"]
        == "retain_native_generic_thomas_field_primitive"
    )
    assert contract["parents"]["p4ht_evidence"]["required_decision"].startswith(
        "retain_fast_native"
    )
    assert contract["candidate"]["cohort_refit_allowed"] is False


class _Amplitude:
    @staticmethod
    def evaluate_channel(
        prior: object, channel: str, exposure: np.ndarray
    ) -> np.ndarray:
        del prior, channel
        return np.full_like(exposure, 0.125, dtype=np.float64)


class _Profile:
    particle_sigma_samples = 0.5
    cluster_sigma_samples = 1.0
    mean_offspring = 2.0
    truncate = 3.0
    component_seeds = (3, 5, 7)
    spatial_profile_id = "spatial-profile"
    amplitude_profile = _Amplitude()

    @staticmethod
    def identity() -> str:
        return "density-profile"


class _Prior:
    curves = tuple(SimpleNamespace(domain=(-1.0, 1.0)) for _ in range(3))

    @staticmethod
    def identity() -> str:
        return "manufacturer-prior"


def test_native_stage_applies_verified_source_and_returns_compact_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _Profile()
    prior = _Prior()
    components = SimpleNamespace(
        profile=profile,
        prior=prior,
        canonical_row_block_height=128,
        rank_bins=65536,
    )
    candidate = {
        "spatial_backend": "native-thomas-field-f32-v1",
        "gamma_backend": "fast-hybrid-v1",
        "copula_iterations": 3,
        "gamma_inverse_iterations": 80,
        "gamma_newton_iterations": 6,
        "gamma_direct_shape_upper": 50.0,
        "high_shape_threshold": 10000.0,
    }

    def render_field(
        library: object, native_profile: object, shape: tuple[int, int]
    ) -> tuple[np.ndarray, dict[str, object]]:
        del library
        return (
            np.full(shape, native_profile.realization_seed, dtype=np.float32),
            {},
        )

    def apply_density(
        copula_library: object,
        gamma_library: object,
        source: np.ndarray,
        fields: np.ndarray,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        del copula_library, gamma_library
        assert fields.shape == (source.shape[0] * source.shape[1], 3)
        assert kwargs["components"] is components
        assert kwargs["fast_newton_iterations"] == 6
        physical = source + np.float32(0.01)
        developed = np.zeros_like(source)
        return physical, developed, {
            "active_fraction": 0.75,
            "minimum_developed_density": 0.2,
        }

    monkeypatch.setattr(integration, "render_native_thomas_field", render_field)
    monkeypatch.setattr(
        integration, "apply_profile_bound_native_density", apply_density
    )
    runtime = NativeDensityStageRuntime(
        candidate=candidate,
        profile_payload={
            "density_profile_id": profile.identity(),
            "manufacturer_prior_id": prior.identity(),
        },
        components=components,
        copula_library=object(),
        gamma_library=object(),
        fast_gamma=True,
        native_spatial=True,
        native_toolchains={"copula": {}, "gamma": {}, "field": {}},
        field_library=object(),
    )
    source = np.full((2, 3, 3), 0.25, dtype=np.float32)
    output, receipt = runtime.apply_source(source, seeds=(11, 13, 17))

    assert np.array_equal(output, source + np.float32(0.01))
    assert receipt["receipt_ids"] == [
        hashlib.sha256(
            np.full((2, 3), seed, dtype=np.float32).tobytes()
        ).hexdigest()
        for seed in (11, 13, 17)
    ]
    assert receipt["rank_bins"] == 65536
    assert receipt["canonical_row_block_height"] == 128
    assert receipt["pass_count"] == 4
    assert receipt["support_degenerate_fraction"] == 0.25
    assert receipt["hard_clipping_used"] == 0.0
    assert receipt["limited_fraction"] == 0.0
    assert not any(isinstance(value, np.ndarray) for value in receipt.values())

    with pytest.raises(ValueError, match="component identity drift"):
        runtime.apply_physical(
            source,
            SimpleNamespace(identity=lambda: "foreign-profile"),
            prior,
            (11, 13, 17),
        )


def test_native_stage_factory_selects_exact_p4hu_kernels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    builds: list[str] = []

    def build(**kwargs: object) -> dict[str, object]:
        source = str(kwargs["source_relative"])
        builds.append(source)
        basename = str(kwargs["basename"])
        return {
            "toolchain": "msvc-x64",
            "source_sha256": "a" * 64,
            "dll_sha256": "b" * 64,
            "dll_path": str(tmp_path / f"{basename}.dll"),
            "compiler_output": "location-specific",
        }

    monkeypatch.setattr(integration, "build_msvc_c11_dll", build)
    monkeypatch.setattr(
        integration, "load_native_histogram_copula_library", lambda path: path
    )
    monkeypatch.setattr(
        integration, "load_native_fast_gamma_density_library", lambda path: path
    )
    monkeypatch.setattr(
        integration, "load_native_thomas_field_library", lambda path: path
    )
    candidate = {
        "spatial_backend": "native-thomas-field-f32-v1",
        "gamma_backend": "fast-hybrid-v1",
    }
    runtime = build_native_density_stage_runtime(
        root=ROOT,
        build_dir=tmp_path / "build",
        candidate=candidate,
        profile_payload={},
        components=SimpleNamespace(),
    )

    assert builds == [
        "native/film_physics/nf_histogram_copula_f32_v1.c",
        "native/film_physics/nf_gamma_density_fast_f64_v1.c",
        "native/film_physics/nf_thomas_field_f32_v1.c",
    ]
    assert runtime.fast_gamma is True
    assert runtime.native_spatial is True
    assert set(runtime.native_toolchains) == {"copula", "gamma", "field"}
    assert all(
        "dll_path" not in receipt and "compiler_output" not in receipt
        for receipt in runtime.native_toolchains.values()
    )
