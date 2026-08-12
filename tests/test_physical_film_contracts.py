from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.film_physics import (
    PROFILE_BUNDLE_SCHEMA,
    QUALITY_TIERS,
    ComponentBinding,
    FilmProfileBundle,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    QualityTier,
    coordinate_counter_u64,
    density_to_transmittance,
    scene_exposure_from_working_image,
    transmittance_to_density,
)
from src.preprocess.types import SourceProfile, WorkingImage

SHA_A = "1" * 64
SHA_B = "2" * 64


def _domain_array(
    values: np.ndarray,
    domain: PhysicalDomain,
    unit: PhysicalUnit,
) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values, domain, unit, ("cyan", "magenta", "yellow"), PhysicalScale(8.0)
    )


def _bundle() -> FilmProfileBundle:
    return FilmProfileBundle(
        profile_id="generic-colour-negative-v1",
        claim_level="generic-physical-inspired",
        stock_id="unknown",
        process_id="unknown",
        scanner_profile_id="unknown",
        evidence_manifest_sha256=None,
        components=(
            ComponentBinding(
                "capture-development",
                "test.capture.v1",
                SHA_A,
                PhysicalDomain.SCENE_LINEAR,
                PhysicalDomain.DEVELOPED_DENSITY,
            ),
            ComponentBinding(
                "scan",
                "test.scan.v1",
                SHA_B,
                PhysicalDomain.TRANSMITTANCE,
                PhysicalDomain.SCAN_LINEAR,
            ),
        ),
    )


def test_domain_array_is_owned_immutable_and_unit_locked() -> None:
    source = np.full((2, 3, 3), 0.25, dtype=np.float32)
    state = _domain_array(
        source, PhysicalDomain.DEVELOPED_DENSITY, PhysicalUnit.OPTICAL_DENSITY
    )
    source[:] = 0.5
    assert np.all(state.values == np.float32(0.25))
    assert not state.values.flags.writeable
    with pytest.raises(ValueError, match="requires unit"):
        _domain_array(
            source,
            PhysicalDomain.DEVELOPED_DENSITY,
            PhysicalUnit.RELATIVE_DISPLAY_LIGHT,
        )


def test_domain_array_explicit_adoption_transfers_owned_storage() -> None:
    source = np.full((2, 3, 3), 0.25, dtype=np.float32)
    state = PhysicalDomainArray.adopt(
        source,
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        ("cyan", "magenta", "yellow"),
    )
    assert state.values is source
    assert not source.flags.writeable
    with pytest.raises(ValueError, match="own contiguous data"):
        PhysicalDomainArray.adopt(
            np.zeros((3, 2, 3), dtype=np.float32)[::2],
            PhysicalDomain.DEVELOPED_DENSITY,
            PhysicalUnit.OPTICAL_DENSITY,
            ("cyan", "magenta", "yellow"),
        )


@pytest.mark.parametrize("dtype,tolerance", [(np.float32, 2e-7), (np.float64, 2e-15)])
def test_density_transmittance_roundtrip(dtype: type[np.floating], tolerance: float) -> None:
    values = np.linspace(0.0, 4.0, 99, dtype=dtype).reshape(11, 3, 3)
    density = _domain_array(
        values, PhysicalDomain.DEVELOPED_DENSITY, PhysicalUnit.OPTICAL_DENSITY
    )
    transmission = density_to_transmittance(density)
    restored = transmittance_to_density(transmission)
    assert transmission.domain is PhysicalDomain.TRANSMITTANCE
    assert restored.values.dtype == values.dtype
    np.testing.assert_allclose(restored.values, values, rtol=0.0, atol=tolerance)


def test_domain_mismatch_and_invalid_transmittance_fail_closed() -> None:
    density = _domain_array(
        np.ones((1, 1, 3), dtype=np.float64),
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
    )
    with pytest.raises(ValueError, match="domain mismatch"):
        transmittance_to_density(density)
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        _domain_array(
            np.zeros((1, 1, 3), dtype=np.float64),
            PhysicalDomain.TRANSMITTANCE,
            PhysicalUnit.TRANSMITTANCE,
        )


def test_physical_scale_uses_cycles_per_mm() -> None:
    scale = PhysicalScale(8.0)
    assert scale.samples_per_mm == 125.0
    assert scale.nyquist_cycles_per_mm == 62.5


def test_working_image_ingress_requires_scene_linear_d65() -> None:
    working = WorkingImage(
        pixels=np.full((2, 2, 3), 0.2, dtype=np.float32),
        working_space="linear_srgb",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "test"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=14,
        source_path=Path("test.raw"),
    )
    admitted = scene_exposure_from_working_image(working)
    assert admitted.domain is PhysicalDomain.SCENE_LINEAR
    working.transfer_state = "display_linear"
    with pytest.raises(ValueError, match="requires scene_linear"):
        scene_exposure_from_working_image(working)


def test_profile_bundle_roundtrip_has_stable_identity() -> None:
    bundle = _bundle()
    rebuilt = FilmProfileBundle.from_dict(bundle.to_dict())
    assert rebuilt.to_dict() == bundle.to_dict()
    assert rebuilt.bundle_sha256 == bundle.bundle_sha256
    assert len(bundle.bundle_sha256) == 64
    assert bundle.schema == PROFILE_BUNDLE_SCHEMA


def test_profile_rejects_backwards_component_and_unknown_calibration() -> None:
    with pytest.raises(ValueError, match="backwards"):
        ComponentBinding(
            "bad",
            "test.bad.v1",
            SHA_A,
            PhysicalDomain.DISPLAY_RGB,
            PhysicalDomain.SCENE_LINEAR,
        )
    with pytest.raises(ValueError, match="requires stock"):
        FilmProfileBundle(
            profile_id="false-calibration",
            claim_level="calibrated-reference",
            stock_id="unknown",
            process_id="process-1",
            scanner_profile_id="scanner-1",
            evidence_manifest_sha256=SHA_A,
            components=_bundle().components,
        )


def test_profile_rejects_field_and_component_order_mutation() -> None:
    payload = _bundle().to_dict()
    payload["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        FilmProfileBundle.from_dict(payload)
    payload = _bundle().to_dict()
    payload["components"][0]["extra"] = True
    with pytest.raises(ValueError, match="component fields"):
        FilmProfileBundle.from_dict(payload)
    with pytest.raises(ValueError, match="canonical domain order"):
        FilmProfileBundle(
            profile_id="bad-order",
            claim_level="look-approximation",
            stock_id="unknown",
            process_id="unknown",
            scanner_profile_id="unknown",
            evidence_manifest_sha256=None,
            components=tuple(reversed(_bundle().components)),
        )


def test_quality_tiers_separate_product_and_reference_semantics() -> None:
    assert QUALITY_TIERS[QualityTier.PREVIEW].product_runtime_allowed
    assert QUALITY_TIERS[QualityTier.STANDARD].nominal_megapixels == (12, 24)
    assert QUALITY_TIERS[QualityTier.REFERENCE].arithmetic == "float64"
    assert not QUALITY_TIERS[QualityTier.REFERENCE].product_runtime_allowed


def test_counter_is_repeatable_coordinate_and_partition_invariant() -> None:
    expected = coordinate_counter_u64(
        profile_sha256=SHA_A, seed=-7, frame=2, x=101, y=55, layer=1
    )
    repeated = coordinate_counter_u64(
        profile_sha256=SHA_A, seed=-7, frame=2, x=101, y=55, layer=1
    )
    assert repeated == expected
    assert coordinate_counter_u64(
        profile_sha256=SHA_A, seed=-7, frame=2, x=102, y=55, layer=1
    ) != expected
    # A tile origin changes local coordinates, not the global counter address.
    tile_origin_x, local_x = 96, 5
    assert coordinate_counter_u64(
        profile_sha256=SHA_A,
        seed=-7,
        frame=2,
        x=tile_origin_x + local_x,
        y=55,
        layer=1,
    ) == expected


def test_counter_rejects_invalid_identity_and_coordinates() -> None:
    with pytest.raises(ValueError, match="sha256"):
        coordinate_counter_u64(
            profile_sha256="bad", seed=0, frame=0, x=0, y=0, layer=0
        )
    with pytest.raises(ValueError, match="nonnegative"):
        coordinate_counter_u64(
            profile_sha256=SHA_A, seed=0, frame=0, x=-1, y=0, layer=0
        )
    with pytest.raises(ValueError, match="signed 64-bit"):
        coordinate_counter_u64(
            profile_sha256=SHA_A, seed=2**63, frame=0, x=0, y=0, layer=0
        )
