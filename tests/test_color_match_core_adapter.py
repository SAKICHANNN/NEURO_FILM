from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.color_match import (
    MATCH_PROFILE_DISPLAY_REC2020,
    MATCH_PROFILE_DISPLAY_SRGB,
    MATCH_PROFILE_SCENE_ACESCG,
    PreparedMatchViewV1,
    ReferenceMatchContractError,
    make_capabilities,
    prepare_working_image_match_view,
    validate_prepared_match_view,
    validate_prepared_view_support,
)
from src.preprocess import DecodeWarning, SourceProfile, WorkingImage


SHA_A = "1" * 64
CONTRACT_ID = "zhuise.transform-bundle.v1"
ALGORITHM_ID = "zhuise.dpct.v1"


def _working(
    *,
    working_space: str = "linear_srgb",
    transfer_state: str = "display_linear",
    alpha_policy: str = "absent",
    orientation_applied: bool = True,
    source_path: str = "source-a.png",
    metadata: dict | None = None,
    sliced: bool = False,
) -> WorkingImage:
    base = np.linspace(
        0.01,
        0.99,
        num=6 * 8 * 3,
        dtype=np.float32,
    ).reshape(6, 8, 3)
    pixels = base[:, ::2, :] if sliced else base.copy()
    return WorkingImage(
        pixels=pixels,
        working_space=working_space,
        transfer_state=transfer_state,
        source_transfer_state="display_referred",
        source_profile=SourceProfile(
            "cicp" if working_space == "linear_rec2020" else "assumed_srgb",
            "adapter fixture",
            12,
        ),
        hdr_metadata={} if metadata is None else metadata,
        orientation_applied=orientation_applied,
        alpha_policy=alpha_policy,
        bit_depth_in=16,
        source_path=Path(source_path),
        warnings=[DecodeWarning("fixture", "adapter test")],
    )


def _capabilities(*, profiles) -> object:
    return make_capabilities(
        producer_id="zhuise-core",
        producer_version="0.1.0",
        producer_build_sha256=SHA_A,
        contract_schema_ids=[CONTRACT_ID],
        supported_profile_ids=profiles,
        supported_algorithm_ids=[ALGORITHM_ID],
        feature_flags=["source-bound-fit"],
    )


@pytest.mark.parametrize(
    ("working_space", "profile_id"),
    [
        ("linear_srgb", MATCH_PROFILE_DISPLAY_SRGB),
        ("linear_rec2020", MATCH_PROFILE_DISPLAY_REC2020),
    ],
)
def test_prepare_working_image_maps_exact_relative_display_profile(
    working_space: str,
    profile_id: str,
) -> None:
    image = _working(working_space=working_space)
    original = image.pixels.copy()
    prepared = prepare_working_image_match_view(image)
    assert prepared.descriptor.profile_id == profile_id
    assert prepared.descriptor.domain == "display-relative-linear"
    assert prepared.descriptor.reference_white_nits is None
    assert prepared.descriptor.shape == tuple(image.pixels.shape)
    assert prepared.descriptor.strides_bytes == prepared.pixels.strides
    assert prepared.pixels.flags.c_contiguous
    assert not prepared.pixels.flags.writeable
    assert np.array_equal(image.pixels, original)
    assert not np.shares_memory(prepared.pixels, image.pixels)


def test_prepare_noncontiguous_input_produces_dense_isolated_view() -> None:
    image = _working(sliced=True)
    assert not image.pixels.flags.c_contiguous
    prepared = prepare_working_image_match_view(image)
    validate_prepared_match_view(prepared)
    assert prepared.pixels.flags.c_contiguous
    assert prepared.descriptor.strides_bytes == (
        prepared.pixels.shape[1] * 12,
        12,
        4,
    )


def test_view_identity_ignores_local_path_but_binds_metadata() -> None:
    first = prepare_working_image_match_view(
        _working(source_path="one/source.png")
    )
    moved = prepare_working_image_match_view(
        _working(source_path="two/source.png")
    )
    changed = prepare_working_image_match_view(
        _working(metadata={"cicp": "09001001"})
    )
    assert first.descriptor == moved.descriptor
    assert first.descriptor.pixel_sha256 == changed.descriptor.pixel_sha256
    assert (
        first.descriptor.provenance_fingerprint
        != changed.descriptor.provenance_fingerprint
    )
    assert first.descriptor.view_id != changed.descriptor.view_id


@pytest.mark.parametrize(
    "transfer_state",
    ["scene_linear", "display_referred", "unknown"],
)
def test_adapter_rejects_non_display_linear_rails(
    transfer_state: str,
) -> None:
    with pytest.raises(
        ReferenceMatchContractError,
        match="display-linear",
    ):
        prepare_working_image_match_view(
            _working(transfer_state=transfer_state)
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"working_space": "acescg"}, "working space"),
        ({"alpha_policy": "composited"}, "alpha-free"),
        ({"orientation_applied": False}, "applied orientation"),
    ],
)
def test_adapter_rejects_unsupported_working_contract(
    changes: dict,
    message: str,
) -> None:
    with pytest.raises(ReferenceMatchContractError, match=message):
        prepare_working_image_match_view(_working(**changes))


def test_prepared_view_detects_buffer_or_descriptor_drift() -> None:
    prepared = prepare_working_image_match_view(_working())
    writable = prepared.pixels.copy()
    with pytest.raises(ReferenceMatchContractError, match="read-only"):
        validate_prepared_match_view(
            PreparedMatchViewV1(prepared.descriptor, writable)
        )
    altered = prepared.pixels.copy()
    altered[0, 0, 0] += np.float32(0.01)
    altered.flags.writeable = False
    with pytest.raises(ReferenceMatchContractError, match="identity"):
        validate_prepared_match_view(
            PreparedMatchViewV1(prepared.descriptor, altered)
        )
    cropped = prepared.pixels[:1, :1, :].copy()
    cropped.flags.writeable = False
    with pytest.raises(ReferenceMatchContractError, match="shape"):
        validate_prepared_match_view(
            PreparedMatchViewV1(prepared.descriptor, cropped)
        )


def test_exact_capabilities_allow_prepared_view() -> None:
    prepared = prepare_working_image_match_view(_working())
    validate_prepared_view_support(
        prepared,
        _capabilities(profiles=[MATCH_PROFILE_DISPLAY_SRGB]),
        algorithm_id=ALGORITHM_ID,
        contract_schema_id=CONTRACT_ID,
        capability_requirements=["source-bound-fit"],
    )


@pytest.mark.parametrize(
    ("profiles", "algorithm", "contract", "requirements", "message"),
    [
        (
            [MATCH_PROFILE_SCENE_ACESCG],
            ALGORITHM_ID,
            CONTRACT_ID,
            (),
            "profile is not advertised",
        ),
        (
            [MATCH_PROFILE_DISPLAY_SRGB],
            "other.algorithm",
            CONTRACT_ID,
            (),
            "algorithm is not advertised",
        ),
        (
            [MATCH_PROFILE_DISPLAY_SRGB],
            ALGORITHM_ID,
            "other.contract",
            (),
            "contract schema is not advertised",
        ),
        (
            [MATCH_PROFILE_DISPLAY_SRGB],
            ALGORITHM_ID,
            CONTRACT_ID,
            ("spatial-context-lut",),
            "missing required core capabilities",
        ),
    ],
)
def test_capability_gate_fails_closed(
    profiles,
    algorithm,
    contract,
    requirements,
    message,
) -> None:
    prepared = prepare_working_image_match_view(_working())
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_prepared_view_support(
            prepared,
            _capabilities(profiles=profiles),
            algorithm_id=algorithm,
            contract_schema_id=contract,
            capability_requirements=requirements,
        )
