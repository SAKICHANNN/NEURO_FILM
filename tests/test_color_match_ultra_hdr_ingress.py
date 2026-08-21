from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.core_contracts import MATCH_PROFILE_ABSOLUTE_REC2020
from src.color_match.ultra_hdr_ingress import (
    ULTRAHDR_DECODER_VERSION,
    ULTRAHDR_EXTERNAL_PROFILE_ID,
    ULTRAHDR_REFERENCE_WHITE_NITS,
    prepare_ultrahdr_match_view_v1,
    validate_prepared_ultrahdr_match_view_v1,
)

SOURCE = b"self-authored-ultrahdr-fixture"
SOURCE_SHA256 = hashlib.sha256(SOURCE).hexdigest()


def _rgba() -> np.ndarray:
    return np.array(
        [
            [[0.0, 0.5, 1.0, 1.0], [2.0, 4.0, 8.0, 1.0]],
            [[16.0, 24.0, 32.0, 1.0], [49.25, 1.5, 3.0, 1.0]],
        ],
        dtype="<f2",
    )


def _prepare(payload: bytes | None = None, **overrides):
    arguments = {
        "source_asset": SOURCE,
        "expected_source_sha256": SOURCE_SHA256,
        "decoded_rgba16f": _rgba().tobytes(order="C") if payload is None else payload,
        "width": 2,
        "height": 2,
        "decoder_version": ULTRAHDR_DECODER_VERSION,
        "producer_profile_id": ULTRAHDR_EXTERNAL_PROFILE_ID,
    }
    arguments.update(overrides)
    return prepare_ultrahdr_match_view_v1(**arguments)


def test_ingress_is_exact_absolute_rec2020_and_deterministic() -> None:
    payload = _rgba().tobytes(order="C")
    first = _prepare(payload)
    second = _prepare(payload)

    expected = _rgba()[..., :3].astype(np.float32)
    expected *= np.float32(ULTRAHDR_REFERENCE_WHITE_NITS)
    assert np.array_equal(first.pixels, expected)
    assert first.pixels.dtype == np.float32
    assert first.pixels.flags.c_contiguous
    assert not first.pixels.flags.writeable
    assert first.descriptor.profile_id == MATCH_PROFILE_ABSOLUTE_REC2020
    assert first.descriptor.domain == "display-absolute-linear"
    assert first.descriptor.primaries == "rec2020"
    assert first.descriptor.reference_white_nits == 203.0
    assert first.descriptor == second.descriptor
    assert first.ingress_id == second.ingress_id
    assert first.decoded_payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert payload == _rgba().tobytes(order="C")


def test_ingress_copies_mutable_inputs_before_publication() -> None:
    source = bytearray(SOURCE)
    payload = bytearray(_rgba().tobytes(order="C"))
    result = _prepare(
        payload,
        source_asset=source,
    )
    frozen_pixels = result.pixels.copy()
    source[0] ^= 0xFF
    payload[0] ^= 0xFF
    assert np.array_equal(result.pixels, frozen_pixels)
    validate_prepared_ultrahdr_match_view_v1(result)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda args: args.update(source_asset=b""), "non-empty"),
        (
            lambda args: args.update(source_asset="not-bytes"),
            "bytes-like",
        ),
        (
            lambda args: args.update(expected_source_sha256="0" * 64),
            "identity mismatch",
        ),
        (
            lambda args: args.update(decoder_version="2.0.1"),
            "decoder version",
        ),
        (
            lambda args: args.update(producer_profile_id="other"),
            "producer profile",
        ),
        (lambda args: args.update(width=0), "positive integer"),
        (
            lambda args: args.update(decoded_rgba16f=args["decoded_rgba16f"][:-2]),
            "length mismatch",
        ),
    ],
)
def test_ingress_rejects_invalid_binding_before_publication(
    mutator,
    match: str,
) -> None:
    arguments = {
        "source_asset": SOURCE,
        "expected_source_sha256": SOURCE_SHA256,
        "decoded_rgba16f": _rgba().tobytes(order="C"),
        "width": 2,
        "height": 2,
        "decoder_version": ULTRAHDR_DECODER_VERSION,
        "producer_profile_id": ULTRAHDR_EXTERNAL_PROFILE_ID,
    }
    mutator(arguments)
    with pytest.raises(ReferenceMatchContractError, match=match):
        prepare_ultrahdr_match_view_v1(**arguments)


@pytest.mark.parametrize(
    ("channel", "value", "match"),
    [
        ((0, 0, 0), np.inf, "finite"),
        ((0, 0, 0), -0.5, "nominal linear range"),
        ((0, 0, 0), 49.5, "nominal linear range"),
        ((0, 0, 3), 0.5, "alpha must be exactly one"),
    ],
)
def test_ingress_rejects_invalid_decoded_samples(
    channel: tuple[int, int, int],
    value: float,
    match: str,
) -> None:
    rgba = _rgba()
    rgba[channel] = value
    with pytest.raises(ReferenceMatchContractError, match=match):
        _prepare(rgba.tobytes(order="C"))


def test_validator_rejects_descriptor_or_ingress_tamper() -> None:
    prepared = _prepare()
    with pytest.raises(ReferenceMatchContractError, match="view_id"):
        validate_prepared_ultrahdr_match_view_v1(
            replace(
                prepared,
                prepared_view=replace(
                    prepared.prepared_view,
                    descriptor=replace(
                        prepared.descriptor,
                        reference_white_nits=100.0,
                    ),
                ),
            )
        )
    with pytest.raises(ReferenceMatchContractError, match="identity"):
        validate_prepared_ultrahdr_match_view_v1(replace(prepared, ingress_id="0" * 64))
