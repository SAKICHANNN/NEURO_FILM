from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from copy import deepcopy

import numpy as np
import pytest

from src.color_match import (
    DpctInvocationProfileV2,
    MATCH_PROFILE_DISPLAY_SRGB,
    PreparedMatchViewV1,
    ReferenceMatchContractError,
    invoke_dpct_package_v2,
    issue_dpct_invocation_profile_v2,
    make_match_view,
    prepare_dpct_invocation_request_v1,
    prepare_dpct_invocation_request_v2,
    verify_dpct_invocation_output_v2,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT.parent / "追色"
WHEEL = (
    PRODUCER
    / "outputs/tmp/producer_invocation_dist_c/"
    "zhuise_research-0.2.0-py3-none-any.whl"
)


def _profile(**changes: object) -> DpctInvocationProfileV2:
    values: dict[str, object] = {
        "schema_id": "placeholder",
        "profile_id": "0" * 64,
        "producer_repository_id": "zhuise-dpct",
        "producer_stable_commit": (
            "e22725d8524ed6ba56f37180abc400213908c6f4"
        ),
        "producer_package_source_commit": (
            "01ef0616610b16246623236301234ff3b4c4a7f2"
        ),
        "producer_fixture_commit": (
            "eb4b889bc18f1aba078e19e7e62fbf8c4add6e54"
        ),
        "producer_package_lock_sha256": (
            "300b95b05954b8f1de7aaec6f86429bf584a40088e15d07d"
            "d439384c73090287"
        ),
        "package_distribution": "zhuise-research",
        "package_version": "0.2.0",
        "package_entrypoint": "zhuise-producer-invoke",
        "wheel_filename": "zhuise_research-0.2.0-py3-none-any.whl",
        "wheel_size_bytes": 95994,
        "wheel_sha256": (
            "fd995ad88c9f30f2136508f7c3879ce6768fde7ff2e149537"
            "d2b58f78e36c292"
        ),
        "python_major_minor": "3.12",
        "numpy_version": "2.4.4",
        "pillow_version": "12.1.1",
        "request_schema": "zhuise.invocation-request.v1",
        "request_schema_sha256": (
            "bac2866842d31c1d42e8b813da5c6632bdb9de49ef296e75"
            "42209639aa9b37b8"
        ),
        "response_schema": "zhuise.invocation-response.v1",
        "response_schema_sha256": (
            "204da4f668e69beba7705f8ab0d7966d962c5d6933870bde"
            "481bbaee8a15473e"
        ),
        "fixture_sha256": (
            "eab24eaed81e3602de519ce4ba89e640bf8185a452d4104b2"
            "577b54fdf15b295"
        ),
        "capability_id": "zhuise.dpct-chroma.cpu-reference.v1",
        "producer_profile_id": (
            "zhuise.display-linear-srgb-d65-relative-f32.v1"
        ),
        "lower_compatibility_profile_id": (
            "neuro-film.dpct-consumer.v2"
        ),
        "evaluation_allowed": True,
        "redistribution_allowed": False,
        "rights_reason": "local research integration only",
        "claim_ceiling": "placeholder",
    }
    values.update(changes)
    return issue_dpct_invocation_profile_v2(
        DpctInvocationProfileV2(**values)  # type: ignore[arg-type]
    )


def _prepared(offset: float) -> PreparedMatchViewV1:
    pixels = np.linspace(
        0.05 + offset,
        0.9 - offset,
        8 * 9 * 3,
        dtype=np.float32,
    ).reshape(8, 9, 3)
    pixels = np.ascontiguousarray(pixels)
    pixels.flags.writeable = False
    wire = pixels.astype(">f4", copy=False).tobytes(order="C")
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=pixels.shape,
        render_bridge_id="neuro-film.invocation-v2-test.v1",
        provenance_fingerprint=hashlib.sha256(
            f"invocation-v2:{offset}".encode()
        ).hexdigest(),
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def test_profile_v2_preserves_existing_wire_request_exactly() -> None:
    source = _prepared(0.0)
    reference = _prepared(0.01)
    assert prepare_dpct_invocation_request_v2(
        profile=_profile(),
        source=source,
        reference=reference,
    ) == prepare_dpct_invocation_request_v1(
        source=source,
        reference=reference,
    )


def test_profile_v2_invokes_exact_wheel_and_binds_profile(
    tmp_path: Path,
) -> None:
    if not WHEEL.is_file():
        pytest.skip("pinned producer wheel is unavailable")
    profile = _profile()
    outcome = invoke_dpct_package_v2(
        profile=profile,
        source=_prepared(0.0),
        reference=_prepared(0.01),
        intent_id="a" * 64,
        wheel_path=WHEEL,
        python_executable=Path(sys.executable),
        scratch_directory=tmp_path,
    )
    assert outcome.status == "candidate"
    assert outcome.candidate is not None
    assert outcome.failure is None
    assert outcome.invocation_profile_id == profile.profile_id
    assert outcome.capability_id == profile.capability_id
    assert outcome.wheel_sha256 == profile.wheel_sha256
    assert outcome.candidate.aliases.capability_id == profile.capability_id
    assert list(tmp_path.iterdir()) == []


def test_profile_v2_wrong_exact_wheel_fails_before_scratch_mutation(
    tmp_path: Path,
) -> None:
    if not WHEEL.is_file():
        pytest.skip("pinned producer wheel is unavailable")
    profile = _profile(wheel_sha256="f" * 64)
    with pytest.raises(
        ReferenceMatchContractError,
        match="wheel identity mismatch",
    ):
        invoke_dpct_package_v2(
            profile=profile,
            source=_prepared(0.0),
            reference=_prepared(0.01),
            intent_id="b" * 64,
            wheel_path=WHEEL,
            python_executable=Path(sys.executable),
            scratch_directory=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda request: request.update(capability_id="zhuise.other.v1"),
            "request identity mismatch",
        ),
        (
            lambda request: request.update(request_id="sha256:" + "0" * 64),
            "request ID mismatch",
        ),
        (
            lambda request: request["source"].update(
                pixels_file="reference.f32be"
            ),
            "source binding mismatch",
        ),
    ],
)
def test_profile_v2_verifier_rejects_tampered_request_before_output_read(
    tmp_path: Path,
    mutation: object,
    match: str,
) -> None:
    profile = _profile()
    source = _prepared(0.0)
    reference = _prepared(0.01)
    request = deepcopy(
        prepare_dpct_invocation_request_v2(
            profile=profile,
            source=source,
            reference=reference,
        )[0]
    )
    mutation(request)  # type: ignore[operator]
    with pytest.raises(ReferenceMatchContractError, match=match):
        verify_dpct_invocation_output_v2(
            profile=profile,
            output_directory=tmp_path,
            request=request,
            source=source,
            reference=reference,
            intent_id="c" * 64,
        )
