from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from src.color_match import (
    MATCH_PROFILE_DISPLAY_SRGB,
    PreparedMatchViewV1,
    evaluate_successor_declaration_v1,
    invoke_dpct_package_v2,
    load_dpct_invocation_profile_v2,
    make_match_view,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT.parent / "追色"
PROFILE_PATH = (
    ROOT / "configs/reference_match_bmkl_invocation_profile_v2.json"
)
DECLARATION_PATH = (
    ROOT / "configs/reference_match_bmkl_successor_declaration_v1.json"
)
PROFILE_SCHEMA_PATH = (
    ROOT
    / "configs/schemas/reference_dpct_invocation_profile_v2.schema.json"
)
DECLARATION_SCHEMA_PATH = (
    ROOT
    / "configs/schemas/reference_match_successor_declaration_v1.schema.json"
)
WHEEL = (
    PRODUCER
    / "outputs/tmp/bmkl-wheel-bf4fd4f/build1/"
    "zhuise_research-0.3.0-py3-none-any.whl"
)
PYTHON = (
    PRODUCER
    / "outputs/tmp/bmkl-wheel-bf4fd4f/venv/Scripts/python.exe"
)
PRODUCER_LOCK = (
    PRODUCER / "docs/freeze/BMKL_PRODUCER_INVOCATION_PACKAGE_V2.json"
)
PRODUCER_REQUEST_SCHEMA = (
    PRODUCER / "schemas/zhuise_invocation_request_v2.schema.json"
)
PRODUCER_RESPONSE_SCHEMA = (
    PRODUCER / "schemas/zhuise_invocation_response_v2.schema.json"
)
PRODUCER_FIXTURE = (
    PRODUCER
    / "tests/fixtures/zhuise_bmkl_invocation_installed_wheel_v2.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared(offset: float) -> PreparedMatchViewV1:
    pixels = np.linspace(
        0.02 + offset,
        0.92 - offset,
        11 * 13 * 3,
        dtype=np.float32,
    ).reshape(11, 13, 3)
    pixels = np.ascontiguousarray(pixels)
    pixels.flags.writeable = False
    wire = pixels.astype(">f4", copy=False).tobytes(order="C")
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=pixels.shape,
        render_bridge_id="neuro-film.bmkl-successor-test.v1",
        provenance_fingerprint=hashlib.sha256(
            f"bmkl:{offset}".encode()
        ).hexdigest(),
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def test_bmkl_profile_and_declaration_are_strict_and_evaluation_ready() -> None:
    profile_payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    declaration = json.loads(
        DECLARATION_PATH.read_text(encoding="utf-8")
    )
    profile_schema = json.loads(
        PROFILE_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    declaration_schema = json.loads(
        DECLARATION_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    Draft202012Validator(profile_schema).validate(profile_payload)
    Draft202012Validator(declaration_schema).validate(declaration)
    profile = load_dpct_invocation_profile_v2(profile_payload)
    decision = evaluate_successor_declaration_v1(declaration)
    assert profile.capability_id == declaration["wire"]["capability_id"]
    assert profile.wheel_sha256 == declaration["package"]["wheel_sha256"]
    assert decision.evaluation_ready is True
    assert decision.evaluation_reasons == ()
    assert decision.product_ready is False
    assert "product-evidence-absent" in decision.product_reasons
    assert "commercial-rights-absent" in decision.product_reasons


def test_bmkl_profile_pins_exact_producer_artifacts() -> None:
    paths = (
        PRODUCER_LOCK,
        PRODUCER_REQUEST_SCHEMA,
        PRODUCER_RESPONSE_SCHEMA,
        PRODUCER_FIXTURE,
        WHEEL,
    )
    if not all(path.is_file() for path in paths):
        pytest.skip("fixed BMKL producer artifacts are unavailable")
    profile = load_dpct_invocation_profile_v2(
        json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    )
    assert _sha256(PRODUCER_LOCK) == profile.producer_package_lock_sha256
    assert _sha256(PRODUCER_REQUEST_SCHEMA) == profile.request_schema_sha256
    assert _sha256(PRODUCER_RESPONSE_SCHEMA) == profile.response_schema_sha256
    assert _sha256(PRODUCER_FIXTURE) == profile.fixture_sha256
    assert WHEEL.stat().st_size == profile.wheel_size_bytes
    assert _sha256(WHEEL) == profile.wheel_sha256


def test_bmkl_exact_wheel_invokes_through_consumer_profile(
    tmp_path: Path,
) -> None:
    if not WHEEL.is_file() or not PYTHON.is_file():
        pytest.skip("fixed BMKL wheel runtime is unavailable")
    profile = load_dpct_invocation_profile_v2(
        json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    )
    outcome = invoke_dpct_package_v2(
        profile=profile,
        source=_prepared(0.0),
        reference=_prepared(0.01),
        intent_id="d" * 64,
        wheel_path=WHEEL,
        python_executable=PYTHON,
        scratch_directory=tmp_path,
    )
    assert outcome.status == "candidate"
    assert outcome.candidate is not None
    assert outcome.failure is None
    assert outcome.invocation_profile_id == profile.profile_id
    assert outcome.capability_id == profile.capability_id
    assert outcome.candidate.aliases.producer_commit == (
        profile.producer_stable_commit
    )
    assert outcome.candidate.aliases.compatibility_profile_id == (
        profile.lower_compatibility_profile_id
    )
    assert outcome.candidate.aliases.capability_id == profile.capability_id
    assert list(tmp_path.iterdir()) == []
