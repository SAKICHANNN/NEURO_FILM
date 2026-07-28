from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from scripts.build_srgb_icc_profile_conformance_v1 import (
    encode_conformance,
)
from src.color_match.contracts import ReferenceMatchContractError
import src.color_match.srgb_icc_profile as profile
from src.color_match.shared_runtime_staging_attested_match_views import (
    prepare_runtime_staging_attested_match_views_v2,
)
from src.color_match.shared_runtime_staging_color_attestation import (
    attest_runtime_staging_srgb_metadata_v1,
)
from src.color_match.shared_runtime_staging_decode import (
    decode_runtime_qualified_shared_staging_bytes_v1,
)
import src.preprocess.output_encode as output_encode
from tests.test_color_match_shared_runtime_staging_decode import (
    _real_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "reference_srgb_icc_profile_conformance_v1.json"
)
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_srgb_icc_profile_conformance_v1.schema.json"
)


def test_pinned_profile_identity_and_header_are_exact() -> None:
    raw = profile.srgb_icc_profile_v1()

    assert len(raw) == profile.SRGB_ICC_PROFILE_SIZE_BYTES == 588
    assert hashlib.sha256(raw).hexdigest() == (
        profile.SRGB_ICC_PROFILE_SHA256
    )
    assert int.from_bytes(raw[0:4], "big") == 588
    assert raw[12:16] == b"mntr"
    assert raw[16:20] == b"RGB "
    assert raw[20:24] == b"XYZ "
    assert raw[36:40] == b"acsp"


def test_conformance_fixture_is_canonical_and_schema_valid() -> None:
    encoded = encode_conformance()
    payload = json.loads(encoded)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    assert FIXTURE.read_bytes() == encoded.encode("utf-8")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    assert base64.b64decode(
        payload["profile_base64"], validate=True
    ) == profile.srgb_icc_profile_v1()


@pytest.mark.parametrize(
    "field",
    ["_SRGB_ICC_PROFILE_BASE64", "SRGB_ICC_PROFILE_SHA256"],
)
def test_pinned_profile_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    value = getattr(profile, field)
    if field == "_SRGB_ICC_PROFILE_BASE64":
        value = value[:1] + ("A" if value[1] != "A" else "B") + value[2:]
    else:
        value = "0" * 64
    monkeypatch.setattr(profile, field, value)

    with pytest.raises(
        ReferenceMatchContractError,
        match="pinned sRGB ICC profile",
    ):
        profile.srgb_icc_profile_v1()


def test_runtime_attestation_consumes_pinned_bytes_not_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _real_snapshot(tmp_path, suffix=".png", depth=16)
    decoded = decode_runtime_qualified_shared_staging_bytes_v1(snapshot)

    def _unexpected_generator_call() -> bytes:
        raise AssertionError("runtime profile generator was called")

    monkeypatch.setattr(
        output_encode,
        "srgb_icc_profile",
        _unexpected_generator_call,
    )
    attested = attest_runtime_staging_srgb_metadata_v1(decoded)
    prepared = prepare_runtime_staging_attested_match_views_v2(attested)

    assert attested.record.expected_profile_sha256 == (
        profile.SRGB_ICC_PROFILE_SHA256
    )
    assert prepared.record.expected_profile_sha256 == (
        profile.SRGB_ICC_PROFILE_SHA256
    )


def test_current_host_encoder_matches_pinned_profile() -> None:
    assert output_encode.srgb_icc_profile() == profile.srgb_icc_profile_v1()
