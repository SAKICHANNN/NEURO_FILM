from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from src.color_match import (
    DPCT_INVOCATION_CLAIM_CEILING,
    DPCT_INVOCATION_COMPATIBILITY_PROFILE_ID,
    DPCT_INVOCATION_STABLE_COMMIT,
    DPCT_INVOCATION_WHEEL_SHA256,
    MATCH_PROFILE_DISPLAY_SRGB,
    PreparedMatchViewV1,
    ReferenceMatchContractError,
    invoke_dpct_package_v1,
    make_match_view,
    prepare_dpct_invocation_request_v1,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT.parent / "追色"
LOCK = ROOT / "configs" / "reference_match_dpct_invocation_v1.json"
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_dpct_invocation_compatibility_lock_v1.schema.json"
)
WHEEL = (
    PRODUCER
    / "outputs"
    / "tmp"
    / "producer_invocation_dist_c"
    / "zhuise_research-0.2.0-py3-none-any.whl"
)
STALE_WHEEL = (
    PRODUCER
    / "outputs"
    / "tmp"
    / "producer_invocation_dist"
    / "zhuise_research-0.2.0-py3-none-any.whl"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared(height: int, width: int, offset: float) -> PreparedMatchViewV1:
    values = np.linspace(
        0.03 + offset,
        0.91 - offset,
        height * width * 3,
        dtype=np.float32,
    ).reshape(height, width, 3)
    values = np.ascontiguousarray(values)
    values.flags.writeable = False
    wire = values.astype(">f4", copy=False).tobytes(order="C")
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=values.shape,
        render_bridge_id="neuro-film.invocation-test.v1",
        provenance_fingerprint=hashlib.sha256(
            f"{height}x{width}:{offset}".encode()
        ).hexdigest(),
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=values)


def test_invocation_lock_is_strict_and_matches_producer_authority() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(lock)
    assert lock["package"]["wheel_sha256"] == DPCT_INVOCATION_WHEEL_SHA256
    assert lock["producer"]["stable_commit"] == (
        DPCT_INVOCATION_STABLE_COMMIT
    )
    producer_files = {
        "package_lock_sha256": (
            PRODUCER / "docs/freeze/PRODUCER_INVOCATION_PACKAGE_V1.json"
        ),
        "request_schema_sha256": (
            PRODUCER / "schemas/zhuise_invocation_request_v1.schema.json"
        ),
        "response_schema_sha256": (
            PRODUCER / "schemas/zhuise_invocation_response_v1.schema.json"
        ),
        "fixture_sha256": (
            PRODUCER
            / "tests/fixtures/"
            "zhuise_producer_invocation_installed_wheel_v1.json"
        ),
    }
    if not all(path.is_file() for path in producer_files.values()):
        pytest.skip("pinned producer audit artifacts are unavailable")
    for key, path in producer_files.items():
        section = (
            lock["producer"] if key == "package_lock_sha256" else lock["wire"]
        )
        assert _sha256(path) == section[key]


def test_request_is_repeatable_and_source_reference_bound() -> None:
    source = _prepared(8, 9, 0.0)
    reference = _prepared(7, 10, 0.01)
    first = prepare_dpct_invocation_request_v1(
        source=source, reference=reference
    )
    second = prepare_dpct_invocation_request_v1(
        source=source, reference=reference
    )
    assert first == second
    request, source_wire, reference_wire = first
    assert request["capability_id"] == (
        "zhuise.dpct-chroma.cpu-reference.v1"
    )
    assert request["source"]["view"]["pixel_sha256"] == (
        "sha256:" + hashlib.sha256(source_wire).hexdigest()
    )
    assert request["reference"]["view"]["pixel_sha256"] == (
        "sha256:" + hashlib.sha256(reference_wire).hexdigest()
    )
    changed = prepare_dpct_invocation_request_v1(
        source=_prepared(8, 9, 0.02),
        reference=reference,
    )[0]
    assert changed["request_id"] != request["request_id"]


def test_same_name_stale_wheel_fails_before_scratch_mutation(
    tmp_path: Path,
) -> None:
    if not STALE_WHEEL.is_file():
        pytest.skip("stale producer wheel audit artifact is unavailable")
    with pytest.raises(
        ReferenceMatchContractError,
        match="wheel identity mismatch",
    ):
        invoke_dpct_package_v1(
            source=_prepared(4, 5, 0.0),
            reference=_prepared(5, 4, 0.01),
            intent_id="1" * 64,
            wheel_path=STALE_WHEEL,
            python_executable=Path(sys.executable),
            scratch_directory=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_exact_wheel_invokes_and_adapts_candidate(
    tmp_path: Path,
) -> None:
    if not WHEEL.is_file():
        pytest.skip("pinned producer wheel is unavailable")
    assert WHEEL.stat().st_size == 95994
    assert _sha256(WHEEL) == DPCT_INVOCATION_WHEEL_SHA256
    outcome = invoke_dpct_package_v1(
        source=_prepared(8, 9, 0.0),
        reference=_prepared(7, 10, 0.01),
        intent_id="2" * 64,
        wheel_path=WHEEL,
        python_executable=Path(sys.executable),
        scratch_directory=tmp_path,
    )
    assert outcome.status == "candidate"
    assert outcome.candidate is not None
    assert outcome.failure is None
    assert outcome.compatibility_profile_id == (
        DPCT_INVOCATION_COMPATIBILITY_PROFILE_ID
    )
    assert outcome.producer_stable_commit == (
        "e22725d8524ed6ba56f37180abc400213908c6f4"
    )
    assert outcome.claim_ceiling == DPCT_INVOCATION_CLAIM_CEILING
    assert outcome.candidate.aliases.capability_id == (
        "zhuise.dpct-chroma.cpu-reference.v1"
    )
    assert outcome.candidate.prepared_output.pixels.shape == (8, 9, 3)
    assert outcome.candidate.prepared_output.pixels.flags.writeable is False
    assert list(tmp_path.iterdir()) == []


def test_wrong_runtime_fails_closed_without_invocation(
    tmp_path: Path,
) -> None:
    if not WHEEL.is_file():
        pytest.skip("pinned producer wheel is unavailable")
    python314 = Path(
        r"C:\Users\hhvrf\AppData\Local\Programs\Python\Python314\python.exe"
    )
    if not python314.is_file():
        pytest.skip("mismatched Python runtime is unavailable")
    with pytest.raises(ReferenceMatchContractError):
        invoke_dpct_package_v1(
            source=_prepared(4, 4, 0.0),
            reference=_prepared(4, 4, 0.01),
            intent_id="3" * 64,
            wheel_path=WHEEL,
            python_executable=python314,
            scratch_directory=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []
