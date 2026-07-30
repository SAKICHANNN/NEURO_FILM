from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
import numpy as np
from PIL import Image
import pytest
import tifffile

from src.color_match.canonical import canonical_sha256
from src.color_match.contracts import ReferenceMatchContractError
import src.color_match.shared_runtime_staging_decode as decode
from src.color_match.shared_runtime_staging_consumption import (
    capture_runtime_qualified_shared_staging_bytes_v1,
)
from src.color_match.shared_runtime_staging_transaction import (
    RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING,
    RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID,
    RuntimeQualifiedExternalSharedStagingRunV1,
    commit_runtime_qualified_external_shared_staging_v1,
    runtime_qualified_external_shared_staging_run_to_json,
)
from src.color_match.shared_staging_transaction import (
    ExternalSharedStagedOutputV1,
)
from tests.test_color_match_shared_runtime_staging_transaction import (
    _pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_runtime_qualified_shared_staging_decode_v1.schema.json"
)


def _real_snapshot(
    tmp_path: Path,
    *,
    suffix: str,
    depth: int,
):
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs = tuple(
        (tmp_path / f"output-{index}{suffix}").resolve()
        for index in range(2)
    )
    report = (tmp_path / "report.json").resolve()
    committed = commit_runtime_qualified_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        runtime_qualification=qualification,
        applies=applies,
        output_paths=outputs,
        report_path=report,
        expected_runtime_qualification_id=qualification.qualification_id,
        output_bit_depth=depth,
    )
    return capture_runtime_qualified_shared_staging_bytes_v1(
        report_path=report,
        expected_report_sha256=committed.report_file_sha256,
        expected_run_id=committed.run.run_id,
        expected_runtime_qualification_id=(
            qualification.qualification_id
        ),
    )


def _bound_snapshot_from_payloads(
    tmp_path: Path,
    payloads: tuple[bytes, ...],
    *,
    format_name: str,
    depth: int,
):
    report_path = (tmp_path / "report.json").resolve()
    outputs: list[ExternalSharedStagedOutputV1] = []
    suffix = {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tif"}[format_name]
    for index, payload in enumerate(payloads):
        path = (tmp_path / f"output-{index}{suffix}").resolve()
        path.write_bytes(payload)
        outputs.append(
            ExternalSharedStagedOutputV1(
                source_index=index,
                source_view_id=hashlib.sha256(
                    f"source-{index}".encode()
                ).hexdigest(),
                apply_receipt_id=hashlib.sha256(
                    f"receipt-{index}".encode()
                ).hexdigest(),
                producer_apply_result_id=(
                    "sha256:"
                    + hashlib.sha256(
                        f"result-{index}".encode()
                    ).hexdigest()
                ),
                diagnostics_id=(
                    "sha256:"
                    + hashlib.sha256(
                        f"diagnostics-{index}".encode()
                    ).hexdigest()
                ),
                output_view_id=hashlib.sha256(
                    f"view-{index}".encode()
                ).hexdigest(),
                output_path=str(path),
                output_file_sha256=hashlib.sha256(payload).hexdigest(),
                output_format=format_name,
                output_bit_depth=depth,
                encode_clipped_fraction=0.0,
            )
        )
    provisional = RuntimeQualifiedExternalSharedStagingRunV1(
        schema_id=RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID,
        run_id="0" * 64,
        runtime_qualification_id=hashlib.sha256(b"qualification").hexdigest(),
        runtime_evidence_id=hashlib.sha256(b"evidence").hexdigest(),
        declaration_id=hashlib.sha256(b"declaration").hexdigest(),
        authorization_id=hashlib.sha256(b"authorization").hexdigest(),
        upstream_batch_id=hashlib.sha256(b"batch").hexdigest(),
        numeric_guard_batch_id=hashlib.sha256(b"numeric").hexdigest(),
        operator_id=hashlib.sha256(b"operator").hexdigest(),
        reference_view_id=hashlib.sha256(b"reference").hexdigest(),
        source_count=len(outputs),
        state="committed-to-runtime-qualified-shared-staging",
        outputs=tuple(outputs),
        report_path=str(report_path),
        claim_ceiling=RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING,
    )
    identity = provisional.to_dict()
    identity.pop("run_id")
    run = replace(provisional, run_id=canonical_sha256(identity))
    encoded = runtime_qualified_external_shared_staging_run_to_json(
        run
    ).encode()
    report_path.write_bytes(encoded)
    return capture_runtime_qualified_shared_staging_bytes_v1(
        report_path=report_path,
        expected_report_sha256=hashlib.sha256(encoded).hexdigest(),
        expected_run_id=run.run_id,
        expected_runtime_qualification_id=run.runtime_qualification_id,
    )


@pytest.mark.parametrize(
    ("suffix", "depth", "format_name", "dtype"),
    [
        (".png", 16, "PNG", np.uint16),
        (".tiff", 16, "TIFF", np.uint16),
        (".png", 8, "PNG", np.uint8),
        (".jpg", 8, "JPEG", np.uint8),
        (".tif", 8, "TIFF", np.uint8),
    ],
)
def test_real_p62_p65_bytes_decode_exactly_without_paths(
    tmp_path: Path,
    suffix: str,
    depth: int,
    format_name: str,
    dtype: np.dtype,
) -> None:
    snapshot = _real_snapshot(tmp_path, suffix=suffix, depth=depth)
    result = decode.decode_runtime_qualified_shared_staging_bytes_v1(
        snapshot
    )

    assert result.record.path_consumption_authorized is False
    assert result.record.persistent_pixels_authorized is False
    assert result.record.delivery_authorized is False
    assert all(row.output_format == format_name for row in result.record.outputs)
    assert all(array.dtype == dtype for array in result.pixels)
    assert all(array.shape == (4, 5, 3) for array in result.pixels)
    assert all(not array.flags.writeable for array in result.pixels)
    assert "report_path" not in result.record.to_dict()
    assert all(
        "output_path" not in output
        for output in result.record.to_dict()["outputs"]
    )
    decode.validate_runtime_qualified_shared_staging_decoded_batch_v1(
        result
    )
    encoded = decode.runtime_qualified_shared_staging_decode_record_to_json(
        result.record
    )
    assert (
        decode.runtime_qualified_shared_staging_decode_record_from_json(
            encoded
        )
        == result.record
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(result.record.to_dict())


def test_invalid_bound_encoded_bytes_fail_before_any_decoded_batch(
    tmp_path: Path,
) -> None:
    snapshot = _bound_snapshot_from_payloads(
        tmp_path,
        (b"not-a-png", b"also-not-a-png"),
        format_name="PNG",
        depth=8,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="PNG signature",
    ):
        decode.decode_runtime_qualified_shared_staging_bytes_v1(snapshot)


def test_second_preflight_failure_allocates_no_decoded_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(buffer, format="PNG")
    snapshot = _bound_snapshot_from_payloads(
        tmp_path,
        (buffer.getvalue(), b"invalid-second-png"),
        format_name="PNG",
        depth=8,
    )
    calls = 0
    original = decode._decode_one

    def tracking(raw: bytes, format_name: str, depth: int) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(raw, format_name, depth)

    monkeypatch.setattr(decode, "_decode_one", tracking)
    with pytest.raises(ReferenceMatchContractError, match="PNG signature"):
        decode.decode_runtime_qualified_shared_staging_bytes_v1(snapshot)
    assert calls == 0


def test_declared_format_mismatch_fails_closed(tmp_path: Path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(buffer, format="PNG")
    snapshot = _bound_snapshot_from_payloads(
        tmp_path,
        (buffer.getvalue(),),
        format_name="JPEG",
        depth=8,
    )
    with pytest.raises(ReferenceMatchContractError, match="JPEG boundary"):
        decode.decode_runtime_qualified_shared_staging_bytes_v1(snapshot)


def test_strict_png_rejects_trailing_crc_alpha_and_animation(
    tmp_path: Path,
) -> None:
    raw = _real_snapshot(tmp_path, suffix=".png", depth=8).output_bytes[0]
    with pytest.raises(ReferenceMatchContractError, match="trailing"):
        decode._decode_png(raw + b"trailing", 8)
    corrupted = bytearray(raw)
    corrupted[-8] ^= 1
    with pytest.raises(ReferenceMatchContractError, match="CRC"):
        decode._decode_png(bytes(corrupted), 8)
    buffer = BytesIO()
    Image.new("RGBA", (2, 2), (1, 2, 3, 4)).save(
        buffer, format="PNG"
    )
    with pytest.raises(ReferenceMatchContractError, match="sample contract"):
        decode._decode_png(buffer.getvalue(), 8)


def test_jpeg_rejects_trailing_and_non_rgb() -> None:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(buffer, format="JPEG")
    raw = buffer.getvalue()
    with pytest.raises(ReferenceMatchContractError, match="boundary"):
        decode._decode_jpeg(raw + b"x", 8)
    cmyk = BytesIO()
    Image.new("CMYK", (2, 2), (1, 2, 3, 4)).save(
        cmyk, format="JPEG"
    )
    with pytest.raises(ReferenceMatchContractError, match="one RGB"):
        decode._decode_jpeg(cmyk.getvalue(), 8)


def test_tiff_rejects_multiple_pages_and_trailing_bytes() -> None:
    buffer = BytesIO()
    with tifffile.TiffWriter(buffer) as writer:
        writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
        writer.write(np.ones((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ReferenceMatchContractError, match="one page"):
        decode._decode_tiff(buffer.getvalue(), 8)
    single = BytesIO()
    tifffile.imwrite(
        single,
        np.zeros((2, 2, 3), dtype=np.uint8),
        photometric="rgb",
        metadata=None,
    )
    with pytest.raises(ReferenceMatchContractError, match="trailing"):
        decode._decode_tiff(single.getvalue() + b"x", 8)


def test_geometry_budget_and_readonly_pixel_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _real_snapshot(tmp_path, suffix=".png", depth=16)
    monkeypatch.setattr(decode, "MAX_DECODED_OUTPUT_PIXELS", 10)
    with pytest.raises(
        ReferenceMatchContractError,
        match="pixel budget|geometry budget",
    ):
        decode.decode_runtime_qualified_shared_staging_bytes_v1(snapshot)
    monkeypatch.setattr(
        decode, "MAX_DECODED_OUTPUT_PIXELS", 128 * 1024 * 1024
    )
    result = decode.decode_runtime_qualified_shared_staging_bytes_v1(
        snapshot
    )
    tampered = result.pixels[0].copy()
    tampered[0, 0, 0] ^= 1
    tampered.flags.writeable = False
    with pytest.raises(
        ReferenceMatchContractError,
        match="pixels do not match",
    ):
        decode.validate_runtime_qualified_shared_staging_decoded_batch_v1(
            replace(result, pixels=(tampered, *result.pixels[1:]))
        )


def test_persisted_decode_record_cannot_escalate_authority(
    tmp_path: Path,
) -> None:
    record = decode.decode_runtime_qualified_shared_staging_bytes_v1(
        _real_snapshot(tmp_path, suffix=".png", depth=8)
    ).record
    for mutation in (
        {"path_consumption_authorized": True},
        {"persistent_pixels_authorized": True},
        {"delivery_authorized": True},
        {"claim_ceiling": "delivered"},
        {"source_count": 3},
    ):
        with pytest.raises(ReferenceMatchContractError):
            decode.validate_runtime_qualified_shared_staging_decode_record_v1(
                replace(record, **mutation)
            )
    payload = record.to_dict()
    payload["delivery_authorized"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(
            json.loads(SCHEMA.read_text(encoding="utf-8"))
        ).validate(payload)
