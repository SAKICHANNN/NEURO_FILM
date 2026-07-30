from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    EXTERNAL_CORE_STAGING_VERIFICATION_CLAIM_CEILING,
    ReferenceMatchContractError,
    commit_external_core_staging_v1,
    external_core_staging_verification_from_json,
    external_core_staging_verification_to_json,
    validate_external_core_staging_verification_v1,
    verify_external_core_staging_v1,
)
from src.inference import sha256_file
from tests.test_color_match_core_staging_transaction import (
    _destinations,
    _pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_core_staging_verification_v1.schema.json"
)


def _committed(tmp_path: Path):
    items, batch, authorization = _pipeline()
    outputs, report = _destinations(tmp_path)
    committed = commit_external_core_staging_v1(
        batch=batch,
        authorization=authorization,
        candidates=tuple(item[1] for item in items),
        output_paths=outputs,
        report_path=report,
    )
    return committed, outputs, report


def _verify(committed, report: Path):
    return verify_external_core_staging_v1(
        report_path=report,
        expected_report_sha256=committed.report_file_sha256,
        expected_run_id=committed.run.run_id,
    )


def test_restart_verification_rebinds_report_and_every_output(
    tmp_path: Path,
) -> None:
    committed, outputs, report = _committed(tmp_path)
    verification = _verify(committed, report)
    assert verification.state == "verified-staging"
    assert (
        verification.claim_ceiling
        == EXTERNAL_CORE_STAGING_VERIFICATION_CLAIM_CEILING
    )
    assert verification.run_id == committed.run.run_id
    assert verification.report_file_sha256 == sha256_file(report)
    assert verification.report_path == str(report.resolve())
    for path, row in zip(outputs, verification.outputs, strict=True):
        assert row.output_path == str(path.resolve())
        assert row.output_file_sha256 == sha256_file(path)
        assert (
            row.apply_receipt_id
            == committed.run.outputs[row.source_index].apply_receipt_id
        )
    encoded = external_core_staging_verification_to_json(verification)
    assert (
        external_core_staging_verification_from_json(encoded)
        == verification
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(verification.to_dict())


def test_report_byte_tamper_fails_before_output_verification(
    tmp_path: Path,
) -> None:
    committed, _outputs, report = _committed(tmp_path)
    report.write_bytes(report.read_bytes() + b" ")
    with pytest.raises(
        ReferenceMatchContractError,
        match="report hash mismatch",
    ):
        _verify(committed, report)


def test_expected_run_id_is_mandatory_and_exact(tmp_path: Path) -> None:
    committed, _outputs, report = _committed(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="run identity mismatch",
    ):
        verify_external_core_staging_v1(
            report_path=report,
            expected_report_sha256=committed.report_file_sha256,
            expected_run_id="0" * 64,
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="lowercase SHA-256",
    ):
        verify_external_core_staging_v1(
            report_path=report,
            expected_report_sha256=committed.report_file_sha256,
            expected_run_id="not-a-hash",
        )


def test_output_tamper_or_missing_file_fails_closed(
    tmp_path: Path,
) -> None:
    committed, outputs, report = _committed(tmp_path)
    original = outputs[0].read_bytes()
    outputs[0].write_bytes(b"tampered")
    with pytest.raises(
        ReferenceMatchContractError,
        match="output 0 hash mismatch",
    ):
        _verify(committed, report)
    outputs[0].write_bytes(original)
    outputs[1].unlink()
    with pytest.raises(
        ReferenceMatchContractError,
        match="output 1 is missing",
    ):
        _verify(committed, report)


def test_report_path_relocation_fails_even_with_new_expected_hash(
    tmp_path: Path,
) -> None:
    committed, _outputs, report = _committed(tmp_path)
    moved = tmp_path / "moved-report.json"
    moved.write_bytes(report.read_bytes())
    with pytest.raises(
        ReferenceMatchContractError,
        match="report path mismatch",
    ):
        verify_external_core_staging_v1(
            report_path=moved,
            expected_report_sha256=sha256_file(moved),
            expected_run_id=committed.run.run_id,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, state="delivered"),
            "state is unsupported",
        ),
        (
            lambda value: replace(
                value,
                outputs=(
                    replace(value.outputs[0], source_index=1),
                    value.outputs[1],
                ),
            ),
            "indices must be contiguous",
        ),
        (
            lambda value: replace(
                value,
                claim_ceiling="applied",
            ),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(value, verification_id="0" * 64),
            "verification_id mismatch",
        ),
    ],
)
def test_verification_mutations_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    committed, _outputs, report = _committed(tmp_path)
    verification = _verify(committed, report)
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_external_core_staging_verification_v1(
            mutation(verification)
        )
