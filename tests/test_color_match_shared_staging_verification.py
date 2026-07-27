from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    EXTERNAL_SHARED_STAGING_VERIFICATION_CLAIM_CEILING,
    ReferenceMatchContractError,
    commit_external_shared_staging_v1,
    external_shared_staging_verification_from_json,
    external_shared_staging_verification_to_json,
    validate_external_shared_staging_verification_v1,
    verify_external_shared_staging_v1,
)
from src.inference import sha256_file
from tests.test_color_match_shared_staging_transaction import (
    _destinations,
    _pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_shared_staging_verification_v1.schema.json"
)


def _committed(tmp_path: Path):
    applies, batch, numeric, authorization = _pipeline()
    outputs, report = _destinations(tmp_path)
    committed = commit_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        applies=applies,
        output_paths=outputs,
        report_path=report,
    )
    return committed, outputs, report


def _verify(committed, report: Path):
    run = committed.run
    return verify_external_shared_staging_v1(
        report_path=report,
        expected_report_sha256=committed.report_file_sha256,
        expected_run_id=run.run_id,
        expected_authorization_id=run.authorization_id,
        expected_numeric_guard_batch_id=run.numeric_guard_batch_id,
        expected_operator_id=run.operator_id,
    )


def test_restart_verification_rebinds_report_chain_and_all_outputs(
    tmp_path: Path,
) -> None:
    committed, outputs, report = _committed(tmp_path)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (*outputs, report)
    }
    first = _verify(committed, report)
    second = _verify(committed, report)
    assert first == second
    assert first.state == "verified-shared-staging"
    assert (
        first.claim_ceiling
        == EXTERNAL_SHARED_STAGING_VERIFICATION_CLAIM_CEILING
    )
    assert first.authorization_id == committed.run.authorization_id
    assert (
        first.numeric_guard_batch_id
        == committed.run.numeric_guard_batch_id
    )
    assert first.operator_id == committed.run.operator_id
    assert first.report_file_sha256 == sha256_file(report)
    for path, row, staged in zip(
        outputs, first.outputs, committed.run.outputs, strict=True
    ):
        assert row.output_path == str(path.resolve())
        assert row.output_file_sha256 == sha256_file(path)
        assert row.apply_receipt_id == staged.apply_receipt_id
        assert (
            row.producer_apply_result_id
            == staged.producer_apply_result_id
        )
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (*outputs, report)
    }
    assert after == before
    encoded = external_shared_staging_verification_to_json(first)
    assert (
        external_shared_staging_verification_from_json(encoded)
        == first
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(first.to_dict())


def test_report_tamper_and_relocation_fail_closed(tmp_path: Path) -> None:
    committed, _outputs, report = _committed(tmp_path)
    original = report.read_bytes()
    report.write_bytes(original + b" ")
    with pytest.raises(
        ReferenceMatchContractError,
        match="report hash mismatch",
    ):
        _verify(committed, report)
    report.write_bytes(original)
    moved = tmp_path / "moved-report.json"
    moved.write_bytes(original)
    with pytest.raises(
        ReferenceMatchContractError,
        match="report path mismatch",
    ):
        verify_external_shared_staging_v1(
            report_path=moved,
            expected_report_sha256=sha256_file(moved),
            expected_run_id=committed.run.run_id,
            expected_authorization_id=committed.run.authorization_id,
            expected_numeric_guard_batch_id=(
                committed.run.numeric_guard_batch_id
            ),
            expected_operator_id=committed.run.operator_id,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("run", "run identity mismatch"),
        ("authorization", "authorization identity mismatch"),
        ("guard", "numeric guard identity mismatch"),
        ("operator", "operator identity mismatch"),
    ],
)
def test_expected_chain_identities_are_mandatory_and_exact(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    committed, _outputs, report = _committed(tmp_path)
    run = committed.run
    kwargs = {
        "report_path": report,
        "expected_report_sha256": committed.report_file_sha256,
        "expected_run_id": run.run_id,
        "expected_authorization_id": run.authorization_id,
        "expected_numeric_guard_batch_id": run.numeric_guard_batch_id,
        "expected_operator_id": run.operator_id,
    }
    key = {
        "run": "expected_run_id",
        "authorization": "expected_authorization_id",
        "guard": "expected_numeric_guard_batch_id",
        "operator": "expected_operator_id",
    }[field]
    kwargs[key] = "0" * 64
    with pytest.raises(ReferenceMatchContractError, match=message):
        verify_external_shared_staging_v1(**kwargs)
    kwargs[key] = "not-a-hash"
    with pytest.raises(
        ReferenceMatchContractError,
        match="lowercase SHA-256",
    ):
        verify_external_shared_staging_v1(**kwargs)


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, state="delivered"),
            "state is invalid",
        ),
        (
            lambda value: replace(
                value,
                outputs=(
                    replace(value.outputs[0], source_index=1),
                    value.outputs[1],
                ),
            ),
            "order is invalid",
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
            "verification identity mismatch",
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
        validate_external_shared_staging_verification_v1(
            mutation(verification)
        )


def test_unknown_json_field_fails_closed(tmp_path: Path) -> None:
    committed, _outputs, report = _committed(tmp_path)
    verification = _verify(committed, report)
    payload = verification.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ReferenceMatchContractError):
        external_shared_staging_verification_from_json(
            json.dumps(payload)
        )
