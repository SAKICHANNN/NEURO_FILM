from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    EXTERNAL_FILMFX_VERIFICATION_CLAIM_CEILING,
    ReferenceMatchContractError,
    commit_external_filmfx_staging_v1,
    external_filmfx_verification_from_json,
    external_filmfx_verification_to_json,
    validate_external_filmfx_staging_verification_v1,
    verify_external_filmfx_staging_v1,
)
from src.inference import sha256_file
from tests.test_color_match_external_filmfx_transaction import (
    _destinations,
    _setup,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_filmfx_staging_verification_v1.schema.json"
)


def _committed(tmp_path: Path):
    _sources, staging_verification, plan = _setup(tmp_path / "source")
    outputs, report = _destinations(tmp_path / "filmfx")
    committed = commit_external_filmfx_staging_v1(
        plan=plan,
        verification=staging_verification,
        output_paths=outputs,
        report_path=report,
        seed=37,
    )
    return committed, outputs, report


def _verify(committed, report: Path):
    return verify_external_filmfx_staging_v1(
        report_path=report,
        expected_report_sha256=committed.report_file_sha256,
        expected_run_id=committed.run.run_id,
    )


def test_restart_verification_rebinds_complete_filmfx_lineage(
    tmp_path: Path,
) -> None:
    committed, outputs, report = _committed(tmp_path)
    verification = _verify(committed, report)
    assert verification.state == "verified-filmfx-staging"
    assert (
        verification.claim_ceiling
        == EXTERNAL_FILMFX_VERIFICATION_CLAIM_CEILING
    )
    assert verification.run_id == committed.run.run_id
    assert verification.report_file_sha256 == sha256_file(report)
    assert verification.report_path == str(report.resolve())
    for output, source_row, row in zip(
        outputs,
        committed.run.outputs,
        verification.outputs,
        strict=True,
    ):
        assert row.input_file_sha256 == sha256_file(
            Path(source_row.input_path)
        )
        assert row.output_path == str(output.resolve())
        assert row.output_file_sha256 == sha256_file(output)
        assert row.grain_seed == source_row.grain_seed
        assert row.dust_seed == source_row.dust_seed
    encoded = external_filmfx_verification_to_json(verification)
    assert (
        external_filmfx_verification_from_json(encoded)
        == verification
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(verification.to_dict())


def test_report_byte_tamper_fails_before_file_verification(
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
        verify_external_filmfx_staging_v1(
            report_path=report,
            expected_report_sha256=committed.report_file_sha256,
            expected_run_id="0" * 64,
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="lowercase SHA-256",
    ):
        verify_external_filmfx_staging_v1(
            report_path=report,
            expected_report_sha256=committed.report_file_sha256,
            expected_run_id="not-a-hash",
        )


def test_input_tamper_or_missing_file_fails_closed(tmp_path: Path) -> None:
    committed, _outputs, report = _committed(tmp_path)
    first = Path(committed.run.outputs[0].input_path)
    second = Path(committed.run.outputs[1].input_path)
    original = first.read_bytes()
    first.write_bytes(b"tampered")
    with pytest.raises(
        ReferenceMatchContractError,
        match="input 0 hash mismatch",
    ):
        _verify(committed, report)
    first.write_bytes(original)
    second.unlink()
    with pytest.raises(
        ReferenceMatchContractError,
        match="input 1 is missing",
    ):
        _verify(committed, report)


def test_output_tamper_or_missing_file_fails_closed(tmp_path: Path) -> None:
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


def test_report_path_relocation_fails_with_new_expected_hash(
    tmp_path: Path,
) -> None:
    committed, _outputs, report = _committed(tmp_path)
    moved = tmp_path / "moved-report.json"
    moved.write_bytes(report.read_bytes())
    with pytest.raises(
        ReferenceMatchContractError,
        match="report path mismatch",
    ):
        verify_external_filmfx_staging_v1(
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
                outputs=(
                    replace(value.outputs[0], dust_seed=999),
                    value.outputs[1],
                ),
            ),
            "dust seed mismatch",
        ),
        (
            lambda value: replace(value, claim_ceiling="delivered"),
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
        validate_external_filmfx_staging_verification_v1(
            mutation(verification)
        )
