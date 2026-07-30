from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    EXTERNAL_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING,
    ReferenceMatchContractError,
    external_local_delivery_verification_from_json,
    external_local_delivery_verification_to_json,
    validate_external_local_delivery_verification_v1,
    verify_external_local_delivery_v1,
)
from src.inference import sha256_file
from tests.test_color_match_external_delivery_authorization import (
    _authorize,
    _chain,
)
from tests.test_color_match_external_local_delivery import (
    _commit,
    _destinations,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_local_delivery_verification_v1.schema.json"
)


def _delivered(tmp_path: Path):
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    outputs, report = _destinations(tmp_path / "delivery")
    committed = _commit(chain, authorization, outputs, report)
    return chain, committed, outputs, report


def _verify(committed, report: Path):
    return verify_external_local_delivery_v1(
        report_path=report,
        expected_report_sha256=committed.report_file_sha256,
        expected_delivery_id=committed.delivery.delivery_id,
    )


def test_restart_verification_rebinds_sources_and_deliveries(
    tmp_path: Path,
) -> None:
    _chain_value, committed, outputs, report = _delivered(tmp_path)
    verification = _verify(committed, report)
    assert verification.state == "verified-local-delivery"
    assert (
        verification.claim_ceiling
        == EXTERNAL_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING
    )
    assert verification.delivery_id == committed.delivery.delivery_id
    assert verification.report_file_sha256 == sha256_file(report)
    for delivered, source_row, row in zip(
        outputs,
        committed.delivery.outputs,
        verification.outputs,
        strict=True,
    ):
        assert row.staging_output_file_sha256 == sha256_file(
            Path(source_row.staging_output_path)
        )
        assert row.delivered_path == str(delivered.resolve())
        assert row.delivered_file_sha256 == sha256_file(delivered)
    encoded = external_local_delivery_verification_to_json(verification)
    assert (
        external_local_delivery_verification_from_json(encoded)
        == verification
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(verification.to_dict())


def test_report_byte_tamper_fails_before_file_verification(
    tmp_path: Path,
) -> None:
    _chain_value, committed, _outputs, report = _delivered(tmp_path)
    report.write_bytes(report.read_bytes() + b" ")
    with pytest.raises(
        ReferenceMatchContractError,
        match="report hash mismatch",
    ):
        _verify(committed, report)


def test_expected_delivery_id_is_mandatory_and_exact(
    tmp_path: Path,
) -> None:
    _chain_value, committed, _outputs, report = _delivered(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="delivery identity mismatch",
    ):
        verify_external_local_delivery_v1(
            report_path=report,
            expected_report_sha256=committed.report_file_sha256,
            expected_delivery_id="0" * 64,
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="lowercase SHA-256",
    ):
        verify_external_local_delivery_v1(
            report_path=report,
            expected_report_sha256=committed.report_file_sha256,
            expected_delivery_id="not-a-hash",
        )


def test_staging_tamper_or_missing_file_fails_closed(
    tmp_path: Path,
) -> None:
    _chain_value, committed, _outputs, report = _delivered(tmp_path)
    first = Path(committed.delivery.outputs[0].staging_output_path)
    second = Path(committed.delivery.outputs[1].staging_output_path)
    original = first.read_bytes()
    first.write_bytes(b"tampered")
    with pytest.raises(
        ReferenceMatchContractError,
        match="staging source 0 hash mismatch",
    ):
        _verify(committed, report)
    first.write_bytes(original)
    second.unlink()
    with pytest.raises(
        ReferenceMatchContractError,
        match="staging source 1 is missing",
    ):
        _verify(committed, report)


def test_delivered_tamper_or_missing_file_fails_closed(
    tmp_path: Path,
) -> None:
    _chain_value, committed, outputs, report = _delivered(tmp_path)
    original = outputs[0].read_bytes()
    outputs[0].write_bytes(b"tampered")
    with pytest.raises(
        ReferenceMatchContractError,
        match="delivered output 0 hash mismatch",
    ):
        _verify(committed, report)
    outputs[0].write_bytes(original)
    outputs[1].unlink()
    with pytest.raises(
        ReferenceMatchContractError,
        match="delivered output 1 is missing",
    ):
        _verify(committed, report)


def test_report_relocation_fails_even_with_new_hash(tmp_path: Path) -> None:
    _chain_value, committed, _outputs, report = _delivered(tmp_path)
    moved = tmp_path / "moved-report.json"
    moved.write_bytes(report.read_bytes())
    with pytest.raises(
        ReferenceMatchContractError,
        match="report path mismatch",
    ):
        verify_external_local_delivery_v1(
            report_path=moved,
            expected_report_sha256=sha256_file(moved),
            expected_delivery_id=committed.delivery.delivery_id,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, state="applied"),
            "state is unsupported",
        ),
        (
            lambda value: replace(value, claim_ceiling="stock"),
            "claim ceiling mismatch",
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
                    replace(
                        value.outputs[0],
                        delivered_file_sha256="0" * 64,
                    ),
                    value.outputs[1],
                ),
            ),
            "byte identity mismatch",
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
    _chain_value, committed, _outputs, report = _delivered(tmp_path)
    verification = _verify(committed, report)
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_external_local_delivery_verification_v1(
            mutation(verification)
        )
