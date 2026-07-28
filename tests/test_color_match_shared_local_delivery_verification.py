from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    SHARED_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING,
    ReferenceMatchContractError,
    shared_local_delivery_verification_from_json,
    shared_local_delivery_verification_to_json,
    validate_shared_local_delivery_verification_v1,
    verify_shared_local_delivery_v1,
)
from src.inference import sha256_file
from tests.test_color_match_shared_delivery_authorization import (
    _authorize,
    _chain,
)
from tests.test_color_match_shared_local_delivery import (
    _commit,
    _destinations,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_shared_local_delivery_verification_v1.schema.json"
)


def _committed(tmp_path: Path):
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    outputs, report = _destinations(tmp_path / "delivery")
    committed = _commit(chain, authorization, outputs, report)
    return committed, chain, outputs, report


def _verify(committed, report: Path):
    return verify_shared_local_delivery_v1(
        report_path=report,
        expected_report_sha256=committed.report_file_sha256,
        expected_delivery_id=committed.delivery.delivery_id,
    )


def test_restart_verification_rebinds_sources_and_deliveries(
    tmp_path: Path,
) -> None:
    committed, _chain_value, outputs, report = _committed(tmp_path)
    tracked = (
        *(
            Path(row.staging_output_path)
            for row in committed.delivery.outputs
        ),
        *outputs,
        report,
    )
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tracked
    }
    first = _verify(committed, report)
    second = _verify(committed, report)
    assert first == second
    assert first.state == "verified-shared-local-delivery"
    assert (
        first.claim_ceiling
        == SHARED_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING
    )
    assert first.delivery_id == committed.delivery.delivery_id
    assert first.authorization_id == committed.delivery.authorization_id
    assert (
        first.filmfx_verification_id
        == committed.delivery.filmfx_verification_id
    )
    for original, row in zip(
        committed.delivery.outputs, first.outputs, strict=True
    ):
        assert row.apply_receipt_id == original.apply_receipt_id
        assert (
            row.producer_apply_result_id
            == original.producer_apply_result_id
        )
        assert row.staging_output_file_sha256 == sha256_file(
            Path(row.staging_output_path)
        )
        assert row.delivered_file_sha256 == sha256_file(
            Path(row.delivered_path)
        )
        assert (
            row.staging_output_file_sha256
            == row.delivered_file_sha256
        )
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tracked
    }
    assert after == before
    encoded = shared_local_delivery_verification_to_json(first)
    assert shared_local_delivery_verification_from_json(encoded) == first
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(first.to_dict())


def test_report_tamper_relocation_and_foreign_id_fail_closed(
    tmp_path: Path,
) -> None:
    committed, _chain_value, _outputs, report = _committed(tmp_path)
    original = report.read_bytes()
    report.write_bytes(original + b" ")
    with pytest.raises(ReferenceMatchContractError, match="report hash"):
        _verify(committed, report)
    report.write_bytes(original)
    moved = tmp_path / "moved.json"
    moved.write_bytes(original)
    with pytest.raises(ReferenceMatchContractError, match="report path"):
        verify_shared_local_delivery_v1(
            report_path=moved,
            expected_report_sha256=sha256_file(moved),
            expected_delivery_id=committed.delivery.delivery_id,
        )
    with pytest.raises(ReferenceMatchContractError, match="identity mismatch"):
        verify_shared_local_delivery_v1(
            report_path=report,
            expected_report_sha256=committed.report_file_sha256,
            expected_delivery_id="0" * 64,
        )


@pytest.mark.parametrize("kind", ["staging", "delivered"])
def test_file_tamper_or_missing_fails_closed(
    tmp_path: Path,
    kind: str,
) -> None:
    committed, _chain_value, outputs, report = _committed(tmp_path)
    paths = (
        tuple(
            Path(row.staging_output_path)
            for row in committed.delivery.outputs
        )
        if kind == "staging"
        else outputs
    )
    original = paths[0].read_bytes()
    paths[0].write_bytes(b"tampered")
    with pytest.raises(ReferenceMatchContractError, match="hash mismatch"):
        _verify(committed, report)
    paths[0].write_bytes(original)
    paths[1].unlink()
    with pytest.raises(ReferenceMatchContractError, match="is missing"):
        _verify(committed, report)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda v: replace(v, state="applied"), "state is invalid"),
        (
            lambda v: replace(v, claim_ceiling="film-stock"),
            "claim ceiling mismatch",
        ),
        (
            lambda v: replace(
                v,
                outputs=tuple(reversed(v.outputs)),
            ),
            "source order is invalid",
        ),
        (
            lambda v: replace(
                v,
                outputs=(
                    replace(
                        v.outputs[0],
                        delivered_file_sha256="0" * 64,
                    ),
                    v.outputs[1],
                ),
            ),
            "byte identity mismatch",
        ),
        (
            lambda v: replace(v, authorization_id="0" * 64),
            "verification identity mismatch",
        ),
        (
            lambda v: replace(v, verification_id="0" * 64),
            "verification identity mismatch",
        ),
    ],
)
def test_mutations_fail_closed(tmp_path: Path, mutation, message: str) -> None:
    committed, _chain_value, _outputs, report = _committed(tmp_path)
    verification = _verify(committed, report)
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_shared_local_delivery_verification_v1(
            mutation(verification)
        )


def test_unknown_fields_fail_closed(tmp_path: Path) -> None:
    committed, _chain_value, _outputs, report = _committed(tmp_path)
    verification = _verify(committed, report)
    payload = verification.to_dict()
    payload["applied"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        shared_local_delivery_verification_from_json(json.dumps(payload))
    payload = verification.to_dict()
    payload["outputs"][0]["unexpected"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        shared_local_delivery_verification_from_json(json.dumps(payload))
