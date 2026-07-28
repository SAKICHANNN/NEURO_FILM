from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from typing import Iterator

from jsonschema import Draft202012Validator, ValidationError
import pytest

from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.handle_verification_io import StableFileHandleLease
import src.color_match.shared_runtime_staging_consumption as consumption
import src.color_match.shared_runtime_staging_verification as verification
from tests.test_color_match_shared_runtime_staging_verification import (
    _committed_p62,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_runtime_qualified_shared_staging_consumption_v1.schema.json"
)


def _capture(tmp_path: Path) -> (
    consumption.RuntimeQualifiedSharedStagingByteSnapshotV1
):
    run, report_sha256, report_path = _committed_p62(tmp_path)
    return consumption.capture_runtime_qualified_shared_staging_bytes_v1(
        report_path=report_path,
        expected_report_sha256=report_sha256,
        expected_run_id=run.run_id,
        expected_runtime_qualification_id=run.runtime_qualification_id,
    )


def test_capture_returns_only_immutable_process_local_bytes_and_strict_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = verification.open_stable_file_handle

    @contextmanager
    def tracking_open(
        path: Path | str,
        *,
        label: str,
    ) -> Iterator[StableFileHandleLease]:
        calls.append(str(path))
        with original(path, label=label) as lease:
            yield lease

    monkeypatch.setattr(
        verification, "open_stable_file_handle", tracking_open
    )
    snapshot = _capture(tmp_path)

    assert calls == [
        str((tmp_path / "p62-report.json").resolve()),
        str((tmp_path / "output-0.png").resolve()),
        str((tmp_path / "output-1.png").resolve()),
    ]
    assert snapshot.output_bytes == (b"output-0", b"output-1")
    assert all(isinstance(value, bytes) for value in snapshot.output_bytes)
    assert snapshot.record.path_consumption_authorized is False
    assert snapshot.record.persistent_snapshot_authorized is False
    assert snapshot.record.delivery_authorized is False
    assert "report_path" not in snapshot.record.to_dict()
    assert all(
        "output_path" not in output
        for output in snapshot.record.to_dict()["outputs"]
    )
    consumption.validate_runtime_qualified_shared_staging_byte_snapshot_v1(
        snapshot
    )

    encoded = (
        consumption
        .runtime_qualified_shared_staging_consumption_record_to_json(
            snapshot.record
        )
    )
    assert (
        consumption
        .runtime_qualified_shared_staging_consumption_record_from_json(
            encoded
        )
        == snapshot.record
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(snapshot.record.to_dict())


def test_snapshot_remains_byte_stable_but_never_authorizes_reopened_path(
    tmp_path: Path,
) -> None:
    snapshot = _capture(tmp_path)
    output = tmp_path / "output-0.png"
    output.write_bytes(b"later-untrusted-path-content")

    assert snapshot.output_bytes[0] == b"output-0"
    consumption.validate_runtime_qualified_shared_staging_byte_snapshot_v1(
        snapshot
    )
    assert snapshot.record.path_consumption_authorized is False


def test_raw_bytes_and_record_authority_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    snapshot = _capture(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="bytes do not match verification",
    ):
        consumption.validate_runtime_qualified_shared_staging_byte_snapshot_v1(
            replace(
                snapshot,
                output_bytes=(b"tampered", *snapshot.output_bytes[1:]),
            )
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="authority is invalid",
    ):
        consumption.validate_runtime_qualified_shared_staging_consumption_record_v1(
            replace(snapshot.record, delivery_authorized=True)
        )
    payload = snapshot.record.to_dict()
    payload["path_consumption_authorized"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(
            json.loads(SCHEMA.read_text(encoding="utf-8"))
        ).validate(payload)


def test_capture_budget_is_enforced_before_any_output_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    output_verify_calls = 0
    original_verify = StableFileHandleLease.verify

    def tracking_verify(
        self: StableFileHandleLease,
        **kwargs: object,
    ) -> object:
        nonlocal output_verify_calls
        if "staged output" in self.label:
            output_verify_calls += 1
        return original_verify(self, **kwargs)

    monkeypatch.setattr(StableFileHandleLease, "verify", tracking_verify)
    monkeypatch.setattr(consumption, "MAX_CAPTURED_OUTPUT_BYTES", 4)
    with pytest.raises(
        ReferenceMatchContractError,
        match="in-memory capture budget",
    ):
        consumption.capture_runtime_qualified_shared_staging_bytes_v1(
            report_path=report_path,
            expected_report_sha256=report_sha256,
            expected_run_id=run.run_id,
            expected_runtime_qualification_id=(
                run.runtime_qualification_id
            ),
        )
    assert output_verify_calls == 0


def test_final_same_handle_failure_returns_no_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    original = StableFileHandleLease.final_check
    calls = 0

    def fail_on_second(self: StableFileHandleLease) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ReferenceMatchContractError("injected final rehash failure")
        original(self)

    monkeypatch.setattr(StableFileHandleLease, "final_check", fail_on_second)
    with pytest.raises(
        ReferenceMatchContractError,
        match="injected final rehash failure",
    ):
        consumption.capture_runtime_qualified_shared_staging_bytes_v1(
            report_path=report_path,
            expected_report_sha256=report_sha256,
            expected_run_id=run.run_id,
            expected_runtime_qualification_id=(
                run.runtime_qualification_id
            ),
        )
    assert calls == 2


@pytest.mark.parametrize(
    "mutation",
    [
        {"persistent_snapshot_authorized": True},
        {"snapshot_scope": "persistent-path-authority"},
        {"claim_ceiling": "delivered"},
        {"source_count": 3},
    ],
)
def test_persisted_record_cannot_escalate_authority(
    tmp_path: Path,
    mutation: dict[str, object],
) -> None:
    record = _capture(tmp_path).record
    with pytest.raises(ReferenceMatchContractError):
        consumption.validate_runtime_qualified_shared_staging_consumption_record_v1(
            replace(record, **mutation)
        )
