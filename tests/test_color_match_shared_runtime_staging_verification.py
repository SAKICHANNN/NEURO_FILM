from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

from jsonschema import Draft202012Validator, ValidationError
import pytest

from src.color_match.canonical import canonical_sha256
from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.handle_verification_io import StableFileHandleLease
import src.color_match.shared_runtime_staging_verification as verification
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
    _destinations,
    _pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_runtime_qualified_shared_staging_verification_v1.schema.json"
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _producer_hash(label: str) -> str:
    return "sha256:" + _hash(label)


def _committed_p62(
    tmp_path: Path,
    *,
    count: int = 2,
    same_bytes: bool = False,
) -> tuple[RuntimeQualifiedExternalSharedStagingRunV1, str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    report_path = (tmp_path / "p62-report.json").resolve()
    outputs: list[ExternalSharedStagedOutputV1] = []
    for index in range(count):
        output_path = (tmp_path / f"output-{index}.png").resolve()
        payload = b"same" if same_bytes else f"output-{index}".encode()
        output_path.write_bytes(payload)
        outputs.append(
            ExternalSharedStagedOutputV1(
                source_index=index,
                source_view_id=_hash(f"source-{index}"),
                apply_receipt_id=_hash(f"receipt-{index}"),
                producer_apply_result_id=_producer_hash(
                    f"producer-result-{index}"
                ),
                diagnostics_id=_producer_hash(f"diagnostics-{index}"),
                output_view_id=_hash(f"output-view-{index}"),
                output_path=str(output_path),
                output_file_sha256=hashlib.sha256(payload).hexdigest(),
                output_format="PNG",
                output_bit_depth=16,
                encode_clipped_fraction=0.0,
            )
        )
    provisional = RuntimeQualifiedExternalSharedStagingRunV1(
        schema_id=RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID,
        run_id="0" * 64,
        runtime_qualification_id=_hash("qualification"),
        runtime_evidence_id=_hash("evidence"),
        declaration_id=_hash("declaration"),
        authorization_id=_hash("authorization"),
        upstream_batch_id=_hash("batch"),
        numeric_guard_batch_id=_hash("numeric"),
        operator_id=_hash("operator"),
        reference_view_id=_hash("reference"),
        source_count=count,
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
    ).encode("utf-8")
    report_path.write_bytes(encoded)
    return run, hashlib.sha256(encoded).hexdigest(), report_path


def _verify(
    run: RuntimeQualifiedExternalSharedStagingRunV1,
    report_sha256: str,
    report_path: Path,
) -> verification.RuntimeQualifiedExternalSharedStagingVerificationV1:
    return verification.verify_runtime_qualified_external_shared_staging_v1(
        report_path=report_path,
        expected_report_sha256=report_sha256,
        expected_run_id=run.run_id,
        expected_runtime_qualification_id=run.runtime_qualification_id,
    )


def test_clean_verification_is_deterministic_strict_and_schema_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
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
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    first = _verify(run, report_sha256, report_path)
    second = _verify(run, report_sha256, report_path)

    assert first == second
    assert first.path_consumption_authorized is False
    assert first.handle_observation_scope == (
        verification.WINDOWS_HANDLE_OBSERVATION_SCOPE
        if os.name == "nt"
        else verification.POSIX_HANDLE_OBSERVATION_SCOPE
    )
    assert calls == [
        str(report_path),
        *(output.output_path for output in run.outputs),
    ] * 2
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.iterdir()
        if path.is_file()
    } == before
    encoded = (
        verification
        .runtime_qualified_external_shared_staging_verification_to_json(
            first
        )
    )
    assert (
        verification
        .runtime_qualified_external_shared_staging_verification_from_json(
            encoded
        )
        == first
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(first.to_dict())
    assert "(?i)" not in schema["properties"]["report_path"]["pattern"]
    impossible_format = first.to_dict()
    impossible_format["outputs"][0]["output_format"] = "TIFF"
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(impossible_format)
    mismatched_scope = first.to_dict()
    mismatched_scope["handle_observation_scope"] = (
        verification.POSIX_HANDLE_OBSERVATION_SCOPE
        if os.name == "nt"
        else verification.WINDOWS_HANDLE_OBSERVATION_SCOPE
    )
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(mismatched_scope)


def test_real_p62_transaction_is_accepted_by_p63(tmp_path: Path) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    committed = commit_runtime_qualified_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        runtime_qualification=qualification,
        expected_runtime_qualification_id=qualification.qualification_id,
        applies=applies,
        output_paths=outputs,
        report_path=report,
    )

    result = _verify(
        committed.run,
        committed.report_file_sha256,
        report,
    )

    assert result.run_id == committed.run.run_id
    assert result.runtime_qualification_id == (
        qualification.qualification_id
    )
    assert tuple(output.output_path for output in result.outputs) == tuple(
        str(path.resolve()) for path in outputs
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("expected_report_sha256", "hash mismatch"),
        ("expected_run_id", "run identity mismatch"),
        (
            "expected_runtime_qualification_id",
            "qualification identity mismatch",
        ),
    ],
)
def test_all_caller_pins_fail_closed(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    arguments = {
        "report_path": report_path,
        "expected_report_sha256": report_sha256,
        "expected_run_id": run.run_id,
        "expected_runtime_qualification_id": run.runtime_qualification_id,
    }
    arguments[field] = _hash(f"wrong-{field}")
    with pytest.raises(ReferenceMatchContractError, match=message):
        verification.verify_runtime_qualified_external_shared_staging_v1(
            **arguments
        )


def test_report_and_output_tamper_missing_and_relocation_fail_closed(
    tmp_path: Path,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    report_original = report_path.read_bytes()
    report_path.write_bytes(report_original + b"x")
    with pytest.raises(ReferenceMatchContractError, match="hash mismatch"):
        _verify(run, report_sha256, report_path)
    report_path.write_bytes(report_original)

    output = Path(run.outputs[0].output_path)
    original = output.read_bytes()
    output.write_bytes(original + b"x")
    with pytest.raises(ReferenceMatchContractError, match="hash mismatch"):
        _verify(run, report_sha256, report_path)
    output.write_bytes(original)

    missing = output.with_suffix(".missing")
    output.rename(missing)
    with pytest.raises(ReferenceMatchContractError):
        _verify(run, report_sha256, report_path)
    missing.rename(output)

    relocated = (tmp_path / "relocated.json").resolve()
    relocated.write_bytes(report_original)
    with pytest.raises(
        ReferenceMatchContractError, match="report path mismatch"
    ):
        _verify(run, report_sha256, relocated)


def test_output_output_hard_link_aliases_fail_closed(
    tmp_path: Path,
) -> None:
    run, report_sha256, report_path = _committed_p62(
        tmp_path, same_bytes=True
    )
    output0 = Path(run.outputs[0].output_path)
    output1 = Path(run.outputs[1].output_path)
    output1.unlink()
    os.link(output0, output1)
    with pytest.raises(
        ReferenceMatchContractError, match="handle identities must be distinct"
    ):
        _verify(run, report_sha256, report_path)


def test_self_consistent_p62_impossible_format_metadata_fails_closed(
    tmp_path: Path,
) -> None:
    run, _report_sha256, report_path = _committed_p62(tmp_path)
    provisional = replace(
        run,
        outputs=(
            replace(run.outputs[0], output_format="TIFF"),
            run.outputs[1],
        ),
    )
    identity = provisional.to_dict()
    identity.pop("run_id")
    impossible = replace(
        provisional,
        run_id=canonical_sha256(identity),
    )
    encoded = runtime_qualified_external_shared_staging_run_to_json(
        impossible
    ).encode("utf-8")
    report_path.write_bytes(encoded)

    with pytest.raises(
        ReferenceMatchContractError,
        match="extension/depth mismatch",
    ):
        _verify(
            impossible,
            hashlib.sha256(encoded).hexdigest(),
            report_path,
        )


def test_same_bytes_new_object_is_content_equivalent_not_inode_continuity(
    tmp_path: Path,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    first = _verify(run, report_sha256, report_path)
    output = Path(run.outputs[0].output_path)
    backup = output.with_suffix(".original")
    output.rename(backup)
    output.write_bytes(backup.read_bytes())

    second = _verify(run, report_sha256, report_path)

    assert first.outputs[0].output_file_sha256 == (
        second.outputs[0].output_file_sha256
    )
    assert first.outputs[0].handle_identity_id != (
        second.outputs[0].handle_identity_id
    )
    assert first.verification_id != second.verification_id


def test_pure_validator_never_resolves_or_stats_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    result = _verify(run, report_sha256, report_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("filesystem access is forbidden")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(os, "stat", forbidden)
    verification.validate_runtime_qualified_external_shared_staging_verification_v1(
        result
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, state="applied"),
            "state is invalid",
        ),
        (
            lambda value: replace(value, claim_ceiling="delivered"),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(
                value,
                outputs=(
                    value.outputs[0],
                    replace(
                        value.outputs[1],
                        handle_identity_id=(
                            value.outputs[0].handle_identity_id
                        ),
                    ),
                ),
            ),
            "identities must be distinct",
        ),
        (
            lambda value: replace(value, verification_id="0" * 64),
            "identity mismatch",
        ),
        (
            lambda value: replace(
                value, handle_identity_scheme="invalid-scheme"
            ),
            "identity scheme is invalid",
        ),
        (
            lambda value: replace(
                value,
                handle_observation_scope="persistent-filesystem-proof",
            ),
            "observation scope is invalid",
        ),
        (
            lambda value: replace(
                value,
                path_consumption_authorized=True,
            ),
            "cannot authorize path consumption",
        ),
        (
            lambda value: replace(
                value,
                report_handle_identity_id=(
                    value.outputs[0].handle_identity_id
                ),
            ),
            "identities must be distinct",
        ),
        (
            lambda value: replace(
                value,
                outputs=(
                    replace(value.outputs[0], source_index=False),
                    value.outputs[1],
                ),
            ),
            "output order is invalid",
        ),
        (
            lambda value: replace(
                value,
                outputs=(
                    replace(value.outputs[0], output_format="TIFF"),
                    value.outputs[1],
                ),
            ),
            "extension/depth mismatch",
        ),
    ],
)
def test_mutated_records_fail_closed(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    result = _verify(run, report_sha256, report_path)
    mutated = mutation(result)  # type: ignore[operator]
    with pytest.raises(ReferenceMatchContractError, match=message):
        verification.validate_runtime_qualified_external_shared_staging_verification_v1(
            mutated
        )


def test_unknown_fields_and_more_than_sixty_four_outputs_fail_closed(
    tmp_path: Path,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    result = _verify(run, report_sha256, report_path)
    payload = result.to_dict()
    payload["unknown"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        verification.runtime_qualified_external_shared_staging_verification_from_json(
            json.dumps(payload)
        )

    # P118 moves the shared 64-source ceiling to the persisted P50/P62
    # validator, so an impossible 65-output run can no longer be serialized
    # merely to exercise the later handle verifier.
    with pytest.raises(
        ReferenceMatchContractError,
        match="source_count is invalid",
    ):
        _committed_p62(tmp_path / "large", count=65)


@pytest.mark.parametrize(
    ("constant", "bound", "message"),
    [
        ("MAX_STAGED_OUTPUT_BYTES", 3, "bounded size contract"),
        (
            "MAX_STAGED_OUTPUT_AGGREGATE_BYTES",
            10,
            "aggregate byte budget",
        ),
    ],
)
def test_output_byte_budgets_reject_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    bound: int,
    message: str,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    monkeypatch.setattr(verification, constant, bound)

    with pytest.raises(ReferenceMatchContractError, match=message):
        _verify(run, report_sha256, report_path)


def test_serialized_record_cannot_exceed_aggregate_byte_budget(
    tmp_path: Path,
) -> None:
    run, report_sha256, report_path = _committed_p62(
        tmp_path,
        count=9,
    )
    result = _verify(run, report_sha256, report_path)
    oversized = replace(
        result,
        outputs=tuple(
            replace(
                output,
                output_file_size_bytes=verification.MAX_STAGED_OUTPUT_BYTES,
            )
            for output in result.outputs
        ),
    )
    identity = oversized.to_dict()
    identity.pop("verification_id")
    oversized = replace(
        oversized,
        verification_id=canonical_sha256(identity),
    )

    with pytest.raises(
        ReferenceMatchContractError, match="aggregate byte budget"
    ):
        verification.runtime_qualified_external_shared_staging_verification_from_json(
            json.dumps(oversized.to_dict())
        )


def test_namespace_race_uses_handle_helper_and_closes_every_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, report_sha256, report_path = _committed_p62(tmp_path)
    target = Path(run.outputs[0].output_path)
    foreign = (tmp_path / "foreign.png").resolve()
    foreign.write_bytes(b"foreign")
    original = verification.open_stable_file_handle
    attack_replaced: list[bool] = []

    @contextmanager
    def attacking_open(
        path: Path | str,
        *,
        label: str,
    ) -> Iterator[StableFileHandleLease]:
        with original(path, label=label) as lease:
            if str(path) == str(target):
                try:
                    os.replace(foreign, target)
                    attack_replaced.append(True)
                except OSError:
                    attack_replaced.append(False)
            yield lease

    monkeypatch.setattr(
        verification, "open_stable_file_handle", attacking_open
    )
    if os.name == "nt":
        result = _verify(run, report_sha256, report_path)
        assert result.outputs[0].output_path == str(target)
        assert attack_replaced == [False]
    else:
        with pytest.raises(ReferenceMatchContractError):
            _verify(run, report_sha256, report_path)
        assert attack_replaced == [True]

    for path in (
        report_path,
        *map(lambda row: Path(row.output_path), run.outputs),
    ):
        assert path.exists()
        moved = path.with_name(path.name + ".moved")
        path.rename(moved)
        assert moved.exists()
