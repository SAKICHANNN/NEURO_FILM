from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from jsonschema import Draft202012Validator
import numpy as np
import pytest

import src.color_match.files as file_module
import src.color_match.shared_runtime_staging_transaction as runtime_staging_module
from src.color_match import (
    FROZEN_GATE_POLICY_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING,
    RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID,
    SUCCESSOR_DECLARATION_SCHEMA,
    SUCCESSOR_POLICY_ID,
    SUPPORTED_PROFILE_ID,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    authorize_shared_product_staging_v1,
    bind_shared_promotion_v1,
    commit_external_shared_staging_v1,
    commit_runtime_qualified_external_shared_staging_v1,
    external_shared_staging_run_from_json,
    guard_shared_numeric_batch_v1,
    make_match_view,
    make_shared_apply_numeric_facts_v1,
    make_shared_reference_operator_v1,
    make_successor_runtime_evidence_v1,
    make_successor_runtime_record_v1,
    prepare_shared_operator_apply_v1,
    qualify_shared_product_authorization_runtime_v1,
    resolve_shared_operator_batch_v1,
    runtime_qualified_external_shared_staging_run_from_json,
    runtime_qualified_external_shared_staging_run_to_json,
    successor_declaration_id_v1,
    validate_runtime_qualified_external_shared_staging_run_v1,
)
from src.inference import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_runtime_qualified_shared_staging_run_v1.schema.json"
)
CAPABILITY = "zhuise.runtime-qualified-staging.shared.v1"
PRODUCER_COMMIT = "1" * 40
MODEL = "d" * 64
OPTIONS = "e" * 64
EVIDENCE = "f" * 64
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
ALL_TARGETS = (
    "windows_x64",
    "macos_arm64",
    "ios_arm64",
    "android_arm64",
)


def _prepared(seed: float) -> PreparedMatchViewV1:
    pixels = np.full((4, 5, 3), seed, dtype=np.float32)
    wire = pixels.astype(">f4", copy=False).tobytes()
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=pixels.shape,
        render_bridge_id="neuro-film.runtime-staging-test.v1",
        provenance_fingerprint=hashlib.sha256(
            str(seed).encode("ascii")
        ).hexdigest(),
    )
    pixels.flags.writeable = False
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _declaration() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": SUCCESSOR_DECLARATION_SCHEMA,
        "policy_id": SUCCESSOR_POLICY_ID,
        "candidate_id": "zhuise.runtime-qualified-staging.shared.v1",
        "producer": {
            "stable_commit": PRODUCER_COMMIT,
            "package_source_commit": "2" * 40,
            "fixture_commit": "3" * 40,
            "package_lock_sha256": SHA_A,
            "conformance_fixture_sha256": SHA_B,
        },
        "package": {
            "distribution": "zhuise-research",
            "version": "0.6.0",
            "entrypoint": "zhuise-shared-invoke",
            "wheel_filename": "zhuise_research-0.6.0-py3-none-any.whl",
            "wheel_size_bytes": 130000,
            "wheel_sha256": SHA_C,
        },
        "wire": {
            "capability_id": CAPABILITY,
            "profile_id": SUPPORTED_PROFILE_ID,
            "lower_compatibility_profile_id": (
                "neuro-film.dpct-consumer.v2"
            ),
            "request_schema": "zhuise.shared-request.v3",
            "request_schema_sha256": SHA_A,
            "response_schema": "zhuise.shared-response.v3",
            "response_schema_sha256": SHA_B,
        },
        "semantics": {
            "fit_semantics": "reference-only-shared",
            "batch_transform_policy": "shared-bundle",
            "deterministic": True,
            "hidden_state": "none",
        },
        "rights": {
            "evaluation_allowed": True,
            "commercial_use_allowed": True,
            "redistribution_allowed": True,
            "evidence_id": "rights-review-v3",
        },
        "runtime_evidence": {
            "windows_x64": True,
            "macos_arm64": True,
            "ios_arm64": True,
            "android_arm64": True,
        },
        "product_evidence": {
            "gate_policy_id": FROZEN_GATE_POLICY_ID,
            "stable_evidence_id": EVIDENCE,
            "a1_passed": True,
            "a4_passed": True,
            "a5_passed": True,
            "blind_aesthetic_passed": True,
        },
    }
    value["declaration_id"] = successor_declaration_id_v1(value)
    return value


def _runtime_record(target: str):
    proof_class = {
        "windows_x64": "host-runtime",
        "macos_arm64": "host-runtime",
        "ios_arm64": "device-runtime",
        "android_arm64": "device-runtime",
    }[target]
    os_name, architecture = {
        "windows_x64": ("Windows", "x86_64"),
        "macos_arm64": ("macOS", "arm64"),
        "ios_arm64": ("iOS", "arm64"),
        "android_arm64": ("Android", "arm64-v8a"),
    }[target]
    return make_successor_runtime_record_v1(
        target_runtime=target,
        proof_class=proof_class,
        os_name=os_name,
        os_version="test-os-1",
        architecture=architecture,
        environment_count=2 if target == "windows_x64" else 1,
        environment_matrix_sha256="4" * 64,
        environment_summary=f"factual-{target}-environment",
        backend_id="test-backend",
        backend_version="test-backend-1",
        runner_sha256=SHA_A,
        executable_sha256=SHA_B,
        report_sha256=SHA_C,
        replay_count=2,
        deterministic_replay_passed=True,
        conformance_passed=True,
        failure_injection_passed=True,
    )


def _pipeline(
    *,
    runtime_targets: tuple[str, ...] = ALL_TARGETS,
    promoted: bool = True,
    unsafe_pixel: bool = False,
    model: str = MODEL,
):
    reference = _prepared(0.2)
    sources = (_prepared(0.3), _prepared(0.4))
    declaration = _declaration()
    operator = make_shared_reference_operator_v1(
        reference=reference,
        compatibility_profile_id="neuro-film.dpct-consumer.v2",
        capability_id=CAPABILITY,
        producer_commit=PRODUCER_COMMIT,
        producer_bundle_id="sha256:" + "9" * 64,
        producer_reference_view_id="sha256:" + "8" * 64,
        model_fingerprint=model,
        options_sha256=OPTIONS,
    )
    output_arrays = [
        np.full((4, 5, 3), 0.31, dtype=np.float32),
        np.full((4, 5, 3), 0.41, dtype=np.float32),
    ]
    if unsafe_pixel:
        output_arrays[1][0, 0, 0] = 1.1
    applies = tuple(
        prepare_shared_operator_apply_v1(
            operator=operator,
            source_index=index,
            source=source,
            producer_source_view_id=(
                "sha256:" + str(index + 1) * 64
            ),
            producer_apply_result_id=(
                "sha256:" + str(index + 3) * 64
            ),
            diagnostics_id="sha256:" + str(index + 5) * 64,
            output_pixels=output_arrays[index],
        )
        for index, source in enumerate(sources)
    )
    batch = resolve_shared_operator_batch_v1(
        operator=operator,
        reference=reference,
        sources=sources,
        applies=applies,
    )
    facts = tuple(
        make_shared_apply_numeric_facts_v1(
            prepared=prepared,
            producer_diagnostics_id=prepared.receipt.diagnostics_id,
            all_finite=True,
            output_minimum=float(np.min(prepared.pixels)),
            output_maximum=float(np.max(prepared.pixels)),
            out_of_gamut_fraction=0.01,
            clipping_fraction=0.01,
            projected_fraction=0.0,
        )
        for prepared in applies
    )
    numeric = guard_shared_numeric_batch_v1(
        batch=batch,
        sources=sources,
        applies=applies,
        facts=facts,
    )
    promotion = (
        PromotionDecision("promoted", ())
        if promoted
        else PromotionDecision(
            "eligible-for-visual-review",
            ("visual-review-required",),
        )
    )
    binding = bind_shared_promotion_v1(
        operator=operator,
        declaration=declaration,
        stable_evidence_id=EVIDENCE,
        promotion=promotion,
    )
    authorization = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    runtime_evidence = make_successor_runtime_evidence_v1(
        declaration_id=str(declaration["declaration_id"]),
        producer_commit=PRODUCER_COMMIT,
        capability_id=CAPABILITY,
        profile_id=SUPPORTED_PROFILE_ID,
        records=tuple(
            _runtime_record(target) for target in runtime_targets
        ),
    )
    qualification = qualify_shared_product_authorization_runtime_v1(
        authorization=authorization,
        runtime_evidence=runtime_evidence,
    )
    return applies, batch, numeric, authorization, qualification


def _destinations(tmp_path: Path) -> tuple[tuple[Path, ...], Path]:
    return (
        (tmp_path / "one.png", tmp_path / "two.png"),
        tmp_path / "runtime-qualified-staging-report.json",
    )


def _assert_no_debris(root: Path) -> None:
    assert not list(root.rglob(".*.reference-match-stage*"))
    assert not list(root.rglob(".*.reference-match-backup"))


def test_exact_runtime_qualification_commits_outputs_and_bound_report(
    tmp_path: Path,
) -> None:
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
    run = committed.run
    assert run.state == (
        "committed-to-runtime-qualified-shared-staging"
    )
    assert run.claim_ceiling == (
        RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING
    )
    assert run.runtime_qualification_id == qualification.qualification_id
    assert run.runtime_evidence_id == qualification.runtime_evidence_id
    assert run.declaration_id == qualification.declaration_id
    assert run.authorization_id == authorization.authorization_id
    assert run.upstream_batch_id == batch.batch_id
    assert run.numeric_guard_batch_id == numeric.guard_batch_id
    assert committed.report_file_sha256 == sha256_file(report)
    for output, row, receipt in zip(
        outputs,
        run.outputs,
        batch.sources,
        strict=True,
    ):
        assert row.output_file_sha256 == sha256_file(output)
        assert row.apply_receipt_id == receipt.receipt_id
        assert row.producer_apply_result_id == (
            receipt.producer_apply_result_id
        )
    decoded = runtime_qualified_external_shared_staging_run_from_json(
        report.read_text(encoding="utf-8")
    )
    assert decoded == run
    assert (
        runtime_qualified_external_shared_staging_run_from_json(
            runtime_qualified_external_shared_staging_run_to_json(run)
        )
        == run
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(run.to_dict())
    _assert_no_debris(tmp_path)


def test_missing_runtime_targets_write_nothing_or_create_directories(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline(
        runtime_targets=("windows_x64",),
    )
    assert qualification.state == "identity-fallback"
    root = tmp_path / "not-created"
    outputs = (root / "a" / "one.png", root / "b" / "two.png")
    report = root / "reports" / "run.json"
    with pytest.raises(
        ReferenceMatchContractError,
        match="requires exact runtime qualification",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert not root.exists()


def test_runtime_cannot_override_upstream_authorization_fallback(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline(
        promoted=False,
    )
    assert qualification.runtime_ready
    assert qualification.state == "identity-fallback"
    outputs, report = _destinations(tmp_path)
    with pytest.raises(ReferenceMatchContractError):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)


def test_foreign_or_tampered_qualification_rejects_before_write(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    _other = _pipeline(model="7" * 64)[4]
    outputs, report = _destinations(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind the upstream",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=_other,
            expected_runtime_qualification_id=_other.qualification_id,
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="identity mismatch",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=replace(
                qualification,
                qualification_id="0" * 64,
            ),
            expected_runtime_qualification_id=qualification.qualification_id,
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)


def test_reordered_apply_and_path_collisions_fail_before_write(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind the chain",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=tuple(reversed(applies)),
            output_paths=outputs,
            report_path=report,
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="output paths must be unique",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=(outputs[0], outputs[0]),
            report_path=report,
        )
    with pytest.raises(ReferenceMatchContractError, match="report must use"):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=outputs,
            report_path=outputs[0],
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)


def test_sparse_out_of_bounds_pixel_fails_without_commit(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline(
        unsafe_pixel=True,
    )
    outputs, report = _destinations(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="bounded sRGB encoding tolerance",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)
    _assert_no_debris(tmp_path)


def test_commit_failure_restores_all_previous_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    previous = {
        outputs[0]: b"old-one",
        outputs[1]: b"old-two",
        report: b"old-report",
    }
    for path, payload in previous.items():
        path.write_bytes(payload)
    original_replace = file_module._replace

    def fail_report(source: Path, destination: Path) -> None:
        if (
            destination == report
            and "reference-match-stage" in source.name
        ):
            raise OSError("injected runtime staging failure")
        original_replace(source, destination)

    monkeypatch.setattr(file_module, "_replace", fail_report)
    with pytest.raises(OSError, match="injected runtime staging failure"):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            expected_prior_output_sha256=tuple(
                hashlib.sha256(payload).hexdigest()
                for payload in (b"old-one", b"old-two")
            ),
            expected_prior_report_sha256=hashlib.sha256(
                b"old-report"
            ).hexdigest(),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    for path, payload in previous.items():
        assert path.read_bytes() == payload
    _assert_no_debris(tmp_path)


def test_consumer_pin_is_required_before_any_path_write(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    root = tmp_path / "not-created"
    with pytest.raises(
        ReferenceMatchContractError,
        match="not consumer-pinned",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id="0" * 64,
            applies=applies,
            output_paths=(
                root / "one.png",
                root / "two.png",
            ),
            report_path=root / "run.json",
        )
    assert not root.exists()


def test_existing_files_require_and_bind_exact_prior_hashes(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    old_outputs = (b"old-one", b"old-two")
    old_report = b"old-report"
    for path, payload in zip(outputs, old_outputs, strict=True):
        path.write_bytes(payload)
    report.write_bytes(old_report)
    with pytest.raises(
        ReferenceMatchContractError,
        match="already exists without an expected prior hash",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    prior_outputs = tuple(
        hashlib.sha256(payload).hexdigest() for payload in old_outputs
    )
    prior_report = hashlib.sha256(old_report).hexdigest()
    committed = commit_runtime_qualified_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        runtime_qualification=qualification,
        expected_runtime_qualification_id=qualification.qualification_id,
        expected_prior_output_sha256=prior_outputs,
        expected_prior_report_sha256=prior_report,
        applies=applies,
        output_paths=outputs,
        report_path=report,
    )
    assert committed.run.prior_output_file_sha256s == prior_outputs
    assert committed.run.prior_report_file_sha256 == prior_report
    assert all(
        path.read_bytes() != prior
        for path, prior in zip(outputs, old_outputs, strict=True)
    )


def test_prior_hash_mismatch_rejects_without_replacement(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    outputs[0].write_bytes(b"current")
    with pytest.raises(
        ReferenceMatchContractError,
        match="prior file hash mismatch",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            expected_prior_output_sha256=(
                "0" * 64,
                None,
            ),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert outputs[0].read_bytes() == b"current"
    assert not outputs[1].exists()
    assert not report.exists()


def test_intervening_destination_creation_during_encoding_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    original_encode = runtime_staging_module.encode_sdr_staging_output
    injected = False

    def create_intervening_destination(*args, **kwargs):
        nonlocal injected
        result = original_encode(*args, **kwargs)
        if not injected:
            outputs[0].write_bytes(b"intervening-owner-bytes")
            injected = True
        return result

    monkeypatch.setattr(
        runtime_staging_module,
        "encode_sdr_staging_output",
        create_intervening_destination,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="already exists without an expected prior hash",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert outputs[0].read_bytes() == b"intervening-owner-bytes"
    assert not outputs[1].exists()
    assert not report.exists()
    _assert_no_debris(tmp_path)


def test_symlink_destination_or_ancestor_rejects(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    target = tmp_path / "target.png"
    target.write_bytes(b"protected")
    link = tmp_path / "link.png"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(
        ReferenceMatchContractError,
        match="symlink or reparse point",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=(link, tmp_path / "two.png"),
            report_path=tmp_path / "run.json",
        )
    assert target.read_bytes() == b"protected"
    assert link.is_symlink()


def test_reparse_component_guard_is_fail_closed_without_os_privilege(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    blocked = tmp_path / "blocked"
    original = runtime_staging_module._is_reparse_point

    def mark_blocked(path: Path) -> bool:
        return path == blocked or original(path)

    monkeypatch.setattr(
        runtime_staging_module,
        "_is_reparse_point",
        mark_blocked,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="symlink or reparse point",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=(
                blocked / "one.png",
                tmp_path / "two.png",
            ),
            report_path=tmp_path / "run.json",
        )
    assert not blocked.exists()


def test_overlapping_target_lock_fails_closed() -> None:
    paths = (
        Path("C:/runtime-lock-test/one.png"),
        Path("C:/runtime-lock-test/report.json"),
    )
    with runtime_staging_module._target_transaction_lock(paths):
        with pytest.raises(
            ReferenceMatchContractError,
            match="already locked",
        ):
            with runtime_staging_module._target_transaction_lock(
                tuple(reversed(paths))
            ):
                raise AssertionError("overlapping lock must not be entered")


def test_overlapping_target_lock_is_cross_process(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    paths = (
        tmp_path / "one.png",
        tmp_path / "run.json",
    )
    script = "\n".join(
        (
            "import sys, time",
            "from pathlib import Path",
            "from src.color_match.shared_runtime_staging_transaction "
            "import _target_transaction_lock",
            "ready, release = Path(sys.argv[1]), Path(sys.argv[2])",
            "paths = tuple(Path(value) for value in sys.argv[3:])",
            "with _target_transaction_lock(paths):",
            "    ready.write_text('ready', encoding='ascii')",
            "    deadline = time.monotonic() + 10.0",
            "    while not release.exists():",
            "        if time.monotonic() >= deadline:",
            "            raise TimeoutError('release timeout')",
            "        time.sleep(0.02)",
        )
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(ready),
            str(release),
            *(str(path) for path in paths),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while not ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("child lock acquisition timeout")
            time.sleep(0.02)
        assert ready.is_file()
        with pytest.raises(
            ReferenceMatchContractError,
            match="already locked",
        ):
            with runtime_staging_module._target_transaction_lock(paths):
                raise AssertionError("cross-process lock must not be entered")
    finally:
        release.write_text("release", encoding="ascii")
        stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, (stdout, stderr)


def test_lock_close_failure_cannot_poison_in_process_lock_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = (tmp_path / "one.png", tmp_path / "run.json")
    original_open = Path.open
    injected = False

    class CloseFailingHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def close(self) -> None:
            nonlocal injected
            self._handle.close()
            if not injected:
                injected = True
                raise OSError("injected lock close failure")

    def open_with_close_failure(path: Path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name.endswith(".lock"):
            return CloseFailingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", open_with_close_failure)
    with runtime_staging_module._target_transaction_lock(paths):
        pass
    assert injected
    assert runtime_staging_module._HELD_LOCK_KEYS.isdisjoint(
        runtime_staging_module._lock_key(path) for path in paths
    )
    with runtime_staging_module._target_transaction_lock(paths):
        pass


def test_caller_pixel_mutation_after_snapshot_cannot_change_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _pipeline()
    baseline_outputs = (
        tmp_path / "baseline-one.png",
        tmp_path / "baseline-two.png",
    )
    baseline_run = commit_runtime_qualified_external_shared_staging_v1(
        batch=baseline[1],
        numeric_guard=baseline[2],
        authorization=baseline[3],
        runtime_qualification=baseline[4],
        expected_runtime_qualification_id=baseline[4].qualification_id,
        applies=baseline[0],
        output_paths=baseline_outputs,
        report_path=tmp_path / "baseline.json",
    )
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    original_validate = (
        runtime_staging_module.validate_sdr_staging_destinations
    )

    def mutate_after_snapshot(*args, **kwargs) -> None:
        applies[0].pixels.flags.writeable = True
        applies[0].pixels.fill(0.99)
        original_validate(*args, **kwargs)

    monkeypatch.setattr(
        runtime_staging_module,
        "validate_sdr_staging_destinations",
        mutate_after_snapshot,
    )
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
    assert outputs[0].read_bytes() == baseline_outputs[0].read_bytes()
    assert committed.run.outputs[0].output_file_sha256 == (
        baseline_run.run.outputs[0].output_file_sha256
    )


def test_staged_output_mutation_after_first_hash_rejects_without_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    original_sha256_file = runtime_staging_module.sha256_file
    mutated: set[Path] = set()

    def mutate_after_first_hash(path: Path) -> str:
        digest = original_sha256_file(path)
        if (
            "reference-match-stage" in path.name
            and path.suffix.casefold() == ".png"
            and path not in mutated
        ):
            mutated.add(path)
            path.write_bytes(path.read_bytes() + b"tamper")
        return digest

    monkeypatch.setattr(
        runtime_staging_module,
        "sha256_file",
        mutate_after_first_hash,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="staged output changed before commit",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert not any(path.exists() for path in outputs)
    assert not report.exists()
    _assert_no_debris(tmp_path)


def test_staged_output_mutation_at_commit_boundary_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    original_commit = runtime_staging_module._commit_staged_batch

    def tamper_then_commit(pairs, **kwargs) -> None:
        first_stage = pairs[0][0]
        first_stage.write_bytes(first_stage.read_bytes() + b"tamper")
        original_commit(pairs, **kwargs)

    monkeypatch.setattr(
        runtime_staging_module,
        "_commit_staged_batch",
        tamper_then_commit,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="staged file changed before batch replacement",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert not any(path.exists() for path in outputs)
    assert not report.exists()
    _assert_no_debris(tmp_path)


def test_destination_creation_at_commit_boundary_rejects_and_preserves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    original_commit = runtime_staging_module._commit_staged_batch

    def create_then_commit(pairs, **kwargs) -> None:
        pairs[0][1].write_bytes(b"intervening-at-commit")
        original_commit(pairs, **kwargs)

    monkeypatch.setattr(
        runtime_staging_module,
        "_commit_staged_batch",
        create_then_commit,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="destination appeared before batch replacement",
    ):
        commit_runtime_qualified_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            runtime_qualification=qualification,
            expected_runtime_qualification_id=(
                qualification.qualification_id
            ),
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert outputs[0].read_bytes() == b"intervening-at-commit"
    assert not outputs[1].exists()
    assert not report.exists()
    _assert_no_debris(tmp_path)


def test_p50_and_p62_reports_cannot_be_cross_parsed(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    legacy_report = tmp_path / "legacy.json"
    commit_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        applies=applies,
        output_paths=(
            tmp_path / "legacy-one.png",
            tmp_path / "legacy-two.png",
        ),
        report_path=legacy_report,
    )
    p62_report = tmp_path / "p62.json"
    commit_runtime_qualified_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        runtime_qualification=qualification,
        expected_runtime_qualification_id=qualification.qualification_id,
        applies=applies,
        output_paths=(
            tmp_path / "p62-one.png",
            tmp_path / "p62-two.png",
        ),
        report_path=p62_report,
    )
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        runtime_qualified_external_shared_staging_run_from_json(
            legacy_report.read_text(encoding="utf-8")
        )
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        external_shared_staging_run_from_json(
            p62_report.read_text(encoding="utf-8")
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, state="applied"),
            "state is invalid",
        ),
        (
            lambda value: replace(
                value,
                runtime_qualification_id="0" * 64,
            ),
            "run identity mismatch",
        ),
        (
            lambda value: replace(
                value,
                runtime_evidence_id="0" * 64,
            ),
            "run identity mismatch",
        ),
        (
            lambda value: replace(
                value,
                declaration_id="0" * 64,
            ),
            "run identity mismatch",
        ),
        (
            lambda value: replace(value, claim_ceiling="delivered"),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(value, run_id="0" * 64),
            "run identity mismatch",
        ),
    ],
)
def test_report_fact_or_identity_mutation_fails_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
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
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_runtime_qualified_external_shared_staging_run_v1(
            mutation(committed.run)
        )


def test_unknown_report_field_and_schema_constants_fail_closed(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    run = commit_runtime_qualified_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        runtime_qualification=qualification,
        expected_runtime_qualification_id=qualification.qualification_id,
        applies=applies,
        output_paths=outputs,
        report_path=report,
    ).run
    payload = deepcopy(run.to_dict())
    payload["unexpected"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        runtime_qualified_external_shared_staging_run_from_json(
            json.dumps(payload)
        )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_id"]["const"] == (
        RUNTIME_QUALIFIED_SHARED_STAGING_SCHEMA_ID
    )
    assert schema["properties"]["claim_ceiling"]["const"] == (
        RUNTIME_QUALIFIED_SHARED_STAGING_CLAIM_CEILING
    )
    assert "(?i)" not in schema["properties"]["report_path"]["pattern"]


def test_relative_report_paths_reject_even_before_identity_check(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization, qualification = _pipeline()
    outputs, report = _destinations(tmp_path)
    run = commit_runtime_qualified_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        runtime_qualification=qualification,
        expected_runtime_qualification_id=qualification.qualification_id,
        applies=applies,
        output_paths=outputs,
        report_path=report,
    ).run
    relative = replace(
        run,
        outputs=(
            replace(run.outputs[0], output_path="relative.png"),
            run.outputs[1],
        ),
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="canonical absolute paths",
    ):
        validate_runtime_qualified_external_shared_staging_run_v1(
            relative
        )
