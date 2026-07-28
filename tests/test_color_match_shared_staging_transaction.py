from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

import src.color_match.files as file_module
import src.color_match.shared_staging_transaction as transaction_module
from src.color_match import (
    EXTERNAL_SHARED_STAGING_CLAIM_CEILING,
    FROZEN_GATE_POLICY_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    SUCCESSOR_DECLARATION_SCHEMA,
    SUCCESSOR_POLICY_ID,
    SUPPORTED_PROFILE_ID,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    authorize_shared_product_staging_v1,
    bind_shared_promotion_v1,
    commit_external_shared_staging_v1,
    external_shared_staging_run_from_json,
    external_shared_staging_run_to_json,
    guard_shared_numeric_batch_v1,
    make_match_view,
    make_shared_apply_numeric_facts_v1,
    make_shared_reference_operator_v1,
    prepare_shared_operator_apply_v1,
    resolve_shared_operator_batch_v1,
    successor_declaration_id_v1,
    validate_external_shared_staging_run_v1,
)
from src.inference import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_shared_staging_run_v1.schema.json"
)
CAPABILITY = "zhuise.spgin.cpu-reference.v1"
PRODUCER_COMMIT = "1" * 40
MODEL = "d" * 64
OPTIONS = "e" * 64
EVIDENCE = "f" * 64


def _prepared(seed: float) -> PreparedMatchViewV1:
    pixels = np.full((10, 10, 3), seed, dtype=np.float32)
    wire = pixels.astype(">f4", copy=False).tobytes()
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=pixels.shape,
        render_bridge_id="neuro-film.shared-staging-test.v1",
        provenance_fingerprint=hashlib.sha256(
            str(seed).encode()
        ).hexdigest(),
    )
    pixels.flags.writeable = False
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _declaration(*, product: bool = True) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": SUCCESSOR_DECLARATION_SCHEMA,
        "policy_id": SUCCESSOR_POLICY_ID,
        "candidate_id": "zhuise.spgin.shared.v1",
        "producer": {
            "stable_commit": PRODUCER_COMMIT,
            "package_source_commit": "2" * 40,
            "fixture_commit": "3" * 40,
            "package_lock_sha256": "a" * 64,
            "conformance_fixture_sha256": "b" * 64,
        },
        "package": {
            "distribution": "zhuise-research",
            "version": "0.5.0",
            "entrypoint": "zhuise-shared-invoke",
            "wheel_filename": "zhuise_research-0.5.0-py3-none-any.whl",
            "wheel_size_bytes": 120000,
            "wheel_sha256": "c" * 64,
        },
        "wire": {
            "capability_id": CAPABILITY,
            "profile_id": SUPPORTED_PROFILE_ID,
            "lower_compatibility_profile_id": (
                "neuro-film.dpct-consumer.v2"
            ),
            "request_schema": "zhuise.shared-request.v1",
            "request_schema_sha256": "a" * 64,
            "response_schema": "zhuise.shared-response.v1",
            "response_schema_sha256": "b" * 64,
        },
        "semantics": {
            "fit_semantics": "reference-only-shared",
            "batch_transform_policy": "shared-bundle",
            "deterministic": True,
            "hidden_state": "none",
        },
        "rights": {
            "evaluation_allowed": True,
            "commercial_use_allowed": product,
            "redistribution_allowed": product,
            "evidence_id": "rights-review-v1",
        },
        "runtime_evidence": {
            "windows_x64": product,
            "macos_arm64": product,
            "ios_arm64": product,
            "android_arm64": product,
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


def _pipeline(
    *,
    product: bool = True,
    promotion: PromotionDecision | None = None,
    second_clip: float = 0.01,
    unsafe_pixel: bool = False,
):
    reference = _prepared(0.2)
    sources = (_prepared(0.3), _prepared(0.4))
    declaration = _declaration(product=product)
    operator = make_shared_reference_operator_v1(
        reference=reference,
        compatibility_profile_id="neuro-film.dpct-consumer.v2",
        capability_id=CAPABILITY,
        producer_commit=PRODUCER_COMMIT,
        producer_bundle_id="sha256:" + "9" * 64,
        producer_reference_view_id="sha256:" + "8" * 64,
        model_fingerprint=MODEL,
        options_sha256=OPTIONS,
    )
    output_arrays = [
        np.full((10, 10, 3), 0.31, dtype=np.float32),
        np.full((10, 10, 3), 0.41, dtype=np.float32),
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
            out_of_gamut_fraction=(
                second_clip if index == 1 else 0.01
            ),
            clipping_fraction=second_clip if index == 1 else 0.01,
            projected_fraction=0.0,
        )
        for index, prepared in enumerate(applies)
    )
    numeric = guard_shared_numeric_batch_v1(
        batch=batch,
        sources=sources,
        applies=applies,
        facts=facts,
    )
    binding = bind_shared_promotion_v1(
        operator=operator,
        declaration=declaration,
        stable_evidence_id=EVIDENCE,
        promotion=promotion or PromotionDecision("promoted", ()),
    )
    authorization = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    return applies, batch, numeric, authorization


def _destinations(tmp_path: Path) -> tuple[tuple[Path, ...], Path]:
    return (
        (tmp_path / "one.png", tmp_path / "two.png"),
        tmp_path / "shared-staging-report.json",
    )


def _assert_no_debris(root: Path) -> None:
    assert not list(root.glob(".*.reference-match-stage*"))
    assert not list(root.glob(".*.reference-match-backup"))


def test_authorized_shared_batch_commits_exact_outputs_and_report(
    tmp_path: Path,
) -> None:
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
    assert committed.run.state == "committed-to-shared-staging"
    assert (
        committed.run.claim_ceiling
        == EXTERNAL_SHARED_STAGING_CLAIM_CEILING
    )
    assert committed.run.authorization_id == authorization.authorization_id
    assert committed.run.numeric_guard_batch_id == numeric.guard_batch_id
    assert committed.run.operator_id == batch.operator.operator_id
    assert committed.report_file_sha256 == sha256_file(report)
    for output, row, receipt in zip(
        outputs, committed.run.outputs, batch.sources, strict=True
    ):
        assert row.output_file_sha256 == sha256_file(output)
        assert row.apply_receipt_id == receipt.receipt_id
        assert row.producer_apply_result_id == (
            receipt.producer_apply_result_id
        )
        assert row.output_bit_depth == 16
    decoded = external_shared_staging_run_from_json(
        report.read_text(encoding="utf-8")
    )
    assert decoded == committed.run
    assert (
        external_shared_staging_run_from_json(
            external_shared_staging_run_to_json(decoded)
        )
        == decoded
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(decoded.to_dict())
    _assert_no_debris(tmp_path)


def test_legacy_shared_staging_still_replaces_existing_destinations(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization = _pipeline()
    outputs, report = _destinations(tmp_path)
    first = commit_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        applies=applies,
        output_paths=outputs,
        report_path=report,
    )
    expected_outputs = tuple(path.read_bytes() for path in outputs)
    expected_report = report.read_bytes()
    for path in (*outputs, report):
        path.write_bytes(b"intervening-existing-destination")
    second = commit_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        applies=applies,
        output_paths=outputs,
        report_path=report,
    )
    assert second == first
    assert tuple(path.read_bytes() for path in outputs) == expected_outputs
    assert report.read_bytes() == expected_report
    _assert_no_debris(tmp_path)


@pytest.mark.parametrize(
    "fallback",
    ["product", "promotion", "numeric"],
)
def test_any_upstream_fallback_writes_nothing(
    tmp_path: Path,
    fallback: str,
) -> None:
    kwargs: dict[str, object] = {}
    if fallback == "product":
        kwargs["product"] = False
    elif fallback == "promotion":
        kwargs["promotion"] = PromotionDecision(
            "eligible-for-visual-review",
            ("blind-aesthetic-review-required",),
        )
    else:
        kwargs["second_clip"] = 0.06
    applies, batch, numeric, authorization = _pipeline(**kwargs)
    assert authorization.state == "identity-fallback"
    outputs, report = _destinations(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="authorized complete batch",
    ):
        commit_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)
    _assert_no_debris(tmp_path)


def test_foreign_apply_guard_and_path_collision_fail_before_write(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization = _pipeline()
    outputs, report = _destinations(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind the chain",
    ):
        commit_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            applies=tuple(reversed(applies)),
            output_paths=outputs,
            report_path=report,
        )
    foreign_applies, _foreign_batch, foreign_numeric, _foreign_auth = (
        _pipeline()
    )
    assert foreign_applies[0].receipt == applies[0].receipt
    changed_guard = replace(
        foreign_numeric,
        upstream_batch_id="7" * 64,
    )
    with pytest.raises(ReferenceMatchContractError):
        commit_external_shared_staging_v1(
            batch=batch,
            numeric_guard=changed_guard,
            authorization=authorization,
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="output paths must be unique",
    ):
        commit_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            applies=applies,
            output_paths=(outputs[0], outputs[0]),
            report_path=report,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)


def test_sparse_out_of_bounds_pixel_fails_encoding_without_commit(
    tmp_path: Path,
) -> None:
    applies, batch, numeric, authorization = _pipeline(unsafe_pixel=True)
    assert numeric.atomic_state == "eligible-for-transaction"
    assert authorization.state == "authorized-for-staging"
    outputs, report = _destinations(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="bounded sRGB encoding tolerance",
    ):
        commit_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)
    _assert_no_debris(tmp_path)


def test_caller_pixel_mutation_after_validation_cannot_change_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_applies, baseline_batch, baseline_numeric, baseline_auth = (
        _pipeline()
    )
    baseline_outputs = (
        tmp_path / "baseline-one.png",
        tmp_path / "baseline-two.png",
    )
    baseline = commit_external_shared_staging_v1(
        batch=baseline_batch,
        numeric_guard=baseline_numeric,
        authorization=baseline_auth,
        applies=baseline_applies,
        output_paths=baseline_outputs,
        report_path=tmp_path / "baseline.json",
    )

    applies, batch, numeric, authorization = _pipeline()
    outputs, report = _destinations(tmp_path)
    original_validate = (
        transaction_module.validate_sdr_staging_destinations
    )

    def mutate_after_snapshot(*args, **kwargs) -> None:
        applies[0].pixels.flags.writeable = True
        applies[0].pixels.fill(0.99)
        original_validate(*args, **kwargs)

    monkeypatch.setattr(
        transaction_module,
        "validate_sdr_staging_destinations",
        mutate_after_snapshot,
    )
    committed = commit_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=authorization,
        applies=applies,
        output_paths=outputs,
        report_path=report,
    )
    assert outputs[0].read_bytes() == baseline_outputs[0].read_bytes()
    assert committed.run.outputs[0].output_file_sha256 == (
        baseline.run.outputs[0].output_file_sha256
    )


def test_commit_failure_restores_every_previous_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applies, batch, numeric, authorization = _pipeline()
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
            raise OSError("injected shared staging failure")
        original_replace(source, destination)

    monkeypatch.setattr(file_module, "_replace", fail_report)
    with pytest.raises(OSError, match="injected shared staging failure"):
        commit_external_shared_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            authorization=authorization,
            applies=applies,
            output_paths=outputs,
            report_path=report,
        )
    for path, payload in previous.items():
        assert path.read_bytes() == payload
    _assert_no_debris(tmp_path)


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
                outputs=(
                    replace(value.outputs[0], source_index=1),
                    value.outputs[1],
                ),
            ),
            "source order is invalid",
        ),
        (
            lambda value: replace(
                value,
                claim_ceiling="delivered",
            ),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(value, run_id="0" * 64),
            "run identity mismatch",
        ),
    ],
)
def test_report_mutations_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
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
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_external_shared_staging_run_v1(mutation(committed.run))


def test_unknown_report_json_field_fails_closed(tmp_path: Path) -> None:
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
    payload = deepcopy(committed.run.to_dict())
    payload["unexpected"] = True
    with pytest.raises(ReferenceMatchContractError):
        external_shared_staging_run_from_json(json.dumps(payload))
