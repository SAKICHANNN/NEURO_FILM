from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

import src.color_match.files as file_module
from src.color_match import (
    EXTERNAL_CORE_STAGING_CLAIM_CEILING,
    MATCH_PROFILE_DISPLAY_SRGB,
    CoreNumericGuardPolicyV1,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    adapt_dpct_candidate_v2,
    admit_core_apply_receipt,
    adjudicate_core_acceptance,
    authorize_core_product_staging_v1,
    commit_external_core_staging_v1,
    external_core_staging_run_from_json,
    external_core_staging_run_to_json,
    guard_core_candidate_numeric_v1,
    guard_core_numeric_batch_v1,
    make_match_view,
    resolve_dpct_batch_v1,
    validate_external_core_staging_run_v1,
)
from src.inference import sha256_file


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "zhuise_producer_contract_exact_bits_v2.json"
)
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_core_staging_run_v1.schema.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _prepared(encoded_hex: str, provenance: str) -> PreparedMatchViewV1:
    wire = bytes.fromhex(encoded_hex)
    pixels = np.frombuffer(wire, dtype=">f4").astype(np.float32, copy=True)
    pixels = np.ascontiguousarray(pixels.reshape(2, 2, 3))
    pixels.flags.writeable = False
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=(2, 2, 3),
        render_bridge_id="neuro-film.staging-transaction-test.v1",
        provenance_fingerprint=provenance * 64,
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _candidate(
    reference: PreparedMatchViewV1,
    *,
    provenance: str,
    intent: str = "1",
    promotion: PromotionDecision,
    research_override: bool,
):
    fixture = _fixture()
    source = _prepared(fixture["source_pixel_f32be_hex"], provenance)
    candidate = adapt_dpct_candidate_v2(
        source=source,
        reference=reference,
        producer_source=fixture["source"],
        producer_reference=fixture["reference"],
        producer_transform=fixture["transform"],
        producer_transform_payload=bytes.fromhex(fixture["payload_hex"]),
        producer_diagnostics=fixture["diagnostics"],
        producer_apply_result=fixture["apply_result"],
        output_pixel_f32be=bytes.fromhex(
            fixture["output_pixel_f32be_hex"]
        ),
        intent_id=intent * 64,
    )
    acceptance = adjudicate_core_acceptance(
        source=source.descriptor,
        reference=reference.descriptor,
        transform=candidate.transform,
        capabilities=candidate.capabilities,
        diagnostics=candidate.diagnostics,
        promotion=promotion,
        allow_research_baseline=research_override,
    )
    admission = admit_core_apply_receipt(
        prepared=candidate.prepared_output,
        acceptance=acceptance,
        source=source.descriptor,
        reference=reference.descriptor,
        transform=candidate.transform,
        capabilities=candidate.capabilities,
        diagnostics=candidate.diagnostics,
    )
    return source, candidate, acceptance, admission


def _pipeline(
    *,
    research_override: bool = False,
    different_intents: bool = False,
):
    fixture = _fixture()
    reference = _prepared(fixture["reference_pixel_f32be_hex"], "a")
    first = _candidate(
        reference,
        provenance="b",
        promotion=PromotionDecision("promoted", ()),
        research_override=False,
    )
    second = _candidate(
        reference,
        provenance="c",
        intent="2" if different_intents else "1",
        promotion=(
            PromotionDecision("rejected", ("research-only",))
            if research_override
            else PromotionDecision("promoted", ())
        ),
        research_override=research_override,
    )
    items = (first, second)
    batch = resolve_dpct_batch_v1(
        reference=reference,
        sources=tuple(item[0] for item in items),
        outcomes=tuple(item[1] for item in items),
        adjudications=tuple((item[2], item[3]) for item in items),
    )
    policy = CoreNumericGuardPolicyV1(
        max_out_of_gamut_fraction=1.0,
        max_clipping_fraction=1.0,
        max_projected_fraction=1.0,
        max_new_boundary_fraction=1.0,
    )
    decisions = tuple(
        guard_core_candidate_numeric_v1(
            source=item[0],
            reference=reference,
            candidate=item[1],
            acceptance=item[2],
            admission=item[3],
            policy=policy,
        )
        for item in items
    )
    numeric = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=decisions,
    )
    authorization = authorize_core_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        acceptances=tuple(item[2] for item in items),
    )
    return items, batch, authorization


def _destinations(tmp_path: Path) -> tuple[tuple[Path, ...], Path]:
    return (
        (tmp_path / "one.png", tmp_path / "two.png"),
        tmp_path / "staging-report.json",
    )


def _assert_no_debris(root: Path) -> None:
    assert not list(root.glob(".*.reference-match-stage*"))
    assert not list(root.glob(".*.reference-match-backup"))


def test_authorized_batch_commits_outputs_and_strict_report(
    tmp_path: Path,
) -> None:
    items, batch, authorization = _pipeline()
    outputs, report = _destinations(tmp_path)
    committed = commit_external_core_staging_v1(
        batch=batch,
        authorization=authorization,
        candidates=tuple(item[1] for item in items),
        output_paths=outputs,
        report_path=report,
    )
    assert committed.run.state == "committed-to-staging"
    assert (
        committed.run.claim_ceiling
        == EXTERNAL_CORE_STAGING_CLAIM_CEILING
    )
    assert committed.run.reference_intent_id == "1" * 64
    assert committed.report_file_sha256 == sha256_file(report)
    for output, row in zip(outputs, committed.run.outputs, strict=True):
        assert output.is_file()
        assert row.output_file_sha256 == sha256_file(output)
        assert row.output_bit_depth == 16
        assert row.apply_receipt_id == batch.sources[
            row.source_index
        ].apply_receipt_id
    decoded = external_core_staging_run_from_json(
        report.read_text(encoding="utf-8")
    )
    assert decoded == committed.run
    assert (
        external_core_staging_run_from_json(
            external_core_staging_run_to_json(decoded)
        )
        == decoded
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(decoded.to_dict())
    _assert_no_debris(tmp_path)


def test_research_override_fallback_writes_nothing(
    tmp_path: Path,
) -> None:
    items, batch, authorization = _pipeline(research_override=True)
    assert authorization.state == "identity-fallback"
    outputs, report = _destinations(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="authorized complete batch",
    ):
        commit_external_core_staging_v1(
            batch=batch,
            authorization=authorization,
            candidates=tuple(item[1] for item in items),
            output_paths=outputs,
            report_path=report,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)
    _assert_no_debris(tmp_path)


def test_candidate_order_and_destination_collisions_fail_before_write(
    tmp_path: Path,
) -> None:
    items, batch, authorization = _pipeline()
    outputs, report = _destinations(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind batch",
    ):
        commit_external_core_staging_v1(
            batch=batch,
            authorization=authorization,
            candidates=tuple(
                reversed(tuple(item[1] for item in items))
            ),
            output_paths=outputs,
            report_path=report,
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="output paths must be unique",
    ):
        commit_external_core_staging_v1(
            batch=batch,
            authorization=authorization,
            candidates=tuple(item[1] for item in items),
            output_paths=(outputs[0], outputs[0]),
            report_path=report,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)
    _assert_no_debris(tmp_path)


def test_shared_reference_intent_is_mandatory(
    tmp_path: Path,
) -> None:
    items, batch, authorization = _pipeline(different_intents=True)
    outputs, report = _destinations(tmp_path)
    with pytest.raises(
        ReferenceMatchContractError,
        match="share one reference intent",
    ):
        commit_external_core_staging_v1(
            batch=batch,
            authorization=authorization,
            candidates=tuple(item[1] for item in items),
            output_paths=outputs,
            report_path=report,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)


def test_commit_failure_restores_all_previous_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items, batch, authorization = _pipeline()
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
            raise OSError("injected external staging failure")
        original_replace(source, destination)

    monkeypatch.setattr(file_module, "_replace", fail_report)
    with pytest.raises(OSError, match="injected external staging failure"):
        commit_external_core_staging_v1(
            batch=batch,
            authorization=authorization,
            candidates=tuple(item[1] for item in items),
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
                claim_ceiling="delivered",
            ),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(value, run_id="0" * 64),
            "run_id mismatch",
        ),
    ],
)
def test_report_mutations_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    items, batch, authorization = _pipeline()
    outputs, report = _destinations(tmp_path)
    committed = commit_external_core_staging_v1(
        batch=batch,
        authorization=authorization,
        candidates=tuple(item[1] for item in items),
        output_paths=outputs,
        report_path=report,
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_external_core_staging_run_v1(mutation(committed.run))
