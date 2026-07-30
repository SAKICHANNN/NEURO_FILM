from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

import src.color_match.files as file_module
from src.color_match import (
    SHARED_LOCAL_DELIVERY_CLAIM_CEILING,
    ReferenceMatchContractError,
    commit_shared_local_delivery_v1,
    shared_local_delivery_from_json,
    shared_local_delivery_to_json,
    validate_shared_local_delivery_v1,
)
from src.inference import sha256_file
from tests.test_color_match_shared_delivery_authorization import (
    _authorize,
    _chain,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_shared_local_delivery_v1.schema.json"
)


def _destinations(root: Path):
    return (
        (root / "delivered-one.png", root / "delivered-two.png"),
        root / "delivery-report.json",
    )


def _commit(chain, authorization, outputs, report):
    filmfx, composition, staging, product, _staging_outputs = chain
    return commit_shared_local_delivery_v1(
        authorization=authorization,
        filmfx_verification=filmfx,
        composition=composition,
        staging_verification=staging,
        product_authorization=product,
        output_paths=outputs,
        report_path=report,
    )


def _assert_no_debris(root: Path) -> None:
    assert not list(root.rglob(".*.reference-match-stage*"))
    assert not list(root.rglob(".*.reference-match-backup"))


def test_authorized_files_and_report_commit_atomically(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    outputs, report = _destinations(tmp_path / "delivery")
    committed = _commit(chain, authorization, outputs, report)
    delivery = committed.delivery
    assert delivery.state == "committed-shared-local-delivery"
    assert delivery.claim_ceiling == SHARED_LOCAL_DELIVERY_CLAIM_CEILING
    assert delivery.authorization_id == authorization.authorization_id
    assert committed.report_file_sha256 == sha256_file(report)
    for source, target, row in zip(
        chain[0].outputs, outputs, delivery.outputs, strict=True
    ):
        assert target.read_bytes() == Path(source.output_path).read_bytes()
        assert row.apply_receipt_id == source.apply_receipt_id
        assert row.producer_apply_result_id == source.producer_apply_result_id
        assert row.delivered_file_sha256 == sha256_file(target)
        assert row.delivered_file_sha256 == row.staging_output_file_sha256
    decoded = shared_local_delivery_from_json(
        report.read_text(encoding="utf-8")
    )
    assert decoded == delivery
    assert shared_local_delivery_from_json(
        shared_local_delivery_to_json(decoded)
    ) == decoded
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(decoded.to_dict())
    _assert_no_debris(tmp_path)


def test_live_tamper_and_foreign_authorization_fail_before_write(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    outputs, report = _destinations(tmp_path / "delivery")
    chain[4][0].write_bytes(b"tampered")
    with pytest.raises(ReferenceMatchContractError, match="hash mismatch"):
        _commit(chain, authorization, outputs, report)
    assert not any(path.exists() for path in outputs)
    chain = _chain(tmp_path / "fresh")
    foreign = _authorize(_chain(tmp_path / "foreign", different=True))
    with pytest.raises(
        ReferenceMatchContractError,
        match="authorization changed on refresh",
    ):
        _commit(chain, foreign, outputs, report)


def test_protected_or_invalid_destinations_fail_closed(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    protected = Path(chain[0].outputs[0].output_path)
    report = tmp_path / "delivery" / "report.json"
    with pytest.raises(ReferenceMatchContractError, match="overwrite staging"):
        _commit(
            chain,
            authorization,
            (protected, tmp_path / "delivery" / "second.png"),
            report,
        )
    with pytest.raises(ReferenceMatchContractError, match="count must match"):
        _commit(chain, authorization, (tmp_path / "one.png",), report)
    with pytest.raises(ReferenceMatchContractError, match="extension"):
        _commit(
            chain,
            authorization,
            (tmp_path / "one.jpg", tmp_path / "two.png"),
            report,
        )


def test_commit_failure_restores_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    outputs, report = _destinations(tmp_path / "delivery")
    previous = {outputs[0]: b"old-1", outputs[1]: b"old-2", report: b"old-r"}
    for path, payload in previous.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    original = file_module._replace

    def fail_report(source: Path, destination: Path) -> None:
        if destination == report and "reference-match-stage" in source.name:
            raise OSError("injected shared delivery failure")
        original(source, destination)

    monkeypatch.setattr(file_module, "_replace", fail_report)
    with pytest.raises(OSError, match="injected shared delivery failure"):
        _commit(chain, authorization, outputs, report)
    for path, payload in previous.items():
        assert path.read_bytes() == payload
    _assert_no_debris(tmp_path)


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
                outputs=(
                    replace(v.outputs[0], delivered_file_sha256="0" * 64),
                    v.outputs[1],
                ),
            ),
            "preserve exact bytes",
        ),
        (
            lambda v: replace(v, delivery_id="0" * 64),
            "delivery identity mismatch",
        ),
    ],
)
def test_mutations_fail_closed(tmp_path: Path, mutation, message: str) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    outputs, report = _destinations(tmp_path / "delivery")
    delivery = _commit(chain, authorization, outputs, report).delivery
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_shared_local_delivery_v1(mutation(delivery))
