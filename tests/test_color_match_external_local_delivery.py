from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

import src.color_match.files as file_module
from src.color_match import (
    EXTERNAL_LOCAL_DELIVERY_CLAIM_CEILING,
    ReferenceMatchContractError,
    commit_external_local_delivery_v1,
    external_local_delivery_from_json,
    external_local_delivery_to_json,
    validate_external_local_delivery_v1,
)
from src.inference import sha256_file
from tests.test_color_match_external_delivery_authorization import (
    _authorize,
    _chain,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_local_delivery_v1.schema.json"
)


def _destinations(root: Path) -> tuple[tuple[Path, ...], Path]:
    return (
        (root / "delivered-one.png", root / "delivered-two.png"),
        root / "delivery-report.json",
    )


def _commit(chain, authorization, outputs, report):
    filmfx, composition, core, product, _staging_outputs = chain
    return commit_external_local_delivery_v1(
        authorization=authorization,
        filmfx_verification=filmfx,
        composition=composition,
        core_verification=core,
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
    assert committed.delivery.state == "committed-local-delivery"
    assert (
        committed.delivery.claim_ceiling
        == EXTERNAL_LOCAL_DELIVERY_CLAIM_CEILING
    )
    assert committed.delivery.authorization_id == authorization.authorization_id
    assert committed.report_file_sha256 == sha256_file(report)
    for source_row, destination, row in zip(
        chain[0].outputs,
        outputs,
        committed.delivery.outputs,
        strict=True,
    ):
        assert destination.read_bytes() == Path(
            source_row.output_path
        ).read_bytes()
        assert row.delivered_file_sha256 == sha256_file(destination)
        assert (
            row.delivered_file_sha256
            == row.staging_output_file_sha256
        )
    decoded = external_local_delivery_from_json(
        report.read_text(encoding="utf-8")
    )
    assert decoded == committed.delivery
    assert (
        external_local_delivery_from_json(
            external_local_delivery_to_json(decoded)
        )
        == decoded
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(decoded.to_dict())
    _assert_no_debris(tmp_path)


def test_live_tamper_revokes_delivery_before_destination_write(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    chain[4][0].write_bytes(b"tampered")
    outputs, report = _destinations(tmp_path / "delivery")
    with pytest.raises(
        ReferenceMatchContractError,
        match="output 0 hash mismatch",
    ):
        _commit(chain, authorization, outputs, report)
    assert not report.exists()
    assert not any(path.exists() for path in outputs)


def test_foreign_valid_authorization_cannot_drive_chain(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path / "first")
    foreign = _authorize(_chain(tmp_path / "second"))
    outputs, report = _destinations(tmp_path / "delivery")
    with pytest.raises(
        ReferenceMatchContractError,
        match="authorization changed on refresh",
    ):
        _commit(chain, foreign, outputs, report)


def test_staging_and_report_paths_are_protected(tmp_path: Path) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    protected = Path(chain[0].outputs[0].output_path)
    _outputs, report = _destinations(tmp_path / "delivery")
    with pytest.raises(
        ReferenceMatchContractError,
        match="must not overwrite staging",
    ):
        _commit(
            chain,
            authorization,
            (protected, tmp_path / "delivery" / "second.png"),
            report,
        )


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (
            lambda root: (root / "only-one.png",),
            "count must match",
        ),
        (
            lambda root: (root / "one.jpg", root / "two.png"),
            "extension must match",
        ),
    ],
)
def test_invalid_destination_shape_or_format_fails_before_write(
    tmp_path: Path,
    outputs,
    message: str,
) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    destination_root = tmp_path / "delivery"
    report = destination_root / "report.json"
    with pytest.raises(ReferenceMatchContractError, match=message):
        _commit(
            chain,
            authorization,
            outputs(destination_root),
            report,
        )
    assert not report.exists()


def test_commit_failure_restores_every_previous_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    outputs, report = _destinations(tmp_path / "delivery")
    previous = {
        outputs[0]: b"old-one",
        outputs[1]: b"old-two",
        report: b"old-report",
    }
    for path, payload in previous.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    original_replace = file_module._move_noreplace

    def fail_report(source: Path, destination: Path) -> None:
        if destination == report and "reference-match-stage" in source.name:
            raise OSError("injected local delivery failure")
        original_replace(source, destination)

    monkeypatch.setattr(file_module, "_move_noreplace", fail_report)
    with pytest.raises(OSError, match="injected local delivery failure"):
        _commit(chain, authorization, outputs, report)
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
            lambda value: replace(value, claim_ceiling="film-stock"),
            "claim ceiling mismatch",
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
            "preserve exact file bytes",
        ),
        (
            lambda value: replace(value, delivery_id="0" * 64),
            "delivery_id mismatch",
        ),
    ],
)
def test_delivery_mutations_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    chain = _chain(tmp_path / "chain")
    authorization = _authorize(chain)
    outputs, report = _destinations(tmp_path / "delivery")
    delivery = _commit(
        chain,
        authorization,
        outputs,
        report,
    ).delivery
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_external_local_delivery_v1(mutation(delivery))
