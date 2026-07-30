from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

import src.color_match.files as file_module
from src.color_match import (
    SHARED_FILMFX_CLAIM_CEILING,
    ReferenceMatchContractError,
    build_shared_reference_composition_v1,
    commit_shared_filmfx_staging_v1,
    shared_filmfx_run_from_json,
    shared_filmfx_run_to_json,
    validate_shared_filmfx_run_v1,
)
from src.inference import sha256_file
from tests.test_color_match_shared_staging_verification import (
    _committed,
    _verify,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_shared_filmfx_run_v1.schema.json"
)
BASE_PROFILE = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"


def _profile(
    tmp_path: Path,
    *,
    model: str = "simple",
    active: bool = True,
) -> tuple[dict, str]:
    profile = json.loads(BASE_PROFILE.read_text(encoding="utf-8"))
    profile["effect_defaults"] = {
        "grain": 0.06 if active else 0.0,
        "halation": 0.04 if active else 0.0,
        "dust": 0.05 if active else 0.0,
        "halation_model": model,
    }
    path = tmp_path / f"profile-{model}-{active}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profile, sha256_file(path)


def _setup(
    tmp_path: Path,
    *,
    model: str = "simple",
    active: bool = True,
):
    committed, staging_outputs, staging_report = _committed(
        tmp_path / "source"
    )
    verification = _verify(committed, staging_report)
    profile, profile_sha = _profile(
        tmp_path,
        model=model,
        active=active,
    )
    plan = build_shared_reference_composition_v1(
        verification,
        include_film_effects=True,
        film_profile=profile,
        film_profile_sha256=profile_sha,
    )
    return staging_outputs, verification, plan


def _destinations(root: Path) -> tuple[tuple[Path, ...], Path]:
    return (
        (root / "fx-one.png", root / "fx-two.png"),
        root / "shared-filmfx-report.json",
    )


def _assert_no_debris(root: Path) -> None:
    assert not list(root.rglob(".*.reference-match-stage*"))
    assert not list(root.rglob(".*.reference-match-backup"))


def test_shared_filmfx_renders_all_sources_atomically(
    tmp_path: Path,
) -> None:
    sources, verification, plan = _setup(tmp_path)
    outputs, report = _destinations(tmp_path / "render")
    committed = commit_shared_filmfx_staging_v1(
        plan=plan,
        verification=verification,
        output_paths=outputs,
        report_path=report,
        seed=37,
    )
    run = committed.run
    assert run.state == "shared-filmfx-rendered-to-staging"
    assert run.claim_ceiling == SHARED_FILMFX_CLAIM_CEILING
    assert run.composition_plan_id == plan.plan_id
    assert run.staging_verification_id == verification.verification_id
    assert run.staging_run_id == verification.run_id
    assert run.authorization_id == verification.authorization_id
    assert run.numeric_guard_batch_id == verification.numeric_guard_batch_id
    assert run.operator_id == verification.operator_id
    assert committed.report_file_sha256 == sha256_file(report)
    for index, (source, output, row, verified) in enumerate(
        zip(
            sources,
            outputs,
            run.outputs,
            verification.outputs,
            strict=True,
        )
    ):
        assert row.source_index == index
        assert row.apply_receipt_id == verified.apply_receipt_id
        assert (
            row.producer_apply_result_id
            == verified.producer_apply_result_id
        )
        assert row.input_file_sha256 == sha256_file(source)
        assert row.output_file_sha256 == sha256_file(output)
        assert row.grain_seed == 37 + index * 1009
        assert row.dust_seed == row.grain_seed + 17
        assert row.output_file_sha256 != row.input_file_sha256
    decoded = shared_filmfx_run_from_json(
        report.read_text(encoding="utf-8")
    )
    assert decoded == run
    assert shared_filmfx_run_from_json(
        shared_filmfx_run_to_json(decoded)
    ) == decoded
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(decoded.to_dict())
    _assert_no_debris(tmp_path)


def test_same_seed_repeats_exact_effect_output_hashes(
    tmp_path: Path,
) -> None:
    _sources, verification, plan = _setup(tmp_path)
    first_outputs, first_report = _destinations(tmp_path / "first")
    second_outputs, second_report = _destinations(tmp_path / "second")
    first = commit_shared_filmfx_staging_v1(
        plan=plan,
        verification=verification,
        output_paths=first_outputs,
        report_path=first_report,
        seed=91,
    )
    second = commit_shared_filmfx_staging_v1(
        plan=plan,
        verification=verification,
        output_paths=second_outputs,
        report_path=second_report,
        seed=91,
    )
    assert [row.output_file_sha256 for row in first.run.outputs] == [
        row.output_file_sha256 for row in second.run.outputs
    ]
    assert first.run.run_id != second.run.run_id


def test_inactive_or_physical_effects_fail_closed(tmp_path: Path) -> None:
    for name, model, active, message in (
        ("empty", "simple", False, "at least one active effect"),
        ("physical", "physical", True, "resolved controls"),
    ):
        root = tmp_path / name
        _sources, verification, plan = _setup(
            root,
            model=model,
            active=active,
        )
        outputs, report = _destinations(root / "render")
        with pytest.raises(ReferenceMatchContractError, match=message):
            commit_shared_filmfx_staging_v1(
                plan=plan,
                verification=verification,
                output_paths=outputs,
                report_path=report,
                seed=7,
            )
        assert not report.exists()
        assert not any(path.exists() for path in outputs)


def test_live_p51_refresh_blocks_tampered_p50_input(
    tmp_path: Path,
) -> None:
    sources, verification, plan = _setup(tmp_path)
    sources[0].write_bytes(b"tampered")
    outputs, report = _destinations(tmp_path / "render")
    with pytest.raises(ReferenceMatchContractError, match="hash mismatch"):
        commit_shared_filmfx_staging_v1(
            plan=plan,
            verification=verification,
            output_paths=outputs,
            report_path=report,
            seed=7,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)


def test_destinations_cannot_overwrite_p50_artifacts(
    tmp_path: Path,
) -> None:
    sources, verification, plan = _setup(tmp_path)
    _outputs, report = _destinations(tmp_path / "render")
    with pytest.raises(
        ReferenceMatchContractError,
        match="must not overwrite P50",
    ):
        commit_shared_filmfx_staging_v1(
            plan=plan,
            verification=verification,
            output_paths=sources,
            report_path=report,
            seed=7,
        )
    assert not report.exists()


def test_foreign_plan_and_verification_fail_cross_binding(
    tmp_path: Path,
) -> None:
    _sources_a, verification_a, plan_a = _setup(tmp_path / "a")
    _sources_b, verification_b, _plan_b = _setup(tmp_path / "b")
    outputs, report = _destinations(tmp_path / "render")
    with pytest.raises(
        ReferenceMatchContractError,
        match="does not bind staging verification",
    ):
        commit_shared_filmfx_staging_v1(
            plan=plan_a,
            verification=verification_b,
            output_paths=outputs,
            report_path=report,
            seed=7,
        )
    assert verification_a.verification_id != verification_b.verification_id


def test_commit_failure_restores_all_previous_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sources, verification, plan = _setup(tmp_path)
    outputs, report = _destinations(tmp_path / "render")
    previous = {
        outputs[0]: b"old-one",
        outputs[1]: b"old-two",
        report: b"old-report",
    }
    for path, payload in previous.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    original_replace = file_module._replace

    def fail_report(source: Path, destination: Path) -> None:
        if (
            destination == report
            and "reference-match-stage" in source.name
        ):
            raise OSError("injected shared FilmFX commit failure")
        original_replace(source, destination)

    monkeypatch.setattr(file_module, "_replace", fail_report)
    with pytest.raises(
        OSError,
        match="injected shared FilmFX commit failure",
    ):
        commit_shared_filmfx_staging_v1(
            plan=plan,
            verification=verification,
            output_paths=outputs,
            report_path=report,
            seed=7,
        )
    for path, payload in previous.items():
        assert path.read_bytes() == payload
    _assert_no_debris(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, state="delivered"),
            "state is invalid",
        ),
        (
            lambda value: replace(value, claim_ceiling="applied"),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(value, operator_id="0" * 64),
            "run identity mismatch",
        ),
        (
            lambda value: replace(
                value,
                outputs=tuple(reversed(value.outputs)),
            ),
            "source order is invalid",
        ),
        (
            lambda value: replace(
                value,
                outputs=(
                    replace(value.outputs[0], grain_seed=8),
                    value.outputs[1],
                ),
            ),
            "grain seed mismatch",
        ),
    ],
)
def test_mutations_fail_closed(tmp_path: Path, mutation, message: str) -> None:
    _sources, verification, plan = _setup(tmp_path)
    outputs, report = _destinations(tmp_path / "render")
    committed = commit_shared_filmfx_staging_v1(
        plan=plan,
        verification=verification,
        output_paths=outputs,
        report_path=report,
        seed=7,
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_shared_filmfx_run_v1(mutation(committed.run))


def test_unknown_json_fields_fail_closed(tmp_path: Path) -> None:
    _sources, verification, plan = _setup(tmp_path)
    outputs, report = _destinations(tmp_path / "render")
    run = commit_shared_filmfx_staging_v1(
        plan=plan,
        verification=verification,
        output_paths=outputs,
        report_path=report,
        seed=7,
    ).run
    payload = run.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        shared_filmfx_run_from_json(json.dumps(payload))
    payload = run.to_dict()
    payload["outputs"][0]["unexpected"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        shared_filmfx_run_from_json(json.dumps(payload))
