from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

import src.color_match.files as file_module
from src.color_match import (
    EXTERNAL_FILMFX_CLAIM_CEILING,
    ReferenceMatchContractError,
    build_external_reference_composition_v1,
    commit_external_filmfx_staging_v1,
    external_filmfx_run_from_json,
    external_filmfx_run_to_json,
    validate_external_filmfx_run_v1,
)
from src.inference import sha256_file
from tests.test_color_match_core_staging_verification import (
    _committed,
    _verify,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_filmfx_run_v1.schema.json"
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
    committed, source_paths, staging_report = _committed(
        tmp_path / "source"
    )
    verification = _verify(committed, staging_report)
    profile, profile_sha = _profile(
        tmp_path,
        model=model,
        active=active,
    )
    plan = build_external_reference_composition_v1(
        verification,
        include_film_effects=True,
        film_profile=profile,
        film_profile_sha256=profile_sha,
    )
    return source_paths, verification, plan


def _destinations(
    root: Path,
) -> tuple[tuple[Path, ...], Path]:
    return (
        (root / "fx-one.png", root / "fx-two.png"),
        root / "filmfx-report.json",
    )


def _assert_no_debris(root: Path) -> None:
    assert not list(root.rglob(".*.reference-match-stage*"))
    assert not list(root.rglob(".*.reference-match-backup"))


def test_simple_filmfx_renders_all_sources_atomically(
    tmp_path: Path,
) -> None:
    sources, verification, plan = _setup(tmp_path)
    outputs, report = _destinations(tmp_path / "render")
    committed = commit_external_filmfx_staging_v1(
        plan=plan,
        verification=verification,
        output_paths=outputs,
        report_path=report,
        seed=37,
    )
    assert committed.run.state == "filmfx-rendered-to-staging"
    assert committed.run.claim_ceiling == EXTERNAL_FILMFX_CLAIM_CEILING
    assert committed.run.composition_plan_id == plan.plan_id
    assert committed.run.staging_verification_id == verification.verification_id
    assert committed.report_file_sha256 == sha256_file(report)
    for index, (source, output, row) in enumerate(
        zip(sources, outputs, committed.run.outputs, strict=True)
    ):
        assert row.source_index == index
        assert row.input_file_sha256 == sha256_file(source)
        assert row.output_file_sha256 == sha256_file(output)
        assert row.grain_seed == 37 + index * 1009
        assert row.dust_seed == row.grain_seed + 17
        assert row.output_file_sha256 != row.input_file_sha256
    decoded = external_filmfx_run_from_json(
        report.read_text(encoding="utf-8")
    )
    assert decoded == committed.run
    assert (
        external_filmfx_run_from_json(
            external_filmfx_run_to_json(decoded)
        )
        == decoded
    )
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
    first = commit_external_filmfx_staging_v1(
        plan=plan,
        verification=verification,
        output_paths=first_outputs,
        report_path=first_report,
        seed=91,
    )
    second = commit_external_filmfx_staging_v1(
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


def test_plan_without_active_effects_and_physical_defaults_fail_closed(
    tmp_path: Path,
) -> None:
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
            commit_external_filmfx_staging_v1(
                plan=plan,
                verification=verification,
                output_paths=outputs,
                report_path=report,
                seed=7,
            )
        assert not report.exists()
        assert not any(path.exists() for path in outputs)


def test_live_p34_refresh_blocks_tampered_staging_input(
    tmp_path: Path,
) -> None:
    sources, verification, plan = _setup(tmp_path)
    sources[0].write_bytes(b"tampered")
    outputs, report = _destinations(tmp_path / "render")
    with pytest.raises(ReferenceMatchContractError, match="hash mismatch"):
        commit_external_filmfx_staging_v1(
            plan=plan,
            verification=verification,
            output_paths=outputs,
            report_path=report,
            seed=7,
        )
    assert not report.exists()
    assert not any(path.exists() for path in outputs)


def test_destinations_cannot_overwrite_p33_artifacts(
    tmp_path: Path,
) -> None:
    sources, verification, plan = _setup(tmp_path)
    _outputs, report = _destinations(tmp_path / "render")
    with pytest.raises(
        ReferenceMatchContractError,
        match="must not overwrite P33",
    ):
        commit_external_filmfx_staging_v1(
            plan=plan,
            verification=verification,
            output_paths=(sources[0], sources[1]),
            report_path=report,
            seed=7,
        )
    assert not report.exists()


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
            raise OSError("injected FilmFX commit failure")
        original_replace(source, destination)

    monkeypatch.setattr(file_module, "_replace", fail_report)
    with pytest.raises(OSError, match="injected FilmFX commit failure"):
        commit_external_filmfx_staging_v1(
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
                outputs=(
                    replace(value.outputs[0], dust_seed=999),
                    value.outputs[1],
                ),
            ),
            "dust seed mismatch",
        ),
        (
            lambda value: replace(value, claim_ceiling="delivered"),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(value, run_id="0" * 64),
            "run_id mismatch",
        ),
    ],
)
def test_run_mutations_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    _sources, verification, plan = _setup(tmp_path)
    outputs, report = _destinations(tmp_path / "render")
    committed = commit_external_filmfx_staging_v1(
        plan=plan,
        verification=verification,
        output_paths=outputs,
        report_path=report,
        seed=7,
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_external_filmfx_run_v1(mutation(committed.run))
