from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import src.eval.fixed_global_policy_confirmation as confirmation
from src.film_physics.display_look import (
    build_source_context_display_look_stages,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bh1_fixed_global_policy_confirmation_v1.json"
OBSERVATIONS = (
    ROOT
    / "configs/u5_r2bh1_fixed_global_policy_confirmation_observations_v1.json"
)
RUNNER = ROOT / "scripts/run_u5_r2bh1_fixed_global_policy_confirmation.py"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_bh1_contract_recovers_exact_population_and_two_fixed_arms() -> None:
    validated = confirmation.validate_contract(ROOT, _config())
    assert len(validated["eligible_ids"]) == 12
    assert (
        len(
            {
                validated["source_rows"][key]["make"]
                for key in validated["eligible_ids"]
            }
        )
        == 12
    )
    assert confirmation.ARMS == (
        "fixed_b0",
        "fixed_ao6_colour_only_t15_c35",
    )
    protocol = _config()["blind_protocol"]
    assert protocol["minimum_b0_round_wins"] == 2
    assert protocol["minimum_b0_aggregate_choices"] == 22
    assert protocol["aggregate_choice_denominator"] == 36
    assert not _config()["training_allowed"]
    assert not _config()["operator_fitting_allowed"]
    assert not _config()["selector_training_allowed"]


def test_bh1_runner_freezes_libraw_openmp_before_project_imports() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    freeze = 'os.environ["OMP_NUM_THREADS"] = "1"'
    assert freeze in source
    assert source.index(freeze) < source.index(
        "from src.eval.fixed_global_policy_confirmation"
    )


def test_bh1_pair_matches_full_source_context_stages() -> None:
    validated = confirmation.validate_contract(ROOT, _config())
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=validated["compiler_config"]
    )
    source = np.linspace(
        0.0, 1.0, num=257 * 9 * 3, dtype=np.float32
    ).reshape(257, 9, 3)
    outputs = confirmation.render_fixed_pair(
        source, artifact, validated["component"]
    )
    encoded = confirmation.linear_srgb_to_encoded(
        source.astype(np.float64)
    ).astype(np.float32)
    base_stage, residual_stage = build_source_context_display_look_stages(
        artifact["component_payloads"][validated["component"]], encoded
    )
    expected_base = np.asarray(base_stage(encoded), dtype=np.float32)
    expected_ao6 = np.asarray(
        residual_stage(expected_base), dtype=np.float32
    )
    assert np.array_equal(outputs[confirmation.ARMS[0]], expected_base)
    assert np.array_equal(outputs[confirmation.ARMS[1]], expected_ao6)
    assert all(value.dtype == np.float32 for value in outputs.values())
    assert all(np.isfinite(value).all() for value in outputs.values())
    assert all(
        np.all((value >= 0.0) & (value <= 1.0))
        for value in outputs.values()
    )


def test_bh1_blind_round_is_complete_and_changes_mapping(
    tmp_path: Path,
) -> None:
    source_rows = {}
    eligible_ids = []
    render_dir = tmp_path / "render"
    for index in range(12):
        source_id = f"s{index:02d}"
        eligible_ids.append(source_id)
        source = tmp_path / "source" / f"{source_id}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (18, 12), (index * 10, 40, 80)).save(source)
        source_rows[source_id] = {
            "decoded_path": source.relative_to(tmp_path).as_posix()
        }
        for arm_index, arm_id in enumerate(confirmation.ARMS):
            path = render_dir / "renders" / arm_id / f"{source_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (18, 12),
                (index * 10, arm_index * 60, 90),
            ).save(path)
    first = confirmation.build_blind_round(
        root=tmp_path,
        render_dir=render_dir,
        source_rows=source_rows,
        eligible_ids=eligible_ids,
        round_index=1,
        output_dir=tmp_path / "blind",
    )
    second = confirmation.build_blind_round(
        root=tmp_path,
        render_dir=render_dir,
        source_rows=source_rows,
        eligible_ids=eligible_ids,
        round_index=2,
        output_dir=tmp_path / "blind",
    )
    mapping_a = json.loads(first["mapping_path"].read_text())
    mapping_b = json.loads(second["mapping_path"].read_text())
    assert len(mapping_a) == len(mapping_b) == 12
    assert all(
        {row["A"], row["B"]} == set(confirmation.ARMS)
        for row in mapping_a + mapping_b
    )
    assert mapping_a != mapping_b
    assert len(first["parts"]) == len(second["parts"]) == 3


def test_bh1_blind_choices_are_complete_and_mapping_sealed() -> None:
    observations = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    eligible = set(confirmation.validate_contract(ROOT, _config())["eligible_ids"])
    assert observations["status"] == (
        "blind_choices_frozen_before_mapping_reveal"
    )
    assert not observations["mapping_files_read"]
    assert observations["repeat_exact_file_count"] == 37
    assert observations["repeat_mismatch_count"] == 0
    assert len(observations["blind_sheets"]) == 9
    assert len(observations["rounds"]) == 3
    for round_row in observations["rounds"]:
        assert set(round_row["choices"]) == eligible
        assert set(round_row["choices"].values()).issubset({"A", "B"})
    assert observations["blind_sheet_severe_review"][
        "confirmed_severe_count"
    ] == 0
    assert observations["full_resolution_severe_review_status"] == (
        "pending_after_blind_choice_freeze"
    )
