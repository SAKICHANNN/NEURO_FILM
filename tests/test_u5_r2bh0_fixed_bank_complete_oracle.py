from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import src.eval.fixed_bank_complete_oracle as oracle


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bh0_fixed_bank_complete_oracle_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_bh0_validator_recovers_exact_population_and_operators() -> None:
    validated = oracle.validate_contract(ROOT, _config())
    assert len(validated["eligible_ids"]) == 10
    assert len({validated["source_rows"][key]["make"] for key in validated["eligible_ids"]}) == 10
    assert validated["ao6_operator"] is not None
    assert validated["ap3_operator"] is not None
    assert validated["az0_candidate"]["neutral_strength"] == 0.15


def test_bh0_renderer_adds_two_fixed_residual_arms_without_refit(
    monkeypatch,
) -> None:
    validated = oracle.validate_contract(ROOT, _config())
    scene = np.full((9, 11, 3), 0.18, dtype=np.float32)
    base = np.full_like(scene, 0.42)
    ao6 = np.full_like(scene, 0.48)
    native = np.full_like(scene, 0.51)

    def fake_render_fixed_arms(**_: object):
        return (
            {
                oracle.ARMS[0]: base,
                oracle.ARMS[1]: ao6,
                oracle.ARMS[2]: native,
            },
            {
                "receipt_sha256": "a" * 64,
                "output": {
                    "array_sha256": "b" * 64,
                },
            },
        )

    monkeypatch.setattr(
        oracle, "render_fixed_arms", fake_render_fixed_arms
    )
    outputs, diagnostics = oracle.render_fixed_bank(
        scene_linear=scene,
        artifact={},
        runtime=object(),
        ap3_operator=validated["ap3_operator"],
        ao6_operator=validated["ao6_operator"],
        ap3_config=validated["ap3_config"],
        az0_candidate=validated["az0_candidate"],
    )
    assert tuple(outputs) == oracle.ARMS
    assert all(value.shape == scene.shape for value in outputs.values())
    assert all(value.dtype == np.float32 for value in outputs.values())
    assert all(np.isfinite(value).all() for value in outputs.values())
    assert all(
        np.all((value >= 0.0) & (value <= 1.0))
        for value in outputs.values()
    )
    assert not np.array_equal(outputs[oracle.ARMS[2]], base)
    assert not np.array_equal(outputs[oracle.ARMS[3]], base)
    assert diagnostics["native_receipt_sha256"] == "a" * 64


def test_bh0_blind_round_is_complete_and_mapping_changes(
    tmp_path: Path,
) -> None:
    source_rows = {}
    eligible_ids = []
    render_dir = tmp_path / "render"
    for index in range(10):
        source_id = f"s{index:02d}"
        eligible_ids.append(source_id)
        source = tmp_path / "source" / f"{source_id}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (18, 12), (index * 10, 40, 80)).save(source)
        source_rows[source_id] = {
            "decoded_path": source.relative_to(tmp_path).as_posix()
        }
        for arm_index, arm_id in enumerate(oracle.ARMS):
            path = render_dir / "renders" / arm_id / f"{source_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (18, 12),
                (index * 10, arm_index * 30, 90),
            ).save(path)
    first = oracle.build_blind_round(
        root=tmp_path,
        render_dir=render_dir,
        source_rows=source_rows,
        eligible_ids=eligible_ids,
        round_index=1,
        output_dir=tmp_path / "blind",
    )
    second = oracle.build_blind_round(
        root=tmp_path,
        render_dir=render_dir,
        source_rows=source_rows,
        eligible_ids=eligible_ids,
        round_index=2,
        output_dir=tmp_path / "blind",
    )
    mapping_a = json.loads(first["mapping_path"].read_text())
    mapping_b = json.loads(second["mapping_path"].read_text())
    assert len(mapping_a) == len(mapping_b) == 10
    assert all(
        set(row[key] for key in "ABCDE") == set(oracle.ARMS)
        for row in mapping_a + mapping_b
    )
    assert mapping_a != mapping_b
    assert len(first["parts"]) == len(second["parts"]) == 2
