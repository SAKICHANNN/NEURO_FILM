from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import src.eval.fixed_bank_complete_oracle as oracle
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)
from src.film_physics.display_look import (
    build_source_context_display_look_stages,
)


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
    candidates = validated["source_contract"]["candidates"]
    assert all(
        Path(row["path"]).suffix.casefold()
        in {
            ".arw",
            ".rw2",
            ".cr3",
            ".nef",
            ".dng",
            ".ori",
            ".mrw",
            ".iiq",
            ".gpr",
            ".raf",
            ".srw",
        }
        for row in candidates
    )


def test_bh0_renderer_adds_two_fixed_residual_arms_without_refit() -> None:
    validated = oracle.validate_contract(ROOT, _config())
    scene = np.full((257, 11, 3), 0.18, dtype=np.float32)
    native = np.full_like(scene, 0.51)
    build_config = validated["build_config"]
    artifact = compile_standalone_profile_artifact(
        root=ROOT,
        config=json.loads(
            (ROOT / build_config["profile_compiler_config"]).read_text()
        ),
    )

    class FakeRuntime:
        def render_to_sink(self, source, *, output_sink):
            output_sink(0, source.shape[0], native)
            return {
                "receipt_sha256": "a" * 64,
                "output": {
                    "array_sha256": oracle.hashlib.sha256(
                        native.tobytes()
                    ).hexdigest(),
                },
            }

    outputs, diagnostics = oracle.render_fixed_bank(
        scene_linear=scene,
        artifact=artifact,
        runtime=FakeRuntime(),
        ap3_operator=validated["ap3_operator"],
        ao6_operator=validated["ao6_operator"],
        ap3_config=validated["ap3_config"],
        az0_candidate=validated["az0_candidate"],
    )
    base = outputs[oracle.ARMS[0]]
    assert not np.array_equal(
        outputs[oracle.ARMS[0]],
        outputs[oracle.ARMS[1]],
    )
    assert np.array_equal(outputs[oracle.ARMS[4]], native)
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
    assert diagnostics["residual_row_chunk"] == 128
    encoded = oracle.linear_srgb_to_encoded(
        scene.astype(np.float64)
    ).astype(np.float32)
    payload = artifact["component_payloads"][
        "ao6-source-context-display-look"
    ]
    full_base, full_residual = build_source_context_display_look_stages(
        payload, encoded
    )
    expected_base = np.asarray(full_base(encoded), dtype=np.float32)
    expected_ao6 = np.asarray(
        full_residual(expected_base), dtype=np.float32
    )
    assert np.array_equal(outputs[oracle.ARMS[0]], expected_base)
    assert np.array_equal(outputs[oracle.ARMS[1]], expected_ao6)

    base_linear = oracle.encoded_srgb_to_linear(base.astype(np.float64))
    ap3_spec = validated["ap3_config"]["fixed_candidate"]
    ap3_controls = ap3_spec["factorization"]
    full_ap3 = oracle.apply_factorized_boundary_guard(
        validated["ap3_operator"],
        base_linear,
        tone_strength=float(ap3_spec["tone_strength"]),
        chroma_strength=float(ap3_spec["chroma_strength"]),
        luma_weights=np.asarray(ap3_controls["luma_weights"]),
        hard_boundary_epsilon_encoded_srgb=float(
            ap3_controls["hard_boundary_epsilon_encoded_srgb"]
        ),
        guard_boundary_epsilon_encoded_srgb=float(
            ap3_controls["guard_boundary_epsilon_encoded_srgb"]
        ),
    )
    full_az0 = oracle.apply_density_residual_guard(
        validated["ao6_operator"],
        base_linear,
        neutral_strength=float(validated["az0_candidate"]["neutral_strength"]),
        opponent_strength=float(validated["az0_candidate"]["opponent_strength"]),
        neutral_weights=np.asarray(validated["az0_candidate"]["neutral_weights"]),
        density_floor=float(validated["az0_candidate"]["density_floor"]),
        hard_boundary_epsilon_encoded_srgb=float(
            validated["az0_candidate"][
                "hard_boundary_epsilon_encoded_srgb"
            ]
        ),
        guard_boundary_epsilon_encoded_srgb=float(
            validated["az0_candidate"][
                "guard_boundary_epsilon_encoded_srgb"
            ]
        ),
    )
    assert np.array_equal(
        outputs[oracle.ARMS[2]],
        oracle.linear_srgb_to_encoded(full_ap3.output).astype(np.float32),
    )
    assert np.array_equal(
        outputs[oracle.ARMS[3]],
        oracle.linear_srgb_to_encoded(full_az0.output).astype(np.float32),
    )


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
