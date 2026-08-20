from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.canoncgt_fixed_atlas_shared_lut import (
    aggregate_atlas_luts,
    apply_lut_bank,
    make_lattice_atlas,
)
from src.eval.global_frontier import sha256_file


def test_lattice_atlas_is_deterministic_and_covers_all_nodes() -> None:
    first = make_lattice_atlas(
        atlas_id="rgb", size=224, bins=17, multiplier=1723
    )
    second = make_lattice_atlas(
        atlas_id="rgb", size=224, bins=17, multiplier=1723
    )
    assert first.shape == (224, 224, 3)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert len(np.unique(first.reshape(-1, 3), axis=0)) == 17**3


def test_channel_permutations_preserve_lattice_population() -> None:
    rgb = make_lattice_atlas(
        atlas_id="rgb", size=224, bins=17, multiplier=1723
    )
    gbr = make_lattice_atlas(
        atlas_id="gbr", size=224, bins=17, multiplier=1723
    )
    assert np.array_equal(rgb[..., [1, 2, 0]], gbr)


def test_aggregation_is_key_order_invariant_and_float32() -> None:
    base = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    forward, forward_metrics = aggregate_atlas_luts(
        {"rgb": base, "gbr": base + 1, "brg": base - 1}
    )
    reverse, reverse_metrics = aggregate_atlas_luts(
        {"brg": base - 1, "gbr": base + 1, "rgb": base}
    )
    assert forward.dtype == np.float32
    assert np.array_equal(forward, reverse)
    assert forward_metrics == reverse_metrics
    assert np.array_equal(forward, base)


def test_runner_requires_explicit_output_root_and_has_three_phases() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_u5_r2repid6_canoncgt_fixed_atlas_shared_lut.py"
    ).read_text(encoding="utf-8")
    assert 'choices=("build", "apply", "evaluate")' in source
    assert 'parser.add_argument("--output-root", type=Path, required=True)' in source
    assert "D:\\" not in source
    assert "P:\\" not in source


def test_build_path_does_not_call_gold_source_loader() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/eval/canoncgt_fixed_atlas_shared_lut.py"
    ).read_text(encoding="utf-8")
    build = source[source.index("def build_lut_bank") : source.index("def apply_lut_bank")]
    assert "_load_gold_samples" not in build
    assert '"application_source_file_reads": 0' in build


def test_apply_decodes_each_source_once_and_binds_manifest_chain(
    monkeypatch, tmp_path: Path
) -> None:
    import torch

    import src.eval.canoncgt_fixed_atlas_shared_lut as module

    class IdentityModel:
        @staticmethod
        def TrilinearInterpolation(source, lut):
            del lut
            return source

    references = [
        {"reference_id": "ref_a", "sha256": "a" * 64},
        {"reference_id": "ref_b", "sha256": "b" * 64},
    ]
    sources = {}
    for index in range(2):
        path = tmp_path / f"source_{index}.png"
        Image.fromarray(
            np.full((3, 4, 3), 32 + index, dtype=np.uint8), mode="RGB"
        ).save(path)
        sources[f"sample_{index}"] = {
            "source_path": str(path),
            "source_sha256": sha256_file(path),
        }
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    build_rows = []
    for reference in references:
        lut_path = build_dir / f"{reference['reference_id']}.npy"
        lut = np.zeros((3, 17, 17, 17), dtype=np.float32)
        np.save(lut_path, lut, allow_pickle=False)
        build_rows.append(
            {
                "reference_id": reference["reference_id"],
                "reference_sha256": reference["sha256"],
                "lut": lut_path.name,
                "lut_file_sha256": sha256_file(lut_path),
                "lut_array_sha256": module._sha256_array(lut),
            }
        )
    build_manifest = build_dir / "build_manifest.json"
    build_manifest.write_text(
        __import__("json").dumps(
            {
                "experiment_id": "test",
                "phase": "build",
                "application_source_file_reads": 0,
                "application_source_pixel_decodes": 0,
                "canonicalizer_executions": 0,
                "records": build_rows,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_load_model_reference_only",
        lambda root, config, device: (
            torch,
            IdentityModel(),
            {"references": references},
            {"device": device},
        ),
    )
    monkeypatch.setattr(module, "_load_gold_samples", lambda root, parent: sources)
    calls = []
    original = module._tensor_from_image

    def counted(path, torch_module, device):
        calls.append(path)
        return original(path, torch_module, device)

    monkeypatch.setattr(module, "_tensor_from_image", counted)
    result = apply_lut_bank(
        root=tmp_path,
        config={"experiment_id": "test", "claim_ceiling": "test-only"},
        build_manifest_path=build_manifest,
        output_dir=tmp_path / "apply",
        device="cpu",
    )
    assert len(calls) == len(sources)
    assert result["manifest"]["application_source_file_reads_after_build_freeze"] == 2
    assert result["manifest"]["application_source_pixel_decodes_after_build_freeze"] == 2
    assert len(result["manifest"]["records"]) == 4
