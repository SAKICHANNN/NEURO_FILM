from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.fresh_native_standard_confirmation import (
    ARMS,
    boundary_metrics,
    load_confirmation_working_image,
    run_comparison,
    validate_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u6_p8bp_fresh_native_standard_confirmation_v1.json"
)


def test_p8bp_preflight_is_hash_bound_and_comparison_is_fixed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = validate_preflight(ROOT, config)
    assert len(manifest) == 9
    assert tuple(config["comparison"]["arms"]) == ARMS
    assert config["comparison"]["automatic_gate"][
        "maximum_new_boundary_fraction_vs_ao6"
    ] == 0.0


def test_boundary_metric_counts_only_new_candidate_boundaries() -> None:
    reference = np.full((2, 2, 3), 0.5, dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0, 0] = 0.0
    metrics = boundary_metrics(candidate, reference)
    assert metrics["output_code_boundary_fraction"] == 0.25
    assert metrics["new_boundary_fraction_vs_ao6"] == 0.25
    reference[0, 0, 1] = 1.0
    metrics = boundary_metrics(candidate, reference)
    assert metrics["new_boundary_fraction_vs_ao6"] == 0.0


def test_ori_research_adapter_uses_existing_raw_decoder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "companion.ORI"
    path.write_bytes(b"frozen-fixture")
    sentinel = object()
    monkeypatch.setattr(
        "src.eval.fresh_native_standard_confirmation."
        "load_raw_working_image",
        lambda candidate: sentinel if candidate == path else None,
    )
    assert load_confirmation_working_image(path) is sentinel


def test_comparison_report_paths_are_run_directory_independent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    output_dir = root / "outputs" / "experiment" / "run_c"
    raw_path = root / "data" / "source.dng"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"raw")
    manifest = [{"id": "source"}]
    config = {
        "candidates": [
            {
                "id": "source",
                "path": raw_path.relative_to(root).as_posix(),
                "sha256": (
                    "d7439bee24773b5e311d800e0685f82f"
                    "8ef46c09ee4e1b308f1b0c5a03f7e428"
                ),
            }
        ],
        "comparison": {
            "automatic_gate": {
                "maximum_output_code_boundary_fraction": 1.0,
                "maximum_new_boundary_fraction_vs_ao6": 1.0,
            }
        },
        "claim_ceiling": "test",
    }
    monkeypatch.setattr(
        "src.eval.fresh_native_standard_confirmation.validate_preflight",
        lambda _root, _config: manifest,
    )
    monkeypatch.setattr(
        "src.eval.fresh_native_standard_confirmation.sha256_file",
        lambda path: config["candidates"][0]["sha256"]
        if path == raw_path
        else "0" * 64,
    )
    working = type(
        "Working",
        (),
        {
            "pixels": np.full((1, 1, 3), 0.5, dtype=np.float32),
            "transfer_state": "scene_linear",
            "working_space": "linear_srgb",
            "orientation_applied": True,
        },
    )()
    monkeypatch.setattr(
        "src.eval.fresh_native_standard_confirmation."
        "load_confirmation_working_image",
        lambda _path: working,
    )
    monkeypatch.setattr(
        "src.eval.fresh_native_standard_confirmation."
        "compile_standalone_profile_artifact",
        lambda **_kwargs: {
            "bundle_sha256": "1" * 64,
            "component_payloads": {},
        },
    )
    monkeypatch.setattr(
        "src.eval.fresh_native_standard_confirmation."
        "create_opt_in_native_standard_runtime",
        lambda **_kwargs: (object(), {"receipt_sha256": "2" * 64}),
    )
    outputs = {
        arm: np.full((1, 1, 3), 0.5, dtype=np.float32)
        for arm in ARMS
    }
    monkeypatch.setattr(
        "src.eval.fresh_native_standard_confirmation.render_fixed_arms",
        lambda **_kwargs: (
            outputs,
            {
                "receipt_sha256": "3" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        "src.eval.fresh_native_standard_confirmation.save_srgb16_png",
        lambda _pixels, path: (
            path.parent.mkdir(parents=True, exist_ok=True),
            path.write_bytes(b"png"),
        ),
    )
    monkeypatch.setattr(
        "src.eval.fresh_native_standard_confirmation.cv2.imread",
        lambda *_args, **_kwargs: np.zeros((1, 1, 3), dtype=np.uint16),
    )
    package_path = root / "package.json"
    compiler_path = root / "compiler.json"
    package_path.write_text("{}", encoding="utf-8")
    compiler_path.write_text("{}", encoding="utf-8")
    build_config = {
        "package": package_path.relative_to(root).as_posix(),
        "profile_compiler_config": compiler_path.relative_to(root).as_posix(),
    }

    report = run_comparison(
        root=root,
        config=config,
        output_dir=output_dir,
        build_components=lambda _config, _path: {},
        build_config=build_config,
    )

    assert {row["output_path"] for row in report["rows"]} == {
        f"renders/{arm}/source.png" for arm in ARMS
    }
