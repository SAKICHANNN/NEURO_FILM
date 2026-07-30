from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.ao6_density_grain_value import (
    AO6DensityGrainError,
    apply_linear_density_grain,
    build_blind_crop_sheets,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u5_r2bc1_ao6_density_grain_value_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_density_grain_preserves_linear_chromaticity_and_cube() -> None:
    y, x = np.mgrid[0:64, 0:96]
    base = np.stack(
        [
            0.05 + 0.9 * x / 95,
            0.08 + 0.7 * y / 63,
            np.full_like(x, 0.35, dtype=np.float64),
        ],
        axis=2,
    ).astype(np.float32)
    output, metrics = apply_linear_density_grain(
        base, density_sigma=0.0225, seed=7
    )
    assert output.shape == base.shape
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    assert metrics["new_raw_clipping_fraction"] == 0.0
    assert metrics["linear_chromaticity_drift_p999"] < 2e-6
    assert metrics["changed_pixel_fraction"] > 0.5


def test_density_grain_is_exact_for_same_seed_and_changes_for_new_seed() -> None:
    base = np.full((31, 47, 3), [0.2, 0.4, 0.7], dtype=np.float32)
    first, first_metrics = apply_linear_density_grain(
        base, density_sigma=0.0225, seed=11
    )
    second, second_metrics = apply_linear_density_grain(
        base, density_sigma=0.0225, seed=11
    )
    third, _ = apply_linear_density_grain(
        base, density_sigma=0.0225, seed=12
    )
    assert np.array_equal(first, second)
    assert first_metrics == second_metrics
    assert not np.array_equal(first, third)


def test_density_grain_rejects_invalid_input() -> None:
    with pytest.raises(AO6DensityGrainError, match="invalid"):
        apply_linear_density_grain(
            np.full((4, 4, 3), np.nan, dtype=np.float32),
            density_sigma=0.0225,
            seed=1,
        )


def test_frozen_contract_binds_parent_and_population() -> None:
    rows = validate_contract(ROOT, _config())
    assert len(rows) == 16


def test_blind_crop_sheets_are_mapping_consistent(tmp_path: Path) -> None:
    ids = [f"sample_{index}" for index in range(9)]
    arms = ["base", "grain", "full"]
    base_dir = tmp_path / "base"
    run_dir = tmp_path / "run"
    blind_dir = run_dir / "blind"
    base_dir.mkdir()
    records = []
    for sample_index, sample_id in enumerate(ids):
        image = np.full(
            (96, 128, 3), 32 + sample_index, dtype=np.uint8
        )
        Image.fromarray(image).save(base_dir / f"{sample_id}.png")
        for arm_index, arm_id in enumerate(arms[1:], start=1):
            relative = Path(arm_id) / f"{sample_id}.png"
            (run_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image + arm_index).save(run_dir / relative)
            records.append(
                {
                    "sample_id": sample_id,
                    "arm_id": arm_id,
                    "output_path": relative.as_posix(),
                }
            )
    config = {
        "population": {
            "expected_ids": ids,
            "base_directory": base_dir.relative_to(tmp_path).as_posix(),
        },
        "arms": arms,
    }
    result = build_blind_crop_sheets(
        root=tmp_path,
        config=config,
        report={"records": records},
        output_dir=blind_dir,
        crop_size=64,
    )

    assert result["crop_size"] == 64
    assert result["mappings"] == [
        {"A": "base", "B": "grain", "C": "full"},
        {"A": "full", "B": "base", "C": "grain"},
        {"A": "grain", "B": "full", "C": "base"},
    ]
    for round_index in range(1, 4):
        with Image.open(
            blind_dir / f"blind_round_{round_index}_crops.png"
        ) as sheet:
            assert sheet.size == ((64 * 2 + 28) * 3, (64 + 34) * 9)


def test_blind_crop_sheets_accept_per_sample_orders(tmp_path: Path) -> None:
    ids = [f"sample_{index}" for index in range(9)]
    arms = ["base", "grain", "full"]
    base_dir = tmp_path / "base"
    run_dir = tmp_path / "run"
    base_dir.mkdir()
    records = []
    for sample_index, sample_id in enumerate(ids):
        image = np.full((96, 128, 3), 32 + sample_index, dtype=np.uint8)
        Image.fromarray(image).save(base_dir / f"{sample_id}.png")
        for arm_index, arm_id in enumerate(arms[1:], start=1):
            relative = Path(arm_id) / f"{sample_id}.png"
            (run_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image + arm_index).save(run_dir / relative)
            records.append(
                {
                    "sample_id": sample_id,
                    "arm_id": arm_id,
                    "output_path": relative.as_posix(),
                }
            )
    config = {
        "population": {
            "expected_ids": ids,
            "base_directory": base_dir.relative_to(tmp_path).as_posix(),
        },
        "arms": arms,
    }
    orders = [
        {
            sample_id: [
                arms[(sample_index + round_index + offset) % 3]
                for offset in range(3)
            ]
            for sample_index, sample_id in enumerate(ids)
        }
        for round_index in range(3)
    ]
    result = build_blind_crop_sheets(
        root=tmp_path,
        config=config,
        report={"records": records},
        output_dir=run_dir / "randomized_blind",
        crop_size=64,
        sample_orders=orders,
    )

    assert result["mappings"][0]["sample_0"] == {
        "A": "base",
        "B": "grain",
        "C": "full",
    }
    assert result["mappings"][1]["sample_0"] == {
        "A": "grain",
        "B": "full",
        "C": "base",
    }
