from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_zero_dc_dog_copula_structure import (
    ZeroDcDogCopulaError,
    evaluate_structure,
    load_contract,
)
from src.film_physics.structure_compiler import (
    zero_dc_dog_kernel_metrics,
    zero_dc_dog_normal_region,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4ao_zero_dc_dog_copula_structure_v1.json"


def test_zero_dc_kernel_and_partition_identity() -> None:
    dc_sum, variance = zero_dc_dog_kernel_metrics(0.65, 1.2)
    assert abs(dc_sum) <= 1e-15
    assert variance > 0.0
    full = zero_dc_dog_normal_region(
        (97, 131),
        origin_yx=(0, 0),
        shape=(97, 131),
        narrow_sigma=0.65,
        broad_sigma=1.2,
        seed=260831,
    )
    assembled = np.empty_like(full)
    for y0 in range(0, 97, 17):
        part = zero_dc_dog_normal_region(
            (97, 131),
            origin_yx=(y0, 0),
            shape=(min(17, 97 - y0), 131),
            narrow_sigma=0.65,
            broad_sigma=1.2,
            seed=260831,
        )
        assembled[y0 : y0 + part.shape[0]] = part
    assert np.array_equal(full, assembled)


def test_invalid_dog_parameters_fail() -> None:
    with pytest.raises(ValueError):
        zero_dc_dog_kernel_metrics(1.2, 0.65)
    with pytest.raises(ValueError):
        zero_dc_dog_kernel_metrics(0.65, 1.2, broad_weight=0.0)


def test_contract_rejects_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["model"]["broad_sigma_pixels"] = 1.21
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ZeroDcDogCopulaError):
        load_contract(path)


def test_formal_evaluation_is_stable() -> None:
    contract = load_contract(CONFIG)
    first = evaluate_structure(contract, ROOT)
    second = evaluate_structure(contract, ROOT)
    assert first == second
    assert first["checks"]["kernel_dc"]
    assert first["checks"]["repeat_exact"]
    assert first["checks"]["row_partition_exact"]
