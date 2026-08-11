from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import absolute_cinema_archival_grain as p4cj

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/u6_p4cj_absolute_cinema_archival_grain_v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_and_parent_are_exact() -> None:
    contract = _contract()
    p4cj._validate_contract(contract)
    parent = p4cj._load_parent(ROOT, contract)
    assert parent["decision"] == contract["parent"]["required_decision"]


def test_phase_scramble_preserves_power_and_is_deterministic() -> None:
    yy, xx = np.indices((96, 96), dtype=np.float64)
    values = np.sin(xx * 0.31) + 0.7 * np.cos(yy * 0.17) + 0.1 * np.sin((xx + yy) * 0.43)
    values -= values.mean()
    values /= np.sqrt(np.mean(np.square(values)))
    first = p4cj._phase_scramble(values, seed=7, patch_index=3, scramble_index=11)
    second = p4cj._phase_scramble(values, seed=7, patch_index=3, scramble_index=11)
    assert np.array_equal(first, second)
    assert np.allclose(np.abs(np.fft.rfft2(first)), np.abs(np.fft.rfft2(values)), atol=1e-10)


def test_model_vectors_are_finite_and_distinct() -> None:
    contract = _contract()
    thomas_signature, thomas_acf = p4cj._model_vectors(contract, thomas=True)
    gaussian_signature, gaussian_acf = p4cj._model_vectors(contract, thomas=False)
    assert np.all(np.isfinite(thomas_signature))
    assert np.all(np.isfinite(thomas_acf))
    assert not np.array_equal(thomas_signature, gaussian_signature)
    assert not np.array_equal(thomas_acf, gaussian_acf)


def test_decode_rejects_nonopaque_rgba(tmp_path: Path) -> None:
    from PIL import Image

    values = np.zeros((8, 8, 4), dtype=np.uint8)
    values[..., 3] = 254
    path = tmp_path / "bad.png"
    Image.fromarray(values, mode="RGBA").save(path)
    with pytest.raises(p4cj.AbsoluteCinemaArchivalGrainError, match="not opaque"):
        p4cj._decode_scalar(path, _contract())


def test_patch_selection_is_deterministic() -> None:
    yy, xx = np.indices((480, 480), dtype=np.float64)
    image = 0.4 + 0.03 * np.sin(xx * 0.11) + 0.02 * np.cos(yy * 0.09)
    first = p4cj._select_patches(image, _contract())
    second = p4cj._select_patches(image, _contract())
    assert len(first) == 8
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
