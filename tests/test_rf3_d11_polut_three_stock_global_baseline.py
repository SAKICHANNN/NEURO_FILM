from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.d_lut_published_assets import CubeAsset
from src.eval.polut_three_stock_global_baseline import (
    PoLUTBaselineError,
    apply_polut,
    git_blob_sha1,
    load_contract,
    trilinear_apply,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "rf3_d11_polut_three_stock_global_baseline_v1.json"


def _identity(size: int = 3) -> CubeAsset:
    axis = np.linspace(0.0, 1.0, size)
    values = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return CubeAsset(values=values, domain_min=np.zeros(3), domain_max=np.ones(3))


def test_contract_freezes_three_stocks_and_no_rescue() -> None:
    contract = load_contract(CONFIG)
    assert [row["stock_id"] for row in contract["external_baseline"]["assets"]] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert contract["external_baseline"]["full_repository_clone_allowed"] is False
    assert contract["colour_execution"]["exposure_adjustment_allowed"] is False
    assert contract["colour_execution"]["tone_or_gamut_rescue_allowed"] is False


def test_trilinear_identity_is_exact_at_nodes_and_interior() -> None:
    asset = _identity()
    probes = np.asarray(
        [[[0.0, 0.0, 0.0], [0.125, 0.625, 0.875], [1.0, 1.0, 1.0]]],
        dtype=np.float64,
    )
    assert np.max(np.abs(trilinear_apply(asset, probes) - probes)) < 1.0e-15


def test_cube_domain_escape_fails_closed() -> None:
    with pytest.raises(PoLUTBaselineError, match="declared domain"):
        trilinear_apply(_identity(), np.asarray([[[-1.0e-4, 0.5, 0.5]]]))


def test_identity_cube_roundtrips_srgb_through_adobe_rgb() -> None:
    contract = load_contract(CONFIG)
    probes = np.asarray(
        [[[0.0, 0.0, 0.0], [0.18, 0.5, 0.9], [1.0, 1.0, 1.0]]],
        dtype=np.float64,
    )
    output, gamut = apply_polut(probes, _identity(33), contract)
    assert np.max(np.abs(output - probes)) < 2.0e-6
    assert gamut["unclipped_out_of_gamut_fraction"] == 0.0


def test_contract_json_is_canonical_object() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["experiment_id"] == "RF3.D11"


def test_git_blob_identity_matches_git_object_format(tmp_path: Path) -> None:
    path = tmp_path / "asset.cube"
    path.write_bytes(b"abc")
    assert git_blob_sha1(path) == "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"
