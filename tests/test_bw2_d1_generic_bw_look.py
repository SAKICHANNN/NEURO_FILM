from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline_color_baseline import load_guardrail_config
from src.inference import (
    list_generic_bw_looks,
    load_render_profile,
    render_generic_bw_look_rgb,
    render_resolved_safe_lab_rgb,
    resolve_generic_bw_look_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/bw2_d1_generic_bw_look_v1.json"
EVIDENCE = ROOT / "docs/evidence/BW2_D0_HP5_TRIX_K1_BASELINE_RESULT.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
STATS = ROOT / "configs/film_color_stats.json"
GUARDS = ROOT / "configs/color_guardrails.json"


@pytest.fixture(scope="module")
def source() -> np.ndarray:
    y, x = np.mgrid[:67, :91]
    return np.ascontiguousarray(
        np.stack(
            (
                ((x * 13 + y * 7) % 251) / 250.0,
                ((x * 3 + y * 17 + 19) % 251) / 250.0,
                ((x * 11 + y * 5 + 43) % 251) / 250.0,
            ),
            axis=-1,
        ).astype(np.float32)
    )


def _inputs():
    profile = load_render_profile(PROFILE, root=ROOT)
    statistics = json.loads(STATS.read_text(encoding="utf-8"))["styles"]["hp5"]
    guardrails = load_guardrail_config(GUARDS, "hp5")
    return profile, statistics, guardrails


def test_contract_binds_closed_bw2_evidence_and_preserves_legacy_profile() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == contract[
        "input_evidence"
    ]["sha256"]
    assert evidence["status"] == contract["input_evidence"]["required_status"]
    assert contract["product_look"]["look_id"] == "generic_bw"
    profile = load_render_profile(PROFILE, root=ROOT)
    assert {"hp5", "tri_x_400"}.issubset(profile["style_parameters"])


def test_catalog_exposes_one_generic_noncalibrated_bw_look() -> None:
    rows = list_generic_bw_looks()
    assert len(rows) == 1
    assert rows[0]["look_id"] == "generic_bw"
    assert rows[0]["film_stock_id"] == "generic_black_and_white"
    claim = rows[0]["claim_ceiling"]
    assert "not an HP5, Tri-X" in claim
    rows[0]["look_id"] = "forged"
    assert list_generic_bw_looks()[0]["look_id"] == "generic_bw"


def test_zero_identity_and_one_exact_legacy_execution(source: np.ndarray) -> None:
    profile, statistics, guardrails = _inputs()
    zero = render_generic_bw_look_rgb(
        source,
        profile=profile,
        look_amount=0.0,
        style_statistics=statistics,
        guardrails=guardrails,
        seed=31,
    )
    one = render_generic_bw_look_rgb(
        source,
        profile=profile,
        look_amount=1.0,
        style_statistics=statistics,
        guardrails=guardrails,
        seed=31,
    )
    expected = render_resolved_safe_lab_rgb(
        source,
        style="hp5",
        style_statistics=statistics,
        style_parameters=profile["style_parameters"]["hp5"],
        guardrails=guardrails,
        seed=31,
    )
    np.testing.assert_array_equal(zero, source)
    np.testing.assert_array_equal(one, expected)
    assert np.array_equal(one[..., 0], one[..., 1])
    assert np.array_equal(one[..., 1], one[..., 2])


def test_intermediate_full_and_tiled_are_exact(source: np.ndarray) -> None:
    profile, statistics, guardrails = _inputs()
    kwargs = {
        "profile": profile,
        "look_amount": 0.5,
        "style_statistics": statistics,
        "guardrails": guardrails,
        "seed": 31,
    }
    full = render_generic_bw_look_rgb(source, **kwargs)
    tiled = render_generic_bw_look_rgb(
        source, **kwargs, tile_size=29, tile_workers=2
    )
    np.testing.assert_array_equal(tiled, full)
    assert np.isfinite(full).all()
    assert float(full.min()) > 0.0
    assert float(full.max()) < 1.0


@pytest.mark.parametrize("amount", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_invalid_amount_fails_closed(amount: object) -> None:
    profile = load_render_profile(PROFILE, root=ROOT)
    with pytest.raises(ValueError, match="look_amount"):
        resolve_generic_bw_look_parameters(profile, look_amount=amount)  # type: ignore[arg-type]


def test_missing_legacy_execution_style_fails_closed() -> None:
    profile = load_render_profile(PROFILE, root=ROOT)
    drifted = copy.deepcopy(profile)
    del drifted["style_parameters"]["hp5"]
    with pytest.raises(ValueError, match="absent"):
        resolve_generic_bw_look_parameters(drifted, look_amount=1.0)
