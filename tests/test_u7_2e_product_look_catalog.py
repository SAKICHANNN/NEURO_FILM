from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline_color_baseline import load_guardrail_config
from src.inference import (
    list_product_looks,
    load_render_profile,
    render_product_look_rgb,
)
from src.inference.generic_bw_look import render_generic_bw_look_rgb
from src.inference.three_stock_look import (
    list_three_stock_looks,
    render_three_stock_look_rgb,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u7_2e_product_look_catalog_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
STATS = ROOT / "configs/film_color_stats.json"
GUARDS = ROOT / "configs/color_guardrails.json"


@pytest.fixture(scope="module")
def source() -> np.ndarray:
    y, x = np.mgrid[:59, :83]
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


@pytest.fixture(scope="module")
def runtime():
    profile = load_render_profile(PROFILE, root=ROOT)
    statistics = json.loads(STATS.read_text(encoding="utf-8"))["styles"]
    guardrails = {
        style: load_guardrail_config(GUARDS, style)
        for style in profile["style_parameters"]
    }
    return profile, statistics, guardrails


def test_contract_binds_both_parent_evidence_files() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["required_order"] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
        "generic_bw",
    ]
    for binding in contract["parent_evidence"]:
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]


def test_catalog_is_exact_ordered_and_does_not_expose_named_bw_stocks() -> None:
    rows = list_product_looks()
    assert [row["look_id"] for row in rows] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
        "generic_bw",
    ]
    assert [row["film_stock_id"] for row in rows[:3]] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert rows[-1]["film_stock_id"] is None
    assert "legacy_execution_style_id" not in rows[-1]
    assert "hp5" not in rows[-1]["display_name"].lower()
    assert "tri-x" not in rows[-1]["display_name"].lower()
    assert "tri-x" in rows[-1]["claim_ceiling"].lower()
    rows[0]["look_id"] = "forged"
    assert list_product_looks()[0]["look_id"] == "velvia_50"


@pytest.mark.parametrize("amount", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "look_id", ["velvia_50", "portra_400", "ektar_100", "generic_bw"]
)
def test_dispatch_is_exact_underlying_renderer(
    source: np.ndarray, runtime, look_id: str, amount: float
) -> None:
    profile, statistics, guardrails = runtime
    actual = render_product_look_rgb(
        source,
        profile=profile,
        look_id=look_id,
        look_amount=amount,
        style_statistics=statistics,
        guardrails=guardrails,
        seed=31,
        tile_size=23,
        tile_workers=2,
    )
    if look_id == "generic_bw":
        expected = render_generic_bw_look_rgb(
            source,
            profile=profile,
            look_amount=amount,
            style_statistics=statistics["hp5"],
            guardrails=guardrails["hp5"],
            seed=31,
            tile_size=23,
            tile_workers=2,
        )
    else:
        row = {item["style_id"]: item for item in list_three_stock_looks()}[
            look_id
        ]
        expected = render_three_stock_look_rgb(
            source,
            profile=profile,
            film_stock_id=row["film_stock_id"],
            look_amount=amount,
            style_statistics=statistics[look_id],
            guardrails=guardrails[look_id],
            seed=31,
            tile_size=23,
            tile_workers=2,
        )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("look_id", ["hp5", "tri_x_400", "unknown", ""])
def test_named_legacy_bw_and_unknown_look_ids_fail_closed(
    source: np.ndarray, runtime, look_id: str
) -> None:
    profile, statistics, guardrails = runtime
    with pytest.raises(ValueError, match="unsupported product look"):
        render_product_look_rgb(
            source,
            profile=profile,
            look_id=look_id,
            look_amount=1.0,
            style_statistics=statistics,
            guardrails=guardrails,
            seed=31,
        )


def test_missing_runtime_assets_fail_closed(source: np.ndarray, runtime) -> None:
    profile, statistics, guardrails = runtime
    missing_statistics = dict(statistics)
    del missing_statistics["hp5"]
    with pytest.raises(ValueError, match="runtime assets are incomplete"):
        render_product_look_rgb(
            source,
            profile=profile,
            look_id="generic_bw",
            look_amount=1.0,
            style_statistics=missing_statistics,
            guardrails=guardrails,
            seed=31,
        )
