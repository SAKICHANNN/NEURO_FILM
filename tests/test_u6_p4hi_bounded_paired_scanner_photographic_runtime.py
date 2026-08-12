from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.bounded_paired_scanner_photographic_runtime import load_contract
from src.eval.layer_gamma_photographic_development import _compact_contact_panel

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4hi_bounded_paired_scanner_photographic_runtime_v1.json"
LIFETIME_CONTRACT = (
    ROOT / "configs/u6_p4hj_lifetime_scheduled_bounded_photographic_runtime_v1.json"
)


def test_p4hi_contract_freezes_bounded_runtime_and_resource_gates() -> None:
    payload = load_contract(CONTRACT)
    assert payload["candidate"]["rank_bins"] == 65536
    assert payload["candidate"]["canonical_row_block_height"] == 128
    assert payload["candidate"]["full_frame_visual_rows_retained"] is False
    assert payload["automatic_gates"]["maximum_peak_process_tree_rss_bytes"] == 2**30
    assert payload["measurement"]["formal_processes"] == 2


def test_compact_contact_panel_is_small_deterministic_uint8() -> None:
    values = np.linspace(
        0.0, 1.0, 1000 * 1200 * 3, dtype=np.float32
    ).reshape(1000, 1200, 3)
    first = _compact_contact_panel(values)
    second = _compact_contact_panel(values.copy())
    assert first.dtype == np.uint8
    assert first.shape[0] <= 180
    assert first.shape[1] <= 256
    assert np.array_equal(first, second)
    assert first.nbytes < values.nbytes


def test_p4hi_contract_parents_are_hash_bound() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert set(payload["parents"]) == {
        "p4he_result",
        "p4he_contract",
        "p4hh_result",
        "p4hh_contract",
    }
    assert all(len(binding["sha256"]) == 64 for binding in payload["parents"].values())


def test_p4hj_contract_preserves_p4hi_pixels_and_budget() -> None:
    payload = load_contract(LIFETIME_CONTRACT)
    assert payload["candidate"][
        "require_exact_p4hi_physical_and_combined_output_sha256"
    ]
    assert payload["automatic_gates"]["maximum_peak_live_temporary_bytes"] == 2**24
    assert payload["parents"]["p4hi_report"]["sha256"] == (
        "ce9b6e5f61aee9aa3d2092ea061b3e616303e2f1e84ccb0980229d0c51166d05"
    )
