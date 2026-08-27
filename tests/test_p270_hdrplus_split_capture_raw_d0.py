from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_p270_hdrplus_split_capture_raw_d0 import (
    P270Error,
    ReadLedger,
    _reduction,
    _select_blocks,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p270_hdrplus_split_capture_raw_d0_v1.json"


def test_p270_roles_and_gates_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    development = config["development_frames"]
    heldout = config["heldout_frames"]
    assert len(development) == len(heldout) == 4
    assert set(development).isdisjoint(heldout)
    assert config["block_selection"]["required_blocks"] == 512
    assert config["gates"]["maximum_result_pixel_reads"] == 0
    assert config["gates"]["minimum_vs_strongest_single_win_rate"] == 0.75


def test_p270_ledger_rejects_heldout_before_freeze(tmp_path: Path) -> None:
    ledger = ReadLedger()
    with pytest.raises(P270Error, match="heldout frame read before candidate freeze"):
        ledger.record(tmp_path / "heldout.dng", "heldout")
    ledger.record(tmp_path / "development.dng", "development")
    ledger.freeze("abc")
    ledger.record(tmp_path / "heldout.dng", "heldout")
    assert ledger.freeze_event_index == 1
    assert [event["role"] for event in ledger.events] == ["development", "heldout"]


def test_p270_block_selection_is_stable_and_source_only() -> None:
    planes = [np.full((64, 64), value, dtype=np.float32) for value in range(4)]
    stds = [np.full((64, 64), 0.01 + value, dtype=np.float32) for value in range(4)]
    masks = [np.ones((64, 64), dtype=bool) for _ in range(4)]
    blocks = _select_blocks(planes, stds, masks, 32, 4, 0.25)
    assert [(row["plane"], row["y"], row["x"]) for row in blocks] == [
        (0, 0, 0),
        (0, 0, 32),
        (0, 32, 0),
        (0, 32, 32),
    ]


def test_p270_reduction_handles_zero_baseline() -> None:
    assert _reduction(0.0, 0.0) == 0.0
    assert _reduction(1.0, 0.0) == -1e9
    assert _reduction(1.0, 2.0) == 0.5
