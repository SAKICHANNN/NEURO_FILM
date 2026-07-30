from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.download_u5_r2aw0_filmmatch_source import (
    FilmMatchAcquisitionError,
    validate_remote_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        ROOT / "configs/u5_r2aw0_filmmatch_ektachrome_paired_source_v1.json"
    ).read_text(encoding="utf-8")
)


def _rows(start: int, end: int, lane: str) -> list[dict]:
    return [
        {
            "file_id": f"id-{number}",
            "name": f"Still 2025-08-21 150231_1.{number}.1.tif",
            "lane": lane,
        }
        for number in range(start, end + 1)
    ]


def test_remote_chart_inventory_sorts_and_accepts_exact_sequence() -> None:
    folder = CONFIG["acquisition"]["folders"][0]
    rows = list(reversed(_rows(315, 347, folder["lane"])))
    validated = validate_remote_inventory(rows, folder)
    assert validated[0]["name"].endswith("_1.315.1.tif")
    assert validated[-1]["name"].endswith("_1.347.1.tif")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "sequence"])
def test_remote_chart_inventory_rejects_drift(mutation: str) -> None:
    folder = copy.deepcopy(CONFIG["acquisition"]["folders"][0])
    rows = _rows(315, 347, folder["lane"])
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[-1]["name"] = rows[-1]["name"].replace(".347.", ".999.")
    with pytest.raises(FilmMatchAcquisitionError):
        validate_remote_inventory(rows, folder)
