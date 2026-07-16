from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.repair_yfcc_full_index_range import _expected_content_range, apply_repair_range


def test_expected_content_range_is_exact_and_inclusive() -> None:
    assert _expected_content_range(10, 19, 100) == "bytes 10-19/100"


def test_apply_repair_range_restores_only_requested_bytes(tmp_path: Path) -> None:
    original_path = tmp_path / "original.sqlite"
    with sqlite3.connect(original_path) as connection:
        connection.execute("CREATE TABLE sample (value TEXT)")
        connection.executemany("INSERT INTO sample VALUES (?)", [(str(index),) for index in range(200)])
    original = original_path.read_bytes()
    damaged_path = tmp_path / "damaged.sqlite"
    damaged = bytearray(original)
    start = len(damaged) // 2
    end = min(start + 128, len(damaged))
    damaged[start:end] = b"\xff" * (end - start)
    damaged_path.write_bytes(damaged)
    repair_path = tmp_path / "repair.bin"
    repair_path.write_bytes(original[start:end])

    apply_repair_range(damaged_path, repair_path, start, end, len(original))

    assert damaged_path.read_bytes() == original
