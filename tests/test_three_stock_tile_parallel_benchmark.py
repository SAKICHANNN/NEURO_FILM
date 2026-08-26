from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_three_stock_tile_parallel_benchmark_is_exact_and_replayable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "report.json"
    pixels = np.arange(31 * 47 * 3, dtype=np.uint16).reshape(31, 47, 3)
    Image.fromarray((pixels % 256).astype(np.uint8), mode="RGB").save(source)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_three_stock_tile_parallel.py"),
        str(source),
        "--output",
        str(output),
        "--width",
        "47",
        "--height",
        "31",
        "--tile-size",
        "11",
        "--workers",
        "4",
    ]
    first = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert first.returncode == 0, first.stderr
    first_report = json.loads(output.read_text(encoding="utf-8"))
    output.unlink()
    second = subprocess.run(
        [*command, "--order", "reverse"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    second_report = json.loads(output.read_text(encoding="utf-8"))

    assert first_report["stable_identity"] == second_report["stable_identity"]
    assert first_report["outputs"] == second_report["outputs"]
    assert all(row["wall_ratio"] > 0.0 for row in first_report["rows"])
