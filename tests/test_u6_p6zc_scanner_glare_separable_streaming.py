from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval.scanner_glare_separable_streaming import (
    ScannerGlareStreamingError,
    evaluate_contract,
    load_contract,
)
from src.film_physics.scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareComponent,
    ScannerGlareDomainError,
)
from src.film_physics.scanner_glare_streaming import (
    apply_scanner_glare_separable_streaming,
    compile_scanner_glare_separable,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6zc_scanner_glare_separable_streaming_v1.json"


def test_p6zc_returns_a_frozen_decision() -> None:
    report = evaluate_contract(ROOT, CONTRACT)
    assert report["decision"] in {
        "retain_separable_streaming_open_performance_benchmark",
        "close_exact_direct_separable_scanner_glare_compiler",
    }
    assert len(report["partition_rows"]) == 5


def test_streaming_rejects_invalid_row_chunk() -> None:
    profile = MultiscaleScannerGlareProfile(
        components=(ScannerGlareComponent(1.0, 2.0),),
        flare_fraction=0.1,
        truncate_sigma=4.0,
    )
    compiled = compile_scanner_glare_separable(profile, kernel_size=17)
    with pytest.raises(ScannerGlareDomainError, match="row chunk"):
        apply_scanner_glare_separable_streaming(
            np.zeros((5, 7)), compiled, flare_fraction=0.1, row_chunk=0
        )


def test_p6zc_contract_drift_rejected(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["fixture"]["row_partitions"] = [64]
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScannerGlareStreamingError, match="boundary drift"):
        load_contract(path)


def test_p6zc_runner_is_byte_deterministic(tmp_path: Path) -> None:
    script = ROOT / "scripts/run_u6_p6zc_scanner_glare_separable_streaming.py"
    outputs = [tmp_path / "a.json", tmp_path / "b.json"]
    for output in outputs:
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            cwd=tmp_path,
            check=True,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
