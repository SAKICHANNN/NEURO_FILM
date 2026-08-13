from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.kodak_vision3_200t_raster_signature import (
    REPORT_SCHEMA,
    Vision3200TRasterSignatureError,
    audit_source_signature,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bu9_kodak_vision3_200t_raster_signature_v1.json"


def test_vision3_200t_raster_signature_is_exact_and_bounded(tmp_path: Path) -> None:
    config = load_contract(CONFIG)
    first = audit_source_signature(config, ROOT, overlay_path=tmp_path / "a.png")
    second = audit_source_signature(config, ROOT, overlay_path=tmp_path / "b.png")
    assert first == second
    assert first["schema"] == REPORT_SCHEMA
    assert first["overlay_sha256"] == second["overlay_sha256"]
    assert set(first["observed_samples"]) == {"characteristic", "mtf", "granularity"}
    assert all(first["gate_results"].values()) == first["signature_pass"]
    assert "not a particular-roll" in first["claim_ceiling"]


def test_vision3_200t_rejects_embedded_graph_drift(tmp_path: Path) -> None:
    config = load_contract(CONFIG)
    config = copy.deepcopy(config)
    config["raster_source"]["mtf"]["png_sha256"] = "0" * 64
    with pytest.raises(Vision3200TRasterSignatureError, match="embedded graph drift"):
        audit_source_signature(config, ROOT, overlay_path=tmp_path / "x.png")


def test_vision3_200t_contract_rejects_gate_relaxation(tmp_path: Path) -> None:
    payload = CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "config.json"
    path.write_text(payload.replace('"minimum_material_vision3_pairs": 3', '"minimum_material_vision3_pairs": 2'), encoding="utf-8")
    with pytest.raises(Vision3200TRasterSignatureError, match="gates drift"):
        load_contract(path)


def test_vision3_200t_real_runner_entrypoint(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_u5_r2bu9_kodak_vision3_200t_raster_signature.py"),
            "--config",
            str(CONFIG.relative_to(ROOT)),
            "--output",
            str((tmp_path / "report.json").relative_to(ROOT) if tmp_path.is_relative_to(ROOT) else tmp_path / "report.json"),
            "--overlay",
            str((tmp_path / "overlay.png").relative_to(ROOT) if tmp_path.is_relative_to(ROOT) else tmp_path / "overlay.png"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"signature_pass":' in result.stdout
