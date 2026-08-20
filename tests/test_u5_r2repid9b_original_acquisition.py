from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_acquisition_requires_explicit_output_root_and_report() -> None:
    source = (ROOT / "scripts/acquire_u5_r2repid9b_originals.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--output-root", type=Path, required=True)' in source
    assert 'parser.add_argument("--report", type=Path, required=True)' in source
    assert "D:\\" not in source
    assert "P:\\" not in source


def test_acquisition_never_constructs_sealed_tasks() -> None:
    source = (ROOT / "scripts/acquire_u5_r2repid9b_originals.py").read_text(encoding="utf-8")
    assert 'manifest["members"]' in source
    assert '"sealed_requests": 0' in source
    assert "selected" not in source
