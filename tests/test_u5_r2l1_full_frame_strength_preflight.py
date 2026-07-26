from __future__ import annotations

from pathlib import Path


def test_runner_normalizes_explicit_relative_paths() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_u5_r2l1_full_frame_strength_preflight.py"
    ).read_text(encoding="utf-8")
    assert "if args.config.is_absolute()" in source
    assert "if args.output.is_absolute()" in source
