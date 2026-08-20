from __future__ import annotations

from pathlib import Path

import pytest

from scripts import acquire_u5_r2repid9b_originals as acquisition

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


def test_transient_transport_failure_is_retried_without_retrying_identity_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0
    sleeps: list[float] = []

    def transient_then_success(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("transient")

    monkeypatch.setattr(acquisition, "_download", transient_then_success)
    monkeypatch.setattr(acquisition.time, "sleep", sleeps.append)
    acquisition._download_with_retry("url", tmp_path / "x", 1, "sha", ".part", (0.5, 1.0, 2.0))
    assert calls == 3
    assert sleeps == [0.5, 1.0]

    def identity_failure(*args: object) -> None:
        raise ValueError("hash")

    monkeypatch.setattr(acquisition, "_download", identity_failure)
    with pytest.raises(ValueError, match="hash"):
        acquisition._download_with_retry("url", tmp_path / "x", 1, "sha", ".part", (0.5,))
