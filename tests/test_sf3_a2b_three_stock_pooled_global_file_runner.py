from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import src.real_film.three_stock_pooled_global_file_runner as runner
from src.real_film.three_stock_k1_baseline import StockFrameSamples

ROOT = Path(__file__).resolve().parents[1]
POOLED_CONFIG = ROOT / "configs/sf3_a2b_three_stock_pooled_global_control_v1.json"
INTEGRITY_CONFIG = (
    ROOT / "configs/sf3_a1_three_stock_file_pixel_alignment_integrity_v1.json"
)
STOCKS = (
    "fujifilm_velvia_50",
    "kodak_portra_400",
    "kodak_ektar_100",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="ascii")


def test_integrity_failure_stops_before_sampling_or_fit(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger.json"
    manifest = tmp_path / "manifest.json"
    _write_json(ledger, {"rows": []})
    _write_json(manifest, {"rows": []})
    monkeypatch.setattr(
        runner,
        "evaluate_integrity",
        lambda *args, **kwargs: {
            "automatic_pass": False,
            "stable_evidence_id": "integrity-fail",
            "decision": "RETAIN_THREE_STOCK_DATA_GAP",
        },
    )
    monkeypatch.setattr(
        runner,
        "load_aligned_file_rows",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("sampling reached")),
    )
    report = runner.evaluate_files(
        root=ROOT,
        integrity_contract_path=INTEGRITY_CONFIG,
        pooled_contract_path=POOLED_CONFIG,
        ledger_path=ledger,
        manifest_path=manifest,
    )
    assert report["automatic_pass"] is False
    assert report["paired_sampling_executed"] is False
    assert report["operator_fits"] == 0


def test_verified_samples_flow_to_pooled_control_with_exact_fit_count(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger.json"
    manifest = tmp_path / "manifest.json"
    _write_json(ledger, {"rows": []})
    _write_json(manifest, {"rows": []})
    monkeypatch.setattr(
        runner,
        "evaluate_integrity",
        lambda *args, **kwargs: {
            "automatic_pass": True,
            "stable_evidence_id": "integrity-pass",
        },
    )
    monkeypatch.setattr(runner, "load_aligned_file_rows", lambda **kwargs: ([], {}))
    sample = np.full((4, 3), 0.25, dtype=np.float64)
    roll_counts = (2, 3, 4)
    development = {
        stock: [
            StockFrameSamples(
                scene_id=f"scene-{roll}",
                frame_id=f"frame-{roll}",
                roll_id=f"roll-{roll}",
                source=sample,
                target=sample,
            )
            for roll in range(roll_count)
        ]
        for stock, roll_count in zip(STOCKS, roll_counts, strict=True)
    }
    confirmation = {stock: [] for stock in STOCKS}
    sampling_facts = {"sampled_rows": sum(roll_counts)}
    monkeypatch.setattr(
        runner,
        "extract_common_paired_samples_streaming",
        lambda rows, config, **kwargs: (
            development,
            confirmation,
            sampling_facts,
        ),
    )
    observed: dict[str, object] = {}

    def fake_pooled(*args, **kwargs):
        observed.update(kwargs)
        return {
            "automatic_pass": True,
            "decision": "PASS",
            "claim_ceiling": "test-only",
            "stable_evidence_id": "pooled-pass",
        }

    monkeypatch.setattr(runner, "evaluate_pooled_global", fake_pooled)
    report = runner.evaluate_files(
        root=ROOT,
        integrity_contract_path=INTEGRITY_CONFIG,
        pooled_contract_path=POOLED_CONFIG,
        ledger_path=ledger,
        manifest_path=manifest,
    )
    assert report["automatic_pass"] is True
    assert report["paired_sampling"] == sampling_facts
    assert report["operator_fits"] == 58
    assert observed["development"] is development
    assert observed["confirmation"] is confirmation


def test_report_identity_is_stable_for_same_verified_handoff(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger.json"
    manifest = tmp_path / "manifest.json"
    _write_json(ledger, {"rows": []})
    _write_json(manifest, {"rows": []})
    monkeypatch.setattr(
        runner,
        "evaluate_integrity",
        lambda *args, **kwargs: {
            "automatic_pass": False,
            "stable_evidence_id": "integrity-fail",
            "decision": "RETAIN_THREE_STOCK_DATA_GAP",
        },
    )
    kwargs = {
        "root": ROOT,
        "integrity_contract_path": INTEGRITY_CONFIG,
        "pooled_contract_path": POOLED_CONFIG,
        "ledger_path": ledger,
        "manifest_path": manifest,
    }
    first = runner.evaluate_files(**kwargs)
    second = runner.evaluate_files(**kwargs)
    assert first == second
    assert first["stable_evidence_id"] == second["stable_evidence_id"]
