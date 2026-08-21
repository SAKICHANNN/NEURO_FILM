from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.real_film.three_stock_confirmation_render import (
    ThreeStockConfirmationRenderError,
    evaluate_and_materialize,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a4_three_stock_confirmation_render_v1.json"
K1 = ROOT / "configs/sf3_a2_three_stock_k1_baseline_v1.json"
INTEGRITY = ROOT / "configs/sf3_a1_three_stock_file_pixel_alignment_integrity_v1.json"
STOCKS = (
    "fujifilm_velvia_50",
    "kodak_portra_400",
    "kodak_ektar_100",
)
DECISION = "OPEN_THREE_STOCK_K1_SEVERE_ARTIFACT_REVIEW_THEN_BLIND_DISTINGUISHABILITY"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _operator(gain: float) -> dict[str, object]:
    return {
        "kind": "bounded_affine",
        "working_space": "interpreted_relative_display_rgb",
        "matrix": np.diag([gain, gain, gain]).tolist(),
        "bias": [0.0, 0.0, 0.0],
        "identity_shrinkage": 0.0,
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / K1.name).write_bytes(K1.read_bytes())
    (config_root / INTEGRITY.name).write_bytes(INTEGRITY.read_bytes())
    ledger_rows = []
    manifest_rows = []
    scenes = ("confirm-01", "confirm-02")
    for index, scene in enumerate(scenes):
        values = np.linspace(8 + index, 220 - index, 8 * 9 * 3, dtype=np.uint8).reshape(
            8, 9, 3
        )
        source = tmp_path / "data" / f"{scene}.png"
        source.parent.mkdir(exist_ok=True)
        assert cv2.imwrite(str(source), values[..., ::-1])
        digest = _sha(source.read_bytes())
        for stock_index, stock in enumerate(STOCKS):
            ledger_rows.append(
                {
                    "digital_reference_path": f"data/{scene}.png",
                    "scan_sample_path": f"data/must-not-read-{scene}-{stock_index}.tif",
                }
            )
            manifest_rows.append(
                {
                    "row_id": f"{scene}-{stock}",
                    "stock_id": stock,
                    "role": "confirmation",
                    "scene_id": scene,
                    "film_frame_id": f"frame-{scene}-{stock}",
                    "roll_id": f"roll-{stock}",
                    "digital_reference_sha256": digest,
                    "scan_sample_sha256": f"{stock_index + 1}" * 64,
                }
            )
    ledger = tmp_path / "ledger.json"
    manifest = tmp_path / "manifest.json"
    ledger_raw = _write_json(ledger, {"rows": ledger_rows})
    manifest_raw = _write_json(manifest, {"rows": manifest_rows})
    k1_result = {
        "schema": "neuro-film.sf3-a2-three-stock-k1-baseline-report.v1",
        "contract_sha256": _sha(K1.read_bytes()),
        "selection_used_confirmation_targets": False,
        "automatic_pass": True,
        "decision": DECISION,
        "common_confirmation_scenes": list(scenes),
        "stocks": {
            stock: {"operator": _operator(gain), "automatic_pass": True}
            for stock, gain in zip(STOCKS, (0.8, 0.9, 1.0), strict=True)
        },
    }
    report = {
        "schema": "neuro-film.sf3-a2-three-stock-k1-file-runner-report.v1",
        "automatic_pass": True,
        "decision": DECISION,
        "k1_contract_sha256": _sha(K1.read_bytes()),
        "ledger_sha256": _sha(ledger_raw),
        "manifest_sha256": _sha(manifest_raw),
        "integrity_automatic_pass": True,
        "paired_sampling_executed": True,
        "operator_fits": 30,
        "k1_result": k1_result,
    }
    report_path = tmp_path / "a2-report.json"
    _write_json(report_path, report)
    return {
        "root": tmp_path,
        "ledger": ledger,
        "manifest": manifest,
        "report": report_path,
    }


def test_contract_binds_k1_parent() -> None:
    _, value = load_contract(CONTRACT, root=ROOT)
    assert value["required_stocks"] == list(STOCKS)
    assert value["source"]["film_target_file_reads_allowed"] == 0


def test_materializes_exact_target_blind_stock_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = evaluate_and_materialize(
        CONTRACT,
        root=fixture["root"],
        a2_report_path=fixture["report"],
        ledger_path=fixture["ledger"],
        manifest_path=fixture["manifest"],
        output_dir=tmp_path / "outputs" / "run-a",
    )
    second = evaluate_and_materialize(
        CONTRACT,
        root=fixture["root"],
        a2_report_path=fixture["report"],
        ledger_path=fixture["ledger"],
        manifest_path=fixture["manifest"],
        output_dir=tmp_path / "outputs" / "run-b",
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert first["film_target_file_reads"] == 0
    assert first["operator_refits"] == 0
    assert first["digital_source_decodes"] == 2
    assert len(first["outputs"]) == 6
    assert len({row["png_sha256"] for row in first["outputs"]}) == 6
    assert (tmp_path / "outputs" / "run-a" / "report.json").is_file()
    for row in first["outputs"]:
        assert row["raw_output_clip_fraction"] == 0.0
        assert (tmp_path / "outputs" / "run-a" / row["relative_path"]).is_file()


def test_nonpassing_parent_creates_no_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = json.loads(fixture["report"].read_text())
    report["automatic_pass"] = False
    _write_json(fixture["report"], report)
    output = tmp_path / "outputs" / "blocked"
    with pytest.raises(ThreeStockConfirmationRenderError, match="did not open"):
        evaluate_and_materialize(
            CONTRACT,
            root=fixture["root"],
            a2_report_path=fixture["report"],
            ledger_path=fixture["ledger"],
            manifest_path=fixture["manifest"],
            output_dir=output,
        )
    assert not output.exists()


def test_out_of_bounds_operator_leaves_no_output_or_stage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = json.loads(fixture["report"].read_text())
    report["k1_result"]["stocks"][STOCKS[0]]["operator"] = _operator(2.0)
    _write_json(fixture["report"], report)
    output = tmp_path / "outputs" / "unsafe"
    with pytest.raises(ThreeStockConfirmationRenderError, match=r"left \[0,1\]"):
        evaluate_and_materialize(
            CONTRACT,
            root=fixture["root"],
            a2_report_path=fixture["report"],
            ledger_path=fixture["ledger"],
            manifest_path=fixture["manifest"],
            output_dir=output,
        )
    assert not output.exists()
    assert not list((tmp_path / "outputs").glob(".unsafe.stage-*"))


def test_output_must_use_logical_outputs_root(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ThreeStockConfirmationRenderError, match="logical outputs"):
        evaluate_and_materialize(
            CONTRACT,
            root=fixture["root"],
            a2_report_path=fixture["report"],
            ledger_path=fixture["ledger"],
            manifest_path=fixture["manifest"],
            output_dir=tmp_path / "outside-output-root",
        )
    assert not (tmp_path / "outside-output-root").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("integrity_automatic_pass", False),
        ("paired_sampling_executed", False),
        ("operator_fits", 0),
    ],
)
def test_incomplete_parent_execution_creates_no_output(
    tmp_path: Path, field: str, value: object
) -> None:
    fixture = _fixture(tmp_path)
    report = json.loads(fixture["report"].read_text())
    report[field] = value
    _write_json(fixture["report"], report)
    output = tmp_path / "outputs" / "incomplete"
    with pytest.raises(ThreeStockConfirmationRenderError, match="chain is incomplete"):
        evaluate_and_materialize(
            CONTRACT,
            root=fixture["root"],
            a2_report_path=fixture["report"],
            ledger_path=fixture["ledger"],
            manifest_path=fixture["manifest"],
            output_dir=output,
        )
    assert not output.exists()
