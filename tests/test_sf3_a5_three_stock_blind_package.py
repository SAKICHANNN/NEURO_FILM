from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.three_stock_blind_package import (
    ThreeStockBlindPackageError,
    adjudicate_package,
    build_package,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a5_three_stock_blind_distinguishability_v1.json"
K1 = ROOT / "configs/sf3_a2_three_stock_k1_baseline_v1.json"
A4 = ROOT / "configs/sf3_a4_three_stock_confirmation_render_v1.json"
STOCKS = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _fixture(tmp_path: Path) -> dict[str, Path]:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / K1.name).write_bytes(K1.read_bytes())
    (configs / A4.name).write_bytes(A4.read_bytes())
    contract = configs / CONTRACT.name
    contract.write_bytes(CONTRACT.read_bytes())
    render_root = tmp_path / "outputs" / "a4"
    outputs = []
    scenes = [f"confirm-{index}" for index in range(1, 5)]
    for scene in scenes:
        for stock in STOCKS:
            relative = Path(stock) / f"{scene}.png"
            raw = f"png:{scene}:{stock}".encode()
            path = render_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            outputs.append(
                {
                    "scene_id": scene,
                    "stock_id": stock,
                    "relative_path": relative.as_posix(),
                    "png_sha256": _sha(raw),
                }
            )
    render = {
        "schema": "neuro-film.sf3-a4-three-stock-confirmation-render-report.v1",
        "automatic_pass": True,
        "decision": "OPEN_THREE_STOCK_K1_SEVERE_ARTIFACT_REVIEW_THEN_DOMAIN_SEPARATED_BLIND_DISTINGUISHABILITY",
        "confirmation_scenes": scenes,
        "stocks": STOCKS,
        "outputs": outputs,
    }
    render_path = render_root / "report.json"
    render_raw = _write(render_path, render)
    severe = {
        "schema": "neuro-film.sf3-a5-three-stock-severe-review.v1",
        "render_report_sha256": _sha(render_raw),
        "confirmed_severe_count": 0,
        "reviewed_outputs": [
            {
                "scene_id": row["scene_id"],
                "stock_id": row["stock_id"],
                "png_sha256": row["png_sha256"],
                "confirmed_severe": False,
            }
            for row in outputs
        ],
    }
    severe_path = tmp_path / "severe.json"
    _write(severe_path, severe)
    return {
        "root": tmp_path,
        "contract": contract,
        "render_root": render_root,
        "render": render_path,
        "severe": severe_path,
    }


def test_builds_hidden_hash_bound_package_and_adjudicates(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "outputs" / "a5"
    report = build_package(
        fixture["contract"],
        root=fixture["root"],
        render_report_path=fixture["render"],
        render_root=fixture["render_root"],
        severe_review_path=fixture["severe"],
        secret="not-published",
        output_dir=output,
    )
    assert report["blind_png_count"] == 36
    sheet_raw = (output / "public_sheet.json").read_bytes()
    assert all(stock.encode() not in sheet_raw for stock in STOCKS)
    mapping = json.loads((output / "private_mapping.json").read_text())
    assignments = [
        {
            "round": row["round"],
            "scene_id": row["scene_id"],
            "label_to_stock": {
                label: facts["stock_id"] for label, facts in row["labels"].items()
            },
        }
        for row in mapping["rows"]
    ]
    package_raw = (output / "report.json").read_bytes()
    observations = {
        "schema": "neuro-film.sf3-a5-three-stock-blind-observations.v1",
        "status": "observations_frozen_mapping_unread",
        "mapping_files_read": False,
        "package_report_sha256": _sha(package_raw),
        "public_sheet_sha256": _sha(sheet_raw),
        "assignments": assignments,
    }
    observations_path = tmp_path / "observations.json"
    observations_raw = _write(observations_path, observations)
    reveal_path = tmp_path / "reveal.json"
    _write(
        reveal_path,
        {
            "schema": "neuro-film.sf3-a5-three-stock-mapping-reveal.v1",
            "status": "mapping_revealed_after_observations_commit",
            "observations_sha256": _sha(observations_raw),
            "private_mapping_sha256": _sha(
                (output / "private_mapping.json").read_bytes()
            ),
            "observations_commit": "a" * 40,
        },
    )
    final = adjudicate_package(
        fixture["contract"],
        package_report_path=output / "report.json",
        public_sheet_path=output / "public_sheet.json",
        private_mapping_path=output / "private_mapping.json",
        observations_path=observations_path,
        reveal_path=reveal_path,
    )
    assert final["automatic_pass"] is True
    assert final["preference_claim_allowed"] is False


def test_severe_or_incomplete_evidence_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    severe = json.loads(fixture["severe"].read_text())
    severe["confirmed_severe_count"] = 1
    _write(fixture["severe"], severe)
    with pytest.raises(ThreeStockBlindPackageError, match="severe review"):
        build_package(
            fixture["contract"],
            root=fixture["root"],
            render_report_path=fixture["render"],
            render_root=fixture["render_root"],
            severe_review_path=fixture["severe"],
            secret="hidden",
            output_dir=tmp_path / "outputs" / "blocked",
        )
