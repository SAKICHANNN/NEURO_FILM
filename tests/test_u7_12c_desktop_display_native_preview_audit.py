from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.audit_u7_12c_desktop_display_native_preview import evaluate_records

ROOT = Path(__file__).resolve().parents[1]


def _run(*, width: int, height: int, wall: float, rss: int, candidate: bool) -> dict:
    outputs = {
        style: {
            "file_sha256": token * 64,
            "decoded_rgb_sha256": token.upper() * 64,
            "width": width,
            "height": height,
        }
        for style, token in (
            ("ektar_100", "a"),
            ("portra_400", "b"),
            ("velvia_50", "c"),
        )
    }
    manifest = {
        "preview_width": width,
        "preview_height": height,
        "jpeg_scaled_decode": False,
        "raw_half_size_decode": True,
    }
    if candidate:
        manifest.update({"max_preview_width": 300, "max_preview_height": 260})
    return {
        "run_index": 0,
        "renderer_wall_seconds": wall,
        "outer_process_wall_seconds": wall + 1.0,
        "peak_process_tree_rss_bytes": rss,
        "manifest": manifest,
        "outputs": outputs,
    }


def _record(source: dict, *, jpeg: bool) -> dict:
    baseline_width, baseline_height = source["baseline_preview_dimensions"]
    candidate_width, candidate_height = source["candidate_preview_dimensions"]
    baseline = _run(
        width=baseline_width,
        height=baseline_height,
        wall=4.0,
        rss=400_000_000,
        candidate=False,
    )
    candidate = _run(
        width=candidate_width,
        height=candidate_height,
        wall=0.5 if jpeg else 2.0,
        rss=200_000_000,
        candidate=True,
    )
    if jpeg:
        for run in (baseline, candidate):
            run["manifest"]["jpeg_scaled_decode"] = True
            run["manifest"].pop("raw_half_size_decode")
    fidelity = [
        [
            {
                "style_id": style,
                "visible_width": candidate_width,
                "visible_height": candidate_height,
                "baseline_visible_rgb_sha256": "d" * 64,
                "candidate_rgb_sha256": "e" * 64,
                "rgb_rmse": 0.01,
                "rgb_absolute_error_p95": 0.02,
                "new_boundary_fraction": 0.0,
            }
            for style in ("ektar_100", "portra_400", "velvia_50")
        ]
        for _ in range(2)
    ]
    second_baseline = deepcopy(baseline)
    second_baseline["run_index"] = 1
    second_candidate = deepcopy(candidate)
    second_candidate["run_index"] = 1
    return {
        "source_id": source["source_id"],
        "source_path": source["path"],
        "source_bytes": source["bytes"],
        "source_sha256": source["sha256"],
        "source_unchanged": True,
        "baseline_runs": [baseline, second_baseline],
        "candidate_runs": [candidate, second_candidate],
        "fidelity": fidelity,
    }


def test_config_freezes_two_sources_visible_box_and_no_rescue() -> None:
    config = json.loads(
        (ROOT / "configs/u7_12c_desktop_display_native_preview_v1.json").read_text(
            "utf-8"
        )
    )
    assert config["display_box"] == {
        "width": 300,
        "height": 260,
        "ui_resample": "Pillow Image.Resampling.LANCZOS",
        "candidate_rule": "aspect fit without upsampling before look execution",
    }
    assert [row["candidate_preview_dimensions"] for row in config["sources"]] == [
        [173, 260],
        [300, 158],
    ]
    assert len(config["stop_rules"]) == 3


def test_evaluator_requires_every_quality_and_performance_gate() -> None:
    config = json.loads(
        (ROOT / "configs/u7_12c_desktop_display_native_preview_v1.json").read_text(
            "utf-8"
        )
    )
    records = [
        _record(source, jpeg=index == 0)
        for index, source in enumerate(config["sources"])
    ]
    passing = evaluate_records(config, records)
    assert all(passing.values())

    slow = deepcopy(records)
    for run in slow[0]["candidate_runs"]:
        run["renderer_wall_seconds"] = 2.0
    failed = evaluate_records(config, slow)
    assert failed["jpeg_renderer_wall_seconds"] is False
    assert failed["jpeg_renderer_wall_ratio"] is False

    drifted = deepcopy(records)
    drifted[1]["candidate_runs"][1]["outputs"]["ektar_100"]["decoded_rgb_sha256"] = (
        "f" * 64
    )
    assert (
        evaluate_records(config, drifted)["candidate_repeat_pixels_and_outputs_exact"]
        is False
    )
