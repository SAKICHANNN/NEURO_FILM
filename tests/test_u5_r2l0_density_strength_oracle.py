from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.density_strength_oracle import (
    DensityStrengthOracleError,
    evaluate_oracle,
)


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _json(path: Path, value: object) -> str:
    return _write(
        path,
        (json.dumps(value, sort_keys=True) + "\n").encode("utf-8"),
    )


def _fixture(tmp_path: Path) -> tuple[dict[str, object], str]:
    candidates: dict[str, object] = {}
    manifest_records = []
    for candidate_id, strength, styles, clips in (
        ("cyan__s50", 0.5, (8.0, 9.0), (0.0, 0.0)),
        ("cyan__s65", 0.65, (11.0, 12.0), (0.001, 0.02)),
    ):
        per_image = []
        for index, sample_id in enumerate(("a", "b")):
            output = f"{candidate_id}/{sample_id}.png"
            output_sha = _write(
                tmp_path / "parent" / "render" / output,
                f"{candidate_id}/{sample_id}".encode(),
            )
            per_image.append(
                {
                    "sample_id": sample_id,
                    "split": "gold" if sample_id == "a" else "stress",
                    "median_style_delta_e76": styles[index],
                    "median_non_basic_residual_delta_e76": styles[index] / 2,
                    "new_hard_clipping_fraction": clips[index],
                    "output_sha256": output_sha,
                }
            )
            manifest_records.append(
                {
                    "candidate_id": candidate_id,
                    "sample_id": sample_id,
                    "output": output,
                    "output_sha256": output_sha,
                }
            )
        candidates[candidate_id] = {
            "witness_id": "cyan",
            "strength": strength,
            "per_image": per_image,
        }
    report_path = tmp_path / "parent" / "report.json"
    report_hash = _json(
        report_path,
        {"frozen_set_sha256": "set", "candidates": candidates},
    )
    manifest_path = tmp_path / "parent" / "render" / "manifest.json"
    manifest_hash = _json(
        manifest_path,
        {"frozen_set_sha256": "set", "records": manifest_records},
    )
    config: dict[str, object] = {
        "experiment_id": "test",
        "parent": {
            "automatic_report": "parent/report.json",
            "automatic_report_sha256": report_hash,
            "render_manifest": "parent/render/manifest.json",
            "render_manifest_sha256": manifest_hash,
            "frozen_set_sha256": "set",
        },
        "witness_id": "cyan",
        "baseline_candidate_id": "cyan__s50",
        "challenger_candidate_id": "cyan__s65",
        "baseline_strength": 0.5,
        "challenger_strength": 0.65,
        "expected_samples": 2,
        "expected_gold_samples": 1,
        "expected_stress_samples": 1,
        "oracle": {"maximum_selected_new_hard_clipping_fraction": 0.005},
        "automatic_gates": {
            "minimum_challenger_selection_fraction": 0.5,
            "minimum_mean_style_gain_delta_e76": 1.0,
            "minimum_gold_mean_style_gain_delta_e76": 1.0,
            "minimum_median_style_gain_delta_e76": 1.0,
            "maximum_selected_new_hard_clipping_fraction": 0.005,
        },
        "claim_ceiling": "test only",
    }
    return config, report_hash


def test_oracle_hard_selects_challenger_or_baseline(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    report, manifest = evaluate_oracle(
        root=tmp_path,
        config=config,
        config_sha256="config",
        software_commit="commit",
    )
    assert report["automatic_checks_passed"]
    assert report["aggregates"]["challenger_selected_count"] == 1
    assert report["aggregates"]["baseline_fallback_count"] == 1
    assert report["aggregates"]["mean_style_gain_delta_e76"] == 1.5
    assert [row["selected_strength"] for row in manifest["records"]] == [
        0.65,
        0.5,
    ]


def test_oracle_fails_closed_on_parent_hash_drift(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    config["parent"]["automatic_report_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(DensityStrengthOracleError, match="parent hash mismatch"):
        evaluate_oracle(
            root=tmp_path,
            config=config,
            config_sha256="config",
            software_commit="commit",
        )


def test_oracle_fails_closed_on_selected_output_drift(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    (tmp_path / "parent/render/cyan__s65/a.png").write_bytes(b"drift")
    with pytest.raises(DensityStrengthOracleError, match="selected output hash"):
        evaluate_oracle(
            root=tmp_path,
            config=config,
            config_sha256="config",
            software_commit="commit",
        )


def test_oracle_fails_closed_on_frozen_set_drift(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    config["parent"]["frozen_set_sha256"] = "other"  # type: ignore[index]
    with pytest.raises(DensityStrengthOracleError, match="frozen-set hash drift"):
        evaluate_oracle(
            root=tmp_path,
            config=config,
            config_sha256="config",
            software_commit="commit",
        )


def test_runner_normalizes_relative_paths() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_u5_r2l0_density_strength_oracle.py"
    ).read_text(encoding="utf-8")
    assert "if args.config.is_absolute()" in source
    assert "if args.output.is_absolute()" in source
    assert "if args.selected_manifest.is_absolute()" in source
