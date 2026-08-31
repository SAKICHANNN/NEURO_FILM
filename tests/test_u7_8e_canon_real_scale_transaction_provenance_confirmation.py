from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.audit_u7_8e_canon_real_scale_transaction_provenance_confirmation import (
    _controller_report,
    _cross_run_report,
    _materialize_historical_runtime,
    _stable_payload,
    _validate_config,
    _validate_recorded_paths,
    _validate_runtime_root_asset_ledger,
)
from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u7_8e_canon_real_scale_transaction_provenance_confirmation_v1.json"
)
CONTRACT = (
    ROOT
    / "docs/planning/U7_8E_CANON_REAL_SCALE_TRANSACTION_PROVENANCE_CONFIRMATION_CONTRACT.md"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _worker() -> dict:
    commit = "1" * 40
    return {
        "schema_version": (
            "neuro-film.u7-8e-canon-real-scale-transaction-provenance-"
            "confirmation-result.v1"
        ),
        "node_id": "U7.8E",
        "requested_order": "forward",
        "implementation_commit": commit,
        "config_git_blob": "2" * 40,
        "execution_lock_sha256": "3" * 64,
        "source_identity": [{"job_id": "a", "sha256": "4" * 64}],
        "source_set_identity": "5" * 64,
        "historical_materialization": {"statistics": {"runtime_sha256": "6" * 64}},
        "transaction": {
            "batch_id": "7" * 64,
            "receipt_sha256": "8" * 64,
            "receipt_bytes": 100,
            "rows": [],
        },
        "software_provenance": {
            "transaction_start_snapshot_count": 1,
            "prepublication_recheck_count": 1,
            "total_observation_count": 2,
            "observed_commits": [commit, commit],
            "child_snapshot_count": 6,
            "all_child_snapshots_equal": True,
            "recipe_commit_count": 18,
            "all_recipe_commits_equal": True,
        },
        "decode_counts": {"a": 1},
        "per_job_wall_seconds": [{"job_id": "a", "wall_seconds": 1.0}],
        "controls": {"late_foreign_complete_batch_preserved": True},
        "gate_results": {
            "all_six_sources_exact_and_immutable": True,
            "historical_materialization_exact": True,
            "successful_decode_calls_exactly_once_per_input": True,
            "six_children_eighteen_outputs_and_recipes": True,
            "transaction_commit_snapshot_recheck_total_1_1_2": True,
            "all_children_and_recipes_bind_implementation_commit": True,
            "all_recorded_paths_equal_fixed_transaction_paths": True,
            "injected_second_child_publishes_nothing": True,
            "failure_controls_have_no_extra_decodes": True,
            "complete_batch_late_foreign_preserved": True,
            "owned_stage_and_file_residue_count": 0,
            "network_requests": 0,
        },
        "claim_ceiling": "bounded",
    }


def test_config_binds_exact_u7_8c_history_sources_and_render_parameters() -> None:
    config = _config()
    _validate_config(config)
    assert CONTRACT.is_file()
    assert config["node_id"] == "U7.8E"
    assert len(config["sources"]) == 6
    assert config["render"] == {
        "look_amount": 1.0,
        "seed": 31,
        "tile_size": 256,
        "tile_workers": 1,
        "png_compression": 0,
        "styles": ["velvia_50", "portra_400", "ektar_100"],
    }
    for row in config["sources"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256_file(path) == row["sha256"]


def test_statistics_materialization_reproduces_frozen_windows_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import (
        audit_u7_8e_canon_real_scale_transaction_provenance_confirmation as module,
    )

    config = _config()
    runtime = json.loads(json.dumps(config))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(module, "SCRATCH", scratch)
    result = _materialize_historical_runtime(config, runtime)
    stats = scratch / "historical-runtime/film_color_stats.json"
    assert stats.stat().st_size == 5271
    assert sha256_file(stats) == (
        "1a45e90765e577edd4ca1773fc8c6cf98fd09b8afbdddfa519b1f097c5dd7c04"
    )
    assert result["statistics"] == {
        "git_blob": "c12ccafcea9ee8a283badba5b473d29f3d54e162",
        "git_lf_bytes": 5031,
        "git_lf_sha256": (
            "064b1b5f1b75671ecfb6ab4bf851c1df6d03f7bb394b1545af2125e070e7254f"
        ),
        "runtime_bytes": 5271,
        "runtime_sha256": (
            "1a45e90765e577edd4ca1773fc8c6cf98fd09b8afbdddfa519b1f097c5dd7c04"
        ),
    }
    assert b"\r\n" in stats.read_bytes()
    assert runtime["render"]["statistics_path"] == str(stats.resolve())
    ledger = _validate_runtime_root_asset_ledger(config, result)
    assert ledger["core_autocrlf"] is True
    assert ledger["scratch_statistics_equals_runtime_root"] is True
    assert ledger["assets"]["style_statistics"] == {
        "bytes": 5271,
        "sha256": ("1a45e90765e577edd4ca1773fc8c6cf98fd09b8afbdddfa519b1f097c5dd7c04"),
    }


def test_historical_blob_bindings_resolve_from_one_immutable_commit() -> None:
    config = _config()
    for row in config["bindings"].values():
        commit = row.get("commit", config["historical_commit"])
        payload = subprocess.check_output(
            ["git", "show", f"{commit}:{row['path']}"], cwd=ROOT
        )
        expected_bytes = row.get("bytes", row.get("git_lf_bytes"))
        expected_sha = row.get("sha256", row.get("git_lf_sha256"))
        assert len(payload) == expected_bytes
        assert hashlib.sha256(payload).hexdigest() == expected_sha
        assert (
            subprocess.check_output(
                ["git", "rev-parse", f"{commit}:{row['path']}"],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
            ).strip()
            == row["git_blob"]
        )


def test_recorded_paths_are_exactly_the_fixed_source_and_transaction_roots(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.cr2"
    source.write_bytes(b"source")
    output_root = tmp_path / "published"
    child_root = output_root / "job-a"
    child_root.mkdir(parents=True)
    rows = []
    for style in ("velvia_50", "portra_400", "ektar_100"):
        output = child_root / f"{style}.png"
        recipe = child_root / f"{style}.recipe.json"
        output.write_bytes(b"png")
        recipe.write_text(
            json.dumps(
                {
                    "input": {"path": str(source.resolve())},
                    "output": {"path": str(output.resolve())},
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "style_id": style,
                "output_path": str(output.resolve()),
                "recipe_path": str(recipe.resolve()),
            }
        )
    (child_root / "batch.json").write_text(
        json.dumps({"input_path": str(source.resolve()), "rows": rows}),
        encoding="utf-8",
    )
    result = _validate_recorded_paths(
        output_root,
        {"sources": [{"job_id": "job-a", "path": str(source.resolve())}]},
    )
    assert result == {
        "recipe_path_count": 3,
        "child_manifest_path_count": 1,
        "all_paths_equal_fixed_source_and_transaction_roots": True,
    }


def test_stable_payload_excludes_order_timing_and_resource_measurements() -> None:
    first = _worker()
    second = _worker()
    second["requested_order"] = "reverse"
    second["per_job_wall_seconds"] = [{"job_id": "a", "wall_seconds": 99.0}]
    second["resource_measurement"] = {
        "controller_wall_seconds": 100.0,
        "process_tree_peak_rss_bytes": 200,
    }
    assert _stable_payload(first) == _stable_payload(second)


def test_controller_requires_stable_and_resource_gates() -> None:
    passed = _controller_report(_worker(), wall_seconds=1199.0, peak_rss_bytes=1)
    assert passed["decision"] == (
        "PASS_PRIVATE_U7_8E_CANON_REAL_SCALE_TRANSACTION_PROVENANCE_CONFIRMATION"
    )
    assert passed["scientific_identity"]

    failed = _controller_report(
        _worker(), wall_seconds=1200.01, peak_rss_bytes=4 * 1024**3 + 1
    )
    assert failed["decision"].startswith("FAIL_CLOSED_")
    assert failed["resource_gate_results"] == {
        "controller_wall_seconds": False,
        "process_tree_peak_rss_bytes": False,
    }


def test_false_boolean_never_passes_as_numeric_zero() -> None:
    worker = _worker()
    worker["gate_results"]["all_six_sources_exact_and_immutable"] = False
    report = _controller_report(worker, wall_seconds=1.0, peak_rss_bytes=1)
    assert report["decision"].startswith("FAIL_CLOSED_")


def test_cross_run_comparison_requires_every_transaction_identity() -> None:
    forward = _controller_report(_worker(), wall_seconds=1.0, peak_rss_bytes=1)
    reverse_worker = _worker()
    reverse_worker["requested_order"] = "reverse"
    reverse = _controller_report(reverse_worker, wall_seconds=2.0, peak_rss_bytes=2)
    comparison = _cross_run_report(forward, reverse)
    assert comparison["decision"].startswith("PASS_PRIVATE_")
    assert all(comparison["gate_results"].values())

    reverse["transaction"]["receipt_sha256"] = "9" * 64
    failed = _cross_run_report(forward, reverse)
    assert failed["decision"].startswith("FAIL_CLOSED_")
    assert not failed["gate_results"][
        "all_rgb16_recipe_manifest_receipt_and_batch_identities_exact"
    ]
