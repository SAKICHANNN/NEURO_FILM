from __future__ import annotations

import json
from pathlib import Path
import re

from src.eval.rawpixls_confirmation_preflight import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk6s_bounded_opponent_fresh_source_v1.json"
DECISION = (
    ROOT
    / "configs/u5_r2bk6s_bounded_opponent_fresh_source_decision_v1.json"
)


def _repository_ids(payload: object) -> set[int]:
    return {
        int(value)
        for value in re.findall(
            r"raw\.pixls\.us/getfile\.php/(\d+)/(?:nice|exif)/",
            json.dumps(payload),
        )
    }


def test_bk6s_contract_is_frozen_before_pixel_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    rows = config["candidates"]
    assert config["status"] == "contract_frozen_acquisition_ready"
    assert len(rows) == 12
    assert len({row["make"].casefold() for row in rows}) == 12
    assert len({row["repository_id"] for row in rows}) == 12
    assert sum(row["bytes_reported_mb"] for row in rows) == 174.39
    assert config["maximum_download_bytes"] == 190 * 1024 * 1024
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["next_leaf"]["strength_retuning_allowed"]
    assert not config["next_leaf"]["router_training_allowed"]


def test_bk6s_sources_are_disjoint_from_every_prior_tracked_pool() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    current_ids = {row["repository_id"] for row in config["candidates"]}
    current_hashes = {row["sha256"] for row in config["candidates"]}
    current_models = {
        (row["make"].casefold(), row["model"].casefold())
        for row in config["candidates"]
    }
    prior_ids: set[int] = set()
    prior_hashes: set[str] = set()
    prior_models: set[tuple[str, str]] = set()
    for path in ROOT.glob("configs/*.json"):
        if path == CONFIG:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        prior_ids.update(_repository_ids(payload))
        rows = payload.get("candidates", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("sha256", "raw_sha256"):
                value = row.get(key)
                if isinstance(value, str) and len(value) == 64:
                    prior_hashes.add(value)
            if "make" in row and "model" in row:
                prior_models.add(
                    (row["make"].casefold(), row["model"].casefold())
                )
    legacy = json.loads(
        (
            ROOT
            / "outputs/color_baseline/velvia50_rawpixls20_s0p58_gamutsafe/manifest.json"
        ).read_text(encoding="utf-8")
    )
    prior_ids.update(_repository_ids(legacy))
    for row in legacy:
        if "make" in row and "model" in row:
            prior_models.add(
                (row["make"].casefold(), row["model"].casefold())
            )
        if isinstance(row.get("raw_sha256"), str):
            prior_hashes.add(row["raw_sha256"])
    assert current_ids.isdisjoint(prior_ids)
    assert current_hashes.isdisjoint(prior_hashes)
    assert current_models.isdisjoint(prior_models)


def test_bk6s_selection_excludes_nonphotographic_metadata() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forbidden = re.compile(r"\b(scanner|chart|flat|target|film|buggy)\b", re.I)
    for row in config["candidates"]:
        text = " ".join(
            str(row[key]) for key in ("make", "model", "mode", "filename")
        )
        assert forbidden.search(text) is None


def test_bk6s_decision_binds_repeatable_eligible_evidence() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    result = decision["result"]
    evidence = decision["evidence"]
    assert decision["status"] == "source_eligible_fixed_comparison_open"
    assert decision["config"]["sha256"] == (
        "ce80b257495d22859b12fec00cde224dba3dcfdba8ce5affba8d8a48c2d57934"
    )
    assert evidence["manifest"]["sha256"] == (
        "444f162f22e6bc40940eb82557a11ed2b1fa67e6736efb1362df7ef1069d5978"
    )
    assert evidence["automatic_report"]["sha256"] == (
        "6b5bb3167e5803d899639f42e427959788bdb660f6b5c123f761a64e55ec9f9b"
    )
    assert evidence["repeat_manifest_sha256_exact"]
    assert evidence["repeat_report_sha256_exact"]
    assert evidence["repeat_contact_sheet_sha256_exact"]
    assert result["eligible_rows"] == 12
    assert result["eligible_camera_makes"] == 12
    assert result["automatic_pass"]
    assert result["visual_pass"]
    assert result["confirmed_severe_source_artifact_count"] == 0
    assert result["fixed_decode_failures"] == []
    assert result["fixed_orientation_stress_rows"] == ["samsung_sm_g950u"]
    assert not decision["training_allowed"]
    assert not decision["operator_fitting_allowed"]
