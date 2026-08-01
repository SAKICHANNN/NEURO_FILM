from __future__ import annotations

import csv
import json
from pathlib import Path

from src.eval.fivek_fresh_pair_preflight import (
    select_pair_names,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq0s0_fivek_casebank_preflight_v1.json"


def _source_names(path: Path) -> set[str]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return {
                str(row["source_name"]) for row in csv.DictReader(handle)
            }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["source_name"]) for row in payload["rows"]}


def test_casebank_contract_is_metadata_only_and_bounded() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    assert config["selection"]["pair_count"] == 512
    assert config["network_preflight"]["maximum_requests"] == 1024
    assert config["network_preflight"]["method"] == "HEAD"
    assert config["network_preflight"]["pixel_payload_download_allowed"] is False
    assert config["image_download_allowed"] is False
    assert config["training_allowed"] is False
    assert config["operator_fitting_allowed"] is False


def test_casebank_selection_is_disjoint_from_all_255_prior_rows() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    exclusion = config["exclusion"]
    excluded = _source_names(ROOT / exclusion["retained_manifest"])
    for item in exclusion["additional_manifests"]:
        excluded.update(_source_names(ROOT / item["path"]))
    assert len(excluded) == exclusion["expected_total_unique_names"] == 255

    licensed = [
        row.strip()
        for row in (ROOT / config["official_source"]["license_file_list"])
        .read_text(encoding="utf-8")
        .splitlines()
        if row.strip()
    ]
    selected = select_pair_names(
        licensed_names=licensed,
        retained_names=excluded,
        seed=config["selection"]["seed"],
        count=config["selection"]["pair_count"],
    )
    assert len(selected) == len(set(selected)) == 512
    assert selected == sorted(selected)
    assert not set(selected) & excluded
