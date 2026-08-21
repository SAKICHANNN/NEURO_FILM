from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_p87_ultrahdr_absolute_rec2020_match_view_v1 import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p87_ultrahdr_absolute_rec2020_match_view_v1.json"


def _isolated_contract(tmp_path: Path) -> tuple[Path, Path]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = {
        "canonical_view": {
            "profile_id": config["producer"]["external_profile_id"],
            "reference_white_nits": config["consumer"]["reference_white_nits"],
        },
        "experiment_id": config["producer"]["experiment_id"],
        "formal_execution": {"stable_identity": "1" * 64},
        "official_source": {
            "commit": config["official_decoder"]["commit"],
            "decoder_version": config["official_decoder"]["version"],
        },
        "producer_commit": config["producer"]["commit"],
        "status": "PASS_PRIVATE_ULTRAHDR_MATCH_VIEW_CANONICALIZATION",
    }
    evidence_path = tmp_path / "producer.json"
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True),
        encoding="utf-8",
    )
    config["producer"]["evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return config_path, evidence_path


def test_p87_audit_is_order_exact_and_passes_all_gates(tmp_path: Path) -> None:
    config, evidence = _isolated_contract(tmp_path)
    forward = run(config, evidence, reverse=False)
    reverse = run(config, evidence, reverse=True)
    assert forward == reverse
    assert forward["status"].startswith("PASS_")
    assert all(forward["gates"].values())
    assert len(forward["records"]) == 3
    assert len(forward["negative_controls"]) == 7
