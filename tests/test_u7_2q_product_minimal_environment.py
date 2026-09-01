from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_2q_product_minimal_environment_v1.json"
REQUIREMENTS = ROOT / "requirements-product.txt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requirements() -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line
        name, version = line.split("==", maxsplit=1)
        canonical = name.lower().replace("_", "-")
        assert canonical not in result
        assert canonical and version
        result[canonical] = version
    return result


def test_product_requirements_are_exactly_the_frozen_runtime_set() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert _requirements() == config["required_distributions"]
    assert set(_requirements()).isdisjoint(config["forbidden_distribution_roots"])
    assert config["python"]["required_major_minor"] == "3.12"
    assert config["environment_count"] == 2
    assert config["binary_only_install"] is True
    assert config["require_pip_check"] is True


def test_product_manifest_binds_the_accepted_parent() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    locks = config["source_locks"]
    assert _sha256(ROOT / "scripts/render_film.py") == locks["render_film_sha256"]
    assert (
        _sha256(
            ROOT
            / "docs/evidence/U7_2P_PRODUCT_CLI_RESEARCH_DEPENDENCY_ISOLATION_RESULT.json"
        )
        == locks["u7_2p_evidence_sha256"]
    )
    report = (
        ROOT
        / "outputs/eval/u7_2p_product_cli_research_dependency_isolation_v1/formal.json"
    )
    assert report.stat().st_size == locks["u7_2p_report_bytes"]
    assert _sha256(report) == locks["u7_2p_report_sha256"]
