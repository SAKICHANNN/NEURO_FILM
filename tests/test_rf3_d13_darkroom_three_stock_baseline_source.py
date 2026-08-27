from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.audit_rf3_d13_darkroom_three_stock_baseline_source import audit

STOCKS = ("Fuji Velvia 50", "Kodak Portra 400", "Kodak Ektar 100")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _source_fixture(root: Path, *, complete_rights: bool) -> dict[str, object]:
    (root / "data").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "README.md").write_text(
        "Capture One\n586 .costyle files parsed\n## License\n\nMIT\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        'license = { file = "LICENSE" }\n', encoding="utf-8"
    )
    (root / "data/color_stocks.py").write_text(
        '"Slide / Fuji Velvia 50"\n'
        '"Neg / Kodak Portra 400"\n'
        '"Neg / Kodak Ektar 100"',
        encoding="utf-8",
    )
    provenance = {
        "source_url": "https://example.invalid/source",
        "source_document": "doc",
        "source_page": "1",
        "extraction_method": "explicit",
        "license": "MIT",
        "license_url": "https://example.invalid/license",
    }
    rows = {name: (provenance if complete_rights else {"brand": "test"}) for name in STOCKS}
    (root / "tools/costyle_data.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    (root / "tools/costyle_report.txt").write_text("report\n", encoding="utf-8")
    (root / "tools/parse_costyles.py").write_text("# parser\n", encoding="utf-8")
    if complete_rights:
        (root / "LICENSE").write_text("MIT\n", encoding="utf-8")

    _git(root, "init")
    _git(root, "config", "user.name", "RF3 D13 Test")
    _git(root, "config", "user.email", "rf3-d13@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    head = _git(root, "rev-parse", "HEAD")
    return {
        "schema": "test.rf3-d13",
        "repository": "https://example.invalid/repo",
        "commit": head,
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test only",
    }


def test_rf3_d13_fails_missing_root_license_and_row_provenance(tmp_path: Path) -> None:
    config = _source_fixture(tmp_path, complete_rights=False)
    scientific = audit(tmp_path, config)["scientific"]

    assert scientific["status"] == (
        "FAIL_CLOSED_MISSING_ROOT_LICENSE_AND_PER_STOCK_PROVENANCE"
    )
    assert scientific["gates"]["three_exact_machine_readable_stock_operators"] is True
    assert scientific["gates"]["root_license_covers_code_and_operator_data"] is False
    assert scientific["gates"]["per_stock_primary_source_and_extraction_provenance"] is False
    assert scientific["observations"]["operator_pixel_executions"] == 0


def test_rf3_d13_admits_only_complete_rights_and_provenance(tmp_path: Path) -> None:
    config = _source_fixture(tmp_path, complete_rights=True)
    scientific = audit(tmp_path, config)["scientific"]

    assert scientific["status"] == "PASS_SOURCE_ONLY"
    assert all(scientific["gates"].values())
    assert scientific["decision"] == "PASS"
