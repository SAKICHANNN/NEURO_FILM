from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rf3_d13_darkroom_three_stock_baseline_source_v1.json"

AUDITED_FILES = (
    "README.md",
    "pyproject.toml",
    "data/color_stocks.py",
    "tools/costyle_data.json",
    "tools/costyle_report.txt",
    "tools/parse_costyles.py",
)
STOCK_ROWS = {
    "fujifilm_velvia_50": ("Fuji Velvia 50", "Slide / Fuji Velvia 50"),
    "kodak_portra_400": ("Kodak Portra 400", "Neg / Kodak Portra 400"),
    "kodak_ektar_100": ("Kodak Ektar 100", "Neg / Kodak Ektar 100"),
}
PROVENANCE_KEYS = {
    "source_url",
    "source_document",
    "source_page",
    "extraction_method",
    "license",
    "license_url",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(source_root: Path, *args: str) -> str:
    physical = source_root.resolve().as_posix()
    result = subprocess.run(
        ["git", "-c", f"safe.directory={physical}", "-C", str(source_root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def audit(source_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source_root = source_root.resolve()
    head = _git(source_root, "rev-parse", "HEAD")
    tracked = set(_git(source_root, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    blobs = {name: _sha256(source_root / name) for name in AUDITED_FILES}

    readme = (source_root / "README.md").read_text(encoding="utf-8")
    pyproject = (source_root / "pyproject.toml").read_text(encoding="utf-8")
    color_source = (source_root / "data/color_stocks.py").read_text(encoding="utf-8")
    costyles = json.loads((source_root / "tools/costyle_data.json").read_text(encoding="utf-8"))

    root_license_candidates = sorted(
        name
        for name in tracked
        if "/" not in name
        and (name.upper() in {"LICENSE", "COPYING", "NOTICE"} or name.upper().startswith("LICENSE."))
    )
    stock_facts: list[dict[str, Any]] = []
    for stock_id, (costyle_name, operator_key) in STOCK_ROWS.items():
        row = costyles.get(costyle_name)
        row_keys = sorted(row) if isinstance(row, dict) else []
        stock_facts.append(
            {
                "stock_id": stock_id,
                "costyle_row": costyle_name,
                "costyle_row_present": isinstance(row, dict),
                "operator_key": operator_key,
                "operator_present": f'"{operator_key}"' in color_source,
                "provenance_keys_present": sorted(PROVENANCE_KEYS.intersection(row_keys)),
                "all_required_provenance_present": PROVENANCE_KEYS.issubset(row_keys),
            }
        )

    gates = {
        "exact_commit_and_blobs": head == config["commit"] and all(name in tracked for name in AUDITED_FILES),
        "root_license_covers_code_and_operator_data": bool(root_license_candidates),
        "three_exact_machine_readable_stock_operators": all(
            row["costyle_row_present"] and row["operator_present"] for row in stock_facts
        ),
        "per_stock_primary_source_and_extraction_provenance": all(
            row["all_required_provenance_present"] for row in stock_facts
        ),
        "headless_deterministic_no_network_execution": True,
        "no_unresolved_third_party_data_rights": bool(root_license_candidates)
        and all(row["all_required_provenance_present"] for row in stock_facts),
        "materially_distinct_from_closed_spectral_film_lut_route": (
            "Capture One" in readme
            and "tools/costyle_data.json" in tracked
            and "data/color_stocks.py" in tracked
        ),
    }
    status = (
        "PASS_SOURCE_ONLY"
        if all(gates.values())
        else "FAIL_CLOSED_MISSING_ROOT_LICENSE_AND_PER_STOCK_PROVENANCE"
    )
    scientific = {
        "protocol": config["schema"],
        "status": status,
        "source": {
            "repository": config["repository"],
            "commit": head,
            "tracked_path_count": len(tracked),
            "audited_blob_sha256": blobs,
            "root_license_candidates": root_license_candidates,
            "pyproject_declares_license_file": 'license = { file = "LICENSE" }' in pyproject,
            "readme_claims_mit": "## License\n\nMIT" in readme,
            "readme_claims_capture_one_data": "586 .costyle files parsed" in readme,
            "costyle_row_count": len(costyles),
        },
        "stock_facts": stock_facts,
        "gates": gates,
        "observations": {
            "image_or_film_scan_reads": 0,
            "operator_pixel_executions": 0,
            "network_reads_beyond_git_source": 0,
        },
        "decision": (
            config["decision_if_pass"] if all(gates.values()) else config["decision_if_fail"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "neuro-film.rf3-d13-darkroom-three-stock-baseline-source-result.v1",
        "scientific": scientific,
        "stable_identity": "sha256:" + hashlib.sha256(canonical).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = audit(args.source_root, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result["scientific"]["gates"], sort_keys=True))
    print(result["scientific"]["status"])


if __name__ == "__main__":
    main()
