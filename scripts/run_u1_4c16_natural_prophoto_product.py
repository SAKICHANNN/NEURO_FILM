#!/usr/bin/env python3
"""Execute the strict natural-ProPhoto Rec.2020 Velvia product contract once."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import atomic_write_json
from src.inference.romm_rec2020_velvia import (
    render_supported_prophoto_velvia_rec2020,
)

CONTRACT_SCHEMA = "neuro-film.u1-4c16-natural-prophoto-rec2020-product-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c16-natural-prophoto-rec2020-product-report.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_file(root: Path, value: str, expected_sha256: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("contract path must be repository-relative")
    path = root / relative
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError(f"contract binding drift: {value}")
    return path


def load_contract(path: Path, *, root: Path = ROOT) -> tuple[dict[str, Any], str]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("experiment_id") != "U1.4C16":
        raise ValueError("unsupported U1.4C16 contract")
    _relative_file(
        root,
        payload["source"]["manifest"],
        payload["source"]["manifest_sha256"],
    )
    _relative_file(
        root,
        payload["product"]["profile"],
        payload["product"]["profile_sha256"],
    )
    _relative_file(
        root,
        payload["product"]["renderer"],
        payload["product"]["renderer_sha256"],
    )
    return payload, _sha256(path)


def _validate_manifest(contract: dict[str, Any], *, root: Path) -> list[dict[str, Any]]:
    source = contract["source"]
    manifest_path = root / source["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    if (
        len(rows) != source["expected_rows"]
        or len({row.get("id") for row in rows}) != len(rows)
        or manifest.get("allowed_use") != source["required_allowed_use"]
        or manifest.get("rights_scope") != source["required_rights_scope"]
        or manifest.get("embedded_icc_sha256")
        != source["required_embedded_icc_sha256"]
    ):
        raise ValueError("U1.4C16 source manifest contract drift")
    for row in rows:
        path = root / row["path"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(f"U1.4C16 source drift: {row.get('id')}")
    return rows


def _inventory(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "report.json"
    ]


def run(contract_path: Path, output_dir: Path, *, root: Path = ROOT) -> dict[str, Any]:
    contract, contract_sha256 = load_contract(contract_path, root=root)
    rows = _validate_manifest(contract, root=root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory must be create-only")
    output_dir.mkdir(parents=True)
    try:
        rendered: list[dict[str, Any]] = []
        profile_path = root / contract["product"]["profile"]
        for row in rows:
            output_path = output_dir / "renders" / f"{row['id']}.png"
            receipt_path = output_dir / "receipts" / f"{row['id']}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt = render_supported_prophoto_velvia_rec2020(
                root / row["path"],
                output_path,
                profile_path=profile_path,
                root=root,
            )
            atomic_write_json(receipt_path, receipt)
            rendered.append(
                {
                    "id": row["id"],
                    "make": row["make"],
                    "model": row["model"],
                    "input_sha256": row["sha256"],
                    "output_sha256": receipt["output"]["sha256"],
                    "receipt_sha256": _sha256(receipt_path),
                    "embedded_icc_sha256": receipt["input"]["embedded_icc_sha256"],
                    "width": receipt["input"]["width"],
                    "height": receipt["input"]["height"],
                    "mapped_pixel_fraction": receipt["ingress"]["diagnostics"][
                        "mapped_pixel_fraction"
                    ],
                    "minimum_chroma_scale": receipt["ingress"]["diagnostics"][
                        "minimum_chroma_scale"
                    ],
                    "median_residual_scale": receipt["look"]["median_residual_scale"],
                    "fraction_residual_scale_below_0p5": receipt["look"][
                        "fraction_residual_scale_below_0p5"
                    ],
                    "exact_sample_readback": receipt["output"][
                        "exact_sample_readback"
                    ],
                }
            )
        gates = contract["gates"]
        gate_results = {
            "all_sources_rendered": len(rendered) == source_count(contract),
            "all_exact_sample_readback": all(
                row["exact_sample_readback"] for row in rendered
            ),
            "all_median_residual_scale_one": all(
                row["median_residual_scale"] == 1.0 for row in rendered
            ),
            "maximum_fraction_residual_scale_below_0p5": max(
                row["fraction_residual_scale_below_0p5"] for row in rendered
            )
            <= gates["maximum_fraction_residual_scale_below_0p5"],
        }
        inventory = _inventory(output_dir)
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "experiment_id": "U1.4C16",
            "contract_sha256": contract_sha256,
            "implementation_commit": contract["implementation_commit"],
            "profile_sha256": contract["product"]["profile_sha256"],
            "source_manifest_sha256": contract["source"]["manifest_sha256"],
            "rows": rendered,
            "ordered_output_inventory": inventory,
            "gate_results": gate_results,
            "automatic_pass": all(gate_results.values()),
            "decision": (
                contract["decision_if_pass"]
                if all(gate_results.values())
                else contract["decision_if_fail"]
            ),
            "production_default_changed": False,
            "claim_ceiling": contract["claim_ceiling"],
        }
        stable_payload = dict(report)
        report["stable_evidence_id"] = hashlib.sha256(
            json.dumps(
                stable_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        atomic_write_json(output_dir / "report.json", report)
        return report
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def source_count(contract: dict[str, Any]) -> int:
    return int(contract["source"]["expected_rows"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u1_4c16_natural_prophoto_rec2020_product_v1.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.contract, args.output_dir)
    print(json.dumps({"automatic_pass": report["automatic_pass"], "stable_evidence_id": report["stable_evidence_id"]}, sort_keys=True))
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
