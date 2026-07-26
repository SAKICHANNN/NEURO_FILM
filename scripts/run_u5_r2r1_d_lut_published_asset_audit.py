#!/usr/bin/env python
"""Run the frozen U5.R2R1 published D-LUT structural audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.d_lut_published_assets import (  # noqa: E402
    PublishedDLUTError,
    analyze_asset,
    canonical_lut_manifest,
    parse_cube,
    sha256_file,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def run_audit(
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    source = config["external_source"]
    asset_root = ROOT / str(source["asset_root"])
    if not asset_root.is_dir():
        raise PublishedDLUTError(f"asset root does not exist: {asset_root}")
    revision = _git_revision(asset_root)
    if revision != str(source["revision"]):
        raise PublishedDLUTError("external source revision mismatch")

    model_path = asset_root / str(source["model_path"])
    if (
        model_path.stat().st_size != int(source["model_bytes"])
        or sha256_file(model_path) != str(source["model_sha256"])
    ):
        raise PublishedDLUTError("published score checkpoint mismatch")

    lut_paths = list(asset_root.glob(str(source["lut_glob"])))
    manifest, manifest_sha256 = canonical_lut_manifest(lut_paths)
    if len(manifest) != int(source["lut_count"]):
        raise PublishedDLUTError("published LUT count mismatch")
    if sum(int(row["bytes"]) for row in manifest) != int(source["lut_total_bytes"]):
        raise PublishedDLUTError("published LUT byte total mismatch")
    if manifest_sha256 != str(source["lut_manifest_sha256"]):
        raise PublishedDLUTError("published LUT manifest hash mismatch")

    by_step: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path, row in zip(
        sorted(lut_paths, key=lambda value: int(value.stem.removeprefix("LUT_"))),
        manifest,
        strict=True,
    ):
        step = int(path.stem.removeprefix("LUT_"))
        by_step[step] = (path, row)
    step_spec = config["audit"]["steps"]
    expected_steps = list(
        range(int(step_spec["minimum"]), int(step_spec["maximum"]) + 1)
    )
    if bool(step_spec["required_contiguous"]) and sorted(by_step) != expected_steps:
        raise PublishedDLUTError("published LUT steps are not contiguous")

    reports = []
    for step in expected_steps:
        path, row = by_step[step]
        asset = parse_cube(path)
        if (
            asset.domain_min.tolist() != [0.0, 0.0, 0.0]
            or asset.domain_max.tolist() != [1.0, 1.0, 1.0]
        ):
            raise PublishedDLUTError(f"unexpected LUT domain at step {step}")
        diagnostics = analyze_asset(
            asset,
            config["gates"],
            config["audit"]["trilinear_subcell_axis"],
        )
        reports.append(
            {
                "step": step,
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                **diagnostics,
            }
        )

    identity_error = float(reports[0]["identity_maximum_absolute_error"])
    identity_pass = identity_error <= float(
        config["gates"]["identity_step_maximum_absolute_error"]
    )
    survivors = [
        int(row["step"])
        for row in reports
        if int(row["step"]) > 0 and bool(row["structural_pass"])
    ]
    survivor_pass = bool(survivors)
    if not bool(
        config["gates"]["require_at_least_one_nonidentity_structural_survivor"]
    ):
        survivor_pass = True
    all_gates_pass = bool(identity_pass and survivor_pass)
    branch = (
        "one_or_more_survivors"
        if survivors
        else "no_nonidentity_survivor"
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "epistemic_status": config["epistemic_status"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "external_source": {
            "revision": revision,
            "license": source["license"],
            "model_path": source["model_path"],
            "model_bytes": model_path.stat().st_size,
            "model_sha256": sha256_file(model_path),
            "lut_count": len(manifest),
            "lut_total_bytes": sum(int(row["bytes"]) for row in manifest),
            "lut_manifest_sha256": manifest_sha256,
        },
        "audit": {
            "trilinear_subcell_axis": config["audit"][
                "trilinear_subcell_axis"
            ],
            "steps": reports,
        },
        "gates": config["gates"],
        "gate_results": {
            "identity_step": identity_pass,
            "nonidentity_structural_survivor": survivor_pass,
            "all": all_gates_pass,
        },
        "nonidentity_structural_survivors": survivors,
        "decision_branch": branch,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2r1_d_lut_published_asset_audit_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--software-commit",
        default=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    report = run_audit(
        config,
        config_sha256=_sha256_bytes(config_bytes),
        software_commit=str(args.software_commit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    args.output.write_bytes(payload)
    print(json.dumps(report["gate_results"], sort_keys=True))
    print(f"decision_branch={report['decision_branch']}")
    print(f"report_sha256={_sha256_bytes(payload)}")


if __name__ == "__main__":
    main()
