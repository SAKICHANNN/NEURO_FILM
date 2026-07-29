#!/usr/bin/env python
"""Run the pinned Filmr Velvia-labelled deterministic external control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmr_external_control import (  # noqa: E402
    evaluate_passes,
    export_deterministic_preset,
    render_pass,
    verify_external_runtime,
)
from src.eval.global_frontier import load_frozen_samples, sha256_file  # noqa: E402


CONFIG_SHA256 = "847e8dd0f0ebb7aada99fc749ea76b258c439d54e1a5ac5d4a818ff151b20999"


def _write_json(path: Path, value: object) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2av1_filmr_external_control_v1.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    args = parser.parse_args()
    if (
        args.expected_config_sha256 != CONFIG_SHA256
        or sha256_file(args.config) != CONFIG_SHA256
    ):
        raise ValueError("AV1 config hash mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = ROOT / str(config["output_root"])
    executable = verify_external_runtime(ROOT, config)
    samples = load_frozen_samples(ROOT, config["inputs"])
    first_sample = next(iter(samples.values()))
    preset_path, preset_receipt = export_deterministic_preset(
        root=ROOT,
        config=config,
        executable=executable,
        source_path=ROOT / str(first_sample["source_path"]),
        output_dir=output_root / "preset",
    )
    manifests = []
    manifest_hashes = []
    for pass_index in (1, 2):
        manifest = render_pass(
            root=ROOT,
            config=config,
            executable=executable,
            preset_path=preset_path,
            pass_id=f"pass{pass_index}",
            output_dir=output_root / f"pass{pass_index}",
        )
        manifests.append(manifest)
        manifest_hashes.append(
            _write_json(output_root / f"pass{pass_index}_manifest.json", manifest)
        )
    result = evaluate_passes(
        root=ROOT,
        config=config,
        first=manifests[0],
        second=manifests[1],
    )
    report = {
        "schema_version": "u5-r2av1-filmr-external-control-result-v1",
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "manifest_sha256": manifest_hashes,
        "preset_receipt": preset_receipt,
        "source_audit": config["source_audit"],
        **result,
        "claim_ceiling": config["claim_ceiling"],
    }
    report_hash = _write_json(output_root / "automatic_report.json", report)
    print(
        json.dumps(
            {
                "report": str(
                    (output_root / "automatic_report.json").relative_to(ROOT)
                ),
                "report_sha256": report_hash,
                "decision": result["decision"],
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
