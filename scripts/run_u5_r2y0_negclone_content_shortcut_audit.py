#!/usr/bin/env python
"""Run the frozen U5.R2Y0 NegClone source and shortcut audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.negclone_content_shortcut import (  # noqa: E402
    evaluate_shortcut_gates,
    inspect_source_contract,
    run_exact_module_probes,
    validate_sdist,
    validate_source_files,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _software_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _resolve_project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("NegClone audit path escapes project root")
    return path


def _load_pinned_fingerprint_module(source_root: Path) -> Any:
    root = source_root.resolve()
    sys.path.insert(0, str(root))
    try:
        for name in tuple(sys.modules):
            if name == "negclone" or name.startswith("negclone."):
                del sys.modules[name]
        module = importlib.import_module("negclone.fingerprint")
    finally:
        if sys.path[0] == str(root):
            sys.path.pop(0)
    module_path = Path(inspect.getfile(module)).resolve()
    expected = (root / "negclone" / "fingerprint.py").resolve()
    if module_path != expected:
        raise RuntimeError("loaded NegClone module is not the pinned source")
    return module


def run_audit(
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
    fingerprint_module: Any | None = None,
) -> dict[str, Any]:
    if config.get("status") != "implemented_frozen_pending_formal_run":
        raise ValueError("unexpected U5.R2Y0 contract status")
    source = config["source"]
    archive = _resolve_project_path(source["sdist_path"])
    source_root = _resolve_project_path(source["extracted_root"])
    members = validate_sdist(
        archive,
        expected_sha256=source["sdist_sha256"],
        expected_member_count=int(source["sdist_member_count"]),
    )
    observed_hashes = validate_source_files(
        source_root, source["source_file_sha256"]
    )
    static = inspect_source_contract(source_root)
    contract = config["static_contract"]
    static_gate_results = {
        name: bool(static.get(name) is True)
        for name in contract["required_true"]
    }
    static_gate_results.update(
        {
            name: bool(static.get(name) is False)
            for name in contract["required_false"]
        }
    )
    module = (
        fingerprint_module
        if fingerprint_module is not None
        else _load_pinned_fingerprint_module(source_root)
    )
    fingerprint_signature = inspect.signature(module.fingerprint_stock)
    probes = run_exact_module_probes(
        module,
        random_seed=int(config["method_scope"]["synthetic_probe_seed"]),
    )
    shortcut_gates = evaluate_shortcut_gates(probes, config["gates"])
    reproducibility_contract = {
        "fingerprint_stock_accepts_seed": (
            "seed" in fingerprint_signature.parameters
        ),
        "global_random_module_used": static[
            "unseeded_image_sampling"
        ]
        and static["unseeded_grain_patch_sampling"],
    }
    all_pass = (
        all(static_gate_results.values())
        and all(shortcut_gates.values())
        and reproducibility_contract["fingerprint_stock_accepts_seed"] is False
        and reproducibility_contract["global_random_module_used"] is True
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "status": "formal_run_pending_exact_repeat",
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "source": {
            **source,
            "sdist_member_count_observed": len(members),
            "source_file_sha256_observed": observed_hashes,
        },
        "static_contract_observed": static,
        "static_gate_results": static_gate_results,
        "reproducibility_contract": reproducibility_contract,
        "synthetic_probes": probes,
        "shortcut_gate_results": shortcut_gates,
        "decision_branch_before_repeat": (
            "content_histogram_texture_shortcut_close"
            if all_pass
            else "source_or_probe_not_reproduced"
        ),
        "external_image_payloads_accessed": 0,
        "network_requests_made": 0,
        "stock_presets_generated": 0,
        "current_stock_pixels_accessed": False,
        "current_stock_training_or_operator_fitting_opened": False,
        "automatic_visual_shortlist_generated": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2y0_negclone_content_shortcut_audit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "eval"
        / "u5_r2y0_negclone_content_shortcut_audit_report.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = run_audit(
        config,
        config_sha256=_sha256_file(config_path),
        software_commit=_software_commit(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(_sha256_file(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
