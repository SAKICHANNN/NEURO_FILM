"""Execute every tuple advertised by output capabilities v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match import (
    match_reference_files,
    reference_file_output_capabilities_payload,
)
from src.preprocess import (
    SourceProfile,
    WorkingImage,
    inspect_input,
    save_rec2020_16_png,
)

CONFIG = ROOT / "configs" / "reference_match_output_capability_execution_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("node") != "P166":
        raise ValueError("config must be the P166 schema v1 contract")
    cases = config.get("cases")
    if not isinstance(cases, list) or len(cases) != 9:
        raise ValueError("P166 must freeze exactly nine advertised tuples")
    if config.get("repeat_count_per_case") != 2:
        raise ValueError("P166 requires exactly two repeats per case")
    return config


def expected_capability_payload(config: dict[str, Any]) -> dict[str, Any]:
    cases = config["cases"]
    return {
        "schema_id": "neuro-film.reference-file-output-capabilities.v1",
        "capabilities": [
            {
                "working_space": "linear_srgb",
                "transfer_state": "display_linear",
                "output_bit_depth": 8,
                "extensions": [row["extension"] for row in cases[:5]],
                "encoding_profile": "srgb-icc.v1",
            },
            {
                "working_space": "linear_srgb",
                "transfer_state": "display_linear",
                "output_bit_depth": 16,
                "extensions": [row["extension"] for row in cases[5:8]],
                "encoding_profile": "srgb-icc.v1",
            },
            {
                "working_space": "linear_rec2020",
                "transfer_state": "display_linear",
                "output_bit_depth": 16,
                "extensions": [cases[8]["extension"]],
                "encoding_profile": "bt2020-sdr-cicp-1-1-0-1.v1",
            },
        ],
    }


def _rgb(seed: int, width: int, height: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        20,
        221,
        size=(height, width, 3),
        dtype=np.uint8,
    )


def _write_input(
    path: Path,
    *,
    working_space: str,
    seed: int,
    width: int,
    height: int,
) -> None:
    rgb = _rgb(seed, width, height)
    if working_space == "linear_srgb":
        Image.fromarray(rgb, mode="RGB").save(path)
        return
    if working_space == "linear_rec2020":
        pixels = rgb.astype(np.float32) / np.float32(255.0)
        save_rec2020_16_png(
            WorkingImage(
                pixels=pixels,
                working_space="linear_rec2020",
                transfer_state="display_linear",
                source_transfer_state="display_referred",
                source_profile=SourceProfile("cicp", "P166 BT.2020 SDR"),
                hdr_metadata={},
                orientation_applied=True,
                alpha_policy="absent",
                bit_depth_in=16,
                source_path=path,
                warnings=[],
            ),
            path,
        )
        return
    raise ValueError("unsupported P166 input rail")


def _residue(run_dir: Path) -> list[str]:
    markers = (
        "reference-match-stage",
        "reference-match-backup",
        ".tmp",
    )
    return sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in run_dir.rglob("*")
        if path.is_file() and any(marker in path.name for marker in markers)
    )


def _normalize_report(
    payload: dict[str, Any],
    *,
    case_id: str,
    extension: str,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    normalized["reference"]["path"] = f"<INPUT>/{case_id}/reference.png"
    normalized["recipe_file"]["path"] = "<RUN>/recipe.json"
    normalized["outputs"][0]["source_path"] = (
        f"<INPUT>/{case_id}/source.png"
    )
    normalized["outputs"][0]["output_path"] = f"<RUN>/output{extension}"
    return normalized


def run_case(
    config: dict[str, Any],
    case: dict[str, Any],
    repeat: int,
    output_root: Path,
) -> dict[str, Any]:
    case_dir = output_root / "inputs" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    reference = case_dir / "reference.png"
    source = case_dir / "source.png"
    if not reference.exists():
        image = config["image"]
        _write_input(
            reference,
            working_space=case["working_space"],
            seed=int(image["reference_seed"]),
            width=int(image["width"]),
            height=int(image["height"]),
        )
        _write_input(
            source,
            working_space=case["working_space"],
            seed=int(image["source_seed"]),
            width=int(image["width"]),
            height=int(image["height"]),
        )
    run_dir = output_root / "runs" / f"{case['case_id']}-{repeat}"
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / f"output{case['extension']}"
    recipe = run_dir / "recipe.json"
    report = run_dir / "report.json"
    result = match_reference_files(
        reference,
        [source],
        [output],
        recipe_path=recipe,
        report_path=report,
        output_bit_depth=int(case["output_bit_depth"]),
    )
    inspection = inspect_input(output)
    normalized = _normalize_report(
        json.loads(report.read_text(encoding="utf-8")),
        case_id=case["case_id"],
        extension=case["extension"],
    )
    row = {
        "case_id": case["case_id"],
        "repeat": repeat,
        "output_format": inspection.format_name,
        "output_bit_depth": inspection.bit_depth,
        "output_profile_kind": inspection.source_profile.kind,
        "output_sha256": _sha256(output),
        "recipe_sha256": _sha256(recipe),
        "normalized_report_sha256": _canonical_sha256(normalized),
        "safety_action": result.outputs[0].safety.action,
        "residual_artifacts": _residue(run_dir),
    }
    row["automatic_pass"] = bool(
        row["output_format"] == case["expected_format"]
        and row["output_bit_depth"] == case["output_bit_depth"]
        and row["output_profile_kind"] == case["expected_profile_kind"]
        and row["safety_action"] == "identity-fallback"
        and not row["residual_artifacts"]
    )
    return row


def evaluate_runs(
    config: dict[str, Any],
    capability_payload: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    inventory_exact = capability_payload == expected_capability_payload(config)
    case_results: dict[str, Any] = {}
    for case in config["cases"]:
        rows = [row for row in runs if row["case_id"] == case["case_id"]]
        normalized = [
            {key: value for key, value in row.items() if key != "repeat"}
            for row in rows
        ]
        exact_replay = bool(
            len(rows) == config["repeat_count_per_case"]
            and len({_canonical_sha256(row) for row in normalized}) == 1
        )
        case_results[case["case_id"]] = {
            "complete": len(rows) == config["repeat_count_per_case"],
            "exact_replay": exact_replay,
            "automatic_pass": bool(
                inventory_exact
                and exact_replay
                and all(row["automatic_pass"] for row in rows)
            ),
        }
    return {
        "public_capability_inventory_exact": inventory_exact,
        "case_results": case_results,
        "automatic_pass": bool(
            inventory_exact
            and all(row["automatic_pass"] for row in case_results.values())
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def _git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def run_audit(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if output_dir.exists():
        raise FileExistsError("output_dir must be create-only")
    if subprocess.run(
        ["git", "diff", "--quiet", config["candidate_commit"], "--", "src"],
        cwd=ROOT,
        check=False,
    ).returncode != 0:
        raise RuntimeError("current product source differs from P166 candidate")
    output_dir.mkdir(parents=True)
    capability_payload = reference_file_output_capabilities_payload()
    runs = [
        run_case(config, case, repeat, output_dir)
        for case in config["cases"]
        for repeat in range(1, config["repeat_count_per_case"] + 1)
    ]
    report = {
        "schema_version": 1,
        "node": "P166",
        "config_sha256": _sha256(config_path),
        "candidate_commit": config["candidate_commit"],
        "execution_head": _git_text("rev-parse", "HEAD"),
        "product_source_diff_from_candidate": False,
        "capability_payload": capability_payload,
        "capability_payload_sha256": _canonical_sha256(capability_payload),
        "runs": runs,
        "gate_result": evaluate_runs(config, capability_payload, runs),
        "claim_ceiling": config["claim_ceiling"],
    }
    _write_json(output_dir / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(report["gate_result"], indent=2))
    return 0 if report["gate_result"]["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
