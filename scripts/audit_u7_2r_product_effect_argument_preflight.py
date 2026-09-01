from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
CONFIG = ROOT / "configs/u7_2r_product_effect_argument_preflight_v1.json"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
SCRATCH_PARENT = ROOT / "tmp"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    payload = subprocess.check_output(["git", *arguments], cwd=ROOT)
    return payload if binary else payload.decode("utf-8").strip()


def _source(path: Path) -> None:
    y, x = np.mgrid[:47, :61]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _artifacts(output: Path) -> tuple[Path, ...]:
    return (
        output,
        output.with_suffix(".recipe.json"),
        output.with_suffix(".metrics.json"),
        output.parent / f"{output.stem}_layers",
    )


def _normalized_json(path: Path, *, metrics: bool = False) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if metrics:
        payload["input"] = "<INPUT>"
        payload["output"] = "<OUTPUT>"
        payload["render_recipe"]["path"] = "<RECIPE>"
        payload["render_recipe"]["sha256"] = "<RECIPE_SHA>"
    else:
        payload["software"]["commit"] = "<COMMIT>"
        payload["input"]["path"] = "<INPUT>"
        payload["output"]["path"] = "<OUTPUT>"
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def _source_locks(config: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for role, binding in sorted(config["source_locks"].items()):
        commit = config["implementation_commit"]
        if role == "contract":
            commit = config["contract_commit"]
        resolved_blob = _git("rev-parse", f"{commit}:{binding['path']}")
        blob_bytes = _git("cat-file", "blob", str(resolved_blob), binary=True)
        rows[role] = {
            "blob_exact": resolved_blob == binding["git_blob"],
            "bytes": len(blob_bytes),
            "sha256": _sha256_bytes(blob_bytes),
            "sha256_exact": _sha256_bytes(blob_bytes) == binding["sha256"],
        }
    return rows


def _invalid_rows(config: dict[str, Any], root: Path, order: str) -> list[dict[str, Any]]:
    cases: list[tuple[str, str, str]] = []
    for option in config["strength_options"]:
        cases.extend(
            (option, value, f"{option} must be finite and in [0,1]")
            for value in config["invalid_strength_values"]
        )
    cases.extend(
        ("--seed", str(value), "--seed must be a signed 32-bit integer")
        for value in config["invalid_seed_values"]
    )
    if order == "reverse":
        cases.reverse()
    rows = []
    for option, value, message in cases:
        token = f"{option}={value}" if value.startswith("-") and option != "--seed" else None
        numeric_arguments = (token,) if token else (option, value)
        source = root / f"must-not-decode-{option[2:]}-{value}.png"
        output = root / f"invalid-{option[2:]}-{value}.png"
        completed = _run(
            str(source),
            "--product-look",
            "ektar_100",
            *numeric_arguments,
            "--write-recipe",
            "--write-layers",
            "--write-metrics",
            "--output",
            str(output),
        )
        rows.append(
            {
                "artifacts_absent": all(not path.exists() for path in _artifacts(output)),
                "input_unmentioned": str(source) not in completed.stderr,
                "message_exact": message in completed.stderr,
                "option": option,
                "returncode": completed.returncode,
                "value": value,
            }
        )
    return sorted(rows, key=lambda row: (row["option"], row["value"]))


def _boundary_rows(config: dict[str, Any], root: Path, source: Path, order: str) -> list[dict[str, Any]]:
    cases = [
        (option, str(value))
        for option in config["strength_options"]
        for value in config["valid_strength_boundaries"]
    ]
    cases.extend(("--seed", str(value)) for value in config["valid_seed_boundaries"])
    if order == "reverse":
        cases.reverse()
    rows = []
    for index, (option, value) in enumerate(cases):
        output = root / f"valid-{index}-{option[2:]}.png"
        completed = _run(
            str(source),
            "--product-look",
            "ektar_100",
            option,
            value,
            "--output",
            str(output),
        )
        rows.append(
            {
                "option": option,
                "output_exists": output.is_file(),
                "output_sha256": _sha256(output) if output.is_file() else None,
                "returncode": completed.returncode,
                "value": value,
            }
        )
    return sorted(rows, key=lambda row: (row["option"], row["value"]))


def _default_look_rows(config: dict[str, Any], root: Path, source: Path, order: str) -> list[dict[str, Any]]:
    looks = list(config["u7_2o_default_output_sha256"])
    if order == "reverse":
        looks.reverse()
    rows = []
    for look in looks:
        output = root / f"default-{look}.png"
        completed = _run(
            str(source),
            "--product-look",
            look,
            "--output",
            str(output),
        )
        observed = _sha256(output) if output.is_file() else None
        rows.append(
            {
                "look": look,
                "output_exact": observed == config["u7_2o_default_output_sha256"][look],
                "output_sha256": observed,
                "returncode": completed.returncode,
            }
        )
    return sorted(rows, key=lambda row: row["look"])


def _full_bundle(config: dict[str, Any], root: Path, source: Path) -> dict[str, Any]:
    reference = root / "bundle-reference.png"
    candidate = root / "bundle-candidate.png"
    common = (
        str(source),
        "--look-amount",
        "0.5",
        "--grain",
        "0.05",
        "--halation",
        "0.15",
        "--dust",
        "0.02",
        "--write-recipe",
        "--write-layers",
        "--write-metrics",
    )
    first = _run(
        *common,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--style",
        "ektar_100",
        "--output",
        str(reference),
    )
    second = _run(
        *common,
        "--product-look",
        "ektar_100",
        "--output",
        str(candidate),
    )
    candidate_layers = candidate.parent / f"{candidate.stem}_layers"
    layer_sha256 = {
        path.name: _sha256(path) for path in sorted(candidate_layers.iterdir())
    }
    oracle = config["u7_2o_full_effects_bundle"]
    return {
        "candidate_returncode": second.returncode,
        "image_byte_exact_to_reference": candidate.read_bytes() == reference.read_bytes(),
        "layer_sha256": layer_sha256,
        "layers_exact_to_u7_2o": layer_sha256 == oracle["layer_sha256"],
        "metrics_semantics_exact_to_reference": _normalized_json(
            candidate.with_suffix(".metrics.json"), metrics=True
        )
        == _normalized_json(reference.with_suffix(".metrics.json"), metrics=True),
        "output_exact_to_u7_2o": _sha256(candidate) == oracle["output_sha256"],
        "output_sha256": _sha256(candidate),
        "recipe_semantics_exact_to_reference": _normalized_json(
            candidate.with_suffix(".recipe.json")
        )
        == _normalized_json(reference.with_suffix(".recipe.json")),
        "reference_returncode": first.returncode,
    }


def _legacy_rows(root: Path, order: str) -> list[dict[str, Any]]:
    cases = [
        ("--grain", "nan"),
        ("--halation", "-0.1"),
        ("--dust", "1.1"),
        ("--seed", str(2**31)),
    ]
    if order == "reverse":
        cases.reverse()
    rows = []
    for option, value in cases:
        source = root / f"legacy-read-{option[2:]}.png"
        output = root / f"legacy-output-{option[2:]}.png"
        completed = _run(
            str(source),
            "--style",
            "ektar_100",
            option,
            value,
            "--output",
            str(output),
        )
        rows.append(
            {
                "artifacts_absent": all(not path.exists() for path in _artifacts(output)),
                "file_read_reached": "FileNotFoundError" in completed.stderr,
                "option": option,
                "parser_rejected": completed.returncode == 2,
                "value": value,
            }
        )
    return sorted(rows, key=lambda row: (row["option"], row["value"]))


def build_report(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    baseline_scratch = {path.name for path in SCRATCH_PARENT.glob("u7_2r_*")}
    runner_blob = str(_git("rev-parse", f"HEAD:{Path(__file__).relative_to(ROOT).as_posix()}"))
    source_locks = _source_locks(config)
    with tempfile.TemporaryDirectory(prefix="u7_2r_", dir=SCRATCH_PARENT) as raw:
        scratch = Path(raw)
        source = scratch / "source.png"
        _source(source)
        source_before = _sha256(source)
        invalid_rows = _invalid_rows(config, scratch, order)
        boundary_rows = _boundary_rows(config, scratch, source, order)
        default_rows = _default_look_rows(config, scratch, source, order)
        full_bundle = _full_bundle(config, scratch, source)
        legacy_rows = _legacy_rows(scratch, order)
        source_immutable = _sha256(source) == source_before
    residue_zero = {path.name for path in SCRATCH_PARENT.glob("u7_2r_*")} == baseline_scratch

    invalid_strengths = [row for row in invalid_rows if row["option"] != "--seed"]
    invalid_seeds = [row for row in invalid_rows if row["option"] == "--seed"]
    invalid_row_pass = lambda row: (
        row["returncode"] == 2
        and row["message_exact"]
        and row["input_unmentioned"]
        and row["artifacts_absent"]
    )
    gates = {
        "all_invalid_seeds_reject_predecode": all(map(invalid_row_pass, invalid_seeds)),
        "all_invalid_strengths_reject_predecode": all(map(invalid_row_pass, invalid_strengths)),
        "invalid_cases_publish_nothing": all(row["artifacts_absent"] for row in invalid_rows),
        "legacy_numeric_behavior_unchanged": all(
            row["file_read_reached"]
            and not row["parser_rejected"]
            and row["artifacts_absent"]
            for row in legacy_rows
        ),
        "owned_runtime_residue_zero": residue_zero,
        "source_immutable": source_immutable,
        "source_locks_exact": all(
            row["blob_exact"] and row["sha256_exact"] for row in source_locks.values()
        ),
        "three_default_product_look_outputs_exact": all(
            row["returncode"] == 0 and row["output_exact"] for row in default_rows
        ),
        "u7_2o_full_effects_bundle_exact": (
            full_bundle["reference_returncode"] == 0
            and full_bundle["candidate_returncode"] == 0
            and full_bundle["image_byte_exact_to_reference"]
            and full_bundle["output_exact_to_u7_2o"]
            and full_bundle["layers_exact_to_u7_2o"]
            and full_bundle["recipe_semantics_exact_to_reference"]
            and full_bundle["metrics_semantics_exact_to_reference"]
        ),
        "valid_numeric_boundaries_execute": all(
            row["returncode"] == 0 and row["output_exists"] for row in boundary_rows
        ),
    }
    return {
        "schema_version": "neuro-film.u7-2r-product-effect-argument-preflight-report.v1",
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "execution_commit": _git("rev-parse", "HEAD"),
        "runner_git_blob": runner_blob,
        "config_sha256": _sha256(config_path),
        "source_locks": source_locks,
        "invalid_rows": invalid_rows,
        "boundary_rows": boundary_rows,
        "default_look_rows": default_rows,
        "full_effects_bundle": full_bundle,
        "legacy_rows": legacy_rows,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.config, args.order)
    encoded = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
