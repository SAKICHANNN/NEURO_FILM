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
CONFIG = ROOT / "configs/u7_2v_product_physical_halation_preflight_v1.json"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
SCRATCH_PARENT = ROOT / "tmp"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


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


def _normalized_recipe(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["software"]["commit"] = "<COMMIT>"
    payload["input"]["path"] = "<INPUT>"
    payload["output"]["path"] = "<OUTPUT>"
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _source_locks(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, lock in sorted(config["source_locks"].items()):
        path = ROOT / lock["path"]
        actual_blob = _git("hash-object", "--", str(path))
        actual_sha = _sha256(path)
        rows[name] = {
            "path": lock["path"],
            "git_blob": actual_blob,
            "sha256": actual_sha,
            "blob_exact": actual_blob == lock["git_blob"],
            "sha256_exact": actual_sha == lock["sha256"],
        }
    return rows


def _cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for look in config["product_looks"]:
        cases.append(
            {
                "case_id": f"product-{look}-expert-reject",
                "kind": "expert-reject",
                "arguments": [
                    "--product-look",
                    look,
                    "--halation-model",
                    "physical",
                    "--halation-expert-controls",
                    "--halation-background-gain",
                    "nan",
                ],
            }
        )

    for option, domain in config["numeric_domains"].items():
        cli = "--" + option.replace("_", "-")
        minimum, maximum = map(float, domain)
        for label, value in (
            ("nan", "nan"),
            ("positive-infinity", "inf"),
            ("below", f"{minimum - 0.0001:g}"),
            ("above", f"{maximum + 0.0001:g}"),
        ):
            cases.append(
                {
                    "case_id": f"product-ektar-{option}-{label}",
                    "kind": "numeric-reject",
                    "option": cli,
                    "arguments": [
                        "--product-look",
                        "ektar_100",
                        "--halation-model",
                        "physical",
                        f"{cli}={value}",
                    ],
                }
            )

    for case_id, arguments in (
        ("unknown-preset", ["--halation-preset", "does-not-exist"]),
        ("bw-type-red-response", ["--halation-type", "bw_clear_base"]),
        (
            "bw-family-red-response",
            [
                "--halation-model-family",
                "bw_density_halation",
                "--halation-color-response",
                "red_orange_core",
            ],
        ),
    ):
        cases.append(
            {
                "case_id": f"product-ektar-invalid-{case_id}",
                "kind": "category-reject",
                "arguments": [
                    "--product-look",
                    "ektar_100",
                    "--halation-model",
                    "physical",
                    *arguments,
                ],
            }
        )

    valid: list[tuple[str, list[str]]] = [("default", [])]
    for option, domain in config["numeric_domains"].items():
        cli = "--" + option.replace("_", "-")
        valid.extend(
            (
                (f"{option}-minimum", [cli, f"{float(domain[0]):g}"]),
                (f"{option}-maximum", [cli, f"{float(domain[1]):g}"]),
            )
        )
    valid.extend(
        (f"preset-{preset}", ["--halation-preset", preset])
        for preset in config["valid_presets"]
    )
    valid.append(
        (
            "bw-valid",
            [
                "--halation-type",
                "bw_clear_base",
                "--halation-color-response",
                "neutral_density",
            ],
        )
    )
    for case_id, arguments in valid:
        cases.append(
            {
                "case_id": f"product-ektar-valid-{case_id}",
                "kind": "valid-cross",
                "arguments": [
                    "--product-look",
                    "ektar_100",
                    "--halation-model",
                    "physical",
                    *arguments,
                ],
            }
        )

    cases.append(
        {
            "case_id": "nonproduct-expert-cross",
            "kind": "nonproduct-cross",
            "arguments": [
                "--style",
                "ektar_100",
                "--use-render-profile",
                "--halation-model",
                "physical",
                "--halation-expert-controls",
                "--halation-background-gain",
                "1.25",
            ],
        }
    )
    return cases


def _case_rows(
    config: dict[str, Any], scratch: Path, order: str
) -> list[dict[str, Any]]:
    cases = _cases(config)
    if order == "reverse":
        cases.reverse()
    rows: list[dict[str, Any]] = []
    for case in cases:
        source = scratch / f"missing-{case['case_id']}.png"
        output = scratch / f"output-{case['case_id']}.png"
        completed = _run(str(source), *case["arguments"], "--output", str(output))
        artifacts_absent = all(not path.exists() for path in _artifacts(output))
        if case["kind"] == "expert-reject":
            message_exact = "requires locked controls" in completed.stderr
            passed = completed.returncode == 2 and message_exact
        elif case["kind"] == "numeric-reject":
            message_exact = f"{case['option']} must be finite" in completed.stderr
            passed = completed.returncode == 2 and message_exact
        elif case["kind"] == "category-reject":
            message_exact = (
                "invalid product physical halation controls" in completed.stderr
            )
            passed = completed.returncode == 2 and message_exact
        else:
            message_exact = "FileNotFoundError" in completed.stderr
            passed = completed.returncode != 2 and message_exact
        rows.append(
            {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "returncode": completed.returncode,
                "message_exact": message_exact,
                "input_unmentioned_on_reject": (
                    str(source) not in completed.stderr
                    if case["kind"].endswith("reject")
                    else None
                ),
                "artifacts_absent": artifacts_absent,
                "passed": passed
                and artifacts_absent
                and (
                    str(source) not in completed.stderr
                    if case["kind"].endswith("reject")
                    else True
                ),
            }
        )
    return sorted(rows, key=lambda row: row["case_id"])


def _parent_rows(
    config: dict[str, Any], scratch: Path, order: str
) -> list[dict[str, Any]]:
    source = scratch / "parent-source.png"
    _source(source)
    source_before = _sha256(source)
    looks = list(config["product_looks"])
    if order == "reverse":
        looks.reverse()
    rows: list[dict[str, Any]] = []
    for look in looks:
        reference = scratch / f"parent-reference-{look}.png"
        candidate = scratch / f"parent-candidate-{look}.png"
        common = (str(source), "--write-recipe")
        explicit = _run(
            *common,
            "--use-render-profile",
            "--render-profile",
            str(PRODUCT_PROFILE),
            "--style",
            look,
            "--output",
            str(reference),
        )
        product = _run(
            *common,
            "--product-look",
            look,
            "--output",
            str(candidate),
        )
        observed = _sha256(candidate) if candidate.is_file() else None
        rows.append(
            {
                "look": look,
                "explicit_returncode": explicit.returncode,
                "product_returncode": product.returncode,
                "output_sha256": observed,
                "output_exact_to_parent": observed
                == config["parent_output_sha256"][look],
                "output_exact_to_explicit": candidate.is_file()
                and reference.is_file()
                and candidate.read_bytes() == reference.read_bytes(),
                "recipe_semantics_exact_to_explicit": candidate.with_suffix(
                    ".recipe.json"
                ).is_file()
                and reference.with_suffix(".recipe.json").is_file()
                and _normalized_recipe(candidate.with_suffix(".recipe.json"))
                == _normalized_recipe(reference.with_suffix(".recipe.json")),
            }
        )
    if _sha256(source) != source_before:
        raise RuntimeError("parent source mutated")
    return sorted(rows, key=lambda row: row["look"])


def build_report(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    baseline = {path.name for path in SCRATCH_PARENT.glob("u7_2v_*")}
    source_locks = _source_locks(config)
    with tempfile.TemporaryDirectory(prefix="u7_2v_", dir=SCRATCH_PARENT) as raw:
        scratch = Path(raw)
        cases = _case_rows(config, scratch, order)
        parents = _parent_rows(config, scratch, order)
    residue_zero = {path.name for path in SCRATCH_PARENT.glob("u7_2v_*")} == baseline

    reject_rows = [row for row in cases if row["kind"].endswith("reject")]
    gates = {
        "source_locks_exact": all(
            row["blob_exact"] and row["sha256_exact"]
            for row in source_locks.values()
        ),
        "product_expert_controls_reject_predecode": all(
            row["passed"] for row in cases if row["kind"] == "expert-reject"
        ),
        "invalid_locked_values_reject_predecode": all(
            row["passed"] for row in cases if row["kind"] == "numeric-reject"
        ),
        "invalid_preset_or_categories_reject_predecode": all(
            row["passed"] for row in cases if row["kind"] == "category-reject"
        ),
        "invalid_controls_publish_nothing": all(
            row["artifacts_absent"] for row in reject_rows
        ),
        "valid_locked_controls_cross_preflight": all(
            row["passed"] for row in cases if row["kind"] == "valid-cross"
        ),
        "nonproduct_expert_route_crosses_preflight": all(
            row["passed"] for row in cases if row["kind"] == "nonproduct-cross"
        ),
        "product_parent_outputs_exact": all(
            row["explicit_returncode"] == 0
            and row["product_returncode"] == 0
            and row["output_exact_to_parent"]
            and row["output_exact_to_explicit"]
            and row["recipe_semantics_exact_to_explicit"]
            for row in parents
        ),
        "owned_runtime_residue_zero": residue_zero,
    }
    return {
        "schema_version": "neuro-film.u7-2v-product-physical-halation-preflight-report.v1",
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "execution_commit": _git("rev-parse", "HEAD"),
        "runner_git_blob": _git(
            "rev-parse", f"HEAD:{Path(__file__).relative_to(ROOT).as_posix()}"
        ),
        "config_sha256": _sha256(config_path),
        "source_locks": source_locks,
        "case_count": len(cases),
        "cases": cases,
        "parent_rows": parents,
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
