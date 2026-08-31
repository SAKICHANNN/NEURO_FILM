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
CONFIG = ROOT / "configs/u7_2o_product_look_cli_entry_v1.json"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
U7_2H_CONFIG = ROOT / "configs/u7_2h_explicit_product_look_selection_v1.json"
SCRATCH_PARENT = ROOT / "tmp"

CONTRACT_COMMIT = "417ff63c"
IMPLEMENTATION_COMMIT = "f02ca086"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _normalized_recipe(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["software"]["commit"] = "<COMMIT>"
    payload["input"]["path"] = "<INPUT>"
    payload["output"]["path"] = "<OUTPUT>"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _normalized_metrics(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["input"] = "<INPUT>"
    payload["output"] = "<OUTPUT>"
    payload["render_recipe"]["path"] = "<RECIPE>"
    payload["render_recipe"]["sha256"] = "<RECIPE_SHA>"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _comparison(
    root: Path,
    source: Path,
    *,
    look: str,
    amount: float,
) -> dict[str, Any]:
    token = str(amount).replace(".", "_")
    reference = root / f"reference-{look}-{token}.png"
    candidate = root / f"candidate-{look}-{token}.png"
    common = (
        str(source),
        "--look-amount",
        str(amount),
        "--write-recipe",
    )
    first = _run(
        *common,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--style",
        look,
        "--output",
        str(reference),
    )
    second = _run(
        *common,
        "--product-look",
        look,
        "--output",
        str(candidate),
    )
    reference_recipe_sha, _ = _normalized_recipe(reference.with_suffix(".recipe.json"))
    candidate_recipe_sha, candidate_recipe = _normalized_recipe(
        candidate.with_suffix(".recipe.json")
    )
    return {
        "amount": amount,
        "candidate_returncode": second.returncode,
        "image_byte_exact": reference.read_bytes() == candidate.read_bytes(),
        "look": look,
        "output_sha256": _sha256(candidate),
        "profile_id": candidate_recipe["profile"]["profile_id"],
        "recipe_claim_calibrated": candidate_recipe["claim"][
            "calibrated_reference_allowed"
        ],
        "recipe_claim_label": candidate_recipe["claim"]["output_label"],
        "recipe_semantics_exact": reference_recipe_sha == candidate_recipe_sha,
        "recipe_style": candidate_recipe["render"]["style"],
        "reference_returncode": first.returncode,
    }


def _bundle(root: Path, source: Path) -> dict[str, Any]:
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
    reference_recipe, _ = _normalized_recipe(reference.with_suffix(".recipe.json"))
    candidate_recipe, _ = _normalized_recipe(candidate.with_suffix(".recipe.json"))
    reference_layers = reference.parent / f"{reference.stem}_layers"
    candidate_layers = candidate.parent / f"{candidate.stem}_layers"
    left = {path.name: _sha256(path) for path in reference_layers.iterdir()}
    right = {path.name: _sha256(path) for path in candidate_layers.iterdir()}
    return {
        "candidate_returncode": second.returncode,
        "image_byte_exact": reference.read_bytes() == candidate.read_bytes(),
        "layer_manifest_exact": left == right,
        "layer_sha256": dict(sorted(right.items())),
        "metrics_semantics_exact": _normalized_metrics(
            reference.with_suffix(".metrics.json")
        )
        == _normalized_metrics(candidate.with_suffix(".metrics.json")),
        "output_sha256": _sha256(candidate),
        "recipe_semantics_exact": reference_recipe == candidate_recipe,
        "reference_returncode": first.returncode,
    }


def _conflicts(root: Path) -> list[dict[str, Any]]:
    cases = [
        ("style", ("--style", "portra_400"), "cannot be combined with --style"),
        (
            "use-render-profile",
            ("--use-render-profile",),
            "cannot be combined with --use-render-profile",
        ),
        (
            "render-profile",
            ("--render-profile", str(PRODUCT_PROFILE)),
            "cannot be combined with --render-profile",
        ),
        (
            "engine",
            ("--color-engine", "analytic-y-chromaticity"),
            "requires the safe_lab color engine",
        ),
    ]
    rows = []
    for name, extra, message in cases:
        output = root / f"conflict-{name}.png"
        missing = root / f"must-not-decode-{name}.png"
        completed = _run(
            str(missing),
            "--product-look",
            "ektar_100",
            *extra,
            "--output",
            str(output),
        )
        rows.append(
            {
                "message_exact": message in completed.stderr,
                "name": name,
                "no_decode_error": str(missing) not in completed.stderr,
                "output_absent": not output.exists(),
                "returncode": completed.returncode,
            }
        )
    discovery = _run("--product-look", "ektar_100", "--list-product-looks")
    rows.append(
        {
            "message_exact": "cannot be combined with --list-product-looks"
            in discovery.stderr,
            "name": "discovery",
            "no_decode_error": True,
            "output_absent": True,
            "returncode": discovery.returncode,
        }
    )
    return rows


def _non_product_values(root: Path) -> list[dict[str, Any]]:
    rows = []
    for value in ("generic_bw", "hp5", "unknown", ""):
        output = root / f"invalid-{value or 'empty'}.png"
        completed = _run(
            str(root / "must-not-decode-invalid.png"),
            "--product-look",
            value,
            "--output",
            str(output),
        )
        rows.append(
            {
                "output_absent": not output.exists(),
                "returncode": completed.returncode,
                "value": value,
            }
        )
    return rows


def build_report(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent = json.loads(U7_2H_CONFIG.read_text(encoding="utf-8"))
    source_locks = config["source_locks"]
    implementation_files = {
        "README.md": _sha256(ROOT / "README.md"),
        "scripts/render_film.py": _sha256(SCRIPT),
        "tests/test_u7_2o_product_look_cli_entry.py": _sha256(
            ROOT / "tests/test_u7_2o_product_look_cli_entry.py"
        ),
    }
    prechange_locks_exact = (
        source_locks["product_profile_sha256"] == _sha256(PRODUCT_PROFILE)
        and source_locks["product_catalog_sha256"]
        == _sha256(ROOT / "src/inference/product_look_catalog.py")
        and source_locks["u7_2k_evidence_sha256"]
        == _sha256(ROOT / "docs/evidence/U7_2K_PRODUCT_LOOK_CLI_DISCOVERY_RESULT.json")
        and source_locks["u7_2n_evidence_sha256"]
        == _sha256(
            ROOT
            / "docs/evidence/U7_2N_PRODUCT_AUXILIARY_OUTPUT_TRANSACTION_RESULT.json"
        )
    )
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    scratch_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="u7_2o_", dir=SCRATCH_PARENT) as raw:
        scratch_path = Path(raw)
        source = scratch_path / "source.png"
        _source(source)
        source_sha = _sha256(source)
        ordered = [
            (look, amount)
            for look in config["available_values"]
            for amount in config["amounts"]
        ]
        if order == "reverse":
            ordered.reverse()
        comparisons = [
            _comparison(scratch_path, source, look=look, amount=amount)
            for look, amount in ordered
        ]
        comparisons.sort(key=lambda row: (row["look"], row["amount"]))
        bundle = _bundle(scratch_path, source)
        conflicts = _conflicts(scratch_path)
        conflicts.sort(key=lambda row: row["name"])
        invalid = _non_product_values(scratch_path)
        invalid.sort(key=lambda row: row["value"])
        discovery = _run("--list-product-looks")
        discovery_payload = json.loads(discovery.stdout)
        legacy = scratch_path / "legacy.png"
        legacy_run = _run(str(source), "--write-recipe", "--output", str(legacy))
        source_immutable = _sha256(source) == source_sha
        legacy_exact = (
            legacy_run.returncode == 0
            and _sha256(legacy)
            == parent["prechange_oracle"]["legacy_default"]["output_sha256"]
        )
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_exact = (
            all(
                marker in readme
                for marker in (
                    "--product-look portra_400",
                    "--look-amount 0.75",
                    "--write-recipe",
                    "not claims of calibrated stock response or physical-film",
                )
            )
            and "--style portra_400" not in readme
        )

    assert scratch_path is not None
    residue_zero = not scratch_path.exists()
    all_comparisons = all(
        row["candidate_returncode"] == row["reference_returncode"] == 0
        and row["image_byte_exact"]
        and row["recipe_semantics_exact"]
        for row in comparisons
    )
    profile_claims_exact = all(
        row["profile_id"] == "safe-rich-product-v1"
        and row["recipe_style"] == row["look"]
        and row["recipe_claim_label"] == "film-inspired"
        and row["recipe_claim_calibrated"] is False
        for row in comparisons
    )
    conflicts_exact = all(
        row["returncode"] == 2
        and row["message_exact"]
        and row["no_decode_error"]
        and row["output_absent"]
        for row in conflicts
    )
    invalid_exact = all(
        row["returncode"] == 2 and row["output_absent"] for row in invalid
    )
    bundle_exact = (
        all(
            bundle[key]
            for key in (
                "image_byte_exact",
                "layer_manifest_exact",
                "metrics_semantics_exact",
                "recipe_semantics_exact",
            )
        )
        and bundle["candidate_returncode"] == bundle["reference_returncode"] == 0
    )
    discovery_exact = (
        discovery.returncode == 0
        and [row["look_id"] for row in discovery_payload["looks"]]
        == ["velvia_50", "portra_400", "ektar_100", "generic_bw"]
        and [row["availability"] for row in discovery_payload["looks"]]
        == ["available", "available", "available", "blocked_severe_artifact"]
    )
    gates = {
        "all_three_looks_all_amounts_exact": all_comparisons,
        "auxiliary_transaction_unchanged": bundle_exact,
        "all_conflicts_predecode": conflicts_exact,
        "discovery_unchanged": discovery_exact,
        "implementation_files_present": all(implementation_files.values()),
        "legacy_output_unchanged": legacy_exact,
        "non_product_values_rejected": invalid_exact,
        "prechange_source_locks_exact": prechange_locks_exact,
        "profile_style_and_claim_exact": profile_claims_exact,
        "readme_quickstart_executable_and_claim_bounded": readme_exact,
        "source_immutable": source_immutable,
        "owned_runtime_residue_zero": residue_zero,
    }
    return {
        "schema_version": "neuro-film.u7-2o-product-look-cli-entry-result.v1",
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "contract_commit": CONTRACT_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_files": implementation_files,
        "comparisons": comparisons,
        "full_bundle": bundle,
        "conflicts": conflicts,
        "non_product_values": invalid,
        "discovery_stdout_sha256": hashlib.sha256(
            discovery.stdout.encode("utf-8")
        ).hexdigest(),
        "fixture_sha256": source_sha,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
