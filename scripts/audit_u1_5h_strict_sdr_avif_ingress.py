#!/usr/bin/env python3
"""Formal committed-head audit for strict ordinary SDR AVIF product ingress."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import replay_style_safe_recipe_to_file
from src.preprocess import (
    inspect_input,
    load_working_image,
    raster_decode,
)
from src.preprocess.avif_sdr import (
    StrictSdrAvifError,
    inspect_strict_sdr_avif,
)

CONFIG = ROOT / "configs/u1_5h_strict_sdr_avif_ingress_v1.json"
SOURCE = ROOT / "data/external/u1_5h_strict_sdr_avif_v1/kodim03_yuv420_8bpc.avif"
P278_ROOT = ROOT / "data/external/p278_libavif_gainmap_v1"
SCRIPT = ROOT / "scripts/render_film.py"
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
BINDINGS = (
    ROOT / "src/preprocess/avif_sdr.py",
    ROOT / "src/preprocess/raster_decode.py",
    ROOT / "scripts/render_film.py",
    ROOT / "tests/test_u1_5h_strict_sdr_avif_ingress.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)


def _verified_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["contract_id"] != "U1_5H_STRICT_SDR_AVIF_INGRESS_V1":
        raise RuntimeError("unexpected U1.5H contract id")
    source = config["official_sdr_fixture"]
    if SOURCE.stat().st_size != source["size"] or _sha256(SOURCE) != source["sha256"]:
        raise RuntimeError("official SDR AVIF source identity mismatch")
    return config


def _p278_rows(config: dict[str, Any], order: str) -> list[dict[str, Any]]:
    parent = json.loads((ROOT / config["negative_bindings"]["p278_config"]).read_text(encoding="utf-8"))
    rows = [*parent["valid_fixtures"], *parent["invalid_fixtures"]]
    if order == "reverse":
        rows.reverse()
    return rows


def _audit_negative_rows(config: dict[str, Any], order: str) -> list[dict[str, Any]]:
    rows = _p278_rows(config, order)
    original_open = raster_decode.Image.open
    pixel_decoder_calls = 0

    def forbidden_open(*args: Any, **kwargs: Any) -> Any:
        nonlocal pixel_decoder_calls
        pixel_decoder_calls += 1
        raise AssertionError("P278 negative reached Pillow before rejection")

    raster_decode.Image.open = forbidden_open
    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            path = P278_ROOT / row["name"]
            if path.stat().st_size < 1 or _sha256(path) != row["sha256"]:
                raise RuntimeError(f"P278 fixture identity mismatch: {row['name']}")
            try:
                inspect_strict_sdr_avif(path)
            except StrictSdrAvifError as exc:
                parser_reason = str(exc)
            else:
                raise AssertionError(f"P278 fixture was admitted: {row['name']}")
            try:
                load_working_image(path)
            except ValueError as exc:
                loader_reason = str(exc)
                if "refusing SDR fallback" not in loader_reason:
                    raise AssertionError("P278 loader rejection lost fail-closed reason") from exc
            else:
                raise AssertionError(f"P278 fixture reached WorkingImage: {row['name']}")
            results.append(
                {
                    "name": row["name"],
                    "sha256": row["sha256"],
                    "parser_reason": parser_reason,
                    "loader_reason": loader_reason,
                }
            )
    finally:
        raster_decode.Image.open = original_open
    if pixel_decoder_calls:
        raise AssertionError("a P278 fixture reached Pillow before rejection")
    return sorted(results, key=lambda row: row["name"])


def _audit_product(config: dict[str, Any]) -> dict[str, Any]:
    strict = inspect_strict_sdr_avif(SOURCE)
    expected = config["official_sdr_fixture"]
    if (strict.width, strict.height) != (expected["width"], expected["height"]):
        raise AssertionError("official SDR AVIF dimensions mismatch")
    inspection = inspect_input(SOURCE)
    if inspection.source_profile.kind != "nclx":
        raise AssertionError("official SDR AVIF lost explicit NCLX identity")
    if any(warning.code == "assumed_srgb" for warning in inspection.warnings):
        raise AssertionError("official SDR AVIF was silently assumed sRGB")
    working = load_working_image(SOURCE)
    if (
        working.pixels.dtype != np.float32
        or working.pixels.shape != (expected["height"], expected["width"], 3)
        or not np.isfinite(working.pixels).all()
        or working.working_space != "linear_srgb"
    ):
        raise AssertionError("official SDR AVIF WorkingImage is invalid")

    scratch_parent = ROOT / "tmp"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="u1_5h_formal_", dir=scratch_parent))
    try:
        output = scratch / "ektar.png"
        replay = scratch / "ektar-replay.png"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(SOURCE),
                "--product-look",
                "ektar_100",
                "--look-amount",
                "0.65",
                "--output",
                str(output),
                "--write-recipe",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": "0"},
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(f"product AVIF render failed: {completed.stderr}")
        recipe_path = output.with_suffix(".recipe.json")
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        claim = recipe["claim"]
        if (
            claim["output_label"] != "film-inspired"
            or claim["evidence_grade"] != "look-approximation"
            or claim["calibrated_reference_allowed"] is not False
        ):
            raise AssertionError("AVIF product recipe exceeded Look Approximation")
        output_sha = _sha256(output)
        replay_style_safe_recipe_to_file(
            recipe, profile_path=PROFILE, output_path=replay, root=ROOT
        )
        replay_sha = _sha256(replay)
        if output.read_bytes() != replay.read_bytes():
            raise AssertionError("AVIF product recipe replay differs")
        return {
            "container": {
                "width": strict.width,
                "height": strict.height,
                "bits_per_channel": list(strict.bits_per_channel),
                "nclx": list(strict.nclx),
                "item_count": strict.item_count,
                "compatible_brands": list(strict.compatible_brands),
            },
            "working_image_sha256": hashlib.sha256(
                np.asarray(working.pixels, dtype="<f4").tobytes(order="C")
            ).hexdigest(),
            "working_min": float(np.min(working.pixels)),
            "working_max": float(np.max(working.pixels)),
            "output_sha256": output_sha,
            "recipe_sha256": _sha256(recipe_path),
            "replay_sha256": replay_sha,
            "claim": claim,
        }
    finally:
        shutil.rmtree(scratch)


def run(order: str) -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("formal U1.5H audit requires a tracked-clean worktree")
    config = _verified_config()
    source_before = _sha256(SOURCE)
    negative_rows = _audit_negative_rows(config, order)
    product = _audit_product(config)
    source_after = _sha256(SOURCE)
    gates = {
        "committed_head_clean": True,
        "official_source_identity_exact": source_before
        == config["official_sdr_fixture"]["sha256"],
        "official_source_immutable": source_after == source_before,
        "official_sdr_container_exact": product["container"]
        == {
            "width": 768,
            "height": 512,
            "bits_per_channel": [8, 8, 8],
            "nclx": [1, 13, 6, True],
            "item_count": 1,
            "compatible_brands": ["avif", "mif1", "miaf", "MA1B"],
        },
        "working_image_finite_linear_srgb": 0.0 <= product["working_min"]
        <= product["working_max"]
        <= 1.0,
        "all_p278_reject_before_pillow": len(negative_rows)
        == config["negative_bindings"]["required_fixture_count"],
        "product_render_recipe_replay_exact": product["output_sha256"]
        == product["replay_sha256"],
        "look_approximation_claim_exact": product["claim"]["output_label"]
        == "film-inspired"
        and product["claim"]["evidence_grade"] == "look-approximation"
        and product["claim"]["calibrated_reference_allowed"] is False,
        "scratch_residue_zero": not any((ROOT / "tmp").glob("u1_5h_formal_*")),
    }
    return {
        "schema_id": "kmcfm.u1-5h-strict-sdr-avif-ingress-result.v1",
        "contract_id": config["contract_id"],
        "status": (
            "PASS_PRIVATE_STRICT_SDR_AVIF_LOOK_APPROXIMATION_INGRESS"
            if all(gates.values())
            else "FAIL_CLOSED_U1_5H_FORMAL"
        ),
        "software": {
            "commit": _git("rev-parse", "HEAD"),
            "bindings": {
                str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                for path in BINDINGS
            },
        },
        "source": {
            "path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "size": SOURCE.stat().st_size,
            "sha256": source_before,
            "repository_commit": config["official_sdr_fixture"]["commit"],
        },
        "product": product,
        "negative_rows": negative_rows,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.order)
    _write_create_only(args.output, report)
    if not report["status"].startswith("PASS_"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

