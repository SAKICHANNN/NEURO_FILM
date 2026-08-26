#!/usr/bin/env python3
"""Formal U1.6G4J staged-density recipe integration audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config
from src.filmfx import (
    composite_staged_density_rows,
    execute_staged_density_halation_default,
)
from src.inference import (
    RECIPE_SCHEMA_ID_V3,
    RECIPE_SCHEMA_ID_V4,
    RenderContractError,
    load_render_profile,
    render_resolved_safe_lab_rgb,
    replay_style_safe_recipe,
    replay_style_safe_recipe_to_file,
    sha256_file,
    validate_render_recipe,
)
from src.preprocess import (
    load_working_image,
    save_srgb16_png,
    srgb_icc_profile_sha256,
    working_image_to_srgb_float,
)

PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
CONTRACT = ROOT / "configs/u1_6g4j_staged_density_recipe_integration_v1.json"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(path: Path) -> None:
    yy, xx = np.mgrid[:192, :256]
    rgb = np.stack(
        (
            (3 * xx + yy) % 256,
            (xx + 5 * yy) % 256,
            ((xx // 8) * 19 + (yy // 6) * 13) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _decoded_facts(path: Path) -> dict[str, object]:
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None or array.dtype != np.uint16 or array.ndim != 3:
        raise ValueError("formal output is not a decoded RGB16 PNG")
    array = array[..., ::-1]
    with Image.open(path) as image:
        image.load()
        icc = image.info.get("icc_profile", b"")
    return {
        "array_sha256": _sha_bytes(array.tobytes()),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bounds": [int(array.min()), int(array.max())],
        "icc_sha256": _sha_bytes(icc),
    }


def _command(source: Path, output: Path, mode: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/render_film.py"),
        str(source),
        "--style",
        "velvia_50",
        "--use-render-profile",
        "--tile-size",
        "64",
        "--output-bit-depth",
        "16",
        "--png-compression",
        "0",
        "--write-recipe",
        "--output",
        str(output),
    ]
    if mode == "candidate":
        command.extend(
            [
                "--halation-model",
                "staged-density-research",
                "--halation",
                "1.0",
            ]
        )
    return command


def _direct_candidate(source: Path, path: Path) -> None:
    profile = load_render_profile(PROFILE, root=ROOT)
    working = load_working_image(source)
    statistics = json.loads(
        (ROOT / "configs/film_color_stats.json").read_text(encoding="utf-8")
    )
    base = render_resolved_safe_lab_rgb(
        working_image_to_srgb_float(working),
        style="velvia_50",
        style_statistics=statistics["styles"]["velvia_50"],
        style_parameters=profile["style_parameters"]["velvia_50"],
        guardrails=load_guardrail_config(
            ROOT / "configs/color_guardrails.json", "velvia_50"
        ),
        seed=7,
        tile_size=64,
    )
    layer, _ = execute_staged_density_halation_default(
        base, tile_size=64, source_row_chunk=64, coarse_row_chunk=7
    )
    rendered = composite_staged_density_rows(
        base, layer, row_chunk=64, output_margin=4
    )
    save_srgb16_png(rendered, path, compression_level=0)


def _worker(mode: str, scratch: Path) -> dict[str, object]:
    source = scratch / "source.png"
    output = scratch / "render.png"
    recipe_path = output.with_suffix(".recipe.json")
    replay = scratch / "replay.png"
    direct = scratch / "direct.png"
    failure = scratch / "failure.png"
    for path in (output, recipe_path, replay, direct, failure):
        path.unlink(missing_ok=True)
    completed = subprocess.run(
        _command(source, output, mode),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    validate_render_recipe(recipe)
    facts: dict[str, object] = {
        "mode": mode,
        "output_sha256": sha256_file(output),
        "recipe_sha256": _sha_bytes(recipe_bytes),
        "recipe_schema_id": recipe["schema_id"],
        "halation_model": recipe["render"]["effects"]["halation"]["model"],
        "decoded": _decoded_facts(output),
    }
    if mode == "candidate":
        replay_style_safe_recipe_to_file(
            recipe,
            profile_path=PROFILE,
            output_path=replay,
            root=ROOT,
        )
        _direct_candidate(source, direct)
        mutation = copy.deepcopy(recipe)
        mutation["render"]["effects"]["halation"]["resolved_parameters"][
            "tile_size"
        ] = 63
        rejected = False
        try:
            replay_style_safe_recipe(
                mutation,
                profile_path=PROFILE,
                root=ROOT,
            )
        except RenderContractError:
            rejected = True
        facts.update(
            {
                "replay_exact": replay.read_bytes() == output.read_bytes(),
                "direct_exact": direct.read_bytes() == output.read_bytes(),
                "injected_failure_rejected": rejected,
                "injected_failure_output_absent": not failure.exists(),
            }
        )
    for path in (output, recipe_path, replay, direct, failure):
        path.unlink(missing_ok=True)
    return facts


def _run_worker(mode: str, scratch: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            "--scratch-root",
            str(scratch),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    return json.loads(completed.stdout)


def _formal(report_path: Path, scratch: Path) -> dict[str, object]:
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    source = scratch / "source.png"
    _fixture(source)
    rows = [_run_worker(mode, scratch) for mode in ("baseline", "candidate", "candidate", "baseline")]
    baseline = [row for row in rows if row["mode"] == "baseline"]
    candidate = [row for row in rows if row["mode"] == "candidate"]
    gates = {
        "legacy_output_and_recipe_exact": baseline[0] == baseline[1]
        and baseline[0]["recipe_schema_id"] == RECIPE_SCHEMA_ID_V3
        and baseline[0]["halation_model"] == "simple",
        "candidate_output_and_recipe_exact": candidate[0] == candidate[1],
        "v4_schema_and_independent_replay_exact": all(
            row["recipe_schema_id"] == RECIPE_SCHEMA_ID_V4
            and row["replay_exact"]
            for row in candidate
        ),
        "direct_executor_parity_exact": all(row["direct_exact"] for row in candidate),
        "finite_bounded_srgb_icc": all(
            row["decoded"]["dtype"] == "uint16"
            and row["decoded"]["shape"] == [192, 256, 3]
            and row["decoded"]["bounds"][0] >= 0
            and row["decoded"]["bounds"][1] <= 65535
            and row["decoded"]["icc_sha256"] == srgb_icc_profile_sha256()
            for row in rows
        ),
        "default_render_unchanged": baseline[0]["halation_model"] == "simple",
        "mutations_fail_closed": all(
            row["injected_failure_rejected"]
            and row["injected_failure_output_absent"]
            for row in candidate
        ),
    }
    source.unlink()
    scratch_empty = not any(scratch.iterdir())
    gates["owned_scratch_empty"] = scratch_empty
    scratch.rmdir()
    report = {
        "schema": "neuro-film.u1-6g4j-staged-density-recipe-integration-result.v1",
        "node": "ULT > U1.6 > U1.6G4J",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip(),
        "contract_sha256": sha256_file(CONTRACT),
        "execution_count": len(rows),
        "baseline": baseline[0],
        "candidate": candidate[0],
        "gates": gates,
        "status": (
            "PASS_PRIVATE_STAGED_DENSITY_RECIPE_INTEGRATION"
            if all(gates.values())
            else "FAIL_CLOSED_STAGED_DENSITY_RECIPE_INTEGRATION"
        ),
        "claim_ceiling": "Private Windows/Python opt-in staged-density Look Approximation integration and exact replay only; no physical-film, stock, calibration, preference, package or default-product claim.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--worker", choices=("baseline", "candidate"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.worker, args.scratch_root), sort_keys=True))
        return 0
    if args.report is None:
        parser.error("--report is required for a formal run")
    report = _formal(args.report, args.scratch_root)
    print(json.dumps({"status": report["status"], "report": str(args.report)}))
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
