"""Audit the product severe veto propagated from BW2.D2."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config
from src.inference import (
    list_product_looks,
    load_render_profile,
    render_product_look_rgb,
    replay_style_safe_recipe_to_file,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _source() -> np.ndarray:
    y, x = np.mgrid[:31, :43]
    return np.ascontiguousarray(
        np.stack(
            (
                ((x * 13 + y * 7) % 251) / 250.0,
                ((x * 3 + y * 17 + 19) % 251) / 250.0,
                ((x * 11 + y * 5 + 43) % 251) / 250.0,
            ),
            axis=-1,
        ).astype(np.float32)
    )


def build_report(*, config_path: Path, order: tuple[str, ...]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent_evidence"]["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    profile = load_render_profile(
        ROOT / "configs/render_profiles/safe_rich_product_v1.json", root=ROOT
    )
    statistics = json.loads(
        (ROOT / "configs/film_color_stats.json").read_text(encoding="utf-8")
    )["styles"]
    catalog = {row["look_id"]: row for row in list_product_looks()}
    source = _source()

    colour_hashes: dict[str, str] = {}
    for look_id in order:
        if look_id == "generic_bw":
            continue
        output = render_product_look_rgb(
            source,
            profile=profile,
            look_id=look_id,
            look_amount=1.0,
            style_statistics=statistics,
            guardrails={look_id: load_guardrail_config(
                ROOT / "configs/color_guardrails.json", look_id
            )},
            seed=7,
            tile_size=17,
        )
        colour_hashes[look_id] = _array_sha256(output)

    dispatch_message = ""
    try:
        render_product_look_rgb(
            np.empty((0, 0, 3), dtype=np.float64),
            profile={},
            look_id="generic_bw",
            look_amount=1.0,
            style_statistics={},
            guardrails={},
            seed=0,
        )
    except ValueError as exc:
        dispatch_message = str(exc)

    with tempfile.TemporaryDirectory(prefix="u7_2g_") as temporary:
        temp = Path(temporary)
        output_path = temp / "blocked.png"
        recipe_message = ""
        try:
            replay_style_safe_recipe_to_file(
                {"render": {"style": "generic_bw"}},
                profile_path=temp / "missing-profile.json",
                output_path=output_path,
                root=temp,
            )
        except ValueError as exc:
            recipe_message = str(exc)
        cli = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/render_film.py"),
                str(temp / "missing-input.png"),
                "--style",
                "generic_bw",
                "--use-render-profile",
                "--render-profile",
                str(ROOT / "configs/render_profiles/safe_rich_product_v1.json"),
                "--output",
                str(output_path),
                "--write-recipe",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        cli_message = next(
            (
                line.strip()
                for line in reversed(cli.stderr.splitlines())
                if "product look 'generic_bw' is unavailable" in line
            ),
            "",
        )
        residue_count = sum(1 for path in temp.rglob("*") if path.is_file())

    expected_message = "3/16" in config["blocked_reason"]
    gates = {
        "parent_sha_exact": _sha256(parent_path)
        == config["parent_evidence"]["sha256"],
        "parent_status_exact": parent["status"]
        == config["parent_evidence"]["required_status"],
        "catalog_order_exact": tuple(catalog) == tuple(config["required_order"]),
        "catalog_availability_exact": all(
            catalog[look_id]["availability"] == config["availability"][look_id]
            for look_id in config["required_order"]
        ),
        "colour_count_exact": len(colour_hashes) == 3,
        "colour_hashes_unique": len(set(colour_hashes.values())) == 3,
        "dispatch_blocked_before_pixel_validation": expected_message
        and "3/16" in dispatch_message,
        "recipe_blocked_before_profile_or_input_read": expected_message
        and "3/16" in recipe_message,
        "cli_blocked_before_input_decode": cli.returncode != 0
        and "3/16" in cli_message,
        "blocked_cli_residue_zero": residue_count == 0,
    }
    return {
        "schema_id": "neuro-film.u7-2g-generic-bw-severe-veto-result.v1",
        "experiment_id": "U7.2G",
        "implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "bindings": {
            "config": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(config_path)},
            "parent_evidence": {"path": str(parent_path.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(parent_path)},
            "catalog": {"path": "src/inference/product_look_catalog.py", "sha256": _sha256(ROOT / "src/inference/product_look_catalog.py")},
            "recipe_replay": {"path": "src/inference/style_safe_engine.py", "sha256": _sha256(ROOT / "src/inference/style_safe_engine.py")},
            "cli": {"path": "scripts/render_film.py", "sha256": _sha256(ROOT / "scripts/render_film.py")},
        },
        "catalog": [catalog[look_id] for look_id in config["required_order"]],
        "colour_output_float32_sha256": dict(sorted(colour_hashes.items())),
        "blocked_messages": {
            "dispatch": dispatch_message,
            "recipe": recipe_message,
            "cli": cli_message,
        },
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "decision": (
            "Keep generic_bw discoverable with an exact BW2.D2 severe-artifact "
            "veto, but reject product dispatch, CLI execution and recipe replay "
            "before input decode or pixel execution. Preserve the three colour "
            "Look Approximation controls unchanged."
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u7_2g_generic_bw_severe_veto_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--order",
        choices=("forward", "reverse"),
        default="forward",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    order = tuple(config["required_order"])
    if args.order == "reverse":
        order = tuple(reversed(order))
    report = build_report(config_path=args.config, order=order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
