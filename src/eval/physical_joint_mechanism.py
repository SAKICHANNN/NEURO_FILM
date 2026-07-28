"""U6.P7A4 mechanism controls and refreshed blind evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.global_frontier import sha256_file
from src.eval.physical_joint_ablation import load_contracts, render_arms
from src.film_physics import required_spatial_response_halo


SCHEMA = "neuro_film.u6_p7a4_mechanism_controlled_severe_audit_contract.v1"


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _verify_inherited_reports(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    first_path = root / config["inherited_report_run_a"]
    second_path = root / config["inherited_report_run_b"]
    first = _load_exact_json(
        root,
        config["inherited_report_run_a"],
        config["inherited_report_run_a_sha256"],
    )
    second = _load_exact_json(
        root,
        config["inherited_report_run_b"],
        config["inherited_report_run_b_sha256"],
    )
    if first != second or first["stable_evidence_id"] != second["stable_evidence_id"]:
        raise ValueError("inherited P7A3 reports are not exact")
    failed = [name for name, passed in first["decisions"].items() if not passed]
    if failed != ["isolated_excursions"]:
        raise ValueError("P7A4 requires exactly the inherited isolated gate failure")
    for row in first["rows"]:
        for arm, digest in row["output_sha256"].items():
            path = first_path.parent / "renders" / arm / f"{row['sample_id']}.png"
            if sha256_file(path) != digest:
                raise ValueError("inherited render hash mismatch")
    return first, first_path


def _mechanism_controls(
    controls: dict[str, Any], runtime: Any
) -> dict[str, Any]:
    shape = tuple(int(value) for value in controls["shape"])
    tolerance = float(controls["maximum_flat_combined_minus_cheap_abs"])
    flat_rows = []
    repeat_exact = True
    for rgb in controls["constant_rgb"]:
        source = np.broadcast_to(
            np.asarray(rgb, dtype=np.float64), (*shape, 3)
        ).copy()
        first = render_arms(source, runtime)
        second = render_arms(source, runtime)
        repeat_exact = repeat_exact and all(
            np.array_equal(first[name], second[name]) for name in first
        )
        spatial_range = {
            name: float(np.max(np.ptp(values, axis=(0, 1))))
            for name, values in first.items()
        }
        flat_rows.append(
            {
                "input_rgb": rgb,
                "combined_minus_cheap_max_abs": float(
                    np.max(np.abs(first["combined"] - first["cheap"]))
                ),
                "maximum_spatial_range": float(max(spatial_range.values())),
            }
        )

    background = np.asarray(
        controls["impulse_background_rgb"], dtype=np.float64
    )
    source = np.broadcast_to(background, (*shape, 3)).copy()
    center = (shape[0] // 2, shape[1] // 2)
    source[center] = np.asarray(controls["impulse_rgb"], dtype=np.float64)
    first = render_arms(source, runtime)
    second = render_arms(source, runtime)
    repeat_exact = repeat_exact and all(
        np.array_equal(first[name], second[name]) for name in first
    )
    difference = np.max(
        np.abs(first["combined"] - first["cheap"]), axis=-1
    )
    halo = required_spatial_response_halo(runtime.profile)
    yy, xx = np.indices(shape)
    outside = (np.abs(yy - center[0]) > halo) | (
        np.abs(xx - center[1]) > halo
    )
    outside_max = float(np.max(difference[outside]))
    decisions = {
        "flat_invariance": max(
            row["maximum_spatial_range"] for row in flat_rows
        )
        <= tolerance,
        "flat_full_vs_cheap": max(
            row["combined_minus_cheap_max_abs"] for row in flat_rows
        )
        <= tolerance,
        "finite_support": outside_max
        <= float(
            controls["maximum_outside_halo_combined_minus_cheap_abs"]
        ),
        "repeat_exact": repeat_exact,
    }
    return {
        "flat_rows": flat_rows,
        "impulse": {
            "shape": list(shape),
            "center_yx": list(center),
            "analytic_halo_pixels": halo,
            "outside_halo_max_abs": outside_max,
            "inside_halo_max_abs": float(np.max(difference[~outside])),
        },
        "decisions": decisions,
    }


def _tile(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(
            ImageOps.exif_transpose(image).convert("RGB"),
            size,
            method=Image.Resampling.LANCZOS,
        )


def _refresh_blind(
    *,
    root: Path,
    config: dict[str, Any],
    runtime: Any,
    inherited_run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = config["visual_refresh"]
    fixed_ids = [
        sample_id
        for sample_id in runtime.eligible_ids
        if sample_id
        in {
            "canon_eos_kiss_f",
            "nikon_d2x",
            "sony_nex_3n",
            "fujifilm_finepix_s5000",
            "olympus_sp550uz",
            "panasonic_dmc_gf2",
            "pentax_k_r",
            "leica_d_lux_6",
            "kodak_dcs_pro_14n",
        }
    ]
    tile_size = (320, 210)
    header = 24
    seed = int(hashlib.sha256(config["node"].encode()).hexdigest()[:8], 16)
    mappings: dict[str, Any] = {}
    hashes = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for round_index in range(1, int(protocol["rounds"]) + 1):
        canvas = Image.new(
            "RGB",
            (3 * tile_size[0], len(fixed_ids) * (tile_size[1] + header)),
            (24, 24, 24),
        )
        draw = ImageDraw.Draw(canvas)
        round_mapping: dict[str, dict[str, str]] = {}
        for row_index, sample_id in enumerate(fixed_ids):
            order = list(protocol["primary_arms"])
            random.Random(seed + round_index * 1009 + row_index).shuffle(order)
            round_mapping[sample_id] = {"A": order[0], "B": order[1]}
            source_path = root / runtime.source_rows[sample_id]["decoded_path"]
            paths = [
                source_path,
                inherited_run_dir / "renders" / order[0] / f"{sample_id}.png",
                inherited_run_dir / "renders" / order[1] / f"{sample_id}.png",
            ]
            y = row_index * (tile_size[1] + header)
            for column, (label, path) in enumerate(
                zip(("SOURCE", "A", "B"), paths, strict=True)
            ):
                canvas.paste(
                    _tile(path, tile_size),
                    (column * tile_size[0], y + header),
                )
                draw.text(
                    (column * tile_size[0] + 4, y + 4),
                    f"{label} {sample_id}" if column == 0 else label,
                    fill=(235, 235, 235),
                )
        mappings[f"round_{round_index}"] = round_mapping
        path = output_dir / f"blind_round_{round_index}.png"
        canvas.save(path, format="PNG", compress_level=6)
        hashes.append(sha256_file(path))
    identities = {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in mappings.values()
    }
    if len(identities) != int(protocol["rounds"]):
        raise RuntimeError("blind round mappings must differ")
    mapping_raw = (
        json.dumps(mappings, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (output_dir / "private_mapping.json").write_bytes(mapping_raw)
    return {
        "fixed_ids": fixed_ids,
        "blind_sha256": hashes,
        "mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
    }


def evaluate_mechanism_audit(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or config.get("post_result_retuning_allowed")
    ):
        raise ValueError("unsupported U6.P7A4 contract")
    parent = _load_exact_json(
        root, config["parent_audit"], config["parent_audit_sha256"]
    )
    _load_exact_json(
        root, config["parent_decision"], config["parent_decision_sha256"]
    )
    contract, runtime = load_contracts(root, parent)
    inherited, inherited_path = _verify_inherited_reports(root, config)
    mechanism = _mechanism_controls(config["synthetic_controls"], runtime)
    visual = _refresh_blind(
        root=root,
        config=config,
        runtime=runtime,
        inherited_run_dir=inherited_path.parent,
        output_dir=output_dir,
    )
    inherited_decisions = {
        name: value
        for name, value in inherited["decisions"].items()
        if name != "isolated_excursions"
    }
    decisions = {**inherited_decisions, **mechanism["decisions"]}
    core = {
        "schema": "neuro_film.u6_p7a4_mechanism_controlled_severe_audit_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "inherited_stable_evidence_id": inherited["stable_evidence_id"],
        "inherited_summary": inherited["summary"],
        "mechanism_controls": mechanism,
        "visual_evidence": visual,
        "decisions": decisions,
        "automatic_pass": all(decisions.values()),
        "branch": config["branch_rule"][
            "complete_pass" if all(decisions.values()) else "automatic_fail"
        ],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = ["evaluate_mechanism_audit", "write_report"]
