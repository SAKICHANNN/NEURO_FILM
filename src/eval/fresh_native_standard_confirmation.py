"""Fresh-population comparison of fixed B0, AO6, and native Standard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.film_physics.display_look import (
    build_source_context_display_look_stages,
)
from src.film_physics.native_standard_factory import (
    create_opt_in_native_standard_runtime,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)
from src.preprocess import load_working_image, save_srgb16_png


SCHEMA = "neuro_film.u6_p8bp_fixed_arm_comparison_result.v1"
ARMS = (
    "fixed_b0",
    "fixed_ao6_colour_only_t15_c35",
    "fixed_native_standard_full_strength_1_0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_exact_json(path: Path, expected_sha256: str) -> Any:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_preflight(
    root: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if (
        config.get("schema")
        != "neuro_film.u6_p8bp_fresh_native_standard_confirmation.v1"
        or tuple(config["comparison"]["arms"]) != ARMS
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
    ):
        raise ValueError("unsupported U6.P8BP contract")
    decision = _load_exact_json(
        root / "configs/u6_p8bp_fresh_source_preflight_decision_v1.json",
        "004fad263ee0ddc437e57f5cc70fc69890c9bbb1bfeb5759383180bfba22cc0e",
    )
    evidence = decision["repeat_evidence"]
    output = (
        root / "outputs/u6_p8bp_fresh_native_standard_confirmation_v1"
    )
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("automatic_report.json", "automatic_report_sha256"),
        ("source_contact_sheet.png", "contact_sheet_sha256"),
        ("visual_source_review.json", "visual_source_review_sha256"),
    ):
        if sha256_file(output / name) != evidence[key]:
            raise ValueError(f"U6.P8BP preflight evidence drift: {name}")
    if (
        decision["result"]["decision"]
        != "pass_fresh_population_and_open_fixed_arm_comparison"
        or not decision["result"]["automatic_pass"]
        or not decision["result"]["visual_pass"]
    ):
        raise ValueError("U6.P8BP source gate did not pass")
    manifest = json.loads((output / "manifest.json").read_text())
    review = json.loads((output / "visual_source_review.json").read_text())
    if (
        len(manifest) != 9
        or review["confirmed_severe_source_artifact_count"] != 0
        or sorted(review["eligible_ids"])
        != sorted(row["id"] for row in manifest)
    ):
        raise ValueError("U6.P8BP source eligibility drift")
    return manifest


def render_fixed_arms(
    *,
    scene_linear: np.ndarray,
    artifact: dict[str, Any],
    runtime: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = np.ascontiguousarray(scene_linear, dtype=np.float32)
    encoded = np.ascontiguousarray(
        linear_srgb_to_encoded(source.astype(np.float64)),
        dtype=np.float32,
    )
    payload = artifact["component_payloads"][
        "ao6-source-context-display-look"
    ]
    apply_base, apply_residual = build_source_context_display_look_stages(
        payload,
        encoded,
    )
    base = np.ascontiguousarray(apply_base(encoded), dtype=np.float32)
    ao6 = np.ascontiguousarray(apply_residual(base), dtype=np.float32)
    native = np.empty_like(source)

    def sink(y0: int, y1: int, rows: np.ndarray) -> None:
        native[y0:y1] = rows

    receipt = runtime.render_to_sink(source, output_sink=sink)
    outputs = {
        ARMS[0]: base,
        ARMS[1]: ao6,
        ARMS[2]: native,
    }
    for arm_id, values in outputs.items():
        if (
            values.shape != source.shape
            or values.dtype != np.float32
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise RuntimeError(f"{arm_id} left display RGB")
    if (
        hashlib.sha256(native.tobytes()).hexdigest()
        != receipt["output"]["array_sha256"]
    ):
        raise RuntimeError("native Standard receipt drift")
    return outputs, receipt


def boundary_metrics(
    candidate: np.ndarray,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    code = np.rint(np.asarray(candidate) * 65535.0).astype(np.uint16)
    boundary = np.any((code == 0) | (code == 65535), axis=-1)
    metrics = {
        "output_code_boundary_fraction": float(np.mean(boundary)),
    }
    if reference is not None:
        reference_code = np.rint(
            np.asarray(reference) * 65535.0
        ).astype(np.uint16)
        reference_boundary = np.any(
            (reference_code == 0) | (reference_code == 65535),
            axis=-1,
        )
        metrics["new_boundary_fraction_vs_ao6"] = float(
            np.mean(boundary & ~reference_boundary)
        )
    return metrics


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_comparison(
    *,
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
    build_components: Any,
    build_config: dict[str, Any],
) -> dict[str, Any]:
    manifest = validate_preflight(root, config)
    if output_dir.exists():
        raise FileExistsError("U6.P8BP comparison is create-only")
    output_dir.mkdir(parents=True)
    builds = build_components(build_config, output_dir / "binaries")
    package = json.loads((root / build_config["package"]).read_text())
    artifact = compile_standalone_profile_artifact(
        root=root,
        config=json.loads(
            (root / build_config["profile_compiler_config"]).read_text()
        ),
    )
    runtime, factory = create_opt_in_native_standard_runtime(
        package=package,
        artifact=artifact,
        library_paths={
            name: Path(row["dll_path"]) for name, row in builds.items()
        },
    )
    rows: list[dict[str, Any]] = []
    for source in manifest:
        config_source = next(
            row
            for row in config["candidates"]
            if row["id"] == source["id"]
        )
        raw_path = root / config_source["path"]
        if sha256_file(raw_path) != config_source["sha256"]:
            raise ValueError("U6.P8BP raw source identity drift")
        working = load_working_image(raw_path)
        if (
            working.transfer_state != "scene_linear"
            or working.working_space
            not in {"linear_srgb", "linear_srgb_d65"}
            or not working.orientation_applied
        ):
            raise ValueError("U6.P8BP WorkingImage contract drift")
        outputs, receipt = render_fixed_arms(
            scene_linear=working.pixels,
            artifact=artifact,
            runtime=runtime,
        )
        ao6 = outputs[ARMS[1]]
        for arm_id in ARMS:
            path = output_dir / "renders" / arm_id / f"{source['id']}.png"
            save_srgb16_png(outputs[arm_id], path)
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if (
                decoded is None
                or decoded.dtype != np.uint16
                or decoded.shape != outputs[arm_id].shape
            ):
                raise RuntimeError("U6.P8BP PNG16 verification failed")
            metrics = boundary_metrics(
                outputs[arm_id],
                ao6 if arm_id == ARMS[2] else None,
            )
            rows.append(
                {
                    "source_id": source["id"],
                    "source_raw_sha256": config_source["sha256"],
                    "arm_id": arm_id,
                    "output_sha256": sha256_file(path),
                    "output_path": path.relative_to(root).as_posix(),
                    **metrics,
                }
            )
        rows[-1]["native_receipt_sha256"] = receipt["receipt_sha256"]
    gates = config["comparison"]["automatic_gate"]
    maximum_boundary = max(
        row["output_code_boundary_fraction"] for row in rows
    )
    native_rows = [row for row in rows if row["arm_id"] == ARMS[2]]
    maximum_new_boundary = max(
        row["new_boundary_fraction_vs_ao6"] for row in native_rows
    )
    automatic_pass = (
        len(rows) == len(manifest) * len(ARMS)
        and maximum_boundary
        <= float(gates["maximum_output_code_boundary_fraction"])
        and maximum_new_boundary
        <= float(gates["maximum_new_boundary_fraction_vs_ao6"])
    )
    core = {
        "schema": SCHEMA,
        "config_sha256": sha256_file(
            root
            / "configs/u6_p8bp_fresh_native_standard_confirmation_v1.json"
        ),
        "preflight_decision_sha256": (
            "004fad263ee0ddc437e57f5cc70fc69890c9bbb1bfeb5759383180bfba22cc0e"
        ),
        "factory_receipt_sha256": factory["receipt_sha256"],
        "bundle_sha256": artifact["bundle_sha256"],
        "source_count": len(manifest),
        "arms": list(ARMS),
        "rows": rows,
        "maximum_output_code_boundary_fraction": maximum_boundary,
        "maximum_new_boundary_fraction_vs_ao6": maximum_new_boundary,
        "automatic_gate_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {
        **core,
        "stable_evidence_id": _canonical_sha256(core),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_blind_sheet(
    *,
    root: Path,
    render_dir: Path,
    manifest: list[dict[str, Any]],
    round_index: int,
    output_path: Path,
    mapping_path: Path,
) -> None:
    font = ImageFont.load_default()
    mapping: list[dict[str, Any]] = []
    tiles: list[Image.Image] = []
    labels = ("A", "B", "C")
    for source in manifest:
        order = list(ARMS)
        random.Random(
            hashlib.sha256(
                f"u6-p8bp:{round_index}:{source['id']}".encode()
            ).digest()
        ).shuffle(order)
        mapping.append(
            {
                "source_id": source["id"],
                **dict(zip(labels, order)),
            }
        )
        with Image.open(root / source["decoded_path"]) as opened:
            original = opened.convert("RGB")
            original.thumbnail((430, 280), Image.Resampling.LANCZOS)
        candidates = []
        for arm_id in order:
            with Image.open(
                render_dir / "renders" / arm_id / f"{source['id']}.png"
            ) as opened:
                image = opened.convert("RGB")
                image.thumbnail((430, 280), Image.Resampling.LANCZOS)
            candidates.append(image)
        tile = Image.new("RGB", (1760, 330), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((8, 5), source["id"], fill="black", font=font)
        for index, (label, image) in enumerate(
            zip(("Source", *labels), (original, *candidates))
        ):
            x = 8 + index * 438
            draw.text((x, 24), label, fill="black", font=font)
            tile.paste(image, (x, 44))
        tiles.append(tile)
    sheet = Image.new("RGB", (1760, len(tiles) * 330 + 32), "white")
    ImageDraw.Draw(sheet).text(
        (8, 8),
        f"U6.P8BP fixed-arm blind round {round_index}",
        fill="black",
        font=font,
    )
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, 32 + index * 330))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "PNG")
    mapping_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ARMS",
    "SCHEMA",
    "boundary_metrics",
    "build_blind_sheet",
    "render_fixed_arms",
    "run_comparison",
    "validate_preflight",
]
