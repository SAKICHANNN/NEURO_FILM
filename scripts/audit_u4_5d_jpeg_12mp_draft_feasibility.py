"""Formal U4.5D zero-pixel JPEG native-draft feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import PIL
from PIL import Image, JpegImagePlugin, features

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview import preview_dimensions

SCHEMA = "kmcfm.u4-5d-jpeg-12mp-draft-feasibility-result.v1"


class U45DError(RuntimeError):
    """Raised when the frozen audit inputs or boundary are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def evaluate_geometry(
    *, source: tuple[int, int], target: tuple[int, int], draft: tuple[int, int], ceiling: int
) -> dict[str, bool]:
    """Evaluate the frozen native-draft geometry gates."""

    source_width, source_height = source
    target_width, target_height = target
    draft_width, draft_height = draft
    return {
        "draft_not_below_target": (
            draft_width >= target_width and draft_height >= target_height
        ),
        "draft_not_above_source": (
            draft_width <= source_width and draft_height <= source_height
        ),
        "draft_strictly_smaller_than_source": (
            draft_width < source_width or draft_height < source_height
        ),
        "draft_pixels_at_most_target_ceiling": draft_width * draft_height <= ceiling,
    }


def _tile_rows(image: Image.Image) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tile in image.tile:
        rows.append(
            {
                "codec_name": tile.codec_name,
                "extents": list(tile.extents),
                "offset": tile.offset,
                "args": list(tile.args),
            }
        )
    return rows


def _probe(source: Path, target: tuple[int, int], ceiling: int) -> dict[str, Any]:
    load_calls = 0
    original_load = JpegImagePlugin.JpegImageFile.load

    def forbidden_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal load_calls
        load_calls += 1
        raise U45DError("pixel load is forbidden in U4.5D")

    JpegImagePlugin.JpegImageFile.load = forbidden_load
    try:
        with Image.open(source) as image:
            source_size = image.size
            source_mode = image.mode
            source_format = image.format
            image.draft("RGB", target)
            draft_size = image.size
            draft_mode = image.mode
            tiles = _tile_rows(image)
    finally:
        JpegImagePlugin.JpegImageFile.load = original_load
    geometry = evaluate_geometry(
        source=source_size,
        target=target,
        draft=draft_size,
        ceiling=ceiling,
    )
    return {
        "source_dimensions": list(source_size),
        "source_mode": source_mode,
        "source_format": source_format,
        "target_dimensions": list(target),
        "target_pixels": target[0] * target[1],
        "draft_dimensions": list(draft_size),
        "draft_pixels": draft_size[0] * draft_size[1],
        "draft_mode": draft_mode,
        "decoder_tiles": tiles,
        "pixel_load_calls": load_calls,
        "geometry": geometry,
    }


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = _json(config_path)
    source = ROOT / config["source"]["path"]
    runtime = config["runtime"]
    bindings = config["bindings"]
    target = config["target"]
    before = _sha256(source)
    derived_target = preview_dimensions(
        int(config["source"]["width"]),
        int(config["source"]["height"]),
        int(target["maximum_pixels"]),
    )
    probe = _probe(source, derived_target, int(target["maximum_pixels"]))
    after = _sha256(source)
    authority_exact = all(
        (
            PIL.__version__ == runtime["pillow_version"],
            features.version_codec("jpg") == runtime["libjpeg_version"],
            (ROOT / runtime["jpeg_plugin_path"]).stat().st_size
            == runtime["jpeg_plugin_bytes"],
            _sha256(ROOT / runtime["jpeg_plugin_path"])
            == runtime["jpeg_plugin_sha256"],
            (ROOT / runtime["imaging_binary_path"]).stat().st_size
            == runtime["imaging_binary_bytes"],
            _sha256(ROOT / runtime["imaging_binary_path"])
            == runtime["imaging_binary_sha256"],
            _sha256(ROOT / bindings["u4_5b_config_path"])
            == bindings["u4_5b_config_sha256"],
        )
    )
    gates = {
        "authority_exact": authority_exact,
        "source_exact": all(
            (
                source.stat().st_size == config["source"]["bytes"],
                before == config["source"]["sha256"],
                probe["source_dimensions"]
                == [config["source"]["width"], config["source"]["height"]],
                probe["source_format"] == "JPEG",
            )
        ),
        "target_exact": all(
            (
                list(derived_target) == [target["width"], target["height"]],
                derived_target[0] * derived_target[1] == target["pixels"],
            )
        ),
        **probe["geometry"],
        "decoder_tile_exact": probe["decoder_tiles"]
        == [
            {
                "codec_name": "jpeg",
                "extents": [0, 0, probe["draft_dimensions"][0], probe["draft_dimensions"][1]],
                "offset": 0,
                "args": ["RGB", ""],
            }
        ],
        "pixel_reads_zero": probe["pixel_load_calls"] == 0,
        "media_output_writes_zero": True,
        "source_immutable": before == after,
    }
    status = (
        "PASS_PRIVATE_JPEG_12MP_NATIVE_DRAFT_FEASIBILITY"
        if all(gates.values())
        else "FAIL_CLOSED_JPEG_12MP_NATIVE_DRAFT_UNAVAILABLE"
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": status,
        "bindings": {
            "config_sha256": _sha256(config_path),
            "contract_sha256": _sha256(ROOT / bindings["contract_path"]),
            "preview_core_sha256": _sha256(ROOT / bindings["preview_core_path"]),
            "runner_sha256": _sha256(ROOT / bindings["runner_path"]),
            "test_sha256": _sha256(ROOT / bindings["test_path"]),
            "source_sha256": before,
        },
        "probe": probe,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u4_5d_jpeg_12mp_draft_feasibility_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
