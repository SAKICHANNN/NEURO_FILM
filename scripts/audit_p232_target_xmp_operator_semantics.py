#!/usr/bin/env python3
"""Audit whether one retained target XMP defines an executable colour operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CRS = "{http://ns.adobe.com/camera-raw-settings/1.0/}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
SCHEMA = "kmcfm.p232-target-xmp-operator-semantics-result.v1"

COLOUR_FIELDS = (
    "WhiteBalance",
    "Temperature",
    "Tint",
    "Saturation",
    "ShadowTint",
    "RedHue",
    "RedSaturation",
    "GreenHue",
    "GreenSaturation",
    "BlueHue",
    "BlueSaturation",
    "Vibrance",
    "Exposure2012",
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
    "Clarity2012",
    "Dehaze",
    "Texture",
)
RENDERER_DEPENDENCY_FIELDS = (
    "Version",
    "ProcessVersion",
    "CameraProfile",
    "CameraProfileDigest",
    "AutoToneDigest",
    "AutoToneDigestNoSat",
)
GEOMETRY_FIELDS = (
    "CropTop",
    "CropLeft",
    "CropBottom",
    "CropRight",
    "CropAngle",
    "CropConstrainToWarp",
    "HasCrop",
    "AlreadyApplied",
    "PerspectiveVertical",
    "PerspectiveHorizontal",
    "PerspectiveRotate",
    "PerspectiveScale",
    "PerspectiveAspect",
    "PerspectiveUpright",
    "PerspectiveX",
    "PerspectiveY",
)
DETAIL_FIELDS = (
    "Sharpness",
    "SharpenRadius",
    "SharpenDetail",
    "SharpenEdgeMasking",
    "LuminanceSmoothing",
    "ColorNoiseReduction",
    "ColorNoiseReductionDetail",
    "ColorNoiseReductionSmoothness",
    "GrainAmount",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _crs_attributes(root: ET.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for element in root.iter():
        for name, value in element.attrib.items():
            if name.startswith(CRS):
                local = name[len(CRS) :]
                if local in values and values[local] != value:
                    raise ValueError(f"conflicting Camera Raw attribute: {local}")
                values[local] = value
    return values


def _direct_curve_sequences(root: ET.Element) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for element in root.iter():
        if not element.tag.startswith(CRS):
            continue
        sequence = element.find(RDF + "Seq")
        if sequence is None:
            continue
        points = [item.text or "" for item in sequence.findall(RDF + "li")]
        sequences.append({"name": element.tag[len(CRS) :], "points": points})
    return sequences


def _validate_numeric_colour_fields(attributes: dict[str, str]) -> None:
    for name in COLOUR_FIELDS:
        if name not in attributes or name in {"WhiteBalance"}:
            continue
        value = float(attributes[name])
        if not math.isfinite(value):
            raise ValueError(f"non-finite Camera Raw field: {name}")


def inspect_xmp(xmp_bytes: bytes) -> dict[str, Any]:
    root = ET.fromstring(xmp_bytes)
    attributes = _crs_attributes(root)
    curves = _direct_curve_sequences(root)
    _validate_numeric_colour_fields(attributes)
    return {
        "camera_raw_attribute_count": len(attributes),
        "direct_curve_sequence_count": len(curves),
        "curve_sequences": curves,
        "observed_colour_fields": {
            name: attributes[name] for name in COLOUR_FIELDS if name in attributes
        },
        "renderer_dependency_fields": {
            name: attributes[name]
            for name in RENDERER_DEPENDENCY_FIELDS
            if name in attributes
        },
        "geometry_fields": {
            name: attributes[name] for name in GEOMETRY_FIELDS if name in attributes
        },
        "detail_fields": {
            name: attributes[name] for name in DETAIL_FIELDS if name in attributes
        },
        "process_version": attributes.get("ProcessVersion"),
        "camera_profile": attributes.get("CameraProfile"),
        "has_settings": attributes.get("HasSettings"),
        "has_crop": attributes.get("HasCrop"),
        "profile_bytes_present": False,
        "renderer_equations_present": False,
        "processing_order_present": False,
    }


def run(config_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = _load_object(config_path)
    source = config["source"]
    named_paths = [
        ("dataset_card", ROOT / source["dataset_card_path"]),
        ("xmp", ROOT / source["xmp_path"]),
        ("prior_audit", ROOT / source["prior_audit_path"]),
    ]
    if reverse:
        named_paths.reverse()
    observed_hashes: dict[str, str] = {}
    observed_bytes: dict[str, int] = {}
    xmp_bytes: bytes | None = None
    for name, path in named_paths:
        observed_hashes[name] = _sha256_file(path)
        observed_bytes[name] = path.stat().st_size
        if name == "xmp":
            xmp_bytes = path.read_bytes()
    if xmp_bytes is None:
        raise RuntimeError("XMP source was not read")
    if observed_hashes["dataset_card"] != source["dataset_card_sha256"]:
        raise ValueError("dataset card identity changed")
    if observed_hashes["xmp"] != source["xmp_sha256"]:
        raise ValueError("XMP identity changed")
    if observed_bytes["xmp"] != int(source["xmp_bytes"]):
        raise ValueError("XMP byte count changed")

    facts = inspect_xmp(xmp_bytes)
    expected = config["expected_exact_sample_facts"]
    for key, expected_value in expected.items():
        if facts[key] != expected_value:
            raise ValueError(
                f"frozen XMP fact changed for {key}: {facts[key]!r}"
            )

    rule = config["strict_semantic_rule"]
    gates = {
        "source_identities_exact": True,
        "minimum_independent_recipes": (
            int(source["locally_retained_recipe_count"])
            >= int(rule["minimum_independent_recipes"])
        ),
        "bulk_recipe_access_authorized": bool(source["bulk_recipe_access_authorized"]),
        "renderer_equations_present": bool(facts["renderer_equations_present"]),
        "processing_order_present": bool(facts["processing_order_present"]),
        "referenced_profile_bytes_present": bool(facts["profile_bytes_present"]),
        "offline_metadata_only": True,
    }
    admitted = all(gates.values())
    result = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": (
            "PASS_PRIVATE_TARGET_XMP_EXPLICIT_OPERATOR_SEMANTICS"
            if admitted
            else "FAIL_CLOSED_BEFORE_OPERATOR_COMPILATION_TARGET_XMP_SEMANTICS_NOT_IDENTIFIABLE"
        ),
        "source": {
            "observed_sha256": dict(sorted(observed_hashes.items())),
            "observed_bytes": dict(sorted(observed_bytes.items())),
            "listed_recipe_count": int(source["listed_recipe_count"]),
            "listed_base_content_count": int(source["listed_base_content_count"]),
            "locally_retained_recipe_count": int(source["locally_retained_recipe_count"]),
            "rights_ceiling": source["rights_ceiling"],
        },
        "facts": facts,
        "gates": gates,
        "execution": {
            "network_reads": 0,
            "image_pixel_reads": 0,
            "operator_compilations": 0,
            "renders": 0,
            "scores": 0,
        },
        "decision": {
            "admitted": admitted,
            "selected_hypothesis": (
                "h1_explicit_operator" if admitted else "h2_renderer_recipe"
            ),
            "descriptor_only_remains_possible": not admitted,
            "interpretation": (
                "The retained XMP records rich edit intent but does not carry an "
                "equation-complete Adobe processing pipeline or referenced profile "
                "bytes, and one authorized recipe cannot establish shared semantics."
            ),
            "no_rescue": config["decision"]["no_rescue"],
        },
        "consumer_mapping": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    payload = canonical_json_bytes(result)
    result["scientific_identity"] = "sha256:" + _sha256_bytes(payload)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/p232_target_xmp_operator_semantics_v1.json",
    )
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, reverse=args.order == "reverse")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))


if __name__ == "__main__":
    main()
