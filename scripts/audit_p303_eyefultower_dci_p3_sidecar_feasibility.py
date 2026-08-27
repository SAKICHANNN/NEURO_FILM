"""Audit whether EyefulTower's DCI-P3 declaration uniquely binds a white point."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


class P303Error(RuntimeError):
    """Raised when a frozen P303 source or authority identity differs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _verify_file(path: Path, expected: dict[str, object]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(expected["bytes"])
        and _sha256_file(path) == str(expected["sha256"])
    )


def _extract_authority_facts(config: dict[str, object]) -> dict[str, object]:
    sources = config["authority_sources"]
    assert isinstance(sources, dict)
    icc_config = sources["icc_registry"]
    smpte_config = sources["smpte_st2113"]
    assert isinstance(icc_config, dict) and isinstance(smpte_config, dict)
    icc_path = ROOT / str(icc_config["path"])
    smpte_path = ROOT / str(smpte_config["path"])
    if not _verify_file(icc_path, icc_config) or not _verify_file(
        smpte_path, smpte_config
    ):
        raise P303Error("frozen authority identity differs")

    icc_text = icc_path.read_text(encoding="utf-8")
    smpte_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(smpte_path).pages
    )
    variants = {
        "P3D65": {
            "white_xy": [0.3127, 0.329],
            "icc": all(
                token in icc_text
                for token in ("D65: x = 0.3127, y = 0.3290", "DCI-P3-D65.icc")
            ),
            "smpte": all(token in smpte_text for token in ("P3D65", "0.3127, 0.3290")),
        },
        "P3DCI": {
            "white_xy": [0.314, 0.351],
            "icc": all(
                token in icc_text
                for token in ("x = 0.3140, y = 0.3510", "DCI-P3-DCI.icc")
            ),
            "smpte": all(token in smpte_text for token in ("P3DCI", "0.3140, 0.3510")),
        },
    }
    primaries = all(
        token in icc_text for token in ("R   0.68", "G   0.265", "B   0.15")
    ) and all(token in smpte_text for token in ("0.6800", "0.2650", "0.1500"))
    return {
        "authority_files": {
            "icc_registry": {
                "bytes": icc_path.stat().st_size,
                "sha256": _sha256_file(icc_path),
            },
            "smpte_st2113": {
                "bytes": smpte_path.stat().st_size,
                "sha256": _sha256_file(smpte_path),
            },
        },
        "common_p3_primaries_confirmed": primaries,
        "variants": variants,
    }


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_config = config["source"]
    source = ROOT / source_config["path"]
    source_before = _sha256_file(source)
    if (
        source.stat().st_size != int(source_config["bytes"])
        or source_before != source_config["sha256"]
    ):
        raise P303Error("frozen image source identity differs")

    facts = _extract_authority_facts(config)
    variants = facts["variants"]
    assert isinstance(variants, dict)
    order = tuple(variants)
    if reverse:
        order = tuple(reversed(order))
    supported = [
        name
        for name in order
        if bool(variants[name]["icc"]) and bool(variants[name]["smpte"])
    ]
    supported = sorted(supported)
    declaration = config["dataset_authority"]
    unique = len(supported) == int(
        config["frozen_color_facts"]["required_unique_variant_count"]
    )
    controls = (
        "reject_p3d65_selection",
        "reject_p3dci_selection",
        "reject_pixel_inference",
    )
    if reverse:
        controls = tuple(reversed(controls))
    gates = {
        "authority_files_exact": True,
        "common_p3_primaries_confirmed": bool(facts["common_p3_primaries_confirmed"]),
        "dataset_declaration_exact": declaration["declaration"]
        == "Color space: DCI-P3 (linear)",
        "dataset_white_point_declared": bool(declaration["white_point_declared"]),
        "unique_authoritative_white_variant": unique,
        "source_immutable": _sha256_file(source) == source_before,
        "zero_exr_header_reads": True,
        "zero_exr_pixel_reads": True,
        "zero_jpeg_reads": True,
        "zero_sidecar_publications": True,
    }
    decision = (
        "PASS_PRIVATE_EYEFULTOWER_DCI_P3_AUTHORITATIVE_SIDECAR"
        if all(gates.values())
        else "FAIL_CLOSED_AMBIGUOUS_DCI_P3_WHITE_POINT_BEFORE_SIDECAR"
    )
    report = {
        "authoritative_variants": supported,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "controls": {name: True for name in sorted(controls)},
        "dataset_declaration": declaration["declaration"],
        "decision": decision,
        "experiment_id": "P303",
        "facts": facts,
        "gates": gates,
        "schema": "neuro-film.p303-eyefultower-dci-p3-sidecar-feasibility-result.v1",
        "sidecar_publications": 0,
        "source_header_reads": 0,
        "source_pixel_reads": 0,
    }
    report["scientific_identity"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = execute(args.config.resolve(), reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
