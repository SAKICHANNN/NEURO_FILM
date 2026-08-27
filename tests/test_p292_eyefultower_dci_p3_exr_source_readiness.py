from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_p292_eyefultower_dci_p3_exr_source_readiness import (
    _extract_facts,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p292_eyefultower_dci_p3_exr_source_readiness_v1.json"


def _responses(config: dict[str, object]) -> dict[str, dict[str, object]]:
    repo = config["official_repository"]
    expected = config["expected"]
    prefix = expected["s3_prefix"]
    objects: list[str] = []
    checksums: list[str] = []
    xml_rows: list[str] = []
    for camera in expected["camera_names"]:
        for index, suffix in enumerate(expected["capture_suffixes"]):
            relative = f"{camera}/{camera}_{suffix}.exr"
            key = prefix + relative
            objects.append(key)
            checksum = (
                expected["selected_md5"]
                if key == expected["selected_key"]
                else f"{len(checksums) + 1:032x}"
            )
            checksums.append(f"{checksum}  {relative}")
            size = expected["selected_size"] if key == expected["selected_key"] else 1
            etag = expected["selected_etag"] if key == expected["selected_key"] else '"x"'
            xml_rows.append(
                f"<Contents><Key>{key}</Key><ETag>{etag}</ETag><Size>{size}</Size></Contents>"
            )
    extra_count = expected["listing_objects"] - len(objects)
    xml_rows.extend(
        f"<Contents><Key>{prefix}meta-{index}.txt</Key><ETag>\"m\"</ETag><Size>1</Size></Contents>"
        for index in range(extra_count)
    )
    expected_exr_total = int(expected["exr_total_bytes"])
    current_total = int(expected["selected_size"]) + len(objects) - 1
    difference = expected_exr_total - current_total
    xml_rows[1] = xml_rows[1].replace("<Size>1</Size>", f"<Size>{difference + 1}</Size>")
    return {
        "commit": {
            "body": json.dumps(
                {
                    "sha": repo["commit"],
                    "tree": {"sha": repo["tree"]},
                    "parents": [{"sha": parent} for parent in repo["parents"]],
                }
            ).encode()
        },
        "tree": {
            "body": json.dumps(
                {
                    "sha": repo["tree"],
                    "truncated": False,
                    "tree": [
                        {"path": "LICENSE", "sha": expected["license_blob"]},
                        {"path": "README.md", "sha": expected["readme_blob"]},
                    ]
                    + [
                        {"path": f"path/{index}", "sha": f"{index:040x}"}
                        for index in range(expected["repository_tree_entries"] - 2)
                    ],
                }
            ).encode()
        },
        "readme": {
            "body": (
                b"High dynamic range images merged from 9-photo raw exposure brackets.\n"
                b"Color space: DCI-P3 (linear)\n"
                b"Stored as EXR images with uncompressed 32-bit floating-point numbers.\n"
                b"The JPEG images are white-balanced and tone-mapped versions of the HDR images.\n"
                b"Relicensed all content under the MIT license.\n"
            )
        },
        "license": {"body": b"Permission is hereby granted, free of charge"},
        "listing": {
            "body": (
                "<ListBucketResult>" + "".join(xml_rows) + "</ListBucketResult>"
            ).encode()
        },
        "checksums": {"body": ("\n".join(checksums) + "\n").encode()},
        "selected_head": {
            "body": b"",
            "body_bytes": 0,
            "content_length": expected["selected_size"],
            "etag": expected["selected_etag"],
            "http_status": 200,
            "method": "HEAD",
        },
    }


def test_p292_parser_binds_inventory_rights_and_source_description() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    facts = _extract_facts(config, _responses(config))
    assert facts["official_identity"]
    assert facts["mit_all_content_rights"]
    assert facts["linear_dci_p3_float32_exr"]
    assert facts["nine_raw_bracket_observation"]
    assert facts["inventory_exact"]
    assert facts["checksum_coverage"]
    assert facts["selected_head_exact"]
    assert facts["capture_group_identities"]
    assert not facts["jpeg_independent_truth"]


def test_p292_selection_is_frozen_and_payloads_are_forbidden() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["expected"]["selected_key"].endswith("40/40_DSC0001.exr")
    assert "Do not download an EXR/JPEG body" in config["stop_rule"]
    assert "candidate 3" in config["stop_rule"]
    assert "independent truth" in config["claim_ceiling"]
    assert config["gates"]["require_zero_payload_and_pixel_reads"]
