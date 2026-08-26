from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_u7_3c_offline_recipe_preview import audit_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_preview_audit_is_order_invariant_for_frozen_shape(tmp_path: Path) -> None:
    entries = []
    for index, style in enumerate(("a", "b", "c")):
        path = tmp_path / f"{style}.png"
        rgb = np.full((48, 64, 3), index * 4000 + 1000, dtype=np.uint16)
        assert cv2.imwrite(str(path), rgb[..., ::-1])
        entries.append(
            {
                "status": "valid",
                "recipe_path": f"{style}.recipe.json",
                "style": style,
                "output_path": str(path),
                "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "output_format": "PNG",
                "output_bit_depth": 16,
                "output_label": "film-inspired",
                "evidence_grade": "look-approximation",
            }
        )
    forward, forward_html = audit_catalog(
        {"entries": entries},
        expected_source_dimensions=(64, 48),
        expected_preview_dimensions=(64, 48),
    )
    reverse, reverse_html = audit_catalog(
        {"entries": entries},
        reverse=True,
        expected_source_dimensions=(64, 48),
        expected_preview_dimensions=(64, 48),
    )
    assert forward == reverse
    assert forward_html == reverse_html
    assert forward["scientific"]["status"] == (
        "PASS_PRIVATE_HASH_BOUND_OFFLINE_RECIPE_PREVIEWS"
    )
    assert all(forward["scientific"]["gates"].values())


def test_tracked_u7_3c_evidence_binds_formal_and_visual_results() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/U7_3C_OFFLINE_RECIPE_PREVIEW_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    formal = evidence["formal_execution"]
    assert evidence["status"] == "PASS_PRIVATE_HASH_BOUND_OFFLINE_RECIPE_PREVIEWS"
    assert formal["reports_byte_exact"] is True
    assert formal["forward_report_sha256"] == formal["reverse_report_sha256"]
    assert formal["html_sha256"] == (
        "1b204dd32111f6a60dda0ccfe09d03ccd4cee461bb1a4f07c7945cc7dcbf2e29"
    )
    assert evidence["pixel_contract"]["sha256_verified_before_decode"] is True
    assert evidence["pixel_contract"]["input_photo_pixels_read"] == 0
    assert evidence["browser_visual_review"]["owned_edge_profile_directories_removed"] == 2
    assert all(evidence["gates"].values())
    assert evidence["relationship_to_u7_2c"]["u7_2c_proxy_separation_failure_unchanged"]
