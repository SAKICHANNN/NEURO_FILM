from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_u7_3c_offline_recipe_preview import audit_catalog


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
