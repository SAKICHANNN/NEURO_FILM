from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_u6_p2c1_route_exposure_polarity import evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_current_shared_sensitometry_closes_only_slide_route() -> None:
    config = json.loads(
        (
            ROOT / "configs/u6_p2c1_route_exposure_polarity_audit_v1.json"
        ).read_text(encoding="utf-8")
    )
    first = evaluate(config)
    second = evaluate(config)
    assert first == second
    assert first["automatic_pass"] is False
    assert first["branch"] == "slide_only_fail"
    assert first["failed_routes"] == ["slide_direct_scan"]
    by_route = {row["route"]: row for row in first["rows"]}
    assert by_route["color_negative_neutral_scan"]["passed"] is True
    assert by_route["color_negative_print"]["passed"] is True
    assert by_route["bw_developer_scan"]["passed"] is True
    assert by_route["slide_direct_scan"]["checks"]["endpoint"] is False
    assert first["non_neutral_bw_rejected"] is True
