import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4ib_freezes_twelve_unique_unused_candidates() -> None:
    payload = json.loads(
        (
            ROOT / "configs/u6_p4ib_analytical_density_fresh_source_preflight_v1.json"
        ).read_text(encoding="utf-8")
    )
    candidates = payload["candidates"]
    assert len(candidates) == 12
    assert len({row["repository_id"] for row in candidates}) == 12
    assert len({row["sha256"] for row in candidates}) == 12
    assert len({row["make"] for row in candidates}) == 12
    assert any(
        "u6_p4hy" in row["path"] for row in payload["preflight"]["comparison_manifests"]
    )
