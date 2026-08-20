from __future__ import annotations

import json
from pathlib import Path

from scripts.build_u5_r2repid9a_original_member_lock import build_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_builder_is_order_invariant_and_reads_no_payload() -> None:
    config = json.loads((ROOT / "configs/u5_r2repid9a_original_member_lock_v1.json").read_text(encoding="utf-8"))
    roles = json.loads((ROOT / config["parent"]["roles_report_path"]).read_text(encoding="utf-8"))
    selected = [row for row in roles["selected"]["rows"] if row["role"] in {"fit", "calibration"}]
    lookup = {
        f"images/original/{row['scene_id']}": {
            "path": f"images/original/{row['scene_id']}",
            "size": 1024,
            "lfs": {"oid": "a" * 64},
        }
        for row in selected
    }

    def fake_post(_url: str, paths: list[str]) -> list[dict]:
        return [lookup[path] for path in paths]

    forward = build_manifest(ROOT, config, post_paths=fake_post)
    reversed_result = build_manifest(ROOT, config, reverse=True, post_paths=fake_post)
    assert forward == reversed_result
    assert forward["automatic_pass"]
    assert forward["execution"] == {
        "member_payload_reads": 0,
        "image_decodes": 0,
        "operator_fits": 0,
        "sealed_scene_requests": 0,
    }


def test_runner_has_no_drive_literal() -> None:
    source = (ROOT / "scripts/build_u5_r2repid9a_original_member_lock.py").read_text(encoding="utf-8")
    assert "D:\\" not in source
    assert "P:\\" not in source
