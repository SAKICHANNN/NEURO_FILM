from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/U7_15B_DESKTOP_VISIBLE_STRENGTH_QUANTIZATION_RESULT.json"
)


def _git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def test_u7_15b_evidence_binds_committed_product_result() -> None:
    payload = json.loads(EVIDENCE.read_text("utf-8"))
    assert payload["status"] == (
        "PASS_PRIVATE_U7_15B_DESKTOP_VISIBLE_STRENGTH_QUANTIZATION"
    )
    assert payload["formal_report"] == {
        "all_gates_pass": True,
        "bytes": 2744,
        "forward_reverse_byte_exact": True,
        "scientific_identity": (
            "b70aad78b7f756754c1f4545d6b8f6f1c6b59edef0f40ae48e60d3a97c3ca4f3"
        ),
        "sha256": ("fccc0a8d1e2a2ab25d58c7c9b5dc206ce4f541c40a7573d5b92e4ac788a0052b"),
    }
    for path, identity in payload["bindings"].items():
        value = _git_bytes(path)
        assert (
            subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True
            ).strip()
            == identity["git_blob"]
        )
        assert hashlib.sha256(value).hexdigest() == identity["sha256"]
    assert payload["observations"]["parent"] == {
        "visible": "65%",
        "variable": 0.654321,
        "workflow_amount": 0.654321,
    }
    assert payload["observations"]["current_render_entry"] == {
        "visible": "65%",
        "variable": 0.65,
        "workflow_amount": 0.65,
    }
    assert payload["claim"]["calibrated_stock_response"] is False
    assert payload["claim"]["physical_film_reproduction"] is False
    assert payload["claim"]["public_release"] is False
