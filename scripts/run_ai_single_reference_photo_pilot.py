"""Fixed three-reference comparison without selecting a visual winner."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ai_deep_photo_pilot import main

if __name__ == "__main__":
    cfg = json.loads(
        (ROOT / "configs/ai_single_reference_photo_pilot_v1.json").read_text()
    )
    parent = json.loads((ROOT / cfg["parent_config"]).read_text())
    refs = json.loads((ROOT / parent["reference_manifest"]).read_text())
    selected = [
        i
        for i, row in sorted(
            enumerate(refs["rows"]), key=lambda item: item[1]["sha256"]
        )[:3]
    ]
    assert selected == cfg["reference_indices"]
    for index in selected:
        main(cfg["parent_config"], structural=True, reference_index=index)
