from __future__ import annotations

import json
from pathlib import Path

from scripts.acquire_u5_r2spcp2_preference_pairs import _inspect_png

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2spcp2a_preference_pair_acquisition_v1.json"


def test_spcp2a_contract_binds_exact_role_manifest_and_budget() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "U5.R2SPCP2A"
    assert payload["acquisition"]["scene_count_exact"] == 128
    assert payload["acquisition"]["member_count_exact"] == 256
    assert payload["acquisition"]["compressed_bytes_exact"] == 193422517
    assert payload["acquisition"]["roles"] == ["fit", "calibration"]


def test_spcp2a_png_inspection_rejects_non_rgb(tmp_path: Path) -> None:
    from io import BytesIO

    import numpy as np
    import pytest
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8), "L").save(buffer, "PNG")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="opaque RGB"):
        _inspect_png(buffer.getvalue(), contract)
