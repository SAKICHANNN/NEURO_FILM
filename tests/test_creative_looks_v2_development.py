import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_creative_looks_v2 import (
    compare,
    matched_controls,
    quantize,
    select_development,
    simple_control,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "creative_config": "creative_hue_look_development_v2.json",
            "assessment": True,
        },
        {"creative_config": "creative_hue_look_development_v2.json", "detail": True},
    ],
)
def test_subject_diagnostic_cannot_change_assessment_or_old_modes(monkeypatch, kwargs):
    original = Path.read_bytes

    def read(path):
        if path.suffix.lower() == ".png":
            pytest.fail("invalid diagnostic must stop before image access")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="separate hue-development"):
        compare("must-not-create", subject_detail=True, **kwargs)


@pytest.mark.parametrize(
    "candidate",
    ["creative_looks_v2_development.json", "creative_hue_look_development_v1.json"],
)
def test_assessment_mismatch_stops_before_source_body(monkeypatch, candidate):
    original = Path.read_bytes

    def read(path):
        if path.suffix.lower() == ".png":
            pytest.fail("candidate mismatch must precede image access")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="candidate mismatch"):
        compare("must-not-create", creative_config=candidate, assessment=True)


def test_matched_controls_recover_affine_and_are_deterministic():
    x = np.random.default_rng(4).random((64, 64, 3), dtype=np.float32)
    target = (x * 0.7 + 0.12).astype(np.float32)
    controls, parameters = matched_controls(x, target)
    other, again = matched_controls(x, target)
    assert parameters == again
    np.testing.assert_allclose(controls["affine"], target, atol=1e-7)
    for name in controls:
        np.testing.assert_array_equal(controls[name], other[name])
        assert np.isfinite(controls[name]).all()
        assert controls[name].min() >= 0 and controls[name].max() <= 1
    constant, _ = matched_controls(
        np.zeros((16, 16, 3), np.float32), np.full((16, 16, 3), 0.4, np.float32)
    )
    np.testing.assert_allclose(constant["affine"], 0.4)


def inputs():
    config = json.loads(
        (ROOT / "configs/creative_looks_v2_photo_development_v1.json").read_text()
    )
    parent = json.loads((ROOT / config["parent_population"]).read_text())
    manifest = json.loads((ROOT / config["manifest"]).read_text())
    return config, parent, manifest


def test_metadata_selection_excludes_extra_row_and_keeps_reserve() -> None:
    config, parent, manifest = inputs()
    rows = select_development(config, manifest, parent)
    assert len(rows) == 8
    assert {r["id"] for r in rows}.isdisjoint(config["reserved_ids"])
    assert "fujifilm_x_s10" not in {r["id"] for r in rows}
    assert select_development(config, list(reversed(manifest)), parent) == rows


def test_selection_and_rights_fail_closed() -> None:
    config, parent, manifest = inputs()
    changed = copy.deepcopy(config)
    changed["development_ids"].reverse()
    with pytest.raises(ValueError, match="selection"):
        select_development(changed, manifest, parent)
    changed = copy.deepcopy(manifest)
    next(r for r in changed if r["id"] == config["development_ids"][0])[
        "rights_scope"
    ] = "unknown"
    with pytest.raises(ValueError, match="rights"):
        select_development(config, changed, parent)


def test_quantization_no_hidden_clamp_and_controls_explicitly_bounded() -> None:
    with pytest.raises(ValueError):
        quantize(np.full((1, 1, 3), 1.1))
    with pytest.raises(ValueError):
        quantize(np.full((1, 1, 3), np.nan))
    source = np.asarray([[[0, 1, 0.5]]], np.float32)
    np.testing.assert_array_equal(quantize(source), [[[0, 255, 128]]])
    for kwargs in ({"saturation": 1.2}, {"contrast": 1.15}):
        result = simple_control(source, **kwargs)
        assert result.min() >= 0 and result.max() <= 1
        assert not np.shares_memory(source, result)
