import json

import numpy as np
import pytest
from PIL import Image, ImageFilter
from pillow_lut import identity_table

from scripts import run_ai_vcg_reference as driver


def test_source_drift_rejects_before_external_module_import(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "MODEL", tmp_path)
    (tmp_path / "source").mkdir()
    (tmp_path / "source/pipeline.py").write_text(
        "raise AssertionError('must not execute')"
    )
    with pytest.raises(ValueError, match="official source drift"):
        driver.initialize({"source_hashes": {"pipeline.py": "0" * 64}})


def test_manifest_identity_and_rgb_domain(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"test": 1}))
    assert driver.checked_json(path, driver.sha(path)) == {"test": 1}
    with pytest.raises(ValueError, match="identity"):
        driver.checked_json(path, "0" * 64)
    image_path = tmp_path / "alpha.png"
    Image.new("RGBA", (2, 2)).save(image_path)
    with pytest.raises(ValueError, match="expected RGB"):
        driver.read_rgb(image_path)


def test_published_cube_layout_identity_and_channel_permutation():
    image = np.random.default_rng(12).integers(0, 256, (17, 19, 3), dtype=np.uint8)
    cube = np.asarray(identity_table(16).table, dtype=np.float32).reshape(64, 64, 3)
    output = Image.fromarray(image).filter(ImageFilter.Color3DLUT(16, cube.flatten()))
    assert np.array_equal(np.asarray(output), image)
    swapped = Image.fromarray(image).filter(
        ImageFilter.Color3DLUT(16, cube[..., ::-1].flatten())
    )
    assert np.array_equal(np.asarray(swapped), image[..., ::-1])
