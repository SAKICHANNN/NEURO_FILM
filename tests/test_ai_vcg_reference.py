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


def test_attention_conversion_uses_pinned_official_renaming_only():
    import ast

    source = (driver.MODEL / "runtime/diffusers/models/modeling_utils.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_convert_deprecated_attention_blocks"
    )
    namespace = {}
    exec(  # noqa: S102 - Tests the reviewed installed migration, not model pickle.
        compile(ast.Module([method], []), "official_migration", "exec"), namespace
    )

    class Block:
        _from_deprecated_attn_block = True

        def named_children(self):
            return []

    class Model:
        _convert_deprecated_attention_blocks = namespace[method.name]

        def named_children(self):
            return [("attention", Block())]

    original = {
        f"attention.{key}.{suffix}": np.arange(4, dtype=np.float32)
        for key in ("query", "key", "value", "proj_attn")
        for suffix in ("weight", "bias")
    }
    converted = driver.convert_legacy_attention(Model(), dict(original))
    for old, new in zip(
        ("query", "key", "value", "proj_attn"),
        ("to_q", "to_k", "to_v", "to_out.0"),
        strict=True,
    ):
        for suffix in ("weight", "bias"):
            assert (
                converted[f"attention.{new}.{suffix}"]
                is original[f"attention.{old}.{suffix}"]
            )
    assert driver.convert_legacy_attention(Model(), converted) == converted
