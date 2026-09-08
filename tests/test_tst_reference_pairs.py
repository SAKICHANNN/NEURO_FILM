import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("pairs", Path(__file__).resolve().parents[1]/"scripts/prepare_tst_reference_pairs.py")
pairs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pairs)


def test_geometry_identity_and_mismatch_no_registration():
    image = np.random.default_rng(48).integers(0, 65536, (80, 96, 3), dtype=np.uint16)
    result = pairs.geometry(image, image)
    assert all(r["best"]["dx"] == r["best"]["dy"] == 0 for r in result["regions"])
    before = image.copy()
    assert pairs.geometry(image, image[:-1])["status"] == "DIMENSION_MISMATCH_NO_PIXEL_PAIRING"
    np.testing.assert_array_equal(image, before)


def test_canonical_hash_and_cross_split_collision():
    image = np.zeros((20, 30, 3), dtype=np.uint16)
    value = pairs.canonical_hash(image)
    assert value == pairs.canonical_hash(image.copy())
    assert value != pairs.canonical_hash(image.reshape(30, 20, 3))
    rows = [{"path": "a", "split": "training", "role": "P", "canonical": value},
            {"path": "b", "split": "diagnostic_development", "role": "R", "canonical": value}]
    assert len(pairs.cross_split_collisions(rows, "canonical")) == 1
    assert not pairs.cross_split_collisions(rows[:1], "canonical")


def test_constant_gradients_are_unknown_not_wrong_mapping():
    image = np.zeros((80, 96, 3), dtype=np.uint16)
    result = pairs.geometry(image, image+10000)
    assert result["status"] == "SAME_NATIVE_DIMENSIONS_DIAGNOSTIC_ONLY"
    assert all(r["best"] is None for r in result["regions"])
