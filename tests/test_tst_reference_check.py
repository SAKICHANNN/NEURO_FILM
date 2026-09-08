import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_entry", ROOT/"scripts/run_tst_reference_check.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_prediction_only_accepts_source_reference_and_real_model_roundtrip():
    assert list(inspect.signature(entry.predict_operators).parameters) == [
        "model", "source_stats", "source_vgg", "reference_stats", "reference_vgg"]
    model = entry.load_model(ROOT/"outputs/tst_reference_response_v1/fit/model.json")
    cache = np.load(ROOT/"outputs/tst_reference_response_v1/features/features.npz", allow_pickle=False)
    source = [cache["00_X_stats"], cache["00_X_vgg"]]
    refs = [[cache[f"00_R{j}_stats"], cache[f"00_R{j}_vgg"]] for j in (1, 2)]
    first = entry.predict_operators(model, *source, *refs[0])
    second = entry.predict_operators(model, *source, *refs[1])
    swapped = [entry.predict_operators(model, *source, *r) for r in refs[::-1]]
    np.testing.assert_array_equal(first["A"], second["A"])
    for arm in "ABC":
        assert first[arm].shape == (343, 3) and np.isfinite(first[arm]).all()
        np.testing.assert_array_equal(swapped[0][arm], second[arm])
        np.testing.assert_array_equal(swapped[1][arm], first[arm])


def test_correct_swapped_and_source_only_switch_metrics():
    rng = np.random.default_rng(20260908)
    x = rng.uniform(.2, .6, size=(100, 3))
    y1, y2 = x.copy(), x.copy()
    y1[:, 0] += .1
    y2[:, 2] += .1
    good = entry.switching([y1, y2], [y1, y2])
    bad = entry.switching([y2, y1], [y1, y2])
    null = entry.switching([x, x], [y1, y2])
    assert good["pass"] and abs(good["margin"]-1) < 1e-12
    assert abs(good["cosine"]-1) < 1e-12 and abs(good["norm_ratio"]-1) < 1e-12
    assert bad["margin"] < 0 and not bad["pass"]
    assert abs(null["margin"]) < 1e-12 and not null["pass"]


def test_source_crop_bounds_and_explicit_absence():
    decoded = {f"{g:02d}_X": {"srgb16_sha256": str(g), "encoded_size": [640, 480]} for g in range(24, 32)}
    crops = {"status": "SOURCE_ONLY_CROPS_FROZEN", "groups": [
        {"group": g, "source_sha256": str(g),
         "regions": {"face": None, "text": None, "dark": [0, 0, 128, 128], "bright": [400, 200, 128, 128]},
         "absence_reasons": {"face": "No face in source", "text": "No text in source"}} for g in range(24, 32)]}
    entry.verify_crops(crops, decoded)
    crops["groups"][0]["regions"]["dark"] = [600, 0, 128, 128]
    with pytest.raises(AssertionError):
        entry.verify_crops(crops, decoded)
