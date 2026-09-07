"""Development composition mechanics; not photographic-value evidence."""

import json
from pathlib import Path

import numpy as np

from scripts.compare_creative_looks_v2 import print_finish
from src.color_engine.creative_print_look import CreativePrintLook, render_print_look


def test_dense_finish_repeat_ownership_and_bounds():
    config = json.loads(Path("configs/creative_print_development_v4.json").read_text())
    x = np.random.default_rng(17).random((32, 39, 3), dtype=np.float32)
    before = x.copy()
    grade = render_print_look(x, CreativePrintLook(**config["looks"]["dense_print"]))
    grade_before = grade.copy()
    a = print_finish(grade, config["finish"])
    b = print_finish(grade, config["finish"])
    assert np.array_equal(a, b)
    assert np.array_equal(x, before) and np.array_equal(grade, grade_before)
    assert a.dtype == np.float32 and a.shape == x.shape
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert not np.shares_memory(a, grade)
    assert np.abs(a - grade).mean() > 0


def test_dense_colour_is_distinct_without_texture():
    config = json.loads(Path("configs/creative_print_development_v4.json").read_text())
    x = np.full((5, 7, 3), 0.5, dtype=np.float32)
    dense = render_print_look(x, CreativePrintLook(**config["looks"]["dense_print"]))
    copper = render_print_look(x, CreativePrintLook(**config["looks"]["copper_light"]))
    assert np.abs(dense - copper).mean() > 0.06
