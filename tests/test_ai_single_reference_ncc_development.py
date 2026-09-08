import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pillow_lut import identity_table

from scripts import run_ai_single_reference_ncc_development as driver


def identity_lut():
    return np.asarray(identity_table(16).table, dtype=np.float32).reshape(64, 64, 3)


def test_ncc_conditioning_uses_unmodified_input_and_true_flag():
    content = Image.new("RGB", (7, 5), (100, 150, 200))
    style = np.zeros((512, 512, 3), dtype=np.uint8)

    def preprocess(images, actual_style, size, ncc):
        assert ncc is True and size == 512 and actual_style is style
        assert np.array_equal(images[0], np.asarray(content))
        return images, "NCC-conditioning"

    assert driver.ncc_conditioning(content, style, preprocess) == "NCC-conditioning"
    with pytest.raises(ValueError, match="changed"):
        driver.ncc_conditioning(content, style, lambda x, *a: (x + 1, None))


@pytest.mark.parametrize("kind", ["shape", "dtype", "nonfinite"])
def test_ncc_render_rejects_invalid_lut(kind):
    lut = identity_lut()
    if kind == "shape":
        lut = lut[:1]
    elif kind == "dtype":
        lut = lut.astype(np.float64)
    else:
        lut[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="representation"):
        driver.render_ncc(Image.new("RGB", (2, 2)), lut)


def test_ncc_identity_lut_keeps_original_pixels():
    content = Image.fromarray(np.random.default_rng(4).integers(0, 256, (13, 17, 3), dtype=np.uint8))
    assert np.array_equal(driver.render_ncc(content, identity_lut()), content)


@pytest.mark.parametrize("reason", ["forward", "event", "unfinished", "owner", "deferred"])
def test_ncc_retry_only_resource_defer_with_no_forward(tmp_path, reason):
    (tmp_path / "scope.json").write_text(json.dumps({"round": driver.ROUND, "config_sha256": "hash"}))
    attempt = tmp_path / "attempt-0001"
    attempt.mkdir()
    if reason != "unfinished":
        (attempt / "report.json").write_text(json.dumps({"status": "DEFERRED_GPU_BUSY", "new_forwards": int(reason == "forward")}))
    if reason == "event":
        driver.baseline.record_event(attempt, {"stage": "FORWARD_START"})
    if reason == "deferred":
        driver.validate_attempts(tmp_path, "hash")
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            driver.validate_attempts(tmp_path, "other" if reason == "owner" else "hash")


def test_worker_eight_ncc_forwards_cache_and_all_36_baseline_bytes(tmp_path, monkeypatch):
    cfg = json.loads((driver.ROOT / driver.CONFIG).read_text())
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    out = tmp_path / cfg["output"] / "attempt-0001"
    out.mkdir(parents=True)
    originals = tmp_path / "originals"
    originals.mkdir()
    rows, records = [], []
    for i in range(9):
        image = Image.new("RGB", (8, 6), (i + 20, i + 40, i + 60))
        content_path = tmp_path / f"source-{i}.png"
        image.save(content_path)
        rows.append({"index": i, "path": str(content_path), "sha256": driver.sha(content_path)})
        arms = {}
        for arm in cfg["arms"][:4]:
            name = f"{i:02d}_{arm}.png"
            image.save(originals / name)
            arms[arm] = {"path": name, "sha256": driver.sha(originals / name)}
        records.append({"arms": arms})
    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (8, 6)).save(ref_path)
    lut_path = tmp_path / "cached.npy"
    np.save(lut_path, identity_lut())
    cfg["cache"]["output_sha256"] = records[2]["arms"]["identity"]["sha256"]
    bundle = {"config": cfg, "config_sha256": "hash", "implementation_sha256": {},
              "baseline_implementation_sha256": {}, "parent": {}, "sources": rows,
              "reference": {"path": "ref.png", "sha256": driver.sha(ref_path), "source_page": "reference"},
              "baseline_report": {"rows": records}, "baseline_directory": str(originals), "cache_lut": str(lut_path)}
    lock_path = out / "execution_lock.json"
    lock_path.write_text(json.dumps({k: bundle[k] for k in ["config_sha256", "implementation_sha256", "baseline_implementation_sha256"]}))
    monkeypatch.setattr(driver, "verify_assets", lambda: bundle)
    snapshot = {"available": True, "free_mib": 10000, "used_mib": 1000, "utilization_percent": 0, "known_compute_pids": []}
    monkeypatch.setattr(driver.baseline, "gpu_snapshot", lambda: snapshot)
    cuda = SimpleNamespace(set_per_process_memory_fraction=lambda *a: None,
                           get_device_properties=lambda *a: SimpleNamespace(total_memory=12 * 1024**3),
                           reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
                           max_memory_reserved=lambda: 0, synchronize=lambda: None, is_initialized=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    seen = []

    def preprocess(images, style, size, ncc):
        assert ncc is True
        return images, images[0]

    def predict(small, style):
        seen.append(int(small[0, 0, 0]) - 20)
        return identity_lut()

    from scripts import run_ai_vcg_reference as parent
    monkeypatch.setattr(parent, "initialize", lambda cfg: (predict, preprocess))
    assert driver.run_worker(lock_path) == 0
    report = json.loads((out / "report.json").read_text())
    assert seen == [0, 1, 3, 4, 5, 6, 7, 8]
    assert report["new_forwards"] == 8 and report["cached_rows"] == 1
    assert len(report["rows"]) == 9
    for i, row in enumerate(report["rows"]):
        assert len(row["arms"]) == 5
        for arm in cfg["arms"][:4]:
            name = row["arms"][arm]["path"]
            assert (out / name).read_bytes() == (originals / name).read_bytes()
        assert row["arms"]["learned_no_precorrection"]["sha256"] == records[i]["arms"]["identity"]["sha256"]
    assert hashlib.sha256((out / "02_learned_no_precorrection.png").read_bytes()).hexdigest() == cfg["cache"]["output_sha256"]
