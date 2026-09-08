import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import imagecodecs
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("curation", ROOT / "scripts/prepare_tst_crossed_corpus.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LIMITS = {"decode_base_bytes": 256 * 1024**2, "rss_bytes": 2 * 1024**3}


def test_uint16_canonical_stream_and_decode(tmp_path):
    array = np.random.default_rng(4).integers(0, 65536, (137, 49, 3), dtype=np.uint16)
    path = tmp_path / "native.tif"
    tifffile.imwrite(path, array, photometric="rgb")
    header = module.inspect_header(path, LIMITS)
    assert header["eligible_decode"]
    decoded, policy = module.decode(path, header)
    np.testing.assert_array_equal(decoded, array)
    assert "ASSUMPTION" in policy
    prefix = json.dumps({"shape": list(array.shape), "dtype": "uint16-le", "space": "encoded_sRGB"}, sort_keys=True).encode()
    assert module.canonical_hash(decoded) == hashlib.sha256(prefix + array.astype("<u2").tobytes()).hexdigest()


def test_tagged_png_and_jpeg_match_baseline_numeric_contract(tmp_path):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    values = np.random.default_rng(9).integers(0, 256, (33, 27, 3), dtype=np.uint8)
    for suffix in ("png", "jpg"):
        path = tmp_path / ("tagged." + suffix)
        Image.fromarray(values).save(path, icc_profile=profile)
        header = module.inspect_header(path, LIMITS)
        assert header["eligible_decode"]
        decoded, policy = module.decode(path, header)
        if suffix == "png":
            native = imagecodecs.png_decode(path.read_bytes())
        else:
            with Image.open(path) as image:
                native = np.asarray(image).copy()
        expected = imagecodecs.cms_transform(native, profile, profile, colorspace="RGB", outcolorspace="RGB", outdtype=np.uint16, intent=1)
        np.testing.assert_array_equal(decoded, expected)
        assert "Embedded ICC" in policy


def test_rgb16_png_preserves_precision(tmp_path):
    values = np.arange(15 * 17 * 3, dtype=np.uint16).reshape(15, 17, 3) * 71
    path = tmp_path / "rgb16.png"
    path.write_bytes(imagecodecs.png_encode(values))
    header = module.inspect_header(path, LIMITS)
    assert header["bits"] == 16 and header["eligible_decode"]
    np.testing.assert_array_equal(module.decode(path, header)[0], values)


def test_alpha_orientation_multiframe_and_resource_are_explicit(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (12, 12)).save(path)
    assert "STRICT_RGB_REQUIRED" in module.inspect_header(path, LIMITS)["reasons"]
    path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (12, 12)).save(path, exif=exif)
    assert "NONIDENTITY_ORIENTATION" in module.inspect_header(path, LIMITS)["reasons"]
    path = tmp_path / "pages.tif"
    with tifffile.TiffWriter(path) as writer:
        for _ in range(2):
            writer.write(np.zeros((12, 12, 3), dtype=np.uint8), photometric="rgb")
    assert not module.inspect_header(path, LIMITS)["eligible_decode"]
    path = tmp_path / "small.png"
    Image.new("RGB", (12, 12)).save(path)
    header = module.inspect_header(path, {"rss_bytes": 1, "decode_base_bytes": 0})
    assert header["reasons"] == ["RESOURCE_DEFERRED"]


def test_non_srgb_untagged_png_gamma_not_silently_assumed(tmp_path):
    path = tmp_path / "gamma.png"
    Image.new("RGB", (12, 12)).save(path)
    data = path.read_bytes()
    payload = struct.pack(">I", 100000)
    chunk = struct.pack(">I", 4) + b"gAMA" + payload + struct.pack(">I", zlib.crc32(b"gAMA" + payload))
    path.write_bytes(data[:33] + chunk + data[33:])
    assert "UNTAGGED_NON_SRGB_GAMMA" in module.inspect_header(path, LIMITS)["reasons"]


def test_compact_only_and_resume_artifact_identity(tmp_path):
    values = np.full((41, 77, 3), 32768, dtype=np.uint16)
    image_bytes, descriptor = module.compact(values)
    assert len(descriptor["dhash64"]) == 16
    output = tmp_path / "preview.png"
    output.write_bytes(image_bytes)
    with Image.open(output) as image:
        assert max(image.size) <= 256
    identity = {"config_sha256": "a", "receipt_sha256": "b"}
    item = {"identity": identity, "encoded_sha256": "c", "artifacts": [{"path": "preview.png", "sha256": module.digest(output)}]}
    assert module.reusable(item, identity, {"sha256": "c"}, tmp_path)
    with pytest.raises(ValueError, match="RESUME_IDENTITY"):
        module.reusable(item, {"config_sha256": "changed"}, {"sha256": "c"}, tmp_path)
    output.write_bytes(b"bad")
    with pytest.raises(ValueError, match="RESUME_ARTIFACT"):
        module.reusable(item, identity, {"sha256": "c"}, tmp_path)


def test_receipt_incomplete_and_missing_hash_fail_closed(tmp_path):
    files = {"content/a.jpg": {"path": "content/a.jpg", "bytes": 3, "sha256": "abc"}}
    config = {"receipt": "receipt.json", "acquisition_config": "config.json"}
    (tmp_path / "config.json").write_text("{}")
    module.save(tmp_path / "receipt.json", {"status": "DOWNLOADING"})
    with pytest.raises(ValueError, match="RECEIPT_NOT_COMPLETE"):
        module.bind_receipt(config, {"expected_bytes": 3}, files, tmp_path)
    receipt = {"status": "ACQUIRED_HASH_VERIFIED_NOT_DECODED", "config_sha256": module.digest(tmp_path / "config.json"),
               "files": list(files.values()), "total_verified_bytes": 3, "accounting": {"cpu_seconds": 1, "wall_seconds": 2}}
    module.save(tmp_path / "receipt.json", receipt)
    assert module.bind_receipt(config, {"expected_bytes": 3}, files, tmp_path)["receipt_sha256"]
    receipt["files"][0]["sha256"] = "wrong"
    module.save(tmp_path / "receipt.json", receipt)
    with pytest.raises(ValueError, match="RECEIPT_FILE_IDENTITY"):
        module.bind_receipt(config, {"expected_bytes": 3}, {"content/a.jpg": {"path": "content/a.jpg", "bytes": 3, "sha256": "abc"}}, tmp_path)
    with pytest.raises(ValueError, match="UNSAFE_RELATIVE"):
        module.confined(tmp_path, "../outside")


def test_exact_collision_quarantines_all_component_members():
    records = [{"path": name, "split": split, "canonical_pixel_sha256": "same"} for name, split in (("a", "fit"), ("b", "monitor"), ("c", "fit"))]
    result = module.duplicate_summary(records)
    assert result["cross_split_quarantine_components"][0]["paths"] == ["a", "b", "c"]
    assert "near-duplicate" in result["claim"]


def synthetic_corpus(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    for name in ("a.png", "b.png", "reference.png"):
        Image.new("RGB", (21, 19), color=(40, 80, 120)).save(originals / name)
    rows = [{"path": name, "bytes": (originals / name).stat().st_size,
             "sha256": module.digest(originals / name)} for name in ("a.png", "b.png", "reference.png")]
    acquisition = {"files": rows, "expected_bytes": sum(r["bytes"] for r in rows)}
    module.save(tmp_path / "acquisition.json", acquisition)
    module.save(tmp_path / "paths.json", {"head_now": rows, "original_Y_only_deferred": [{"path": "forbiddenY.png"}],
                                          "real_reserved_no_network": [{"path": "forbiddenReal.png"}]})
    module.save(tmp_path / "sources.json", [{"path": "a.png", "split": "fit", "roles": ["fit:query_X"]},
                                            {"path": "b.png", "split": "monitor", "roles": ["monitor:donor_P"]}])
    module.save(tmp_path / "receipt.json", {"status": "ACQUIRED_HASH_VERIFIED_NOT_DECODED", "files": rows,
        "config_sha256": module.digest(tmp_path / "acquisition.json"), "total_verified_bytes": acquisition["expected_bytes"],
        "accounting": {"cpu_seconds": 1, "wall_seconds": 2}})
    config = {**LIMITS, "pins": {}, "acquisition_config": "acquisition.json", "source_queue": "sources.json",
              "path_manifest": "paths.json", "originals": "originals", "receipt": "receipt.json", "expected_files": 3,
              "expected_sources": 2, "output": "output", "threads": 2, "preview_cache_bytes": 1024**2}
    module.save(tmp_path / "config.json", config)
    return config


def test_synthetic_s0_s1_and_exact_resume(tmp_path, monkeypatch):
    config = synthetic_corpus(tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    # Default arguments are fixed at import, so route this fixture's manifest root explicitly.
    original_manifest, original_receipt = module.manifest, module.bind_receipt
    monkeypatch.setattr(module, "manifest", lambda c: original_manifest(c, tmp_path))
    monkeypatch.setattr(module, "bind_receipt", lambda c, a, f: original_receipt(c, a, f, tmp_path))
    config_path = tmp_path / "config.json"
    assert module.preflight(config_path, "sources")[1]["status"] == "WAIT_COMPLETE_HEADERS"
    for stage in ("headers", "sources"):
        checks = module.preflight(config_path, stage)[1]
        assert checks["status"] == "READY"
        attempt = tmp_path / "output" / stage / "attempts/0001"
        module.save(attempt / "identity.json", checks["identity"])
        module.worker(config_path, stage, attempt)
        report = module.read(attempt / "worker_report.json")
        module.save(tmp_path / "output" / stage / "report.json", report)
        assert report["processed"] == (3 if stage == "headers" else 2)
        module.worker(config_path, stage, attempt)
        assert module.read(attempt / "worker_report.json")["processed"] == report["processed"]
    report = module.read(tmp_path / "output/sources/report.json")
    assert report["duplicates"]["cross_split_quarantine_components"][0]["paths"] == ["a.png", "b.png"]
    assert len(list((tmp_path / "output/sources/previews").glob("*.png"))) == 2
    assert not list((tmp_path / "output").rglob("*.tif"))
    assert config["expected_sources"] == 2


def test_manifest_rejects_reserved_or_after_as_source(tmp_path):
    config = synthetic_corpus(tmp_path)
    sources = module.read(tmp_path / "sources.json")
    sources[0] = {"path": "reference.png", "split": "fit", "roles": ["fit:donor_R"]}
    module.save(tmp_path / "sources.json", sources)
    with pytest.raises(ValueError, match="SOURCE_ROLE_NOT_X_OR_P"):
        module.manifest(config, tmp_path)
    sources[0] = {"path": "forbiddenY.png", "split": "fit", "roles": ["fit:query_X"]}
    module.save(tmp_path / "sources.json", sources)
    with pytest.raises(ValueError, match="SOURCE_OUTSIDE_ALLOWLIST"):
        module.manifest(config, tmp_path)


def test_failed_attempt_cpu_budget_cannot_reset(tmp_path, monkeypatch):
    config = {"output": "output", "stages": {"headers": {"cpu_seconds": 4, "wall_seconds": 10}},
              "prior_reserve_wall_seconds": 1, "study_wall_seconds": 100}
    module.save(tmp_path / "output/headers/attempts/0001/accounting.json",
                {"stage": "headers", "aggregate_cpu_seconds": 5, "active_wall_seconds": 1, "failure": "CPU_LIMIT"})
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "preflight", lambda path, stage: (config, {"status": "READY", "identity": {"download_accounting": {"wall_seconds": 1}}}))
    with pytest.raises(ValueError, match="CUMULATIVE_BUDGET_EXHAUSTED"):
        module.launch(tmp_path / "config.json", "headers")
    assert not (tmp_path / "output/running.lock").exists()


def test_supervisor_records_actual_synthetic_cpu_failure(tmp_path, monkeypatch):
    config = {"output": "output", "stages": {"headers": {"cpu_seconds": .05, "wall_seconds": 10}},
              "prior_reserve_wall_seconds": 1, "prior_reserve_cpu_seconds": 1,
              "study_wall_seconds": 100, "rss_bytes": 2 * 1024**3}
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "preflight", lambda path, stage: (config, {"status": "READY", "identity": {"download_accounting": {"wall_seconds": 1}}}))
    real_popen = subprocess.Popen
    monkeypatch.setattr(module.subprocess, "Popen", lambda command, **kwargs: real_popen([sys.executable, "-c", "while True: pass"], **kwargs))
    assert module.launch(tmp_path / "config.json", "headers") != 0
    record = module.read(tmp_path / "output/headers/attempts/0001/accounting.json")
    assert record["failure"] == "CPU_LIMIT" and record["aggregate_cpu_seconds"] >= .05
    assert (tmp_path / "output/headers/attempts/0001/failure.json").exists()
