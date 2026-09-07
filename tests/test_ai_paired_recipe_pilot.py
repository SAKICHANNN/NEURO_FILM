import hashlib
import json

import numpy as np
import pytest
from PIL import Image, ImageCms

from scripts import run_ai_paired_recipe_pilot as pilot


def test_srgb_decode_and_missing_profile_reject(tmp_path):
    root = tmp_path
    folder = root / "train" / "input"
    folder.mkdir(parents=True)
    path = folder / "sample.png"
    a = np.full((8, 8, 3), 127, dtype=np.uint8)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(a).save(path, icc_profile=profile)
    row = {"path": "train/input/sample.png", "sha256": pilot.sha(path)}
    decoded = pilot.decode(root, row)
    np.testing.assert_allclose(decoded.numpy(), 127 / 255, rtol=0, atol=1e-7)
    assert pilot.sha(path) == row["sha256"]
    Image.fromarray(a).save(path)
    row["sha256"] = pilot.sha(path)
    with pytest.raises(ValueError, match="Missing RGB profile"):
        pilot.decode(root, row)
    with pytest.raises(ValueError, match="Forbidden path"):
        pilot.decode(root, {"path": "test/input/sample.png", "sha256": "unused"})


def test_freeze_deduplicates_groups_and_never_includes_lockbox(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    source, inventory = [], []
    for i, group in enumerate(["a", "a", "b", "c"]):
        name = f"{i}.png"
        source.append(
            {
                "research_pool": "source_train",
                "distributed_split": "train",
                "duplicate_cluster_id": group,
                "content_id": name,
                "path": f"train/input/{name}",
                "sha256": str(i),
                "bytes": 1,
            }
        )
        inventory.append(
            {"path": f"train/ClassNeg/{name}", "sha256": str(i), "bytes": 1}
        )
    source_path, inventory_path = (
        tmp_path / "source.jsonl",
        tmp_path / "inventory.jsonl",
    )
    source_path.write_text("\n".join(json.dumps(row) for row in source))
    inventory_path.write_text("\n".join(json.dumps(row) for row in inventory))
    cfg = {
        "source_manifest": source_path.name,
        "source_manifest_sha256": pilot.sha(source_path),
        "inventory": inventory_path.name,
        "styles": ["ClassNeg"],
        "fit_count": 2,
        "evaluation_count": 1,
    }
    result = pilot.freeze(cfg)
    assert [r["group"] for r in result["rows"]] == ["a", "b", "c"]
    assert [r["role"] for r in result["rows"]] == [
        "paired_fit",
        "paired_fit",
        "paired_development_evaluation",
    ]
    source[0]["research_pool"] = "internal_dev_lockbox"
    source_path.write_text("\n".join(json.dumps(row) for row in source))
    cfg["source_manifest_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Forbidden role"):
        pilot.freeze(cfg)
