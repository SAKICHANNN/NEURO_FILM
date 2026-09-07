import copy
import json
from pathlib import Path

import pytest

from scripts.audit_ai_classneg_supervision import select_rows


def test_diagnostic_evidence_is_descriptive_and_complete():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (
            root / "docs/evidence/AI_CLASSNEG_SUPERVISION_DIAGNOSTIC_20260907.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["decision"] == "DIAGNOSTIC_COMPLETE_NO_PROMOTION"
    assert [row["index"] for row in evidence["rows"]] == list(range(16))
    for arm in ("target", "prediction"):
        assert sum(row[f"{arm}_preferred"] for row in evidence["rows"]) == 8
        assert evidence["review"][f"{arm}_preferred"] == 8
    assert evidence["review"]["population_preference"] is False
    assert evidence["review"]["full_resolution_product_clearance"] is False


def test_fixed_consumed_development_roles():
    manifest = {
        "config": {"styles": ["Cinema", "ClassNeg", "Velvia"]},
        "rows": [
            {
                "role": "paired_development_evaluation",
                "group": str(i),
                "files": [
                    {"path": f"train/{s}/{i}.png"}
                    for s in ("input", "Cinema", "ClassNeg")
                ],
            }
            for i in range(16)
        ],
    }
    assert len(select_rows(manifest)) == 16
    for path in ("test/input/0.png", "train/input/../0.png"):
        bad = copy.deepcopy(manifest)
        bad["rows"][0]["files"][0]["path"] = path
        with pytest.raises(ValueError, match="role path"):
            select_rows(bad)
    bad = copy.deepcopy(manifest)
    bad["rows"].append({"role": "paired_fit", "group": "0"})
    with pytest.raises(ValueError, match="Fit overlap"):
        select_rows(bad)
    with pytest.raises(ValueError, match="exactly16"):
        select_rows({**manifest, "rows": manifest["rows"][:-1]})
