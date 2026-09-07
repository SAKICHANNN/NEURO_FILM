import copy

import pytest

from scripts.audit_ai_classneg_supervision import select_rows


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
