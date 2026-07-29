from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.fresh_native_standard_adjudication import (
    adjudicate_fresh_native_standard,
)
from src.eval.fresh_native_standard_confirmation import ARMS


def _write_json(root: Path, name: str, payload: object) -> dict[str, str]:
    path = root / name
    encoded = json.dumps(payload, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return {
        "path": name,
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    experiment = {
        "comparison": {
            "arms": list(ARMS),
            "preference_gate": {
                "minimum_native_standard_round_wins_vs_ao6": 2,
                "minimum_native_standard_total_choices_vs_ao6": 0.5,
                "ties_allowed": True,
            },
        }
    }
    rows = [
        {
            "source_id": source_id,
            "arm_id": arm,
            "output_sha256": hashlib.sha256(
                f"{source_id}:{arm}".encode()
            ).hexdigest(),
        }
        for source_id in ("one", "two", "three")
        for arm in ARMS
    ]
    automatic = {
        "automatic_gate_pass": True,
        "blind_review_allowed": True,
        "stable_evidence_id": "4" * 64,
        "rows": rows,
    }
    mapping = [
        {
            "source_id": source_id,
            "A": ARMS[0],
            "B": ARMS[1],
            "C": ARMS[2],
        }
        for source_id in ("one", "two", "three")
    ]
    observations = {
        "mapping_unread_when_recorded": True,
        "rounds": [
            {
                "round": 1,
                "votes": {"one": "C", "two": "C", "three": "A"},
            },
            {
                "round": 2,
                "votes": {"one": "C", "two": "B", "three": "C"},
            },
            {
                "round": 3,
                "votes": {"one": "B", "two": "B", "three": "tie"},
            },
        ],
        "contact_sheet_severe_findings": {
            "confirmed_new_severe_count": 0,
        },
    }
    return {
        "schema": "neuro_film.u6_p8bp_fixed_arm_adjudication.v1",
        "experiment": _write_json(tmp_path, "experiment.json", experiment),
        "automatic_reports": [
            _write_json(tmp_path, f"run_{index}.json", automatic)
            for index in (1, 2)
        ],
        "blind_observations": _write_json(
            tmp_path,
            "observations.json",
            observations,
        ),
        "blind_mappings": [
            _write_json(tmp_path, f"mapping_{index}.json", mapping)
            for index in (1, 2, 3)
        ],
        "full_resolution_review": {
            "confirmed_new_severe_count": 0,
        },
        "claim_ceiling": "test",
    }


def test_adjudication_retains_native_only_after_stable_pairwise_gain(
    tmp_path: Path,
) -> None:
    report = adjudicate_fresh_native_standard(
        root=tmp_path,
        config=_fixture(tmp_path),
    )
    assert report["native_vs_ao6_round_wins"] == {
        ARMS[1]: 1,
        ARMS[2]: 2,
        "tie": 0,
    }
    assert report["arm_total_choices"] == {
        ARMS[0]: 1,
        ARMS[1]: 3,
        ARMS[2]: 4,
        "tie": 1,
    }
    assert report["preference_gate_pass"]
    assert report["severe_artifact_gate_pass"]
    assert report["decision"] == (
        "retain_native_standard_as_product_challenger"
    )


def test_adjudication_rejects_native_on_any_severe_finding(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    config["full_resolution_review"][
        "confirmed_new_severe_count"
    ] = 1
    report = adjudicate_fresh_native_standard(
        root=tmp_path,
        config=config,
    )
    assert not report["severe_artifact_gate_pass"]
    assert report["decision"] == "reject_native_standard_severe_failure"


def test_adjudication_rejects_nonidentical_automatic_replay(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    changed = json.loads(
        (tmp_path / config["automatic_reports"][1]["path"]).read_text()
    )
    changed["rows"][0]["output_sha256"] = "5" * 64
    config["automatic_reports"][1] = _write_json(
        tmp_path,
        "run_changed.json",
        changed,
    )
    try:
        adjudicate_fresh_native_standard(
            root=tmp_path,
            config=config,
        )
    except ValueError as error:
        assert "output inventories" in str(error)
    else:
        raise AssertionError("nonidentical replays must fail closed")
