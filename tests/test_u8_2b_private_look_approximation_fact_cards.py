from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u8_2b_private_look_approximation_fact_cards_v1.json"


def test_u8_2b_contract_freezes_private_claim_and_current_runtime() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    claim = config["claim_ceiling"]
    assert config["input"]["runtime_receipt_sha256"] == (
        "548a80583811329260083bd37d0abd122b033141b1c358ef700f07ee044ea756"
    )
    assert config["input"]["profile_sha256"] == (
        "09722cb48c55d1c4c460220c9f59b9ced81801c1b6fe856708a27f5cc99d39fe"
    )
    assert config["input"]["root_license_expected"] == "absent"
    assert config["required_card_order"] == [
        "product",
        "data",
        "model",
        "profiles",
        "release",
    ]
    assert config["required_profile_order"] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
        "generic_bw",
    ]
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["calibrated_stock_response"] is False
    assert claim["physical_film_reproduction"] is False
    assert claim["public_release"] is False
    assert claim["legal_clearance"] is False


def test_u8_2b_builder_surface_is_reserved() -> None:
    from src.inference import product_fact_cards as cards

    assert callable(cards.build_product_fact_cards)
    assert callable(cards.validate_product_fact_cards)
    assert callable(cards.publish_product_fact_cards)


def _inputs() -> dict[str, object]:
    from src.inference.product_look_catalog import list_product_looks

    config = json.loads(CONFIG.read_text("utf-8"))
    receipt_path = ROOT / config["input"]["runtime_receipt"]
    profile_path = ROOT / config["input"]["profile"]
    return {
        "config": config,
        "receipt": json.loads(receipt_path.read_text("utf-8")),
        "receipt_sha256": config["input"]["runtime_receipt_sha256"],
        "profile": json.loads(profile_path.read_text("utf-8")),
        "profile_sha256": config["input"]["profile_sha256"],
        "catalog": list_product_looks(),
        "evidence_sha256": {
            role: row["sha256"] for role, row in config["evidence"].items()
        },
        "root_license_present": False,
    }


def test_u8_2b_bundle_is_deterministic_complete_and_honest() -> None:
    from src.inference import product_fact_cards as cards

    inputs = _inputs()
    first = cards.build_product_fact_cards(**inputs)
    reordered = dict(inputs)
    reordered["evidence_sha256"] = dict(
        reversed(list(inputs["evidence_sha256"].items()))
    )
    second = cards.build_product_fact_cards(**reordered)
    assert cards.canonical_json(first) == cards.canonical_json(second)
    assert list(first["cards"]) == first["card_order"]
    assert [row["look_id"] for row in first["cards"]["profiles"]] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
        "generic_bw",
    ]
    assert first["cards"]["release"]["root_license"] == "unresolved"
    assert first["cards"]["data"]["product_training_dataset"] is None
    assert first["cards"]["model"]["learned_final_rgb_generator"] is False


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda data: data.update(root_license_present=True), "root licence"),
        (
            lambda data: data.update(receipt_sha256="0" * 64),
            "runtime receipt",
        ),
        (
            lambda data: data["catalog"].reverse(),
            "catalog order",
        ),
    ],
)
def test_u8_2b_frozen_input_drift_rejects(mutator, match: str) -> None:
    from src.inference import product_fact_cards as cards

    inputs = _inputs()
    inputs["catalog"] = list(inputs["catalog"])
    mutator(inputs)
    with pytest.raises(cards.ProductFactCardError, match=match):
        cards.build_product_fact_cards(**inputs)


def test_u8_2b_overclaim_rejects() -> None:
    from src.inference import product_fact_cards as cards

    inputs = _inputs()
    bundle = cards.build_product_fact_cards(**inputs)
    overclaim = deepcopy(bundle)
    overclaim["cards"]["profiles"][0]["calibrated_stock_response"] = True
    with pytest.raises(cards.ProductFactCardError, match="profile claim"):
        cards.validate_product_fact_cards(overclaim, config=inputs["config"])


def test_u8_2b_publication_is_create_only(tmp_path: Path) -> None:
    from src.inference import product_fact_cards as cards

    payload = cards.canonical_json(cards.build_product_fact_cards(**_inputs()))
    destination = tmp_path / "cards.json"
    cards.publish_product_fact_cards(destination, payload)
    assert destination.read_bytes() == payload
    with pytest.raises(FileExistsError):
        cards.publish_product_fact_cards(destination, payload)
    assert destination.read_bytes() == payload
    assert not list(tmp_path.glob(".cards.json.u8-2b-*.stage"))


def test_u8_2b_formal_audit_binds_owned_sources() -> None:
    from scripts import audit_u8_2b_private_look_approximation_fact_cards as audit

    assert {path.as_posix() for path in audit.SOURCE_PATHS} == {
        "configs/u8_2b_private_look_approximation_fact_cards_v1.json",
        "docs/planning/U8_2B_PRIVATE_LOOK_APPROXIMATION_FACT_CARDS_CONTRACT.md",
        "scripts/audit_u8_2b_private_look_approximation_fact_cards.py",
        "scripts/build_private_product_fact_cards.py",
        "src/inference/product_fact_cards.py",
        "tests/test_u8_2b_private_look_approximation_fact_cards.py",
    }


def test_u8_2b_forward_reverse_reports_are_exact(tmp_path: Path) -> None:
    from scripts import audit_u8_2b_private_look_approximation_fact_cards as audit

    forward = audit.run_audit(CONFIG, "forward")
    reverse = audit.run_audit(CONFIG, "reverse")
    assert audit.canonical_json(forward) == audit.canonical_json(reverse)
    assert forward["status"] == "PASS_PRIVATE_LOOK_APPROXIMATION_FACT_CARDS"
    assert all(forward["gates"].values())
