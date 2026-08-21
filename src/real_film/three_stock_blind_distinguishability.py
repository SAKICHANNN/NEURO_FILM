"""Blind three-way stock assignment for the SF3.A5 confirmation gate."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "neuro-film.sf3-a5-three-stock-blind-distinguishability-contract.v1"


class ThreeStockBlindError(ValueError):
    """Raised for incomplete or malformed blind stock assignments."""


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "FROZEN_BEFORE_PHYSICAL_TARGET_ADMISSION_OR_RENDER"
    ):
        raise ThreeStockBlindError("unsupported or unfrozen SF3.A5 contract")
    if value.get("required_stocks") != [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]:
        raise ThreeStockBlindError("SF3.A5 stock inventory drift")
    return value


def build_mapping(
    scene_ids: Iterable[str], stock_ids: Iterable[str], *, seed: str
) -> list[dict[str, Any]]:
    scenes, stocks = list(scene_ids), list(stock_ids)
    if (
        len(scenes) != 4
        or len(set(scenes)) != 4
        or len(stocks) != 3
        or len(set(stocks)) != 3
        or not seed
    ):
        raise ThreeStockBlindError("SF3.A5 scene, stock or seed inventory drift")
    rows = []
    for round_index in range(1, 4):
        for scene_id in scenes:
            ordered = stocks.copy()
            token = hashlib.sha256(
                f"sf3-a5-v1:{seed}:{round_index}:{scene_id}".encode()
            ).digest()
            random.Random(int.from_bytes(token[:8], "big")).shuffle(ordered)
            rows.append(
                {
                    "round": round_index,
                    "scene_id": scene_id,
                    "label_to_stock": dict(zip(("A", "B", "C"), ordered, strict=True)),
                }
            )
    return rows


def adjudicate(
    mapping: list[dict[str, Any]],
    observations: Iterable[dict[str, Any]],
    *,
    gates: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        (row["round"], row["scene_id"]): row["label_to_stock"] for row in mapping
    }
    seen: set[tuple[int, str]] = set()
    correct = Counter()
    total = Counter()
    exact = 0
    round_correct = Counter()
    for row in observations:
        key = (int(row.get("round", 0)), str(row.get("scene_id", "")))
        assignment = row.get("label_to_stock")
        if (
            key not in expected
            or key in seen
            or not isinstance(assignment, dict)
            or set(assignment) != {"A", "B", "C"}
            or set(assignment.values()) != set(expected[key].values())
        ):
            raise ThreeStockBlindError(
                "unknown, duplicate or incomplete blind assignment"
            )
        seen.add(key)
        row_exact = True
        for label, truth in expected[key].items():
            total[truth] += 1
            if assignment[label] == truth:
                correct[truth] += 1
                round_correct[key[0]] += 1
            else:
                row_exact = False
        exact += int(row_exact)
    if seen != set(expected):
        raise ThreeStockBlindError("blind observations are incomplete")
    per_stock = {stock: correct[stock] / total[stock] for stock in sorted(total)}
    overall = sum(correct.values()) / sum(total.values())
    exact_rate = exact / len(expected)
    passing_rounds = sum(round_correct[index] / 12 >= 2 / 3 for index in (1, 2, 3))
    passed = (
        overall >= float(gates["minimum_overall_assignment_accuracy"])
        and min(per_stock.values())
        >= float(gates["minimum_per_stock_assignment_accuracy"])
        and exact_rate >= float(gates["minimum_exact_triplet_rate"])
        and passing_rounds
        >= int(gates["minimum_rounds_with_overall_accuracy_at_least_two_thirds"])
    )
    return {
        "overall_assignment_accuracy": overall,
        "per_stock_assignment_accuracy": per_stock,
        "exact_triplet_rate": exact_rate,
        "rounds_at_or_above_two_thirds": passing_rounds,
        "automatic_pass": passed,
    }


__all__ = ["ThreeStockBlindError", "adjudicate", "build_mapping", "load_contract"]
