"""Blind three-way stock assignment for the SF3.A5 confirmation gate."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable
from typing import Any


class ThreeStockBlindError(ValueError):
    """Raised for incomplete or malformed blind stock assignments."""


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
    mapping: list[dict[str, Any]], observations: Iterable[dict[str, Any]]
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
        overall >= 0.75
        and min(per_stock.values()) >= 2 / 3
        and exact_rate >= 0.5
        and passing_rounds >= 2
    )
    return {
        "overall_assignment_accuracy": overall,
        "per_stock_assignment_accuracy": per_stock,
        "exact_triplet_rate": exact_rate,
        "rounds_at_or_above_two_thirds": passing_rounds,
        "automatic_pass": passed,
    }


__all__ = ["ThreeStockBlindError", "adjudicate", "build_mapping"]
