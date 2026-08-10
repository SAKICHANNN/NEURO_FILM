"""Hard trusted-support selector test following the BZ0 proxy failure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.eval.fivek_proxy_exploitation_control import (
    _aggregate,
    _evaluate_rows,
    _rewards,
    _sha256,
    apply_operator_bank,
    candidate_bank,
    canonical_bytes,
    load_manifest,
    load_pair,
    scorer_features,
    split_rows,
)
from src.eval.fivek_proxy_exploitation_control import (
    validate_contract as validate_parent_contract,
)


class FiveKTrustedSupportError(ValueError):
    """Raised when the frozen BZ1 contract or parent evidence drifts."""


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKTrustedSupportError("contract is not frozen")
    parent = config["parent"]
    parent_contract_path = root / str(parent["contract"])
    parent_evidence_path = root / str(parent["evidence"])
    if not parent_contract_path.is_file() or _sha256(parent_contract_path) != str(
        parent["contract_sha256"]
    ):
        raise FiveKTrustedSupportError("parent contract drift")
    if not parent_evidence_path.is_file() or _sha256(parent_evidence_path) != str(
        parent["evidence_sha256"]
    ):
        raise FiveKTrustedSupportError("parent evidence drift")
    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
    parent_evidence = json.loads(parent_evidence_path.read_text(encoding="utf-8"))
    validate_parent_contract(root, parent_contract)
    if parent_evidence.get("decision") != parent["required_decision"]:
        raise FiveKTrustedSupportError("parent decision drift")
    failed_gate = str(parent["required_failed_gate"])
    if parent_evidence.get("gates", {}).get(failed_gate) is not False:
        raise FiveKTrustedSupportError("parent failed gate drift")
    if (
        parent_evidence["hidden_test"]["fixed_trusted_support_control"][
            "mean_true_gain"
        ]
        != parent["required_fixed_trusted_hidden_mean_gain"]
    ):
        raise FiveKTrustedSupportError("parent fixed-control metric drift")
    only = config["only_change"]
    if (
        only.get("broad_512_member_bank_used_for_selection") is not False
        or only.get("projection_or_refit") is not False
    ):
        raise FiveKTrustedSupportError("trusted-support boundary drift")
    return parent_contract, parent_evidence


def evaluate_gates(
    *,
    primary: Mapping[str, float],
    fixed: Mapping[str, float],
    oracle: Mapping[str, float],
    win_fraction: float,
    config: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, float]]:
    evaluation = config["evaluation"]
    metrics = {
        "oracle_mean_gain_advantage_over_fixed": float(
            oracle["mean_true_gain"] - fixed["mean_true_gain"]
        ),
        "primary_mean_gain_advantage_over_fixed": float(
            primary["mean_true_gain"] - fixed["mean_true_gain"]
        ),
        "primary_win_fraction_over_fixed": float(win_fraction),
        "primary_p95_rmse_ratio_to_fixed": float(
            primary["p95_rmse"] / fixed["p95_rmse"]
        ),
        "primary_worst_gain_advantage_over_fixed": float(
            primary["worst_true_gain"] - fixed["worst_true_gain"]
        ),
        "primary_mean_positive_proxy_excess": float(
            primary["mean_positive_proxy_excess"]
        ),
        "maximum_new_boundary_fraction": float(
            max(
                primary["maximum_new_boundary_fraction"],
                fixed["maximum_new_boundary_fraction"],
                oracle["maximum_new_boundary_fraction"],
            )
        ),
    }
    gates = {
        "oracle_has_material_routing_value": metrics[
            "oracle_mean_gain_advantage_over_fixed"
        ]
        >= float(evaluation["minimum_oracle_mean_gain_advantage_over_fixed_k1"]),
        "primary_improves_mean": metrics[
            "primary_mean_gain_advantage_over_fixed"
        ]
        >= float(evaluation["minimum_primary_mean_gain_advantage_over_fixed_k1"]),
        "primary_wins_sources": metrics["primary_win_fraction_over_fixed"]
        >= float(evaluation["minimum_primary_win_fraction_over_fixed_k1"]),
        "primary_improves_p95": metrics["primary_p95_rmse_ratio_to_fixed"]
        <= float(evaluation["maximum_primary_p95_rmse_ratio_to_fixed_k1"]),
        "primary_improves_worst": metrics[
            "primary_worst_gain_advantage_over_fixed"
        ]
        >= float(evaluation["minimum_primary_worst_gain_advantage_over_fixed_k1"]),
        "primary_proxy_excess_bounded": metrics[
            "primary_mean_positive_proxy_excess"
        ]
        <= float(evaluation["maximum_primary_mean_positive_proxy_excess"]),
        "boundary_safe": metrics["maximum_new_boundary_fraction"]
        <= float(evaluation["maximum_new_boundary_fraction"]),
    }
    return gates, metrics


def run_control(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    parent, parent_evidence = validate_contract(root, config)
    rows = load_manifest(root, parent)
    splits = split_rows(rows, parent)
    side = int(parent["data"]["decode_side"])
    operator = parent["operator"]
    trusted = candidate_bank(
        operator["trusted_support_bounds"],
        int(operator["trusted_candidate_count"]),
        int(operator["trusted_candidate_seed"]),
    )
    quantiles = parent["frozen_scorer"]["quantiles"]

    feature_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    fit_candidate_rewards: list[np.ndarray] = []
    for row in splits["scorer_fit"]:
        source, target = load_pair(root, row, side)
        outputs = apply_operator_bank(source, trusted)
        features, _ = scorer_features(source, outputs, quantiles)
        rewards, _ = _rewards(outputs, source, target)
        feature_rows.append(features)
        reward_rows.append(rewards)
        fit_candidate_rewards.append(rewards)
    train_x = np.concatenate(feature_rows, axis=0)
    train_y = np.concatenate(reward_rows, axis=0)
    scaler = StandardScaler().fit(train_x)
    scorer = Ridge(alpha=float(parent["frozen_scorer"]["ridge_alpha"])).fit(
        scaler.transform(train_x), train_y
    )
    fixed_index = int(np.argmax(np.mean(np.stack(fit_candidate_rewards), axis=0)))

    split_results: dict[str, Any] = {}
    selected_rows: dict[str, Any] = {}
    for split_name in ("validation", "hidden_test"):
        selections, controls, predicted_all, true_all = _evaluate_rows(
            root=root,
            rows=splits[split_name],
            side=side,
            bank=trusted,
            quantiles=quantiles,
            scaler=scaler,
            scorer=scorer,
            fixed_trusted_parameters=trusted[fixed_index],
        )
        selector_aggregate = {
            name: _aggregate(value) for name, value in selections.items()
        }
        control_aggregate = {
            name: _aggregate(value) for name, value in controls.items()
        }
        primary_rows = selections["fidelity_plus_gap"]
        fixed_rows = controls["fixed_trusted"]
        wins = sum(
            left["true_gain"] > right["true_gain"]
            for left, right in zip(primary_rows, fixed_rows, strict=True)
        )
        win_fraction = wins / len(primary_rows)
        split_results[split_name] = {
            "scorer_candidate_spearman": float(
                spearmanr(predicted_all, true_all).statistic
            ),
            "selectors": selector_aggregate,
            "controls": control_aggregate,
            "primary_win_fraction_over_fixed": float(win_fraction),
        }
        selected_rows[split_name] = {
            "selectors": selections,
            "controls": controls,
        }

    hidden = split_results["hidden_test"]
    gates, metrics = evaluate_gates(
        primary=hidden["selectors"]["fidelity_plus_gap"],
        fixed=hidden["controls"]["fixed_trusted"],
        oracle=hidden["controls"]["oracle"],
        win_fraction=float(hidden["primary_win_fraction_over_fixed"]),
        config=config,
    )
    oracle_gate = gates["oracle_has_material_routing_value"]
    automatic_pass = all(gates.values())
    if not oracle_gate:
        decision = "retain_fixed_k1_no_material_trusted_bank_oracle_gain"
    elif automatic_pass:
        decision = "retain_trusted_support_hard_selector_for_confirmation"
    else:
        decision = "retain_fixed_k1_scorer_does_not_earn_routing"

    report = {
        "schema": "neuro_film.u5_r2bz1_fivek_trusted_support_selector.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": _sha256(
            root / "configs/u5_r2bz1_fivek_trusted_support_selector_v1.json"
        ),
        "parent_evidence_sha256": config["parent"]["evidence_sha256"],
        "parent_stable_evidence_id": parent_evidence["formal_reports"][
            "stable_evidence_id"
        ],
        "trusted_bank_sha256": hashlib.sha256(trusted.tobytes()).hexdigest(),
        "scorer": {
            "training_example_count": int(train_x.shape[0]),
            "feature_count": int(train_x.shape[1]),
            "fit_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            scorer.predict(scaler.transform(train_x)) - train_y
                        )
                        ** 2
                    )
                )
            ),
            "coefficient_sha256": hashlib.sha256(
                np.asarray(scorer.coef_, dtype=np.float64).tobytes()
            ).hexdigest(),
        },
        "fixed_trusted_control": {
            "candidate_index": fixed_index,
            "parameters": trusted[fixed_index].tolist(),
        },
        "results": split_results,
        "selected_rows": selected_rows,
        "primary_metrics": metrics,
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": decision,
        "confirmation_opened": automatic_pass,
        "product_integration_opened": False,
        "film_or_stock_claim_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    return report


__all__ = [
    "FiveKTrustedSupportError",
    "evaluate_gates",
    "run_control",
    "validate_contract",
]
