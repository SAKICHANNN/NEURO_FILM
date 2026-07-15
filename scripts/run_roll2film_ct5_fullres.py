"""Render frozen CT5 finalists at full resolution and build worst-case review sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.baselines import (  # noqa: E402
    LAB_STATS_SCHEMA,
    SLICED_SCHEMA,
    LabMeanStdOperator,
    SlicedTransportOperator,
)
from src.roll2film.ct5_data import (  # noqa: E402
    CT5DataContract,
    load_ct5_internal_dev_rows,
    load_ct5_working_image,
)
from src.roll2film.ct5_fullres import (  # noqa: E402
    full_resolution_diagnostics,
    linear_to_u8,
    make_review_sheet,
)
from src.roll2film.operators import OPERATOR_SCHEMA, AffineColorOperator  # noqa: E402
from src.roll2film.splines import (  # noqa: E402
    L2_OPERATOR_SCHEMA,
    AffineMonotoneSplineOperator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=ROOT / "configs" / "roll2film_ct5_baselines.json",
    )
    parser.add_argument(
        "--pilot-decision",
        type=Path,
        default=ROOT / "configs" / "roll2film_ct5_pilot_decision.json",
    )
    parser.add_argument(
        "--confirmatory-decision",
        type=Path,
        default=ROOT / "configs" / "roll2film_ct5_confirmatory_decision.json",
    )
    parser.add_argument(
        "--pilot-report",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "pilot_report.json",
    )
    parser.add_argument(
        "--confirmatory-report",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "confirmatory_report.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "data_cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "fullres",
    )
    parser.add_argument(
        "--adjudication-domain",
        choices=("cinema", "classneg", "velvia"),
        help="Render only named confirmatory cases for targeted visual adjudication.",
    )
    parser.add_argument(
        "--adjudication-content-id",
        action="append",
        default=[],
        help="Confirmatory content ID to render; may be supplied more than once.",
    )
    return parser.parse_args()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _operator(payload: dict[str, Any]) -> Any:
    schema = payload.get("schema")
    if schema == OPERATOR_SCHEMA:
        return AffineColorOperator.from_dict(payload)
    if schema == LAB_STATS_SCHEMA:
        return LabMeanStdOperator.from_dict(payload)
    if schema == SLICED_SCHEMA:
        return SlicedTransportOperator.from_dict(payload)
    if schema == L2_OPERATOR_SCHEMA:
        return AffineMonotoneSplineOperator.from_dict(payload)
    raise ValueError(f"unsupported frozen CT5 operator schema: {schema!r}")


def _worst_indices(rows: list[dict[str, Any]], count: int) -> list[int]:
    axes = (
        "raw_out_of_range_fraction",
        "mean_delta_e00_to_target",
        "red_cyan_boundary_occupancy",
        "speckle_candidate_percent",
    )
    score = np.zeros(len(rows), dtype=np.float64)
    for axis in axes:
        values = np.asarray(
            [
                row["metrics"]["chroma_speckle"][axis]
                if axis == "speckle_candidate_percent"
                else row["metrics"][axis]
                for row in rows
            ],
            dtype=np.float64,
        )
        order = np.argsort(np.argsort(values))
        score += order / max(len(rows) - 1, 1)
    return list(np.argsort(score)[::-1][:count])


def _save_png(path: Path, linear: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(linear_to_u8(linear), mode="RGB").save(path, "PNG")


def _candidate_names(
    final_decision: dict[str, Any], domain: str
) -> tuple[dict[str, Any], list[str]]:
    recipe = final_decision["recipe_finalists"][domain]
    names = list(dict.fromkeys([recipe["primary"], *recipe["comparators"]]))
    return recipe, names


def _render_targeted_adjudication(
    *,
    args: argparse.Namespace,
    policy: dict[str, Any],
    pilot_decision: dict[str, Any],
    final_decision: dict[str, Any],
    pilot: dict[str, Any],
    contract: CT5DataContract,
    by_content: dict[str, dict[str, dict[str, Any]]],
    confirm_ids: set[str],
) -> int:
    if not args.adjudication_domain:
        raise ValueError("--adjudication-domain is required with targeted content IDs")
    requested = list(dict.fromkeys(args.adjudication_content_id))
    forbidden = sorted(set(requested) - confirm_ids)
    if forbidden:
        raise ValueError(
            "targeted adjudication is restricted to frozen confirmatory IDs: "
            + ", ".join(forbidden)
        )
    domain = args.adjudication_domain
    recipe, candidate_names = _candidate_names(final_decision, domain)
    operators = {
        name: _operator(pilot["operator_bundles"][domain][name]) for name in candidate_names
    }
    primary_stratum = pilot_decision["candidate_stratum_by_domain"][domain].get(
        recipe["primary"], "strong"
    )
    best_basic_name = pilot_decision["best_basic_by_domain_and_stratum"][domain][
        primary_stratum
    ]
    if best_basic_name not in operators:
        operators[best_basic_name] = _operator(
            pilot["operator_bundles"][domain][best_basic_name]
        )
    output_dir = args.output_dir.resolve() / "targeted_adjudication" / domain
    report_rows: list[dict[str, Any]] = []
    for content_id in requested:
        domain_rows = by_content[content_id]
        source = load_ct5_working_image(contract, domain_rows["input"]).pixels
        target = load_ct5_working_image(contract, domain_rows[domain]).pixels
        case_dir = output_dir / content_id.replace(":", "_")
        _save_png(case_dir / "input.png", source)
        _save_png(case_dir / "target.png", target)
        candidate_metrics: dict[str, Any] = {}
        for name, operator in operators.items():
            rendered = operator.apply(source.reshape(-1, 3)).reshape(source.shape)
            _save_png(case_dir / f"candidate_{name}.png", rendered)
            candidate_metrics[name] = full_resolution_diagnostics(source, target, rendered)
        report_rows.append(
            {
                "content_id": content_id,
                "case_dir": str(case_dir),
                "candidates": candidate_metrics,
            }
        )
    report = {
        "schema_version": 1,
        "mode": "targeted_confirmatory_visual_adjudication",
        "experiment_id": policy["experiment_id"],
        "domain": domain,
        "primary": recipe["primary"],
        "best_basic": best_basic_name,
        "confirmatory_decision_sha256": _sha(args.confirmatory_decision),
        "confirmatory_report_sha256": _sha(args.confirmatory_report),
        "software_commit": _commit(),
        "rows": report_rows,
        "final_628_parsed_or_decoded": False,
    }
    report_path = output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"report": str(report_path), "cases": len(report_rows)}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    policy = json.loads(args.policy_config.read_text(encoding="utf-8"))
    pilot_decision = json.loads(args.pilot_decision.read_text(encoding="utf-8"))
    final_decision = json.loads(args.confirmatory_decision.read_text(encoding="utf-8"))
    if _sha(args.confirmatory_report) != final_decision["confirmatory_report_sha256"]:
        raise ValueError("frozen confirmatory report hash mismatch")
    if _sha(args.pilot_report) != pilot_decision["pilot_report_sha256"]:
        raise ValueError("frozen pilot report hash mismatch")
    confirmatory = json.loads(args.confirmatory_report.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_report.read_text(encoding="utf-8"))
    if confirmatory["final_628_parsed_or_decoded"] is not False:
        raise ValueError("confirmatory run did not preserve final-628 seal")
    contract = CT5DataContract.from_config(args.policy_config, ROOT)
    rows = load_ct5_internal_dev_rows(contract)
    membership_path = args.cache_dir / "membership.json"
    cache_report_path = args.cache_dir / "report.json"
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    if _sha(membership_path) != cache_report["arrays_sha256"]["membership.json"]:
        raise ValueError("CT5 membership hash mismatch")
    membership = json.loads(membership_path.read_text(encoding="utf-8"))["internal_dev"]
    confirm_ids = {
        str(row["content_id"]) for row in membership if row["fold"] == "confirmatory"
    }
    by_content: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        content_id = str(row["content_id"])
        if content_id in confirm_ids:
            by_content.setdefault(content_id, {})[str(row["domain"])] = row
    if set(by_content) != confirm_ids:
        raise ValueError("full-resolution confirmatory membership mismatch")

    if args.adjudication_content_id:
        return _render_targeted_adjudication(
            args=args,
            policy=policy,
            pilot_decision=pilot_decision,
            final_decision=final_decision,
            pilot=pilot,
            contract=contract,
            by_content=by_content,
            confirm_ids=confirm_ids,
        )
    if args.adjudication_domain:
        raise ValueError(
            "--adjudication-domain requires at least one --adjudication-content-id"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_domains: dict[str, Any] = {}
    save_count = int(
        final_decision["full_resolution_adjudication"]
        ["save_worst_cases_per_recipe_candidate"]
    )
    for domain in policy["dataset"]["domains"]:
        print(f"CT5 fullres {domain}", flush=True)
        recipe, candidate_names = _candidate_names(final_decision, domain)
        operators = {
            name: _operator(pilot["operator_bundles"][domain][name]) for name in candidate_names
        }
        primary_stratum = pilot_decision["candidate_stratum_by_domain"][domain].get(
            recipe["primary"], "strong"
        )
        best_basic_name = pilot_decision["best_basic_by_domain_and_stratum"][domain][
            primary_stratum
        ]
        if best_basic_name not in operators:
            operators[best_basic_name] = _operator(
                pilot["operator_bundles"][domain][best_basic_name]
            )
        metrics_by_candidate = {name: [] for name in candidate_names}
        for index, content_id in enumerate(sorted(confirm_ids), start=1):
            domain_rows = by_content[content_id]
            source = load_ct5_working_image(contract, domain_rows["input"]).pixels
            target = load_ct5_working_image(contract, domain_rows[domain]).pixels
            if source.shape != target.shape:
                raise ValueError(f"full-resolution pair shape mismatch: {content_id}/{domain}")
            for name in candidate_names:
                rendered = operators[name].apply(source.reshape(-1, 3)).reshape(source.shape)
                metrics_by_candidate[name].append(
                    {
                        "content_id": content_id,
                        "metrics": full_resolution_diagnostics(source, target, rendered),
                    }
                )
            if index % 25 == 0 or index == len(confirm_ids):
                print(f"{domain} {index}/{len(confirm_ids)}", flush=True)

        domain_report: dict[str, Any] = {}
        for name, metric_rows in metrics_by_candidate.items():
            selected = _worst_indices(metric_rows, save_count)
            review_rows: list[dict[str, Path | str]] = []
            for selected_index in selected:
                content_id = metric_rows[selected_index]["content_id"]
                domain_rows = by_content[content_id]
                source = load_ct5_working_image(contract, domain_rows["input"]).pixels
                target = load_ct5_working_image(contract, domain_rows[domain]).pixels
                candidate = operators[name].apply(source.reshape(-1, 3)).reshape(source.shape)
                basic = operators[best_basic_name].apply(source.reshape(-1, 3)).reshape(source.shape)
                case_dir = output_dir / "cases" / domain / name / content_id.replace(":", "_")
                paths = {
                    "input": case_dir / "input.png",
                    "target": case_dir / "target.png",
                    "best_basic": case_dir / f"best_basic_{best_basic_name}.png",
                    "candidate": case_dir / f"candidate_{name}.png",
                }
                _save_png(paths["input"], source)
                _save_png(paths["target"], target)
                _save_png(paths["best_basic"], basic)
                _save_png(paths["candidate"], candidate)
                review_rows.append({"content_id": content_id, **paths})
            sheet_path = output_dir / "sheets" / f"{domain}__{name}.jpg"
            make_review_sheet(review_rows, sheet_path)
            domain_report[name] = {
                "images": len(metric_rows),
                "per_image": metric_rows,
                "worst_case_content_ids": [metric_rows[value]["content_id"] for value in selected],
                "review_sheet": str(sheet_path),
                "display_output_transform": "explicit hard clip to sRGB8 for review PNG only",
                "automatic_metrics_do_not_clear_severe": True,
            }
        report_domains[domain] = {
            "primary": recipe["primary"],
            "best_basic": best_basic_name,
            "candidates": domain_report,
        }
    report = {
        "schema_version": 1,
        "experiment_id": policy["experiment_id"],
        "confirmatory_decision_sha256": _sha(args.confirmatory_decision),
        "confirmatory_report_sha256": _sha(args.confirmatory_report),
        "pilot_report_sha256": _sha(args.pilot_report),
        "cache_report_sha256": _sha(cache_report_path),
        "software_commit": _commit(),
        "domains": report_domains,
        "final_628_parsed_or_decoded": False,
        "decision_state": "awaiting_visual_severe_adjudication",
    }
    report_path = output_dir / "report.json"
    report_path.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"report": str(report_path), "final_628_parsed_or_decoded": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
