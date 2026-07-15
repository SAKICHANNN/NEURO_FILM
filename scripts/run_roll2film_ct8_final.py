"""Run the frozen one-shot FilmSet final-628 recipe-bank evaluation."""

from __future__ import annotations

import argparse
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

from src.roll2film.ct5_data import CT5DataContract, load_ct5_working_image  # noqa: E402
from src.roll2film.ct5_fullres import (  # noqa: E402
    full_resolution_diagnostics,
    linear_to_u8,
    make_review_sheet,
)
from src.roll2film.ct8_final import (  # noqa: E402
    cluster_bootstrap_improvement,
    load_final_manifest,
    operator_from_frozen_bundle,
    select_review_cases,
    sha256_file,
    verify_payloads,
)


UPSTREAM_DEFAULTS = {
    "filmset_evidence_report": "outputs/roll2film/filmset_evidence/report.json",
    "pilot_report": "outputs/roll2film/ct5_v1/pilot_report.json",
    "pilot_decision": "configs/roll2film_ct5_pilot_decision.json",
    "confirmatory_report": "outputs/roll2film/ct5_v1/confirmatory_report.json",
    "confirmatory_decision": "configs/roll2film_ct5_confirmatory_decision.json",
    "fullres_report": "outputs/roll2film/ct5_v1/fullres/report.json",
    "fullres_decision": "configs/roll2film_ct5_fullres_decision.json",
    "blueneg_confirmatory_decision": "configs/roll2film_blueneg_confirmatory_decision.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "configs" / "roll2film_ct8_final_policy.json"
    )
    parser.add_argument(
        "--ct5-config", type=Path, default=ROOT / "configs" / "roll2film_ct5_baselines.json"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "filmset_evidence" / "final_628_lockbox.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "roll2film" / "ct8_final_628"
    )
    for name, relative in UPSTREAM_DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=ROOT / relative)
    return parser.parse_args()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _verify_upstream(args: argparse.Namespace, policy: dict[str, Any]) -> None:
    contract = policy["source_contract"]
    for name in UPSTREAM_DEFAULTS:
        path = getattr(args, name)
        expected = contract[f"{name}_sha256"]
        observed = sha256_file(path)
        if observed.lower() != expected.lower():
            raise ValueError(f"upstream hash mismatch for {name}: {observed}")


def _save_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(linear_to_u8(values), "RGB").save(path, "PNG", optimize=True)


def _read_progress(path: Path, policy_sha: str, software: str) -> dict[tuple[str, str], dict]:
    completed: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if row["policy_sha256"] != policy_sha or row["software_commit"] != software:
            raise ValueError("progress belongs to another frozen policy or software commit")
        key = (str(row["domain"]), str(row["content_id"]))
        if key in completed:
            raise ValueError(f"duplicate progress row at line {line_number}: {key}")
        completed[key] = row
    return completed


def _append_progress(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def _render_review_cases(
    *,
    output_dir: Path,
    domain: str,
    selection: dict[str, Any],
    by_content: dict[str, dict[str, dict[str, Any]]],
    contract: CT5DataContract,
    primary: Any,
    basic: Any,
    primary_name: str,
    basic_name: str,
) -> dict[str, Any]:
    review_rows: list[dict[str, Path | str]] = []
    for content_id in selection["union"]:
        rows = by_content[content_id]
        source = load_ct5_working_image(contract, rows["input"]).pixels
        target = load_ct5_working_image(contract, rows[domain]).pixels
        primary_values = primary.apply(source.reshape(-1, 3)).reshape(source.shape)
        basic_values = basic.apply(source.reshape(-1, 3)).reshape(source.shape)
        case_dir = output_dir / "cases" / domain / content_id.replace(":", "_")
        paths = {
            "input": case_dir / "input.png",
            "target": case_dir / "target.png",
            "best_basic": case_dir / f"best_basic_{basic_name}.png",
            "candidate": case_dir / f"primary_{primary_name}.png",
        }
        _save_png(paths["input"], source)
        _save_png(paths["target"], target)
        _save_png(paths["best_basic"], basic_values)
        _save_png(paths["candidate"], primary_values)
        review_rows.append({"content_id": content_id, **paths})
    sheet = output_dir / "sheets" / f"{domain}__primary.jpg"
    make_review_sheet(review_rows, sheet)
    return {"selection": selection, "review_sheet": str(sheet), "saved_cases": len(review_rows)}


def main() -> int:
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    policy_sha = sha256_file(args.policy)
    software = _commit()
    _verify_upstream(args, policy)
    ct5_contract = CT5DataContract.from_config(args.ct5_config, ROOT)
    expected = policy["expected"]
    by_content = load_final_manifest(
        args.manifest,
        expected_sha256=policy["source_contract"]["final_628_manifest_sha256"],
        expected_identities=int(expected["identities"]),
        expected_rows=int(expected["manifest_rows"]),
        domains=expected["domains"],
    )
    print("CT8 manifest contract passed; verifying every payload before decode", flush=True)
    verify_payloads(by_content, ct5_contract.dataset_root)
    print(f"CT8 payload preflight passed: {expected['manifest_rows']} files", flush=True)

    pilot = json.loads(args.pilot_report.read_text(encoding="utf-8"))
    bundles = pilot["operator_bundles"]
    output_dir = args.output_dir.resolve()
    progress_path = output_dir / "progress.jsonl"
    completed = _read_progress(progress_path, policy_sha, software)
    domains = tuple(expected["domains"])
    identities = tuple(sorted(by_content))
    total = len(domains) * len(identities)
    done = len(completed)
    for domain in domains:
        names = policy["frozen_recipe_bank"][domain]
        primary = operator_from_frozen_bundle(bundles[domain][names["primary"]])
        basic = operator_from_frozen_bundle(bundles[domain][names["best_basic"]])
        for content_id in identities:
            key = (domain, content_id)
            if key in completed:
                continue
            rows = by_content[content_id]
            source = load_ct5_working_image(ct5_contract, rows["input"]).pixels
            target = load_ct5_working_image(ct5_contract, rows[domain]).pixels
            if source.shape != target.shape:
                raise ValueError(f"final pair shape mismatch: {content_id}/{domain}")
            primary_values = primary.apply(source.reshape(-1, 3)).reshape(source.shape)
            basic_values = basic.apply(source.reshape(-1, 3)).reshape(source.shape)
            row = {
                "policy_sha256": policy_sha,
                "software_commit": software,
                "content_id": content_id,
                "cluster_id": str(rows["input"]["duplicate_cluster_id"]),
                "domain": domain,
                "primary": names["primary"],
                "best_basic": names["best_basic"],
                "primary_metrics": full_resolution_diagnostics(source, target, primary_values),
                "best_basic_metrics": full_resolution_diagnostics(source, target, basic_values),
            }
            _append_progress(progress_path, row)
            completed[key] = row
            done += 1
            if done % 10 == 0 or done == total:
                print(f"CT8 fullres {done}/{total}", flush=True)

    statistics = policy["statistics"]
    severe = policy["severe_review"]
    report_domains: dict[str, Any] = {}
    for domain_index, domain in enumerate(domains):
        rows = [completed[(domain, content_id)] for content_id in identities]
        primary_rows = [
            {"content_id": row["content_id"], "metrics": row["primary_metrics"]} for row in rows
        ]
        comparison = cluster_bootstrap_improvement(
            [row["best_basic_metrics"]["mean_delta_e00_to_target"] for row in rows],
            [row["primary_metrics"]["mean_delta_e00_to_target"] for row in rows],
            [row["cluster_id"] for row in rows],
            seed=int(statistics["bootstrap_seed"]) + domain_index,
            resamples=int(statistics["bootstrap_resamples"]),
        )
        primary_fidelity = float(
            np.mean([row["primary_metrics"]["mean_delta_e00_to_target"] for row in rows])
        )
        basic_fidelity = float(
            np.mean([row["best_basic_metrics"]["mean_delta_e00_to_target"] for row in rows])
        )
        primary_style = float(
            np.mean([row["primary_metrics"]["median_delta_e00_from_input"] for row in rows])
        )
        basic_style = float(
            np.mean([row["best_basic_metrics"]["median_delta_e00_from_input"] for row in rows])
        )
        style_ratio = primary_style / max(basic_style, 1e-12)
        selection = select_review_cases(
            primary_rows,
            axes=severe["axes"],
            composite_count=int(severe["save_composite_worst_per_domain"]),
            per_axis_count=int(severe["save_axis_extremes_per_domain"]),
        )
        names = policy["frozen_recipe_bank"][domain]
        review = _render_review_cases(
            output_dir=output_dir,
            domain=domain,
            selection=selection,
            by_content=by_content,
            contract=ct5_contract,
            primary=operator_from_frozen_bundle(bundles[domain][names["primary"]]),
            basic=operator_from_frozen_bundle(bundles[domain][names["best_basic"]]),
            primary_name=names["primary"],
            basic_name=names["best_basic"],
        )
        triggers = [
            row["content_id"]
            for row in rows
            if row["primary_metrics"]["raw_excursion_max"]
            >= severe["automatic_review_triggers"]["raw_excursion_max_at_least"]
            or row["primary_metrics"]["new_display_clip_pixel_fraction_vs_target"]
            >= severe["automatic_review_triggers"]
            ["new_display_clip_pixel_fraction_vs_target_at_least"]
        ]
        report_domains[domain] = {
            "primary": names["primary"],
            "best_basic": names["best_basic"],
            "identities": len(rows),
            "primary_mean_delta_e00_to_target": primary_fidelity,
            "best_basic_mean_delta_e00_to_target": basic_fidelity,
            "cluster_bootstrap_improvement": comparison,
            "fidelity_ci_pass": comparison["ci95_low"]
            > statistics["primary_must_beat_best_basic_ci95_low"],
            "primary_mean_per_image_median_style": primary_style,
            "best_basic_mean_per_image_median_style": basic_style,
            "style_ratio": style_ratio,
            "style_ratio_pass": style_ratio >= statistics["primary_style_ratio_to_best_basic_minimum"],
            "automatic_trigger_content_ids": triggers,
            "review": review,
            "per_image": rows,
            "visual_severe_veto": "pending_human_or_codex_original_resolution_adjudication",
        }
    report = {
        "schema_version": 1,
        "experiment_id": policy["experiment_id"],
        "policy_sha256": policy_sha,
        "software_commit": software,
        "final_manifest_sha256": sha256_file(args.manifest),
        "payload_preflight_before_decode": True,
        "final_628_parsed_or_decoded": True,
        "operators_refit_on_final": False,
        "post_hoc_changes": False,
        "domains": report_domains,
        "decision_state": "awaiting_original_resolution_visual_severe_adjudication",
        "claim_boundary": policy["claim_boundary"],
    }
    report_path = output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"report": str(report_path), "domains": {
        domain: {
            "fidelity_ci_pass": value["fidelity_ci_pass"],
            "style_ratio_pass": value["style_ratio_pass"],
            "automatic_triggers": len(value["automatic_trigger_content_ids"]),
        } for domain, value in report_domains.items()
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
