"""Integrity and presentation tooling for the frozen U5.R2H2 visual audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.eval.global_frontier import sha256_file


class HardRoutingVisualAuditError(ValueError):
    """Raised when the frozen H2 evidence or policy does not replay exactly."""


def blind_assignment(seed: int, round_id: str, sample_id: str) -> bool:
    """Return True when candidate A is routed, using a stable hash bit."""

    digest = hashlib.sha256(
        f"{seed}|{round_id}|{sample_id}".encode("utf-8")
    ).digest()
    return bool(digest[0] & 1)


def _verify(root: Path, config: Mapping[str, Any]) -> None:
    pairs = (
        ("input_manifest", "input_manifest_sha256"),
        ("routing_report", "routing_report_sha256"),
        ("routing_config", "routing_config_sha256"),
        ("anchor_manifest", "anchor_manifest_sha256"),
        ("anchor_report", "anchor_report_sha256"),
        ("density_manifest", "density_manifest_sha256"),
        ("density_report", "density_report_sha256"),
    )
    for path_key, hash_key in pairs:
        actual = sha256_file(root / str(config[path_key]))
        if actual != str(config[hash_key]):
            raise HardRoutingVisualAuditError(f"{path_key} hash mismatch")


def _resolve_record_path(
    root: Path, manifest_path: Path, prefix: str, record: Mapping[str, Any]
) -> Path:
    path = Path(str(record["output"]))
    if path.is_absolute():
        return path
    if prefix == "density":
        return manifest_path.parent / path
    return root / path


def build_replay_manifest(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify all frozen inputs and return the exact 41-row routed policy."""

    _verify(root, config)
    source_payload = json.loads(
        (root / str(config["input_manifest"])).read_text(encoding="utf-8")
    )
    samples = source_payload.get("frozen_set", source_payload)["samples"]
    if len(samples) != 41:
        raise HardRoutingVisualAuditError("H2 requires exactly 41 source rows")
    by_id = {str(row["id"]): dict(row) for row in samples}
    if len(by_id) != 41:
        raise HardRoutingVisualAuditError("duplicate source ID")
    for sample_id, row in by_id.items():
        if sha256_file(root / row["source_path"]) != row["source_sha256"]:
            raise HardRoutingVisualAuditError(
                f"source hash mismatch: {sample_id}"
            )

    class_map = {int(k): str(v) for k, v in config["class_map"].items()}
    banks: dict[int, dict[str, dict[str, Any]]] = {}
    metrics: dict[int, dict[str, dict[str, Any]]] = {}
    for class_id, prefix in ((0, "anchor"), (1, "density")):
        candidate_id = class_map[class_id]
        manifest_path = root / str(config[f"{prefix}_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = {
            str(row["sample_id"]): dict(row)
            for row in manifest["records"]
            if row["candidate_id"] == candidate_id
        }
        report = json.loads(
            (root / str(config[f"{prefix}_report"])).read_text(encoding="utf-8")
        )
        report_rows = {
            str(row["sample_id"]): dict(row)
            for row in report["candidates"][candidate_id]["per_image"]
        }
        if records.keys() != by_id.keys() or report_rows.keys() != by_id.keys():
            raise HardRoutingVisualAuditError(
                f"{prefix} bank does not cover the frozen 41 rows"
            )
        for sample_id, record in records.items():
            output_path = _resolve_record_path(
                root, manifest_path, prefix, record
            )
            if sha256_file(output_path) != record["output_sha256"]:
                raise HardRoutingVisualAuditError(
                    f"{prefix} output hash mismatch: {sample_id}"
                )
            if record["source_sha256"] != by_id[sample_id]["source_sha256"]:
                raise HardRoutingVisualAuditError(
                    f"{prefix} source lineage mismatch: {sample_id}"
                )
            record["resolved_output"] = str(output_path.resolve())
        banks[class_id] = records
        metrics[class_id] = report_rows

    routing = json.loads(
        (root / str(config["routing_report"])).read_text(encoding="utf-8")
    )
    router = routing["views"][str(config["routing_view"])]["routers"][
        str(config["router_id"])
    ]
    test_ids = [str(value) for value in router["test_ids"]]
    predictions = [int(value) for value in router["predictions"]]
    traces = {
        str(row["test_id"]): [str(v) for v in row["selected_training_ids"]]
        for row in router["traces"]
    }
    if not (len(test_ids) == len(predictions) == len(traces) == 37):
        raise HardRoutingVisualAuditError("H1 assigned replay must have 37 rows")
    if set(test_ids) - by_id.keys():
        raise HardRoutingVisualAuditError("H1 contains an unknown source ID")
    if sum(value == 0 for value in predictions) != 10:
        raise HardRoutingVisualAuditError(
            "frozen H1 policy must contain exactly ten anchor interventions"
        )
    predicted = dict(zip(test_ids, predictions))
    fallback = int(config["unassigned_fallback_class"])
    margin = float(
        json.loads(
            (root / str(config["routing_config"])).read_text(encoding="utf-8")
        )["minimum_assignment_margin"]
    )

    records_out = []
    for sample_id in sorted(by_id):
        anchor_metric = metrics[0][sample_id]
        density_metric = metrics[1][sample_id]
        anchor_utility = float(
            anchor_metric["median_style_delta_e76"]
        ) + float(anchor_metric["median_non_basic_residual_delta_e76"])
        density_utility = float(
            density_metric["median_style_delta_e76"]
        ) + float(density_metric["median_non_basic_residual_delta_e76"])
        oracle_assigned = abs(density_utility - anchor_utility) >= margin
        oracle_class = int(density_utility > anchor_utility)
        if oracle_assigned != (sample_id in predicted):
            raise HardRoutingVisualAuditError(
                f"H1 assignment mismatch: {sample_id}"
            )
        route_class = predicted.get(sample_id, fallback)
        source_path = (root / by_id[sample_id]["source_path"]).resolve()
        routed = banks[route_class][sample_id]
        global_row = banks[1][sample_id]
        with Image.open(source_path) as source_image:
            source_size = list(ImageOps.exif_transpose(source_image).size)
        with Image.open(routed["resolved_output"]) as routed_image:
            routed_size = list(routed_image.size)
        records_out.append(
            {
                "sample_id": sample_id,
                "source_path": str(source_path),
                "source_sha256": by_id[sample_id]["source_sha256"],
                "source_size": source_size,
                "assigned": sample_id in predicted,
                "prediction_class": (
                    predicted[sample_id] if sample_id in predicted else None
                ),
                "fallback_applied": sample_id not in predicted,
                "route_class": route_class,
                "route_candidate_id": class_map[route_class],
                "selected_training_ids": traces.get(sample_id, []),
                "routed_path": routed["resolved_output"],
                "routed_sha256": routed["output_sha256"],
                "routed_size": routed_size,
                "global_path": global_row["resolved_output"],
                "global_sha256": global_row["output_sha256"],
                "oracle_assigned": oracle_assigned,
                "oracle_class": oracle_class if oracle_assigned else None,
                "oracle_agreement": (
                    route_class == oracle_class if oracle_assigned else None
                ),
                "anchor_utility": anchor_utility,
                "density_utility": density_utility,
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "sample_count": 41,
        "assigned_count": 37,
        "fallback_count": 4,
        "intervention_count": 10,
        "global_class": 1,
        "records": records_out,
    }


def _fit_image(path: Path, size: tuple[int, int], crop: bool) -> Image.Image:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        if crop:
            width, height = rgb.size
            left = int(round(width * 0.2))
            top = int(round(height * 0.2))
            right = int(round(width * 0.8))
            bottom = int(round(height * 0.8))
            rgb = rgb.crop((left, top, right, bottom))
        contained = ImageOps.contain(rgb, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        contained,
        ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
    )
    return canvas


def write_presentations(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write blinded intervention sheets and complete severe-review sheets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    interventions = [
        row for row in manifest["records"] if int(row["route_class"]) == 0
    ]
    key_rows = []
    panel_rows = []
    for round_spec in config["rounds"]:
        round_id = str(round_spec["round_id"])
        crop = str(round_spec["presentation"]) == "central_60_percent"
        ordering = sorted(
            interventions,
            key=lambda row: hashlib.sha256(
                f"{config['blind_seed']}|order|{round_id}|{row['sample_id']}".encode()
            ).hexdigest(),
        )
        for sheet_index in range(0, len(ordering), 5):
            subset = ordering[sheet_index : sheet_index + 5]
            canvas = Image.new("RGB", (984, 5 * 226 + 38), "#e8e8e8")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (8, 8),
                f"{round_id} / sheet {sheet_index // 5 + 1} / blind A-B",
                fill="black",
                font=font,
            )
            for row_index, row in enumerate(subset):
                sample_id = str(row["sample_id"])
                a_is_routed = blind_assignment(
                    int(config["blind_seed"]), round_id, sample_id
                )
                routed_path = Path(str(row["routed_path"]))
                global_path = Path(str(row["global_path"]))
                a_path = routed_path if a_is_routed else global_path
                b_path = global_path if a_is_routed else routed_path
                y = 38 + row_index * 226
                for column, (label, path) in enumerate(
                    (
                        ("Source", Path(str(row["source_path"]))),
                        ("A", a_path),
                        ("B", b_path),
                    )
                ):
                    x = column * 328
                    tile = _fit_image(path, (320, 196), crop)
                    canvas.paste(tile, (x + 4, y + 22))
                    draw.text(
                        (x + 8, y + 4),
                        f"{sample_id}  {label}",
                        fill="black",
                        font=font,
                    )
                key_rows.append(
                    {
                        "round_id": round_id,
                        "sample_id": sample_id,
                        "a_is_routed": a_is_routed,
                    }
                )
            panel_path = (
                output_dir
                / f"blind_{round_id}_{sheet_index // 5 + 1:02d}.png"
            )
            canvas.save(panel_path)
            panel_rows.append(
                {
                    "round_id": round_id,
                    "path": str(panel_path.resolve()),
                    "sha256": sha256_file(panel_path),
                    "sample_ids": [str(row["sample_id"]) for row in subset],
                }
            )

    severe_rows = []
    all_records = list(manifest["records"])
    for sheet_index in range(0, len(all_records), 7):
        subset = all_records[sheet_index : sheet_index + 7]
        canvas = Image.new("RGB", (984, 7 * 226 + 38), "#e8e8e8")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 8),
            f"all-routed severe review / sheet {sheet_index // 7 + 1}",
            fill="black",
            font=font,
        )
        for row_index, row in enumerate(subset):
            y = 38 + row_index * 226
            paths = (
                ("Source", Path(str(row["source_path"])), False),
                ("Routed full", Path(str(row["routed_path"])), False),
                ("Routed crop", Path(str(row["routed_path"])), True),
            )
            for column, (label, path, crop) in enumerate(paths):
                x = column * 328
                tile = _fit_image(path, (320, 196), crop)
                canvas.paste(tile, (x + 4, y + 22))
                draw.text(
                    (x + 8, y + 4),
                    f"{row['sample_id']}  {label}",
                    fill="black",
                    font=font,
                )
        panel_path = output_dir / f"severe_{sheet_index // 7 + 1:02d}.png"
        canvas.save(panel_path)
        severe_rows.append(
            {
                "path": str(panel_path.resolve()),
                "sha256": sha256_file(panel_path),
                "sample_ids": [str(row["sample_id"]) for row in subset],
            }
        )

    scoring = {
        "schema_version": 1,
        "instructions": "Score A/B blind before opening blind_key.json.",
        "rounds": [
            {
                "round_id": str(spec["round_id"]),
                "rows": [
                    {
                        "sample_id": str(row["sample_id"]),
                        "a_severe": None,
                        "b_severe": None,
                        "style_salience": None,
                        "colour_content_coherence": None,
                        "overall": None,
                        "reason": None,
                    }
                    for row in interventions
                ],
            }
            for spec in config["rounds"]
        ],
    }
    return {
        "blind_panels": panel_rows,
        "severe_panels": severe_rows,
        "blind_key": key_rows,
        "scoring_template": scoring,
    }


def adjudicate_scores(
    scoring: Mapping[str, Any],
    blind_key: list[Mapping[str, Any]],
    severe_review: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Decode already-frozen A/B scores and apply the preregistered gates."""

    if scoring.get("status") != "frozen_before_blind_key_open":
        raise HardRoutingVisualAuditError("blind scores were not frozen")
    key = {
        (str(row["round_id"]), str(row["sample_id"])): bool(
            row["a_is_routed"]
        )
        for row in blind_key
    }
    if len(key) != len(blind_key):
        raise HardRoutingVisualAuditError("duplicate blind key row")
    round_results = []
    intervention_ids: set[str] | None = None
    blind_severe = 0
    for round_row in scoring["rounds"]:
        round_id = str(round_row["round_id"])
        rows = list(round_row["rows"])
        ids = {str(row["sample_id"]) for row in rows}
        if len(rows) != 10 or len(ids) != 10:
            raise HardRoutingVisualAuditError(
                f"{round_id} must score all ten interventions"
            )
        if intervention_ids is None:
            intervention_ids = ids
        elif ids != intervention_ids:
            raise HardRoutingVisualAuditError(
                "blind rounds do not cover identical interventions"
            )
        routed_wins = global_wins = ties = 0
        decoded_rows = []
        for row in rows:
            sample_id = str(row["sample_id"])
            lookup = (round_id, sample_id)
            if lookup not in key:
                raise HardRoutingVisualAuditError(f"missing blind key: {lookup}")
            if bool(row["a_severe"]) or bool(row["b_severe"]):
                blind_severe += 1
            choice = str(row["overall"])
            if choice not in {"A", "B", "tie"}:
                raise HardRoutingVisualAuditError(
                    f"invalid overall score: {choice}"
                )
            if choice == "tie":
                decoded = "tie"
                ties += 1
            else:
                chose_routed = (choice == "A") == key[lookup]
                decoded = "routed" if chose_routed else "global"
                if chose_routed:
                    routed_wins += 1
                else:
                    global_wins += 1
            decoded_rows.append(
                {
                    "sample_id": sample_id,
                    "blind_choice": choice,
                    "decoded_choice": decoded,
                }
            )
        score = routed_wins - global_wins
        round_results.append(
            {
                "round_id": round_id,
                "routed_wins": routed_wins,
                "global_wins": global_wins,
                "ties": ties,
                "routed_win_plus_tie": routed_wins + ties,
                "score": score,
                "win_plus_tie_gate": routed_wins + ties
                >= int(gates["minimum_routed_win_plus_tie_per_round"]),
                "rows": decoded_rows,
            }
        )
    expected_ids = {str(row["sample_id"]) for row in severe_review["records"]}
    if len(expected_ids) != 41 or len(severe_review["records"]) != 41:
        raise HardRoutingVisualAuditError(
            "severe review must contain 41 unique rows"
        )
    routed_severe = sum(
        bool(row["routed_severe"]) for row in severe_review["records"]
    )
    new_severe = sum(
        bool(row["routed_severe"]) and not bool(row["global_severe"])
        for row in severe_review["records"]
    )
    positive_rounds = sum(row["score"] > 0 for row in round_results)
    gate_results = {
        "routed_severe": routed_severe
        <= int(gates["maximum_routed_severe_failures"]),
        "new_intervention_severe": new_severe
        <= int(gates["maximum_new_intervention_severe_failures"]),
        "each_round_win_plus_tie": all(
            row["win_plus_tie_gate"] for row in round_results
        ),
        "positive_rounds": positive_rounds
        >= int(gates["minimum_positive_score_rounds"]),
    }
    passed = all(gate_results.values())
    return {
        "schema_version": 1,
        "decision": (
            "retain_exact_hard_1nn_as_a0_research_challenger"
            if passed
            else "close_exact_hard_1nn_visual_policy"
        ),
        "intervention_count": len(intervention_ids or set()),
        "routed_severe_count": routed_severe,
        "new_intervention_severe_count": new_severe,
        "blind_presented_severe_marks": blind_severe,
        "positive_score_rounds": positive_rounds,
        "rounds": round_results,
        "gates": gate_results,
    }
