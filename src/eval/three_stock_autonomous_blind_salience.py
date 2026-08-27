"""Hash-bound blind visible-salience review for existing three-stock outputs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

CONTRACT_SCHEMA = "neuro-film.rf3-d15-three-stock-autonomous-blind-salience-contract.v1"
PACKAGE_SCHEMA = "neuro-film.rf3-d15-three-stock-autonomous-blind-salience-package.v1"
OBSERVATIONS_SCHEMA = (
    "neuro-film.rf3-d15-three-stock-autonomous-blind-salience-observations.v1"
)
REPORT_SCHEMA = "neuro-film.rf3-d15-three-stock-autonomous-blind-salience-report.v1"


class BlindSalienceError(ValueError):
    """Raised when frozen inputs or the blind-review boundary drift."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise BlindSalienceError(f"expected JSON object: {path}")
    return raw, value


def _load_contract(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw, contract = _json_object(path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != "FROZEN_BEFORE_BLIND_MATERIAL_BUILD_OR_REVIEW"
    ):
        raise BlindSalienceError("unsupported or unfrozen contract")
    return raw, contract


def _bound_parent(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    relative = Path(str(binding.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise BlindSalienceError("invalid parent path")
    path = root.joinpath(*relative.parts)
    raw, value = _json_object(path)
    if _sha256(raw) != binding.get("sha256"):
        raise BlindSalienceError(f"parent identity drift: {relative}")
    required = binding.get("required_status")
    if required is not None and value.get("status") != required:
        raise BlindSalienceError(f"parent status drift: {relative}")
    return value


def _mapping(
    source_ids: list[str], arm_ids: list[str], seed: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prior: dict[str, tuple[str, ...]] = {}
    for round_index in (1, 2):
        for source_id in source_ids:
            ordered = arm_ids.copy()
            token = hashlib.sha256(
                f"rf3-d15:{seed}:{round_index}:{source_id}".encode()
            ).digest()
            random.Random(int.from_bytes(token[:8], "big")).shuffle(ordered)
            if round_index == 2 and tuple(ordered) == prior[source_id]:
                ordered = ordered[1:] + ordered[:1]
            prior.setdefault(source_id, tuple(ordered))
            rows.append(
                {
                    "round": round_index,
                    "source_id": source_id,
                    "label_to_arm": dict(zip(("A", "B", "C"), ordered, strict=True)),
                }
            )
    return rows


def _sheet(paths: dict[str, Path], destination: Path) -> None:
    opened = {label: Image.open(path).convert("RGB") for label, path in paths.items()}
    try:
        width = 720
        resized: dict[str, Image.Image] = {}
        for label, image in opened.items():
            height = max(1, round(image.height * width / image.width))
            resized[label] = image.resize((width, height), Image.Resampling.LANCZOS)
        height = max(image.height for image in resized.values())
        canvas = Image.new("RGB", (width * 3, height + 56), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for index, label in enumerate(("A", "B", "C")):
            image = resized[label]
            y = 56 + (height - image.height) // 2
            canvas.paste(image, (index * width, y))
            draw.text((index * width + 16, 18), label, fill=(0, 0, 0))
        canvas.save(destination, format="PNG", optimize=False)
    finally:
        for image in opened.values():
            image.close()


def build_package(config_path: Path, root: Path, output_dir: Path) -> dict[str, Any]:
    """Create two label-hidden layouts without rendering or changing pixels."""

    config_raw, config = _load_contract(config_path)
    _bound_parent(root, config["parents"]["severe_review"])
    _bound_parent(root, config["parents"]["automatic_separation"])
    structural = _bound_parent(root, config["parents"]["structural_report"])
    rows = structural.get("scientific_payload", {}).get("rows")
    if not isinstance(rows, list):
        raise BlindSalienceError("structural row inventory missing")
    arm_ids = list(config["arm_ids"])
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    source_ids: set[str] = set()
    render_root = root / config["render_root"]
    for row in rows:
        arm_id = row.get("arm_id")
        if arm_id not in arm_ids:
            continue
        source_id = str(row.get("source_id", ""))
        output = row.get("output", {})
        relative = Path(str(output.get("relative_path", "")))
        if (
            not source_id
            or relative.is_absolute()
            or ".." in relative.parts
            or (source_id, arm_id) in selected
        ):
            raise BlindSalienceError("invalid selected output identity")
        path = render_root.joinpath(*relative.parts)
        if not path.is_file() or _sha256_file(path) != output.get("sha256"):
            raise BlindSalienceError("selected output bytes drift")
        selected[(source_id, arm_id)] = {"path": path, "sha256": output["sha256"]}
        source_ids.add(source_id)
    ordered_sources = sorted(source_ids)
    if len(ordered_sources) != int(config["source_count"]) or set(selected) != {
        (source, arm) for source in ordered_sources for arm in arm_ids
    }:
        raise BlindSalienceError("selected 16x3 population drift")

    logical_outputs = (root / "outputs").resolve()
    destination = output_dir.resolve()
    if (
        destination.exists()
        or destination == logical_outputs
        or not destination.is_relative_to(logical_outputs)
    ):
        raise BlindSalienceError("output must be a new logical outputs subdirectory")
    mapping = _mapping(
        ordered_sources, arm_ids, str(config["blind_protocol"]["seed_sha256"])
    )
    stage = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        public_rows: list[dict[str, Any]] = []
        private_rows: list[dict[str, Any]] = []
        for row in mapping:
            round_index, source_id = row["round"], row["source_id"]
            public_labels = []
            private_labels = {}
            anonymous_paths: dict[str, Path] = {}
            base = Path("blind") / f"round-{round_index}" / source_id
            for label, arm_id in row["label_to_arm"].items():
                source = selected[(source_id, arm_id)]
                relative = base / f"{label}.png"
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source["path"], target)
                if _sha256_file(target) != source["sha256"]:
                    raise BlindSalienceError("anonymous copy identity drift")
                anonymous_paths[label] = target
                public_labels.append(
                    {
                        "label": label,
                        "relative_path": relative.as_posix(),
                        "sha256": source["sha256"],
                    }
                )
                private_labels[label] = {"arm_id": arm_id, "sha256": source["sha256"]}
            sheet_relative = (
                Path("sheets") / f"round-{round_index}" / f"{source_id}.png"
            )
            sheet_path = stage / sheet_relative
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            _sheet(anonymous_paths, sheet_path)
            public_rows.append(
                {
                    "round": round_index,
                    "source_id": source_id,
                    "labels": public_labels,
                    "sheet_relative_path": sheet_relative.as_posix(),
                    "sheet_sha256": _sha256_file(sheet_path),
                }
            )
            private_rows.append(
                {"round": round_index, "source_id": source_id, "labels": private_labels}
            )
        public = {
            "schema": f"{PACKAGE_SCHEMA}.public-sheet",
            "rows": public_rows,
            "stock_labels_present": False,
        }
        private = {
            "schema": f"{PACKAGE_SCHEMA}.private-mapping",
            "domain": config["blind_protocol"]["mapping_domain"],
            "rows": private_rows,
        }
        public_raw, private_raw = _canonical(public), _canonical(private)
        (stage / "public_sheet.json").write_bytes(public_raw)
        (stage / "private_mapping.json").write_bytes(private_raw)
        core = {
            "schema": PACKAGE_SCHEMA,
            "experiment_id": config["experiment_id"],
            "contract_sha256": _sha256(config_raw),
            "source_count": len(ordered_sources),
            "arm_count": len(arm_ids),
            "round_count": 2,
            "anonymous_png_count": len(public_rows) * 3,
            "sheet_count": len(public_rows),
            "public_sheet_sha256": _sha256(public_raw),
            "private_mapping_sha256": _sha256(private_raw),
            "render_calls": 0,
            "network_reads": 0,
            "automatic_pass": True,
            "decision": "OPEN_LABEL_HIDDEN_AUTONOMOUS_VISUAL_OBSERVATIONS",
            "claim_ceiling": config["claim_ceiling"],
        }
        report = {**core, "stable_evidence_id": _sha256(_canonical(core))}
        (stage / "report.json").write_bytes(_canonical(report))
        os.rename(stage, destination)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def adjudicate_package(
    config_path: Path,
    root: Path,
    package_dir: Path,
    observations_path: Path,
    observations_commit: str,
) -> dict[str, Any]:
    """Reveal mappings only after the observations blob exists in Git."""

    config_raw, config = _load_contract(config_path)
    package_raw, package = _json_object(package_dir / "report.json")
    public_raw, public = _json_object(package_dir / "public_sheet.json")
    mapping_raw, mapping = _json_object(package_dir / "private_mapping.json")
    observations_raw, observations = _json_object(observations_path)
    if (
        package.get("schema") != PACKAGE_SCHEMA
        or package.get("contract_sha256") != _sha256(config_raw)
        or package.get("public_sheet_sha256") != _sha256(public_raw)
        or package.get("private_mapping_sha256") != _sha256(mapping_raw)
        or public.get("stock_labels_present") is not False
    ):
        raise BlindSalienceError("blind package identity drift")
    relative_observations = (
        observations_path.resolve().relative_to(root.resolve()).as_posix()
    )
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{observations_commit}:{relative_observations}",
        ],
        check=False,
        capture_output=True,
    )
    committed = completed.stdout if completed.returncode == 0 else b""
    if not committed or _sha256(committed) != _sha256(observations_raw):
        raise BlindSalienceError("observations were not frozen in the claimed commit")
    if (
        observations.get("schema") != OBSERVATIONS_SCHEMA
        or observations.get("status") != "OBSERVATIONS_FROZEN_BEFORE_MAPPING_REVEAL"
        or observations.get("mapping_read_before_freeze") is not False
        or observations.get("public_sheet_sha256") != _sha256(public_raw)
    ):
        raise BlindSalienceError("observation boundary drift")

    truth = {
        (int(row["round"]), row["source_id"]): {
            label: facts["arm_id"] for label, facts in row["labels"].items()
        }
        for row in mapping["rows"]
    }
    expected_keys = set(truth)
    seen: set[tuple[int, str]] = set()
    decisions: dict[tuple[int, str, tuple[str, str]], bool] = {}
    for row in observations.get("rows", []):
        key = (int(row.get("round", 0)), str(row.get("source_id", "")))
        if key not in expected_keys or key in seen:
            raise BlindSalienceError("unknown or duplicate observation row")
        seen.add(key)
        pairs = row.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != 3:
            raise BlindSalienceError("incomplete observation pairs")
        labels_seen: set[tuple[str, str]] = set()
        for pair in pairs:
            labels = tuple(
                sorted((str(pair.get("left", "")), str(pair.get("right", ""))))
            )
            if (
                labels not in {("A", "B"), ("A", "C"), ("B", "C")}
                or labels in labels_seen
            ):
                raise BlindSalienceError("invalid or duplicate label pair")
            if not isinstance(pair.get("visibly_distinguishable"), bool):
                raise BlindSalienceError("pair decision must be boolean")
            labels_seen.add(labels)
            arms = tuple(sorted((truth[key][labels[0]], truth[key][labels[1]])))
            decisions[(key[0], key[1], arms)] = pair["visibly_distinguishable"]
    if seen != expected_keys:
        raise BlindSalienceError("observation inventory incomplete")

    pair_rows = []
    all_pairs = sorted(
        tuple(sorted(pair)) for pair in combinations(config["arm_ids"], 2)
    )
    passed = True
    for arms in all_pairs:
        both_visible = 0
        disagreements = 0
        for source_id in sorted({key[1] for key in expected_keys}):
            first = decisions[(1, source_id, arms)]
            second = decisions[(2, source_id, arms)]
            both_visible += int(first and second)
            disagreements += int(first != second)
        pair_pass = both_visible >= int(
            config["gates"]["minimum_visible_source_count_per_stock_pair"]
        ) and disagreements <= int(
            config["gates"]["maximum_round_disagreement_count_per_stock_pair"]
        )
        passed = passed and pair_pass
        pair_rows.append(
            {
                "arm_ids": list(arms),
                "both_rounds_visible_source_count": both_visible,
                "round_disagreement_count": disagreements,
                "source_count": int(config["source_count"]),
                "pair_gate_pass": pair_pass,
            }
        )
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "contract_sha256": _sha256(config_raw),
        "package_report_sha256": _sha256(package_raw),
        "observations_sha256": _sha256(observations_raw),
        "observations_commit": observations_commit,
        "pair_results": pair_rows,
        "prior_labeled_exposure_disclosed": True,
        "automatic_pass": passed,
        "decision": config["decision_if_pass" if passed else "decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**core, "scientific_identity": _sha256(_canonical(core))}


__all__ = ["BlindSalienceError", "adjudicate_package", "build_package"]
