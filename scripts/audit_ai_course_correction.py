from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CATALOG = "outputs/user_style_catalog_20260908/catalog.json"
EVIDENCE = "docs/evidence/"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(root: Path, path: str) -> dict:
    return json.loads((root / path).read_text(encoding="utf-8"))


def verify_report(root: Path, path: str, expected: str) -> dict:
    actual = sha(root / path)
    if actual != expected:
        raise ValueError(f"Report drift: {path}")
    return {"path": path, "sha256": actual, "verified": True}


def gallery_role(path: str) -> dict:
    # Exact reviewed artifacts, never infer learning or quality from a filename.
    examples = {
        "ai_classneg_photo_review_v1/03_learned.png": ("classneg", "development"),
        "ai_classneg_photo_review_v1/09_learned.png": ("classneg", "development"),
        "ai_photo_distribution_pilot_v1/04_lut.png": ("distribution", "training"),
        "ai_photo_distribution_pilot_v1/06_lut.png": (
            "distribution",
            "transfer_development",
        ),
        "ai_vcg_reference_development_v3/02_learned.png": (
            "vcg",
            "consumed_development",
        ),
    }
    family, role = examples.get(path, (None, "unverified_or_control"))
    return {
        "family": family,
        "data_role": role,
        "learned_example": family is not None,
        "promotion": False,
        "independent_assessment": False,
    }


def recovery_matrix(root: Path, images: dict) -> dict:
    prefixes = (
        "ai_recovery_20260907/",
        "ai_nlut_reference_development_v2/",
        "ai_vcg_reference_development_v3/",
        "ai_vcg_ncc_ablation_v1/",
        "ai_photo_distribution_pilot_v1/",
        "ai_clip_lut_pilot_v1/",
    )
    paths = {im["path"]: im for im in images.values()}
    groups, unresolved, metadata = {}, [], {}
    manifest_links = {}
    output_root = (root / "outputs").resolve()
    parent_path = "outputs/ai_vcg_reference_development_v3/report.json"
    ncc_path = "outputs/ai_vcg_ncc_ablation_v1/report.json"
    if (root / parent_path).exists() and (root / ncc_path).exists():
        ncc = read(root, ncc_path)
        verify_report(root, parent_path, ncc["parent_report_sha256"])
        parent = read(root, parent_path)
        for row in ncc["rows"]:
            pair = parent["rows"][row["id"]]["pair"]
            source_path = Path(pair["content_path"]).resolve()
            if source_path.is_relative_to(output_root):
                source_key = source_path.relative_to(output_root).as_posix()
                if source_key.startswith(
                    "color_baseline/velvia50_rawpixls20_s0p50_gamutsafe/inputs/"
                ):
                    verify_report(root, "outputs/" + source_key, pair["content_sha256"])
                    manifest_links[
                        f"ai_vcg_ncc_ablation_v1/{row['id']:02d}_ncc.png"
                    ] = {
                        "path": source_key,
                        "link_evidence": ncc_path
                        + " -> hash-bound parent content_path",
                    }
    for manifest in (root / "outputs/ai_recovery_20260907").glob("**/manifest.csv"):
        with manifest.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("before") or not row.get("after"):
                    continue
                before_path, after_path = (
                    Path(row["before"]).resolve(),
                    Path(row["after"]).resolve(),
                )
                if before_path.is_relative_to(
                    output_root
                ) and after_path.is_relative_to(output_root):
                    before_key = before_path.relative_to(output_root).as_posix()
                    after_key = after_path.relative_to(output_root).as_posix()
                    if before_key.startswith(
                        "color_baseline/velvia50_rawpixls20_s0p50_gamutsafe/inputs/"
                    ):
                        manifest_links[after_key] = {
                            "path": before_key,
                            "link_evidence": str(manifest.relative_to(root)),
                        }

    def identity(path):
        if path not in metadata:
            file = root / "outputs" / path
            with Image.open(file) as image:
                metadata[path] = {
                    "sha256": sha(file),
                    "size": list(image.size),
                    "icc_sha256": hashlib.sha256(
                        image.info.get("icc_profile", b"")
                    ).hexdigest(),
                }
        return metadata[path]

    for im in images.values():
        if not im["path"].startswith(prefixes):
            continue
        path = Path(im["path"])
        if any(
            word in path.name
            for word in (
                "comparison",
                "contact_sheet",
                "_identity",
                "_original",
                "_reference",
            )
        ) or any(part in path.parts for part in ("diff_maps", "before")):
            continue
        before = images.get(im.get("before_id"))
        if before is None:
            number = path.name.split("_")[0]
            candidates = [
                path.parent / (number + suffix)
                for suffix in ("_identity.png", "_original.png")
            ]
            if path.parent.name in ("after", "safe_rich"):
                candidates.append(
                    path.parent.parent
                    / "before"
                    / f"{number}_{path.parent.parent.name}_before.png"
                )
            before = next(
                (paths[p.as_posix()] for p in candidates if p.as_posix() in paths), None
            )
        if before is None:
            before = manifest_links.get(im["path"])
        if before is None:
            unresolved.append(im["path"])
            continue
        source = identity(before["path"])
        group = groups.setdefault(
            source["sha256"],
            {"original": before["path"], "original_metadata": source, "outputs": []},
        )
        group["outputs"].append(
            {
                "path": im["path"],
                "metadata": identity(im["path"]),
                "family": im["path"].split("/")[0],
                "original_link": before.get(
                    "link_evidence", "catalog or same-directory identity filename"
                ),
            }
        )
    return {
        "grouping": "exact saved-original byte hash only; no semantic matching, preprocessing or training-role equivalence implied",
        "groups": list(groups.values()),
        "unresolved_originals": unresolved,
        "comparison_ready": False,
    }


def audit(root: Path) -> dict:
    checks = []
    decisions = {}
    for family, filename in [
        ("classneg", "AI_CLASSNEG_PHOTOGRAPHIC_REVIEW_20260907.json"),
        ("vcg", "AI_VCG_REFERENCE_DEVELOPMENT_20260907.json"),
        ("nlut", "AI_NLUT_REFERENCE_DEVELOPMENT_20260907.json"),
    ]:
        source = read(root, EVIDENCE + filename)
        report = source["report"]
        path, expected = (
            (report["path"], report["sha256"])
            if isinstance(report, dict)
            else (report, source["report_sha256"])
        )
        checks.append(verify_report(root, path, expected))
        decisions[family] = {
            "record": EVIDENCE + filename,
            "record_sha256": sha(root / EVIDENCE / filename),
            "decision": source.get("decision", source.get("status")),
        }
    ncc = read(root, EVIDENCE + "AI_VCG_STAGE_GAIN_AND_NCC_DIAGNOSTIC_20260907.json")
    for key in ("gain_report", "ncc_report"):
        checks.append(verify_report(root, ncc[key], ncc[key + "_sha256"]))
    checks.append(
        verify_report(
            root,
            "outputs/ai_photo_distribution_pilot_v1/report.json",
            "1e4cd6d8e9364260e051ab867ca7a73db01cef8ace4e61c9d66e409f805a52ee",
        )
    )
    for config in ("ai_classneg_photo_review_v1", "ai_nlut_reference_development_v1"):
        cfg = read(root, f"configs/{config}.json")
        checks.append(verify_report(root, cfg["checkpoint"], cfg["checkpoint_sha256"]))
    teachers = []
    for family in (
        "scheme_a_distilled_v1",
        "scheme_b_nilut_distilled_v1",
        "scheme_c_context4d_distilled_v1",
    ):
        path = f"outputs/neural_film_lut_v2/{family}/metrics.json"
        metrics = read(root, path)
        teachers.append(
            {
                "family": family,
                "record": path,
                "sha256": sha(root / path),
                "teacher": metrics["teacher_root"],
                "role": "authored_teacher_control",
            }
        )
    catalog = read(root, CATALOG)
    images = {im["id"]: im for im in catalog["images"]}
    reviewed = []
    for im in catalog["images"]:
        if im.get("visibility_review") != "visible":
            continue
        role = gallery_role(im["path"])
        row = {"image_id": im["id"], "path": im["path"], **role}
        if role["learned_example"]:
            source = images[im["before_id"]]
            row.update(
                {
                    "before_path": source["path"],
                    "before_sha256": sha(root / "outputs" / source["path"]),
                    "after_sha256": sha(root / "outputs" / im["path"]),
                }
            )
        reviewed.append(row)
    expected = 5
    if sum(r["learned_example"] for r in reviewed) != expected:
        raise ValueError(
            "Reviewed-example inventory changed; re-audit rather than infer new eligibility"
        )
    violations = []
    for im in images.values():
        if im.get("selection_eligible") and (
            not gallery_role(im["path"])["learned_example"]
            or im.get("product_promotion") is not False
            or not im.get("family_decision")
        ):
            violations.append(im["path"])
    if violations:
        raise ValueError(f"Catalog claims/eligibility drift: {violations}")
    return {
        "schema": "neuro-film.ai-course-correction.v1",
        "scope": "Local evidence/provenance audit; no inference, training or new photographic assessment",
        "authority": "docs/ops/ASTRA_NATIVE_HANDOFF_20260908.md",
        "verified_report_and_checkpoint_bindings": checks,
        "family_decisions": decisions,
        "teacher_lineage": teachers,
        "existing_recovery_matrix": recovery_matrix(root, images),
        "gallery_reviewed_examples": reviewed,
        "gallery_coverage": {
            "catalog_images": len(images),
            "visually_reviewed_pairs": 135,
            "visible_selected_images": len(reviewed),
            "learned_browsing_examples": expected,
            "exhaustive_best_model_recovery": False,
        },
        "gallery_contract_violations": violations,
        "promotion": False,
        "unresolved": [
            "earlier preferred AI checkpoint identity",
            "matched-input/preprocessing full family comparison",
            "independent photographic superiority",
            "native-resolution face/text/content coverage",
            "credible successor supervision",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(ROOT)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        # Never overwrite a previous audit result.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
