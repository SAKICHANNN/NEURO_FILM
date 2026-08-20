#!/usr/bin/env python3
"""Build the metadata-only REPID fit/calibration original-member lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_u5_r2repid2_shared_operator_roles import _post_paths


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifest(
    root: Path,
    config: dict[str, Any],
    *,
    reverse: bool = False,
    post_paths=_post_paths,
) -> dict[str, Any]:
    parent = config["parent"]
    roles_path = root / parent["roles_report_path"]
    evidence_path = root / parent["roles_evidence_path"]
    roles_bytes = roles_path.read_bytes()
    evidence_bytes = evidence_path.read_bytes()
    roles_report = json.loads(roles_bytes)
    included = set(config["roles"]["included"])
    selected = [row for row in roles_report["selected"]["rows"] if row["role"] in included]
    if reverse:
        selected.reverse()
    template = config["source"]["member_path_template"]
    requested = [template.format(scene_id=row["scene_id"]) for row in selected]
    resolved_rows = post_paths(config["source"]["paths_info_url"], requested)
    resolved = {row["path"]: row for row in resolved_rows}
    members = []
    role_by_scene = {row["scene_id"]: row["role"] for row in selected}
    for path in requested:
        row = resolved.get(path, {})
        lfs = row.get("lfs") or {}
        scene_id = Path(path).name
        members.append(
            {
                "scene_id": scene_id,
                "role": role_by_scene[scene_id],
                "path": path,
                "size": row.get("size"),
                "sha256": lfs.get("oid"),
            }
        )
    members.sort(key=lambda row: (str(row["role"]), str(row["scene_id"])))
    role_counts = Counter(str(row["role"]) for row in members)
    total_bytes = sum(int(row["size"] or 0) for row in members)
    gates_spec = config["gates"]
    gates = {
        "roles_report_exact": _sha256(roles_bytes) == parent["roles_report_sha256"],
        "roles_evidence_exact": _sha256(evidence_bytes) == parent["roles_evidence_sha256"],
        "selected_identity_exact": roles_report["selected"]["identity_sha256"] == parent["selected_identity_sha256"],
        "scene_count_exact": len({row["scene_id"] for row in members}) == int(gates_spec["required_scene_count"]),
        "member_count_exact": len(members) == int(gates_spec["required_member_count"]),
        "role_counts_exact": role_counts == Counter({"fit": 80, "calibration": 24}),
        "all_paths_resolved": len(resolved) == len(requested),
        "all_paths_lfs_sha256_present": all(row["sha256"] for row in members),
        "all_paths_jpeg_extension": all(Path(str(row["path"])).suffix.lower() in {".jpg", ".jpeg"} for row in members),
        "total_bytes": total_bytes <= int(gates_spec["maximum_total_bytes"]),
        "sealed_scene_requests_zero": int(config["roles"]["sealed_scene_requests"]) == 0,
        "member_payload_reads_zero": int(gates_spec["member_payload_reads"]) == 0,
        "image_decodes_zero": int(gates_spec["image_decodes"]) == 0,
        "operator_fits_zero": int(gates_spec["operator_fits"]) == 0,
    }
    passed = all(gates.values())
    canonical_members = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "source": config["source"],
        "prospective_adoption": config["prospective_adoption"],
        "role_counts": dict(sorted(role_counts.items())),
        "scene_count": len({row["scene_id"] for row in members}),
        "member_count": len(members),
        "total_bytes": total_bytes,
        "member_identity_sha256": _sha256(canonical_members),
        "members": members,
        "execution": {"member_payload_reads": 0, "image_decodes": 0, "operator_fits": 0, "sealed_scene_requests": 0},
        "gates": gates,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u5_r2repid9a_original_member_lock_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    manifest = build_manifest(ROOT, json.loads(config_path.read_text(encoding="utf-8")), reverse=args.reverse)
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({"decision": manifest["decision"], "sha256": _sha256(encoded)}, sort_keys=True))
    return 0 if manifest["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
