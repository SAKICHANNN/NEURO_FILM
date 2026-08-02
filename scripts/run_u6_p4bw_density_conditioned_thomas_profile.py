from __future__ import annotations

import json
from pathlib import Path

from src.eval.density_conditioned_thomas_profile import compile_and_evaluate, write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "configs/u6_p4bw_density_conditioned_thomas_profile_v1.json").read_text(
            encoding="utf-8"
        )
    )
    bundle_a, report_a = compile_and_evaluate(contract, root)
    bundle_a_sha = write_json(bundle_a, root / contract["outputs"]["bundle_a"])
    report_a_sha = write_json(report_a, root / contract["outputs"]["report_a"])
    bundle_b, report_b = compile_and_evaluate(contract, root)
    bundle_b_sha = write_json(bundle_b, root / contract["outputs"]["bundle_b"])
    report_b_sha = write_json(report_b, root / contract["outputs"]["report_b"])
    if (
        bundle_a != bundle_b
        or report_a != report_b
        or bundle_a_sha != bundle_b_sha
        or report_a_sha != report_b_sha
    ):
        raise SystemExit("P4BW repeat drift")
    print(
        json.dumps(
            {
                "bundle_sha256": bundle_a_sha,
                "report_sha256": report_a_sha,
                **report_a,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
