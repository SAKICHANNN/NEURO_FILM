#!/usr/bin/env python3
"""Build and execute the frozen native Standard sources across toolchains."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402
from src.eval.native_standard_portable import (  # noqa: E402
    build_android_libraries,
    build_apple_abi_objects,
    build_hashes,
    build_windows_dlls,
    sha256_file,
    validate_contract,
    validate_llvm_toolchain,
    validate_ndk,
)
from src.film_physics.native_standard_factory import (  # noqa: E402
    create_opt_in_native_standard_runtime,
)
from src.film_physics.profile_consumer import (  # noqa: E402
    compile_standalone_profile_artifact,
)


DEFAULT_LLVM = Path(
    r"C:\Users\hhvrf\Documents\追色\outputs\tmp\tools"
    r"\llvm-mingw-20260616-ucrt-x86_64"
)
DEFAULT_NDK = Path(
    r"C:\Users\hhvrf\Documents\追色\outputs\tmp\tools"
    r"\android-ndk-r27d"
)


def _render(runtime: Any, height: int, width: int) -> tuple[bytes, dict]:
    source = np.ascontiguousarray(
        p8aq._source_rows(
            y0=0,
            y1=height,
            height=height,
            width=width,
        ),
        dtype=np.float32,
    )
    chunks: list[bytes] = []
    receipt = runtime.render_to_sink(
        source,
        output_sink=lambda y0, y1, rows: chunks.append(
            rows.tobytes()
        ),
    )
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != receipt["output"][
        "array_sha256"
    ]:
        raise RuntimeError("P8BO runtime receipt drift")
    return payload, receipt


def _repeat_equal(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise RuntimeError(f"P8BO {label} repeat build drift")


def run(
    *,
    contract: dict[str, Any],
    llvm: Path,
    ndk: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_contract(ROOT, contract)
    llvm_clang, readobj = validate_llvm_toolchain(
        llvm, contract["toolchains"]["llvm_mingw"]
    )
    ndk_clang, readelf = validate_ndk(
        ndk, contract["toolchains"]["android_ndk"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    windows_a = build_windows_dlls(
        root=ROOT,
        contract=contract,
        clang=llvm_clang,
        output_dir=output_dir / "llvm_windows_a",
    )
    windows_b = build_windows_dlls(
        root=ROOT,
        contract=contract,
        clang=llvm_clang,
        output_dir=output_dir / "llvm_windows_b",
    )
    _repeat_equal(
        build_hashes(windows_a),
        build_hashes(windows_b),
        "LLVM Windows",
    )

    android_a = build_android_libraries(
        root=ROOT,
        contract=contract,
        clang=ndk_clang,
        readelf=readelf,
        output_dir=output_dir / "android_a",
    )
    android_b = build_android_libraries(
        root=ROOT,
        contract=contract,
        clang=ndk_clang,
        readelf=readelf,
        output_dir=output_dir / "android_b",
    )
    _repeat_equal(android_a, android_b, "Android")

    apple_a = build_apple_abi_objects(
        root=ROOT,
        contract=contract,
        clang=llvm_clang,
        readobj=readobj,
        output_dir=output_dir / "apple_a",
    )
    apple_b = build_apple_abi_objects(
        root=ROOT,
        contract=contract,
        clang=llvm_clang,
        readobj=readobj,
        output_dir=output_dir / "apple_b",
    )
    _repeat_equal(apple_a, apple_b, "Apple ABI object")

    package = json.loads(
        (ROOT / contract["windows_package"]).read_text(encoding="utf-8")
    )
    profile_config = json.loads(
        (ROOT / contract["profile_compiler_config"]).read_text(
            encoding="utf-8"
        )
    )
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=profile_config
    )
    p8aw._patch_runtime()
    msvc_builds = p8aq._build_components(
        {
            "component_dll_sha256": {
                name: row["sha256"]
                for name, row in package["components"].items()
            }
        },
        output_dir / "msvc",
    )
    msvc_runtime, msvc_factory = create_opt_in_native_standard_runtime(
        package=package,
        artifact=artifact,
        library_paths={
            name: Path(row["dll_path"])
            for name, row in msvc_builds.items()
        },
    )

    llvm_package = deepcopy(package)
    for name, row in windows_a.items():
        llvm_package["components"][name]["sha256"] = row["sha256"]
    llvm_runtime, llvm_factory = create_opt_in_native_standard_runtime(
        package=llvm_package,
        artifact=artifact,
        library_paths={
            name: Path(row["path"]) for name, row in windows_a.items()
        },
    )
    height = int(contract["oracle"]["height"])
    width = int(contract["oracle"]["width"])
    msvc_bytes, msvc_receipt = _render(msvc_runtime, height, width)
    llvm_bytes, llvm_receipt = _render(llvm_runtime, height, width)
    if msvc_bytes != llvm_bytes:
        raise RuntimeError("P8BO Windows compiler output mismatch")

    stable = {
        "schema": "neuro_film.u6_p8bo_portable_native_standard_result.v1",
        "contract_sha256": hashlib.sha256(
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "toolchains": {
            "llvm_clang_sha256": sha256_file(llvm_clang),
            "android_source_properties_sha256": sha256_file(
                ndk / "source.properties"
            ),
        },
        "windows": {
            "llvm_component_sha256": build_hashes(windows_a),
            "repeat_byte_exact": True,
            "msvc_component_sha256": {
                name: row["sha256"]
                for name, row in package["components"].items()
            },
            "oracle_shape": [height, width, 3],
            "oracle_output_sha256": hashlib.sha256(msvc_bytes).hexdigest(),
            "msvc_output_receipt_sha256": msvc_receipt["receipt_sha256"],
            "llvm_output_receipt_sha256": llvm_receipt["receipt_sha256"],
            "msvc_factory_receipt_sha256": msvc_factory[
                "receipt_sha256"
            ],
            "llvm_factory_receipt_sha256": llvm_factory[
                "receipt_sha256"
            ],
            "output_byte_exact": True,
        },
        "android": {
            "targets": android_a,
            "repeat_byte_exact": True,
            "claim": "compile-link-only",
        },
        "apple": {
            "targets": apple_a,
            "repeat_byte_exact": True,
            "claim": "header-and-ABI-object-only",
        },
        "strength": 1.0,
        "all_gates_pass": True,
        "production_default_changed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable["stable_evidence_id"] = hashlib.sha256(
        json.dumps(
            stable,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return stable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p8bo_portable_native_standard_v1.json",
    )
    parser.add_argument("--llvm", type=Path, default=DEFAULT_LLVM)
    parser.add_argument("--ndk", type=Path, default=DEFAULT_NDK)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "outputs/u6_p8bo_portable_native_standard_v1/build",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u6_p8bo_portable_native_standard_v1/report.json",
    )
    args = parser.parse_args()
    contract = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(
        contract=contract,
        llvm=args.llvm,
        ndk=args.ndk,
        output_dir=args.build_dir,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"oracle_output_sha256={report['windows']['oracle_output_sha256']}")
    print(
        "report_sha256="
        + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
