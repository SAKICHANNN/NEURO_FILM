"""Build a minimal physical-device package for the consumer sRGB quantizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_srgb_oetf_quantize_streaming_v1 import _linear_samples
from scripts.build_srgb_oetf_quantize_c_v1 import (
    accepted_linear_bounds,
    quantize_reference,
    thresholds,
)
from scripts.build_srgb_oetf_quantize_native_v1 import build_android
from scripts.build_reference_chain_android import _validate_ndk


HARNESS = ROOT / "runtime" / "android_srgb_quantizer_testlab"
JAVA_SOURCE = (
    HARNESS
    / "test/src/com/neurofilm/srgbquantizer/QuantizerInstrumentation.java"
)
JNI_SOURCE = HARNESS / "test/jni/nf_srgb_quantizer_testlab.c"
APP_MANIFEST = HARNESS / "app/AndroidManifest.xml"
TEST_MANIFEST = HARNESS / "test/AndroidManifest.xml"
PACKAGE_PREFIX = "nf-019f9f37-p90"
DEFAULT_TOOL_ROOT = (
    ROOT.parent / "\u8ffd\u8272" / "outputs" / "tmp" / "tools"
)
DEFAULT_SDK = DEFAULT_TOOL_ROOT / "android-sdk"
DEFAULT_NDK = DEFAULT_TOOL_ROOT / "android-ndk-r27d"
VECTOR_COUNT = 4096
VECTOR_PREFIX = b"neuro-film.android-srgb-quantizer-vector.v1\0"


def _run(arguments: list[object], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        [str(argument) for argument in arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=input_text,
    )
    if completed.returncode != 0:
        command = " ".join(str(argument) for argument in arguments)
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command}\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return completed.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_zip_entries(archive: Path, entries: dict[str, Path]) -> None:
    with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_STORED) as apk:
        for name, source in sorted(entries.items()):
            info = zipfile.ZipInfo(name)
            info.date_time = (2009, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            apk.writestr(info, source.read_bytes())


def _ensure_debug_key(keytool: Path, output: Path) -> Path:
    keystore = output / "testlab-debug.keystore"
    if not keystore.is_file():
        _run(
            [
                keytool,
                "-genkeypair",
                "-keystore",
                keystore,
                "-storepass",
                "android",
                "-keypass",
                "android",
                "-alias",
                "nf-srgb-quantizer",
                "-keyalg",
                "RSA",
                "-keysize",
                "2048",
                "-validity",
                "3650",
                "-dname",
                (
                    "CN=Neuro-Film sRGB Quantizer,OU=Research,"
                    "O=Local,L=Local,C=ZZ"
                ),
            ]
        )
    return keystore


def _vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    linear = _linear_samples(0, VECTOR_COUNT).copy()
    lower, upper = accepted_linear_bounds()
    linear[:8] = np.asarray(
        [
            lower,
            -0.0,
            0.0,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.float32(0.0031308),
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            np.float32(1.0),
            upper,
        ],
        dtype=np.float32,
    )
    q8 = quantize_reference(linear, 8)
    q16 = quantize_reference(linear, 16)
    _, _, threshold_identity = thresholds()
    vector_payload = (
        VECTOR_PREFIX
        + linear.view(np.uint32).astype(">u4", copy=False).tobytes()
        + q8.tobytes()
        + q16.astype(">u2", copy=False).tobytes()
    )
    identities = {
        "threshold_identity": threshold_identity,
        "vector_sha256": hashlib.sha256(vector_payload).hexdigest(),
        "input_sha256": hashlib.sha256(linear.tobytes()).hexdigest(),
        "q8_sha256": hashlib.sha256(q8.tobytes()).hexdigest(),
        "q16_sha256": hashlib.sha256(q16.tobytes()).hexdigest(),
    }
    return linear, q8, q16, identities


def _encode_values(
    name: str,
    c_type: str,
    values: np.ndarray,
    formatter,
) -> str:
    rows = []
    for offset in range(0, values.size, 12):
        rows.append(
            "    "
            + ", ".join(
                formatter(value) for value in values[offset : offset + 12]
            )
            + ","
        )
    return (
        f"static const {c_type} {name}[{values.size}u] = {{\n"
        + "\n".join(rows)
        + "\n};\n"
    )


def encode_vector_header() -> tuple[str, dict[str, str]]:
    linear, q8, q16, identities = _vectors()
    input_bits = linear.view(np.uint32)
    return (
        f"""#ifndef NF_SRGB_QUANTIZER_VECTORS_V1_H
#define NF_SRGB_QUANTIZER_VECTORS_V1_H

#include <stdint.h>

#define NF_SRGB_QUANTIZER_VECTOR_COUNT {VECTOR_COUNT}u
#define NF_SRGB_QUANTIZER_THRESHOLD_ID "{identities["threshold_identity"]}"
#define NF_SRGB_QUANTIZER_VECTOR_SHA256 "{identities["vector_sha256"]}"
#define NF_SRGB_QUANTIZER_INPUT_SHA256 "{identities["input_sha256"]}"
#define NF_SRGB_QUANTIZER_Q8_SHA256 "{identities["q8_sha256"]}"
#define NF_SRGB_QUANTIZER_Q16_SHA256 "{identities["q16_sha256"]}"

{_encode_values("NF_SRGB_QUANTIZER_INPUT_BITS", "uint32_t", input_bits, lambda value: f"0x{int(value):08x}u")}
{_encode_values("NF_SRGB_QUANTIZER_EXPECTED_Q8", "uint8_t", q8, lambda value: f"{int(value)}u")}
{_encode_values("NF_SRGB_QUANTIZER_EXPECTED_Q16", "uint16_t", q16, lambda value: f"{int(value)}u")}
#endif
""",
        identities,
    )


def build(
    sdk: Path,
    ndk: Path,
    output: Path,
) -> dict[str, Any]:
    sdk = sdk.resolve()
    ndk = ndk.resolve()
    output = output.resolve()
    tools = sdk / "build-tools/35.0.0"
    android_jar = sdk / "platforms/android-35/android.jar"
    aapt2 = tools / "aapt2.exe"
    d8 = tools / "d8.bat"
    zipalign = tools / "zipalign.exe"
    apksigner = tools / "apksigner.bat"
    javac = Path(
        _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Command javac).Source",
            ]
        ).strip()
    )
    keytool = javac.with_name("keytool.exe")
    jar = javac.with_name("jar.exe")
    for required in (
        android_jar,
        aapt2,
        d8,
        zipalign,
        apksigner,
        javac,
        keytool,
        jar,
        JAVA_SOURCE,
        JNI_SOURCE,
        APP_MANIFEST,
        TEST_MANIFEST,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    ndk_lock = _validate_ndk(ndk)
    output.mkdir(parents=True, exist_ok=True)
    classes = output / "classes"
    dex = output / "dex"
    native_output = output / "native"
    generated = output / "generated"
    for directory in (classes, dex, native_output, generated):
        directory.mkdir(exist_ok=True)

    core_report = build_android(ndk, native_output)
    core_library = (
        native_output
        / "libneuro_film_srgb_oetf_quantize_arm64-v8a.so"
    )
    vector_source, vector_identities = encode_vector_header()
    vector_header = generated / "nf_srgb_quantizer_vectors_v1.h"
    vector_header.write_text(
        vector_source,
        encoding="ascii",
        newline="\n",
    )
    compiler = (
        ndk
        / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
    )
    readelf = compiler.with_name("llvm-readelf.exe")
    jni_library = native_output / "libnf_srgb_quantizer_testlab.so"
    _run(
        [
            compiler,
            "--target=aarch64-linux-android24",
            "-std=c11",
            "-O2",
            "-fPIC",
            "-fvisibility=hidden",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            JNI_SOURCE,
            "-I",
            generated,
            "-I",
            ndk
            / "toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include",
            "-ldl",
            "-Wl,--no-undefined",
            "-Wl,--build-id=sha1",
            "-o",
            jni_library,
        ]
    )
    dynamic = _run([readelf, "--dynamic", jni_library])
    if "libdl.so" not in dynamic or "libc.so" not in dynamic:
        raise ValueError("JNI library dependency closure mismatch")

    _run(
        [
            javac,
            "-source",
            "8",
            "-target",
            "8",
            "-encoding",
            "UTF-8",
            "-classpath",
            android_jar,
            "-d",
            classes,
            JAVA_SOURCE,
        ]
    )
    _run([jar, "cf", output / "classes.jar", "-C", classes, "."])
    _run(
        [
            d8,
            "--min-api",
            "24",
            "--lib",
            android_jar,
            "--output",
            dex,
            output / "classes.jar",
        ]
    )

    app_unsigned = output / "nf-srgb-quantizer-app-unsigned.apk"
    test_unsigned = output / "nf-srgb-quantizer-test-unsigned.apk"
    for manifest, destination in (
        (APP_MANIFEST, app_unsigned),
        (TEST_MANIFEST, test_unsigned),
    ):
        _run(
            [
                aapt2,
                "link",
                "-I",
                android_jar,
                "--manifest",
                manifest,
                "--min-sdk-version",
                "24",
                "--target-sdk-version",
                "35",
                "-o",
                destination,
            ]
        )
    _add_zip_entries(
        test_unsigned,
        {
            "classes.dex": dex / "classes.dex",
            "lib/arm64-v8a/libnf_srgb_quantizer_testlab.so": jni_library,
            (
                "lib/arm64-v8a/"
                "libneuro_film_srgb_oetf_quantize_arm64-v8a.so"
            ): core_library,
        },
    )
    app_badging = _run([aapt2, "dump", "badging", app_unsigned])
    test_badging = _run([aapt2, "dump", "badging", test_unsigned])
    if (
        "package: name='com.neurofilm.srgbquantizer.target'"
        not in app_badging
    ):
        raise ValueError("app APK package mismatch")
    if (
        "package: name='com.neurofilm.srgbquantizer.test'"
        not in test_badging
    ):
        raise ValueError("test APK package mismatch")
    test_xmltree = _run(
        [
            aapt2,
            "dump",
            "xmltree",
            test_unsigned,
            "--file",
            "AndroidManifest.xml",
        ]
    )
    if (
        '="com.neurofilm.srgbquantizer.QuantizerInstrumentation"'
        not in test_xmltree
        or '="com.neurofilm.srgbquantizer.target"' not in test_xmltree
        or "E: instrumentation" not in test_xmltree
    ):
        raise ValueError("instrumentation runner/target mismatch")

    keystore = _ensure_debug_key(keytool, output)
    artifacts: dict[str, dict[str, object]] = {}
    for role, unsigned in (
        ("app", app_unsigned),
        ("test", test_unsigned),
    ):
        aligned = output / f"nf-srgb-quantizer-{role}-aligned.apk"
        signed = output / f"nf-srgb-quantizer-{role}.apk"
        _run([zipalign, "-f", "4", unsigned, aligned])
        shutil.copyfile(aligned, signed)
        _run(
            [
                apksigner,
                "sign",
                "--ks",
                keystore,
                "--ks-pass",
                "pass:android",
                "--key-pass",
                "pass:android",
                "--ks-key-alias",
                "nf-srgb-quantizer",
                signed,
            ]
        )
        certificate = _run(
            [
                apksigner,
                "verify",
                "--verbose",
                "--print-certs",
                signed,
            ]
        )
        match = re.search(
            r"Signer #1 certificate SHA-256 digest: ([0-9a-fA-F]+)",
            certificate,
        )
        if match is None:
            raise ValueError("signed APK certificate digest missing")
        artifacts[role] = {
            "path": signed.as_posix(),
            "bytes": signed.stat().st_size,
            "sha256": _sha256(signed),
            "certificate_sha256": match.group(1).lower(),
        }

    source_hashes = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in (
            JAVA_SOURCE,
            JNI_SOURCE,
            APP_MANIFEST,
            TEST_MANIFEST,
        )
    }
    canonical_inputs = json.dumps(
        {
            "core_library_sha256": _sha256(core_library),
            "jni_library_sha256": _sha256(jni_library),
            "package_prefix": PACKAGE_PREFIX,
            "sources": source_hashes,
            "vector_header_sha256": _sha256(vector_header),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema": "neuro-film.android-srgb-quantizer-testlab-package.v1",
        "status": "PASS",
        "package_prefix": PACKAGE_PREFIX,
        "target": {
            "abi": "arm64-v8a",
            "min_api": 24,
            "compile_api": 35,
            "test_model": "shiba",
            "test_version": "34",
            "test_model_name": "Pixel 8",
            "device_form": "PHYSICAL",
        },
        "sdk": {
            "build_tools": "35.0.0",
            "android_jar_sha256": _sha256(android_jar),
            "aapt2_sha256": _sha256(aapt2),
            "d8_sha256": _sha256(tools / "lib/d8.jar"),
            "apksigner_sha256": _sha256(
                tools / "lib/apksigner.jar"
            ),
            "zipalign_sha256": _sha256(zipalign),
        },
        "ndk": ndk_lock,
        "sources": source_hashes,
        "vector": {
            **vector_identities,
            "count": VECTOR_COUNT,
            "generated_header_sha256": _sha256(vector_header),
        },
        "core_cross_compile_report": core_report,
        "native": {
            "core_library_sha256": _sha256(core_library),
            "jni_library_sha256": _sha256(jni_library),
        },
        "manifest_validation": {
            "app_package": "com.neurofilm.srgbquantizer.target",
            "test_package": "com.neurofilm.srgbquantizer.test",
            "runner": (
                "com.neurofilm.srgbquantizer.QuantizerInstrumentation"
            ),
            "runner_target": "com.neurofilm.srgbquantizer.target",
            "aapt2_badging_and_xmltree_checked": True,
        },
        "artifacts": artifacts,
        "apk_reproducibility": (
            "actual signed APK and signing certificate are bound by hash; "
            "fresh build directories create fresh test-only keys, so APK "
            "byte reproducibility is not claimed"
        ),
        "package_identity": hashlib.sha256(canonical_inputs).hexdigest(),
        "runtime_protocol": {
            "outer_replays": 2,
            "inner_replays_per_depth": 2,
            "sample_count": VECTOR_COUNT,
            "success_check": (
                "sRGB8 and sRGB16 outputs exactly match frozen arrays"
            ),
            "failure_injection": (
                "nonfinite input and insufficient capacity both reject "
                "before output mutation"
            ),
            "claim_ceiling": (
                "one Android physical-device consumer quantizer runtime; "
                "no media, producer algorithm or product admission"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--ndk", type=Path, default=DEFAULT_NDK)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("outputs/native/android_srgb_quantizer_testlab"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/eval/android_srgb_quantizer_testlab_package_v1.json"
        ),
    )
    args = parser.parse_args()
    report = build(args.sdk, args.ndk, args.build_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
