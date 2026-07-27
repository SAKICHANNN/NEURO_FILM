"""Exact-wheel local invocation adapter for the pinned D-PCT SDR package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping
import zipfile

from .contracts import ReferenceMatchContractError
from .core_adapter import PreparedMatchViewV1, validate_prepared_match_view
from .dpct_adapter import (
    DPCT_CANONICAL_JSON,
    DPCT_MATCH_VIEW_SCHEMA,
    DPCT_PIXEL_LAYOUT,
    DPCT_PRODUCER_PROFILE_ID,
    AdaptedDpctCandidateV2,
    DpctProducerFailureV2,
    adapt_dpct_candidate_v2,
    dpct_producer_view_id_for_prepared_v2,
    verify_dpct_failed_diagnostics_v2,
)


DPCT_INVOCATION_COMPATIBILITY_PROFILE_ID = (
    "neuro-film.dpct-invocation-consumer.v1"
)
DPCT_INVOCATION_STABLE_COMMIT = "e22725d"
DPCT_INVOCATION_SOURCE_COMMIT = (
    "01ef0616610b16246623236301234ff3b4c4a7f2"
)
DPCT_INVOCATION_WHEEL_SIZE = 95994
DPCT_INVOCATION_WHEEL_SHA256 = (
    "fd995ad88c9f30f2136508f7c3879ce6768fde7ff2e149537d2b58f78e36c292"
)
DPCT_INVOCATION_CAPABILITY_ID = "zhuise.dpct-chroma.cpu-reference.v1"
DPCT_INVOCATION_REQUEST_SCHEMA = "zhuise.invocation-request.v1"
DPCT_INVOCATION_RESPONSE_SCHEMA = "zhuise.invocation-response.v1"
DPCT_INVOCATION_CLAIM_CEILING = (
    "candidate-only-not-promoted-not-applied-not-delivered"
)
_EXPECTED_FILES_CANDIDATE = {
    "output.f32be",
    "transform.payload",
    "response.json",
}
_EXPECTED_FILES_FAILED = {"response.json"}


@dataclass(frozen=True)
class DpctInvocationOutcomeV1:
    compatibility_profile_id: str
    producer_stable_commit: str
    producer_source_commit: str
    wheel_sha256: str
    request_id: str
    response_id: str
    status: str
    claim_ceiling: str
    candidate: AdaptedDpctCandidateV2 | None
    failure: DpctProducerFailureV2 | None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation JSON is not canonicalizable"
        ) from exc


def _domain_id(domain: bytes, payload: Mapping[str, Any]) -> str:
    return "sha256:" + _sha256(domain + _canonical_json(payload))


def _prepared_wire(
    prepared: PreparedMatchViewV1,
) -> tuple[dict[str, Any], bytes]:
    validate_prepared_match_view(prepared)
    height, width, channels = prepared.descriptor.shape
    if channels != 3:
        raise ReferenceMatchContractError(
            "D-PCT invocation requires three-channel pixels"
        )
    pixels = prepared.pixels.astype(">f4", copy=False).tobytes(order="C")
    view = {
        "schema": DPCT_MATCH_VIEW_SCHEMA,
        "profile_id": DPCT_PRODUCER_PROFILE_ID,
        "width": width,
        "height": height,
        "channels": 3,
        "pixel_layout": DPCT_PIXEL_LAYOUT,
        "pixel_byte_length": len(pixels),
        "pixel_sha256": "sha256:" + _sha256(pixels),
        "view_id": dpct_producer_view_id_for_prepared_v2(prepared),
    }
    return view, pixels


def prepare_dpct_invocation_request_v1(
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
) -> tuple[dict[str, Any], bytes, bytes]:
    source_view, source_pixels = _prepared_wire(source)
    reference_view, reference_pixels = _prepared_wire(reference)
    request: dict[str, Any] = {
        "schema": DPCT_INVOCATION_REQUEST_SCHEMA,
        "canonical_json": DPCT_CANONICAL_JSON,
        "operation": "fit-apply",
        "capability_id": DPCT_INVOCATION_CAPABILITY_ID,
        "source": {
            "view": source_view,
            "pixels_file": "source.f32be",
        },
        "reference": {
            "view": reference_view,
            "pixels_file": "reference.f32be",
        },
    }
    request["request_id"] = _domain_id(
        b"ZhuiseInvocationRequestV1\0", request
    )
    return request, source_pixels, reference_pixels


def _strict_object(
    value: Any,
    keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the pinned invocation contract"
        )
    return value


def _load_response(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation response is unavailable"
        ) from exc
    if len(raw) > 1024 * 1024 or not raw.endswith(b"\n"):
        raise ReferenceMatchContractError(
            "D-PCT invocation response size/terminator is invalid"
        )
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation response is not strict UTF-8 JSON"
        ) from exc
    if raw != _canonical_json(response) + b"\n":
        raise ReferenceMatchContractError(
            "D-PCT invocation response is not canonical JSON"
        )
    return response, raw


def verify_dpct_invocation_output_v1(
    *,
    output_directory: Path,
    request: Mapping[str, Any],
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    intent_id: str,
) -> DpctInvocationOutcomeV1:
    output_directory = output_directory.resolve()
    response, _ = _load_response(output_directory / "response.json")
    response = _strict_object(
        response,
        {
            "schema",
            "canonical_json",
            "producer_package",
            "request_id",
            "status",
            "source",
            "reference",
            "transform",
            "diagnostics",
            "apply_result",
            "artifacts",
            "failure",
            "response_id",
        },
        "D-PCT invocation response",
    )
    if (
        response["schema"] != DPCT_INVOCATION_RESPONSE_SCHEMA
        or response["canonical_json"] != DPCT_CANONICAL_JSON
        or response["request_id"] != request["request_id"]
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation response/request binding mismatch"
        )
    package = _strict_object(
        response["producer_package"],
        {"distribution", "version", "entrypoint"},
        "D-PCT producer package",
    )
    if dict(package) != {
        "distribution": "zhuise-research",
        "version": "0.2.0",
        "entrypoint": "zhuise-producer-invoke",
    }:
        raise ReferenceMatchContractError(
            "D-PCT invocation package identity mismatch"
        )
    identity = dict(response)
    response_id = identity.pop("response_id")
    if response_id != _domain_id(
        b"ZhuiseInvocationResponseV1\0", identity
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation response ID mismatch"
        )
    if (
        response["source"] != request["source"]["view"]
        or response["reference"] != request["reference"]["view"]
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation returned foreign input views"
        )
    actual_files = {
        item.name for item in output_directory.iterdir() if item.is_file()
    }
    status = response["status"]
    candidate = None
    failure = None
    if status == "candidate":
        if actual_files != _EXPECTED_FILES_CANDIDATE:
            raise ReferenceMatchContractError(
                "D-PCT candidate artifact set mismatch"
            )
        artifacts = _strict_object(
            response["artifacts"],
            {
                "output_pixels_file",
                "output_pixel_sha256",
                "bundle_payload_file",
                "bundle_payload_sha256",
            },
            "D-PCT candidate artifacts",
        )
        if (
            artifacts["output_pixels_file"] != "output.f32be"
            or artifacts["bundle_payload_file"] != "transform.payload"
            or response["failure"] is not None
        ):
            raise ReferenceMatchContractError(
                "D-PCT candidate artifact names/failure mismatch"
            )
        output_path = output_directory / "output.f32be"
        payload_path = output_directory / "transform.payload"
        expected_output_size = (
            source.descriptor.shape[0]
            * source.descriptor.shape[1]
            * 12
        )
        if (
            output_path.stat().st_size != expected_output_size
            or payload_path.stat().st_size > 16 * 1024 * 1024
        ):
            raise ReferenceMatchContractError(
                "D-PCT invocation artifact size is invalid"
            )
        output = output_path.read_bytes()
        payload = payload_path.read_bytes()
        if (
            artifacts["output_pixel_sha256"]
            != "sha256:" + _sha256(output)
            or artifacts["bundle_payload_sha256"]
            != "sha256:" + _sha256(payload)
        ):
            raise ReferenceMatchContractError(
                "D-PCT invocation artifact hash mismatch"
            )
        candidate = adapt_dpct_candidate_v2(
            source=source,
            reference=reference,
            producer_source=response["source"],
            producer_reference=response["reference"],
            producer_transform=response["transform"],
            producer_transform_payload=payload,
            producer_diagnostics=response["diagnostics"],
            producer_apply_result=response["apply_result"],
            output_pixel_f32be=output,
            intent_id=intent_id,
        )
        if (
            candidate.aliases.capability_id
            != DPCT_INVOCATION_CAPABILITY_ID
        ):
            raise ReferenceMatchContractError(
                "D-PCT invocation capability is not pinned"
            )
    elif status == "failed":
        if actual_files != _EXPECTED_FILES_FAILED:
            raise ReferenceMatchContractError(
                "failed D-PCT invocation emitted candidate artifacts"
            )
        if any(
            response[key] is not None
            for key in ("transform", "apply_result", "artifacts")
        ):
            raise ReferenceMatchContractError(
                "failed D-PCT invocation carries candidate state"
            )
        failure = verify_dpct_failed_diagnostics_v2(
            response["diagnostics"]
        )
        failure_envelope = _strict_object(
            response["failure"],
            {"code", "message"},
            "D-PCT invocation failure",
        )
        if failure_envelope["code"] != failure.failure_code:
            raise ReferenceMatchContractError(
                "D-PCT failure envelope/diagnostics mismatch"
            )
    else:
        raise ReferenceMatchContractError(
            "D-PCT invocation status is unsupported"
        )
    return DpctInvocationOutcomeV1(
        compatibility_profile_id=(
            DPCT_INVOCATION_COMPATIBILITY_PROFILE_ID
        ),
        producer_stable_commit=DPCT_INVOCATION_STABLE_COMMIT,
        producer_source_commit=DPCT_INVOCATION_SOURCE_COMMIT,
        wheel_sha256=DPCT_INVOCATION_WHEEL_SHA256,
        request_id=str(request["request_id"]),
        response_id=str(response_id),
        status=str(status),
        claim_ceiling=DPCT_INVOCATION_CLAIM_CEILING,
        candidate=candidate,
        failure=failure,
    )


def _verify_runtime(python_executable: Path) -> None:
    code = (
        "import json,sys,numpy,PIL;"
        "print(json.dumps({'python':f'{sys.version_info.major}."
        "{sys.version_info.minor}','numpy':numpy.__version__,"
        "'pillow':PIL.__version__},sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [python_executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation runtime preflight failed closed"
        ) from exc
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation runtime identity is invalid"
        ) from exc
    if identity != {
        "python": "3.12",
        "numpy": "2.4.4",
        "pillow": "12.1.1",
    }:
        raise ReferenceMatchContractError(
            "D-PCT invocation runtime identity mismatch"
        )


def invoke_dpct_package_v1(
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    intent_id: str,
    wheel_path: Path,
    python_executable: Path,
    scratch_directory: Path,
    timeout_seconds: float = 300.0,
) -> DpctInvocationOutcomeV1:
    wheel_path = wheel_path.resolve()
    python_executable = python_executable.resolve()
    scratch_directory = scratch_directory.resolve()
    if (
        not wheel_path.is_file()
        or wheel_path.name
        != "zhuise_research-0.2.0-py3-none-any.whl"
        or wheel_path.stat().st_size != DPCT_INVOCATION_WHEEL_SIZE
        or _sha256(wheel_path.read_bytes())
        != DPCT_INVOCATION_WHEEL_SHA256
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation wheel identity mismatch"
        )
    if not python_executable.is_file():
        raise ReferenceMatchContractError(
            "D-PCT invocation Python executable is unavailable"
        )
    if not scratch_directory.is_dir():
        raise ReferenceMatchContractError(
            "D-PCT invocation scratch directory is unavailable"
        )
    _verify_runtime(python_executable)
    request, source_pixels, reference_pixels = (
        prepare_dpct_invocation_request_v1(
            source=source,
            reference=reference,
        )
    )
    with tempfile.TemporaryDirectory(
        prefix="nfcm-dpct-invocation-",
        dir=scratch_directory,
    ) as temporary:
        root = Path(temporary)
        package_directory = root / "package"
        request_directory = root / "request"
        output_directory = root / "output"
        package_directory.mkdir()
        with zipfile.ZipFile(wheel_path) as archive:
            members = archive.infolist()
            for member in members:
                member_path = Path(member.filename)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member.filename.endswith("\\")
                ):
                    raise ReferenceMatchContractError(
                        "D-PCT wheel contains an unsafe archive path"
                    )
            archive.extractall(package_directory)
        request_directory.mkdir()
        (request_directory / "source.f32be").write_bytes(source_pixels)
        (request_directory / "reference.f32be").write_bytes(
            reference_pixels
        )
        (request_directory / "request.json").write_bytes(
            _canonical_json(request) + b"\n"
        )
        code = (
            "import importlib,os,sys;"
            "package=os.path.normcase(os.path.abspath(sys.argv.pop(1)));"
            "sys.path.insert(0,package);"
            "module=importlib.import_module('zhuise.producer_invocation');"
            "origin=os.path.normcase(os.path.abspath(module.__file__));"
            "assert origin.startswith(package + os.sep),origin;"
            "module.main()"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        try:
            completed = subprocess.run(
                [
                    python_executable,
                    "-c",
                    code,
                    package_directory,
                    "--request",
                    request_directory / "request.json",
                    "--output-directory",
                    output_directory,
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReferenceMatchContractError(
                "D-PCT exact-wheel invocation process failed closed"
            ) from exc
        if completed.returncode != 0:
            raise ReferenceMatchContractError(
                "D-PCT exact-wheel invocation failed closed: "
                + completed.stderr.strip()[:512]
            )
        return verify_dpct_invocation_output_v1(
            output_directory=output_directory,
            request=request,
            source=source,
            reference=reference,
            intent_id=intent_id,
        )


__all__ = [
    "DPCT_INVOCATION_CLAIM_CEILING",
    "DPCT_INVOCATION_COMPATIBILITY_PROFILE_ID",
    "DPCT_INVOCATION_SOURCE_COMMIT",
    "DPCT_INVOCATION_STABLE_COMMIT",
    "DPCT_INVOCATION_WHEEL_SHA256",
    "DpctInvocationOutcomeV1",
    "invoke_dpct_package_v1",
    "prepare_dpct_invocation_request_v1",
    "verify_dpct_invocation_output_v1",
]
