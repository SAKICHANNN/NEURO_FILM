"""Build the independent P28-P30 C++17 conformance verifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


VSWHERE = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
)


def _developer_environment() -> dict[str, str]:
    if not VSWHERE.is_file():
        raise FileNotFoundError("vswhere.exe is unavailable")
    installation = subprocess.run(
        [
            VSWHERE,
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not installation:
        raise FileNotFoundError("Visual Studio C++ tools are unavailable")
    vsdevcmd = (
        Path(installation)
        / "Common7"
        / "Tools"
        / "VsDevCmd.bat"
    )
    if not vsdevcmd.is_file() or '"' in str(vsdevcmd):
        raise FileNotFoundError("VsDevCmd.bat is unavailable")
    command = (
        f'call "{vsdevcmd}" -no_logo -arch=x64 -host_arch=x64 '
        ">nul && set"
    )
    completed = subprocess.run(
        command,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    environment = os.environ.copy()
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        environment[name] = value
    return environment


def build(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file() or source.suffix.casefold() != ".cpp":
        raise ValueError("source must be an existing .cpp file")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = _developer_environment()
    compiler = shutil.which("cl.exe", path=environment.get("Path"))
    if compiler is None:
        raise FileNotFoundError("cl.exe is unavailable")
    subprocess.run(
        [
            compiler,
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/W4",
            "/WX",
            str(source),
            f"/Fe:{output}",
            f"/Fo:{output.with_suffix('.obj')}",
        ],
        env=environment,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.source, arguments.output)
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
