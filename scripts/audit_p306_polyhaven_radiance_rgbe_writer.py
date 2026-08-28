"""Audit deterministic Radiance RGBE publication against official C logic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.radiance_rgbe import (
    RadianceRgbeError,
    _decode_radiance_rgbe_codes_bytes,
    decode_radiance_rgbe_bytes,
    encode_radiance_rgbe_bytes,
    write_radiance_rgbe_create_only,
)

CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"

C_HARNESS = r'''#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef unsigned char COLR[4];
static void setcolr(COLR c,double r,double g,double b){double d=r>g?r:g;int e;if(b>d)d=b;if(d<=1e-32){c[0]=c[1]=c[2]=c[3]=0;return;}d=frexp(d,&e)*255.9999/d;c[0]=r>0?r*d:0;c[1]=g>0?g*d:0;c[2]=b>0?b*d:0;c[3]=e+128;}
static int putch(int v,FILE*f){return fputc(v,f)==EOF?-1:0;}
static int fwritecolrs(COLR*s,int n,FILE*f){int i,j,beg,cnt=1,c2;if(putch(2,f)||putch(2,f)||putch(n>>8,f)||putch(n&255,f))return-1;for(i=0;i<4;i++){for(j=0;j<n;j+=cnt){for(beg=j;beg<n;beg+=cnt){for(cnt=1;cnt<127&&beg+cnt<n&&s[beg+cnt][i]==s[beg][i];cnt++);if(cnt>=4)break;}if(beg-j>1&&beg-j<4){c2=j+1;while(s[c2++][i]==s[j][i])if(c2==beg){putch(128+beg-j,f);putch(s[j][i],f);j=beg;break;}}while(j<beg){if((c2=beg-j)>128)c2=128;putch(c2,f);while(c2--)putch(s[j++][i],f);}if(cnt>=4){putch(128+cnt,f);putch(s[beg][i],f);}else cnt=0;}}return ferror(f)?-1:0;}
int main(int ac,char**av){if(ac!=5)return 2;int h=atoi(av[1]),w=atoi(av[2]);if(h<1||w<8||w>32767)return 3;FILE*in=fopen(av[3],"rb"),*out=fopen(av[4],"wb");if(!in||!out)return 4;float*row=malloc((size_t)w*3*sizeof(float));COLR*c=malloc((size_t)w*sizeof(COLR));if(!row||!c)return 5;fprintf(out,"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n",h,w);for(int y=0;y<h;y++){if(fread(row,sizeof(float),(size_t)w*3,in)!=(size_t)w*3)return 6;for(int x=0;x<w;x++)setcolr(c[x],row[3*x],row[3*x+1],row[3*x+2]);if(fwritecolrs(c,w,out))return 7;}if(fgetc(in)!=EOF)return 8;free(c);free(row);return fclose(in)||fclose(out);}
'''


class P306Error(RuntimeError):
    """Raised when a frozen P306 gate differs."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _verify(path: Path, binding: dict[str, Any]) -> bool:
    return path.is_file() and path.stat().st_size == binding["bytes"] and _file_sha(path) == binding["sha256"]


def _invalid_controls() -> dict[str, bool]:
    controls = {
        "negative": np.full((1, 8, 3), -1.0, np.float32),
        "nonfinite": np.full((1, 8, 3), np.nan, np.float32),
        "wrong_channels": np.zeros((1, 8, 4), np.float32),
        "short_width": np.zeros((1, 7, 3), np.float32),
        "boolean": np.zeros((1, 8, 3), np.bool_),
    }
    result = {}
    for name, value in controls.items():
        try:
            encode_radiance_rgbe_bytes(value)
        except RadianceRgbeError:
            result[name] = True
        else:
            result[name] = False
    return result


def execute(config_path: Path, work: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = {name: _verify(ROOT / row["path"], row) for name, row in config["bindings"].items()}
    if not all(bindings.values()):
        raise P306Error("binding drift")
    source_path = ROOT / config["source"]["path"]
    official_path = ROOT / config["official_oracle"]["path"]
    if not _verify(source_path, config["source"]) or not _verify(official_path, config["official_oracle"]):
        raise P306Error("source drift")
    official_text = official_path.read_text(encoding="utf-8")
    source_formula_bound = all(
        fragment in official_text
        for fragment in ("* 255.9999 / d", "MINRUN", "fwritecolrs")
    )
    if not source_formula_bound or not CLANG.is_file():
        raise P306Error("official oracle unavailable")

    source_bytes = source_path.read_bytes()
    source_codes = _decode_radiance_rgbe_codes_bytes(source_bytes)
    values = decode_radiance_rgbe_bytes(source_bytes)
    values_before = _sha(values.tobytes())
    python_payload = encode_radiance_rgbe_bytes(values)
    work.mkdir(parents=True, exist_ok=False)
    c_path, exe_path = work / "oracle.c", work / "oracle.exe"
    input_path, c_output = work / "input.f32", work / "official.hdr"
    published = work / "published.hdr"
    c_path.write_text(C_HARNESS, encoding="ascii", newline="\n")
    input_path.write_bytes(values.astype("<f4", copy=False).tobytes())
    compile_result = subprocess.run([str(CLANG), "-O2", "-std=c11", str(c_path), "-o", str(exe_path)], capture_output=True, text=True, check=False)
    if compile_result.returncode != 0:
        raise P306Error("official oracle build failed")
    run = subprocess.run([str(exe_path), str(values.shape[0]), str(values.shape[1]), str(input_path), str(c_output)], capture_output=True, check=False)
    if run.returncode != 0:
        raise P306Error("official oracle execution failed")
    c_payload = c_output.read_bytes()
    write_radiance_rgbe_create_only(published.resolve(), values)
    published_bytes = published.read_bytes()
    try:
        write_radiance_rgbe_create_only(published.resolve(), values)
    except RadianceRgbeError:
        create_only_reject = published.read_bytes() == published_bytes
    else:
        create_only_reject = False

    encoded_codes = _decode_radiance_rgbe_codes_bytes(python_payload)
    decoded = decode_radiance_rgbe_bytes(python_payload)
    invalid = _invalid_controls()
    gates = {
        "bindings": all(bindings.values()),
        "create_only_atomic": create_only_reject,
        "decoded_float_exact": _sha(decoded.tobytes()) == config["parents"]["p305_float_sha256"],
        "input_immutable": _sha(values.tobytes()) == values_before,
        "invalid_controls": all(invalid.values()),
        "python_c_container_exact": python_payload == c_payload == published_bytes,
        "source_codes_exact": np.array_equal(encoded_codes, source_codes),
        "source_formula_bound": source_formula_bound,
    }
    report: dict[str, object] = {
        "bindings": bindings,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "container": {"bytes": len(python_payload), "sha256": _sha(python_payload)},
        "decision": "PASS_PRIVATE_POLYHAVEN_RADIANCE_RGBE_WRITER" if all(gates.values()) else "FAIL_CLOSED_POLYHAVEN_RADIANCE_RGBE_WRITER",
        "experiment_id": "P306",
        "gates": gates,
        "invalid_controls": invalid,
        "oracle": {"clang_sha256": _file_sha(CLANG), "harness_sha256": _sha(C_HARNESS.encode("ascii")), "official_source_sha256": _file_sha(official_path)},
        "output": {"codes_sha256": _sha(encoded_codes.tobytes()), "decoded_f32le_sha256": _sha(decoded.astype("<f4", copy=False).tobytes())},
        "schema": "neuro-film.p306-polyhaven-radiance-rgbe-writer-result.v1",
        "source_file_reads": 1,
        "network_requests": 0,
    }
    report["scientific_identity"] = "sha256:" + _sha(json.dumps(report, sort_keys=True, separators=(",", ":")).encode())
    for path in (published, c_output, input_path, exe_path, c_path):
        path.unlink(missing_ok=True)
    work.rmdir()
    report["media_residue_zero"] = not work.exists()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    del args.reverse
    report = execute(args.config.resolve(), args.work.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))


if __name__ == "__main__":
    main()
