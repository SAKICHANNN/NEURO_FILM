"""P4EL Android virtual-device runtime for the native cloud chain."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.eval.native_cloud_attenuation_android_runtime import _cleanup_owned_launchers
from src.eval.native_thomas_rgb16_png_android_runtime import (
    _android_env,
    _finish_owned_emulator_processes,
    _run,
    _wait_for_boot,
)

SOURCES = [
    "native/film_physics/nf_density_conditioned_poisson_u16_v2.c",
    "native/film_physics/nf_cloud_spatial_response_f32_v2.c",
    "native/film_physics/nf_cloud_chain_runtime_probe_v1.c",
]


def _build(root: Path, compiler: Path, target: str, output: Path, android: bool) -> None:
    command = [str(compiler), f"--target={target}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-model=strict", "-I", str(root / "native/film_physics"), *[str(root / source) for source in SOURCES]]
    if android:
        command += ["-fPIE", "-pie", "-Wl,--build-id=none", "-lm"]
    else:
        command += ["-Wl,--no-insert-timestamp"]
    command += ["-o", str(output)]
    _run(command, cwd=root)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(root: Path, contract_path: Path, ndk: Path, host_clang: Path, sdk: Path, avd_home: Path, output: Path, port: int = 5584) -> dict:
    contract=json.loads(contract_path.read_text()); parent=root/contract["parent"]["path"]; evidence=json.loads(parent.read_text())
    if _sha(parent)!=contract["parent"]["sha256"] or evidence["decision"]!=contract["parent"]["required_decision"]: raise RuntimeError("P4EL parent drift")
    output.mkdir(parents=True,exist_ok=True); host=output/"host.exe"; android=output/"android-probe"
    _build(root,host_clang,"x86_64-w64-windows-gnu",host,False); _build(root,ndk/"toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe","x86_64-linux-android34",android,True)
    host_files=[output/"host-count.u16",output/"host-density.f32",output/"host-t.f32"]
    host_stdout=_run([str(host),*map(str,host_files)],cwd=output)
    env=_android_env(sdk,avd_home); emulator=sdk/"emulator/emulator.exe"; adb=sdk/"platform-tools/adb.exe"; serial=f"emulator-{port}"; boots=[]
    for boot in range(contract["runtime"]["cold_boots"]):
        process=subprocess.Popen([str(emulator),"-avd",contract["runtime"]["avd_name"],"-port",str(port),"-no-window","-no-audio","-no-boot-anim","-no-snapshot","-wipe-data","-gpu","swiftshader_indirect"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            _wait_for_boot(adb,serial,process,env=env,timeout=180.0); remote="/data/local/tmp/nf_p4el_probe"
            _run([str(adb),"-s",serial,"push",str(android),remote],cwd=output,env=env); _run([str(adb),"-s",serial,"shell","chmod","755",remote],cwd=output,env=env)
            rows=[]
            for run in range(contract["runtime"]["fresh_processes_per_boot"]):
                remote_files=[f"/data/local/tmp/p4el-{boot}-{run}-{suffix}" for suffix in ("c.u16","d.f32","t.f32")]
                stdout=_run([str(adb),"-s",serial,"shell",remote,*remote_files],cwd=output,env=env); local=[]
                for remote_file in remote_files:
                    path=output/Path(remote_file).name; _run([str(adb),"-s",serial,"pull",remote_file,str(path)],cwd=output,env=env); local.append(path)
                rows.append({"stdout":stdout,"paths":local})
            boots.append({"abi":_run([str(adb),"-s",serial,"shell","getprop","ro.product.cpu.abi"],cwd=output,env=env),"runs":rows})
        finally:
            try: _run([str(adb),"-s",serial,"emu","kill"],cwd=output,env=env,timeout=15)
            except Exception: process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            _finish_owned_emulator_processes(emulator,contract["runtime"]["avd_name"],port)
            if not _cleanup_owned_launchers(emulator,contract["runtime"]["avd_name"],port): raise RuntimeError("owned emulator survived cleanup")
    rows=[row for boot in boots for row in boot["runs"]]; host_count=np.fromfile(host_files[0],dtype='<u2'); host_d=np.fromfile(host_files[1],dtype='<f4'); host_t=np.fromfile(host_files[2],dtype='<f4')
    counts=[np.fromfile(row["paths"][0],dtype='<u2') for row in rows]; densities=[np.fromfile(row["paths"][1],dtype='<f4') for row in rows]; trans=[np.fromfile(row["paths"][2],dtype='<f4') for row in rows]
    max_d=max(float(np.max(np.abs(value-host_d))) for value in densities); max_t=max(float(np.max(np.abs(value-host_t))) for value in trans); gates_cfg=contract["gates"]
    gates={"count":all(np.array_equal(value,host_count) for value in counts),"density":max_d<=gates_cfg["maximum_host_android_density_absolute_error"],"transmittance":max_t<=gates_cfg["maximum_host_android_transmittance_absolute_error"],"repeat":all(np.array_equal(value,counts[0]) for value in counts) and all(np.array_equal(value,densities[0]) for value in densities) and all(np.array_equal(value,trans[0]) for value in trans),"atomic":all("invalid=2" in row["stdout"] for row in rows),"abi":all(boot["abi"]=="x86_64" for boot in boots),"cleanup":_cleanup_owned_launchers(emulator,contract["runtime"]["avd_name"],port)}
    stable={"contract_sha256":_sha(contract_path),"host_count_sha256":_sha(host_files[0]),"host_density_sha256":_sha(host_files[1]),"host_transmittance_sha256":_sha(host_files[2]),"maximum_host_android_density_absolute_error":max_d,"maximum_host_android_transmittance_absolute_error":max_t,"gates":gates,"decision":contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],"claim_ceiling":contract["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4el_native_cloud_chain_android_runtime.v1","automatic_pass":all(gates.values()),"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest(),"host_stdout":host_stdout}


__all__=["evaluate"]
