"""P4EP Android virtual-device runtime for the single-call cloud row chain."""

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

SOURCES=["native/film_physics/nf_conditioned_cloud_row_chain_f32_v1.c","native/film_physics/nf_density_conditioned_poisson_u16_v3.c","native/film_physics/nf_cloud_spatial_response_f32_v2.c","native/film_physics/nf_cloud_attenuation_f32_v1.c","native/film_physics/nf_deterministic_log10_f32_v1.c","native/film_physics/nf_cloud_row_chain_runtime_probe_v1.c"]


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(root: Path, compiler: Path, target: str, output: Path, android: bool) -> None:
    command=[str(compiler),f"--target={target}","-std=c11","-O2","-Wall","-Wextra","-Werror","-ffp-model=strict","-I",str(root/"native/film_physics"),*[str(root/source) for source in SOURCES]]
    command += ["-fPIE","-pie","-Wl,--build-id=none","-lm"] if android else ["-Wl,--no-insert-timestamp"]
    command += ["-o",str(output)]; _run(command,cwd=root)


def evaluate(root: Path, contract_path: Path, ndk: Path, host_clang: Path, sdk: Path, avd_home: Path, output: Path, port: int=5586) -> dict:
    c=json.loads(contract_path.read_text()); parent=root/c["parent"]["path"]
    if _sha(parent)!=c["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=c["parent"]["required_decision"]: raise RuntimeError("P4EP parent drift")
    output.mkdir(parents=True,exist_ok=True); host=output/"host.exe"; android=output/"android-probe"
    _build(root,host_clang,"x86_64-w64-windows-gnu",host,False); _build(root,ndk/"toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe","x86_64-linux-android34",android,True)
    host_paths=[output/"host-density.f32",output/"host-t.f32"]; host_stdout=_run([str(host),*map(str,host_paths)],cwd=output)
    env=_android_env(sdk,avd_home); emulator=sdk/"emulator/emulator.exe"; adb=sdk/"platform-tools/adb.exe"; serial=f"emulator-{port}"; boots=[]
    for boot in range(c["runtime"]["cold_boots"]):
        process=subprocess.Popen([str(emulator),"-avd",c["runtime"]["avd_name"],"-port",str(port),"-no-window","-no-audio","-no-boot-anim","-no-snapshot","-wipe-data","-gpu","swiftshader_indirect"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            _wait_for_boot(adb,serial,process,env=env,timeout=180.); remote="/data/local/tmp/nf_p4ep_probe"; _run([str(adb),"-s",serial,"push",str(android),remote],cwd=output,env=env);_run([str(adb),"-s",serial,"shell","chmod","755",remote],cwd=output,env=env); runs=[]
            for run in range(c["runtime"]["fresh_processes_per_boot"]):
                remote_paths=[f"/data/local/tmp/p4ep-{boot}-{run}-{suffix}" for suffix in ("d.f32","t.f32")]; stdout=_run([str(adb),"-s",serial,"shell",remote,*remote_paths],cwd=output,env=env); paths=[]
                for rp in remote_paths:
                    path=output/Path(rp).name;_run([str(adb),"-s",serial,"pull",rp,str(path)],cwd=output,env=env);paths.append(path)
                runs.append({"stdout":stdout,"paths":paths})
            boots.append({"abi":_run([str(adb),"-s",serial,"shell","getprop","ro.product.cpu.abi"],cwd=output,env=env),"runs":runs})
        finally:
            try:_run([str(adb),"-s",serial,"emu","kill"],cwd=output,env=env,timeout=15)
            except (RuntimeError,subprocess.SubprocessError):process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            _finish_owned_emulator_processes(emulator,c["runtime"]["avd_name"],port)
            if not _cleanup_owned_launchers(emulator,c["runtime"]["avd_name"],port):raise RuntimeError("owned emulator survived cleanup")
    rows=[r for b in boots for r in b["runs"]]; hd=np.fromfile(host_paths[0],dtype='<f4'); ht=np.fromfile(host_paths[1],dtype='<f4'); ds=[np.fromfile(r["paths"][0],dtype='<f4') for r in rows]; ts=[np.fromfile(r["paths"][1],dtype='<f4') for r in rows]
    max_density=max(float(np.max(np.abs(x.astype(np.float64)-hd.astype(np.float64)))) for x in ds)
    density_differing=max(int(np.count_nonzero(x!=hd)) for x in ds)
    gates={"density":all(np.array_equal(x,hd) for x in ds),"transmittance":all(np.array_equal(x,ht) for x in ts),"repeat":all(np.array_equal(x,ds[0]) for x in ds) and all(np.array_equal(x,ts[0]) for x in ts),"workspace":all("counts=6909 convolution=2303 core=4371" in r["stdout"] for r in rows),"atomic":all("invalid=2" in r["stdout"] for r in rows),"abi":all(b["abi"]=="x86_64" for b in boots),"cleanup":_cleanup_owned_launchers(emulator,c["runtime"]["avd_name"],port)}
    stable={"contract_sha256":_sha(contract_path),"host_density_sha256":_sha(host_paths[0]),"host_transmittance_sha256":_sha(host_paths[1]),"maximum_density_absolute_error":max_density,"density_differing_values":density_differing,"gates":gates,"decision":c["decision_if_pass"] if all(gates.values()) else c["decision_if_fail"],"claim_ceiling":c["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4ep_native_cloud_row_chain_android_runtime.v1","automatic_pass":all(gates.values()),"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"host_stdout":host_stdout}


__all__=["evaluate"]
