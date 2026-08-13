"""Run CHAM9 publisher-parameter SpektraFilm chart baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _worker(config_path: Path, output: Path) -> None:
    contract = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = ROOT / contract["external_runtime"]["source_root"] / "src"
    sys.path.insert(0, str(source_root))
    from spektrafilm import digest_params, init_params, simulate
    from spektrafilm.utils.raw_file_processor import load_and_process_raw_file

    publisher = json.loads(
        (ROOT / contract["parents"]["publisher_parameters"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    simulation = publisher["simulation"]
    params = init_params(
        film_profile=simulation["film_stock"],
        print_profile="fujifilm_crystal_archive_typeii",
    )
    input_image = publisher["input_image"]
    params.io.input_color_space = input_image["input_color_space"]
    params.io.input_cctf_decoding = bool(input_image["apply_cctf_decoding"])
    params.io.output_color_space = simulation["output_color_space"]
    params.io.output_cctf_encoding = True
    params.camera.filter_uv = tuple(input_image["filter_uv"])
    params.camera.filter_ir = tuple(input_image["filter_ir"])
    params.camera.film_format_mm = float(simulation["film_format_mm"])
    params.camera.exposure_compensation_ev = float(
        simulation["exposure_compensation_ev"]
    )
    params.camera.auto_exposure = bool(simulation["auto_exposure"])
    params.camera.auto_exposure_method = simulation["auto_exposure_method"]
    params.enlarger.illuminant = simulation["print_illuminant"]
    params.enlarger.print_exposure = float(simulation["print_exposure"])
    params.enlarger.print_exposure_compensation = bool(
        simulation["print_exposure_compensation"]
    )
    params.enlarger.y_filter_shift = float(simulation["print_y_filter_shift"])
    params.enlarger.m_filter_shift = float(simulation["print_m_filter_shift"])
    preflash = publisher["preflashing"]
    params.enlarger.preflash_exposure = float(preflash["exposure"])
    params.enlarger.preflash_y_filter_shift = float(preflash["y_filter_shift"])
    params.enlarger.preflash_m_filter_shift = float(preflash["m_filter_shift"])
    couplers = publisher["couplers"]
    params.film_render.dir_couplers.active = bool(couplers["active"])
    params.film_render.dir_couplers.amount = float(couplers["amount"])
    params.film_render.dir_couplers.inhibition_samelayer = float(
        couplers["inhibition_samelayer"]
    )
    params.film_render.dir_couplers.inhibition_interlayer = float(
        couplers["inhibition_interlayer"]
    )
    params.film_render.dir_couplers.gamma_samelayer_rgb = tuple(
        couplers["gamma_samelayer_rgb"]
    )
    params.film_render.dir_couplers.gamma_interlayer_r_to_gb = tuple(
        couplers["gamma_interlayer_r_to_gb"]
    )
    params.film_render.dir_couplers.gamma_interlayer_g_to_rb = tuple(
        couplers["gamma_interlayer_g_to_rb"]
    )
    params.film_render.dir_couplers.gamma_interlayer_b_to_rg = tuple(
        couplers["gamma_interlayer_b_to_rg"]
    )
    params.film_render.dir_couplers.diffusion_size_um = 0.0
    params.film_render.density_curve_gamma = float(
        publisher["special"]["film_gamma_factor"]
    )
    params.film_render.grain.active = False
    params.film_render.halation.active = False
    params.print_render.glare.active = False
    params.debug.deactivate_spatial_effects = True
    params.debug.deactivate_stochastic_effects = True
    params.settings.use_enlarger_lut = False
    params.settings.use_scanner_lut = False
    raw = ROOT / "data/quarantine/spektrafilm_portra400_same_scene_v1/Digital Lumix S5ii Color Chart.RW2"
    image = load_and_process_raw_file(
        raw,
        white_balance=publisher["load_raw"]["white_balance"],
        temperature=float(publisher["load_raw"]["temperature"]),
        tint=float(publisher["load_raw"]["tint"]),
        lens_correction=bool(publisher["load_raw"]["lens_correction"]),
        output_colorspace=input_image["input_color_space"],
        output_cctf_encoding=bool(input_image["apply_cctf_decoding"]),
    )
    if image.shape[0] % 2 or image.shape[1] % 2:
        raise RuntimeError("CHAM9 RAW geometry must support exact 2x2 area reduction")
    image = image.reshape(
        image.shape[0] // 2, 2, image.shape[1] // 2, 2, 3
    ).mean(axis=(1, 3))
    result = np.asarray(
        simulate(image, digest_params(params), digest_params_first=False),
        dtype=np.float64,
    )
    if result.shape != image.shape or not np.isfinite(result).all():
        raise RuntimeError("CHAM9 external render invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result, allow_pickle=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2cham9_spektrafilm_author_baseline_d1_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        _worker(args.config, args.output)
        return 0
    from src.eval.portra400_spektrafilm_author_baseline_d1 import (
        evaluate,
        load_contract,
        write_report,
    )

    contract = load_contract(args.config)
    render = args.output.with_suffix(".render.npy")
    report = evaluate(contract, ROOT, render)
    digest = write_report(report, args.output)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": digest,
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
