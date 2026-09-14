import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.fable_reference_photometry import canonical_donor, dequantize8, quantize8, transform
from src.eval.tst_reference_response import descriptors, load_vgg
from src.eval.fable_paired_contrast import predict_after_raw_blocks, predict_paired_control_raw_blocks
from src.eval.fable_photometry_ridge import predict_head


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(model):
    for key in ['coefficients', 'intercept', 'mean_identity_z']:
        if key in model:
            model[key] = np.asarray(model[key], dtype=np.float64)
    for standard in model['standards']:
        for key in ['mean', 'scale']:
            standard[key] = np.asarray(standard[key], dtype=np.float64)
        standard['active'] = np.asarray(standard['active'], dtype=bool)
    return model


def main():
    started = time.perf_counter()
    cp = ROOT / 'configs/fable_paired_contrast_v1.json'
    cfg = json.loads(cp.read_text())
    root = ROOT / cfg['output']
    mp, dp = root / 'fit_head/model.json', root / 'fresh_draws.json'
    draw_record = json.loads(dp.read_text())
    if digest(cp) != draw_record['config_sha256'] or digest(mp) != draw_record['model_sha256']:
        raise ValueError('post-lock draw provenance mismatch')
    for path, expected in cfg['frozen_inputs'].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'frozen input changed: {path}')
    if digest(ROOT / cfg['source_config']) != cfg['source_config_sha256']:
        raise ValueError('new sources changed')
    if digest(ROOT / 'src/eval/fable_paired_contrast.py') != cfg['decoder_module_sha256']:
        raise ValueError('decoder changed')
    draws = np.asarray(draw_record['draws'], dtype=np.float64)
    if draws.shape != (32, 4) or not np.array_equal(draws[::2], -draws[1::2]):
        raise ValueError('32 frozen antithetic draws required')
    old_root = ROOT / 'outputs/fable_reference_photometry_v1'
    old_cfg = json.loads((ROOT / 'configs/fable_reference_photometry_v1.json').read_text())
    sources = json.loads((ROOT / cfg['source_config']).read_text())['sources']
    with np.load(old_root / 'fit_features/features.npz', allow_pickle=False) as fit:
        bases = list(fit['canonical_bases'].copy())
    for source in sources:
        path = ROOT / source['path']
        if digest(path) != source['sha256']:
            raise ValueError('source preview changed')
        with Image.open(path) as im:
            bases.append(canonical_donor(np.asarray(im.convert('RGB'), dtype=np.float32) / 255,
                                        side=cfg['descriptor']['canonical_side']))
    ids = list(range(6)) + [s['index'] for s in sources]
    if ids != [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]:
        raise ValueError('donor order changed')
    out = root / 'fresh_features'
    pred_out = root / 'fresh_predictions'
    if out.exists() or pred_out.exists():
        raise FileExistsError('fresh outputs already exist; no overwrite or redraw')
    out.mkdir()
    torch.set_num_threads(cfg['descriptor']['cpu_threads'])
    vgg = load_vgg(ROOT)
    with np.load(root / 'fit_identities/features.npz', allow_pickle=False) as identity:
        identity_stats = list(identity['stats'].copy())
        identity_deep = list(identity['deep'].copy())
    for base in bases[6:]:
        a, b = descriptors(base, vgg)
        identity_stats.append(a)
        identity_deep.append(b)
    stats, deep, refs, donors, draw_ids = [], [], [], [], []
    for donor, base in zip(ids, bases, strict=True):
        latent = dequantize8(base, seed=cfg['prospective']['dequantization_seeds'][str(donor)])
        for j, u in enumerate(draws):
            after = quantize8(transform(latent, u, slope_limit=old_cfg['slope_limit'], offset_limit=old_cfg['offset_limit']))
            a, b = descriptors(after, vgg)
            stats.append(a)
            deep.append(b)
            refs.append(np.floor(after * 255 + .5).astype(np.uint8))
            donors.append(donor)
            draw_ids.append(j)
        print(json.dumps({'donor_done': donor, 'rows': len(stats)}), flush=True)
    payload = dict(stats=np.asarray(stats), deep=np.asarray(deep), references=np.asarray(refs),
        donor_ids=np.asarray(donors), draw_ids=np.asarray(draw_ids), pair_ids=np.asarray(draw_ids) // 2,
        targets=draws[draw_ids], identities_stats=np.asarray(identity_stats), identities_deep=np.asarray(identity_deep),
        identity_donor_ids=np.asarray(ids), canonical_bases=np.asarray(bases))
    if payload['stats'].shape != (320, 30) or payload['deep'].shape != (320, 3968):
        raise ValueError('unexpected feature shape')
    np.savez(out / 'features.npz', **payload)
    receipt = dict(status='FRESH_FEATURES_FROZEN_NO_REFIT', config_sha256=digest(cp), model_sha256=digest(mp),
        fresh_draws_sha256=digest(dp), features_sha256=digest(out / 'features.npz'), entry_sha256=digest(Path(__file__)),
        after_rows=320, new_identity_rows=4, wall_seconds=time.perf_counter() - started)
    (out / 'report.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    model = load_model(json.loads(mp.read_text())['model'])
    frozen = json.loads((old_root / 'fit_head/model.json').read_text())
    if digest(ROOT / 'src/eval/fable_photometry_ridge.py') != frozen['ridge_module_sha256']:
        raise ValueError('old ridge module changed')
    frozen_model = load_model(frozen['models']['combined'])
    blocks = [payload['stats'], payload['deep']]
    identity_rows = np.repeat(np.arange(10), 32)
    before = [payload['identities_stats'][identity_rows], payload['identities_deep'][identity_rows]]
    with threadpool_limits(limits=cfg['descriptor']['cpu_threads']):
        results = dict(after_only=predict_after_raw_blocks(model, blocks),
                       paired_control=predict_paired_control_raw_blocks(model, blocks, before),
                       frozen_combined=predict_head(frozen_model, blocks))
    predictions = {key: payload[key] for key in ['donor_ids', 'draw_ids', 'pair_ids', 'targets']}
    for name, result in results.items():
        predictions[name] = result['u']
        predictions[name + '_raw'] = result['raw']
        predictions[name + '_clipped'] = result['clipped']
    pred_out.mkdir()
    np.savez(pred_out / 'predictions.npz', **predictions)
    receipt = {**receipt, 'status': 'FRESH_PREDICTIONS_FROZEN_NO_REFIT',
        'predictions_sha256': digest(pred_out / 'predictions.npz'), 'wall_seconds': time.perf_counter() - started}
    (pred_out / 'report.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
