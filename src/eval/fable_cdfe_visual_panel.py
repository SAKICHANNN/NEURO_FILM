import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.fable_reference_photometry import transform
from src.preprocess.fable_canonical_raw import linear16_to_q8


def render_codes(codes: np.ndarray, u: np.ndarray) -> np.ndarray:
    if codes.dtype != np.uint8 or codes.ndim != 3 or codes.shape[-1] != 3:
        raise ValueError('native uint8 RGB required')
    ramp = np.repeat((np.arange(256)/255)[:, None], 3, axis=1)
    table = np.floor(255*transform(ramp, u, slope_limit=1.25, offset_limit=.35)+.5).astype(np.uint8)
    return np.stack([table[codes[..., c], c] for c in range(3)], axis=-1)


def render_comfort_panel(directory: Path, plan: dict, report: dict, arrays: dict) -> dict:
    if report['gates']['numeric_passed'] is not True:
        raise ValueError('numeric gate must pass before comfort opening')
    cases = plan['comfort']['cases']
    if len(cases) != 64 or len({(c['donor'], c['treatment'], c['query']) for c in cases}) != 64:
        raise ValueError('fixed64 unique comfort cases required')
    directory.mkdir(parents=True, exist_ok=False)
    donor_index = {identity: j for j, identity in enumerate(report['identities'])}
    treatment_index = {identity: j for j, identity in enumerate(report['treatment_ids'])}
    rows = []
    for q, query in enumerate(plan['queries']):
        path = Path(query['native_path'])
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != query['native_sha256']:
                raise ValueError('native query changed')
        codes = linear16_to_q8(np.load(path, mmap_mode='r', allow_pickle=False))
        source_name = f'query_{q}.png'
        Image.fromarray(codes).save(directory / source_name)
        for index, case in enumerate(cases):
            if case['query'] != query['identity']:
                continue
            d, t = donor_index[case['donor']], treatment_index[case['treatment']]
            parameters = {'ideal': np.asarray(plan['treatments'][t]['u']),
                'learned': arrays['learned_u'][d, t], 'paired_oracle': arrays['paired_oracle_u'][d, t]}
            files = {'source': source_name}
            for method, u in parameters.items():
                name = f'case_{index:02d}_{method}.png'
                Image.fromarray(render_codes(codes, u)).save(directory / name)
                files[method] = name
            rows.append({'index': index, **case, 'files': files, 'native_shape': list(codes.shape)})
        del codes
    rows.sort(key=lambda row: row['index'])
    if [row['index'] for row in rows] != list(range(64)):
        raise ValueError('not every fixed case was rendered exactly once')
    labels = {'source': '原图', 'ideal': '理想目标', 'learned': '模型结果', 'paired_oracle': '真实统计量对照'}
    body = ['<!doctype html><meta charset="utf-8"><title>固定64例视觉检查</title>',
        '<style>body{font:16px sans-serif;background:#202020;color:#eee;margin:24px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}img{width:100%;height:auto}section{margin-bottom:40px}a{color:#ddd}</style>',
        '<h1>固定64例视觉检查</h1><p>点击图片查看原生分辨率。模型与真实统计量对照分别判定：明显改变、舒适度、事实损失。</p>']
    for row in rows:
        body.append(f'<section><h2>{row["index"]+1:02d} · {html.escape(row["query"])} · {html.escape(row["treatment"])}</h2><div class="grid">')
        for method, label in labels.items():
            name = row['files'][method]
            body.append(f'<div>{label}<a href="{name}" target="_blank"><img loading="lazy" src="{name}" alt="{label}"></a></div>')
        body.append('</div></section>')
    (directory / 'index.html').write_text('\n'.join(body), encoding='utf-8')
    manifest = {'status': 'AWAITING_VISUAL_REVIEW', 'cases': rows,
        'image_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(directory.glob('*.png'))}}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
