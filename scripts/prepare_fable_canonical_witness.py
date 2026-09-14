import hashlib
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rawpy
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cp = ROOT / 'outputs/fable_canonical_source_metadata_v1/witness_protocol.json'
    cfg = json.loads(cp.read_text())
    out, raw_root = ROOT / cfg['preview_root'], ROOT / cfg['raw_root']
    out.mkdir(exist_ok=False)
    raw_root.mkdir(parents=True, exist_ok=True)

    def obtain(row):
        identifier = str(row['repository_id'])
        raw_path = raw_root / (identifier + row['extension'])
        record = {'id': row['repository_id'], 'camera': [row['make'], row['model']],
            'raw_path': str(raw_path.relative_to(ROOT)), 'expected_raw_sha256': row['sha256']}
        try:
            if not raw_path.exists():
                part = raw_path.with_suffix(raw_path.suffix + '.part')
                total = 0
                with urllib.request.urlopen(row['download_url'], timeout=60) as response, part.open('xb') as target:
                    while chunk := response.read(1048576):
                        total += len(chunk)
                        if total > cfg['maximum_file_bytes'][identifier]:
                            raise ValueError('declared file byte cap exceeded')
                        target.write(chunk)
                if digest(part) != row['sha256']:
                    raise ValueError('download hash mismatch')
                part.rename(raw_path)
            if digest(raw_path) != row['sha256']:
                raise ValueError('cached RAW hash mismatch')
            record['raw_bytes'] = raw_path.stat().st_size
            with rawpy.imread(str(raw_path)) as raw:
                thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                preview = out / (identifier + '.jpg')
                preview.write_bytes(thumb.data)
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                preview = out / (identifier + '.png')
                Image.fromarray(thumb.data).save(preview)
            else:
                raise ValueError('unsupported embedded preview')
            with Image.open(preview) as image:
                size = list(image.size)
            record.update(status='EMBEDDED_PREVIEW_EXTRACTED_NOT_REVIEWED', preview=str(preview.relative_to(ROOT)),
                          preview_sha256=digest(preview), size=size)
        except Exception as exc:
            record.update(status='UNASSESSABLE', error_type=type(exc).__name__, error=str(exc))
        (out / (identifier + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps({'id': row['repository_id'], 'status': record['status']}), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(obtain, cfg['rows']))
    total = sum(r.get('raw_bytes', 0) for r in rows)
    if total > cfg['maximum_total_bytes']:
        raise ValueError('total byte cap exceeded')
    for camera_index, group in enumerate(cfg['cameras']):
        sheet = Image.new('RGB', (1200, 400), 'white')
        for column, identifier in enumerate(group['selected_ids']):
            row = next(r for r in rows if r['id'] == identifier)
            ImageDraw.Draw(sheet).text((column * 300 + 4, 4), str(identifier) + ' ' + ' '.join(group['camera']), fill='black')
            if 'preview' in row:
                with Image.open(ROOT / row['preview']) as im:
                    thumb = im.convert('RGB'); thumb.thumbnail((300, 360))
                    sheet.paste(thumb, (column * 300, 30))
        sheet.save(out / f'camera{camera_index}.png')
    report = {'status': 'SIXTEEN_SOURCE_WITNESSES_READY_FOR_REVIEW', 'protocol_sha256': digest(cp),
        'entry_sha256': digest(Path(__file__)), 'raw_bytes': total, 'rows': rows,
        'admitted_scene_lineages': 0, 'training': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'raw_bytes': total}))


if __name__ == '__main__':
    main()
