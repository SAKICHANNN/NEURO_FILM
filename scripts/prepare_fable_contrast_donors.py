import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_rawpixls_velvia_preview import render_raw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cp = ROOT / 'configs/fable_contrast_donor_candidates_v1.json'
    cfg = json.loads(cp.read_text())
    if digest(ROOT / cfg['renderer'].split(':')[0]) != cfg['renderer_sha256']:
        raise ValueError('source decoder changed')
    raw_root, out = ROOT / cfg['download_root'], ROOT / cfg['preview_root']
    raw_root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for row in cfg['candidates']:
        name = str(row['repository_id'])
        path = raw_root / (name + row['extension'])
        receipt_path = out / (name + '.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if digest(path) != row['sha256'] or digest(ROOT / receipt['path']) != receipt['sha256']:
                raise ValueError('existing source receipt mismatch')
            continue
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(row['download_url'], timeout=60) as response, temporary.open('wb') as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            if digest(temporary) != row['sha256']:
                raise ValueError('download checksum mismatch')
            temporary.rename(path)
        if digest(path) != row['sha256']:
            raise ValueError('raw checksum mismatch')
        preview = out / (name + '.jpg')
        render_raw(path, preview, cfg['max_side'])
        receipt = {**row, 'path': str(preview.relative_to(ROOT)), 'sha256': digest(preview),
            'raw_path': str(path.relative_to(ROOT)), 'raw_sha256': row['sha256'],
            'config_sha256': digest(cp), 'entry_sha256': digest(Path(__file__)),
            'status': 'SOURCE_ONLY_DECODE_PENDING_VISUAL_DUPLICATE_CHECK'}
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps({'decoded': name, 'preview': str(preview)}), flush=True)


if __name__ == '__main__':
    main()
