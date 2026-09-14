import hashlib
import json
from pathlib import Path

import tifffile


def read_fields(path):
    fields = []
    with tifffile.TiffFile(path) as tiff:
        queue = list(tiff.pages)
        while queue:
            page = queue.pop(0)
            values = {name: page.tags[name].value for name in ('Make', 'Model', 'DateTime') if name in page.tags}
            if 'ExifTag' in page.tags:
                exif = page.tags['ExifTag'].value
                if isinstance(exif, dict):
                    values.update({k: exif[k] for k in ('DateTimeOriginal', 'DateTimeDigitized', 'BodySerialNumber') if k in exif})
            fields.append({'ifd_offset': page.offset, 'fields': values})
            if page.pages is not None:
                queue.extend(page.pages)
    return fields


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'outputs/fable_cdfe68_exposure_v1/history_locators.json'
    ledger = json.loads(source.read_bytes())
    output = root / 'outputs/fable_cdfe68_exposure_v1/local_history_audit.jsonl'
    cache = {}
    with output.open('x', encoding='utf-8') as stream:
        for row in ledger['rows']:
            candidates = {}
            for evidence in row['evidence']:
                record = evidence['record']
                for path_key, sha_key in [('raw_path', 'raw_sha256'), ('dng_path', 'dng_sha256')]:
                    if record.get(path_key) and record.get(sha_key):
                        path = root / record[path_key]
                        if path.is_file():
                            candidates.setdefault(str(path), set()).add(record[sha_key])
            audited = []
            for name, expected in sorted(candidates.items()):
                if name not in cache:
                    path = Path(name)
                    with path.open('rb') as raw:
                        digest = hashlib.file_digest(raw, 'sha256').hexdigest()
                    cache[name] = {'path': name, 'sha256': digest, 'bytes': path.stat().st_size}
                item = dict(cache[name], expected_sha256=sorted(expected))
                if expected != {item['sha256']}:
                    item['status'] = 'HASH_CONFLICT'
                else:
                    try:
                        item['metadata'] = read_fields(Path(name))
                        item['status'] = 'HASH_MATCH_METADATA_READ'
                    except (ValueError, OSError, TypeError) as exc:
                        item.update(status='HASH_MATCH_METADATA_UNSUPPORTED', error=str(exc))
                audited.append(item)
            stream.write(json.dumps({'identity': row['identity'], 'originals': audited,
                                     'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}, ensure_ascii=False) + '\n')
            stream.flush()
    print(json.dumps({'identities': len(ledger['rows']), 'local_files_hashed': len(cache)}))


if __name__ == '__main__':
    main()
