import hashlib
import json
import re
import csv
import io
from pathlib import Path


def build(root):
    ledger_path = root / 'outputs/fable_cdfe68_exposure_v1/ledger.json'
    ledger = json.loads(ledger_path.read_bytes())
    missing = set(ledger['history_without_recorded_raw_metadata'])
    parents = {Path(d['path']).parent.parent for r in ledger['rows'] if r['identity'] in missing
               and r['identity'].startswith('rawpixls/') for d in r['representations']}
    files = {p for parent in parents for pattern in ('manifest.json', 'rows.json') for p in parent.rglob(pattern)}
    discovery = root / 'outputs/fable_source_population_bridge_v1/expanded_history_discovery.json'
    files.update(root / b['path'] for b in json.loads(discovery.read_bytes())['batches'])
    records = {identity: [] for identity in sorted(missing)}
    hashes = {str(ledger_path.relative_to(root)): hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
              str(discovery.relative_to(root)): hashlib.sha256(discovery.read_bytes()).hexdigest()}

    def visit(value, source, pointer):
        if isinstance(value, list):
            for i, item in enumerate(value): visit(item, source, pointer + f'/{i}')
        elif isinstance(value, dict):
            identity = None
            url = value.get('source_url') or value.get('download_url', '')
            match = re.search(r'https://raw\.pixls\.us/getfile\.php/(\d+)/', str(url))
            if match: identity = 'rawpixls/' + match[1]
            elif isinstance(value.get('repository_id'), int): identity = 'rawpixls/' + str(value['repository_id'])
            elif value.get('source_name'): identity = 'fivek/' + value['source_name']
            if identity in records:
                keys = ('raw_path', 'raw_sha256', 'dng_path', 'dng_sha256', 'dng_url', 'source_url',
                        'metadata_sha256', 'exif_url', 'fields', 'make', 'model', 'date',
                        'download_url', 'sha256', 'raw_tar_member', 'raw_gold')
                evidence = {k: value[k] for k in keys if k in value}
                if evidence:
                    records[identity].append({'source': source, 'pointer': pointer, 'record': evidence,
                        'recorded_path_exists_now': {k: (root / value[k]).is_file() for k in ('raw_path', 'dng_path') if value.get(k)}})
            for k, item in value.items():
                if isinstance(item, (dict, list)): visit(item, source, pointer + '/' + k)

    absent = []
    for path in sorted(files):
        source = str(path.relative_to(root))
        if not path.is_file():
            absent.append(source)
            continue
        blob = path.read_bytes()
        hashes[source] = hashlib.sha256(blob).hexdigest()
        visit(json.loads(blob), source, '')
    for source in ['outputs/fivek_auto_optimize/freeze_v1/manifest.csv',
                   'outputs/fable_contrast_source_preparation_v1/repository.json']:
        blob = (root / source).read_bytes()
        hashes[source] = hashlib.sha256(blob).hexdigest()
        if source.endswith('.csv'):
            visit(list(csv.DictReader(io.StringIO(blob.decode('utf-8')))), source, '')
        else:
            for index, row in enumerate(json.loads(blob)['data']):
                match = re.search(r"href='(https://raw\.pixls\.us/getfile\.php/(\d+)/nice/[^']+)'", row[7])
                digest = re.search(r"sha256 Checksum'>([0-9a-f]{64})", row[7])
                if match and digest and 'rawpixls/' + match[2] in missing:
                    visit({'download_url': match[1], 'sha256': digest[1], 'make': row[0],
                           'model': row[1], 'date': row[6]}, source, f'/data/{index}')
    return {'status': 'EXISTING_RECORD_LOCATORS_NOT_RESOLVED_HISTORY', 'source_sha256': hashes,
            'missing_record_files': absent, 'rows': [{'identity': i, 'evidence': e} for i, e in records.items()],
            'matched_identities': sum(bool(v) for v in records.values()),
            'limits': ['Existing metadata records are leads pending source/hash reconciliation.',
                       'Repository dates are not assumed capture timestamps; no freshness conclusions.',
                       'One bounded pass through representation-parent manifests/rows and recorded FiveK batches; no external requests.']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    report = build(root)
    with (root / 'outputs/fable_cdfe68_exposure_v1/history_locators_v2.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'matched': report['matched_identities'], 'total': len(report['rows']), 'source_files': len(report['source_sha256'])}))
