import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1152)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError('positive object limit required')
    cfg_path = ROOT / 'configs/fable_fivek_preview_protocol_v1.json'
    cfg = json.loads(cfg_path.read_text())
    cohort_path = ROOT / cfg['cohort']
    if hashlib.sha256(cohort_path.read_bytes()).hexdigest() != cfg['cohort_sha256']:
        raise ValueError('frozen cohort hash mismatch')
    cohort = json.loads(cohort_path.read_text())
    objects = [(kind, row) for kind in ('history_restore', 'candidates') for row in cohort[kind]]
    output = ROOT / 'outputs/fable_source_population_bridge_v1/previews'
    output.mkdir(exist_ok=True)
    ledger_path = output / 'ledger.json'
    protocol_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'protocol_sha256': protocol_hash, 'body_bytes': 0, 'objects': {}, 'status': 'PREPARED'}
    if ledger['protocol_sha256'] != protocol_hash:
        raise ValueError('protocol changed')
    if ledger['status'] in ('TRANSPORT_STOP', 'BYTE_CAP_STOP') or any(
            a['status'] == 'STARTED' for v in ledger['objects'].values() for a in v['attempts']):
        raise ValueError('terminal or uncertain transfer requires explicit local adjudication; no automatic retry')
    session = requests.Session()
    failures = 0
    for previous in reversed(list(ledger['objects'].values())):
        if previous['status'] != 'FAILED':
            break
        failures += 1
    if failures >= 3:
        raise ValueError('three consecutive failed objects already recorded; no automatic retry')
    processed = 0
    for kind, row in objects:
        key = kind + '/' + row['dataset_id']
        if key in ledger['objects']:
            continue
        if processed >= args.limit:
            break
        record = {'source_name': row['source_name'], 'url': row['preview_url'], 'attempts': [], 'status': 'PENDING'}
        ledger['objects'][key] = record
        url = urlsplit(row['preview_url'])
        if url.scheme != 'https' or url.hostname != 'data.csail.mit.edu' or not url.path.startswith('/graphics/fivek/img/thmb_c/'):
            raise ValueError('unexpected publisher thumbnail URL')
        encoded_url = urlunsplit((url.scheme, url.netloc, quote(url.path, safe='/%'), url.query, ''))
        for number in range(cfg['max_attempts_per_object']):
            attempt = {'number': number + 1, 'status': 'STARTED', 'body_bytes': 0}
            record['attempts'].append(attempt)
            ledger['status'] = 'RUNNING'
            save(ledger_path, ledger)
            start = time.monotonic()
            data = bytearray()
            try:
                remaining = cfg['max_preview_bytes_including_retries'] - ledger['body_bytes']
                if remaining <= 0:
                    raise RuntimeError('BYTE_CAP')
                with session.get(encoded_url, timeout=(15, 30), stream=True, allow_redirects=False) as response:
                    attempt['http_status'] = response.status_code
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise ValueError('non-200 response; no redirect fallback')
                    while remaining:
                        block = response.raw.read(min(65536, remaining))
                        if not block:
                            break
                        data.extend(block)
                        ledger['body_bytes'] += len(block)
                        attempt['body_bytes'] += len(block)
                        remaining -= len(block)
                    if not remaining:
                        raise RuntimeError('BYTE_CAP')
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    record['size'] = list(image.size)
                    record['format'] = image.format
                target = output / (kind + '_' + row['dataset_id'] + '.image')
                target.write_bytes(data)
                record.update(status='FETCHED', path=target.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(data).hexdigest())
                attempt['status'] = 'SUCCESS'
            except Exception as error:
                attempt.update(status='FAILED', error=type(error).__name__ + ': ' + str(error)[:350])
                if str(error) == 'BYTE_CAP':
                    ledger['status'] = 'BYTE_CAP_STOP'
            attempt['elapsed_seconds'] = time.monotonic() - start
            save(ledger_path, ledger)
            if record['status'] == 'FETCHED' or ledger['status'] == 'BYTE_CAP_STOP':
                break
        if record['status'] != 'FETCHED':
            record['status'] = 'FAILED'
            failures += 1
        else:
            failures = 0
        processed += 1
        if failures >= 3:
            ledger['status'] = 'TRANSPORT_STOP'
        save(ledger_path, ledger)
        if ledger['status'] in ('TRANSPORT_STOP', 'BYTE_CAP_STOP'):
            break
    if ledger['status'] == 'RUNNING':
        ledger['status'] = 'FETCH_COMPLETE' if len(ledger['objects']) == len(objects) else 'BATCH_PAUSED'
    save(ledger_path, ledger)
    print(json.dumps({'status': ledger['status'], 'objects': len(ledger['objects']),
                      'fetched': sum(r['status'] == 'FETCHED' for r in ledger['objects'].values()),
                      'body_bytes': ledger['body_bytes']}))


if __name__ == '__main__':
    main()
