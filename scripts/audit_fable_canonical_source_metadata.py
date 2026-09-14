import hashlib
import json
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_rawpixls_velvia_preview import parse_repository_row


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'outputs/fable_contrast_source_preparation_v1/repository.json'
    output = ROOT / 'outputs/fable_canonical_source_metadata_v1'
    output.mkdir(exist_ok=False)
    groups = defaultdict(list)
    for raw in json.loads(source.read_text())['data']:
        parsed = parse_repository_row(raw)
        if parsed and parsed['sha256']:
            parsed['repository_id'] = int(re.search(r'getfile.php/(\d+)/', parsed['download_url'])[1])
            groups[(parsed['make'], parsed['model'])].append(parsed)
    eligible = [k for k, rows in groups.items() if len(rows) >= 4 and len({r['date'] for r in rows}) >= 2]
    ranked = sorted(eligible, key=lambda k: (-len({r['date'] for r in groups[k]}), -len(groups[k]), k))[:20]
    rows = [r for k in ranked for r in groups[k]]
    protocol = {'status': 'METADATA_CANDIDATES_NOT_SCENE_ADMISSION', 'source_sha256': digest(source),
        'parser_sha256': digest(ROOT / 'scripts/make_rawpixls_velvia_preview.py'), 'entry_sha256': digest(Path(__file__)),
        'selection': 'supportedCC0camera with>=4rows and>=2upload dates; sort distinctuploaddatesdesc,rowsdesc,make/model;first20',
        'eligible_camera_count': len(eligible), 'selected_cameras': ranked, 'rows': rows,
        'no_raw_download': True, 'no_lineage_claim': True}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')

    def fetch(row):
        result = {k: row[k] for k in ['repository_id','make','model','date','mode','sha256','size_mb','exif_url']}
        if not row['exif_url']:
            return {**result, 'status': 'NO_EXIF_URL'}
        try:
            with urllib.request.urlopen(row['exif_url'], timeout=45) as response:
                data = response.read(1048577)
            if len(data) > 1048576:
                return {**result, 'status': 'EXIF_TOO_LARGE'}
            fields = {}
            allowed = {'Exif.Photo.DateTimeOriginal','Exif.Photo.DateTimeDigitized','Exif.Image.DateTime',
                       'Exif.Image.Make','Exif.Image.Model','Exif.Image.ImageWidth','Exif.Image.ImageLength'}
            for line in data.decode('utf-8', 'replace').splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0] in allowed:
                    fields[parts[0]] = parts[1].strip()
            return {**result, 'status': 'FETCHED_METADATA_ONLY', 'metadata_sha256': hashlib.sha256(data).hexdigest(),
                    'bytes': len(data), 'fields': fields}
        except Exception as exc:
            return {**result, 'status': 'FETCH_FAILED', 'error_type': type(exc).__name__}

    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(fetch, rows))
    summaries = []
    for make, model in ranked:
        camera_rows = [r for r in fetched if (r['make'], r['model']) == (make, model)]
        stamps = [r.get('fields', {}).get('Exif.Photo.DateTimeOriginal') for r in camera_rows]
        known = [v for v in stamps if v]
        summaries.append({'make': make, 'model': model, 'files': len(camera_rows), 'original_timestamp_rows': len(known),
            'distinct_timestamps': len(set(known)), 'distinct_capture_dates': len({v[:10] for v in known}),
            'missing_original_timestamp': len(stamps) - len(known),
            'limitation': 'Dates/timestamps neither prove unique scene nor exclude alternate render/burst; no pixel check or historical exclusion yet.'})
    (output / 'rows.json').write_text(json.dumps(fetched, indent=2), encoding='utf-8')
    report = {'status': 'BOUNDED_METADATA_RETRIEVAL_COMPLETE_NO_ADMISSION', 'cameras': summaries,
        'files_requested': len(rows), 'fetched': sum(r['status'] == 'FETCHED_METADATA_ONLY' for r in fetched),
        'protocol_sha256': digest(output / 'protocol.json'), 'rows_sha256': digest(output / 'rows.json'),
        'raw_files_downloaded': 0, 'admitted_scene_lineages': 0,
        'website_limitation': 'Rawpixls requests format/crop/bitdepth variations and explicitly discourages people photos; source suitability must be established, not inferred from file count.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
