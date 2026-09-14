import hashlib
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    destination = parser.parse_args().output
    if destination.exists():
        raise FileExistsError(destination)
    index = ROOT / 'data/raw/fivek/fivek_index.html'
    legal = ROOT / 'data/raw/fivek/legal'
    files = [index] + [legal / n for n in ('filesAdobe.txt', 'filesAdobeMIT.txt', 'LicenseAdobe.txt', 'LicenseAdobeMIT.txt')]
    partitions = {name: {line.strip() for line in (legal / f'files{name}.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()}
                  for name in ('Adobe', 'AdobeMIT')}
    if partitions['Adobe'] & partitions['AdobeMIT']:
        raise ValueError('overlapping license membership')
    rows = []
    for tr in BeautifulSoup(index.read_text(encoding='utf-8'), 'html.parser').select('table.data tr'):
        link = tr.find('a', href=re.compile(r'^img/dng/.*\.dng$'))
        if link is None:
            continue
        stem = Path(link['href']).stem
        preview = tr.find('img')
        if preview is None or Path(preview.get('src', '')).stem != stem:
            raise ValueError(f'preview identity mismatch: {stem}')
        membership = [name for name, values in partitions.items() if stem in values]
        if len(membership) != 1:
            raise ValueError(f'license mapping missing: {stem}')
        fields = {}
        for cell in tr.find_all('td'):
            value = cell.get_text(' ', strip=True)
            if ':' in value:
                key, text = value.split(':', 1)
                if key in ('Subject', 'Light', 'Location', 'Time', 'EXIF'):
                    fields[key] = text.strip()
        if 'EXIF' not in fields:
            raise ValueError(f'missing camera evidence: {stem}')
        camera_match = re.match(r'^(.*?)\s+(?:\d+(?:\.\d+)?mm\b|F/)', fields['EXIF'])
        rows.append({'source_name': stem, 'dataset_id': stem.split('-')[0],
                     'dng_url': urljoin('https://data.csail.mit.edu/graphics/fivek/', link['href']),
                     'preview_url': urljoin('https://data.csail.mit.edu/graphics/fivek/', preview['src']),
                     'preview_role': 'Expert C thumbnail; admission only, not canonical target',
                     'license_partition': membership[0], 'publisher_fields': fields,
                     'camera_label_unvalidated': camera_match.group(1) if camera_match else None,
                     'photographer': 'UNKNOWN', 'capture_timestamp': 'UNKNOWN', 'sequence': 'UNKNOWN'})
    stems = {r['source_name'] for r in rows}
    if len(rows) != 5000 or len(stems) != 5000 or len({r['dataset_id'] for r in rows}) != 5000:
        raise ValueError('expected 5000 unique publisher identities')
    if stems != partitions['Adobe'] | partitions['AdobeMIT']:
        raise ValueError('license lists and index coverage differ')
    payload = {'status': 'CACHED_INDEX_MAPPING_VERIFIED_NOT_CANDIDATE_FREEZE',
               'input_sha256': {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
               'rows': rows, 'license_counts': {k: len(v) for k, v in partitions.items()},
               'unparsed_camera_labels': sum(r['camera_label_unvalidated'] is None for r in rows),
               'limits': ['Cached index provenance only; current remote availability unverified',
                          'Camera labels not alias-normalized or RAW-verified',
                          'License membership is not legal/product clearance',
                          'No candidate selection, history closure or lineage admission']}
    destination.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
