import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.tst_reference_response import descriptors, load_vgg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    fit = ROOT / 'outputs/fable_reference_photometry_v1/fit_features'
    report = json.loads((fit / 'report.json').read_text())
    entry = ROOT / 'src/eval/tst_reference_response.py'
    if digest(fit / 'features.npz') != report['features_sha256'] or digest(entry) != report['descriptor_module_sha256']:
        raise ValueError('frozen feature provenance mismatch')
    with np.load(fit / 'features.npz', allow_pickle=False) as data:
        bases = data['canonical_bases'].copy()
    if bases.shape != (6, 128, 128, 3):
        raise ValueError('six frozen canonical identity references required')
    out = ROOT / 'outputs/fable_paired_contrast_v1/fit_identities'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = load_vgg(ROOT)
    rows = [descriptors(base, model) for base in bases]
    np.savez(out / 'features.npz', stats=np.stack([r[0] for r in rows]),
             deep=np.stack([r[1] for r in rows]), donor_ids=np.arange(6))
    receipt = {'status': 'SIX_CANONICAL_IDENTITIES_EXTRACTED_NO_FIT',
        'fit_features_sha256': report['features_sha256'], 'descriptor_sha256': digest(entry),
        'entry_sha256': digest(Path(__file__)), 'features_sha256': digest(out / 'features.npz'),
        'identity_rule': 'Use frozen canonical Q8 bases, not native JPEG descriptors; no regression labels.'}
    (out / 'report.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
