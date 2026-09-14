import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from src.data.fable_windows_memory import limit_current_process_committed_memory
from src.training.fable_source_closure import local_import_closure
from src.training.fable_runtime import computation_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--assessment', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    blob = args.contract.read_bytes()
    seal = json.loads(blob)
    if seal.get('status') != 'SEALED_FOR_SINGLE_FITTING_RUN':
        raise ValueError('sealed training contract required before validation')
    if seal['computation_runtime'] != computation_runtime():
        raise ValueError('computation environment differs from seal')
    required = {'scripts/validate_fable_cdfe.py', 'src/eval/fable_cdfe_validation.py',
        'src/eval/fable_cdfe_gates.py', 'src/training/fable_cdfe_checkpoint.py',
        'src/models/canonical_photometry.py', 'src/data/fable_windows_memory.py',
        seal['training_config'], seal['validation_plan'], seal['cache_plan']}
    if args.assessment:
        required.update({seal['assessment_plan'], *['src/eval/'+name+'.py' for name in
            ['fable_cdfe_assessment', 'fable_cdfe_assessment_predictions', 'fable_cdfe_controls',
             'fable_cdfe_query_counts', 'fable_cdfe_case_metrics', 'fable_cdfe_assessment_gates',
             'fable_photometry_metrics', 'fable_protected_regions', 'fable_paired_contrast']]})
    if not required.issubset(seal['source_sha256']):
        raise ValueError('validation dependency bindings incomplete')
    for relative, expected in local_import_closure(root, ['scripts/validate_fable_cdfe.py']).items():
        if seal['source_sha256'].get(relative) != expected:
            raise ValueError(f'unbound evaluation import: {relative}')
    cache_plan = json.loads((root / seal['cache_plan']).read_text())
    plan = json.loads((root / seal['validation_plan']).read_text())
    dependency_plans = [cache_plan, plan]
    if args.assessment:
        dependency_plans.append(json.loads((root / seal['assessment_plan']).read_text()))
    for dependency_plan in dependency_plans:
        for relative, expected in dependency_plan['source_sha256'].items():
            if seal['source_sha256'].get(relative) != expected:
                raise ValueError('computation bindings differ')
    for relative, expected in seal['source_sha256'].items():
        with (root / relative).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'sealed source changed: {relative}')
    config = json.loads((root / seal['training_config']).read_text())
    for key in ['role_lock', 'model_config', 'native_measurement_config']:
        if config[key] not in seal['source_sha256']:
            raise ValueError(f'unbound {key}')
    job = limit_current_process_committed_memory(config['training']['allocation_limit_bytes'])
    import numpy as np
    import torch
    from src.models.canonical_photometry import CanonicalPhotometryCNN
    from src.training.fable_cdfe_checkpoint import load_final_checkpoint
    from src.eval.fable_cdfe_validation import evaluate_validation

    if seal['runtime'] != {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__}:
        raise ValueError('validation runtime differs')
    torch.set_num_threads(config['training']['cpu_threads'])
    torch.set_num_interop_threads(config['training']['interop_threads'])
    torch.use_deterministic_algorithms(True)
    output = root / seal['output_directory']
    contract = hashlib.sha256(blob).hexdigest()
    checkpoint = load_final_checkpoint(output, contract_sha256=contract, expected_steps=4000)
    architecture = json.loads((root / config['model_config']).read_text())['architecture']
    model = CanonicalPhotometryCNN(**architecture)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    lock = json.loads((root / config['role_lock']).read_text())
    plan = json.loads((root / seal['validation_plan']).read_text())
    native = json.loads((root / config['native_measurement_config']).read_text())
    checkpoint_sha = hashlib.sha256((output / 'final.pt').read_bytes()).hexdigest()
    if args.assessment:
        from src.eval.fable_cdfe_gates import validation_gate
        from src.eval.fable_cdfe_assessment import evaluate_assessment
        from src.eval.fable_cdfe_visual_panel import render_comfort_panel

        validation = json.loads((output / 'validation.json').read_text())
        if (validation['contract_sha256'] != contract or validation['checkpoint_sha256'] != checkpoint_sha
                or validation['identities'] != lock['assignment']['validation']
                or validation['treatment_ids'] != plan['treatment_ids']):
            raise ValueError('validation provenance differs from final checkpoint')
        if hashlib.sha256((output / 'validation.npz').read_bytes()).hexdigest() != validation['arrays_sha256']:
            raise ValueError('validation arrays changed')
        with np.load(output / 'validation.npz', allow_pickle=False) as values:
            gate = validation_gate(values['predicted'], values['targets'], np.zeros(4))
        if not gate['passed'] or gate != validation['gate']:
            raise ValueError('fixed validation gate failed or report differs')
        with (output / 'assessment.claim').open('x') as stream:
            json.dump({'contract_sha256': contract, 'checkpoint_sha256': checkpoint_sha}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        assessment_plan = dependency_plans[-1]
        result = evaluate_assessment(model, assessment_plan, assignment=lock['assignment'],
            normalizer={k: v.numpy() for k, v in checkpoint['normalizer'].items()},
            native_config=native, progress=lambda row: print(json.dumps(row), flush=True))
        with (output / 'assessment.npz.partial').open('xb') as stream:
            np.savez(stream, **result['arrays'])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(output / 'assessment.npz.partial', output / 'assessment.npz')
        report = {k: v for k, v in result.items() if k != 'arrays'}
        report.update(contract_sha256=contract, checkpoint_sha256=checkpoint_sha,
            arrays_sha256=hashlib.sha256((output / 'assessment.npz').read_bytes()).hexdigest())
        with (output / 'assessment.json').open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        if report['gates']['numeric_passed']:
            render_comfort_panel(output / 'comfort', assessment_plan, report, result['arrays'])
        print(json.dumps({'numeric_passed': report['gates']['numeric_passed'],
                          'visual_review_required': True}), flush=True)
        return
    with (output / 'validation.claim').open('x') as stream:
        json.dump({'contract_sha256': contract,
                   'checkpoint_sha256': hashlib.sha256((output / 'final.pt').read_bytes()).hexdigest()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = evaluate_validation(model, plan['rows'], plan['treatments'],
        locked_ids=lock['assignment']['validation'], locked_treatment_ids=plan['treatment_ids'],
        normalizer={k: v.numpy() for k, v in checkpoint['normalizer'].items()}, native_config=native)
    temporary = output / 'validation.npz.partial'
    with temporary.open('xb') as stream:
        np.savez(stream, predicted=result['predicted'], targets=result['targets'])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output / 'validation.npz')
    report = {k: v for k, v in result.items() if k not in ['predicted', 'targets']}
    report.update(contract_sha256=contract, checkpoint_sha256=checkpoint_sha,
        arrays_sha256=hashlib.sha256((output / 'validation.npz').read_bytes()).hexdigest())
    with (output / 'validation.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report['gate']), flush=True)


if __name__ == '__main__':
    main()
