import hashlib
import json
from pathlib import Path

from src.training.fable_runtime import computation_runtime
from src.training.fable_source_closure import local_import_closure


def main():
    root = Path(__file__).resolve().parents[1]
    base = 'outputs/fable_cdfe68_computation_v1/'
    fields = {'training_config': 'configs/fable_cdfe68_training_draft_v1.json',
        'cache_plan': base+'fit_cache_plan_v2.json', 'validation_plan': base+'validation_plan_v1.json',
        'assessment_plan': base+'assessment_plan_v1.json',
        'cache_directory': 'outputs/fable_cdfe68_fit_cache_v1',
        'output_directory': 'outputs/fable_cdfe68_training_v1'}
    sources = local_import_closure(root, ['scripts/train_fable_cdfe.py', 'scripts/validate_fable_cdfe.py'])
    config = json.loads((root / fields['training_config']).read_text())
    paths = [fields[k] for k in ['training_config', 'cache_plan', 'validation_plan', 'assessment_plan']]
    paths += [config[k] for k in ['role_lock', 'model_config', 'native_measurement_config']]
    paths += ['configs/fable_cdfe68_evaluation_draft_v1.json', base+'cached_backward_engineering.json',
              base+'query_counts_check.json', 'scripts/prepare_fable_cdfe_seal.py']
    for relative in paths:
        blob = (root / relative).read_bytes()
        sources[relative] = hashlib.sha256(blob).hexdigest()
        if relative.endswith('.json'):
            for dependency, expected in json.loads(blob).get('source_sha256', {}).items():
                actual = hashlib.sha256((root / dependency).read_bytes()).hexdigest()
                if actual != expected or (dependency in sources and sources[dependency] != expected):
                    raise ValueError(f'bound dependency changed: {dependency}')
                sources[dependency] = expected
    runtime = computation_runtime()
    candidate = dict(status='CANDIDATE_NOT_TRAINING_ADMISSION', **fields,
        source_sha256=sources, computation_runtime=runtime,
        runtime={k: runtime['python'] if k == 'python' else runtime['packages'][k]
                 for k in ['python', 'numpy', 'torch']},
        unresolved=['Complete cache receipt binding', 'Final integration and visual-output readiness review'])
    path = root / (base+'training_seal_candidate_v1.json')
    with path.open('x') as stream:
        json.dump(candidate, stream, indent=2)
    print(json.dumps({'status': candidate['status'], 'source_bindings': len(sources)}))


if __name__ == '__main__':
    main()
