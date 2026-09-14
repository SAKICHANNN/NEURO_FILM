import argparse
import hashlib
import json
from pathlib import Path

from src.training.fable_runtime import computation_runtime
from src.training.fable_source_closure import local_import_closure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = 'outputs/fable_cdfe68_computation_v1/'
    fields = {'training_config': 'configs/fable_cdfe68_training_draft_v1.json',
        'cache_plan': base+'fit_cache_plan_v2.json', 'validation_plan': base+'validation_plan_v1.json',
        'assessment_plan': base+'assessment_plan_v1.json',
        'cache_directory': 'outputs/fable_cdfe68_fit_cache_v1',
        'output_directory': 'outputs/fable_cdfe68_training_v1'}
    sources = local_import_closure(root, ['scripts/train_fable_cdfe.py', 'scripts/validate_fable_cdfe.py',
                                         'scripts/prepare_fable_cdfe_seal.py'])
    config = json.loads((root / fields['training_config']).read_text())
    paths = [fields[k] for k in ['training_config', 'cache_plan', 'validation_plan', 'assessment_plan']]
    paths += [config[k] for k in ['role_lock', 'model_config', 'native_measurement_config']]
    paths += ['configs/fable_cdfe68_evaluation_draft_v1.json', base+'cached_backward_engineering.json',
              base+'query_counts_check.json', base+'consumed_oracle_engineering.json',
              base+'mock_cli_integration.json', 'scripts/prepare_fable_cdfe_seal.py']
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
        unresolved=['Complete cache receipt binding and full fitting array/normalizer validation'])
    filename = 'training_seal_candidate_v2.json'
    if args.final:
        from src.data.fable_windows_memory import limit_current_process_committed_memory
        job = limit_current_process_committed_memory(config['training']['allocation_limit_bytes'])
        from src.data.fable_cdfe_cache import exclusive_cache_writer
        from src.training.fable_cache_seal import completed_cache_bindings
        from src.data.fable_cdfe_training_data import load_fitting_cache

        cache = root / fields['cache_directory']
        if not (cache / 'complete.json').exists():
            raise ValueError('fitting cache unfinished; final seal cannot be created')
        output = root / fields['output_directory']
        if output.exists() and any(output.iterdir()):
            raise ValueError('fixed training output already contains an execution; do not redirect or retry')
        backward = json.loads((root / (base+'cached_backward_engineering.json')).read_text())
        oracle = json.loads((root / (base+'consumed_oracle_engineering.json')).read_text())
        mock = json.loads((root / (base+'mock_cli_integration.json')).read_text())
        if (backward['optimizer_updates'] != 0 or not backward['finite_gradients']
                or not backward['weights_unchanged'] or not oracle['gate']['passed']
                or mock['status'] != 'MOCK_CLI_INTEGRATION_PASSED_NOT_TRAINING'
                or mock['actual_optimizer_updates'] != 0 or mock['real_held_data_opened']):
            raise ValueError('required engineering evidence not satisfied')
        lock = json.loads((root / config['role_lock']).read_text())
        with exclusive_cache_writer(cache):
            bindings = completed_cache_bindings(cache, root / fields['cache_plan'],
                                                locked_ids=lock['assignment']['fit'])
            data = load_fitting_cache(cache, root / fields['cache_plan'],
                locked_fitting_ids=lock['assignment']['fit'], receipt_sha256=bindings['receipt_sha256'])
            candidate.update(bindings)
            candidate['verified_normalizer'] = {key: data['normalizer'][key].tolist() for key in ['mean', 'scale']}
            candidate['verified_input_shape'] = list(data['inputs'].shape)
        candidate.update(status='SEALED_FOR_SINGLE_FITTING_RUN', unresolved=[])
        filename = 'training_seal_final_v1.json'
    path = root / (base+filename)
    with path.open('x') as stream:
        json.dump(candidate, stream, indent=2)
    print(json.dumps({'status': candidate['status'], 'source_bindings': len(sources)}))


if __name__ == '__main__':
    main()
