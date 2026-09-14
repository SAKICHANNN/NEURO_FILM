import numpy as np

from src.eval.fable_cdfe_assessment_predictions import assessment_predictions
from src.eval.fable_cdfe_controls import decode_assessment_controls
from src.eval.fable_cdfe_query_counts import load_query_counts
from src.eval.fable_cdfe_case_metrics import paired_case_metrics
from src.eval.fable_cdfe_assessment_gates import assessment_gates


def evaluate_assessment(model, plan: dict, *, assignment: dict, normalizer: dict,
                        native_config: dict, progress) -> dict:
    ids = assignment['represented_assessment'] + [i for camera in assignment['held_cameras']
                                                 for i in assignment['held_assessment'][camera]]
    predictions = assessment_predictions(model, plan['rows'], plan['treatments'], locked_ids=ids,
        locked_treatment_ids=plan['treatment_ids'], normalizer=normalizer,
        native_config=native_config, progress=progress)
    cameras = [r['camera'] for r in plan['rows']]
    parameters = decode_assessment_controls(predictions['predicted_canonical'],
        predictions['after_mu'], predictions['after_scale'], fitting_mean=normalizer['mean'],
        donor_ids=ids, cameras=cameras, shuffle_mapping=plan['shuffle_mapping'],
        slope_limit=1.25, offset_limit=.35)
    queries = load_query_counts(plan['queries'], plan['masks'], locked_ids=assignment['queries'])
    ideal = np.tile(np.asarray([t['u'] for t in plan['treatments']]), (32, 1))
    cases = paired_case_metrics({name: u.reshape(1024, 4) for name, u in parameters.items()},
        ideal, queries['counts'], slope_limit=1.25, offset_limit=.35)
    gates = assessment_gates(cases, donor_ids=ids, query_ids=assignment['queries'],
        donor_cameras=cameras, treatment_regions=plan['treatment_regions'],
        query_counts=queries['counts'], query_regions=queries['regions'])
    arrays = {key: predictions[key] for key in ['predicted_canonical', 'after_mu', 'after_scale']}
    arrays['D'] = cases['D']
    for name, u in parameters.items():
        arrays[name+'_u'] = u
        for metric in ['E', 'P']:
            arrays[name+'_'+metric] = cases['methods'][name][metric]
    return {'arrays': arrays, 'gates': gates, 'identities': ids,
            'query_ids': assignment['queries'], 'treatment_ids': plan['treatment_ids']}
