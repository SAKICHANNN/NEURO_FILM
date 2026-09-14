import numpy as np

from src.eval.fable_cdfe_gates import donor_control_gates, stratified_photometry_gates
from src.eval.fable_protected_regions import protected_region_accuracy, saturation_excess


def assessment_gates(cases: dict, *, donor_ids: list[str], query_ids: list[str],
                     donor_cameras: list[str], treatment_regions: list[str],
                     query_counts: list[np.ndarray], query_regions: list[dict]) -> dict:
    if len(query_counts) != 4 or len(query_regions) != 4 or any(not r for r in query_regions):
        raise ValueError('four query histograms with named regions required')
    methods = cases['methods']
    if any(name not in methods for name in ['learned', 'constant', 'shuffled']):
        raise ValueError('all fixed comparison methods required')
    if np.asarray(cases['D']).shape != (1024, 4):
        raise ValueError('complete donor-major treatment-minor cases required')
    d = cases['D'].reshape(32, 32, 4)
    e = {name: methods[name]['E'].reshape(32, 32, 4) for name in ['learned', 'constant', 'shuffled']}
    photometry = stratified_photometry_gates(e['learned'], d,
        methods['learned']['P'].reshape(32, 32, 4), donor_ids=donor_ids, query_ids=query_ids,
        donor_cameras=donor_cameras, treatment_regions=treatment_regions)
    controls = donor_control_gates(e['learned'], e['constant'], e['shuffled'], d)
    regions, saturation = {}, {}
    for j, identity in enumerate(query_ids):
        regions[identity] = protected_region_accuracy(query_regions[j],
            methods['learned']['tables'], cases['oracle_tables'])
        saturation[identity] = saturation_excess(query_counts[j],
            methods['learned']['tables'], cases['oracle_tables'])
    numeric_passed = (photometry['photometry_passed'] and controls['passed']
        and all(r['passed'] for r in regions.values())
        and all(r['passed'] for r in saturation.values()))
    return {'photometry': photometry, 'controls': controls, 'regions': regions,
            'saturation': saturation, 'numeric_passed': numeric_passed,
            'visual_review_required': True,
            'limits': 'Numerical gates do not establish visual comfort or preservation of semantic facts.'}
