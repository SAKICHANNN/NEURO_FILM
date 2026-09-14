import pytest

from src.eval.fable_cdfe_assignment import validate_cdfe_assignment, REQUIRED_COVERAGE


def fixture():
    rows, assignment = [], {'held_cameras': ['H0', 'H1', 'H2', 'H3'], 'held_assessment': {}}
    for role, count in [('fit', 256), ('validation', 32), ('represented_assessment', 16), ('queries', 4)]:
        assignment[role] = []
        for n in range(count):
            identity = f'{role}{n}'
            assignment[role].append(identity)
            rows.append({'identity': identity, 'camera': 'A', 'component': 'development' if role == 'fit' else identity,
                         'exposure': 'SUBSTANTIVE_DEVELOPMENT' if role == 'fit' else 'ADMISSION_ONLY_NOT_YET_FRESH_CERTIFIED',
                         'technical_eligible': True, 'history_separation_established': True,
                         'query_coverage_reviewed': role == 'queries',
                         'query_coverage': sorted(REQUIRED_COVERAGE) if role == 'queries' else []})
    for camera in assignment['held_cameras']:
        assignment['held_assessment'][camera] = []
        for n in range(4):
            identity = f'{camera}-{n}'
            assignment['held_assessment'][camera].append(identity)
            rows.append({'identity': identity, 'camera': camera, 'component': identity,
                         'exposure': 'ADMISSION_ONLY_NOT_YET_FRESH_CERTIFIED', 'technical_eligible': True,
                         'history_separation_established': True, 'query_coverage': []})
    return rows, assignment


def test_consumed_fitting_can_share_component():
    rows, assignment = fixture()
    assert validate_cdfe_assignment(rows, assignment, [])['fit_components'] == 1


@pytest.mark.parametrize('defect', ['history_link', 'fresh_overlap', 'unknown_exposure', 'unseen_query', 'missing_coverage', 'not_technical', 'not_separated', 'held_smoke', 'single_rep'])
def test_rejects_protocol_violations(defect):
    rows, assignment = fixture()
    indexed = {r['identity']: r for r in rows}
    smoke = []
    if defect == 'history_link': indexed['queries0']['component'] = 'development'
    if defect == 'fresh_overlap': indexed['queries0']['component'] = 'validation0'
    if defect == 'unknown_exposure': indexed['queries0']['exposure'] = 'UNKNOWN_HISTORY_NOT_FRESH'
    if defect == 'unseen_query': indexed['queries0']['camera'] = 'B'
    if defect == 'missing_coverage':
        for i in assignment['queries']: indexed[i]['query_coverage'] = ['face']
    if defect == 'not_technical': indexed['fit0']['technical_eligible'] = False
    if defect == 'not_separated': indexed['validation0']['history_separation_established'] = False
    if defect == 'held_smoke':
        rows.append(dict(indexed['fit0'], identity='smoke', camera='H0')); smoke = ['smoke']
    if defect == 'single_rep':
        indexed['fit0']['camera'] = 'B'; indexed['represented_assessment0']['camera'] = 'B'
    with pytest.raises(ValueError): validate_cdfe_assignment(rows, assignment, smoke)
