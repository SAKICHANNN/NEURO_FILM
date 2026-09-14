from collections import Counter


REQUIRED_COVERAGE = {'text', 'face', 'object', 'shadow', 'highlight', 'colored_illumination'}


def validate_cdfe_assignment(rows: list[dict], assignment: dict, smoke_ids: list[str]) -> dict:
    indexed = {row['identity']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate inventory identity')
    counts = {'fit': 256, 'validation': 32, 'represented_assessment': 16, 'queries': 4}
    for role, count in counts.items():
        if len(assignment[role]) != count:
            raise ValueError(f'wrong {role} count')
    held = assignment['held_assessment']
    if len(held) != 4 or any(len(ids) != 4 for ids in held.values()):
        raise ValueError('four held cameras with four donors each required')
    if set(assignment['held_cameras']) != set(held) or len(assignment['held_cameras']) != 4:
        raise ValueError('held camera declaration mismatch')
    selected = [identity for role in counts for identity in assignment[role]] + [i for ids in held.values() for i in ids]
    if len(set(selected)) != 324 or not set(selected + smoke_ids) <= indexed.keys():
        raise ValueError('roles overlap or identity absent from inventory')
    if any(indexed[i]['technical_eligible'] is not True for i in selected):
        raise ValueError('selected source lacks technical eligibility')
    if any(indexed[i]['query_coverage_reviewed'] is not True for i in assignment['queries']):
        raise ValueError('query coverage has not been actually reviewed')
    fit = assignment['fit']
    if any(indexed[i]['exposure'] != 'SUBSTANTIVE_DEVELOPMENT' for i in fit + smoke_ids):
        raise ValueError('fitting and smoke require substantive consumed captures')
    forbidden = {r['component'] for r in rows if r['exposure'] != 'ADMISSION_ONLY_NOT_YET_FRESH_CERTIFIED'}
    fresh = assignment['validation'] + assignment['represented_assessment'] + assignment['queries'] + [i for ids in held.values() for i in ids]
    if any(indexed[i]['exposure'] != 'ADMISSION_ONLY_NOT_YET_FRESH_CERTIFIED' or
           indexed[i]['history_separation_established'] is not True or
           indexed[i]['component'] in forbidden for i in fresh):
        raise ValueError('fresh source touches history or lacks separation evidence')
    if len({indexed[i]['component'] for i in fresh}) != 68:
        raise ValueError('fresh roles share acquisition components')
    fit_cameras = {indexed[i]['camera'] for i in fit}
    represented = assignment['validation'] + assignment['represented_assessment'] + assignment['queries']
    if any(indexed[i]['camera'] not in fit_cameras for i in represented):
        raise ValueError('validation assessment and queries must be fitting-represented')
    rep_counts = Counter(indexed[i]['camera'] for i in assignment['represented_assessment'])
    if min(rep_counts.values()) < 2:
        raise ValueError('represented-camera shuffle requires at least two donors per camera')
    seen = fit_cameras | {indexed[i]['camera'] for i in represented + smoke_ids}
    if set(held) & seen:
        raise ValueError('held camera exposed to fitting smoke validation or queries')
    if any(indexed[i]['camera'] != camera for camera, ids in held.items() for i in ids):
        raise ValueError('held donor camera mismatch')
    coverage = set().union(*(set(indexed[i]['query_coverage']) for i in assignment['queries']))
    if not REQUIRED_COVERAGE <= coverage:
        raise ValueError('query content coverage incomplete')
    return {'status': 'ROLE_CONSTRAINTS_PASS_CONDITIONAL_ON_BOUND_INVENTORY',
            'fit_captures': 256, 'fit_components': len({indexed[i]['component'] for i in fit}),
            'fresh_components': 68, 'query_coverage': sorted(coverage),
            'limits': 'Certificate checks supplied evidence flags and components; it does not independently establish provenance or native ROI readability.'}
