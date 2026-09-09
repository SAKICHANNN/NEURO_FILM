import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
spec = importlib.util.spec_from_file_location('staged_review_test', ROOT/'scripts/prepare_tst_crossed_photo_review_v2.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def record(stage='portrait_face'):
    return {'operator_key': 'key', 'stage': stage, 'probe_id': None if stage == 'donor' else 'probe', 'artifact': {'sha256': 'a'*64}}


def review(stage='portrait_face', decision='PASS'):
    return {'operator_key': 'key', 'stage': stage, 'stage_record_sha256': 'b'*64,
            'probe_id': None if stage == 'donor' else 'probe', 'reviewed_artifact_hashes': ['a'*64],
            'actual_viewed': True, 'reviewer_kind': 'test_agent', 'decision': decision, 'reason': 'synthetic label',
            'comfortable': True, 'visibly_changed_comfortably_rich': True, 'objectionable_new_content_or_detail_loss': False,
            'pair_content_geometry_compatible': True, 'donor_reconstruction_appearance_compatible': True}


def test_stable_round_robin_and_imports():
    rows = [{'operator_key': k, 'split': s} for s, keys in [('constructed_test',['c0','c1','c2']),('fit',['f0','f1']),('monitor',['m0'])] for k in keys]
    assert m.ordered_queue(rows, {'c0': {}}) == ['f0','m0','c1','f1','c2']
    with pytest.raises(m.old.s2.IdentityFailure):
        m.ordered_queue(rows+rows[:1], {})


@pytest.mark.parametrize('mutation', [
    {'stage_record_sha256':'wrong'}, {'probe_id':'wrong'}, {'stage':'donor'}, {'operator_key':'wrong'},
    {'reviewed_artifact_hashes':[]}, {'reviewed_artifact_hashes':['a'*64]*2},
    {'reviewed_artifact_hashes':['c'*64]}, {'actual_viewed':False}, {'reason':''}, {'training_admitted':True},
    {'decision':'FAIL'}, {'comfortable':None}, {'comfortable':1}, {'decision':'AUTOPASS'}])
def test_ingest_rejects_missing_fabricated_conflicting_evidence(mutation):
    value={**review(), **mutation}
    with pytest.raises(m.old.s2.IdentityFailure):
        m.validate_observation(value, record(), 'b'*64)


def test_uncertainty_distinct_and_no_false_failure():
    v=review(decision='UNCERTAIN');v.update(comfortable=None,visibly_changed_comfortably_rich=None)
    m.validate_observation(v,record(),'b'*64)
    assert m.outcome([v])=='UNCERTAIN_NOT_ADMITTED'
    v['decision']='FAIL'
    with pytest.raises(m.old.s2.IdentityFailure):m.validate_observation(v,record(),'b'*64)


def test_exact_conjunction_short_circuit_and_full_pass():
    import itertools
    for comfort in itertools.product([False,True],repeat=4):
        full=[];p=0
        for stage in m.STAGES:
            v=review(stage)
            if stage!='donor':
                v.update(comfortable=comfort[p],visibly_changed_comfortably_rich=comfort[p],decision='PASS' if comfort[p] else 'FAIL');p+=1
            full.append(v)
        prefix=[]
        for v in full:
            m.validate_observation(v,record(v['stage']),'b'*64);prefix.append(v)
            if v['decision']=='FAIL':break
        result=m.outcome(prefix)
        assert (result=='PASS_CURATOR_ONLY')==all(comfort)
    full=[review(s) for s in m.STAGES]
    assert all(m.outcome(full[:n])=='PENDING' for n in range(5))
    full[0]['visibly_changed_comfortably_rich']=False
    assert m.outcome(full)=='PASS_CURATOR_ONLY'
    full[2]['visibly_changed_comfortably_rich']=False
    assert m.outcome(full)=='FAIL_INSUFFICIENT_COMFORTABLE_RICH_CHANGE'
    with pytest.raises(m.old.s2.IdentityFailure):m.outcome([full[0],full[0]])


def test_real_file_hash_and_exclusive_review_write(tmp_path):
    p=tmp_path/'sheet.png';p.write_bytes(b'actual')
    a={'path':'sheet.png','bytes':6,'sha256':hashlib.sha256(b'actual').hexdigest()}
    m.verify_artifact(tmp_path,a)
    p.write_bytes(b'forged')
    with pytest.raises(m.old.s2.IdentityFailure):m.verify_artifact(tmp_path,a)
    m.create_json(tmp_path/'review.json',{'test':1})
    with pytest.raises(FileExistsError):m.create_json(tmp_path/'review.json',{'test':2})


def test_manifest_row_to_frozen_loader_integration_and_independent_pixels(monkeypatch):
    source=np.random.default_rng(48).integers(0,65536,(21,25,3),dtype=np.uint16)
    probe={'component':'probe','split':'fit','category':'portrait_face','representative':'synthetic.png','crops':[{'box_xyxy_exclusive':[2,3,13,17]}]}
    expanded={'probe_id':'probe','source':'synthetic.png','encoded':{'sha256':'encoded'},'header_path':'header.json','header_sha256':'header','shape':list(source.shape),'canonical_sha256':'canonical','color_policy':'synthetic','split':'fit'}
    monkeypatch.setattr(m.old.s3,'probe_metadata',lambda cfg:[expanded])
    monkeypatch.setattr(m.old.s2,'verify_encoded',lambda p,e:None)
    monkeypatch.setattr(m.old.s2,'verified_header',lambda p,h:{})
    monkeypatch.setattr(m.old.s2.corpus,'confined',lambda p,n:Path(n))
    monkeypatch.setattr(m.old.s2.corpus,'decode',lambda p,h:(source,'synthetic'))
    monkeypatch.setattr(m.old.s2.corpus,'canonical_hash',lambda a:'canonical')
    axis=np.linspace(0,1,7);mesh=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1)
    grid=(mesh**.8+.06).reshape(343,3)
    cfg={'originals':'unused','chunk_pixels':32768,'overview_side':384}
    board,layout,boxes=m.stage_panel(cfg,{'split':'fit'},'portrait_face',probe,grid)
    rgb=source.astype(np.float64)/65535
    expected=np.rint(np.clip(RegularGridInterpolator((axis,)*3,grid.reshape(7,7,7,3))(rgb),0,1)*65535).astype(np.uint16)
    np.testing.assert_array_equal(m.old.render_native(source,grid,32768),expected)
    for row in layout:
        x0,y0,x1,y1=row['source_xyxy'];bx0,by0,bx1,by1=row['board_xyxy'];a=source if row['column']==0 else expected
        display=((a[y0:y1,x0:x1].astype(np.uint32)+128)//257).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(board)[by0:by1,bx0:bx1],display)
    original_board,old_layout=m.old.panel([source,expected],['Original source','F(source) strength1'],boxes,'portrait_face | synthetic.png',384)
    assert board.tobytes()==original_board.tobytes() and layout==old_layout


def test_constant_prefix_cursor_and_historical_corruption_audit_boundary(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    m.create_json(tmp_path/'probes.json',{'rows':[{'split':'fit','category':s,'component':s} for s in m.STAGES if s!='donor']})
    cfg={'output':'out','probe_manifest':'probes.json'}; identity={'pin':'x'}
    inputs={'queue':['old','next'],'rows':[{'operator_key':'old','split':'fit'},{'operator_key':'next','split':'fit'}]}
    m.create_json(tmp_path/'out/terminal/old.json',{'identity':identity,'next_index':1})
    m.create_json(tmp_path/'out/cursor.json',{'index':1,'previous_terminal_sha256':m.digest(tmp_path/'out/terminal/old.json')})
    state,_,_=m.cursor(cfg,inputs,identity)
    assert state['operator_key']=='next' and state['status']=='READY_TO_RENDER'
    # Historical sheets are deliberately not rehashed during scheduling; final audit must do so.
    m.create_json(tmp_path/'out/terminal/unrelated.json',{'not_current':True})
    assert m.cursor(cfg,inputs,identity)[0]==state
    (tmp_path/'out/terminal/old.json').write_text('{}')
    with pytest.raises(m.old.s2.IdentityFailure):m.cursor(cfg,inputs,identity)


def test_cumulative_budget_keeps_prior_photo_and_external(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m.old.s3,'ledger',lambda cfg:{'study_cpu':100.,'study_wall':200.})
    monkeypatch.setattr(m.old.s3,'attempts_accounting',lambda p:[{'aggregate_cpu_seconds':10.,'active_wall_seconds':20.}])
    cfg={'output':'out','legacy_output':'old','external_photo_cpu_seconds':8.,'external_photo_wall_reserve_seconds':15.,'legacy_candidate_bytes':500}
    result=m.costs(cfg)
    assert result['photo_cpu']==18 and result['photo_wall']==35
    assert result['study_cpu']==108 and result['study_wall']==215 and result['retained_bytes']==500


def test_terminal_chain_full_audit_catches_historical_image_change(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    cfg={'output':'out','probe_manifest':'probes.json'};identity={'pin':'x'}
    m.create_json(tmp_path/'probes.json',{'rows':[{'split':'fit','category':'portrait_face','component':'probe'}]})
    directory=tmp_path/'out/candidate/key/portrait_face';directory.mkdir(parents=True)
    (directory/'sheet.png').write_bytes(b'pixels')
    rec={**record(),'identity':identity,'artifact':{'path':'sheet.png','sha256':m.digest(directory/'sheet.png'),'bytes':6}}
    m.create_json(directory/'record.json',rec)
    v=review(decision='FAIL');v.update(comfortable=False,visibly_changed_comfortably_rich=False,stage_record_sha256=m.digest(directory/'record.json'),reviewed_artifact_hashes=[rec['artifact']['sha256']])
    m.validate_observation(v,rec,v['stage_record_sha256']);m.create_json(directory/'review.json',v)
    m.commit_terminal_if_needed(cfg,identity,'key',0)
    size=sum(p.stat().st_size for p in directory.iterdir())
    acc={'previous_accounting_sha256':None,'failure':None,'aggregate_cpu_seconds':1.,'active_wall_seconds':2.,'added_candidate_bytes':size}
    m.create_json(tmp_path/'out/attempts/0001/accounting.json',acc)
    checkpoint={'attempts':1,'cpu':1.,'wall':2.,'bytes':size}
    monkeypatch.setattr(m,'costs',lambda cfg,active=None:{'checkpoint':checkpoint})
    inputs={'queue':['key'],'rows':[{'operator_key':'key','split':'fit'}],'population_keys':['key'],'imported':{}}
    assert m.audit_history(cfg,inputs,identity)['status']=='FULL_HISTORY_INTEGRITY_VERIFIED_NOT_TRAINING'
    (directory/'sheet.png').write_bytes(b'forged')
    with pytest.raises(m.old.s2.IdentityFailure):m.audit_history(cfg,inputs,identity)
