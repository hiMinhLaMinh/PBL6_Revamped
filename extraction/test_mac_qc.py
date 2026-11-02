import json
from pathlib import Path
import sys

import numpy as np
import pytest

import quality_control as qc
from run_front_batch import plan_parts


def arrays(hands=(1,1,1,1)):
    hand = np.zeros((4,2,21,3),np.float32)
    for i,n in enumerate(hands): hand[i,n:]=np.nan
    return dict(raw_pose_xyz=np.zeros((4,8,3),np.float32),raw_hand_xyz=hand,
                pose_finite=np.ones((4,8),bool),hand_finite=np.isfinite(hand).all(-1),
                raw_pose_visibility=np.ones((4,8),np.float32))


def meta():
    return dict(decoded_frames=4,reported_frames=4,completion_status='complete',
                official_metadata=dict(num_frames=4,fps=25),fps=25,warnings=[],
                timestamp_source='opencv_pos_msec')


def test_one_hand_sign_not_automatically_rejected():
    result=qc.metrics(arrays(),meta(),qc.DEFAULT_POLICY)
    assert result['one_hand_fraction']==1 and result['two_hands_fraction']==0
    assert result['qc_status']=='not_flagged'


def test_missing_hand_run_and_oob_denominator():
    a=arrays((1,0,0,2))
    a['raw_pose_xyz'][0,0,0]=1.5
    result=qc.metrics(a,meta(),qc.DEFAULT_POLICY)
    assert result['longest_no_hand_run_frames']==2
    assert result['no_hand_fraction']==0.5
    # 32 observed pose + 63 hand points, NaNs are excluded from the denominator.
    assert result['xy_outside_observed_fraction']==pytest.approx(1/95)
    assert 'frequent_no_hand' in result['review_flags']


def test_qc_never_mutates_input_or_fills_nan():
    a=arrays((0,1,2,0)); before={k:v.copy() for k,v in a.items()}
    qc.metrics(a,meta(),qc.DEFAULT_POLICY)
    for k in a: np.testing.assert_array_equal(a[k],before[k])


def test_incomplete_and_timestamp_fallback_are_reviewed():
    m=meta();m['completion_status']='incomplete';m['timestamp_source']='frame_index_over_fps'
    result=qc.metrics(arrays(),m,qc.DEFAULT_POLICY)
    assert not result['decode_complete']
    assert result['review_flags']==['decode_incomplete_or_unverified','timestamp_fps_fallback']


def test_part_commands_are_front_only_and_write_separate_ledgers(tmp_path):
    for name in ('Part_1','Part_2'):
        (tmp_path/'input'/name/'front_view').mkdir(parents=True)
    plans=plan_parts(tmp_path/'input',tmp_path/'out',Path('config.json'),2)
    assert len(plans)==2
    assert plans[0][1][plans[0][1].index('--view')+1]=='front'
    outputs=[command[command.index('--output')+1] for _,command in plans]
    assert len(set(outputs))==2 and outputs[0].endswith('Part_1')


def test_coverage_detects_missing_clips_and_keeps_raw_unchanged(tmp_path,monkeypatch):
    inp=tmp_path/'raw';inp.mkdir();(inp/'a.npz').write_bytes(b'raw')
    dataset=tmp_path/'dataset/Part_1';dataset.mkdir(parents=True)
    records=[dict(video_id=f'{i:06}',signer_id='001',gloss='Anh',num_frames=4,fps=25) for i in range(2)]
    (dataset/'front_view.json').write_text(json.dumps(records))
    m=dict(meta(),source_part='Part_1',view='front',video_id='000000',signer_id='001',gloss='Anh',official_metadata=records[0])
    monkeypatch.setattr(qc,'load_raw',lambda _: (arrays(),m))
    policy=tmp_path/'policy.json';policy.write_text(json.dumps(qc.DEFAULT_POLICY))
    out=tmp_path/'qc'
    monkeypatch.setattr(sys,'argv',['quality_control.py',str(inp),'--output',str(out),
        '--policy',str(policy),'--dataset-root',str(dataset.parent)])
    qc.main()
    summary=json.loads((out/'qc_summary.json').read_text())
    assert summary['coverage']['missing']==[['Part_1','front','000001']]
    assert (inp/'a.npz').read_bytes()==b'raw'


def test_qc_counts_failed_and_interrupted_clips_without_npz(tmp_path):
    from batch_io import manifest_row
    ledger=tmp_path/'manifest.csv'
    for path,status,time in [('a.mp4','started','01'),('a.mp4','complete','02'),
                             ('b.mp4','started','01'),('b.mp4','failed','02'),
                             ('c.mp4','started','01')]:
        manifest_row(ledger,dict(source_path=path,status=status,timestamp_utc=time))
    counts,failures=qc.audit_ledgers(tmp_path)
    assert counts=={'complete':1,'failed':1,'started':1}
    assert len(failures)==2


def test_v3_overflow_qc_uses_all_candidates_not_empty_legacy_slots():
    a=arrays((0,0,0,0))
    a['hand_candidate_offsets']=np.array([0,3,6,9,12],np.int64)
    a['hand_candidate_xyz']=np.zeros((12,21,3),np.float32)
    a['hand_candidate_xyz'][2,0,0]=1.5
    a['association_method']=np.array([['pose','ambiguous']]*4)
    a['association_indices']=np.array([[0,-1]]*4,np.int32)
    a['association_valid']=np.array([[True,False]]*4)
    a['associated_hand_observed']=np.zeros((4,2,21),bool)
    result=qc.metrics(a,meta(),qc.DEFAULT_POLICY)
    assert result['no_hand_fraction']==0
    assert result['hand_candidate_max_count']==3
    assert result['hand_candidate_overflow_fraction']==1
    assert result['xy_outside_observed_fraction']==pytest.approx(1/(32+12*21))
    assert result['association_ambiguous_frame_fraction']==1
    assert result['association_unassigned_side_fraction']==.5
    assert 'frequent_no_hand' not in result['review_flags']
    assert 'hand_candidate_overflow' in result['review_flags']
