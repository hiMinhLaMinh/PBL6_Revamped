import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

import extract_raw as raw
import hand_association as association
from test_extract_raw import results, arrays_for_test


def scene(xs, visibility=1.):
    c = np.zeros((len(xs),21,3),np.float32)
    for i,x in enumerate(xs): c[i,:,0]=x;c[i,:,1]=.5
    pose = np.zeros((8,3),np.float32);pose[4,:2]=[.2,.5];pose[5,:2]=[.8,.5]
    return c,c.copy(),pose,np.full(8,visibility,np.float32)


def run(xs, state=None, frame=0, visibility=1.):
    return association.associate_frame(*scene(xs,visibility),state or [None,None],frame,1.,association.load_policy())


def test_third_candidate_is_considered_and_two_sides_unique():
    result,_=run([.2,.5,.8])
    assert result['association_indices'].tolist()==[0,2]
    assert result['association_valid'].tolist()==[True,True]


def test_order_swap_does_not_swap_anatomical_sides():
    a,state=run([.2,.8])
    b,_=run([.8,.2],state,1)
    assert b['association_indices'].tolist()==[1,0]
    np.testing.assert_array_equal(a['associated_hand_xyz'],b['associated_hand_xyz'])


@pytest.mark.parametrize('xs,expected', [([],[-1,-1]),([.2],[0,-1]),([.8],[-1,0]),([.5],[-1,-1])])
def test_zero_one_or_far_candidates_allow_unmatched(xs,expected):
    result,_=run(xs)
    assert result['association_indices'].tolist()==expected


def test_duplicate_near_left_is_ambiguous_not_forced():
    result,_=run([.2,.201,.8])
    assert result['association_indices'].tolist()==[-1,2]
    assert result['association_method'][0]=='ambiguous'
    assert np.isnan(result['associated_hand_xyz'][0]).all()


def test_crossed_pose_wrists_with_equal_cost_abstains():
    c,w,p,v=scene([.2,.8]);p[4,:2]=p[5,:2]=[.5,.5]
    policy=association.load_policy();policy['pose_max_distance']=.4
    result,_=association.associate_frame(c,w,p,v,[None,None],0,1.,policy)
    assert result['association_indices'].tolist()==[-1,-1]
    assert list(result['association_method'])==['ambiguous','ambiguous']


def test_tracking_is_uncertain_and_expires_without_pose_confirmation():
    _,state=run([.2,.8])
    tracked,state=run([.21,.79],state,1,visibility=0.)
    assert list(tracked['association_method'])==['tracking','tracking']
    assert not tracked['association_valid'].any()
    expired,_=run([.22,.78],state,6,visibility=0.)
    assert expired['association_indices'].tolist()==[-1,-1]


def test_one_pose_wrist_missing_does_not_invent_opposite_side():
    c,w,p,v=scene([.2,.8]);v[5]=0
    result,_=association.associate_frame(c,w,p,v,[None,None],0,1.,association.load_policy())
    assert result['association_indices'].tolist()==[0,-1]


def test_aspect_correction_changes_distance_gate():
    c,w,p,v=scene([.2]);c[0,:,1]=.6
    a,_=association.associate_frame(c,w,p,v,[None,None],0,1.,association.load_policy())
    b,_=association.associate_frame(c,w,p,v,[None,None],0,2.,association.load_policy())
    assert a['association_indices'][0]==0 and b['association_indices'][0]==-1


def test_pack_preserves_every_candidate_including_overflow():
    pose,hands=results()
    third=NS(landmark=[NS(x=.9,y=.5,z=0.) for _ in range(21)])
    hands.multi_hand_landmarks.append(third)
    frame=raw.pack_frame(pose,hands)
    assert frame['hand_candidate_xyz'].shape==(3,21,3)
    assert np.isnan(frame['raw_hand_xyz']).all()  # no arbitrary first-two truncation
    assert frame['hand_candidate_xyz'][2,0,0]==np.float32(.9)
    a=raw.assemble_frames([frame]);a.update(frame_indices=np.array([0],np.int32),
        timestamps=np.array([0.],np.float64),decoder_timestamps=np.array([0.],np.float64))
    raw.validate_arrays(a)
    assert a['hand_candidate_offsets'].tolist()==[0,3]


def test_roundtrip_variable_counts_and_invalid_mapping_rejected(tmp_path):
    a=arrays_for_test(); meta=dict(schema=raw.SCHEMA,stage='detector_raw',decoded_frames=2)
    path=tmp_path/'raw.npz';raw.save_raw(path,a,meta)
    loaded,_=raw.load_raw(path)
    np.testing.assert_array_equal(loaded['hand_candidate_offsets'],a['hand_candidate_offsets'])
    loaded['association_indices'][0,0]=100
    with pytest.raises(ValueError,match='index out'): raw.validate_arrays(loaded)


def test_offline_replay_does_not_change_raw_and_needs_no_mediapipe(tmp_path):
    a=arrays_for_test(); path=tmp_path/'raw.npz'; output=tmp_path/'associated.npz'
    raw.save_raw(path,a,dict(schema=raw.SCHEMA,stage='detector_raw',decoded_frames=2,height=1080,width=1080))
    before=raw.sha256(path)
    result=subprocess.run([sys.executable,str(Path(association.__file__)),str(path),'--output',str(output)],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert raw.sha256(path)==before
    with np.load(output,allow_pickle=False) as data:
        meta=json.loads(data['meta_json'].item())
        assert meta['source_npz_sha256']==before
        np.testing.assert_array_equal(data['association_indices'],a['association_indices'])


def test_legacy_v2_cache_stays_readable_but_has_no_candidates(tmp_path):
    a=arrays_for_test()
    for key in (*association.CANDIDATE_FIELDS,*association.ASSOCIATION_SHAPES,'hand_candidate_offsets'):
        a.pop(key)
    path=tmp_path/'old.npz'
    np.savez_compressed(path,**a,meta_json=np.array(json.dumps(dict(schema='vsl400.detector_raw.v2',stage='detector_raw',decoded_frames=2))))
    assert raw.load_raw(path)[0]['raw_pose_xyz'].shape==(2,8,3)


def test_variable_detection_counts_have_lossless_offsets():
    frames=[]
    for n in (3,0,2):
        pose,hands=results()
        hands.multi_hand_landmarks=[hands.multi_hand_landmarks[0]]*n
        hands.multi_hand_world_landmarks=[];hands.multi_handedness=[]
        frames.append(raw.pack_frame(pose,hands))
    a=raw.assemble_frames(frames)
    a.update(frame_indices=np.arange(3,dtype=np.int32),timestamps=np.arange(3,dtype=np.float64),
             decoder_timestamps=np.arange(3,dtype=np.float64))
    raw.validate_arrays(a)
    assert a['hand_candidate_offsets'].tolist()==[0,3,3,5]
    assert len(a['hand_candidate_xyz'])==5
    a['hand_candidate_offsets'][-1]=4
    with pytest.raises(ValueError): raw.validate_arrays(a)


def test_new_policy_can_change_association_without_changing_candidates():
    a=arrays_for_test();before={k:v.copy() for k,v in a.items()}
    policy=association.load_policy();policy['pose_visibility_min']=0.;policy['ambiguity_margin']=0.
    alternate=association.associate_sequence(a,policy,1.)
    assert np.any(alternate['association_indices']!=a['association_indices'])
    for key in a: np.testing.assert_array_equal(a[key],before[key])
