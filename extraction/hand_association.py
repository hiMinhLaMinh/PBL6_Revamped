"""Conservative N-candidate to two anatomical sides, with reversible provenance."""
import argparse
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np

DEFAULT_POLICY = Path(__file__).with_name('association_policy.json')
CANDIDATE_FIELDS = ('hand_candidate_xyz', 'hand_candidate_world_xyz',
                    'hand_candidate_label', 'hand_candidate_score')
ASSOCIATION_SHAPES = {
    'associated_hand_xyz': (2,21,3), 'associated_hand_world_xyz': (2,21,3),
    'associated_hand_observed': (2,21), 'association_indices': (2,),
    'association_proposed_indices': (2,), 'association_method': (2,),
    'association_cost': (2,), 'association_margin': (2,), 'association_valid': (2,),
}


def load_policy(path=DEFAULT_POLICY):
    policy = json.loads(Path(path).read_text())
    keys = {'version','pose_visibility_min','pose_max_distance','tracking_max_distance',
            'ambiguity_margin','max_tracking_gap_frames'}
    if set(policy) != keys or policy['version'] != 'pose_tracking_v1':
        raise ValueError('Unknown association policy keys/version')
    for key in keys - {'version','max_tracking_gap_frames'}:
        x = policy[key]
        if type(x) not in (int,float) or not np.isfinite(x) or not 0 <= x <= 1:
            raise ValueError(f'Invalid association policy: {key}')
    if min(policy['pose_max_distance'],policy['tracking_max_distance']) <= 0:
        raise ValueError('Distance thresholds must be positive')
    if type(policy['max_tracking_gap_frames']) is not int or policy['max_tracking_gap_frames'] < 0:
        raise ValueError('Invalid tracking gap')
    return policy


def associate_frame(candidates, worlds, pose, visibility, state, frame_index, aspect, policy):
    """Minimize two-side assignment cost, including unmatched dummy choices.

    Distances are in image-width units. Tracking remains uncertain and expires
    relative to the last pose confirmation, not the last tracked frame.
    """
    n = len(candidates)
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError('Aspect height/width must be positive')
    costs = np.full((2,n),np.inf)
    methods = ['none','none']
    limits = [0.,0.]
    scale = np.array([1.,aspect])
    for side, wrist in enumerate((4,5)):  # indices in the selected eight Pose joints
        if np.isfinite(pose[wrist,:2]).all() and np.isfinite(visibility[wrist]) and visibility[wrist] >= policy['pose_visibility_min']:
            anchor, methods[side], limits[side] = pose[wrist,:2], 'pose', policy['pose_max_distance']
        elif state[side] is not None and frame_index-state[side]['confirmed_frame'] <= policy['max_tracking_gap_frames']:
            anchor, methods[side], limits[side] = state[side]['xy'], 'tracking', policy['tracking_max_distance']
        else:
            continue
        distances = np.linalg.norm((candidates[:,0,:2]-anchor)*scale,axis=-1)
        costs[side] = np.where(np.isfinite(distances) & (distances <= limits[side]),distances,np.inf)
    # At most two candidates selected; all N candidates compete, not just indices 0/1.
    feasible = []
    for left,right in itertools.product(range(-1,n),repeat=2):
        if left >= 0 and left == right:
            continue
        pair = (left,right)
        value = sum(limits[side] if idx < 0 else costs[side,idx] for side,idx in enumerate(pair))
        if np.isfinite(value): feasible.append((float(value),pair))
    feasible.sort(key=lambda x:(x[0],x[1]))
    best, pair = feasible[0]
    result = dict(associated_hand_xyz=np.full((2,21,3),np.nan,np.float32),
        associated_hand_world_xyz=np.full((2,21,3),np.nan,np.float32),
        association_indices=np.full(2,-1,np.int32),association_proposed_indices=np.array(pair,np.int32),
        association_method=np.full(2,'none',dtype='<U12'),association_cost=np.full(2,np.nan,np.float32),
        association_margin=np.full(2,np.nan,np.float32),association_valid=np.zeros(2,bool))
    next_state = list(state)
    for side,idx in enumerate(pair):
        alternatives = [value for value,other in feasible if other[side] != idx]
        margin = min(alternatives)-best if alternatives else float('inf')
        if np.isfinite(margin): result['association_margin'][side] = max(0.,margin)
        if methods[side] != 'none' and margin < policy['ambiguity_margin']:
            result['association_method'][side] = 'ambiguous'
            continue
        if idx < 0:
            continue
        result['association_indices'][side] = idx
        result['association_method'][side] = methods[side]
        result['association_cost'][side] = costs[side,idx]
        result['association_valid'][side] = methods[side] == 'pose'
        result['associated_hand_xyz'][side] = candidates[idx]
        result['associated_hand_world_xyz'][side] = worlds[idx]
        confirmed = frame_index if methods[side] == 'pose' else state[side]['confirmed_frame']
        next_state[side] = dict(xy=candidates[idx,0,:2].copy(),confirmed_frame=confirmed)
    result['associated_hand_observed'] = np.isfinite(result['associated_hand_xyz']).all(-1)
    return result,next_state


def associate_sequence(arrays, policy, aspect):
    state = [None,None]; rows = []
    offsets = arrays['hand_candidate_offsets']
    for i in range(len(arrays['raw_pose_xyz'])):
        start,end = offsets[i:i+2]
        result,state = associate_frame(arrays['hand_candidate_xyz'][start:end],
            arrays['hand_candidate_world_xyz'][start:end],arrays['raw_pose_xyz'][i],
            arrays['raw_pose_visibility'][i],state,i,aspect,policy)
        rows.append(result)
    return {key:np.stack([row[key] for row in rows]) for key in ASSOCIATION_SHAPES}


def main():
    parser = argparse.ArgumentParser(description='Re-run association from a v3 NPZ, without MediaPipe or modifying raw')
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True,help='New derived NPZ file, not the raw input')
    parser.add_argument('--policy',type=Path,default=DEFAULT_POLICY)
    args = parser.parse_args()
    from extract_raw import load_raw, sha256
    source, target = args.input.resolve(),args.output.resolve()
    if source == target or target.exists():
        parser.error('Output must be a new file distinct from raw')
    arrays,meta = load_raw(source)
    if 'hand_candidate_offsets' not in arrays:
        parser.error('Legacy v1/v2 lacks all candidates; re-extract video to create v3')
    policy = load_policy(args.policy)
    result = associate_sequence(arrays,policy,meta['height']/meta['width'])
    result['frame_indices'] = arrays['frame_indices']
    result['meta_json'] = np.array(json.dumps(dict(schema='vsl400.hand_association.v1',
        source_npz=str(source),source_npz_sha256=sha256(source),policy=policy,
        association_code_sha256=sha256(__file__),side_order=['left','right']),ensure_ascii=False,allow_nan=False))
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,suffix='.npz',delete=False) as stream:
            temporary = Path(stream.name); np.savez_compressed(stream,**result)
        temporary.replace(target)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    print(f'Association saved: {target}; raw unchanged')


if __name__ == '__main__': main()
