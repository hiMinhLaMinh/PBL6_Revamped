"""Read-only raw landmark QC. Review flags are heuristics, never automatic exclusions."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np

from batch_io import completion, FIELDS as MANIFEST_FIELDS
from extract_raw import load_raw, sha256

DEFAULT_POLICY = dict(pose_visibility_cutoff=0.5, review_pose_observed_below=0.9,
                      review_no_hand_above=0.2, review_xy_outside_above=0.05)


def longest_run(mask):
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def metrics(arrays, metadata, policy):
    pose = arrays['pose_finite']
    hand = arrays['hand_finite']
    hand_count = np.diff(arrays['hand_candidate_offsets']) if 'hand_candidate_offsets' in arrays else hand.all(-1).sum(-1)
    candidates = arrays.get('hand_candidate_xyz', arrays['raw_hand_xyz'])
    xyz = np.concatenate([arrays['raw_pose_xyz'].reshape(-1,3),candidates.reshape(-1,3)],axis=0)
    finite = np.isfinite(xyz).all(-1)
    outside = ((xyz[..., :2] < 0) | (xyz[..., :2] > 1)).any(-1) & finite
    outside_fraction = float(outside.sum()/finite.sum()) if finite.any() else None
    vis = arrays['raw_pose_visibility']
    vis_valid = np.isfinite(vis)
    low_vis = float((vis[vis_valid] < policy['pose_visibility_cutoff']).mean()) if vis_valid.any() else None
    complete, completeness_warnings = completion(metadata, metadata.get('official_metadata'))
    integrity = metadata.get('completion_status') == 'complete' and complete
    flags = []
    if not integrity: flags.append('decode_incomplete_or_unverified')
    if pose.mean() < policy['review_pose_observed_below']: flags.append('low_pose_observation')
    if (hand_count == 0).mean() > policy['review_no_hand_above']: flags.append('frequent_no_hand')
    if outside_fraction is not None and outside_fraction > policy['review_xy_outside_above']:
        flags.append('frequent_xy_outside_image')
    if metadata.get('timestamp_source') == 'frame_index_over_fps': flags.append('timestamp_fps_fallback')
    if (hand_count>2).any(): flags.append('hand_candidate_overflow')
    if 'association_method' in arrays:
        ambiguous = (arrays['association_method']=='ambiguous').any(-1)
        uncertain = ((arrays['association_indices']>=0) & ~arrays['association_valid']).any(-1)
        if ambiguous.any(): flags.append('association_ambiguous')
        if uncertain.any(): flags.append('association_tracking_uncertain')
    else:
        ambiguous = uncertain = None
    result = dict(hand_candidate_max_count=int(hand_count.max()),
                  hand_candidate_overflow_fraction=float((hand_count>2).mean()),
                  association_ambiguous_frame_fraction=float(ambiguous.mean()) if ambiguous is not None else None,
                  association_uncertain_frame_fraction=float(uncertain.mean()) if uncertain is not None else None,
                  association_unassigned_side_fraction=float((arrays['association_indices']<0).mean()) if ambiguous is not None else None,
                  frames=len(pose), pose_observed_fraction=float(pose.mean()),
                  pose_low_visibility_fraction=low_vis,
                  no_hand_fraction=float((hand_count == 0).mean()),
                  one_hand_fraction=float((hand_count == 1).mean()),
                  two_hands_fraction=float((hand_count == 2).mean()),
                  longest_no_hand_run_frames=longest_run(hand_count == 0),
                  xy_outside_observed_fraction=outside_fraction,
                  decode_complete=bool(integrity), review_flags=flags,
                  qc_status='review' if flags else 'not_flagged',
                  completeness_warnings=completeness_warnings,
                  pose_missing_fraction_per_joint=(1-pose.mean(0)).tolist(),
                  hand_missing_fraction_per_slot_joint=(1-hand.mean(0)).tolist())
    if 'associated_hand_observed' in arrays:
        result['associated_hand_missing_fraction_per_side_joint'] = (1-arrays['associated_hand_observed'].mean(0)).tolist()
    return result


BODY_EDGES = [(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24)]
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
              (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
              (13,17),(0,17),(17,18),(18,19),(19,20)]


def overlay(npz, destination):
    arrays, meta = load_raw(npz)
    video = Path(meta['source_path'])
    if sha256(video) != meta['source_sha256']:
        raise ValueError('Source video hash differs from extraction; overlay refused')
    cap = cv2.VideoCapture(str(video))
    images = []
    try:
        if not cap.isOpened(): raise ValueError(f'Cannot open {video}')
        for position in np.unique(np.linspace(0,len(arrays['frame_indices'])-1,3).astype(int)):
            index = int(arrays['frame_indices'][position])
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok, frame = cap.read()
            if not ok: raise ValueError(f'Cannot decode frame {index}')
            h,w = frame.shape[:2]
            def draw(points, edges, color):
                valid = np.isfinite(points[...,:2]).all(-1)
                # Outside-image points remain in raw; do not pass extreme coordinates to drawing.
                valid &= (np.abs(points[...,:2]) < 10).all(-1)
                pixels = [(int(p[0]*w),int(p[1]*h)) if v else None for p,v in zip(points,valid)]
                for a,b in edges:
                    if pixels[a] is not None and pixels[b] is not None:
                        cv2.line(frame,pixels[a],pixels[b],color,3)
                for p in pixels:
                    if p is not None: cv2.circle(frame,p,4,color,-1)
            mapping = {source:i for i,source in enumerate(meta.get('pose_indices',range(arrays['raw_pose_xyz'].shape[1])))}
            edges = [(mapping[a],mapping[b]) for a,b in BODY_EDGES if a in mapping and b in mapping]
            draw(arrays['raw_pose_xyz'][position],edges,(0,255,255))
            if 'hand_candidate_offsets' in arrays:
                start,end = arrays['hand_candidate_offsets'][position:position+2]
                for candidate in arrays['hand_candidate_xyz'][start:end]:
                    draw(candidate,HAND_EDGES,(160,160,160))
            hand_view = arrays.get('associated_hand_xyz',arrays['raw_hand_xyz'])
            for slot,color in enumerate(((0,255,0),(255,100,0))):
                draw(hand_view[position,slot],HAND_EDGES,color)
            caption = "associated L/R; gray=candidates" if "associated_hand_xyz" in arrays else "detector slots"
            cv2.putText(frame,f"frame {index} | {caption}",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            images.append(cv2.resize(frame,(480,round(h*480/w))))
        destination = Path(destination)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(destination),np.concatenate(images,axis=1)):
            raise ValueError('Preview write failed')
    finally:
        cap.release()


def audit_ledgers(source):
    latest = {}
    for path in sorted(source.rglob('manifest.csv')) if source.is_dir() else []:
        with path.open(newline='',encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                key = row.get('source_path') or str(path)
                previous = latest.get(key)
                if previous is None or row['timestamp_utc'] >= previous['timestamp_utc']:
                    latest[key] = row
    counts = defaultdict(int)
    failures = []
    for row in latest.values():
        counts[row['status']] += 1
        if row['status'] not in ('complete','skipped_complete'):
            failures.append(row)
    return dict(counts),failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='Extraction output root or one NPZ')
    parser.add_argument('--output',required=True,type=Path,help='Separate QC report directory')
    parser.add_argument('--policy',type=Path,default=Path(__file__).with_name('qc_policy.json'))
    parser.add_argument('--previews',type=int,default=0,help='Up to N clip contact sheets: flagged plus stratified random sample')
    parser.add_argument('--dataset-root',type=Path,help='Compare coverage with all front_view.json files')
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    if args.previews < 0: parser.error('previews must be nonnegative')
    policy = json.loads(args.policy.read_text())
    if set(policy) != set(DEFAULT_POLICY) or any(type(x) not in (int,float) or not 0<=x<=1 for x in policy.values()):
        parser.error('Policy keys must match default and values must be finite fractions [0,1]')
    source = args.input.resolve(); out = args.output.resolve()
    files = [source] if source.is_file() else sorted(source.rglob('*.npz'))
    if not files: parser.error('No NPZs found')
    if out == source or (source.is_file() and out == source.parent):
        parser.error('Use a separate QC report directory')
    out.mkdir(parents=True,exist_ok=True)
    fields = ['npz','video_id','source_part','view','signer_id','gloss','frames','qc_status',
              'pose_observed_fraction','pose_low_visibility_fraction','no_hand_fraction',
              'one_hand_fraction','two_hands_fraction','longest_no_hand_run_frames',
              'xy_outside_observed_fraction','decode_complete','hand_candidate_max_count',
              'hand_candidate_overflow_fraction','association_ambiguous_frame_fraction',
              'association_uncertain_frame_fraction','association_unassigned_side_fraction','review_flags','error']
    expected = {}
    metadata_hashes = {}
    if args.dataset_root:
        for metadata_path in sorted(args.dataset_root.glob('Part_*/front_view.json')):
            metadata_hashes[str(metadata_path.resolve())] = sha256(metadata_path)
            for record in json.loads(metadata_path.read_text(encoding='utf-8-sig')):
                key = (metadata_path.parent.name, 'front', record['video_id'])
                if key in expected: raise ValueError(f'Duplicate official identity: {key}')
                expected[key] = record
        if not expected: parser.error('No official front-view metadata at dataset-root')
    observed = set(); duplicates = []
    totals = defaultdict(int); groups = {}; flagged = []; rng = random.Random(args.seed)
    with (out/'qc_report.csv').open('w',newline='',encoding='utf-8') as stream, \
         (out/'qc_details.jsonl').open('w',encoding='utf-8') as details:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
        for path in files:
            row = dict(npz=str(path))
            try:
                arrays,meta = load_raw(path)
                row.update({key:meta.get(key,'') for key in ('video_id','source_part','view','signer_id','gloss')})
                row.update(metrics(arrays,meta,policy))
                identity = (row['source_part'], row['view'], row['video_id'])
                if identity in observed: duplicates.append(identity)
                observed.add(identity)
                if args.dataset_root:
                    record = expected.get(identity)
                    if record is None or meta.get('official_metadata') != record:
                        row['review_flags'].append('current_official_metadata_mismatch')
                        row['qc_status'] = 'review'
                key = (row['source_part'],row['signer_id'])
                # One reservoir candidate per Part/signer, not always the first clip/class.
                count, candidate = groups.get(key,(0,None)); count += 1
                if rng.randrange(count)==0: candidate = path
                groups[key] = count,candidate
                if row['review_flags'] and len(flagged)<args.previews: flagged.append(path)
            except Exception as exc:
                row.update(qc_status='invalid',error=f'{type(exc).__name__}: {exc}',review_flags=['invalid_cache'])
            totals[row['qc_status']] += 1
            details.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            flat=dict(row,review_flags=json.dumps(row['review_flags'],ensure_ascii=False))
            writer.writerow(flat)
    candidates=[value[1] for value in groups.values()]; rng.shuffle(candidates)
    # Reserve roughly half of the preview budget for random Part/signer candidates.
    selected=list(dict.fromkeys(flagged[:(args.previews+1)//2]+candidates))[:args.previews]
    previews=[]
    for path in selected:
        target=out/'previews'/(hashlib.sha256(str(path).encode()).hexdigest()[:12]+'.jpg')
        event=dict(npz=str(path),preview=str(target))
        try: overlay(path,target); event['status']='created'
        except Exception as exc: event.update(status='failed',error=str(exc))
        previews.append(event)
    ledger_counts, failures = audit_ledgers(source)
    with (out/'batch_failures.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=MANIFEST_FIELDS,extrasaction='ignore')
        writer.writeheader(); writer.writerows(failures)
    summary=dict(batch_latest_status_counts=ledger_counts,batch_noncomplete=len(failures),schema='vsl400.qc.v2',total=len(files),counts=dict(totals),policy=policy,
                 policy_sha256=sha256(args.policy),qc_code_sha256=sha256(__file__),
                 seed=args.seed,previews=previews,
                 coverage=dict(checked=bool(args.dataset_root),expected=len(expected) if args.dataset_root else None,
                     observed_unique=len(observed),missing=sorted(set(expected)-observed) if args.dataset_root else [],
                     unexpected=sorted(observed-set(expected)) if args.dataset_root else [],duplicates=duplicates,
                     metadata_sha256=metadata_hashes),
                 warning='not_flagged does not mean correct; review flags are uncalibrated heuristics, not automatic exclusions')
    (out/'qc_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    coverage = summary['coverage']
    summary['dataset_complete'] = bool(args.dataset_root) and not (
        coverage['missing'] or coverage['unexpected'] or coverage['duplicates'] or totals['invalid'] or failures)
    (out/'qc_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(dict(total=len(files),counts=dict(totals),batch_noncomplete=len(failures),dataset_complete=summary['dataset_complete'],
        coverage_checked=bool(args.dataset_root),missing=len(coverage['missing']),
        unexpected=len(coverage['unexpected']),duplicates=len(coverage['duplicates']),
        summary=str(out/'qc_summary.json')),ensure_ascii=False))
    if totals['invalid'] or failures or any(p['status']=='failed' for p in previews): raise SystemExit(1)


if __name__=='__main__': main()
