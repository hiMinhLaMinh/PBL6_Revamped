"""Cache detector outputs before any ISLR feature engineering.

This is a separate schema from extract_single/extract_batch's normalized NPZs.
See ../README/EXTRACTION.md for coordinate, missing-data and hand-slot conventions.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import traceback
from uuid import uuid4

from batch_io import MetadataResolver, completion, manifest_row, error_event
from hand_association import (load_policy, DEFAULT_POLICY, CANDIDATE_FIELDS,
                              ASSOCIATION_SHAPES, associate_sequence)
from zipfile import BadZipFile

import cv2
import numpy as np

SCHEMA = "vsl400.detector_raw.v3"
POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
POSE_NAMES = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
              "left_wrist", "right_wrist", "left_hip", "right_hip"]
DEFAULT_CONFIG = Path(__file__).with_name("raw_extraction.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"static_image_mode", "input_mirrored", "pose", "hands"}
    if set(cfg) != expected:
        raise ValueError(f"Config requires exactly {sorted(expected)}")
    for key in ("static_image_mode", "input_mirrored"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    options = {
        "pose": {"model_complexity", "smooth_landmarks", "min_detection_confidence", "min_tracking_confidence"},
        "hands": {"model_complexity", "max_num_hands", "min_detection_confidence", "min_tracking_confidence"},
    }
    for group, keys in options.items():
        if set(cfg[group]) != keys:
            raise ValueError(f"{group} requires exactly {sorted(keys)}")
        for name in ("min_detection_confidence", "min_tracking_confidence"):
            value = cfg[group][name]
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"{group}.{name} must be in [0, 1]")
    if type(cfg["hands"]["max_num_hands"]) is not int or cfg["hands"]["max_num_hands"] != 2:
        raise ValueError("This single-signer schema requires 2 hand slots")
    if cfg["pose"]["smooth_landmarks"] is not False:
        raise ValueError("Raw foundation requires pose.smooth_landmarks=false; smooth downstream")
    for group, allowed in (("pose", (0, 1, 2)), ("hands", (0, 1))):
        value = cfg[group]["model_complexity"]
        if type(value) is not int or value not in allowed:
            raise ValueError(f"Invalid {group}.model_complexity")
    return cfg


def xyz(landmarks, count):
    if landmarks is None:
        return np.full((count, 3), np.nan, np.float32)
    if len(landmarks.landmark) != count:
        raise ValueError(f"Expected {count} landmarks, got {len(landmarks.landmark)}")
    return np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)


def pack_frame(pose, hands):
    """Preserve hand detection order, unfiltered pose visibility and missing NaNs."""
    pose_lm = pose.pose_landmarks
    hand_lms = hands.multi_hand_landmarks or []
    world_lms = hands.multi_hand_world_landmarks or []
    labels = hands.multi_handedness or []
    frame = {
        "raw_pose_xyz": xyz(pose_lm, 33)[POSE_INDICES],
        "raw_pose_visibility": np.array([p.visibility for p in pose_lm.landmark], np.float32)[POSE_INDICES]
        if pose_lm is not None else np.full(len(POSE_INDICES), np.nan, np.float32),
        "raw_pose_world_xyz": xyz(pose.pose_world_landmarks, 33)[POSE_INDICES],
        "raw_hand_xyz": np.full((2, 21, 3), np.nan, np.float32),
        "raw_hand_world_xyz": np.full((2, 21, 3), np.nan, np.float32),
        "handedness_label": np.full(2, "", dtype="<U5"),
        "handedness_score": np.full(2, np.nan, np.float32),
        "detector_returned": np.array([pose_lm is not None,
                                       len(hand_lms) > 0, len(hand_lms) > 1], dtype=bool),
    }
    n = len(hand_lms)
    frame.update(hand_candidate_xyz=np.stack([xyz(lm,21) for lm in hand_lms]) if n else np.empty((0,21,3),np.float32),
                 hand_candidate_world_xyz=np.full((n,21,3),np.nan,np.float32),
                 hand_candidate_label=np.full(n,'',dtype='<U5'),
                 hand_candidate_score=np.full(n,np.nan,np.float32))
    for slot, landmarks in enumerate(hand_lms):
        if slot < len(world_lms):
            frame['hand_candidate_world_xyz'][slot] = xyz(world_lms[slot],21)
        if slot < len(labels) and labels[slot].classification:
            classification = labels[slot].classification[0]
            if classification.label not in ('Left','Right'):
                raise ValueError(f'Unexpected handedness: {classification.label}')
            frame['hand_candidate_label'][slot] = classification.label
            frame['hand_candidate_score'][slot] = classification.score
        # Legacy detector-slot view is undefined at overflow; never truncate candidates.
        if n > 2:
            continue
        frame["raw_hand_xyz"][slot] = xyz(landmarks, 21)
        if slot < len(world_lms):
            frame["raw_hand_world_xyz"][slot] = xyz(world_lms[slot], 21)
        if slot < len(labels) and labels[slot].classification:
            label = labels[slot].classification[0]
            if label.label not in ("Left", "Right"):
                raise ValueError(f"Unexpected handedness: {label.label}")
            frame["handedness_label"][slot] = label.label
            frame["handedness_score"][slot] = label.score
    for group in ("pose", "hand"):
        key = f"raw_{group}_xyz"
        if key in frame:
            # Finite means returned coordinates, not ground-truth visibility/correctness.
            frame[f"{group}_finite"] = np.isfinite(frame[key]).all(axis=-1)
    return frame


def choose_timestamps(decoder_msec, fps):
    """Keep decoder timestamps separately; never silently assume 25 FPS."""
    pts = np.asarray(decoder_msec, np.float64) / 1000.0
    if np.isfinite(pts).all() and np.all(pts >= 0) and (len(pts) == 1 or np.all(np.diff(pts) > 0)):
        return pts, "opencv_pos_msec"
    if fps is not None and np.isfinite(fps) and fps > 0:
        return np.arange(len(pts), dtype=np.float64) / fps, "frame_index_over_fps"
    raise ValueError("Neither valid decoder timestamps nor valid FPS; cannot assign time")


def _validate_base_arrays(arrays, pose_count=8):
    t = len(arrays["frame_indices"])
    shapes = {
        "raw_pose_xyz": (t, pose_count, 3), "raw_pose_visibility": (t, pose_count),
        "raw_pose_world_xyz": (t, pose_count, 3), "raw_hand_xyz": (t, 2, 21, 3),
        "raw_hand_world_xyz": (t, 2, 21, 3), "handedness_label": (t, 2),
        "handedness_score": (t, 2), "detector_returned": (t, 3),
        "pose_finite": (t, pose_count), "hand_finite": (t, 2, 21),
        "frame_indices": (t,), "timestamps": (t,), "decoder_timestamps": (t,),
    }
    if t == 0 or set(arrays) != set(shapes):
        raise ValueError("Empty extraction or unexpected raw schema keys")
    for key, shape in shapes.items():
        arr = arrays[key]
        if arr.shape != shape or arr.dtype.hasobject:
            raise ValueError(f"Invalid {key} shape/dtype: {arr.shape}/{arr.dtype}")
        expected_dtype = (
            np.dtype("bool") if key in {"detector_returned", "pose_finite", "hand_finite"}
            else np.dtype("int32") if key == "frame_indices"
            else np.dtype("float64") if key in {"timestamps", "decoder_timestamps"}
            else np.dtype("<U5") if key == "handedness_label"
            else np.dtype("float32")
        )
        if arr.dtype != expected_dtype:
            raise ValueError(f"Invalid dtype for {key}: {arr.dtype}, expected {expected_dtype}")
        if np.issubdtype(arr.dtype, np.number) and np.isinf(arr).any():
            raise ValueError(f"Infinite values in {key}")
    if not np.array_equal(arrays["frame_indices"], np.arange(t)):
        raise ValueError("Raw frame indices must be consecutive from zero")
    times = arrays["timestamps"]
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid selected timestamps")
    for group in ("pose", "hand"):
        key = f"raw_{group}_xyz"
        if key in arrays:
            finite = np.isfinite(arrays[key])
            if np.any(finite.any(-1) != finite.all(-1)):
                raise ValueError(f"Partially missing XYZ in {group}")
            mask = arrays[f"{group}_finite"]
            if mask.dtype != bool or not np.array_equal(mask, finite.all(-1)):
                raise ValueError(f"Invalid finite mask for {group}")


def assemble_frames(frames, policy=None, aspect=1.):
    policy = policy if policy is not None else load_policy()
    arrays = {key:np.stack([frame[key] for frame in frames]) for key in frames[0]
              if key not in CANDIDATE_FIELDS}
    counts = [len(frame['hand_candidate_xyz']) for frame in frames]
    arrays['hand_candidate_offsets'] = np.r_[0,np.cumsum(counts,dtype=np.int64)].astype(np.int64)
    for key in CANDIDATE_FIELDS:
        arrays[key] = np.concatenate([frame[key] for frame in frames],axis=0)
    arrays.update(associate_sequence(arrays,policy,aspect))
    return arrays


def validate_arrays(arrays, pose_count=8):
    if 'hand_candidate_offsets' not in arrays:
        return _validate_base_arrays(arrays,pose_count)
    extra = set(CANDIDATE_FIELDS) | {'hand_candidate_offsets'} | set(ASSOCIATION_SHAPES)
    _validate_base_arrays({key:value for key,value in arrays.items() if key not in extra},pose_count)
    if not extra.issubset(arrays): raise ValueError('Missing v3 candidate/association arrays')
    t = len(arrays['frame_indices']); offsets = arrays['hand_candidate_offsets']
    if offsets.shape != (t+1,) or offsets.dtype != np.int64 or offsets[0] != 0 or np.any(np.diff(offsets)<0):
        raise ValueError('Invalid candidate frame offsets')
    d = int(offsets[-1])
    shapes = {'hand_candidate_xyz':(d,21,3),'hand_candidate_world_xyz':(d,21,3),
              'hand_candidate_label':(d,),'hand_candidate_score':(d,)}
    shapes.update({key:(t,)+shape for key,shape in ASSOCIATION_SHAPES.items()})
    for key,shape in shapes.items():
        value = arrays[key]
        dtype = (np.int32 if key in ('association_indices','association_proposed_indices') else
                 bool if key in ('association_valid','associated_hand_observed') else
                 '<U12' if key == 'association_method' else '<U5' if key == 'hand_candidate_label' else np.float32)
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f'Invalid v3 array {key}: {value.shape}/{value.dtype}')
        if value.dtype.kind == 'f' and np.isinf(value).any(): raise ValueError(f'Infinite {key}')
    for key in ('hand_candidate_xyz','hand_candidate_world_xyz'):
        finite = np.isfinite(arrays[key])
        if np.any(finite.any(-1) != finite.all(-1)): raise ValueError(f'Partial XYZ: {key}')
    if not np.isin(arrays['association_method'],['none','pose','tracking','ambiguous']).all():
        raise ValueError('Unknown association method')
    for i in range(t):
        start,end = offsets[i:i+2]; n = int(end-start)
        selected = arrays['association_indices'][i]
        proposed = arrays['association_proposed_indices'][i]
        if np.any(selected < -1) or np.any(selected >= n) or np.any(proposed < -1) or np.any(proposed >= n):
            raise ValueError('Association index out of frame bounds')
        used = selected[selected >= 0]
        if len(set(used.tolist())) != len(used): raise ValueError('Same candidate assigned twice')
        for side,idx in enumerate(selected):
            for output,source in [('associated_hand_xyz','hand_candidate_xyz'),
                                  ('associated_hand_world_xyz','hand_candidate_world_xyz')]:
                expected = arrays[source][start+idx] if idx >= 0 else np.full((21,3),np.nan,np.float32)
                if not np.array_equal(arrays[output][i,side],expected,equal_nan=True):
                    raise ValueError('Associated coordinates do not match selected raw candidate')
            if arrays['association_valid'][i,side] and (idx < 0 or arrays['association_method'][i,side] != 'pose'):
                raise ValueError('Invalid confident association')
        if n > 2 and np.isfinite(arrays['raw_hand_xyz'][i]).any():
            raise ValueError('Overflow must not silently truncate raw detector slots')
    if not np.array_equal(arrays['associated_hand_observed'],np.isfinite(arrays['associated_hand_xyz']).all(-1)):
        raise ValueError('Invalid associated observation mask')


def extract_video(video, cfg, max_frames=None, association_policy=None):
    # Lazy import allows schema/adapter tests without a working MediaPipe backend.
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("Use the pinned MediaPipe Solutions runtime; see ../README/EXTRACTION.md")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps_value = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_value) if np.isfinite(fps_value) and fps_value > 0 else None
        reported = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        reported = int(reported) if np.isfinite(reported) and reported > 0 else None
        frames, decoder_msec, sizes = [], [], set()
        with ExitStack() as stack:
            pose = stack.enter_context(mp.solutions.pose.Pose(
                static_image_mode=cfg["static_image_mode"], enable_segmentation=False, **cfg["pose"]))
            hands = stack.enter_context(mp.solutions.hands.Hands(
                static_image_mode=cfg["static_image_mode"], **cfg["hands"]))
            while max_frames is None or len(frames) < max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                decoder_msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                sizes.add((int(bgr.shape[1]), int(bgr.shape[0])))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                frames.append(pack_frame(pose.process(rgb), hands.process(rgb)))
        if not frames:
            raise ValueError(f"No frames decoded: {video}")
        if len(sizes) != 1:
            raise ValueError("Frame dimensions change inside video; inspect before caching")
        width, height = next(iter(sizes))
        arrays = assemble_frames(frames, association_policy, height/width)
        arrays["frame_indices"] = np.arange(len(frames), dtype=np.int32)
        arrays["decoder_timestamps"] = np.asarray(decoder_msec, dtype=np.float64) / 1000.0
        arrays["decoder_timestamps"][~np.isfinite(arrays["decoder_timestamps"])] = np.nan
        arrays["timestamps"], time_source = choose_timestamps(decoder_msec, fps)
        validate_arrays(arrays)
        width, height = next(iter(sizes))
        warnings = []
        if reported is not None and reported != len(frames):
            warnings.append(f"decoded_frames={len(frames)}, reported_frames={reported}; check truncation/decode")
        if time_source != "opencv_pos_msec":
            warnings.append("Decoder timestamps unusable; selected timestamps use FPS and assume CFR")
        overflow_frames = int((np.diff(arrays['hand_candidate_offsets'])>2).sum())
        ambiguous_frames = int((arrays['association_method']=='ambiguous').any(-1).sum())
        uncertain_frames = int(((arrays['association_indices']>=0) & ~arrays['association_valid']).any(-1).sum())
        if overflow_frames: warnings.append(f'Hand candidate overflow in {overflow_frames} frames; all candidates retained')
        if ambiguous_frames: warnings.append(f'Association ambiguous in {ambiguous_frames} frames; abstained sides remain NaN')
        if uncertain_frames: warnings.append(f'Tracking-only association in {uncertain_frames} frames; marked uncertain')
        return arrays, {
            "hand_candidate_overflow_frames": overflow_frames,
            "association_ambiguous_frames": ambiguous_frames,
            "association_uncertain_frames": uncertain_frames,
            "fps": fps, "width": width, "height": height,
            "decoded_frames": len(frames), "reported_frames": reported,
            "max_frames": max_frames, "timestamp_source": time_source,
            "frame_limit_reached": max_frames is not None and len(frames) == max_frames,
            "warnings": warnings,
        }
    finally:
        cap.release()


def load_raw(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["meta_json"].item())
        if meta.get("schema") not in (SCHEMA, "vsl400.detector_raw.v1", "vsl400.detector_raw.v2") or meta.get("stage") != "detector_raw":
            raise ValueError("Expected detector_raw foundation NPZ, not normalized/repaired features")
        arrays = {key: data[key] for key in data.files if key != "meta_json"}
    if meta["schema"] == SCHEMA and "hand_candidate_offsets" not in arrays:
        raise ValueError("v3 requires all raw candidates")
    validate_arrays(arrays, 33 if meta["schema"] == "vsl400.detector_raw.v1" else 8)
    if meta["schema"] == SCHEMA and meta.get("pose_indices", POSE_INDICES) != POSE_INDICES:
        raise ValueError("Unexpected pose index mapping")
    if len(arrays["frame_indices"]) != meta["decoded_frames"]:
        raise ValueError("Decoded-frame metadata mismatch")
    return arrays, meta


def save_raw(path, arrays, metadata):
    """Atomic cache writes; a failed extraction cannot leave a valid-looking NPZ."""
    path = Path(path)
    validate_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temp_path = Path(stream.name)
            np.savez_compressed(stream, **arrays, meta_json=np.array(canonical_json(metadata)))
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def runtime_info():
    import mediapipe as mp
    return {"python": platform.python_version(), "mediapipe": mp.__version__,
            "opencv": cv2.__version__, "numpy": np.__version__,
            "extractor_sha256": sha256(__file__),
            "batch_io_sha256": sha256(Path(__file__).with_name("batch_io.py")),
            "association_code_sha256": sha256(Path(__file__).with_name("hand_association.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Video or dataset/Part/view directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--association-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--max-frames", type=int, help="Capped pilot; never automatically marked complete")
    parser.add_argument("--max-videos", type=int, help="First N selected videos, including resume candidates")
    parser.add_argument("--view", choices=("all", "front", "left", "right"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/metadata without inference or output writes")
    parser.add_argument("--allow-missing-metadata", action="store_true", help="Explicit opt-out for standalone non-dataset videos")
    parser.add_argument("--resume", action="store_true", help="Skip complete matching caches; retry matching incomplete caches")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace mismatched/corrupt caches")
    args = parser.parse_args()
    for name in ("max_frames", "max_videos"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    cfg = load_config(args.config)
    association_policy = load_policy(args.association_policy)
    source = args.input.resolve()
    output = args.output.resolve()
    if source.is_file():
        videos, base = [source], source.parent
    elif source.is_dir():
        videos = sorted(p for p in source.rglob("*") if p.is_file()
                        and not p.name.startswith(".")
                        and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"})
        base = source
        if output == source or source in output.parents:
            parser.error("Output must be outside input dataset")
    else:
        parser.error(f"Input does not exist: {source}")
    if args.view != "all":
        videos = [p for p in videos if p.parent.name == args.view + "_view"]
    discovered = len(videos)
    if args.max_videos is not None:
        videos = videos[:args.max_videos]
    if not videos:
        parser.error("No matching videos")
    resolver = MetadataResolver(args.allow_missing_metadata)
    # Snapshot official metadata before inference; fail preflight before writing NPZs.
    plans, invalid = [], []
    for video in videos:
        relative = video.relative_to(base)
        dest = output / relative.parent / (relative.name + ".raw.npz")
        try:
            official = resolver.resolve(video)
            plans.append((video, relative, dest, official))
        except Exception as exc:
            invalid.append({"video": str(video), "error": f"{type(exc).__name__}: {exc}"})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    if args.dry_run:
        print(canonical_json({"dry_run": True, "discovered": discovered,
              "selected": len(videos), "valid": len(plans), "invalid": len(invalid),
              "schema": SCHEMA, "pose_indices": POSE_INDICES, "landmarks_per_frame": 50,
              "association_policy": association_policy,
              "errors": invalid[:10], "examples": [
                  {"video": str(v), "npz": str(d), "video_id": m["video_id"],
                   "source_part": m["source_part"], "view": m["view"],
                   "signer_id": m["signer_id"], "gloss": m["gloss"]}
                  for v, _, d, m in plans[:3]]}))
        if invalid:
            raise SystemExit(1)
        return
    output.mkdir(parents=True, exist_ok=True)
    ledger, errors = output / "manifest.csv", output / "batch_errors.jsonl"
    errors.touch(exist_ok=True)
    if invalid:
        for event in invalid:
            now = datetime.now(timezone.utc).isoformat()
            manifest_row(ledger, {"run_id": run_id, "timestamp_utc": now,
                         "status": "metadata_failed", "source_path": event["video"],
                         "error": event["error"], "schema": SCHEMA})
            error_event(errors, dict(event, run_id=run_id, timestamp_utc=now, status="metadata_failed"))
        raise SystemExit(f"Metadata preflight failed for {len(invalid)} videos; no extraction started")
    try:
        runtime = runtime_info()
    except Exception as exc:
        event = dict(run_id=run_id, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                     status="runtime_failed", error=f"{type(exc).__name__}: {exc}", schema=SCHEMA)
        manifest_row(ledger, event)
        error_event(errors, dict(event, traceback=traceback.format_exc()))
        raise SystemExit(event["error"])
    failures = 0
    for index, (video, relative, dest, official) in enumerate(plans, 1):
        expected = official["official_metadata"]
        row = dict(official, run_id=run_id, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   schema=SCHEMA, source_relative_path=relative.as_posix(), source_path=str(video),
                   npz_path=str(dest.relative_to(output)), expected_frames=expected["num_frames"] if expected else "",
                   official_fps=expected["fps"] if expected else "")
        manifest_row(ledger, dict(row, status="started"))
        try:
            identity = {"source_sha256": sha256(video), "source_relative_path": relative.as_posix(),
                        "config": cfg, "runtime": runtime, "max_frames": args.max_frames,
                        "schema": SCHEMA, "association_policy": association_policy, **official}
            row["source_sha256"] = identity["source_sha256"]
            if dest.exists():
                retry_incomplete = False
                if args.resume:
                    try:
                        _, old = load_raw(dest)
                    except (ValueError, OSError, KeyError, EOFError, BadZipFile):
                        if not args.overwrite:
                            raise
                        old = {}
                    matching = all(old.get(key) == value for key, value in identity.items())
                    if matching:
                        verified, _ = completion(old, expected)
                        if old.get("completion_status") == "complete" and verified:
                            manifest_row(ledger, dict(row, status="skipped_complete",
                                decoded_frames=old["decoded_frames"], reported_frames=old.get("reported_frames"),
                                fps=old.get("fps"), warnings=canonical_json(old.get("warnings", []))))
                            print(f"[{index}/{len(plans)}] verified skip: {relative}", flush=True)
                            continue
                        retry_incomplete = True
                if not retry_incomplete and not args.overwrite:
                    raise ValueError(f"Mismatched/existing cache: {dest}; use separate output or --overwrite")
            arrays, details = extract_video(video, cfg, args.max_frames, association_policy)
            complete, warnings = completion(details, expected)
            status = "complete" if complete else "incomplete"
            metadata = {
                **identity, **details, "stage": "detector_raw", "completion_status": status,
                "run_id": run_id, "source_path": str(video), "source_bytes": video.stat().st_size,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "pose_indices": POSE_INDICES, "pose_names": POSE_NAMES,
                "backend": "separate_mediapipe_solutions_pose_hands",
                "hand_order": "raw_slots_only_when_candidates_le_2; associated_side_order_left_right",
                "association_side_order": ["left", "right"],
                "association_distance_units": "image_width",
                "detector_returned_order": ["pose", "hand_slot_0", "hand_slot_1"],
                "coordinates": "native MediaPipe XYZ; XY normalized by image dimensions; Z local to each model",
                "world_coordinates": "native estimates; pose hip origin; separate hand-center origins",
                "warnings": warnings, "postprocessing": [],
            }
            save_raw(dest, arrays, metadata)
            row.update(decoded_frames=details["decoded_frames"], reported_frames=details.get("reported_frames"),
                       fps=details.get("fps"), warnings=canonical_json(warnings))
            row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_row(ledger, dict(row, status=status))
            if not complete:
                failures += 1
                error_event(errors, dict(row, status=status, error="Decode completeness not verified"))
            print(f"[{index}/{len(plans)}] {status}: {relative}; T={details['decoded_frames']}; {warnings}", flush=True)
        except Exception as exc:
            failures += 1
            row.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), status="failed", error=f"{type(exc).__name__}: {exc}")
            manifest_row(ledger, row)
            error_event(errors, dict(row, traceback=traceback.format_exc()))
            print(f"[{index}/{len(plans)}] FAILED {relative}: {row['error']}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
