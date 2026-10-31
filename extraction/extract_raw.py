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
from zipfile import BadZipFile

import cv2
import numpy as np

SCHEMA = "vsl400.detector_raw.v1"
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
    if len(hand_lms) > 2:
        raise ValueError("More than two hands returned")
    frame = {
        "raw_pose_xyz": xyz(pose_lm, 33),
        "raw_pose_visibility": np.array([p.visibility for p in pose_lm.landmark], np.float32)
        if pose_lm is not None else np.full(33, np.nan, np.float32),
        "raw_pose_world_xyz": xyz(pose.pose_world_landmarks, 33),
        "raw_hand_xyz": np.full((2, 21, 3), np.nan, np.float32),
        "raw_hand_world_xyz": np.full((2, 21, 3), np.nan, np.float32),
        "handedness_label": np.full(2, "", dtype="<U5"),
        "handedness_score": np.full(2, np.nan, np.float32),
        "detector_returned": np.array([pose_lm is not None,
                                       len(hand_lms) > 0, len(hand_lms) > 1], dtype=bool),
    }
    for slot, landmarks in enumerate(hand_lms):
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


def validate_arrays(arrays):
    t = len(arrays["frame_indices"])
    shapes = {
        "raw_pose_xyz": (t, 33, 3), "raw_pose_visibility": (t, 33),
        "raw_pose_world_xyz": (t, 33, 3), "raw_hand_xyz": (t, 2, 21, 3),
        "raw_hand_world_xyz": (t, 2, 21, 3), "handedness_label": (t, 2),
        "handedness_score": (t, 2), "detector_returned": (t, 3),
        "pose_finite": (t, 33), "hand_finite": (t, 2, 21),
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


def extract_video(video, cfg, max_frames=None):
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
        arrays = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
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
        return arrays, {
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
        if meta.get("schema") != SCHEMA or meta.get("stage") != "detector_raw":
            raise ValueError("Expected detector_raw foundation NPZ, not normalized/repaired features")
        arrays = {key: data[key] for key in data.files if key != "meta_json"}
    validate_arrays(arrays)
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
            "extractor_sha256": sha256(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Single video or video directory (recursive)")
    parser.add_argument("--output", type=Path, required=True, help="Separate raw cache directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-frames", type=int, help="Pilot only; recorded in cache identity")
    parser.add_argument("--resume", action="store_true", help="Skip only verified matching raw caches")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing mismatched cache files")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    cfg = load_config(args.config)
    source = args.input.resolve()
    if source.is_file():
        videos, base = [source], source.parent
    elif source.is_dir():
        videos = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"})
        base = source
    else:
        parser.error(f"Input does not exist: {source}")
    if not videos:
        parser.error("No videos found")
    runtime = runtime_info()
    # Preserve directories and original extension: front/0001.mp4.raw.npz.
    # This avoids collisions across Part, view and video extension.
    failures = 0
    for index, video in enumerate(videos, 1):
        relative = video.relative_to(base)
        dest = args.output / relative.parent / (relative.name + ".raw.npz")
        try:
            identity = {"source_sha256": sha256(video), "source_relative_path": relative.as_posix(),
                        "config": cfg, "runtime": runtime, "max_frames": args.max_frames}
            if dest.exists():
                if args.resume:
                    try:
                        _, old = load_raw(dest)
                    except (ValueError, OSError, KeyError, EOFError, BadZipFile):
                        if not args.overwrite:
                            raise
                        old = {}
                    if all(old.get(key) == value for key, value in identity.items()):
                        print(f"[{index}/{len(videos)}] verified skip: {relative}", flush=True)
                        continue
                if not args.overwrite:
                    raise ValueError(f"Existing cache does not qualify for skip: {dest}; use another output or --overwrite")
            arrays, details = extract_video(video, cfg, args.max_frames)
            metadata = {
                **identity, **details, "schema": SCHEMA, "stage": "detector_raw",
                "source_path": str(video), "source_bytes": video.stat().st_size,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "backend": "separate_mediapipe_solutions_pose_hands",
                "hand_order": "detector_slots_not_anatomical_sides",
                "detector_returned_order": ["pose", "hand_slot_0", "hand_slot_1"],
                "coordinates": "native_mediapipe_xyz; image XY normalized by width/height; Z local to each model",
                "world_coordinates": "native model estimates in meters; pose hip origin, each hand separate hand-center origin",
                "postprocessing": [],
            }
            save_raw(dest, arrays, metadata)
            print(f"[{index}/{len(videos)}] saved {relative}: T={details['decoded_frames']}; {details['warnings']}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(videos)}] FAILED {relative}: {type(exc).__name__}: {exc}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
