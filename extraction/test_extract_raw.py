"""Scientific contract: retain observations and make cache reuse auditable."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import extract_raw as raw


def landmarks(n, offset=0.0, visibility=0.05):
    return NS(landmark=[NS(x=offset + i / 100, y=0.2, z=-i / 1000,
                           visibility=visibility) for i in range(n)])


def results(missing_pose=False, reverse=False):
    pose = NS(pose_landmarks=None if missing_pose else landmarks(33),
              pose_world_landmarks=None if missing_pose else landmarks(33, offset=2.0))
    hand_sets = [landmarks(21, 0.1), landmarks(21, 0.8)]
    labels = [NS(classification=[NS(label="Left", score=0.91)]),
              NS(classification=[NS(label="Right", score=0.82)])]
    if reverse:
        hand_sets, labels = hand_sets[::-1], labels[::-1]
    return pose, NS(multi_hand_landmarks=hand_sets, multi_hand_world_landmarks=hand_sets,
                    multi_handedness=labels)


def arrays_for_test():
    frames = [raw.pack_frame(*results()), raw.pack_frame(*results(missing_pose=True, reverse=True))]
    arrays = raw.assemble_frames(frames)
    arrays.update(frame_indices=np.arange(2, dtype=np.int32),
                  timestamps=np.array([0.0, 0.04]), decoder_timestamps=np.array([0.0, 0.04]))
    return arrays


def test_preserves_xyz_low_visibility_and_hand_slot_order():
    first = raw.pack_frame(*results())
    second = raw.pack_frame(*results(reverse=True))
    np.testing.assert_allclose(first["raw_pose_visibility"], 0.05)
    assert np.isfinite(first["raw_pose_xyz"]).all()  # No visibility threshold at extraction.
    assert first["raw_pose_xyz"][7, 2] == np.float32(-0.024)
    assert first["raw_hand_xyz"][0, 0, 0] == np.float32(0.1)
    np.testing.assert_array_equal(second["raw_hand_xyz"][1], first["raw_hand_xyz"][0])
    assert list(first["handedness_label"]) == ["Left", "Right"]
    assert list(second["handedness_label"]) == ["Right", "Left"]
    assert not any("face" in key for key in first)


def test_hands_survive_missing_pose_and_do_not_require_body_normalization():
    frame = raw.pack_frame(*results(missing_pose=True))
    assert np.isnan(frame["raw_pose_xyz"]).all()
    assert np.isnan(frame["raw_pose_visibility"]).all()
    assert frame["hand_finite"].all()
    assert list(frame["detector_returned"]) == [False, True, True]


def test_missing_hand_is_nan_and_not_zero_or_last_frame_copy():
    pose, hands = results()
    hands.multi_hand_landmarks = hands.multi_hand_landmarks[:1]
    hands.multi_hand_world_landmarks = []
    hands.multi_handedness = []
    frame = raw.pack_frame(pose, hands)
    assert np.isnan(frame["raw_hand_xyz"][1]).all()
    assert np.isnan(frame["raw_hand_world_xyz"]).all()
    assert not frame["hand_finite"][1].any()
    assert list(frame["detector_returned"]) == [True, True, False]
    assert list(frame["handedness_label"]) == ["", ""]
    assert np.isnan(frame["handedness_score"]).all()


def test_timestamps_preserve_variable_intervals_and_record_fallback():
    times, source = raw.choose_timestamps([12, 51, 109], 25)
    np.testing.assert_allclose(times, [0.012, 0.051, 0.109])
    assert source == "opencv_pos_msec"
    times, source = raw.choose_timestamps([0, 0, 0], 20)
    np.testing.assert_allclose(times, [0.0, 0.05, 0.1])
    assert source == "frame_index_over_fps"
    with pytest.raises(ValueError, match="Neither valid"):
        raw.choose_timestamps([0, 0], None)


def test_round_trip_no_pickle_and_corruption_rejected(tmp_path):
    arrays = arrays_for_test()
    meta = {"schema": raw.SCHEMA, "stage": "detector_raw", "decoded_frames": 2,
            "config": raw.load_config(raw.DEFAULT_CONFIG)}
    path = tmp_path / "raw.npz"
    raw.save_raw(path, arrays, meta)
    actual, loaded_meta = raw.load_raw(path)
    assert loaded_meta == meta
    for key in arrays:
        np.testing.assert_array_equal(actual[key], arrays[key])
    assert list(tmp_path.iterdir()) == [path]
    actual["hand_finite"][0, 0, 0] = False
    with pytest.raises(ValueError, match="Invalid finite mask"):
        raw.validate_arrays(actual)


@pytest.mark.parametrize("mutation", ["partial_nan", "frame_gap", "resampled_time", "wrong_shape", "normalized"])
def test_reject_invalid_cache_contract(mutation):
    arrays = arrays_for_test()
    if mutation == "partial_nan":
        arrays["raw_pose_xyz"][0, 0, 2] = np.nan
    elif mutation == "frame_gap":
        arrays["frame_indices"][1] = 5
    elif mutation == "resampled_time":
        arrays["timestamps"][1] = 0
    elif mutation == "wrong_shape":
        arrays["raw_pose_xyz"] = arrays["raw_pose_xyz"][:, :7]
    else:
        arrays["upper_body"] = np.zeros((2, 7, 2))
    with pytest.raises(ValueError):
        raw.validate_arrays(arrays)


def test_config_rejects_hidden_smoothing_and_unknown_settings(tmp_path):
    cfg = raw.load_config(raw.DEFAULT_CONFIG)
    path = tmp_path / "config.json"
    cfg["pose"]["smooth_landmarks"] = True
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="smooth"):
        raw.load_config(path)
    cfg["pose"]["smooth_landmarks"] = False
    cfg["hands"]["typo"] = 1
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="exactly"):
        raw.load_config(path)


def test_cli_resume_checks_source_config_and_partial_run_identity(monkeypatch, tmp_path):
    video_dir = tmp_path / "videos"
    for view in ("front_view", "left_view"):
        directory = video_dir / view
        directory.mkdir(parents=True)
        (directory / "0001.mp4").write_bytes(view.encode())
    output = tmp_path / "cache"
    calls = []

    def fake_extract(video, cfg, max_frames, association_policy=None):
        calls.append(video)
        return arrays_for_test(), {"decoded_frames": 2, "reported_frames": 2, "warnings": []}

    monkeypatch.setattr(raw, "extract_video", fake_extract)
    monkeypatch.setattr(raw, "runtime_info", lambda: {"test_backend": "fixed"})
    args = ["extract_raw.py", str(video_dir), "--output", str(output), "--allow-missing-metadata"]
    monkeypatch.setattr("sys.argv", args)
    raw.main()
    assert len(calls) == 2
    assert len(list(output.rglob("*.npz"))) == 2  # Views never overwrite each other.
    monkeypatch.setattr("sys.argv", args + ["--resume"])
    raw.main()
    assert len(calls) == 2
    for extra in (["--max-frames", "3"], []):
        if not extra:
            (video_dir / "front_view" / "0001.mp4").write_bytes(b"changed source")
        monkeypatch.setattr("sys.argv", args + ["--resume"] + extra)
        with pytest.raises(SystemExit):
            raw.main()
        assert len(calls) == 2
    config = deepcopy(raw.load_config(raw.DEFAULT_CONFIG))
    config["hands"]["min_detection_confidence"] = 0.7
    config_path = tmp_path / "changed_config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("sys.argv", args + ["--resume", "--config", str(config_path)])
    with pytest.raises(SystemExit):
        raw.main()
    assert len(calls) == 2
    # Explicit replacement also recovers a corrupt NPZ under --resume.
    (output / "front_view" / "0001.mp4.raw.npz").write_bytes(b"corrupt")
    monkeypatch.setattr("sys.argv", args + ["--resume", "--overwrite"])
    raw.main()
    assert len(calls) == 3


def test_extraction_keeps_all_decoded_frames_and_disables_smoothing(monkeypatch):
    """Exercise the real extraction loop with fake detector and decoder results."""
    class Capture:
        def __init__(self, *args):
            self.index = 0
            self.released = False
        def isOpened(self):
            return True
        def get(self, key):
            return {raw.cv2.CAP_PROP_FPS: 25, raw.cv2.CAP_PROP_FRAME_COUNT: 2,
                    raw.cv2.CAP_PROP_POS_MSEC: (self.index - 1) * 40}.get(key, 0)
        def read(self):
            if self.index == 2:
                return False, None
            self.index += 1
            return True, np.zeros((12, 16, 3), np.uint8)
        def release(self):
            self.released = True

    options = {}
    class Detector:
        def __init__(self, kind, **kwargs):
            self.kind, self.index = kind, 0
            options[kind] = kwargs
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def process(self, rgb):
            self.index += 1
            return results(missing_pose=self.index == 2)[self.kind == "hands"]

    backend = NS(solutions=NS(
        pose=NS(Pose=lambda **kw: Detector("pose", **kw)),
        hands=NS(Hands=lambda **kw: Detector("hands", **kw))))
    monkeypatch.setitem(__import__("sys").modules, "mediapipe", backend)
    cap = Capture()
    monkeypatch.setattr(raw.cv2, "VideoCapture", lambda *args: cap)
    arrays, meta = raw.extract_video(Path("fake.mp4"), raw.load_config(raw.DEFAULT_CONFIG))
    assert options["pose"]["smooth_landmarks"] is False
    assert not options["pose"]["static_image_mode"]
    assert cap.released
    assert meta["decoded_frames"] == 2
    assert np.isnan(arrays["raw_pose_xyz"][1]).all()
    assert arrays["hand_finite"][1].all()
    np.testing.assert_array_equal(arrays["frame_indices"], [0, 1])
