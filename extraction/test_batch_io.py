import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import extract_raw as raw
from batch_io import MetadataResolver, completion
from test_extract_raw import arrays_for_test, results


def dataset(tmp_path, views=('front', 'left'), count=2):
    root = tmp_path / 'input'
    for view in views:
        folder = root / 'Part_1' / (view + '_view')
        folder.mkdir(parents=True)
        rows = []
        for i in range(count):
            vid = f'{i:06}'
            (folder / (vid + '.mp4')).write_bytes((view + vid).encode())
            rows.append(dict(video_id=vid, signer_id='001', gloss='Ngày Nhà giáo Việt Nam',
                             fps=25.0, num_frames=2, resolution=1080, length_seconds=0.08))
        (folder.parent / (view + '_view.json')).write_text(json.dumps(rows, ensure_ascii=False))
    return root


def invoke(monkeypatch, root, output, *flags):
    monkeypatch.setattr(sys, 'argv', ['extract_raw.py', str(root), '--output', str(output), *flags])
    raw.main()


def fake_runtime(monkeypatch):
    monkeypatch.setattr(raw, 'runtime_info', lambda: {'test': 'fixed'})


def test_pose_selection_has_no_face_or_legs():
    frame = raw.pack_frame(*results())
    assert raw.POSE_INDICES == [11, 12, 13, 14, 15, 16, 23, 24]
    expected = raw.xyz(results()[0].pose_landmarks, 33)[raw.POSE_INDICES]
    np.testing.assert_array_equal(frame['raw_pose_xyz'], expected)
    assert frame['raw_pose_world_xyz'].shape == (8, 3)
    assert frame['raw_pose_visibility'].shape == (8,)


def test_metadata_unicode_leading_zeros_and_hash(tmp_path):
    root = dataset(tmp_path)
    resolver = MetadataResolver()
    record = resolver.resolve(root / 'Part_1/front_view/000000.mp4')
    assert record['gloss'] == 'Ngày Nhà giáo Việt Nam'
    assert record['signer_id'] == '001' and record['video_id'] == '000000'
    assert record['view'] == 'front' and record['source_part'] == 'Part_1'
    assert len(record['metadata_sha256']) == 64
    assert record['official_metadata']['num_frames'] == 2


@pytest.mark.parametrize('change', ['missing_file', 'missing_id', 'duplicate', 'invalid_frames'])
def test_bad_metadata_fails_before_extraction(tmp_path, monkeypatch, change):
    root = dataset(tmp_path, views=('front',), count=1)
    path = root / 'Part_1/front_view.json'
    rows = json.loads(path.read_text())
    if change == 'missing_file':
        path.unlink()
    else:
        if change == 'missing_id': rows[0]['video_id'] = '999999'
        if change == 'duplicate': rows.append(rows[0])
        if change == 'invalid_frames': rows[0]['num_frames'] = 0
        path.write_text(json.dumps(rows))
    monkeypatch.setattr(raw, 'extract_video', lambda *a: pytest.fail('Should not extract'))
    monkeypatch.setattr(raw, 'runtime_info', lambda: pytest.fail('No inference runtime needed'))
    out = tmp_path / 'out'
    with pytest.raises(SystemExit): invoke(monkeypatch, root, out)
    assert not list(out.rglob('*.npz'))
    assert json.loads((out / 'batch_errors.jsonl').read_text())['status'] == 'metadata_failed'
    assert list(csv.DictReader((out / 'manifest.csv').open()))[-1]['status'] == 'metadata_failed'


def test_dry_run_filters_and_caps_without_writes_or_runtime(tmp_path, monkeypatch, capsys):
    root = dataset(tmp_path)
    out = tmp_path / 'out'
    monkeypatch.setattr(raw, 'runtime_info', lambda: pytest.fail('dry-run must not load MediaPipe'))
    monkeypatch.setattr(raw, 'extract_video', lambda *a: pytest.fail('dry-run must not decode'))
    invoke(monkeypatch, root, out, '--view', 'left', '--max-videos', '1', '--dry-run')
    plan = json.loads(capsys.readouterr().out)
    assert plan['selected'] == 1 and plan['discovered'] == 2
    assert plan['examples'][0]['view'] == 'left'
    assert not out.exists()


@pytest.mark.parametrize('expected,reported,cap', [(3, 2, False), (2, 3, False), (2, 2, True), (1, 2, False)])
def test_incomplete_if_official_header_or_cap_disagree(expected, reported, cap):
    valid, reasons = completion(dict(decoded_frames=2, reported_frames=reported,
        frame_limit_reached=cap, fps=25, warnings=[]), dict(num_frames=expected, fps=25))
    assert not valid and reasons


def test_incomplete_retried_then_complete_skipped(tmp_path, monkeypatch):
    root = dataset(tmp_path, views=('front',), count=1)
    out = tmp_path / 'out'
    fake_runtime(monkeypatch)
    calls = []
    def extract(*args):
        calls.append(1)
        full = arrays_for_test()
        incomplete = len(calls) == 1
        if incomplete:
            from hand_association import CANDIDATE_FIELDS
            arrays = {key:arr[:1] for key,arr in full.items() if key not in CANDIDATE_FIELDS and key != 'hand_candidate_offsets'}
            arrays['hand_candidate_offsets'] = full['hand_candidate_offsets'][:2]
            for key in CANDIDATE_FIELDS:
                arrays[key] = full[key][:int(arrays['hand_candidate_offsets'][-1])]
        else:
            arrays = full
        return arrays, dict(decoded_frames=len(arrays['frame_indices']), reported_frames=2,
                            fps=25, warnings=[], frame_limit_reached=False)
    monkeypatch.setattr(raw, 'extract_video', extract)
    with pytest.raises(SystemExit): invoke(monkeypatch, root, out)
    npz = next(out.rglob('*.npz'))
    _, meta = raw.load_raw(npz)
    assert meta['completion_status'] == 'incomplete'
    assert meta['official_metadata']['num_frames'] == 2
    assert meta['signer_id'] == '001' and meta['gloss'] == 'Ngày Nhà giáo Việt Nam'
    invoke(monkeypatch, root, out, '--resume')  # no --overwrite necessary for matching incomplete
    assert raw.load_raw(npz)[1]['completion_status'] == 'complete'
    invoke(monkeypatch, root, out, '--resume')
    assert len(calls) == 2
    events = list(csv.DictReader((out / 'manifest.csv').open()))
    assert [r['status'] for r in events] == ['started','incomplete','started','complete','started','skipped_complete']
    assert events[-1]['gloss'] == meta['gloss']


def test_changed_official_metadata_prevents_skip(tmp_path, monkeypatch):
    root = dataset(tmp_path, views=('front',), count=1)
    out = tmp_path / 'out'
    fake_runtime(monkeypatch)
    monkeypatch.setattr(raw, 'extract_video', lambda *a: (arrays_for_test(), dict(decoded_frames=2,
        reported_frames=2, fps=25, warnings=[])))
    invoke(monkeypatch, root, out)
    path = root / 'Part_1/front_view.json'
    rows = json.loads(path.read_text()); rows[0]['gloss'] = 'Nhãn được sửa'
    path.write_text(json.dumps(rows, ensure_ascii=False))
    with pytest.raises(SystemExit): invoke(monkeypatch, root, out, '--resume')
    event = json.loads((out / 'batch_errors.jsonl').read_text().splitlines()[-1])
    assert event['status'] == 'failed' and 'Mismatched' in event['error']


def test_max_videos_limits_real_batch_and_logs_failure(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    out = tmp_path / 'out'
    fake_runtime(monkeypatch)
    calls = []
    def fail(*a):
        calls.append(1)
        raise RuntimeError('Synthetic decode failure')
    monkeypatch.setattr(raw, 'extract_video', fail)
    with pytest.raises(SystemExit): invoke(monkeypatch, root, out, '--max-videos', '1')
    assert len(calls) == 1
    event = json.loads((out / 'batch_errors.jsonl').read_text())
    assert event['video_id'] == '000000' and 'Synthetic decode failure' in event['traceback']


def test_legacy_v1_cache_remains_readable(tmp_path):
    arrays = arrays_for_test()
    for name in ('raw_pose_xyz','raw_pose_world_xyz'):
        arrays[name] = np.zeros((2,33,3), np.float32)
    arrays['raw_pose_visibility'] = np.zeros((2,33), np.float32)
    arrays['pose_finite'] = np.ones((2,33), bool)
    from hand_association import CANDIDATE_FIELDS, ASSOCIATION_SHAPES
    for key in (*CANDIDATE_FIELDS, *ASSOCIATION_SHAPES, 'hand_candidate_offsets'):
        arrays.pop(key)
    meta = dict(schema='vsl400.detector_raw.v1', stage='detector_raw', decoded_frames=2)
    path = tmp_path / 'legacy.npz'
    np.savez_compressed(path, **arrays, meta_json=np.array(json.dumps(meta)))
    assert raw.load_raw(path)[0]['raw_pose_xyz'].shape == (2,33,3)


def test_batch_continues_after_one_failure(tmp_path, monkeypatch):
    root = dataset(tmp_path, views=('front',), count=2)
    out = tmp_path / 'out'
    fake_runtime(monkeypatch)
    def extract(video, *args):
        if video.stem == '000000':
            raise RuntimeError('First file broken')
        return arrays_for_test(), dict(decoded_frames=2, reported_frames=2, fps=25, warnings=[])
    monkeypatch.setattr(raw, 'extract_video', extract)
    with pytest.raises(SystemExit): invoke(monkeypatch, root, out)
    assert len(list(out.rglob('*.npz'))) == 1
    rows = list(csv.DictReader((out / 'manifest.csv').open()))
    assert [r['status'] for r in rows] == ['started','failed','started','complete']


def test_runtime_failure_is_persisted(tmp_path, monkeypatch):
    root = dataset(tmp_path, views=('front',), count=1)
    out = tmp_path / 'out'
    def fail():
        raise RuntimeError('Missing backend')
    monkeypatch.setattr(raw, 'runtime_info', fail)
    with pytest.raises(SystemExit): invoke(monkeypatch, root, out)
    event = json.loads((out / 'batch_errors.jsonl').read_text())
    assert event['status'] == 'runtime_failed' and 'Missing backend' in event['traceback']
