"""Official VSL400 metadata, completion checks and durable batch ledgers."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MetadataResolver:
    def __init__(self, allow_missing=False):
        self.allow_missing = allow_missing
        self.cache = {}

    def resolve(self, video):
        video = Path(video)
        view_folder = video.parent.name
        part = next((p.name for p in video.parents if p.name.startswith('Part_')), '')
        path = video.parent.parent / (view_folder + '.json')
        base = dict(video_id=video.stem, source_part=part,
                    view=view_folder.removesuffix('_view'), signer_id='', gloss='',
                    source_metadata='', metadata_sha256='', official_metadata=None)
        if view_folder not in ('front_view', 'left_view', 'right_view') or not path.is_file():
            if self.allow_missing:
                return base
            raise ValueError(f'Official metadata not found for {video}: expected {path}')
        if path not in self.cache:
            content = path.read_bytes()
            rows = json.loads(content.decode('utf-8-sig'))
            if not isinstance(rows, list):
                raise ValueError(f'Metadata must be a list: {path}')
            mapping = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f'Invalid metadata row in {path}')
                vid = row.get('video_id')
                if not isinstance(vid, str) or not vid or vid in mapping:
                    raise ValueError(f'Missing/non-string/duplicate video_id {vid!r} in {path}')
                for field in ('signer_id', 'gloss'):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f'Invalid {field} for {vid} in {path}')
                n = row.get('num_frames')
                fps = row.get('fps')
                if type(n) is not int or n <= 0:
                    raise ValueError(f'Invalid num_frames for {vid} in {path}')
                if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
                    raise ValueError(f'Invalid fps for {vid} in {path}')
                if 'view' in row and row['view'].removesuffix('_view') != base['view']:
                    raise ValueError(f'Metadata view disagrees with folder: {path}, {vid}')
                mapping[vid] = row
            self.cache[path] = mapping, hashlib.sha256(content).hexdigest()
        mapping, checksum = self.cache[path]
        if video.stem not in mapping:
            raise ValueError(f'Video {video.stem} missing from official metadata {path}')
        row = mapping[video.stem]
        return dict(base, signer_id=row['signer_id'], gloss=row['gloss'],
                    source_metadata=str(path.resolve()), metadata_sha256=checksum,
                    official_metadata=row)


def completion(details, official):
    """A capped pilot or any disagreement in frame counts is not complete."""
    count = details['decoded_frames']
    expected = official['num_frames'] if official else None
    reported = details.get('reported_frames')
    reasons = []
    if details.get('frame_limit_reached', False):
        reasons.append('Frame limit reached; full decode not verified')
    if expected is not None and count != expected:
        reasons.append(f'Official frame mismatch: decoded={count}, expected={expected}')
    if reported is not None and count != reported:
        reasons.append(f'Header frame mismatch: decoded={count}, reported={reported}')
    if expected is None and reported is None:
        reasons.append('No official/header frame count to verify completeness')
    warnings = list(details.get('warnings', []))
    if official and details.get('fps') is not None and not math.isclose(
            details['fps'], official['fps'], rel_tol=1e-4, abs_tol=1e-3):
        warnings.append(f"FPS differs: decoded/header={details['fps']}, official={official['fps']}")
    return not reasons, list(dict.fromkeys(warnings + reasons))


FIELDS = ['run_id', 'timestamp_utc', 'status', 'video_id', 'source_part', 'view',
          'signer_id', 'gloss', 'source_relative_path', 'source_path', 'npz_path',
          'source_sha256', 'metadata_sha256', 'source_metadata', 'expected_frames',
          'decoded_frames', 'reported_frames', 'fps', 'official_fps',
          'schema', 'warnings', 'error']


def manifest_row(path, row):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with path.open(encoding='utf-8', newline='') as stream:
            if next(csv.reader(stream)) != FIELDS:
                raise ValueError(f'Incompatible manifest header: {path}')
    new = not path.exists() or path.stat().st_size == 0
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def error_event(path, event):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
