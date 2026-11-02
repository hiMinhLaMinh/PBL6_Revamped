"""Run independent front-view Part processes; each has its own manifest and tracker."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from uuid import uuid4

from extract_raw import sha256


def plan_parts(source, output, config, limit=None, association_policy=None):
    plans = []
    for part in sorted(Path(source).glob('Part_*')):
        if not part.is_dir() or not (part / 'front_view').is_dir():
            continue
        command = [sys.executable, str(Path(__file__).with_name('extract_raw.py')),
                   str(part.resolve()), '--output', str((Path(output) / part.name).resolve()),
                   '--config', str(Path(config).resolve()), '--view', 'front']
        if association_policy is not None:
            command += ['--association-policy',str(Path(association_policy).resolve())]
        if limit is not None:
            command += ['--max-videos', str(limit)]
        plans.append((part.name, command))
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Root containing Part_* directories')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('raw_extraction.json'))
    parser.add_argument('--association-policy', type=Path, default=Path(__file__).with_name('association_policy.json'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-videos-per-part', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or (args.max_videos_per_part is not None and args.max_videos_per_part < 1):
        parser.error('workers and max-videos-per-part must be positive')
    source, output = args.input.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        parser.error('Output must be outside input')
    plans = plan_parts(source, output, args.config, args.max_videos_per_part, args.association_policy)
    if not plans:
        parser.error('No Part_* directories with front_view found')
    if args.resume:
        for _, command in plans:
            command.append('--resume')
    if args.dry_run:
        reports = []
        for part, command in plans:
            result = subprocess.run(command + ['--dry-run'], capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(f'{part}: {result.stdout}\n{result.stderr}')
            reports.append(dict(part=part, plan=json.loads(result.stdout)))
        print(json.dumps({'dry_run': True, 'workers': args.workers,
              'selected': sum(r['plan']['selected'] for r in reports), 'parts': reports}, ensure_ascii=False))
        return
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8]
    logs = output / '_batch' / run_id
    logs.mkdir(parents=True)
    environment = os.environ.copy()
    # These limit unrelated library pools, not XNNPACK's internal threadpool.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'OPENCV_FOR_THREADS_NUM'):
        environment[key] = '1'
    report = dict(run_id=run_id, input=str(source), output=str(output), workers=args.workers,
                  config_sha256=sha256(args.config), association_policy_sha256=sha256(args.association_policy), runner_sha256=sha256(__file__),
                  extractor_sha256=sha256(Path(__file__).with_name('extract_raw.py')),
                  platform=platform.platform(), machine=platform.machine(),
                  max_videos_per_part=args.max_videos_per_part, resume=args.resume,
                  thread_limits={k: environment[k] for k in environment if k in
                  ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OPENCV_FOR_THREADS_NUM')},
                  parts=[])
    report_path = logs / 'run_summary.json'
    started = time.perf_counter()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    def execute(part, command):
        begin = time.perf_counter()
        try:
            with (logs / (part + '.log')).open('w') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            return dict(part=part, returncode=result.returncode,
                        seconds=time.perf_counter()-begin, command=command)
        except Exception as exc:
            return dict(part=part, returncode=-1, seconds=time.perf_counter()-begin,
                        command=command, error=f'{type(exc).__name__}: {exc}')

    # Threads only supervise fresh Python processes; no MediaPipe graph is forked/shared.
    with ThreadPoolExecutor(max_workers=min(args.workers, len(plans))) as pool:
        futures = [pool.submit(execute, part, command) for part, command in plans]
        for future in as_completed(futures):
            result = future.result()
            report['parts'].append(result)
            report['elapsed_seconds'] = time.perf_counter()-started
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"{result['part']}: exit={result['returncode']}, {result['seconds']:.2f}s", flush=True)
    print(f"Total: {report['elapsed_seconds']:.2f}s; summary: {report_path}", flush=True)
    if any(part['returncode'] for part in report['parts']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
