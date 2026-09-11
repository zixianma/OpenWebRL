#!/usr/bin/env python3
"""Fetch only pinned OpenWebRL ARM inputs, bounded network concurrency, no models."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import resource
import time
import urllib.request

REV = '0d83b48c1659cac47a1044ef88fb573d3c16e180'
ROOT = 'https://huggingface.co'
REPO = 'PTeterwak/action-reward-models-data'


def listing(path):
    url = f'{ROOT}/api/datasets/{REPO}/tree/{REV}/{path}?limit=1000'
    result = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            result.extend(json.load(response))
            links = response.headers.get('Link', '')
        url = None
        for part in links.split(','):
            if 'rel="next"' in part:
                url = part.split('<')[1].split('>')[0]
    return result


def fetch(entry, output):
    path = output / entry['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get('lfs', {}).get('oid')
    def checksum(p):
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    if path.exists() and path.stat().st_size == entry['size']:
        actual = checksum(path)
        if expected is None or actual == expected:
            return {'path': str(path), 'sha256': actual, 'size': entry['size'], 'reused': True}
    partial = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            h = hashlib.sha256()
            url = f'{ROOT}/datasets/{REPO}/resolve/{REV}/{entry["path"]}'
            with urllib.request.urlopen(url, timeout=90) as response, partial.open('wb') as out:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    out.write(chunk); h.update(chunk)
            if partial.stat().st_size != entry['size'] or (expected and h.hexdigest() != expected):
                raise ValueError('download integrity mismatch: ' + entry['path'])
            partial.replace(path)
            return {'path': str(path), 'sha256': h.hexdigest(), 'size': entry['size'], 'reused': False}
        except Exception:
            if attempt == 2: raise
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 185))
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = listing('openwebrl_actor')
    wanted = {'states_full.jsonl', 'candidates_merged.jsonl', 'labels_drawlevel.jsonl'}
    entries = [r for r in rows if r['type'] == 'file' and Path(r['path']).name in wanted]
    entries.extend(r for r in listing('openwebrl_actor/state_images') if r['type'] == 'file')
    (args.output / 'remote-manifest.json').write_text(json.dumps({'revision': REV, 'entries': entries}, indent=2))
    print(json.dumps({'files': len(entries), 'total_bytes': sum(r['size'] for r in entries)}), flush=True)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, r, args.output): r['path'] for r in entries}
        for i, future in enumerate(as_completed(futures), 1):
            try: results.append(future.result())
            except Exception as error: failures.append({'path': futures[future], 'error': str(error)})
            if i % 200 == 0: print(f'fetched/verified {i}/{len(entries)}; failures={len(failures)}', flush=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {'revision': REV, 'files': results, 'failures': failures, 'wall_seconds': time.monotonic()-start,
              'cpu_seconds': usage.ru_utime+usage.ru_stime, 'peak_rss_mib': usage.ru_maxrss/1024}
    (args.output / 'download-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k,v in report.items() if k != 'files'}), flush=True)
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
