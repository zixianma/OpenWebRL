"""Rejudge collaborator-owned saved endpoints with the frozen Terra protocol.

Default is a read-only plan. --execute requires explicit dollar, call, wall and
HTTP concurrency caps and a new output directory. No actors, browsers or Slurm.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path if exclusive else path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def science():
    """Keep archived function bodies byte-equivalent; relocate config only."""
    index = read(ROOT / 'source-provenance.json')['snapshots']['browsercode_luna_effort']['files']
    entries = {row['file']: row for row in index}
    paths = {}
    for name in ('common_judge.py', 'claude_budget.py', 'request_budget.py'):
        entry = entries[name]
        path = ROOT / entry['published_path']
        need(sha(path) == entry['sha256'], 'Frozen source hash differs: ' + name)
        paths[name] = path
    prompt = ROOT / 'portable_assets/support/openwebrl/eval/reward_online_mind2web.py'
    expected = read(ROOT / 'portable_assets/provenance.json')['files']['support/openwebrl/eval/reward_online_mind2web.py']['sha256']
    need(sha(prompt) == expected, 'Frozen prompt hash differs')
    helper = load_module('_portable_rejudge_helpers', paths['claude_budget.py'])
    budget = load_module('_portable_rejudge_budget', paths['request_budget.py'])
    tree = ast.parse(paths['common_judge.py'].read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'campaign' for target in node.targets)]
    need(len(assignments) == 1, 'Frozen config adaptation anchor changed')
    assignments[0].value = ast.Dict(keys=[], values=[])
    judge = types.ModuleType('_portable_rejudge_terra')
    judge.__file__ = str(paths['common_judge.py'])
    previous = sys.modules.get('claude_budget')
    sys.modules['claude_budget'] = helper
    try:
        exec(compile(ast.fix_missing_locations(tree), judge.__file__, 'exec'), judge.__dict__)
    finally:
        if previous is None:
            sys.modules.pop('claude_budget', None)
        else:
            sys.modules['claude_budget'] = previous
    judge.campaign = {'judge_protocol': deepcopy(judge.TERRA_PROTOCOL), 'frozen_support_source': str(ROOT / 'portable_assets/support')}
    sources = {str(path.relative_to(ROOT)): sha(path) for path in [*paths.values(), prompt]}
    return judge, budget, sources


def positive_money(value):
    try:
        amount = Decimal(value)
    except Exception as error:
        raise argparse.ArgumentTypeError('Use a positive finite USD cap') from error
    if not amount.is_finite() or amount <= 0:
        raise argparse.ArgumentTypeError('Use a positive finite USD cap')
    return str(amount)


def endpoints(manifest, judge):
    manifest = Path(manifest).resolve()
    data = read(manifest)
    need(data.get('schema') == 'terra-endpoint-input-v1' and type(data.get('episodes')) is list and data['episodes'], 'Use the documented nonempty endpoint manifest')
    rows, seen, inputs = [], set(), {str(manifest): sha(manifest)}
    for value in data['episodes']:
        row = deepcopy(value)
        ident = row['id']
        need(type(ident) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,160}', ident) and ident not in seen, 'Duplicate or unsafe endpoint ID')
        seen.add(ident)
        need(type(row['original_valid']) is bool and type(row['eligible']) is bool, 'Explicit boolean original_valid/eligible required')
        need(not row['eligible'] or row['original_valid'], 'Original invalid episodes cannot become eligible')
        cohort = row['cohort']
        fields = ('benchmark', 'harness', 'arm', 'actor_model', 'actor_protocol_sha256')
        need(type(cohort) is dict and all(type(cohort.get(key)) is str and cohort[key] for key in fields), 'Preserve benchmark, harness, arm, actor model and protocol identity')
        need(re.fullmatch(r'[0-9a-f]{64}', cohort['actor_protocol_sha256']), 'Actor protocol identity must be a SHA256 digest')
        if not row['eligible']:
            need(type(row.get('exclusion_reason')) is str and row['exclusion_reason'], 'Excluded episodes require a reason')
            rows.append(row)
            continue
        need(type(row['task']) is str and type(row['actions']) is list and all(type(action) is str for action in row['actions']), 'Exact task and selected-action strings required')
        scope = row['endpoint_scope']
        need(scope in ('dedicated_terminal_capture', 'last_browser_action_capture'), 'Declare the terminal capture policy')
        need(scope != 'last_browser_action_capture' or row.get('last_action_capture_verified') is True, 'A last-action PNG requires an explicit verified endpoint binding')
        image = Path(row['screenshot'])
        image = image.resolve() if image.is_absolute() else (manifest.parent / image).resolve()
        need(sha(image) == row['screenshot_sha256'], 'Original screenshot hash differs')
        with image.open('rb') as stream:
            header = stream.read(24)
        need(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR', 'Original PNG required')
        size = struct.unpack('>II', header[16:24])
        need(size in ((1280, 1800), (1280, 1657)), 'Supported archival PNG geometry is1280x1800 or1280x1657; never resize historical evidence')
        from PIL import Image
        with Image.open(image) as opened:
            need(opened.format == 'PNG' and opened.size == size, 'Malformed PNG')
            opened.load()
        request = judge.build_judge_request(row['task'], row['actions'], image)
        reserve, bound, long_context = judge.judge_reservation(request)
        row.update(screenshot=str(image), screenshot_dimensions=list(size), reserved_usd=str(reserve), input_token_upper_bound=bound, long_context=long_context)
        inputs[str(image)] = row['screenshot_sha256']
        for binding in row.get('evidence_bindings', []):
            path = Path(binding['path'])
            path = path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()
            need(sha(path) == binding['sha256'], 'Endpoint action/provenance evidence differs')
            inputs[str(path)] = binding['sha256']
        rows.append(row)
    return rows, inputs


def plan(args):
    need(type(args.judge_calls) is int and args.judge_calls > 0, 'Positive judge call cap required')
    need(type(args.wall_seconds) is int and args.wall_seconds > 0, 'Positive wall cap required')
    need(type(args.concurrency) is int and 1 <= args.concurrency <= 8, 'HTTP concurrency must be1 through8')
    judge, _, sources = science()
    rows, inputs = endpoints(args.manifest, judge)
    eligible = [row for row in rows if row['eligible']]
    return {'schema': 'portable-terra-rejudge-v1', 'execute': args.execute, 'judge_protocol': judge.protocol(), 'judge_usd': positive_money(args.judge_usd),
            'judge_calls': args.judge_calls, 'wall_seconds': args.wall_seconds, 'concurrency': args.concurrency, 'actor_calls': 0,
            'runner_sha256': sha(__file__), 'frozen_sources': sources, 'input_manifest_sha256': sha(args.manifest), 'input_sha256': inputs, 'episodes': rows,
            'eligible': len(eligible), 'original_invalid_preserved': sum(not row['original_valid'] for row in rows),
            'sum_individual_full_reservations_usd': str(sum((Decimal(row['reserved_usd']) for row in eligible), Decimal(0))),
            'automatic_retries': False, 'wall_policy': 'Parent kills and awaits its sole judge process at the wall deadline; outstanding reservations remain fully charged.'}


def check_inputs(record):
    need(record['runner_sha256'] == sha(__file__), 'Rejudge runner changed')
    for path, expected in record['input_sha256'].items():
        need(sha(path) == expected, 'Original saved evidence changed: ' + path)
    for relative, expected in record['frozen_sources'].items():
        need(sha(ROOT / relative) == expected, 'Frozen judge/budget source changed')


def cache_environment(root):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TMPDIR='/tmp', DO_NOT_TRACK='1')
    for name in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME', 'WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR'):
        path = root / 'runtime-cache' / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def summarize(root, record, budget_module):
    results = {}
    for path in (root / 'episodes').glob('*/result.json'):
        value = read(path)
        need(value['id'] not in results, 'Duplicate saved judgment')
        results[value['id']] = value
    cells = {}
    for row in record['episodes']:
        key = json.dumps(row['cohort'], sort_keys=True)
        cell = cells.setdefault(key, {'cohort': row['cohort'], 'requested': 0, 'original_invalid_preserved': 0, 'eligible': 0, 'recorded': 0, 'valid_judgments': 0, 'invalid_judgments': 0, 'successes': 0})
        cell['requested'] += 1
        cell['original_invalid_preserved'] += int(not row['original_valid'])
        cell['eligible'] += int(row['eligible'])
        if row['id'] in results:
            value = results[row['id']]
            cell['recorded'] += 1
            cell['valid_judgments'] += int(value['valid'])
            cell['invalid_judgments'] += int(not value['valid'])
            cell['successes'] += int(value['valid'] and value['reward'] == 1)
    for cell in cells.values():
        cell['missing_judgments'] = cell['eligible'] - cell['recorded']
        cell['original_valid_ineligible'] = cell['requested'] - cell['original_invalid_preserved'] - cell['eligible']
        cell['valid_only_success_rate'] = cell['successes'] / cell['valid_judgments'] if cell['valid_judgments'] else None
    pool = budget_module.ReservationBudget(root / 'budget/judge', record['judge_usd'], record['judge_calls']).snapshot()
    # Physical IDs, local paths and request metadata stay in the private ledger.
    pool = {key: value for key, value in pool.items() if key != 'requests'}
    summary = {'schema': 'portable-terra-rejudge-summary-v1', 'judge_protocol': record['judge_protocol'], 'input_manifest_sha256': record['input_manifest_sha256'],
               'runner_sha256': record['runner_sha256'], 'frozen_sources': record['frozen_sources'], 'cells': list(cells.values()), 'all_attempt_judge_pool': pool,
               'actor_calls': 0, 'originals_unchanged': True, 'complete': sum(cell['recorded'] for cell in cells.values()) == record['eligible'],
               'final_audit_complete': False, 'cost_scope': 'Separate new judge pool includes every known charge and full unknown/in-flight reserve; original actor costs and latencies are unchanged.'}
    write(root / 'summary.json', summary)
    return summary


def perform(root, record, client_factory):
    """One new process owns this complete bounded cohort; tests inject fake HTTP."""
    check_inputs(record)
    write(root / 'worker-started.json', {'epoch': time.time(), 'plan_sha256': sha(root / 'rejudge-plan.json')}, True)
    judge, budget_module, _ = science()
    budget = budget_module.ReservationBudget(root / 'budget/judge', record['judge_usd'], record['judge_calls'])
    queue = [row for row in record['episodes'] if row['eligible']][:record['judge_calls']]
    failures = []

    def one(row):
        directory = root / 'episodes' / row['id']
        write(directory / 'claimed.json', {'id': row['id'], 'attempt': 1, 'plan_sha256': sha(root / 'rejudge-plan.json'), 'epoch': time.time()}, True)
        value = {'id': row['id'], 'cohort': row['cohort'], 'valid': False, 'reward': None, 'actor_calls': 0, 'judge_attempt': 1}
        client = None
        try:
            need(sha(row['screenshot']) == row['screenshot_sha256'], 'Saved PNG changed')
            client = client_factory()
            need(client.max_retries == 0, 'SDK retries must be disabled')
            value['reward'] = judge.judge(client, budget, row['task'], row['actions'], row['screenshot'], directory)
            value['valid'] = True
        except BaseException as error:
            value.update(error_type=type(error).__name__, error=str(error))
            budget.halt('Judge attempt failed; preserve it without automatic retry')
            raise
        finally:
            if client is not None:
                client.close()
            value['request_artifacts_sha256'] = {str(path.relative_to(directory)): sha(path) for path in (directory / 'requests').glob('*.json')}
            write(directory / 'result.json', value, True)

    try:
        with ThreadPoolExecutor(max_workers=record['concurrency']) as executor:
            active = {}
            while queue or active:
                while queue and len(active) < record['concurrency'] and not failures and not budget.halted():
                    row = queue.pop(0)
                    active[executor.submit(one, row)] = row['id']
                if not active:
                    break
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    ident = active.pop(future)
                    try:
                        future.result()
                    except BaseException as error:
                        failures.append({'id': ident, 'error_type': type(error).__name__})
                write(root / 'heartbeat.json', {'epoch': time.time(), 'queued': len(queue), 'active': len(active), 'failures': failures})
    finally:
        check_inputs(record)
        summary = summarize(root, record, budget_module)
    return 0 if summary['complete'] and not failures else 1


def worker(root, env_file=None):
    root = Path(root).resolve()
    record = read(root / 'rejudge-plan.json')
    need(record['execute'] is True and record['output_root'] == str(root), 'An explicitly executed bounded plan is required')
    from dotenv import dotenv_values
    from openai import OpenAI
    keys = dotenv_values(env_file) if env_file else os.environ
    key = keys.get('JUDGE_API_KEY') or keys.get('OPENAI_API_KEY')
    need(bool(key), 'JUDGE_API_KEY or OPENAI_API_KEY required')
    return perform(root, record, lambda: OpenAI(api_key=key, max_retries=0, timeout=min(180, record['wall_seconds'])))


def run(args):
    record = plan(args)
    if not args.execute:
        print(json.dumps({key: value for key, value in record.items() if key not in ('episodes', 'input_sha256')}, indent=2))
        return 0
    root = args.output.resolve()
    need(not root.is_relative_to(Path('/gpfs/home/zixianma')), 'Choose a writable project or temporary output')
    need(not root.exists() and not root.is_symlink(), 'Use a new output; earlier attempts and claims are never overwritten or retried')
    root.mkdir(parents=True)
    record['output_root'] = str(root)
    write(root / 'rejudge-plan.json', record, True)
    arguments = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(root), '--execute']
    if args.env_file:
        arguments += ['--env-file', str(args.env_file.resolve())]
    timed_out = False
    with (root / 'worker.log').open('ab') as stream:
        try:
            code = subprocess.run(arguments, env=cache_environment(root), stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, timeout=args.wall_seconds).returncode
        except subprocess.TimeoutExpired:
            timed_out, code = True, 1  # subprocess.run kills and awaits the sole worker.
    check_inputs(record)
    _, budget_module, _ = science()
    summary = summarize(root, record, budget_module)
    write(root / 'execution.json', {'returncode': code, 'wall_deadline_terminated_worker': timed_out, 'unknown_or_inflight_reservations_never_released': True})
    print(json.dumps(summary, indent=2))
    return code


def add_arguments(parser):
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--judge-usd', type=positive_money, required=True)
    parser.add_argument('--judge-calls', type=int, required=True)
    parser.add_argument('--wall-seconds', type=int, required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--execute', action='store_true')


def main(argv=None):
    values = sys.argv[1:] if argv is None else argv
    if '--worker' in values:
        parser = argparse.ArgumentParser()
        parser.add_argument('--worker', type=Path, required=True)
        parser.add_argument('--execute', action='store_true', required=True)
        parser.add_argument('--env-file', type=Path)
        args = parser.parse_args(values)
        return worker(args.worker, args.env_file)
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(values)
    try:
        return run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
