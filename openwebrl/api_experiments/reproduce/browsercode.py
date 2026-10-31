"""Set up, check, run and analyze the frozen BrowserCode experiments."""
from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ALIASES = {
    'luna': 'browsercode_luna_effort',
    'claude': 'browsercode_claude',
    'gpt': 'browsercode_gpt_high',
    'gpt-luna-single': 'browsercode_luna_single_high',
    'gpt-crossbench': 'browsercode_gpt_high_wv_ds',
    'sol61-pair20': 'browsercode_sol61_pair20',
    'sol61-pair80': 'browsercode_sol61_pair80',
    'sol61-onecall20': 'browsercode_sol61_onecall20',
    'sol61-onecall80': 'browsercode_sol61_onecall80',
    'sol61-low-medium': 'browsercode_sol61_low_medium30',
    'luna-high100': 'browsercode_luna_high100',
    'luna-high100-initial': 'browsercode_luna_high100_initial',
    'luna-high100-recovery': 'browsercode_luna_high100_recovery4096',
}
MARKER = 'browsercode-workspace.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_money(value):
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError('Use a positive finite USD amount') from exc
    if not amount.is_finite() or amount <= 0:
        raise argparse.ArgumentTypeError('Use a positive finite USD amount')
    return str(amount)


def campaign(value):
    if value in ALIASES:
        return ALIASES[value]
    if value in ALIASES.values():
        return value
    raise argparse.ArgumentTypeError('Unknown campaign; choose ' + ', '.join(ALIASES))


def writable(path):
    path = Path(path).resolve()
    if path.is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError('Choose a writable project or temporary directory')
    return path


def call(python, script, *arguments):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([str(python), str(HERE / script), *map(str, arguments)], env=env).returncode


def browsercode_preflight_only(runtime):
    """Restrict a historical mechanical check to the selected public executor.

    Episode/model/judge functions remain unchanged. Historical setup also pins
    its declared viewport in the active session, independently of other adapters.
    Record changes and refresh both generated runtime inventories before release.
    """
    runtime = Path(runtime)
    worker = runtime / 'worker.py'
    text = worker.read_text()
    lines = text.splitlines(keepends=True)
    edits = []
    tree = ast.parse(text)
    preflight = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'preflight')
    for node in ast.walk(preflight):
        if isinstance(node, ast.For) and isinstance(node.iter, (ast.List, ast.Tuple)):
            values = [item.value for item in node.iter.elts if isinstance(item, ast.Constant)]
            if 'browsercode' in values and values != ['browsercode']:
                if node.iter.lineno != node.iter.end_lineno:
                    raise ValueError('Unsupported multiline preflight inventory')
                edits.append((node.iter.lineno - 1, node.iter.col_offset, node.iter.end_col_offset))
    before = digest(worker)
    for line, start, end in sorted(edits, reverse=True):
        lines[line] = lines[line][:start] + "['browsercode']" + lines[line][end:]
    adapted = ''.join(lines)
    pinned_viewport = False
    if any(isinstance(node, ast.FunctionDef) and node.name == 'setup_native' for node in tree.body) and 'viewport-fixed' not in adapted:
        anchor = "target=live_page(ws,navigate=item['start_url']);local.start(ws,target);result={'observation':{'success':True},'native_browsercode_bound':True}"
        if adapted.count(anchor) != 1:
            raise ValueError('Historical BrowserCode viewport setup anchor changed')
        replacement = "\n".join([
            "target=live_page(ws,navigate=item['start_url']);local.start(ws,target)",
            "  viewport={'code':'await session.Emulation.setDeviceMetricsOverride({width:1280,height:1800,deviceScaleFactor:1,mobile:false}); console.log(\"viewport-fixed\")','description':'Pin the frozen 1280x1800 viewport in the BrowserCode session','timeout':30000}",
            "  checked=local.execute({'command':json.dumps(viewport)})",
            "  if not checked['observation']['success'] or 'viewport-fixed' not in checked['output']:raise RuntimeError('Could not pin BrowserCode viewport')",
            "  write_json(local._workspace_dir().parent/'browsercode-viewport.json',checked)",
            "  result={'observation':{'success':True},'native_browsercode_bound':True,'viewport':'1280x1800'}",
        ])
        adapted = adapted.replace(anchor, replacement)
        pinned_viewport = True
    if adapted != text:
        compile(adapted, str(worker), 'exec')
        worker.write_text(adapted)
    build = read(runtime / 'portable-build.json')
    build['public_executor'] = 'browsercode'
    if edits or pinned_viewport:
        build['adaptations'].append({'file': 'worker.py', 'reason': 'Check only BrowserCode; pin the declared 1280x1800 viewport in its active session independently of other startup adapters. Episode/model/judge functions unchanged.',
                                     'original_sha256': before, 'adapted_sha256': digest(worker)})
    write(runtime / 'portable-build.json', build)
    manifest = read(runtime / 'source-manifest.json')
    manifest['source_hashes'][str(worker.resolve())] = digest(worker)
    manifest['source_hashes'][str((runtime / 'portable-build.json').resolve())] = digest(runtime / 'portable-build.json')
    write(runtime / 'source-manifest.json', manifest)
    write(runtime / 'execution-source-manifest.json', manifest)


def workspace(path):
    path = Path(path).resolve()
    record = read(path / MARKER)
    if record.get('schema') != 'browsercode-workspace-v1':
        raise ValueError('Use a workspace created by browsercode.py setup')
    runtime = path / 'runtime'
    if not runtime.is_dir() or runtime.is_symlink():
        raise ValueError('Prepared runtime is missing or was replaced')
    python = Path(record['python'])
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('The installed Python interpreter is unavailable')
    return runtime, python


def provider(runtime, arm):
    plan = read(Path(runtime) / 'campaign-plan.json')
    config = read(Path(runtime) / 'runtime-config.json')
    selected = next((row for row in plan['arms'] if row['arm'] == arm), None)
    if selected is None:
        raise ValueError('Arm must be one of: ' + ', '.join(row['arm'] for row in plan['arms']))
    model = selected['model']
    if model in ('claude-opus-5-5', 'claude-sonnet-5-5') and config.get('portable_campaign') == 'browsercode_claude':
        return 'claude'
    if model == 'gpt-6-luna' and config.get('portable_campaign') == 'browsercode_luna_effort':
        return 'luna'
    if model in ('gpt-6-luna', 'gpt-6-sol', 'gpt-6.1-sol'):
        return 'gpt'
    raise ValueError('Prepared campaign has an unsupported model adapter')


def setup(args):
    target = writable(args.workspace)
    if target.exists() and any(path.name != 'dependencies' for path in target.iterdir()):
        raise ValueError('Use a new workspace; prepared experiments are never overwritten')
    dependencies = args.dependencies.resolve() if args.dependencies else target / 'dependencies/dependencies.json'
    if args.dependencies and not dependencies.is_file():
        raise ValueError('Dependency manifest does not exist')
    target.mkdir(parents=True, exist_ok=True)
    if not args.dependencies:
        spec = importlib.util.spec_from_file_location('browsercode_installer', HERE / 'install.py')
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        installer.install(target / 'dependencies', profile='browsercode')
    installed = read(dependencies)
    python = Path(installed['python'])
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Dependency manifest has no usable installed Python')
    code = call(python, 'prepare_browsercode.py', '--campaign', args.campaign, '--dependencies', dependencies, '--output', target / 'runtime')
    if code:
        return code
    browsercode_preflight_only(target / 'runtime')
    write(target / MARKER, {'schema': 'browsercode-workspace-v1', 'campaign': args.campaign,
                           'dependencies': str(dependencies), 'python': str(python), 'executor': 'browsercode'})
    print('Workspace ready: ' + str(target))
    return 0


def run(args):
    runtime, python = workspace(args.workspace)
    adapter = provider(runtime, args.arm)
    forwarded = ['--runtime', runtime, '--arm', args.arm, '--harness', 'browsercode', '--output', writable(args.output),
                 '--limit', args.limit, '--model-usd', args.model_usd, '--judge-usd', args.judge_usd,
                 '--model-calls', args.model_calls, '--judge-calls', args.judge_calls, '--wandb-mode', args.wandb_mode]
    for name in ('benchmark', 'browser_starts', 'env_file'):
        value = getattr(args, name)
        if value is not None:
            forwarded += ['--' + name.replace('_', '-'), value]
    if args.execute:
        forwarded.append('--execute')
    return call(python, 'eval_' + adapter + '.py', *forwarded)


def check(args):
    runtime, python = workspace(args.workspace)
    flags = ['--browser-preflight'] if args.browser_preflight else []
    return call(python, 'verify_runtime.py', '--runtime', runtime, *flags)


def report(args):
    for item in args.run:
        if '=' not in item:
            raise ValueError('--run requires NAME=OUTPUT_DIRECTORY')
        _, directory = item.split('=', 1)
        plan = read(Path(directory) / 'reproduction-plan.json')
        if plan.get('harness', 'browsercode') != 'browsercode':
            raise ValueError('The BrowserCode report accepts BrowserCode outputs only')
    forwarded = ['--output', writable(args.output), '--seed', args.seed, '--bootstrap-draws', args.bootstrap_draws]
    for item in args.run:
        forwarded += ['--run', item]
    if args.markdown:
        forwarded.append('--markdown')
    return call(sys.executable, 'report.py', *forwarded)


def rejudge(args):
    forwarded = ['--manifest', args.manifest, '--output', writable(args.output), '--judge-usd', args.judge_usd,
                 '--judge-calls', args.judge_calls, '--wall-seconds', args.wall_seconds, '--concurrency', args.concurrency]
    if args.env_file:
        forwarded += ['--env-file', args.env_file]
    if args.execute:
        forwarded.append('--execute')
    return call(sys.executable, 'rejudge.py', *forwarded)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    s = commands.add_parser('setup', help='Install dependencies and prepare one experiment; no browser or model calls')
    s.add_argument('--workspace', type=Path, required=True)
    s.add_argument('--campaign', type=campaign, required=True, metavar='CAMPAIGN', help='luna, claude, gpt, gpt-luna-single or gpt-crossbench; historical experiment keys also accepted')
    s.add_argument('--dependencies', type=Path, help='Reuse an existing dependency manifest without downloads')
    s.set_defaults(action=setup)
    c = commands.add_parser('check', help='Check real imports, optionally a local browser; zero model calls')
    c.add_argument('--workspace', type=Path, required=True)
    c.add_argument('--browser-preflight', action='store_true')
    c.set_defaults(action=check)
    r = commands.add_parser('run', help='Print a bounded experiment plan; add --execute to run')
    r.add_argument('--workspace', type=Path, required=True)
    r.add_argument('--arm', required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--model-usd', type=positive_money, required=True)
    r.add_argument('--judge-usd', type=positive_money, required=True)
    r.add_argument('--limit', type=int, default=1)
    r.add_argument('--benchmark', choices=('online-mind2web', 'webvoyager', 'deepshop'))
    r.add_argument('--model-calls', type=int, default=10000)
    r.add_argument('--judge-calls', type=int, default=1100)
    r.add_argument('--browser-starts', type=int)
    r.add_argument('--env-file', type=Path)
    r.add_argument('--wandb-mode', choices=('offline', 'online', 'disabled'), default='offline')
    r.add_argument('--execute', action='store_true')
    r.set_defaults(action=run)
    a = commands.add_parser('report', help='Analyze saved BrowserCode outputs and paired outcomes')
    a.add_argument('--run', action='append', required=True, help='NAME=OUTPUT_DIRECTORY; repeat for comparisons')
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--seed', type=int, default=20261029)
    a.add_argument('--bootstrap-draws', type=int, default=10000)
    a.add_argument('--markdown', action='store_true')
    a.set_defaults(action=report)
    j = commands.add_parser('rejudge', help='Regrade saved endpoints with Terra; no actor/browser execution; dry-run by default')
    j.add_argument('--manifest', type=Path, required=True)
    j.add_argument('--output', type=Path, required=True)
    j.add_argument('--judge-usd', type=positive_money, required=True)
    j.add_argument('--judge-calls', type=int, required=True)
    j.add_argument('--wall-seconds', type=int, required=True)
    j.add_argument('--concurrency', type=int, required=True)
    j.add_argument('--env-file', type=Path)
    j.add_argument('--execute', action='store_true')
    j.set_defaults(action=rejudge)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        return args.action(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
