"""Independent pinned official WebJudge; only local gateway is permitted.

Invoke in one subprocess per case: official utils/module imports are process local.
The gateway owns durable physical-call accounting, no retries, and budget caps.
"""
import argparse
import ast
import asyncio
import hashlib
import importlib.util
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/serial-vs-parallel-reasoning-20261013/official-webjudge-regrade-20261020-preparation-v1')
MODEL = 'gpt-4o-2024-08-06'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def read(path):
    return json.loads(Path(path).read_text())

def strict_status(text):
    found = re.findall(r'^\s*Status\s*:\s*["\']?(success|failure)["\']?\s*$', text or '', re.I | re.M)
    return int(found[0].lower() == 'success') if len(found) == 1 else None

def official_status(text):
    try:
        return int('success' in text.lower().split('status:')[1])
    except Exception:
        return 0

def manual_capture_provenance(episode, rows, target):
    """Bind an unlisted raw page capture to pinned recorded browser-step code."""
    for row in rows:
        ref = row.get('code_path')
        if not ref:
            continue
        code = (episode / ref).resolve()
        if not code.is_relative_to(episode) or not code.is_file():
            raise ValueError('Manual capture code is missing or escapes episode')
        raw = code.read_bytes()
        if sha(raw) != row.get('code_sha256'):
            raise ValueError('Manual capture code hash mismatch')
        env = {'workspace': str(episode)}
        def value(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name):
                return env.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left, right = value(node.left), value(node.right)
                return str(Path(left) / right) if left is not None and right is not None else None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('Path', 'str') and len(node.args) == 1:
                return value(node.args[0])
            return None
        tree = ast.parse(raw.decode())
        for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0))):
            if isinstance(node, ast.Assign):
                resolved = value(node.value)
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        env[name.id] = resolved
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'screenshot' and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'page'):
                continue
            raw_path = next((value(k.value) for k in node.keywords if k.arg == 'path'), None)
            if raw_path is not None and (episode / raw_path).resolve() == target:
                return {'browser_step': row['browser_step'], 'code_path': str(code.relative_to(episode)),
                        'code_sha256': sha(raw), 'line': node.lineno, 'capture_api': 'page.screenshot'}
    raise ValueError('Current-folder image lacks recorded raw page.screenshot provenance')


def evidence_v2(task, episode_dir):
    episode = Path(episode_dir).resolve()
    manifest = episode / 'browser-steps.jsonl'
    # Unlike the permissive native reader, never silently drop a corrupt record.
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Non-object browser step')
    numbers = [int(row['browser_step']) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('Browser steps are unordered or duplicated')
    actions, captures, missing = [], [], []
    seen = set()
    for row in rows:
        action = row.get('action')
        if not isinstance(action, str):
            raise ValueError('Browser step lacks recorded action text')
        actions.append(action)
        ref = row.get('screenshot_path')
        if not ref:
            if row.get('screenshot_error'):
                missing.append({'browser_step': row['browser_step'], 'reason': 'capture_error'})
            continue
        path = (episode / ref).resolve()
        if not path.is_relative_to(episode):
            raise ValueError('Capture escapes current episode')
        if not path.is_file():
            missing.append({'browser_step': row['browser_step'], 'reason': 'referenced_file_missing', 'path': ref})
            continue
        if path in seen:
            raise ValueError('Capture path reused by multiple steps')
        seen.add(path)
        captures.append({'browser_step': row['browser_step'], 'path': str(path), 'sha256': sha(path.read_bytes())})
    # Released native task_screenshot_paths includes every supported image directly
    # under screenshots/, including actor-requested raw page.screenshot captures.
    # Archived images are deliberately excluded; missing references stay inventoried.
    screenshots = episode / 'screenshots'
    if screenshots.is_dir():
        for candidate in sorted(screenshots.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            if not candidate.is_file() or candidate.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            path = candidate.resolve()
            if not path.is_relative_to(episode):
                raise ValueError('Capture escapes current episode')
            if path in seen:
                continue
            provenance = manual_capture_provenance(episode, rows, path)
            seen.add(path)
            captures.append({'browser_step': provenance['browser_step'], 'path': str(path),
                             'sha256': sha(path.read_bytes()), 'manual_capture_provenance': provenance})
        # Keep this campaign's chronological external-export order; native
        # task_screenshot_paths enumerates the same current folder alphabetically.
        captures.sort(key=lambda c: (int(c['browser_step']), Path(c['path']).name))
    if not captures:
        raise ValueError('No recorded current-episode screenshot')
    trajectory = episode / 'trajectory.json'
    data = read(trajectory)
    answer = data.get('info', {}).get('submission', '')
    if not isinstance(answer, str):
        raise ValueError('Native submission is not text')
    if (episode / 'final-answer.json').exists():
        final = read(episode / 'final-answer.json').get('final_answer')
        if final != answer:
            raise ValueError('Final answer differs from native submission')
    if answer:
        actions.append('Final answer: ' + answer)
    return {'task': task['intent'], 'action_history': actions, 'captures': captures,
            'browser_steps_sha256': sha(manifest.read_bytes()), 'trajectory_sha256': sha(trajectory.read_bytes()),
            'browser_step_count': len(rows), 'final_answer': answer,
            'failed_capture_inventory': missing}

def evidence_v1(task, episode_dir):
    episode = Path(episode_dir).resolve()
    manifest = episode / 'browser-steps.jsonl'
    # Unlike the permissive native reader, never silently drop a corrupt record.
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Non-object browser step')
    numbers = [int(row['browser_step']) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('Browser steps are unordered or duplicated')
    actions, captures, missing = [], [], []
    seen = set()
    for row in rows:
        action = row.get('action')
        if not isinstance(action, str):
            raise ValueError('Browser step lacks recorded action text')
        actions.append(action)
        ref = row.get('screenshot_path')
        if not ref:
            if row.get('screenshot_error'):
                missing.append({'browser_step': row['browser_step'], 'reason': 'capture_error'})
            continue
        path = (episode / ref).resolve()
        if not path.is_relative_to(episode):
            raise ValueError('Capture escapes current episode')
        if not path.is_file():
            raise ValueError('Referenced capture is missing')
        if path in seen:
            raise ValueError('Capture path reused by multiple steps')
        seen.add(path)
        captures.append({'browser_step': row['browser_step'], 'path': str(path), 'sha256': sha(path.read_bytes())})
    if not captures:
        raise ValueError('No recorded current-episode screenshot')
    trajectory = episode / 'trajectory.json'
    data = read(trajectory)
    answer = data.get('info', {}).get('submission', '')
    if not isinstance(answer, str):
        raise ValueError('Native submission is not text')
    if (episode / 'final-answer.json').exists():
        final = read(episode / 'final-answer.json').get('final_answer')
        if final != answer:
            raise ValueError('Final answer differs from native submission')
    if answer:
        actions.append('Final answer: ' + answer)
    return {'task': task['intent'], 'action_history': actions, 'captures': captures,
            'browser_steps_sha256': sha(manifest.read_bytes()), 'trajectory_sha256': sha(trajectory.read_bytes()),
            'browser_step_count': len(rows), 'final_answer': answer,
            'failed_capture_inventory': missing}

def manual_capture_provenance_alias(episode, rows, target):
    """Bind an unlisted raw page capture to pinned recorded browser-step code."""
    for row in rows:
        ref = row.get('code_path')
        if not ref:
            continue
        code = (episode / ref).resolve()
        if not code.is_relative_to(episode) or not code.is_file():
            raise ValueError('Manual capture code is missing or escapes episode')
        raw = code.read_bytes()
        if sha(raw) != row.get('code_sha256'):
            raise ValueError('Manual capture code hash mismatch')
        env = {'workspace': str(episode)}
        object_types = {'page': 'page', 'context': 'context', 'browser': 'browser'}
        def object_type(node):
            if isinstance(node, ast.Await):
                return object_type(node.value)
            if isinstance(node, ast.Name):
                return object_types.get(node.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = object_type(node.func.value)
                if node.func.attr == 'new_context' and receiver == 'browser':
                    return 'context'
                if node.func.attr == 'new_page' and receiver in ('browser', 'context'):
                    return 'page'
            return None
        def value(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name):
                return env.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left, right = value(node.left), value(node.right)
                return str(Path(left) / right) if left is not None and right is not None else None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('Path', 'str') and len(node.args) == 1:
                return value(node.args[0])
            return None
        tree = ast.parse(raw.decode())
        for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0))):
            if isinstance(node, ast.Assign):
                resolved = value(node.value)
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        env[name.id] = resolved
                        object_types[name.id] = object_type(node.value)
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'screenshot' and isinstance(node.func.value, ast.Name)
                    and object_type(node.func.value) == 'page'):
                continue
            raw_path = next((value(k.value) for k in node.keywords if k.arg == 'path'), None)
            if raw_path is not None and (episode / raw_path).resolve() == target:
                return {'browser_step': row['browser_step'], 'code_path': str(code.relative_to(episode)),
                        'code_sha256': sha(raw), 'line': node.lineno, 'capture_api': 'page.screenshot', 'receiver_alias': node.func.value.id}
    raise ValueError('Current-folder image lacks recorded raw page.screenshot provenance')


def automatic_capture_provenance(episode, rows, target):
    """Recover only independently bound native emitted capture paths, never images."""
    proof_path = ROOT / 'repairs/available-captures-v2/independent-c004-original-captures.json'
    if not proof_path.is_file():
        raise ValueError('No independently bound automatic capture provenance')
    proof = read(proof_path)
    source = ROOT / 'upstream/src/cuawright/webwright/tools/browser_session.py'
    pinned = 'e618afb3e90d0c7584791fd36c19226f6389fb32672a097830f8b5fc347822e6'
    if (proof.get('verified') is not True or proof.get('case') != episode.name
            or proof.get('browser_session_source_sha256') != pinned or sha(source.read_bytes()) != pinned):
        raise ValueError('Automatic capture provenance source/case mismatch')
    for step, binding in proof['bindings'].items():
        debug_path = episode / ('debug/steps/step_%04d.json' % int(step))
        debug = read(debug_path)
        command = debug.get('bash_command', '')
        if not command.startswith('python -m cuawright.webwright.tools.browser_session step '):
            continue
        request_path = Path(binding['path']).resolve()
        if not request_path.is_relative_to(ROOT / 'meter'):
            raise ValueError('Automatic proof request escapes meter')
        raw = request_path.read_bytes()
        if sha(raw) != binding['sha256']:
            raise ValueError('Automatic proof physical request changed')
        calls = read(request_path).get('input', [])
        emitted = [item.get('output', '') for item in calls if item.get('type') == 'function_call_output'
                   and item.get('call_id') == binding['call_id']]
        for output in debug.get('outputs', []):
            text = output.get('output', '')
            if not text or not any(text in physical for physical in emitted):
                continue
            observation = json.loads(text)
            ref = observation.get('screenshot_path')
            if not ref or (episode / ref).resolve() != target:
                continue
            number = observation['browser_step']
            if target.name != 'browser_step_%04d.png' % int(number):
                raise ValueError('Automatic capture does not use pinned default path')
            row = next((row for row in rows if row['browser_step'] == number), None)
            if row is None:
                raise ValueError('Automatic capture missing browser step')
            code = (episode / row['code_path']).resolve()
            if not code.is_relative_to(episode) or sha(code.read_bytes()) != row['code_sha256']:
                raise ValueError('Automatic capture selected code changed')
            return {'browser_step': number, 'capture_api': 'pinned browser_session page.screenshot',
                    'debug_path': str(debug_path.relative_to(episode)), 'debug_sha256': sha(debug_path.read_bytes()),
                    'physical_request_path': str(request_path.relative_to(ROOT)), 'physical_request_sha256': sha(raw),
                    'call_id': binding['call_id'], 'code_path': row['code_path'], 'code_sha256': row['code_sha256'],
                    'command_sha256': sha(command.encode()), 'native_step_success': observation.get('success'),
                    'pinned_capture_source_sha256': pinned, 'independent_proof_sha256': sha(proof_path.read_bytes())}
    raise ValueError('Image lacks independently emitted automatic capture provenance')


def evidence_v2_alias(task, episode_dir):
    episode = Path(episode_dir).resolve()
    manifest = episode / 'browser-steps.jsonl'
    # Unlike the permissive native reader, never silently drop a corrupt record.
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Non-object browser step')
    numbers = [int(row['browser_step']) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('Browser steps are unordered or duplicated')
    actions, captures, missing = [], [], []
    seen = set()
    for row in rows:
        action = row.get('action')
        if not isinstance(action, str):
            raise ValueError('Browser step lacks recorded action text')
        actions.append(action)
        ref = row.get('screenshot_path')
        if not ref:
            if row.get('screenshot_error'):
                missing.append({'browser_step': row['browser_step'], 'reason': 'capture_error'})
            continue
        path = (episode / ref).resolve()
        if not path.is_relative_to(episode):
            raise ValueError('Capture escapes current episode')
        if not path.is_file():
            missing.append({'browser_step': row['browser_step'], 'reason': 'referenced_file_missing', 'path': ref})
            continue
        if path in seen:
            raise ValueError('Capture path reused by multiple steps')
        seen.add(path)
        captures.append({'browser_step': row['browser_step'], 'path': str(path), 'sha256': sha(path.read_bytes())})
    # Released native task_screenshot_paths includes every supported image directly
    # under screenshots/, including actor-requested raw page.screenshot captures.
    # Archived images are deliberately excluded; missing references stay inventoried.
    screenshots = episode / 'screenshots'
    if screenshots.is_dir():
        for candidate in sorted(screenshots.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            if not candidate.is_file() or candidate.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            path = candidate.resolve()
            if not path.is_relative_to(episode):
                raise ValueError('Capture escapes current episode')
            if path in seen:
                continue
            try:
                # Preserve the already audited v2 direct-page evidence rule.
                provenance = manual_capture_provenance(episode, rows, path)
            except ValueError as error:
                if str(error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                    raise
                try:
                    provenance = manual_capture_provenance_alias(episode, rows, path)
                except ValueError as alias_error:
                    if str(alias_error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                        raise
                    provenance = automatic_capture_provenance(episode, rows, path)
            seen.add(path)
            captures.append({'browser_step': provenance['browser_step'], 'path': str(path),
                             'sha256': sha(path.read_bytes()), 'manual_capture_provenance': provenance})
        # Keep this campaign's chronological external-export order; native
        # task_screenshot_paths enumerates the same current folder alphabetically.
        captures.sort(key=lambda c: (int(c['browser_step']), Path(c['path']).name))
    if not captures:
        raise ValueError('No recorded current-episode screenshot')
    trajectory = episode / 'trajectory.json'
    data = read(trajectory)
    answer = data.get('info', {}).get('submission', '')
    if not isinstance(answer, str):
        raise ValueError('Native submission is not text')
    if (episode / 'final-answer.json').exists():
        final = read(episode / 'final-answer.json').get('final_answer')
        if final != answer:
            raise ValueError('Final answer differs from native submission')
    if answer:
        actions.append('Final answer: ' + answer)
    return {'task': task['intent'], 'action_history': actions, 'captures': captures,
            'browser_steps_sha256': sha(manifest.read_bytes()), 'trajectory_sha256': sha(trajectory.read_bytes()),
            'browser_step_count': len(rows), 'final_answer': answer,
            'failed_capture_inventory': missing}

def evidence_v2_alias_lf(task, episode_dir):
    episode = Path(episode_dir).resolve()
    manifest = episode / 'browser-steps.jsonl'
    # Unlike the permissive native reader, never silently drop a corrupt record.
    rows = [json.loads(line) for line in manifest.read_text().split('\n') if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Non-object browser step')
    numbers = [int(row['browser_step']) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('Browser steps are unordered or duplicated')
    actions, captures, missing = [], [], []
    seen = set()
    for row in rows:
        action = row.get('action')
        if not isinstance(action, str):
            raise ValueError('Browser step lacks recorded action text')
        actions.append(action)
        ref = row.get('screenshot_path')
        if not ref:
            if row.get('screenshot_error'):
                missing.append({'browser_step': row['browser_step'], 'reason': 'capture_error'})
            continue
        path = (episode / ref).resolve()
        if not path.is_relative_to(episode):
            raise ValueError('Capture escapes current episode')
        if not path.is_file():
            missing.append({'browser_step': row['browser_step'], 'reason': 'referenced_file_missing', 'path': ref})
            continue
        if path in seen:
            raise ValueError('Capture path reused by multiple steps')
        seen.add(path)
        captures.append({'browser_step': row['browser_step'], 'path': str(path), 'sha256': sha(path.read_bytes())})
    # Released native task_screenshot_paths includes every supported image directly
    # under screenshots/, including actor-requested raw page.screenshot captures.
    # Archived images are deliberately excluded; missing references stay inventoried.
    screenshots = episode / 'screenshots'
    if screenshots.is_dir():
        for candidate in sorted(screenshots.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            if not candidate.is_file() or candidate.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            path = candidate.resolve()
            if not path.is_relative_to(episode):
                raise ValueError('Capture escapes current episode')
            if path in seen:
                continue
            try:
                # Preserve the already audited v2 direct-page evidence rule.
                provenance = manual_capture_provenance(episode, rows, path)
            except ValueError as error:
                if str(error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                    raise
                try:
                    provenance = manual_capture_provenance_alias(episode, rows, path)
                except ValueError as alias_error:
                    if str(alias_error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                        raise
                    provenance = automatic_capture_provenance(episode, rows, path)
            seen.add(path)
            captures.append({'browser_step': provenance['browser_step'], 'path': str(path),
                             'sha256': sha(path.read_bytes()), 'manual_capture_provenance': provenance})
        # Keep this campaign's chronological external-export order; native
        # task_screenshot_paths enumerates the same current folder alphabetically.
        captures.sort(key=lambda c: (int(c['browser_step']), Path(c['path']).name))
    if not captures:
        raise ValueError('No recorded current-episode screenshot')
    trajectory = episode / 'trajectory.json'
    data = read(trajectory)
    answer = data.get('info', {}).get('submission', '')
    if not isinstance(answer, str):
        raise ValueError('Native submission is not text')
    if (episode / 'final-answer.json').exists():
        final = read(episode / 'final-answer.json').get('final_answer')
        if final != answer:
            raise ValueError('Final answer differs from native submission')
    if answer:
        actions.append('Final answer: ' + answer)
    return {'task': task['intent'], 'action_history': actions, 'captures': captures,
            'browser_steps_sha256': sha(manifest.read_bytes()), 'trajectory_sha256': sha(trajectory.read_bytes()),
            'browser_step_count': len(rows), 'final_answer': answer,
            'failed_capture_inventory': missing}


def manual_capture_provenance_syntax(episode, rows, target):
    """Bind an unlisted raw page capture to pinned recorded browser-step code."""
    for row in rows:
        ref = row.get('code_path')
        if not ref:
            continue
        code = (episode / ref).resolve()
        if not code.is_relative_to(episode) or not code.is_file():
            raise ValueError('Manual capture code is missing or escapes episode')
        raw = code.read_bytes()
        if sha(raw) != row.get('code_sha256'):
            raise ValueError('Manual capture code hash mismatch')
        env = {'workspace': str(episode)}
        def value(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name):
                return env.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left, right = value(node.left), value(node.right)
                return str(Path(left) / right) if left is not None and right is not None else None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('Path', 'str') and len(node.args) == 1:
                return value(node.args[0])
            return None
        try:
            tree = ast.parse(raw.decode())
        except SyntaxError:
            if (row.get('success') is False and row.get('error_kind') == 'action'
                    and 'SyntaxError:' in row.get('traceback', '')):
                continue  # Python compiles the complete snippet before any action.
            raise
        for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0))):
            if isinstance(node, ast.Assign):
                resolved = value(node.value)
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        env[name.id] = resolved
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'screenshot' and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'page'):
                continue
            raw_path = next((value(k.value) for k in node.keywords if k.arg == 'path'), None)
            if raw_path is not None and (episode / raw_path).resolve() == target:
                return {'browser_step': row['browser_step'], 'code_path': str(code.relative_to(episode)),
                        'code_sha256': sha(raw), 'line': node.lineno, 'capture_api': 'page.screenshot'}
    raise ValueError('Current-folder image lacks recorded raw page.screenshot provenance')


def manual_capture_provenance_alias_syntax(episode, rows, target):
    """Bind an unlisted raw page capture to pinned recorded browser-step code."""
    for row in rows:
        ref = row.get('code_path')
        if not ref:
            continue
        code = (episode / ref).resolve()
        if not code.is_relative_to(episode) or not code.is_file():
            raise ValueError('Manual capture code is missing or escapes episode')
        raw = code.read_bytes()
        if sha(raw) != row.get('code_sha256'):
            raise ValueError('Manual capture code hash mismatch')
        env = {'workspace': str(episode)}
        object_types = {'page': 'page', 'context': 'context', 'browser': 'browser'}
        def object_type(node):
            if isinstance(node, ast.Await):
                return object_type(node.value)
            if isinstance(node, ast.Name):
                return object_types.get(node.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = object_type(node.func.value)
                if node.func.attr == 'new_context' and receiver == 'browser':
                    return 'context'
                if node.func.attr == 'new_page' and receiver in ('browser', 'context'):
                    return 'page'
            return None
        def value(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name):
                return env.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left, right = value(node.left), value(node.right)
                return str(Path(left) / right) if left is not None and right is not None else None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('Path', 'str') and len(node.args) == 1:
                return value(node.args[0])
            return None
        try:
            tree = ast.parse(raw.decode())
        except SyntaxError:
            if (row.get('success') is False and row.get('error_kind') == 'action'
                    and 'SyntaxError:' in row.get('traceback', '')):
                continue  # Python compiles the complete snippet before any action.
            raise
        for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0))):
            if isinstance(node, ast.Assign):
                resolved = value(node.value)
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        env[name.id] = resolved
                        object_types[name.id] = object_type(node.value)
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'screenshot' and isinstance(node.func.value, ast.Name)
                    and object_type(node.func.value) == 'page'):
                continue
            raw_path = next((value(k.value) for k in node.keywords if k.arg == 'path'), None)
            if raw_path is not None and (episode / raw_path).resolve() == target:
                return {'browser_step': row['browser_step'], 'code_path': str(code.relative_to(episode)),
                        'code_sha256': sha(raw), 'line': node.lineno, 'capture_api': 'page.screenshot', 'receiver_alias': node.func.value.id}
    raise ValueError('Current-folder image lacks recorded raw page.screenshot provenance')


def evidence_v2_alias_lf_syntax(task, episode_dir):
    episode = Path(episode_dir).resolve()
    manifest = episode / 'browser-steps.jsonl'
    # Unlike the permissive native reader, never silently drop a corrupt record.
    rows = [json.loads(line) for line in manifest.read_text().split('\n') if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Non-object browser step')
    numbers = [int(row['browser_step']) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('Browser steps are unordered or duplicated')
    actions, captures, missing = [], [], []
    seen = set()
    for row in rows:
        action = row.get('action')
        if not isinstance(action, str):
            raise ValueError('Browser step lacks recorded action text')
        actions.append(action)
        ref = row.get('screenshot_path')
        if not ref:
            if row.get('screenshot_error'):
                missing.append({'browser_step': row['browser_step'], 'reason': 'capture_error'})
            continue
        path = (episode / ref).resolve()
        if not path.is_relative_to(episode):
            raise ValueError('Capture escapes current episode')
        if not path.is_file():
            missing.append({'browser_step': row['browser_step'], 'reason': 'referenced_file_missing', 'path': ref})
            continue
        if path in seen:
            raise ValueError('Capture path reused by multiple steps')
        seen.add(path)
        captures.append({'browser_step': row['browser_step'], 'path': str(path), 'sha256': sha(path.read_bytes())})
    # Released native task_screenshot_paths includes every supported image directly
    # under screenshots/, including actor-requested raw page.screenshot captures.
    # Archived images are deliberately excluded; missing references stay inventoried.
    screenshots = episode / 'screenshots'
    if screenshots.is_dir():
        for candidate in sorted(screenshots.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            if not candidate.is_file() or candidate.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            path = candidate.resolve()
            if not path.is_relative_to(episode):
                raise ValueError('Capture escapes current episode')
            if path in seen:
                continue
            try:
                # Preserve the already audited v2 direct-page evidence rule.
                provenance = manual_capture_provenance_syntax(episode, rows, path)
            except ValueError as error:
                if str(error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                    raise
                try:
                    provenance = manual_capture_provenance_alias_syntax(episode, rows, path)
                except ValueError as alias_error:
                    if str(alias_error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                        raise
                    provenance = automatic_capture_provenance(episode, rows, path)
            seen.add(path)
            captures.append({'browser_step': provenance['browser_step'], 'path': str(path),
                             'sha256': sha(path.read_bytes()), 'manual_capture_provenance': provenance})
        # Keep this campaign's chronological external-export order; native
        # task_screenshot_paths enumerates the same current folder alphabetically.
        captures.sort(key=lambda c: (int(c['browser_step']), Path(c['path']).name))
    if not captures:
        raise ValueError('No recorded current-episode screenshot')
    trajectory = episode / 'trajectory.json'
    data = read(trajectory)
    answer = data.get('info', {}).get('submission', '')
    if not isinstance(answer, str):
        raise ValueError('Native submission is not text')
    if (episode / 'final-answer.json').exists():
        final = read(episode / 'final-answer.json').get('final_answer')
        if final != answer:
            raise ValueError('Final answer differs from native submission')
    if answer:
        actions.append('Final answer: ' + answer)
    return {'task': task['intent'], 'action_history': actions, 'captures': captures,
            'browser_steps_sha256': sha(manifest.read_bytes()), 'trajectory_sha256': sha(trajectory.read_bytes()),
            'browser_step_count': len(rows), 'final_answer': answer,
            'failed_capture_inventory': missing}


def manual_capture_provenance_page_iterator(episode, rows, target):
    """Bind an unlisted raw page capture to pinned recorded browser-step code."""
    for row in rows:
        ref = row.get('code_path')
        if not ref:
            continue
        code = (episode / ref).resolve()
        if not code.is_relative_to(episode) or not code.is_file():
            raise ValueError('Manual capture code is missing or escapes episode')
        raw = code.read_bytes()
        if sha(raw) != row.get('code_sha256'):
            raise ValueError('Manual capture code hash mismatch')
        env = {'workspace': str(episode)}
        object_types = {'page': 'page', 'context': 'context', 'browser': 'browser'}
        def object_type(node):
            if isinstance(node, ast.Await):
                return object_type(node.value)
            if isinstance(node, ast.Name):
                return object_types.get(node.id)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'next' and len(node.args) == 1 and not node.keywords
                    and isinstance(node.args[0], ast.GeneratorExp)):
                generator = node.args[0]
                if len(generator.generators) == 1:
                    loop = generator.generators[0]
                    if (not loop.is_async and isinstance(loop.target, ast.Name)
                            and isinstance(generator.elt, ast.Name) and generator.elt.id == loop.target.id
                            and isinstance(loop.iter, ast.Attribute) and loop.iter.attr == 'pages'
                            and object_type(loop.iter.value) == 'context'):
                        return 'page'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = object_type(node.func.value)
                if node.func.attr == 'new_context' and receiver == 'browser':
                    return 'context'
                if node.func.attr == 'new_page' and receiver in ('browser', 'context'):
                    return 'page'
            return None
        def value(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name):
                return env.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left, right = value(node.left), value(node.right)
                return str(Path(left) / right) if left is not None and right is not None else None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('Path', 'str') and len(node.args) == 1:
                return value(node.args[0])
            return None
        try:
            tree = ast.parse(raw.decode())
        except SyntaxError:
            if (row.get('success') is False and row.get('error_kind') == 'action'
                    and 'SyntaxError:' in row.get('traceback', '')):
                continue  # Python compiles the complete snippet before any action.
            raise
        for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0))):
            if isinstance(node, ast.Assign):
                resolved = value(node.value)
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        env[name.id] = resolved
                        object_types[name.id] = object_type(node.value)
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'screenshot' and isinstance(node.func.value, ast.Name)
                    and object_type(node.func.value) == 'page'):
                continue
            raw_path = next((value(k.value) for k in node.keywords if k.arg == 'path'), None)
            if raw_path is not None and (episode / raw_path).resolve() == target:
                return {'browser_step': row['browser_step'], 'code_path': str(code.relative_to(episode)),
                        'code_sha256': sha(raw), 'line': node.lineno, 'capture_api': 'page.screenshot', 'receiver_alias': node.func.value.id}
    raise ValueError('Current-folder image lacks recorded raw page.screenshot provenance')


def evidence_v2_alias_lf_page_iterator(task, episode_dir):
    episode = Path(episode_dir).resolve()
    manifest = episode / 'browser-steps.jsonl'
    # Unlike the permissive native reader, never silently drop a corrupt record.
    rows = [json.loads(line) for line in manifest.read_text().split('\n') if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Non-object browser step')
    numbers = [int(row['browser_step']) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('Browser steps are unordered or duplicated')
    actions, captures, missing = [], [], []
    seen = set()
    for row in rows:
        action = row.get('action')
        if not isinstance(action, str):
            raise ValueError('Browser step lacks recorded action text')
        actions.append(action)
        ref = row.get('screenshot_path')
        if not ref:
            if row.get('screenshot_error'):
                missing.append({'browser_step': row['browser_step'], 'reason': 'capture_error'})
            continue
        path = (episode / ref).resolve()
        if not path.is_relative_to(episode):
            raise ValueError('Capture escapes current episode')
        if not path.is_file():
            missing.append({'browser_step': row['browser_step'], 'reason': 'referenced_file_missing', 'path': ref})
            continue
        if path in seen:
            raise ValueError('Capture path reused by multiple steps')
        seen.add(path)
        captures.append({'browser_step': row['browser_step'], 'path': str(path), 'sha256': sha(path.read_bytes())})
    # Released native task_screenshot_paths includes every supported image directly
    # under screenshots/, including actor-requested raw page.screenshot captures.
    # Archived images are deliberately excluded; missing references stay inventoried.
    screenshots = episode / 'screenshots'
    if screenshots.is_dir():
        for candidate in sorted(screenshots.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            if not candidate.is_file() or candidate.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            path = candidate.resolve()
            if not path.is_relative_to(episode):
                raise ValueError('Capture escapes current episode')
            if path in seen:
                continue
            try:
                # Preserve the already audited v2 direct-page evidence rule.
                provenance = manual_capture_provenance_syntax(episode, rows, path)
            except ValueError as error:
                if str(error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                    raise
                try:
                    provenance = manual_capture_provenance_page_iterator(episode, rows, path)
                except ValueError as alias_error:
                    if str(alias_error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                        raise
                    provenance = automatic_capture_provenance(episode, rows, path)
            seen.add(path)
            captures.append({'browser_step': provenance['browser_step'], 'path': str(path),
                             'sha256': sha(path.read_bytes()), 'manual_capture_provenance': provenance})
        # Keep this campaign's chronological external-export order; native
        # task_screenshot_paths enumerates the same current folder alphabetically.
        captures.sort(key=lambda c: (int(c['browser_step']), Path(c['path']).name))
    if not captures:
        raise ValueError('No recorded current-episode screenshot')
    trajectory = episode / 'trajectory.json'
    data = read(trajectory)
    answer = data.get('info', {}).get('submission', '')
    if not isinstance(answer, str):
        raise ValueError('Native submission is not text')
    if (episode / 'final-answer.json').exists():
        final = read(episode / 'final-answer.json').get('final_answer')
        if final != answer:
            raise ValueError('Final answer differs from native submission')
    if answer:
        actions.append('Final answer: ' + answer)
    return {'task': task['intent'], 'action_history': actions, 'captures': captures,
            'browser_steps_sha256': sha(manifest.read_bytes()), 'trajectory_sha256': sha(trajectory.read_bytes()),
            'browser_step_count': len(rows), 'final_answer': answer,
            'failed_capture_inventory': missing}


def manual_capture_provenance_path_join(episode, rows, target):
    """Bind an unlisted raw page capture to pinned recorded browser-step code."""
    import os
    for row in rows:
        ref = row.get('code_path')
        if not ref:
            continue
        code = (episode / ref).resolve()
        if not code.is_relative_to(episode) or not code.is_file():
            raise ValueError('Manual capture code is missing or escapes episode')
        raw = code.read_bytes()
        if sha(raw) != row.get('code_sha256'):
            raise ValueError('Manual capture code hash mismatch')
        env = {'workspace': str(episode)}
        object_types = {'page': 'page', 'context': 'context', 'browser': 'browser'}
        imported_os = set()
        def object_type(node):
            if isinstance(node, ast.Await):
                return object_type(node.value)
            if isinstance(node, ast.Name):
                return object_types.get(node.id)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'next' and len(node.args) == 1 and not node.keywords
                    and isinstance(node.args[0], ast.GeneratorExp)):
                generator = node.args[0]
                if len(generator.generators) == 1:
                    loop = generator.generators[0]
                    if (not loop.is_async and isinstance(loop.target, ast.Name)
                            and isinstance(generator.elt, ast.Name) and generator.elt.id == loop.target.id
                            and isinstance(loop.iter, ast.Attribute) and loop.iter.attr == 'pages'
                            and object_type(loop.iter.value) == 'context'):
                        return 'page'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = object_type(node.func.value)
                if node.func.attr == 'new_context' and receiver == 'browser':
                    return 'context'
                if node.func.attr == 'new_page' and receiver in ('browser', 'context'):
                    return 'page'
            return None
        def value(node):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'join'
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'path'
                    and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id in imported_os
                    and node.args and not node.keywords and all(not isinstance(arg, ast.Starred) for arg in node.args)):
                components = [value(arg) for arg in node.args]
                if all(isinstance(item, str) for item in components):
                    return os.path.join(*components)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name):
                return env.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left, right = value(node.left), value(node.right)
                return str(Path(left) / right) if left is not None and right is not None else None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('Path', 'str') and len(node.args) == 1:
                return value(node.args[0])
            return None
        try:
            tree = ast.parse(raw.decode())
        except SyntaxError:
            if (row.get('success') is False and row.get('error_kind') == 'action'
                    and 'SyntaxError:' in row.get('traceback', '')):
                continue  # Python compiles the complete snippet before any action.
            raise
        for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'os':
                        imported_os.add(alias.asname or 'os')
            if isinstance(node, ast.Assign):
                resolved = value(node.value)
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        imported_os.discard(name.id)
                        env[name.id] = resolved
                        object_types[name.id] = object_type(node.value)
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'screenshot' and isinstance(node.func.value, ast.Name)
                    and object_type(node.func.value) == 'page'):
                continue
            raw_path = next((value(k.value) for k in node.keywords if k.arg == 'path'), None)
            if raw_path is not None and (episode / raw_path).resolve() == target:
                return {'browser_step': row['browser_step'], 'code_path': str(code.relative_to(episode)),
                        'code_sha256': sha(raw), 'line': node.lineno, 'capture_api': 'page.screenshot', 'receiver_alias': node.func.value.id}
    raise ValueError('Current-folder image lacks recorded raw page.screenshot provenance')


def evidence_v2_alias_lf_path_join(task, episode_dir):
    episode = Path(episode_dir).resolve()
    manifest = episode / 'browser-steps.jsonl'
    # Unlike the permissive native reader, never silently drop a corrupt record.
    rows = [json.loads(line) for line in manifest.read_text().split('\n') if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Non-object browser step')
    numbers = [int(row['browser_step']) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('Browser steps are unordered or duplicated')
    actions, captures, missing = [], [], []
    seen = set()
    for row in rows:
        action = row.get('action')
        if not isinstance(action, str):
            raise ValueError('Browser step lacks recorded action text')
        actions.append(action)
        ref = row.get('screenshot_path')
        if not ref:
            if row.get('screenshot_error'):
                missing.append({'browser_step': row['browser_step'], 'reason': 'capture_error'})
            continue
        path = (episode / ref).resolve()
        if not path.is_relative_to(episode):
            raise ValueError('Capture escapes current episode')
        if not path.is_file():
            missing.append({'browser_step': row['browser_step'], 'reason': 'referenced_file_missing', 'path': ref})
            continue
        if path in seen:
            raise ValueError('Capture path reused by multiple steps')
        seen.add(path)
        captures.append({'browser_step': row['browser_step'], 'path': str(path), 'sha256': sha(path.read_bytes())})
    # Released native task_screenshot_paths includes every supported image directly
    # under screenshots/, including actor-requested raw page.screenshot captures.
    # Archived images are deliberately excluded; missing references stay inventoried.
    screenshots = episode / 'screenshots'
    if screenshots.is_dir():
        for candidate in sorted(screenshots.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            if not candidate.is_file() or candidate.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            path = candidate.resolve()
            if not path.is_relative_to(episode):
                raise ValueError('Capture escapes current episode')
            if path in seen:
                continue
            try:
                # Preserve the already audited v2 direct-page evidence rule.
                provenance = manual_capture_provenance_syntax(episode, rows, path)
            except ValueError as error:
                if str(error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                    raise
                try:
                    provenance = manual_capture_provenance_path_join(episode, rows, path)
                except ValueError as alias_error:
                    if str(alias_error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                        raise
                    provenance = automatic_capture_provenance(episode, rows, path)
            seen.add(path)
            captures.append({'browser_step': provenance['browser_step'], 'path': str(path),
                             'sha256': sha(path.read_bytes()), 'manual_capture_provenance': provenance})
        # Keep this campaign's chronological external-export order; native
        # task_screenshot_paths enumerates the same current folder alphabetically.
        captures.sort(key=lambda c: (int(c['browser_step']), Path(c['path']).name))
    if not captures:
        raise ValueError('No recorded current-episode screenshot')
    trajectory = episode / 'trajectory.json'
    data = read(trajectory)
    answer = data.get('info', {}).get('submission', '')
    if not isinstance(answer, str):
        raise ValueError('Native submission is not text')
    if (episode / 'final-answer.json').exists():
        final = read(episode / 'final-answer.json').get('final_answer')
        if final != answer:
            raise ValueError('Final answer differs from native submission')
    if answer:
        actions.append('Final answer: ' + answer)
    return {'task': task['intent'], 'action_history': actions, 'captures': captures,
            'browser_steps_sha256': sha(manifest.read_bytes()), 'trajectory_sha256': sha(trajectory.read_bytes()),
            'browser_step_count': len(rows), 'final_answer': answer,
            'failed_capture_inventory': missing}

def manual_capture_provenance_string_add(episode, rows, target):
    """Bind an unlisted raw page capture to pinned recorded browser-step code."""
    import os
    for row in rows:
        ref = row.get('code_path')
        if not ref:
            continue
        code = (episode / ref).resolve()
        if not code.is_relative_to(episode) or not code.is_file():
            raise ValueError('Manual capture code is missing or escapes episode')
        raw = code.read_bytes()
        if sha(raw) != row.get('code_sha256'):
            raise ValueError('Manual capture code hash mismatch')
        env = {'workspace': str(episode)}
        object_types = {'page': 'page', 'context': 'context', 'browser': 'browser'}
        imported_os = set()
        def object_type(node):
            if isinstance(node, ast.Await):
                return object_type(node.value)
            if isinstance(node, ast.Name):
                return object_types.get(node.id)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'next' and len(node.args) == 1 and not node.keywords
                    and isinstance(node.args[0], ast.GeneratorExp)):
                generator = node.args[0]
                if len(generator.generators) == 1:
                    loop = generator.generators[0]
                    if (not loop.is_async and isinstance(loop.target, ast.Name)
                            and isinstance(generator.elt, ast.Name) and generator.elt.id == loop.target.id
                            and isinstance(loop.iter, ast.Attribute) and loop.iter.attr == 'pages'
                            and object_type(loop.iter.value) == 'context'):
                        return 'page'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = object_type(node.func.value)
                if node.func.attr == 'new_context' and receiver == 'browser':
                    return 'context'
                if node.func.attr == 'new_page' and receiver in ('browser', 'context'):
                    return 'page'
            return None
        def value(node):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'join'
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'path'
                    and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id in imported_os
                    and node.args and not node.keywords and all(not isinstance(arg, ast.Starred) for arg in node.args)):
                components = [value(arg) for arg in node.args]
                if all(isinstance(item, str) for item in components):
                    return os.path.join(*components)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name):
                return env.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left, right = value(node.left), value(node.right)
                return left + right if isinstance(left, str) and isinstance(right, str) else None
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left, right = value(node.left), value(node.right)
                return str(Path(left) / right) if left is not None and right is not None else None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('Path', 'str') and len(node.args) == 1:
                return value(node.args[0])
            return None
        try:
            tree = ast.parse(raw.decode())
        except SyntaxError:
            if (row.get('success') is False and row.get('error_kind') == 'action'
                    and 'SyntaxError:' in row.get('traceback', '')):
                continue  # Python compiles the complete snippet before any action.
            raise
        for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'os':
                        imported_os.add(alias.asname or 'os')
            if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)) and isinstance(node.target, ast.Name):
                env.pop(node.target.id, None)
                object_types.pop(node.target.id, None)
                imported_os.discard(node.target.id)
            if isinstance(node, ast.Delete):
                for target_node in node.targets:
                    if isinstance(target_node, ast.Name):
                        env.pop(target_node.id, None)
                        object_types.pop(target_node.id, None)
                        imported_os.discard(target_node.id)
            if isinstance(node, ast.Assign):
                resolved = value(node.value)
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        imported_os.discard(name.id)
                        env[name.id] = resolved
                        object_types[name.id] = object_type(node.value)
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'screenshot' and isinstance(node.func.value, ast.Name)
                    and object_type(node.func.value) == 'page'):
                continue
            raw_path = next((value(k.value) for k in node.keywords if k.arg == 'path'), None)
            if raw_path is not None and (episode / raw_path).resolve() == target:
                return {'browser_step': row['browser_step'], 'code_path': str(code.relative_to(episode)),
                        'code_sha256': sha(raw), 'line': node.lineno, 'capture_api': 'page.screenshot', 'receiver_alias': node.func.value.id}
    raise ValueError('Current-folder image lacks recorded raw page.screenshot provenance')

def evidence_v2_alias_lf_string_add(task, episode_dir):
    episode = Path(episode_dir).resolve()
    manifest = episode / 'browser-steps.jsonl'
    # Unlike the permissive native reader, never silently drop a corrupt record.
    rows = [json.loads(line) for line in manifest.read_text().split('\n') if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Non-object browser step')
    numbers = [int(row['browser_step']) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('Browser steps are unordered or duplicated')
    actions, captures, missing = [], [], []
    seen = set()
    for row in rows:
        action = row.get('action')
        if not isinstance(action, str):
            raise ValueError('Browser step lacks recorded action text')
        actions.append(action)
        ref = row.get('screenshot_path')
        if not ref:
            if row.get('screenshot_error'):
                missing.append({'browser_step': row['browser_step'], 'reason': 'capture_error'})
            continue
        path = (episode / ref).resolve()
        if not path.is_relative_to(episode):
            raise ValueError('Capture escapes current episode')
        if not path.is_file():
            missing.append({'browser_step': row['browser_step'], 'reason': 'referenced_file_missing', 'path': ref})
            continue
        if path in seen:
            raise ValueError('Capture path reused by multiple steps')
        seen.add(path)
        captures.append({'browser_step': row['browser_step'], 'path': str(path), 'sha256': sha(path.read_bytes())})
    # Released native task_screenshot_paths includes every supported image directly
    # under screenshots/, including actor-requested raw page.screenshot captures.
    # Archived images are deliberately excluded; missing references stay inventoried.
    screenshots = episode / 'screenshots'
    if screenshots.is_dir():
        for candidate in sorted(screenshots.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            if not candidate.is_file() or candidate.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            path = candidate.resolve()
            if not path.is_relative_to(episode):
                raise ValueError('Capture escapes current episode')
            if path in seen:
                continue
            try:
                # Preserve the already audited v2 direct-page evidence rule.
                provenance = manual_capture_provenance_syntax(episode, rows, path)
            except ValueError as error:
                if str(error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                    raise
                try:
                    provenance = manual_capture_provenance_string_add(episode, rows, path)
                except ValueError as alias_error:
                    if str(alias_error) != 'Current-folder image lacks recorded raw page.screenshot provenance':
                        raise
                    provenance = automatic_capture_provenance(episode, rows, path)
            seen.add(path)
            captures.append({'browser_step': provenance['browser_step'], 'path': str(path),
                             'sha256': sha(path.read_bytes()), 'manual_capture_provenance': provenance})
        # Keep this campaign's chronological external-export order; native
        # task_screenshot_paths enumerates the same current folder alphabetically.
        captures.sort(key=lambda c: (int(c['browser_step']), Path(c['path']).name))
    if not captures:
        raise ValueError('No recorded current-episode screenshot')
    trajectory = episode / 'trajectory.json'
    data = read(trajectory)
    answer = data.get('info', {}).get('submission', '')
    if not isinstance(answer, str):
        raise ValueError('Native submission is not text')
    if (episode / 'final-answer.json').exists():
        final = read(episode / 'final-answer.json').get('final_answer')
        if final != answer:
            raise ValueError('Final answer differs from native submission')
    if answer:
        actions.append('Final answer: ' + answer)
    return {'task': task['intent'], 'action_history': actions, 'captures': captures,
            'browser_steps_sha256': sha(manifest.read_bytes()), 'trajectory_sha256': sha(trajectory.read_bytes()),
            'browser_step_count': len(rows), 'final_answer': answer,
            'failed_capture_inventory': missing}

def evidence_v3_host_cdp_raw(task, episode_dir):
    from native_capture import host_capture_provenance
    episode = Path(episode_dir).resolve()
    manifest = episode / 'browser-steps.jsonl'
    # Unlike the permissive native reader, never silently drop a corrupt record.
    rows = [json.loads(line) for line in manifest.read_text().split('\n') if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Non-object browser step')
    numbers = [int(row['browser_step']) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('Browser steps are unordered or duplicated')
    actions, captures, missing = [], [], []
    seen = set()
    for row in rows:
        action = row.get('action')
        if not isinstance(action, str):
            raise ValueError('Browser step lacks recorded action text')
        actions.append(action)
        ref = row.get('screenshot_path')
        if not ref:
            if row.get('screenshot_error'):
                missing.append({'browser_step': row['browser_step'], 'reason': 'capture_error'})
            continue
        path = (episode / ref).resolve()
        if not path.is_relative_to(episode):
            raise ValueError('Capture escapes current episode')
        if not path.is_file():
            missing.append({'browser_step': row['browser_step'], 'reason': 'referenced_file_missing', 'path': ref})
            continue
        if path in seen:
            raise ValueError('Capture path reused by multiple steps')
        seen.add(path)
        provenance=host_capture_provenance(ROOT,episode,path)
        if provenance['browser_step'] != row['browser_step']:raise ValueError('Host raw capture belongs to a different native step')
        captures.append({'browser_step': row['browser_step'], 'path': str(path), 'sha256': sha(path.read_bytes()), 'host_capture_provenance':provenance})
    # Released native task_screenshot_paths includes every supported image directly
    # under screenshots/, including actor-requested raw page.screenshot captures.
    # Archived images are deliberately excluded; missing references stay inventoried.
    screenshots = episode / 'screenshots'
    if screenshots.is_dir():
        for candidate in sorted(screenshots.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
            if not candidate.is_file() or candidate.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            path = candidate.resolve()
            if not path.is_relative_to(episode):
                raise ValueError('Capture escapes current episode')
            if path in seen:
                continue
            provenance=host_capture_provenance(ROOT,episode,path)
            seen.add(path)
            captures.append({'browser_step': provenance['browser_step'], 'path': str(path),
                             'sha256': sha(path.read_bytes()), 'host_capture_provenance': provenance})
        # Keep this campaign's chronological external-export order; native
        # task_screenshot_paths enumerates the same current folder alphabetically.
        captures.sort(key=lambda c: (int(c['browser_step']), Path(c['path']).name))
    if not captures:
        raise ValueError('No recorded current-episode screenshot')
    trajectory = episode / 'trajectory.json'
    data = read(trajectory)
    answer = data.get('info', {}).get('submission', '')
    if not isinstance(answer, str):
        raise ValueError('Native submission is not text')
    if (episode / 'final-answer.json').exists():
        final = read(episode / 'final-answer.json').get('final_answer')
        if final != answer:
            raise ValueError('Final answer differs from native submission')
    if answer:
        actions.append('Final answer: ' + answer)
    return {'task': task['intent'], 'action_history': actions, 'captures': captures,
            'browser_steps_sha256': sha(manifest.read_bytes()), 'trajectory_sha256': sha(trajectory.read_bytes()),
            'browser_step_count': len(rows), 'final_answer': answer,
            'failed_capture_inventory': missing}

def evidence(task, episode_dir, adapter_version='available-captures-v3-host-cdp-raw'):
    if adapter_version == 'available-captures-v3-host-cdp-raw':
        return evidence_v3_host_cdp_raw(task, episode_dir)
    if adapter_version == 'available-captures-v2-alias-lf-string-add':
        return evidence_v2_alias_lf_string_add(task, episode_dir)
    if adapter_version == 'available-captures-v2-alias-lf-path-join':
        return evidence_v2_alias_lf_path_join(task, episode_dir)
    if adapter_version == 'available-captures-v2-alias-lf-page-iterator':
        return evidence_v2_alias_lf_page_iterator(task, episode_dir)
    if adapter_version == 'available-captures-v2-alias-lf-syntax':
        return evidence_v2_alias_lf_syntax(task, episode_dir)
    if adapter_version == 'available-captures-v2-alias-lf':
        return evidence_v2_alias_lf(task, episode_dir)
    if adapter_version == 'available-captures-v2-alias':
        return evidence_v2_alias(task, episode_dir)
    if adapter_version == 'available-captures-v2':
        return evidence_v2(task, episode_dir)
    if adapter_version == 'available-captures-v1':
        return evidence_v1(task, episode_dir)
    raise ValueError('Unknown evidence adapter version')


class GatewayEngine:
    def __init__(self, endpoint, directory):
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('Judge endpoint must be the local metered gateway')
        if not re.fullmatch(r'/judge/[^/]+/chat/completions', parsed.path):
            raise ValueError('Unexpected judge gateway route')
        self.endpoint = endpoint
        self.directory = Path(directory)
        self.directory.mkdir()
        self.calls = []
        self.lock = threading.Lock()
    def generate(self, messages, **kwargs):
        if kwargs:
            raise ValueError('Official judge attempted extra arguments')
        system = messages[0]['content']
        stage = 'key_points' if 'identify the key points' in system else 'image_relevance' if 'determining whether an image' in system else 'outcome'
        request = {'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': 512}
        with self.lock:
            index = len(self.calls)
            rec = {'index': index, 'stage': stage, 'request_sha256': sha(canonical(request).encode()), 'state': 'reserved'}
            self.calls.append(rec)
        prefix = self.directory / f'{index:04d}-{stage}'
        write(str(prefix) + '.request.json', request)
        started = time.monotonic()
        try:
            req = urllib.request.Request(self.endpoint, canonical(request).encode(), {'Content-Type': 'application/json', 'Authorization': 'Bearer local-metered-gateway'})
            # Exactly one attempt; no retry middleware or upstream SDK retries.
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=180) as response:
                raw = json.load(response)
            write(str(prefix) + '.response.json', raw)
            if raw.get('model') != MODEL:
                raise ValueError('Judge model snapshot mismatch')
            choice = raw['choices'][0]
            text = choice['message']['content']
            if not isinstance(text, str):
                raise ValueError('Judge returned non-text')
            usage = raw.get('usage')
            known = isinstance(usage, dict) and all(isinstance(usage.get(k), int) for k in ('prompt_tokens', 'completion_tokens'))
            rec.update(state='complete', finish_reason=choice.get('finish_reason'), usage=usage, unknown_usage=not known, text=text)
            return [text]
        except Exception as exc:
            rec.update(state='failed_or_unknown', error_class=type(exc).__name__)
            raise
        finally:
            rec['latency_seconds'] = time.monotonic() - started
            write(str(prefix) + '.record.json', rec)

def load_official():
    # Verify against the already independently audited frozen official source manifest.
    plan = read(PRIOR / 'plan.json')
    for rel in ('official-source/src/methods/webjudge_online_mind2web.py', 'official-source/src/utils.py'):
        expected = plan['source_hashes'][rel]
        if sha((PRIOR / rel).read_bytes()) != expected:
            raise ValueError('Pinned official source changed: ' + rel)
    sys.path.insert(0, str(PRIOR / 'deps'))
    sys.path.insert(0, str(PRIOR / 'official-source/src'))
    spec = importlib.util.spec_from_file_location('pinned_official_external_webjudge', PRIOR / 'official-source/src/methods/webjudge_online_mind2web.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_external_judge(taskdict, episode_dir, endpoint, output_path):
    output = Path(output_path)
    if output.exists():
        raise ValueError('Existing judge result cannot be overwritten or unchanged-retried')
    started = time.monotonic()
    result = {'evidence_adapter_version': 'available-captures-v3-host-cdp-raw', 'state': 'invalid', 'valid': False, 'official_label': None, 'strict_label': None, 'flags': []}
    engine = None
    try:
        ev = evidence(taskdict, episode_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        write(output.with_name(output.stem + '-evidence.json'), ev)
        engine = GatewayEngine(endpoint, output.with_name(output.stem + '-calls'))
        module = load_official()
        images = [frame['path'] for frame in ev['captures']]
        messages, text, system, records, keypoints = asyncio.run(module.WebJudge_Online_Mind2Web_eval(ev['task'], ev['action_history'], images, engine, 3))
        response = engine.generate(messages)[0]
        strict = strict_status(response)
        flags = []
        if strict is None:
            flags.append('strict_final_parse_failure')
        if any(c.get('finish_reason') == 'length' for c in engine.calls):
            flags.append('output_truncation')
        if any(c.get('unknown_usage') for c in engine.calls):
            flags.append('unknown_usage')
        if any(rec['Score'] == 0 for rec in records):
            flags.append('image_score_parse_failure')
        if len(engine.calls) != len(images) + 2:
            flags.append('native_stage_count_mismatch')
        result.update(state='complete', valid=not flags, official_label=official_status(response), strict_label=strict,
                      flags=flags, response=response, key_points=keypoints, image_judge_record=records,
                      input_screenshot_count=len(images), selected_screenshot_count=len(messages[1]['content']) - 1,
                      final_input_text=text, evidence_sha256=sha(canonical(ev).encode()), browser_step_count=ev['browser_step_count'])
    except Exception as exc:
        result['flags'].append(type(exc).__name__)
        result['error'] = str(exc)
    result['latency_seconds'] = time.monotonic() - started
    result['judge_only_latency_seconds'] = result['latency_seconds']
    result['call_count'] = len(engine.calls) if engine else 0
    result['calls'] = engine.calls if engine else []
    result['model'] = MODEL
    write(output, result)
    return result

def audit_external_result(taskdict, episode_dir, output_path):
    """Offline case closeout: physical endpoint, source binding, parser, call files.

    Budget, scheduler, native actor configuration and gateway accounting are audited
    independently by the campaign controller; this function does not dispatch.
    """
    output = Path(output_path)
    result = read(output)
    errors = []
    try:
        ev = evidence(taskdict, episode_dir, result.get('evidence_adapter_version', 'available-captures-v1'))
        if result.get('evidence_sha256') != sha(canonical(ev).encode()):
            errors.append('source_evidence_changed')
        if result.get('state') != 'complete':
            errors.append('judge_incomplete')
        calls = result.get('calls', [])
        if len(calls) != len(ev['captures']) + 2:
            errors.append('native_call_count')
        stages = [call['stage'] for call in calls]
        if not stages or stages[0] != 'key_points' or stages[-1] != 'outcome' or any(stage != 'image_relevance' for stage in stages[1:-1]):
            errors.append('native_stage_order')
        for call in calls:
            prefix = output.with_name(output.stem + '-calls') / f"{call['index']:04d}-{call['stage']}"
            request = read(str(prefix) + '.request.json')
            response = read(str(prefix) + '.response.json')
            record = read(str(prefix) + '.record.json')
            if record != call or sha(canonical(request).encode()) != call['request_sha256']:
                errors.append('call_binding')
            if (request.get('model'), request.get('temperature'), request.get('max_tokens')) != (MODEL, 0, 512):
                errors.append('native_settings')
            if response.get('model') != MODEL or response['choices'][0]['message']['content'] != call.get('text'):
                errors.append('response_binding')
        if result.get('official_label') != official_status(result.get('response', '')) or result.get('strict_label') != strict_status(result.get('response', '')):
            errors.append('label_parser_binding')
    except Exception as exc:
        errors.append(type(exc).__name__)
    return {'verified': not errors, 'errors': sorted(set(errors)), 'judge_valid': result.get('valid') is True,
            'official_label': result.get('official_label'), 'strict_label': result.get('strict_label'),
            'flags': result.get('flags', []), 'call_count': result.get('call_count', 0),
            'result_sha256': sha(output.read_bytes())}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-json', required=True)
    parser.add_argument('--episode-dir', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run_external_judge(read(args.task_json), args.episode_dir, args.endpoint, args.output)
    print(json.dumps({k: result[k] for k in ('state', 'valid', 'official_label', 'strict_label', 'flags', 'call_count')}))

if __name__ == '__main__':
    main()
