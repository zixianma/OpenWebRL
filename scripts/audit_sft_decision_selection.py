#!/usr/bin/env python3
"""Audit the approved private pilot; never launch tasks or mark a run complete."""
import argparse
import ast
import base64
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import time
import traceback

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sft-decision-selection-pilot-20261004')
MODES = ('sft', 'jev', 'kev-0.8b', 'kev-27b')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def decode_image(value):
    if isinstance(value, dict):
        value = value['url']
    return base64.b64decode(value.split(',', 1)[-1], validate=True)


def normalized(value):
    return value.strip().removesuffix('<|im_end|>').strip()


def split_output(value):
    if '</think>' in value:
        thought, action = value.split('</think>', 1)
        return dict(thought=thought.rsplit('<think>', 1)[-1].strip(), action=action.strip())
    return dict(thought='', action=value.strip())


def expected_seed(task, turn, candidate):
    raw = f'42:{task}:{turn}:{candidate}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big') % 2147483647


def verify_image(raw):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        dimensions = image.size
        image.verify()
    return list(dimensions)


def align_partial_screenshots(screenshots, decision_count, diagnosis):
    """Keep an explicitly diagnosed, unselected next observation separate.

    Cancellation during the next proposal leaves mm_messages one image ahead of
    the last completed turn sample. It is evidence, but not a selected action or
    a terminal screenshot. Never silently discard an unexplained extra image.
    """
    extras = screenshots[decision_count:]
    expected = diagnosis.get('unselected_observation_sha256', [])
    assert len(screenshots) >= decision_count and len(extras) <= 1
    assert [digest(raw) for raw in extras] == expected, 'Unexplained unselected observation'
    for raw in extras:
        verify_image(raw)
    return screenshots[:decision_count], expected


def verify_pre_action_abort(sample, decisions, task_id, initial_image):
    """Verify a preserved initial observation with no executed model action."""
    assert sample['sample_id'] == task_id and sample['status'] == 'aborted'
    assert sample['total_steps'] == 0 and sample['llm_response'] == ''
    assert not decisions, 'Pre-action abort contains selector requests'
    assert sample['terminate_reason'] == 'generation_error: Selector halted or request cap reached'
    assert len(sample['images']) == 1 and len(sample['images'][0]) == 1
    assert decode_image(sample['images'][0][0]) == initial_image
    return verify_image(initial_image)


def audit(root=ROOT):
    root = Path(root)
    plan, approval = read(root / 'plan.json'), read(root / 'approval.json')
    modes = plan.get('modes', MODES)
    task_count = len(plan['task_ids'])
    expected_results = len(modes) * task_count
    assert digest((root / 'plan.json').read_bytes()) == approval['plan_sha256']
    source = Path(plan['source'])
    assert digest((source / 'reference_manifest.json').read_bytes()) == plan['source_manifest_sha256']
    for name, expected in read(source / 'reference_manifest.json')['recipe_files_sha256'].items():
        assert digest((source / name).read_bytes()) == expected, name
    assert digest(Path(plan['tasks']).read_bytes()) == plan['tasks_sha256']
    tasks = {r['metadata']['task_id']: r for r in
             map(json.loads, Path(plan['tasks']).read_text().splitlines())}
    assert set(tasks) == set(plan['task_ids']) and len(tasks) == task_count and task_count > 0
    prompt_ast = ast.parse((source / 'openwebrl/eval/reward_online_mind2web.py').read_text())
    system_prompt = next(ast.literal_eval(n.value) for n in prompt_ast.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in n.targets))
    issues, rows, counts, summaries, unfinished = [], [], Counter(), {}, []
    counts['jev_requests'] += len(list((root / 'recovery-probes').glob('request-*.json')))
    diagnosed = {(r['mode'], r['task_id']): r for r in
        (read(root / 'diagnosed-invalid.json') if (root / 'diagnosed-invalid.json').exists() else [])}
    for mode in modes:
        folder = root / mode
        # Freeze the completed set before reading traces; live workers may
        # finish another episode while this audit scans earlier evidence.
        result_paths = set((folder / 'results').glob('*.json'))
        superseded_path = folder / 'superseded-selections.json'
        superseded = read(superseded_path) if superseded_path.exists() else {}
        traces = defaultdict(list)
        if mode == 'sft':
            for path in (folder / 'selections').glob('*.jsonl'):
                for line in path.read_text().splitlines():
                    record = json.loads(line)
                    traces[record['task_id']].append(record)
                    counts['actor_proposals_saved'] += len(record['candidates'])
        else:
            for path in sorted((folder / 'selections').glob('request-*.json')):
                record = read(path); record['_path'] = path
                counts['jev_requests' if mode == 'jev' else 'local_kev_requests'] += len(record.get('http_attempts', [None]))
                counts['actor_proposals_saved'] += len(record['actor_outputs'])
                if path.name in superseded:
                    prior = superseded[path.name]
                    assert digest(path.read_bytes()) == prior['sha256']
                    assert digest(path.with_suffix('.png').read_bytes()) == prior['image_sha256']
                    assert record['task_id'] == prior['task_id']
                    counts['superseded_selection_requests'] += 1
                    continue
                traces[record['task_id']].append(record)
        counts['local_kev_requests'] += len(list(folder.glob('warmup-request.json')))
        counts['judge_http_attempts'] += len(list((folder / 'judge').rglob('request-*.json')))
        markers = folder / 'browser_sessions'
        active = [p.name for p in markers.glob('*') if p.is_file()]
        stopped = [p.name for p in (markers / 'stopped').glob('*') if p.is_file()]
        counts['browser_sessions'] += len(active) + len(stopped)
        summary = dict(completed=0, valid=0, successes=0, active_browser_sessions=len(active),
            stopped_browser_sessions=len(stopped), selection_turns=sum(map(len, traces.values())))
        for task_id in plan['task_ids']:
            key = digest(task_id.encode()); path = folder / 'results' / (key + '.json')
            if path not in result_paths:
                started = folder / 'started' / (key + '.json')
                decisions = sorted(traces[task_id], key=lambda r: r['turn'])
                unfinished.append(dict(mode=mode, task_id=task_id, started=started.exists(),
                    start_receipt=read(started) if started.exists() else None,
                    saved_selection_records=len(decisions),
                    last_saved_turn=decisions[-1]['turn'] if decisions else None,
                    final_action_execution_verified=False,
                    saved_rollouts=[str(p) for p in (folder / 'samples' / key).rglob('*.json')]))
                continue
            result = read(path); summary['completed'] += 1
            corrected = root / 'judge-recovery' / mode / 'corrected-results' / path.name
            original = root / 'judge-recovery' / mode / 'original-results' / path.name
            staged = corrected.exists() and original.exists() and read(original) == result
            if staged:
                result = read(corrected)
            row = dict(mode=mode, task_id=task_id, valid=result.get('valid'),
                reward=result.get('reward'), steps=result.get('total_steps'),
                termination=result.get('terminate_reason'), result_sha256=digest(path.read_bytes()),
                terminal_correction_staged=staged, issues=[])
            try:
                assert result['task_id'] == task_id
                assert result['mode'] == mode
                diagnosis = diagnosed.get((mode, task_id))
                if diagnosis and diagnosis.get('pre_action_abort'):
                    assert not result['valid'] and result['reward'] is None
                    assert result['error_type'] == diagnosis['error_type'] == 'ValueError'
                    assert digest(path.read_bytes()) == diagnosis['original_result_sha256']
                    archive = Path(diagnosis['partial_rollout'])
                    assert list((folder / 'samples' / key).rglob('*.json')) == [archive]
                    assert digest(archive.read_bytes()) == diagnosis['partial_rollout_sha256']
                    initial = (folder / 'final' / (key + '.png')).read_bytes()
                    assert digest(initial) == diagnosis['initial_image_sha256']
                    dimensions = verify_pre_action_abort(read(archive), traces[task_id], task_id, initial)
                    assert read(folder / 'started' / (key + '.json'))['task_id'] == task_id
                    assert not list((folder / 'judge' / key).glob('request-*.json'))
                    row.update(diagnosed_invalid=True, partial_rollout_verified=True,
                        partial_rollout=str(archive), decisions=0, final_action_execution_verified=False,
                        image_role='initial observation only', final_image_dimensions=dimensions,
                        final_sha256=digest(initial), diagnosis=diagnosis['diagnosis'])
                    row['issues'].append('Diagnosed invalid: ' + diagnosis['classification'])
                    issues.append(dict(mode=mode, task_id=task_id, error=row['issues'][-1], diagnosed=True))
                    rows.append(row)
                    continue
                partial = result.get('error_type') == 'TimeoutError' and 'metadata' not in result
                if result['valid']:
                    assert result['reward'] in (0, 1)
                if partial:
                    assert not result['valid'] and diagnosis and diagnosis['error_type'] == 'TimeoutError'
                    archive = Path(diagnosis['partial_rollout'])
                    assert archive in list((folder / 'samples' / key).rglob('*.json'))
                    assert digest(archive.read_bytes()) == diagnosis['partial_rollout_sha256']
                    sample = read(archive)
                    assert sample['sample_id'] == task_id and sample['status'] == 'pending'
                    prompt = sample['llm_input_texts']
                    assert tasks[task_id]['metadata']['start_url'] in prompt
                    assistants = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', prompt, re.S)
                    assistants.append(sample['llm_response'])
                    assert all(len(images) == 1 for images in sample['images'])
                    screenshots = [decode_image(images[0]) for images in sample['images']]
                    row.update(partial_rollout=str(archive), termination='TimeoutError',
                        final_action_execution_verified=False)
                else:
                    assert result['start_url'] == tasks[task_id]['metadata']['start_url']
                    messages = result['metadata']['messages']
                    assistants = [m['content'] for m in messages if m['role'] == 'assistant']
                    screenshots = [decode_image(c['image_url']) for m in messages if m['role'] == 'user'
                                   for c in m['content'] if c['type'] == 'image_url']
                all_decisions = sorted(traces[task_id], key=lambda r: r['turn'])
                failed_decisions = [r for r in all_decisions if mode != 'sft' and r['status'] != 'selected']
                decisions = [r for r in all_decisions if r not in failed_decisions]
                if partial:
                    screenshots, extra_hashes = align_partial_screenshots(screenshots, len(decisions), diagnosis)
                    row['unselected_observation_sha256'] = extra_hashes
                assert not failed_decisions or not result['valid'], 'Unexecuted reservation in valid episode'
                assert len(assistants) == len(decisions) <= 30
                if result['valid']:
                    assert len(decisions) == result['total_steps']
                assert len(screenshots) == len(decisions)
                assert [r['turn'] for r in decisions] == list(range(len(decisions)))
                for turn, (record, actual, screenshot) in enumerate(zip(decisions, assistants, screenshots)):
                    assert record['fallback'] is None
                    assert record['candidate_seeds'] == [expected_seed(task_id, turn, j)
                        for j in range(1 if mode == 'sft' else 5)], 'Seed schedule differs'
                    assert digest(screenshot) == record['screenshot_sha256'], 'Actor state differs from selection trace'
                    if mode == 'sft':
                        assert record['selected_index'] == 0 and record['mode'] == 'baseline'
                        chosen = record['candidates'][0]
                    else:
                        assert record['status'] == 'selected' and record['http_status'] == 200
                        assert record['sampling'] == dict(temperature=plan['protocol']['temperature'],
                            top_p=plan['protocol']['top_p'], top_k=plan['protocol']['top_k'],
                            max_new_tokens=4096, repetition_penalty=1.)
                        request, response = record['request'], record['response']
                        http_attempts = record.get('http_attempts')
                        if http_attempts is not None:
                            assert 1 <= len(http_attempts) <= 1 + plan.get('worker_options', {}).get('selector_validation_retries', 0)
                            assert http_attempts[-1]['response'] == response
                            for rejected in http_attempts[:-1]:
                                assert mode == 'jev' and rejected['http_status'] == 200
                                assert rejected['validation_error'] == 'Decision choice disagrees with probability argmax'
                                rejected_answer = rejected['response']['answers']['selection']
                                assert rejected_answer['probabilities'][rejected_answer['choice']] < max(rejected_answer['probabilities'].values())
                        assert response['model'] == request['model'] == ('jev-1.13.0' if mode == 'jev' else 'kev-latest')
                        assert not response.get('truncated')
                        assert record['identity'] == ({'model': 'jev-1.13.0'} if mode == 'jev'
                            else plan['kev_specs'][mode.removeprefix('kev-')])
                        answer = response['answers']['selection']; probs = answer['probabilities']
                        assert set(probs) == set('12345') and all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
                        rounding = (mode == 'jev' and all(abs(p * 100 - round(p * 100)) < 1e-10
                            for p in probs.values()))
                        assert abs(sum(probs.values()) - 1) <= (.025 + 1e-12 if rounding else .001)
                        index = record['selected_index']; assert str(index + 1) == answer['choice']
                        assert probs[str(index + 1)] >= max(probs.values()) - 1e-7
                        candidates = request['state']['candidates']
                        assert len(record['actor_outputs']) == len(candidates) == 5
                        for j, output in enumerate(record['actor_outputs']):
                            assert split_output(output) == candidates[str(j + 1)]
                        assert len(request['state']['recent_history']) <= 5
                        assert len(request['state']['page']['text']) <= 16000
                        assert record['_path'].with_suffix('.png').read_bytes() == screenshot
                        chosen = candidates[str(index + 1)]
                    actual_split = split_output(actual)
                    assert actual_split['thought'] == chosen['thought'], 'Selected reasoning changed'
                    assert normalized(actual_split['action']) == normalized(chosen['action']), 'Selected action changed'
                if not result['valid']:
                    assert diagnosis, 'Invalid result requires diagnosis'
                    assert diagnosis['original_result_sha256'] == row['result_sha256']
                    assert result['reward'] == diagnosis['saved_reward']
                    assert partial or 'ABORTED' in result['status']
                    final_path = folder / 'final' / (key + '.png')
                    assert final_path.exists() == diagnosis['saved_final_image']
                    if final_path.exists():
                        final = final_path.read_bytes()
                        row['final_image_dimensions'] = verify_image(final)
                        row['final_sha256'] = digest(final)
                        assert decode_image(result['metadata']['full_image_list'][-1]) == final
                    assert len(failed_decisions) == len(diagnosis.get('unexecuted_requests', {}))
                    for failed in failed_decisions:
                        trace_path = failed['_path']
                        assert digest(trace_path.read_bytes()) == diagnosis['unexecuted_requests'][trace_path.name]
                        assert failed['status'] == 'failed' and failed['turn'] == len(decisions)
                        assert failed['candidate_seeds'] == [expected_seed(task_id, failed['turn'], j) for j in range(5)]
                        assert digest(trace_path.with_suffix('.png').read_bytes()) == failed['screenshot_sha256']
                        assert len(failed['actor_outputs']) == 5 and 'selected_index' not in failed
                    assert not list((folder / 'judge' / key).glob('request-*.json'))
                    row.update(diagnosed_invalid=True, trajectory_verified=not partial,
                        partial_rollout_verified=partial,
                        decisions=len(decisions), unexecuted_reservations=len(failed_decisions), diagnosis=diagnosis['diagnosis'])
                    row['issues'].append('Diagnosed invalid: ' + diagnosis['classification'])
                    issues.append(dict(mode=mode, task_id=task_id, error=row['issues'][-1], diagnosed=True))
                    rows.append(row)
                    continue
                final = (folder / 'final' / (key + '.png')).read_bytes()
                row['final_image_dimensions'] = verify_image(final)
                row['final_sha256'] = digest(final)
                assert decode_image(result['metadata']['full_image_list'][-1]) == final
                judge_folder = folder / 'judge' / key
                requests = sorted(judge_folder.glob('request-*.json')); assert 1 <= len(requests) <= 4
                for req in requests:
                    request = read(req)
                    assert request['model'] == 'o4-mini' and request['max_completion_tokens'] == 4096 and request['seed'] == 42
                    assert request['messages'][0]['content'] == system_prompt, 'Judge protocol changed'
                    parts = request['messages'][1]['content']
                    assert result['intent'] in parts[0]['text']
                    assert decode_image(parts[1]['image_url']) == final, 'Judge saw a different final image'
                response = read(judge_folder / requests[-1].name.replace('request-', 'response-'))
                assert response['model'].startswith('o4-mini-')
                assert response['usage']['completion_tokens'] <= 4096
                text = response['choices'][0]['message']['content']
                assert text == result['metadata']['reward']['judge_text']
                assert 'status:' in text.lower()
                assert float('success' in text.lower().split('status:', 1)[1]) == result['reward']
                samples = list((folder / 'samples' / key).rglob('*.json'))
                assert len(samples) == 1, 'Expected one final rollout archive'
                sample = read(samples[0]); assert sample['total_steps'] == result['total_steps']
                assert normalized(sample['llm_response']).removeprefix('<think>').strip() == normalized(assistants[-1]).removeprefix('<think>').strip()
                row.update(evidence_verified=True, judge_model=response['model'], judge_text=text,
                    rollout=str(samples[0]), decisions=len(decisions))
                summary['valid'] += 1; summary['successes'] += int(result['reward'])
            except (AssertionError, KeyError, ValueError, OSError, TypeError) as exc:
                row['issues'].append(str(exc) or type(exc).__name__)
                issues.append(dict(mode=mode, task_id=task_id, error=row['issues'][-1],
                    audit_line=traceback.extract_tb(exc.__traceback__)[-1].lineno))
            rows.append(row)
        summary['success_rate_all_scheduled'] = summary['successes'] / task_count
        summary['success_rate_valid'] = summary['successes'] / summary['valid'] if summary['valid'] else None
        summaries[mode] = summary
    for key in ('browser_sessions', 'jev_requests', 'local_kev_requests', 'judge_http_attempts'):
        assert counts[key] <= approval['limits'][key], f'Exceeded {key} approval'
    assert counts['actor_proposals_saved'] <= approval['limits']['actor_proposals']
    jobs = ','.join(a['job_id'] for a in approval['attempts'])
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', jobs,
        '--format=JobIDRaw,State,ElapsedRaw'], text=True)
    attempts = [dict(job_id=r[0], state=r[1], seconds=int(r[2])) for line in accounting.splitlines()
                if (r := line.split('|'))[0] in jobs.split(',')]
    assert len(attempts) == len(approval['attempts'])
    seconds = sum(a['seconds'] for a in attempts); assert seconds <= approval['resources']['total_seconds']
    terminal = all(a['state'].split()[0] in ('COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED') for a in attempts)
    return dict(audited_unix=time.time(), source=str(source), plan_sha256=approval['plan_sha256'],
        expected_results=expected_results,
        all_results_audited=len(rows) == expected_results and all(r.get('evidence_verified') or r.get('diagnosed_invalid') for r in rows),
        all_evidence_verified=len(rows) == expected_results and not issues and not any(r['terminal_correction_staged'] for r in rows),
        completed_results=len(rows), all40_evidence_verified=expected_results == 40 and len(rows) == 40 and not issues and not any(r['terminal_correction_staged'] for r in rows),
        staged_terminal_corrections=sum(r['terminal_correction_staged'] for r in rows),
        all40_results_audited=expected_results == 40 and len(rows) == 40 and all(r.get('evidence_verified') or r.get('diagnosed_invalid') for r in rows),
        scheduler_terminal=terminal, all_browsers_closed=not any(s['active_browser_sessions'] for s in summaries.values()),
        unfinished_tasks=unfinished,
        scheduler_seconds=seconds, attempts=attempts, accounting=dict(counts), summaries=summaries, issues=issues, rows=rows,
        limitations=['Canonical AgentTrek verdicts are not independently relabeled ground truth.',
                    'Saved proposal counts exclude SGLang startup warmup and any generation interrupted before trace persistence.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = audit(args.root)
    output = args.root / 'independent-audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))
