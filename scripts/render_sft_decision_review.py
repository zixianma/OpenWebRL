#!/usr/bin/env python3
"""Render a private, self-contained review of completed SFT selection episodes."""
import argparse
import base64
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
ROOT = RUNTIME / 'evaluations/sft-decision-selection-pilot-20261004'
MODES = [('sft', 'SFT alone'), ('jev', 'SFT + Jev'), ('kev-0.8b', 'SFT + Kev 0.8B'), ('kev-27b', 'SFT + Kev 27B')]


def read(path):
    return json.loads(Path(path).read_text())


def build(root=ROOT):
    plan, audit = read(root / 'plan.json'), read(root / 'independent-audit.json')
    evidence = {(r['mode'], r['task_id']): r for r in audit['rows']}
    tasks = []
    for line in Path(plan['tasks']).read_text().splitlines():
        row = json.loads(line)['metadata']
        tasks.append(dict(task_id=row['task_id'], intent=row['intent'], start_url=row['start_url'], models={}))
    images = {}

    def image(raw):
        key = hashlib.sha256(raw).hexdigest()
        if key not in images:
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as original:
                preview = original.convert('RGB')
                preview.thumbnail((1600, 1600))
                buffer = io.BytesIO(); preview.save(buffer, format='JPEG', quality=85)
            images[key] = base64.b64encode(buffer.getvalue()).decode()
        return key

    for mode, label in MODES:
        directory = root / mode
        traces = defaultdict(list)
        if mode == 'sft':
            for path in (directory / 'selections').glob('*.jsonl'):
                for line in path.read_text().splitlines():
                    row = json.loads(line); traces[row['task_id']].append(row)
        else:
            for path in (directory / 'selections').glob('request-*.json'):
                row = read(path); row['_path'] = path; traces[row['task_id']].append(row)
        for task in tasks:
            key = hashlib.sha256(task['task_id'].encode()).hexdigest()
            result_path = directory / 'results' / (key + '.json')
            unfinished = not result_path.exists()
            status = 'completed'
            if unfinished:
                started = (directory / 'started' / (key + '.json')).exists()
                status = ('interrupted' if started else 'not run') if audit['scheduler_terminal'] else ('running' if started else 'pending')
                if not traces[task['task_id']]:
                    task['models'][mode] = dict(status=status)
                    continue
                start = read(directory / 'started' / (key + '.json'))['started_unix']
                end = audit['audited_unix']
                if audit['scheduler_terminal'] and (root / 'heartbeat.json').exists():
                    end = min(end, read(root / 'heartbeat.json')['updated_unix'])
                result = dict(valid=False, reward=None, elapsed_seconds=max(0, end - start),
                    error_type='No terminal result; partial trace only')
            else:
                result = read(result_path)
            corrected = root / 'judge-recovery' / mode / 'corrected-results' / result_path.name
            original = root / 'judge-recovery' / mode / 'original-results' / result_path.name
            if corrected.exists() and original.exists() and read(original) == result:
                result = read(corrected)
            if not unfinished and (mode, task['task_id']) not in evidence:
                task['models'][mode] = dict(status='awaiting audit')
                continue
            receipt = evidence.get((mode, task['task_id']), dict(issues=[
                'No terminal result. These saved proposals are not an audited completed trajectory.'],
                partial_rollout_verified=False, final_action_execution_verified=False))
            messages = result.get('metadata', {}).get('messages', [])
            observations = [m['content'] for m in messages if m['role'] == 'user']
            decisions = sorted(traces[task['task_id']], key=lambda r: r['turn'])
            frames = []
            for i, decision in enumerate(decisions):
                observation = observations[i] if i < len(observations) else []
                picture = next((c['image_url'] for c in observation if c['type'] == 'image_url'), None)
                if isinstance(picture, dict): picture = picture['url']
                picture = image(base64.b64decode(picture.split(',', 1)[-1], validate=True)) if picture else None
                if picture is None and decision.get('_path'):
                    picture = image(decision['_path'].with_suffix('.png').read_bytes())
                if mode == 'sft':
                    candidates = decision['candidates']; probabilities = {'1': 1.0}; page = None
                else:
                    candidates = list(decision['request']['state']['candidates'].values())
                    probabilities = decision.get('response', {}).get('answers', {}).get('selection', {}).get('probabilities', {})
                    page = decision['request']['state']['page']
                following = observations[i + 1] if i + 1 < len(observations) else []
                frames.append(dict(turn=i, screenshot=picture, candidates=candidates,
                    selected_index=decision.get('selected_index'), probabilities=probabilities,
                    seeds=decision['candidate_seeds'], page=page,
                    tool_result='\n'.join(c['text'] for c in following if c['type'] == 'text'),
                    latency_seconds=decision.get('seconds'),
                    raw_response=decision.get('response'), status=decision.get('status', 'selected')))
                if receipt.get('final_action_execution_verified') is False and i == len(decisions) - 1:
                    frames[-1]['execution_unverified'] = True
            final = directory / 'final' / (key + '.png')
            task['models'][mode] = dict(status=status, valid=result.get('valid'), reward=result.get('reward'),
                steps=receipt.get('decisions', max(result.get('total_steps', 0), len(frames))),
                termination=result.get('terminate_reason') or result.get('error_type'),
                partial_rollout=unfinished or receipt.get('partial_rollout_verified', False),
                elapsed_seconds=result.get('elapsed_seconds'), frames=frames,
                final=image(final.read_bytes()) if final.exists() else None,
                judge=result.get('metadata', {}).get('reward', {}).get('judge_text'),
                evidence_verified=receipt.get('evidence_verified', False), issues=receipt['issues'])
    return dict(tasks=tasks, models=[dict(key=k, label=v) for k,v in MODES], images=images,
        audit={k:v for k,v in audit.items() if k != 'rows'}, protocol=plan['protocol'], rendered_unix=time.time())


def render(output):
    output = Path(output).resolve()
    if output.is_relative_to(REPO):
        raise ValueError('Raw task data must remain private')
    payload = build()
    template = (REPO / 'scripts/templates/sft_decision_review.html').read_text()
    assert template.count('__REVIEW_DATA__') == 1
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    output.write_text(template.replace('__REVIEW_DATA__', encoded))
    receipt = dict(html=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        bytes=output.stat().st_size, completed_results=payload['audit']['completed_results'],
        all40_evidence_verified=payload['audit']['all40_evidence_verified'], public=False)
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUNTIME / 'visualizations/sft-decision-selection-review-20261004.html')
    print(json.dumps(render(parser.parse_args().output), indent=2))
