#!/usr/bin/env python3
"""Render private, self-contained HTML from verified Jev/Kev pilot artifacts."""
import argparse
import base64
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def target_mapping(decision, observed_pages):
    """Resolve indices from the exact model request; geometry needs a matched state.

    Never map an old target index onto the post-action page, where the numbered
    element list may have changed. Observed bounds are not an execution trace.
    """
    if not decision or decision.get('target') is None:
        return None
    question = decision['request']['questions'][decision['operation'].lower() + '_target']
    criterion = question['criteria'][decision['target']]
    result = dict(operation=decision['operation'], model_target=decision['target'],
        label=criterion['element'], role=criterion.get('role'),
        current_value=criterion.get('current_value'), harness_action_id=decision['choice'],
        geometry_status='No matching pre-decision snapshot was saved',
        actual_execution_coordinates_recorded=False)
    observed = observed_pages.get(decision['fingerprint'])
    if observed:
        action = next((a for a in observed['actions'] if a['id'] == decision['choice']), None)
        if action:
            result.update(cached_dom_node=action.get('node'), action=action,
                observed_rect=action.get('rect'), before_screenshot=observed.get('image'),
                viewport=dict(width=observed['w'], height=observed['h']),
                geometry_status='Bounds from a saved page with the exact decision fingerprint')
    return result


def build():
    roots = [('jev', 'Jev 1.13.0', RUNTIME / 'evaluations/jev-ultrafast-om2w-pilot-20261004'),
             ('kev08', 'Kev 0.8B', RUNTIME / 'evaluations/kev-pair-om2w-pilot-20261004/0.8b'),
             ('kev27', 'Kev 27B', RUNTIME / 'evaluations/kev-pair-om2w-pilot-20261004/27b')]
    tasks, images, models = {}, {}, []

    def image(value):
        if not value:
            return None
        raw = base64.b64decode(value, validate=True)
        key = hashlib.sha256(raw).hexdigest()
        images[key] = value
        return key

    for key, label, root in roots:
        if not read(root / 'summary.json')['verified_complete']:
            raise ValueError('Only verified completed cohorts belong in this review')
        summary = read(root / 'summary.json')
        models.append(dict(key=key, label=label, summary=summary))
        for directory in sorted((root / 'tasks').iterdir()):
            if not directory.is_dir(): continue
            task, trajectory, result = [read(directory / name) for name in
                                        ('task.json', 'trajectory.json', 'result.json')]
            ident = task['task_id']
            if ident not in tasks:
                tasks[ident] = dict(**task, models={})
            elif task != {k: tasks[ident][k] for k in task}:
                raise ValueError('Task identities differ across cohorts')
            frames = []; previous_history = []; previous_decisions = 0; observed_pages = {}
            for path in sorted(directory.glob('state-*.json.gz')):
                with gzip.open(path, 'rt') as handle: state = json.load(handle)
                page = state['page']; history = state['history']; decisions = state['decisions']
                final = path.name == 'state-0061.json.gz'
                delta = history[len(previous_history):]
                decision = decisions[-1] if len(decisions) > previous_decisions else None
                outcome = 'initial' if not frames else 'observation'
                if decision:
                    outcome = 'executed' if delta else ('terminal' if state['status'] in ('done','blocked') else 'not_executed')
                if final: outcome = 'final'
                screenshot = image(page.get('screenshot'))
                # For an unexecuted attempt the current snapshot can still be
                # the exact decision state; require its fingerprint to agree.
                observed_pages.setdefault(page['fingerprint'], dict(actions=page['actions'], image=screenshot,
                    w=page['w'], h=page['h']))
                frames.append(dict(number=int(path.name[6:10]), status=state['status'], outcome=outcome,
                    screenshot=screenshot, url=page['url'], title=page.get('title',''),
                    target_mapping=target_mapping(decision, observed_pages),
                    page_text=page.get('text',''), omitted_actions=page.get('omitted_actions',0),
                    elapsed_ms=state['elapsed_ms'], total_actions=len(history), total_decisions=len(decisions),
                    actions=delta, available_actions=[{k: a[k] for k in ('id','kind','label','value','rect') if k in a}
                                                    for a in page.get('actions',[])],
                    decision={k:v for k,v in decision.items() if k in ('choice','operation','target','confidence',
                        'operation_probabilities','target_probabilities','raw_answers','model','usage','latency_ms')}
                        if decision else None))
                previous_history, previous_decisions = history, len(decisions)
            # The final frame must show the independently verified fresh JPEG.
            final_image = image(base64.b64encode((directory / 'final.jpg').read_bytes()).decode())
            if frames[-1]['screenshot'] != final_image:
                raise ValueError('Final screenshot differs from saved terminal evidence')
            tasks[ident]['models'][key] = dict(result=result, frames=frames,
                history=trajectory['history'], text_calls=trajectory['text_calls'],
                operations=dict(Counter(d['operation'] for d in trajectory['decisions'])),
                nonexecuted_decisions=sum(f['outcome']=='not_executed' for f in frames),
                judge=read(directory / 'judge-response.json'))
    plan = read(roots[0][2] / 'plan.json')
    ordered = [tasks[t['task_id']] for t in plan['tasks']]
    if len(ordered) != 10 or any(set(t['models']) != {'jev','kev08','kev27'} for t in ordered):
        raise ValueError('Expected 10 matched tasks and all three providers')
    return dict(tasks=ordered, models=models, images=images,
        upstream_commit='1231850a0bf1a0c0341fe408ef1668dbbfdfac46',
        kev_commit='fe64b1274ea7f80d4095866df90666abb03e9cf6',
        protocol=dict(browser='Browser Use remote, no proxy, fresh session per task',
            viewport='1120 × 780', max_actions=30, max_decisions=60, timeout_seconds=600,
            text_helper='GPT-4.1-mini-2025-04-14', judge='o4-mini / AgentTrek',
            jev_seconds=148, kev_seconds_including_retries=600))


def render(output):
    output = Path(output).resolve()
    if output.is_relative_to(REPO):
        raise ValueError('Task payloads and screenshots must remain in private runtime storage')
    payload = build()
    # Safe JSON embedding even when hostile page text contains closing script tags.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',',':')).replace('<','\\u003c')
    template = (REPO / 'scripts/templates/jev_kev_review.html').read_text()
    if template.count('__REVIEW_DATA__') != 1:
        raise ValueError('Unexpected review template')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__REVIEW_DATA__', encoded))
    receipt = dict(html=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        bytes=output.stat().st_size, tasks=len(payload['tasks']), cohorts=len(payload['models']),
        unique_screenshots=len(payload['images']), self_contained=True, public=False)
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=RUNTIME / 'visualizations/jev-kev-review-20261004.html')
    print(json.dumps(render(parser.parse_args().output), indent=2))
