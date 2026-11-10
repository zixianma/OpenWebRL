#!/usr/bin/env python3
"""Answer-grounding check for episodes the training judge (GPT-4.1) scored as successes.

A vision model splits the final answer into atomic claims and, for each claim, cites where the
trajectory supports it: screenshot ids (S<k>), browser-response ids (T<k>) or agent-reasoning
ids (R<k>, not evidence on its own), or marks it not found. Uses the remaining budget of the
approved $30 judge-audit cap (shared budget file). Episodes: GPT-4.1 successes from the judge
audit (56-group run, iterations 10 and 90). Private per-episode output; aggregate summary.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_rl_eval_failures import TOOL_CALL  # noqa: E402
from audit_judge_agreement import Budget, capped_client, load_episode  # noqa: E402

SYSTEM = """You audit whether a web agent's final answer is grounded in what it actually observed.
You get the task, the agent's final answer, and the trajectory as numbered items:
  R<k> = the agent's own reasoning before step k (NOT evidence by itself),
  A<k> = the action it took at step k,
  T<k> = the browser's text response to step k,
  S<k> = a screenshot of the page the agent saw before acting at step k.
Split the final answer into its atomic factual claims (ignore politeness and restated task text).
For each claim decide whether the observations support it:
  "supported": a screenshot or browser response shows it; cite those ids.
  "partial": only part of it is visible, or it is only weakly implied; cite ids and say what is missing.
  "not_found": no screenshot or browser response shows it (it may appear only in reasoning; cite R ids if so).
Also report whether the answer actually completes the task. Reply with JSON only:
{"claims":[{"claim":"...","status":"supported|partial|not_found","evidence":["S3","T4"],"note":"..."}],
 "answer_completes_task": true|false, "unsupported_claims": <int>, "summary": "<one sentence>"}"""


def build_messages(record, d, max_shots):
    turns = d['turns']
    task = (turns[0].get('metadata') or {}).get('intent', '')
    parts, shots = [], []
    for k, t in enumerate(turns, 1):
        m = t.get('metadata') or {}
        resp = t.get('response') or ''
        reasoning = re.sub(r'<tool_call>.*', '', resp, flags=re.S).replace('<think>', '').replace('</think>', '').strip()
        call = TOOL_CALL.search(resp)
        tools = ' | '.join(str(x.get('tool_response'))[:400] for x in m.get('step_tool_responses') or [])
        parts.append(f'R{k}: {reasoning[-700:]}\nA{k}: {call.group(1) if call else "(none)"}\nT{k}: {tools or "(no response recorded)"}')
        images = (t.get('multimodal_inputs') or {}).get('images') or []
        if images:
            shots.append((k, images[-1]))
    answer = ''
    for t in reversed(turns):
        c = TOOL_CALL.search(t.get('response') or '')
        if c and '"done"' in c.group(1):
            try:
                answer = json.loads(c.group(1)).get('arguments', {}).get('response', '')
            except json.JSONDecodeError:
                answer = c.group(1)
            break
    content = [{'type': 'text', 'text': f'TASK: {task}\n\nFINAL ANSWER: {answer}\n\nTRAJECTORY:\n' + '\n\n'.join(parts)}]
    for k, img in shots[-max_shots:]:
        url = img if img.startswith('data:') else 'data:image/png;base64,'+img
        content += [{'type': 'text', 'text': f'S{k}:'}, {'type': 'image_url', 'image_url': {'url': url, 'detail': 'high'}}]
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}], answer, [k for k, _ in shots[-max_shots:]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--model', default='gpt-5.6-terra')
    p.add_argument('--max-shots', type=int, default=6)
    p.add_argument('--cap-usd', type=float, default=30.)
    p.add_argument('--concurrency', type=int, default=4)
    a = p.parse_args()
    from dotenv import dotenv_values
    for k, v in dotenv_values(Path(__file__).resolve().parents[1]/'.env').items():
        if v and k.startswith(('OPENAI_', 'JUDGE_')):
            os.environ.setdefault(k, v)
    os.environ.setdefault('JUDGE_API_BASE', 'https://api.openai.com/v1')
    import openwebrl.reward_browser as rb
    budget = Budget(a.audit_dir/'budget.json', a.cap_usd)  # shared with the judge audit: one $30 cap
    client = capped_client(rb._get_openai_client(method='served'), budget, a.model)
    verdicts = [json.loads(x) for x in (a.audit_dir/'verdicts.jsonl').read_text().splitlines()]
    second = {(r['cohort'], r['task_id']): r for r in verdicts if r['model'] == a.model}
    picks = sorted(k for k, r in second.items() if r['original'] == 1)
    out = a.audit_dir/'grounding.jsonl'
    done = {(r['cohort'], r['task_id']) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()

    async def one(cohort, task_id, sem):
        if (cohort, task_id) in done:
            return
        async with sem:
            record, d = await asyncio.to_thread(load_episode, cohort, task_id)
            messages, answer, shot_ids = build_messages(record, d, a.max_shots)
            try:
                resp = await client.chat.completions.create(model=a.model, messages=messages, seed=42,
                                                            response_format={'type': 'json_object'})
            except RuntimeError as exc:
                if 'cap' in str(exc):
                    return
                raise
            text = resp.choices[0].message.content or ''
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            with out.open('a') as f:
                f.write(json.dumps(dict(cohort=cohort, task_id=task_id, iteration=10 if 'iter10' in cohort else 90,
                                        answer=answer, screenshots_shown=shot_ids, second_judge=second[(cohort, task_id)]['verdict'],
                                        grounding=parsed, raw=None if parsed else text))+'\n')

    async def run():
        sem = asyncio.Semaphore(a.concurrency)
        await asyncio.gather(*(one(c, t, sem) for c, t in picks))
    asyncio.run(run())

    rows = [json.loads(x) for x in out.read_text().splitlines()]
    cells = {}
    for r in rows:
        g = r['grounding'] or {}
        claims = g.get('claims') or []
        key = f"iter{r['iteration']}_second_judge_{'success' if r['second_judge'] == 1 else 'failure'}"
        c = cells.setdefault(key, dict(episodes=0, parsed=0, claims=0, supported=0, partial=0, not_found=0,
                                       episodes_with_unsupported=0, answer_completes_task=0))
        c['episodes'] += 1
        if r['grounding'] is None:
            continue
        c['parsed'] += 1
        c['claims'] += len(claims)
        for cl in claims:
            c[cl.get('status') if cl.get('status') in ('supported', 'partial', 'not_found') else 'not_found'] += 1
        c['episodes_with_unsupported'] += any(cl.get('status') == 'not_found' for cl in claims)
        c['answer_completes_task'] += bool(g.get('answer_completes_task'))
    summary = dict(model=a.model, episodes='GPT-4.1 action_history successes from the judge audit', budget=budget.state, cells=cells)
    (a.audit_dir/'grounding-summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
