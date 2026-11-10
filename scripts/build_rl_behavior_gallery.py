#!/usr/bin/env python3
"""Private HTML gallery of representative held-out episodes per behavior category.

Contains task text, screenshots and answers, so it is written under the runtime analysis
directory and must not be published or committed. Picks a fixed number of episodes per
category with a seeded choice among episodes of at most --max-steps steps, then renders
each turn's screenshot (downscaled JPEG), the model's action, the browser response and
the judge's reasoning. Run in a CPU job: it loads full rollout archives.
"""
import argparse
import base64
import html
import io
import json
from pathlib import Path
import random
import re

from analyze_rl_eval_failures import EVALS, TOOL_CALL, classify

SPECS = [  # (category, cohort to draw from, label)
    ('success', 'outcome-only-56groups-20261006-iter90', 'Success'),
    ('wrong_answer', 'outcome-only-56groups-20261006-iter90', 'Wrong answer (finished, judged failure)'),
    ('repeat_loop', 'outcome-only-56groups-20261006-iter90', 'Repeated action (same action 3+ times)'),
    ('claims_infeasible', 'outcome-only-56groups-20261006-iter90', 'Claims the task is impossible'),
    ('turn_limit', 'outcome-only-56groups-20261006-iter10', 'Turn limit (early checkpoint, iteration 10)'),
    ('done_then_step_error', 'outcome-only-56groups-20261006-iter90', 'Invalid: browser step error right after done'),
    ('repeat_loop', 'arm-mixed-bonus-iter90-340425', 'ARM bonus iteration 90: repeated action'),
]


def category(r, max_steps):
    c = classify(r, max_steps)
    if c == 'invalid:env_step_error':
        calls = [s['call'] for s in r.get('steps') or [] if s.get('call')]
        if calls and calls[-1].get('name') == 'done':
            return 'done_then_step_error'
    return c


def thumbnail(data_url, width):
    from PIL import Image
    raw = base64.b64decode(data_url.split(',', 1)[1])
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    image.thumbnail((width, width*2))
    out = io.BytesIO()
    image.save(out, 'JPEG', quality=70)
    return out.getvalue()


def render_episode(record, cohort, out_dir, key, width):
    import torch
    root = EVALS/cohort/'rollouts'
    summary = json.loads(next(p for p in root.glob('*.json') if json.loads(p.read_text())['task_id'] == record['task_id']).read_text())
    d = torch.load(root/summary['rollout_file'], weights_only=False, map_location='cpu')
    parts = []
    for i, t in enumerate(d['turns']):
        m = t.get('metadata') or {}
        images = ((t.get('multimodal_inputs') or {}).get('images') or [])
        img = ''
        if images and isinstance(images[-1], str) and images[-1].startswith('data:image'):
            name = f'{key}-t{i:02d}.jpg'
            (out_dir/name).write_bytes(thumbnail(images[-1], width))
            img = f'<img src="{name}" loading="lazy">'
        response = t.get('response') or ''
        think = re.sub(r'<tool_call>.*', '', response, flags=re.S).replace('<think>', '').replace('</think>', '').strip()
        call = TOOL_CALL.search(response)
        tools = '<br>'.join(html.escape(f"{x.get('tool_name')}: {str(x.get('tool_response'))[:300]}") for x in m.get('step_tool_responses') or [])
        parts.append(f'<div class="turn"><div class="shot">{img}</div><div class="txt"><b>Step {i+1}</b>'
                     f'<pre class="call">{html.escape(call.group(1) if call else "(no tool call)")}</pre>'
                     f'<div class="tool">{tools or "<i>no browser response recorded</i>"}</div>'
                     f'<details><summary>Model reasoning</summary><pre>{html.escape(think[-1500:])}</pre></details></div></div>')
    return (f'<p class="meta">Task: {html.escape(str(record.get("intent")))}<br>Start: {html.escape(str(record.get("start_url")))}'
            f' · steps {record.get("total_steps")} · end {html.escape(str(record.get("terminate_reason")))}'
            f' · success {record["success"]} · valid {record["valid"]}</p>{"".join(parts)}'
            f'<details open><summary>Judge reasoning</summary><pre>{html.escape(record.get("judge_text") or "(no judge verdict: invalid episode)")}</pre></details>')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compact-dir', type=Path, required=True, help='JSONL written by analyze_rl_eval_failures.py --extract')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--per-category', type=int, default=2)
    p.add_argument('--max-steps', type=int, default=12)
    p.add_argument('--width', type=int, default=520)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    sections, toc = [], []
    for n, (cat, cohort, label) in enumerate(SPECS):
        rows = [json.loads(x) for x in (a.compact_dir/f'{cohort}.jsonl').read_text().splitlines()]
        pool = sorted((r for r in rows if category(r, 30) == cat and (r.get('total_steps') or 99) <= max(a.max_steps, 30 if cat == 'turn_limit' else 0)),
                      key=lambda r: r['task_id'])
        picks = rng.sample(pool, min(a.per_category, len(pool)))
        toc.append(f'<li><a href="#s{n}">{html.escape(label)}</a> ({len(pool)} eligible in {cohort})</li>')
        body = ''.join(f'<h3>Example {k+1}</h3>{render_episode(r, cohort, a.out_dir, f"s{n}e{k}", a.width)}' for k, r in enumerate(picks))
        sections.append(f'<section id="s{n}"><h2>{html.escape(label)}</h2>{body}</section>')
    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL Behavior Gallery</title><style>
body{{font:14px/1.45 system-ui,sans-serif;margin:16px auto;max-width:1200px;padding:0 16px;color:#1d1d1f;background:#fff}}
h2{{border-top:2px solid #ddd;padding-top:12px}} .meta{{background:#f4f4f6;padding:8px;border-radius:6px}}
.turn{{display:flex;gap:12px;margin:10px 0;border-bottom:1px solid #eee;padding-bottom:8px}} .shot img{{width:{a.width}px;max-width:45vw;border:1px solid #ccc}}
.txt{{flex:1;min-width:0}} pre{{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:4px 0}} .call{{background:#eef6ff;padding:6px}}
.tool{{font-size:12px;color:#444}} @media (max-width:700px){{.turn{{flex-direction:column}} .shot img{{max-width:100%}}}}
</style></head><body><h1>RL behavior gallery</h1>
<p>Private: contains task text, screenshots and answers from saved held-out evaluations. Do not publish.
Categories use the rules in RL_BEHAVIOR.md; examples are a seeded random choice (seed {a.seed}) among episodes of at most {a.max_steps} steps (30 for turn-limit).</p>
<ul>{"".join(toc)}</ul>{"".join(sections)}</body></html>'''
    (a.out_dir/'index.html').write_text(page)
    print(a.out_dir/'index.html')


if __name__ == '__main__':
    main()
