#!/usr/bin/env python3
"""Private spot-check page for judge disagreements (human labels are the ground truth).

Shows, for a seeded sample of audited episodes, exactly what the judges saw: task, action
history, final answer and the last three screenshots, plus both judges' reasoning hidden
behind a toggle so the reviewer can label first. Labels are kept in the browser and
exported as JSON. Contains task payloads: write under the runtime analysis directory only.
"""
import argparse
import html
import json
from pathlib import Path
import random

from audit_judge_agreement import build_sample, load_episode
from build_rl_behavior_gallery import thumbnail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--model', default='gpt-5.6-terra')
    p.add_argument('--disagreements', type=int, default=40)
    p.add_argument('--agreements', type=int, default=10, help='agreeing success verdicts mixed in as controls')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    import openwebrl.reward_browser as rb
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (a.audit_dir/'verdicts.jsonl').read_text().splitlines()]
    second = {(r['cohort'], r['task_id']): r for r in rows if r['model'] == a.model}
    rng = random.Random(a.seed)
    flipped = sorted(k for k, r in second.items() if r['original'] == 1 and r['verdict'] == 0)
    agreed = sorted(k for k, r in second.items() if r['original'] == 1 and r['verdict'] == 1)
    grounding_path = a.audit_dir/'grounding.jsonl'
    grounding = {(r['cohort'], r['task_id']): r for r in map(json.loads, grounding_path.read_text().splitlines())} if grounding_path.exists() else {}
    picks = rng.sample(flipped, min(a.disagreements, len(flipped))) + rng.sample(agreed, min(a.agreements, len(agreed)))
    rng.shuffle(picks)
    cards = []
    args = type('A', (), dict(browser_response_format_mode='browser_env'))()
    for n, (cohort, task_id) in enumerate(picks):
        record, d = load_episode(cohort, task_id)
        sample = build_sample(d)
        imgs = ''
        steps = [k for k, t in enumerate(d['turns'], 1) if ((t.get('multimodal_inputs') or {}).get('images'))][-3:]
        for k, img in enumerate(sample.multimodal_inputs['judge_images']):
            name = f'j{n:02d}-{k}.jpg'
            (a.out_dir/name).write_bytes(thumbnail(img if img.startswith('data:') else 'data:image/png;base64,'+img, 640))
            label = f'S{steps[k]}' if k < len(steps) else ''
            imgs += f'<figure><img src="{name}" loading="lazy"><figcaption>{label}</figcaption></figure>'
        g = (grounding.get((cohort, task_id)) or {}).get('grounding') or {}
        ground = ''
        if g:
            rows_html = ''.join(f'<tr class="{html.escape(str(c.get("status")))}"><td>{html.escape(str(c.get("claim")))}</td>'
                                f'<td>{html.escape(str(c.get("status")))}</td><td>{html.escape(", ".join(map(str, c.get("evidence") or [])))}</td>'
                                f'<td>{html.escape(str(c.get("note") or ""))}</td></tr>' for c in g.get('claims') or [])
            ground = (f'<details><summary>Answer grounding check ({html.escape(str(g.get("unsupported_claims")))} unsupported claims; '
                      f'completes task: {html.escape(str(g.get("answer_completes_task")))})</summary><p>{html.escape(str(g.get("summary") or ""))}</p>'
                      f'<table><tr><th>Claim</th><th>Status</th><th>Evidence</th><th>Note</th></tr>{rows_html}</table>'
                      f'<p class="hint">S = screenshot before step k, T = browser response, R = agent reasoning (not evidence).</p></details>')
        answer = rb._extract_final_answer(args, sample.response or '')
        r2 = second[(cohort, task_id)]
        cards.append(f'''<section class="card" data-id="{html.escape(cohort)}|{html.escape(task_id)}">
<h2>#{n+1}</h2><p class="meta"><b>Task:</b> {html.escape(sample.metadata.get("intent", ""))}<br>
<b>Final answer:</b> {html.escape(answer)}</p>
<details><summary>Action history</summary><pre>{html.escape(sample.metadata.get("trajectory_action_history_text") or "")}</pre></details>
<div class="shots">{imgs}</div>
<fieldset><legend>Your verdict (from the screenshots, history and answer)</legend>
<label><input type="radio" name="v{n}" value="success"> Success</label>
<label><input type="radio" name="v{n}" value="failure"> Failure</label>
<label><input type="radio" name="v{n}" value="unsure"> Unsure</label>
<input type="text" name="c{n}" placeholder="optional note"></fieldset>
<details><summary>Reveal judges</summary><p><b>GPT-4.1 (training judge):</b> success</p>
<pre>{html.escape(record.get("reward_metadata", {}).get("judge_text", "") or "")}</pre>
<p><b>{html.escape(a.model)}:</b> {"success" if r2["verdict"] == 1 else "failure"}</p><pre>{html.escape(r2["text"] or "")}</pre></details>
{ground}
</section>''')
    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Judge Spot Check</title><style>
body{{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:16px auto;padding:0 16px;color:#1d1d1f;background:#fff}}
.card{{border-top:2px solid #ddd;padding:8px 0 16px}} .meta{{background:#f4f4f6;padding:8px;border-radius:6px}}
.shots{{display:flex;flex-wrap:wrap;gap:6px}} .shots figure{{margin:0;width:32%;min-width:240px}} .shots img{{width:100%;border:1px solid #ccc}}
figcaption{{font-size:12px;color:#555}} table{{border-collapse:collapse;font-size:12px;width:100%}} td,th{{border:1px solid #ddd;padding:4px;vertical-align:top}}
tr.not_found td{{background:#fdecea}} tr.partial td{{background:#fff7e0}} tr.supported td{{background:#eaf6ec}} .hint{{font-size:12px;color:#666}} pre{{white-space:pre-wrap;font-size:12px;background:#fafafa;padding:6px}}
fieldset{{margin:8px 0}} label{{margin-right:14px}} #bar{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd}}
</style></head><body><div id="bar"><b>Judge spot check</b> · {len(picks)} episodes · labelled <span id="count">0</span>
· <button onclick="exportLabels()">Export labels (JSON)</button></div>
<p>Private: task payloads. All shown episodes were judged <b>success</b> by GPT-4.1 (the training reward). A mix of cases where the
second judge disagreed and where it agreed is shown in random order; label before revealing the judges. Labels are saved in this browser.</p>
{"".join(cards)}
<script>
const KEY='judge-spotcheck-20261024';
function load(){{try{{return JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{return {{}}}}}}
function save(s){{try{{localStorage.setItem(KEY,JSON.stringify(s))}}catch(e){{}}}}
const state=load();
document.querySelectorAll('.card').forEach((card,n)=>{{
  const id=card.dataset.id, s=state[id]||{{}};
  card.querySelectorAll('input[type=radio]').forEach(r=>{{ if(r.value===s.verdict) r.checked=true;
    r.addEventListener('change',()=>{{state[id]={{...(state[id]||{{}}),verdict:r.value}};save(state);count()}})}});
  const t=card.querySelector('input[type=text]'); t.value=s.note||'';
  t.addEventListener('input',()=>{{state[id]={{...(state[id]||{{}}),note:t.value}};save(state)}});
}});
function count(){{document.getElementById('count').textContent=Object.values(state).filter(v=>v.verdict).length}}
function exportLabels(){{const b=new Blob([JSON.stringify(state,null,1)],{{type:'application/json'}});
  const u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='judge-spotcheck-labels.json';a.click();URL.revokeObjectURL(u)}}
count();
</script></body></html>'''
    (a.out_dir/'index.html').write_text(page)
    (a.out_dir/'picks.json').write_text(json.dumps([dict(cohort=c, task_id=t, second_judge=second[(c, t)]['verdict']) for c, t in picks], indent=1))
    print(a.out_dir/'index.html')


if __name__ == '__main__':
    main()
