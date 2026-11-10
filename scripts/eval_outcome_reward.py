#!/usr/bin/env python3
"""Evaluate the RL outcome reward as a verifier, on two benchmarks.

Rewards scored:
  train  - the training reward: GPT-4.1 action_history judge (reward_browser.compute_judge_reward_actionhistory,
           unchanged code path, last 3 screenshots).
  gated  - train x G, the grounded-answer gate proposed in docs/RL_REWARD_DESIGN.md. G = no not_found claim and
           answer_completes_task (scripts/audit_answer_grounding.py prompt, last 6 screenshots); only run where train = 1.
  checklist - train x C: requirements extracted once per task from the task text alone (frozen, cached in
           checklists.jsonl), then checked against the same trajectory evidence as G; C = every requirement
           satisfied. Reported alone (train x C) and with the grounding gate (train x G x C).
Reference judges:
  eval   - the Online-Mind2Web evaluation judge (AgentTrek prompt from openwebrl/eval/reward_online_mind2web.py,
           final screenshot), default gpt-5.6-terra (o4-mini was retired 2026-10-23).

Benchmarks:
  heldout - saved held-out full300 episodes of RL checkpoints (default: outcome-only 56-group, iterations 10/90).
            `train` is the saved evaluation verdict (same judge, model and prompt), so it costs nothing; `eval` and
            `gated` are new calls. Reports agreement of each reward with the eval judge.
  cuavb   - CUAVerifierBench (microsoft/CUAVerifierBench, revision c19eb32; Fara-7B trajectories with human outcome
            labels). Fara steps are mapped to our judge inputs: thought -> reasoning, action -> tool call, terminate ->
            done(response=final_answer), screenshot k-1 -> the page before step k (om2w split; the internal split has
            fewer screenshots than actions, so its screenshots carry no step labels). Reports agreement with the
            adjudicated human label, alongside the dataset's own verifier columns and individual reviewers.

Every call goes through a budget-capped client (--cap-usd, upper-bound reservation per request). --dry-run lists
the planned calls and an upper-bound cost without calling any API. Per-item outputs (private: contain task text) go
to --out-dir/<benchmark>/<judge>.jsonl and are resumed; summary.json holds the metrics with task-bootstrap 95% CIs.
Run as a CPU Slurm job: loading saved episodes does not fit the login node.
"""
import argparse
import asyncio
import base64
import glob
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_rl_eval_failures import EVALS  # noqa: E402
from audit_answer_grounding import SYSTEM as GROUNDING_SYSTEM, build_messages as grounding_messages_heldout  # noqa: E402
from audit_judge_agreement import COHORTS, Budget, build_sample, capped_client, judge_args  # noqa: E402

BENCH = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/datasets/cuaverifierbench-c19eb32/data')
BENCH_SPLITS = ('fara7b_om2w_browserbase', 'internal')
UPPER_USD = {'train': 0.03, 'eval': 0.05, 'gated': 0.10, 'checklist': 0.10}  # per-call upper bounds for --dry-run only


# ---- items -------------------------------------------------------------------------------------------

def heldout_items(cohorts):
    items = []
    for cohort in cohorts:
        for p in sorted((EVALS/cohort/'rollouts').glob('*.json')):
            r = json.loads(p.read_text())
            if r['metrics']['valid_trajectories'] > 0 and r.get('terminal_status') == 'completed':
                items.append(dict(bench='heldout', group=cohort, id=f"{cohort}/{r['task_id']}", task_id=r['task_id'],
                                  rollout_file=r['rollout_file'], saved_train=float(r['metrics']['successes'] > 0)))
    return items


def bench_rows(split, columns=None):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(str(BENCH/'trajectories'/f'{split}-*.parquet'))):
        rows += pq.read_table(f, columns=columns).to_pylist()
    return rows


def bench_items(splits):
    cols = ['task_id', 'final_human_outcome_label', 'uv_outcome_success', 'mm_is_success', 'verifier_is_success', 'gpt_eval_json']
    items = []
    for split in splits:
        for r in bench_rows(split, cols):
            gpt = json.loads(r['gpt_eval_json'] or '{}').get('score')
            items.append(dict(bench='cuavb', group=split, id=f"{split}/{r['task_id']}", task_id=r['task_id'],
                              human=r['final_human_outcome_label'],
                              baselines=dict(uv=r['uv_outcome_success'], legacy_mm=r['mm_is_success'],
                                             legacy_text=r['verifier_is_success'],
                                             dataset_gpt=gpt if gpt in (0, 1) else None)))
    return items


def fara_steps(row):
    """(thought, action name, action params) per Fara step, plus the final answer."""
    steps = []
    for line in row['web_surfer_log'].splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        args = e.get('arguments') or {}
        if isinstance(args, str):
            args = json.loads(args)
        name = args.get('action') or e.get('action') or 'unknown'
        params = {k: v for k, v in args.items() if k not in ('action', 'thoughts')}
        steps.append((args.get('thoughts') or '', name, params))
    answer = (row['final_answer'] or '').strip()
    answer = '' if answer == '<no_answer>' else answer
    if steps and steps[-1][1] == 'terminate':
        steps[-1] = (steps[-1][0], 'done', {'response': answer})
    return steps, answer


def data_url(raw):
    mime = 'image/jpeg' if raw[:3] == b'\xff\xd8\xff' else 'image/webp' if raw[8:12] == b'WEBP' else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(raw).decode()


def fara_screens(row, split):
    """[(step label or None, image data URL)] with label k = page before step k where the alignment is known."""
    shots = [data_url(s['bytes']) for s in row['screenshots']]
    n_steps = len(fara_steps(row)[0])
    if split == 'fara7b_om2w_browserbase' and len(shots) == n_steps + 1:
        return [(k + 1, s) for k, s in enumerate(shots)]  # the last one is the page after the final step
    return [(None, s) for s in shots]


# ---- judge inputs ------------------------------------------------------------------------------------

def train_sample_for_fara(row, split):
    from openwebrl.reward_browser import _format_tool_call_for_action_history
    from slime.utils.types import Sample
    steps, answer = fara_steps(row)
    screens = [(k, s) for k, s in fara_screens(row, split) if k is None or k <= len(steps)][-3:]
    history = '\n'.join(f'Step {k}: action=[{_format_tool_call_for_action_history(n, p)}]'
                        for k, (_, n, p) in enumerate(steps, 1))
    sample = Sample()
    sample.response = '<tool_call>' + json.dumps({'name': 'done', 'arguments': {'response': answer}}) + '</tool_call>' if answer else ''
    labelled = screens and screens[-1][0] is not None
    sample.metadata = dict(intent=row['instruction'], trajectory_action_history_text=history,
                           **({'turn_index': screens[-1][0] - 1} if labelled else {}))
    sample.multimodal_inputs = dict(judge_images=[s for _, s in screens])
    return sample


def eval_messages(task, thoughts, actions, last_b64):
    from openwebrl.eval.reward_online_mind2web import SYSTEM_PROMPT, USER_PROMPT, _format_trajectory
    url = last_b64 if last_b64.startswith('data:') else f'data:image/png;base64,{last_b64}'
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': USER_PROMPT.format(task=task, thoughts_and_actions=_format_trajectory(thoughts, actions))},
                {'type': 'image_url', 'image_url': {'url': url, 'detail': 'high'}}]}]


def eval_inputs_heldout(d):
    from openwebrl.base.utils import ToolParser
    from openwebrl.eval._shared import _TOOLS_INFO
    from openwebrl.eval.reward_online_mind2web import _extract_thoughts_and_actions
    last = next(t for t in d['turns'] if (t.get('metadata') or {}).get('is_last_turn'))
    m = last['metadata']
    thoughts, actions = _extract_thoughts_and_actions(m['messages'], ToolParser(_TOOLS_INFO, parser_type='qwen25'))
    return m['intent'], thoughts, actions, (m.get('full_image_list') or [None])[-1]


def eval_inputs_fara(row, split):
    from openwebrl.eval.reward_online_mind2web import _format_action
    steps, _ = fara_steps(row)
    screens = fara_screens(row, split)
    return (row['instruction'], [t for t, _, _ in steps], [_format_action(n, p) for _, n, p in steps],
            screens[-1][1] if screens else None)


def grounding_messages_fara(row, split, max_shots=6):
    steps, answer = fara_steps(row)
    parts = [f'R{k}: {t[-700:]}\nA{k}: {json.dumps({"name": n, "arguments": p})}\nT{k}: (no response recorded)'
             for k, (t, n, p) in enumerate(steps, 1)]
    screens = fara_screens(row, split)
    if screens and screens[0][0] is not None:
        shots = [(k, s) for k, s in screens if k <= len(steps)][-max_shots:]
    else:  # unknown alignment: show the last screenshots under their position in the trajectory
        shots = [(f'{i + 1}', s) for i, (_, s) in enumerate(screens)][-max_shots:]
    content = [{'type': 'text', 'text': f'TASK: {row["instruction"]}\n\nFINAL ANSWER: {answer}\n\nTRAJECTORY:\n' + '\n\n'.join(parts)}]
    for k, s in shots:
        content += [{'type': 'text', 'text': f'S{k}:'}, {'type': 'image_url', 'image_url': {'url': s, 'detail': 'high'}}]
    return [{'role': 'system', 'content': GROUNDING_SYSTEM}, {'role': 'user', 'content': content}]


CHECKLIST_EXTRACT = """You list the explicit requirements of a web task, before seeing any attempt at it.
Write one item per requirement the task states or directly implies: the target item or information, sort order or
ranking ("most", "best", "top", "cheapest", "latest"), filters and attributes (color, size, price, rating, type),
date or time range, location or radius, quantity, and the required end state (for example "added to cart",
"page opened", "form submitted", "answer reported"). Do not add requirements the task does not ask for.
Reply with JSON only: {"constraints": [{"id": "C1", "requirement": "...", "kind": "target|ranking|filter|date|location|quantity|end_state|other"}]}"""

CHECKLIST_CHECK = """You check a web agent's finished attempt against a fixed list of task requirements.
You get the task, the requirement list, the agent's final answer, and the trajectory as numbered items:
  R<k> = the agent's own reasoning before step k (NOT evidence by itself),
  A<k> = the action it took at step k,
  T<k> = the browser's text response to step k,
  S<k> = a screenshot of the page the agent saw before acting at step k.
For each requirement decide, from the observations and actions (not from the agent's reasoning or claims):
  "satisfied": the observations show the requirement was met (cite S/T/A ids);
  "not_satisfied": the observations show it was not met, or the agent skipped it;
  "not_verifiable": nothing observed shows whether it was met.
A requirement the task makes impossible counts as satisfied only if the observations show it is impossible and the
answer says so. Reply with JSON only:
{"requirements": [{"id": "C1", "status": "satisfied|not_satisfied|not_verifiable", "evidence": ["S3"], "note": "..."}],
 "summary": "<one sentence>"}"""


def checklist_gate(parsed, constraints):
    if not isinstance(parsed, dict) or not isinstance(parsed.get('requirements'), list):
        return None
    status = {r.get('id'): r.get('status') for r in parsed['requirements'] if isinstance(r, dict)}
    return float(all(status.get(c['id']) == 'satisfied' for c in constraints))


def checklist_messages(grounding_msgs, constraints):
    """Same trajectory evidence as the grounding check, with the checklist prompt and the frozen requirement list."""
    user = [dict(c) for c in grounding_msgs[1]['content']]
    reqs = '\n'.join(f"{c['id']}: {c['requirement']}" for c in constraints)
    user[0]['text'] = user[0]['text'].replace('\n\nFINAL ANSWER:', f'\n\nREQUIREMENTS:\n{reqs}\n\nFINAL ANSWER:', 1)
    return [{'role': 'system', 'content': CHECKLIST_CHECK}, {'role': 'user', 'content': user}]


def gate(parsed):
    if not isinstance(parsed, dict) or not isinstance(parsed.get('claims'), list):
        return None
    ok = all(c.get('status') in ('supported', 'partial') for c in parsed['claims'] if isinstance(c, dict))
    return float(ok and parsed.get('answer_completes_task') is True)


# ---- metrics -----------------------------------------------------------------------------------------

def binary_metrics(pairs):
    """pairs: [(prediction, reference)] with both in {0, 1}."""
    n = len(pairs)
    tp = sum(p == 1 and r == 1 for p, r in pairs); fp = sum(p == 1 and r == 0 for p, r in pairs)
    fn = sum(p == 0 and r == 1 for p, r in pairs); tn = n - tp - fp - fn
    acc = (tp + tn)/n if n else None
    pe = ((tp + fp)*(tp + fn) + (fn + tn)*(fp + tn))/n**2 if n else None
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn, accuracy=acc,
                kappa=(acc - pe)/(1 - pe) if n and pe < 1 else None,
                precision=tp/(tp + fp) if tp + fp else None, recall=tp/(tp + fn) if tp + fn else None,
                false_positive_rate=fp/(fp + tn) if fp + tn else None,
                predicted_success_rate=(tp + fp)/n if n else None, reference_success_rate=(tp + fn)/n if n else None)


def with_ci(pairs, n_boot=2000, seed=0):
    out = binary_metrics(pairs)
    rng = random.Random(seed)
    boots = [binary_metrics([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(n_boot)] if pairs else []
    for k in ('accuracy', 'kappa', 'precision', 'recall', 'false_positive_rate'):
        xs = sorted(b[k] for b in boots if b[k] is not None)
        out[k + '_ci95'] = [xs[int(0.025*len(xs))], xs[int(0.975*len(xs)) - 1]] if len(xs) > 40 else None
    return out


# ---- running -----------------------------------------------------------------------------------------

def read_jsonl(path):
    return {r['id']: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


async def call_json(client, model, messages):
    resp = await client.chat.completions.create(model=model, messages=messages, seed=42, response_format={'type': 'json_object'})
    text = resp.choices[0].message.content or ''
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        return None, text


class Checklists:
    """Requirement lists extracted once per task text from the task alone, cached in a file and frozen thereafter."""

    def __init__(self, path, client, model):
        self.path, self.client, self.model, self.locks = path, client, model, {}
        self.cache = {r['task']: r['constraints'] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}

    async def get(self, task):
        lock = self.locks.setdefault(task, asyncio.Lock())
        async with lock:
            if task not in self.cache:
                parsed, _ = await call_json(self.client, self.model, [{'role': 'system', 'content': CHECKLIST_EXTRACT},
                                                                      {'role': 'user', 'content': f'TASK: {task}'}])
                cons = (parsed or {}).get('constraints')
                if not isinstance(cons, list) or not cons or not all(isinstance(c, dict) and c.get('id') and c.get('requirement') for c in cons):
                    return None
                self.cache[task] = cons
                with self.path.open('a') as f:
                    f.write(json.dumps(dict(task=task, constraints=cons, model=self.model)) + '\n')
            return self.cache[task]


async def run_judge(name, items, out, a, clients, loader):
    """Score `items` with judge `name`; results appended to out (resumable)."""
    import openwebrl.reward_browser as rb
    from openwebrl.eval.reward_online_mind2web import _parse_verdict
    done = read_jsonl(out)
    todo = [it for it in items if it['id'] not in done]
    checklists = Checklists(a.out_dir/'checklists.jsonl', clients['gated'], a.grounding_model) if name == 'checklist' else None
    sem = asyncio.Semaphore(a.concurrency)

    async def one(it):
        async with sem:
            src = await asyncio.to_thread(loader, it)
            row = dict(id=it['id'], group=it['group'], task_id=it['task_id'])
            try:
                if name == 'train':
                    rb._get_openai_client = lambda method='served': clients['train']
                    sample = build_sample(src) if it['bench'] == 'heldout' else train_sample_for_fara(*src)
                    score, text, timeout = await rb.compute_judge_reward_actionhistory(
                        judge_args(a.train_model), sample,
                        trajectory_action_history_text=sample.metadata.get('trajectory_action_history_text'))
                    invalid = timeout or bool((sample.metadata or {}).get('judge_invalid'))
                    row.update(verdict=None if invalid else score, text=text, model=a.train_model)
                elif name == 'eval':
                    task, thoughts, actions, last = eval_inputs_heldout(src) if it['bench'] == 'heldout' else eval_inputs_fara(*src)
                    if not actions or not last:
                        row.update(verdict=0.0, text='no actions or no screenshot (judge not called, as in evaluation)')
                    else:
                        resp = await clients['eval'].chat.completions.create(model=a.eval_model, seed=42,
                                                                            messages=eval_messages(task, thoughts, actions, last))
                        text = resp.choices[0].message.content or ''
                        row.update(verdict=_parse_verdict(text), text=text)
                    row['model'] = a.eval_model
                else:  # grounding gate or checklist gate: same trajectory evidence
                    messages = (grounding_messages_heldout(None, src, 6)[0] if it['bench'] == 'heldout'
                                else grounding_messages_fara(*src))
                    if name == 'gated':
                        parsed, text = await call_json(clients['gated'], a.grounding_model, messages)
                        row.update(gate=gate(parsed), grounding=parsed, raw=None if parsed else text, model=a.grounding_model)
                    else:
                        task = messages[1]['content'][0]['text'].split('\n\nFINAL ANSWER:', 1)[0].removeprefix('TASK: ')
                        constraints = await checklists.get(task)
                        if constraints is None:
                            row.update(gate=None, raw='checklist extraction failed', model=a.grounding_model)
                        else:
                            parsed, text = await call_json(clients['gated'], a.grounding_model, checklist_messages(messages, constraints))
                            row.update(gate=checklist_gate(parsed, constraints), constraints=constraints, check=parsed,
                                       raw=None if parsed else text, model=a.grounding_model)
            except RuntimeError as exc:
                if 'cap' in str(exc):
                    return
                raise
            with out.open('a') as f:
                f.write(json.dumps(row) + '\n')
    await asyncio.gather(*(one(it) for it in todo))
    return read_jsonl(out)


def make_loader(bench):
    if bench == 'heldout':
        import torch
        return lambda it: torch.load(EVALS/it['group']/'rollouts'/it['rollout_file'], weights_only=False, map_location='cpu')
    cache = {}

    def load(it):
        split = it['group']
        if split not in cache:
            cache[split] = {r['task_id']: r for r in bench_rows(split)}
        return cache[split][it['task_id']], split
    return load


def summarize(bench, items, res):
    out = {}
    groups = sorted({it['group'] for it in items}) + ['all']
    for g in groups:
        its = [it for it in items if g == 'all' or it['group'] == g]
        train = {it['id']: (it['saved_train'] if bench == 'heldout' and 'saved_train' in it and res['train'] is None
                            else (res['train'] or {}).get(it['id'], {}).get('verdict')) for it in its}
        gated = {}
        for it in its:
            t = train[it['id']]
            if t == 0:
                gated[it['id']] = 0.0
            elif t == 1 and res.get('gated') is not None and it['id'] in res['gated']:
                gated[it['id']] = res['gated'][it['id']]['gate']
        preds = {'train': train, 'gated': gated}
        if res.get('checklist') is not None:
            chk = {k: (0.0 if train[k] == 0 else (res['checklist'].get(k) or {}).get('gate') if train[k] == 1 else None) for k in train}
            preds['checklist'] = chk
            preds['gated_checklist'] = {k: (None if gated.get(k) is None or chk[k] is None else gated[k]*chk[k]) for k in train}
        if bench == 'heldout':
            ref = {k: v.get('verdict') for k, v in (res.get('eval') or {}).items()}
            ref_name = 'eval judge'
        else:
            ref = {it['id']: it['human'] for it in its}
            ref_name = 'human (final adjudicated label)'
            if res.get('eval') is not None:
                preds['eval'] = {k: v.get('verdict') for k, v in res['eval'].items()}
            for b in ('uv', 'legacy_mm', 'legacy_text', 'dataset_gpt'):
                preds['dataset_' + b if not b.startswith('dataset') else b] = {it['id']: it['baselines'][b] for it in its}
        cell = dict(reference=ref_name, items=len(its), scores={})
        for name, pred in preds.items():
            pairs = [(pred.get(it['id']), ref.get(it['id'])) for it in its]
            valid = [(p, r) for p, r in pairs if p in (0, 1) and r in (0, 1)]
            cell['scores'][name] = dict(invalid_or_missing=len(pairs) - len(valid), **with_ci(valid))
        out[g] = cell
    return out


def human_reviewers():
    """UV-blind individual reviewer labels vs the final adjudicated label (om2w split; reviewers fed that label)."""
    import pyarrow.parquet as pq
    split = 'fara7b_om2w_browserbase'
    final = {r['task_id']: r['final_human_outcome_label'] for r in bench_rows(split, ['task_id', 'final_human_outcome_label'])}
    ann = pq.read_table(str(BENCH/'annotations'/f'{split}-00000-of-00001.parquet')).to_pylist()
    lab = {'correct': 1, 'incorrect': 0}
    pairs = [(lab[a['human_judgement_outcome'].strip().lower()], final[a['task_id']]) for a in ann
             if (a['human_judgement_outcome'] or '').strip().lower() in lab and final.get(a['task_id']) in (0, 1)]
    return dict(note='individual UV-blind reviewer vs final label; not independent (the reviewer contributed to it)',
                annotations=len(ann), **with_ci(pairs))


def seed_grounding(src, out, model):
    """Copy earlier grounding results made with the same prompt, model and 6 screenshots (judge audit 2026-10-24)."""
    rows = [json.loads(x) for x in Path(src).read_text().splitlines()]
    with out.open('w') as f:
        for r in rows:
            f.write(json.dumps(dict(id=f"{r['cohort']}/{r['task_id']}", group=r['cohort'], task_id=r['task_id'],
                                    gate=gate(r['grounding']), grounding=r['grounding'], raw=r.get('raw'), model=model,
                                    reused_from=str(src))) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--bench', choices=['heldout', 'cuavb', 'both'], default='both')
    p.add_argument('--judges', default='train,eval,gated', help='subset of train,eval,gated,checklist; finished judges not listed are reused')
    p.add_argument('--cohorts', default=','.join(COHORTS.values()), help='held-out evaluation cohorts under evaluations/')
    p.add_argument('--splits', default=','.join(BENCH_SPLITS))
    p.add_argument('--rejudge-train', action='store_true', help='held-out: re-run the training judge instead of using saved verdicts')
    p.add_argument('--limit', type=int, default=None, help='first N items per benchmark (smoke test)')
    p.add_argument('--train-model', default='gpt-4.1')
    p.add_argument('--eval-model', default='gpt-5.6-terra')
    p.add_argument('--grounding-model', default='gpt-5.6-terra')
    p.add_argument('--reuse-grounding', type=Path,
                   default=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/analysis/judge-audit-20261024/grounding.jsonl'),
                   help='held-out: seed gated.jsonl from earlier grounding results with the same prompt and model')
    p.add_argument('--cap-usd', type=float, required=True)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    judges = [j for j in a.judges.split(',') if j]
    benches = ['heldout', 'cuavb'] if a.bench == 'both' else [a.bench]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    plan = {}
    for bench in benches:
        items = heldout_items(a.cohorts.split(',')) if bench == 'heldout' else bench_items(a.splits.split(','))
        plan[bench] = items[:a.limit] if a.limit else items
    if a.dry_run:
        est = {}
        for bench, items in plan.items():
            n_train = 0 if bench == 'heldout' and not a.rejudge_train else len(items)
            n_pos = sum(it.get('saved_train', 0.5) for it in items) if bench == 'heldout' else 0.6*len(items)
            calls = dict(train=n_train if 'train' in judges else 0, eval=len(items) if 'eval' in judges else 0,
                         gated=round(n_pos) if 'gated' in judges else 0, checklist=round(n_pos) if 'checklist' in judges else 0)
            est[bench] = dict(items=len(items), calls=calls, upper_usd=round(sum(UPPER_USD[k]*v for k, v in calls.items()), 2))
        print(json.dumps(dict(plan=est, cap_usd=a.cap_usd, note='gated calls on cuavb assume ~60% train successes'), indent=1))
        return

    from dotenv import dotenv_values
    for k, v in dotenv_values(Path(__file__).resolve().parents[1]/'.env').items():
        if v and k.startswith(('OPENAI_', 'JUDGE_')):
            os.environ.setdefault(k, v)
    os.environ.setdefault('JUDGE_API_BASE', 'https://api.openai.com/v1')
    import openwebrl.reward_browser as rb
    budget = Budget(a.out_dir/'budget.json', a.cap_usd)
    inner = rb._get_openai_client(method='served')
    clients = dict(train=capped_client(inner, budget, a.train_model), eval=capped_client(inner, budget, a.eval_model),
                   gated=capped_client(inner, budget, a.grounding_model))
    summary = dict(budget=None, models=dict(train=a.train_model, eval=a.eval_model, grounding=a.grounding_model),
                   bench_revision='microsoft/CUAVerifierBench@c19eb323cd802add5c3d2840ff13044061364867', results={})
    for bench, items in plan.items():
        d = a.out_dir/bench; d.mkdir(exist_ok=True)
        loader = make_loader(bench)
        res = dict(train=None, eval=None, gated=None, checklist=None)
        for name in ('train', 'eval', 'gated', 'checklist'):  # reuse finished judges not rerun in this call
            if name not in judges and (d/f'{name}.jsonl').exists() and (name != 'train' or bench == 'cuavb' or a.rejudge_train):
                res[name] = read_jsonl(d/f'{name}.jsonl')
        if 'train' in judges and (bench == 'cuavb' or a.rejudge_train):
            res['train'] = asyncio.run(run_judge('train', items, d/'train.jsonl', a, clients, loader))
        if 'eval' in judges:
            res['eval'] = asyncio.run(run_judge('eval', items, d/'eval.jsonl', a, clients, loader))
        if 'gated' in judges and bench == 'heldout' and a.reuse_grounding and not (d/'gated.jsonl').exists():
            seed_grounding(a.reuse_grounding, d/'gated.jsonl', a.grounding_model)
        pos = [it for it in items if (it.get('saved_train') if res['train'] is None and bench == 'heldout'
                                      else (res['train'] or {}).get(it['id'], {}).get('verdict')) == 1]
        if 'gated' in judges:
            res['gated'] = asyncio.run(run_judge('gated', pos, d/'gated.jsonl', a, clients, loader))
        if 'checklist' in judges:
            res['checklist'] = asyncio.run(run_judge('checklist', pos, d/'checklist.jsonl', a, clients, loader))
        summary['results'][bench] = summarize(bench, items, res)
        if bench == 'cuavb':
            summary['results'][bench]['human_reviewers_om2w'] = human_reviewers()
    summary['budget'] = budget.state
    (a.out_dir/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
