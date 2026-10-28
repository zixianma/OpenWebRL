"""Per-step / per-cell latency and API cost for native WebArena cells (v8 or v9 packages), from durable evidence only.

Inputs per cell (written by every native worker): api.jsonl (each model call's request/response wall time, purpose,
payload, usage) and lifecycle.jsonl (episode phases; one 'browser_step' per executed action). No model/browser calls.

Per decision step (window = previous action start .. this step's action start; step 0 starts at the first observation,
the last step ends at native evaluation):
  policy_sampling   native_policy calls (v9 arbiter7b/shepherd8b: the single n=20 call; no_search/v8: one call)
  selector          native_reward (WebArbiter-7B tournament, local GPU), selector_checklist / selector_pointwise
                    (WebShepherd-8B, local GPU); wall = union of call intervals (tournament calls may overlap)
  terminal_detector native_auxiliary calls whose system prompt is the released task-completion detector
  state_diff        native_auxiliary calls whose system prompt is the released UI state-difference analyst
  browser           previous action start .. first model call of this step (action execution + next observation)
  other             remaining wall time of the window
Tokens and USD per paid call use the package worker template's prices_usd_per_token (gpt-4o-2024-08-06
[2.5e-06, 1e-05] per token, source runtime pricing-table5-20261011.md); local critic calls cost USD 0 (GPU time is in
'selector' seconds). Candidate generation (policy_sampling) and selection (selector) are reported separately so the
Best-of-20 cost can be compared with selectors that take N=5 directly.
Usage: cost_latency.py OUT.json PACKAGE_DIR [PACKAGE_DIR ...]
"""
import glob, json, statistics, sys
from collections import defaultdict
from pathlib import Path

CATEGORIES = ('policy_sampling', 'selector', 'terminal_detector', 'state_diff', 'browser', 'other')
PRICE_SOURCE = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/pricing-table5-20261011.md'


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def category(request):
    purpose = request.get('purpose')
    if purpose == 'native_policy':
        return 'policy_sampling'
    if purpose in ('native_reward', 'selector_checklist', 'selector_pointwise', 'selector_pair'):
        return 'selector'
    if purpose == 'native_auxiliary':
        system = request['payload']['messages'][0]['content'] if request['payload']['messages'][0]['role'] == 'system' else ''
        if 'task completion detector' in system:
            return 'terminal_detector'
        if 'state difference analyst' in system:
            return 'state_diff'
        return 'other'
    return 'other'  # native_fuzzy (evaluation), canaries


def union(intervals):
    total, end = 0., None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a; end = b
        elif b > end:
            total += b - end; end = b
    return total


def calls(api_path, prices):
    """Pair each request with its response (sequential per cell journal; local tournaments may interleave)."""
    rows, open_requests, out = lines(api_path), [], []
    for row in rows:
        if row['event'] == 'request':
            open_requests.append(row)
        elif row['event'] in ('response', 'response_error') and open_requests:
            # responses carry the reservation (paid) or arrive in order (local); match by reservation, else FIFO
            index = next((i for i, r in enumerate(open_requests) if r.get('reservation') is not None
                          and r.get('reservation') == row.get('reservation')), 0)
            request = open_requests.pop(index)
            usage = (row.get('response') or {}).get('usage') or {}
            model = request['payload']['model']; local = bool(request.get('local'))
            rate = prices.get(model)
            usd = 0. if local or rate is None else usage.get('prompt_tokens', 0)*rate[0] + usage.get('completion_tokens', 0)*rate[1]
            out.append(dict(category=category(request), purpose=request.get('purpose'), model=model, local=local,
                            start=request['time'], end=row['time'], seconds=row['time']-request['time'],
                            n=request['payload'].get('n', 1), prompt_tokens=usage.get('prompt_tokens', 0),
                            completion_tokens=usage.get('completion_tokens', 0), usd=usd, error=row['event'] == 'response_error'))
    return out


def cell_metrics(episode, prices):
    phases = lines(episode/'lifecycle.jsonl')
    start = next((p['started_at'] for p in phases if p['phase'] == 'native_observation'), None)
    actions = [p['started_at'] for p in phases if p['phase'] == 'browser_step']
    end = next((p['started_at'] for p in phases if p['phase'] in ('native_evaluation', 'cleanup')), None)
    if start is None or end is None:
        return None
    model_calls = calls(episode/'api.jsonl', prices)
    bounds = [start] + actions + [end]
    steps = []
    for k in range(len(bounds)-1):
        lo, hi = bounds[k], bounds[k+1]
        inside = [c for c in model_calls if lo <= c['start'] < hi]
        row = dict(step=k, wall_seconds=hi-lo)
        for cat in ('policy_sampling', 'selector', 'terminal_detector', 'state_diff'):
            mine = [c for c in inside if c['category'] == cat]
            row[cat+'_seconds'] = union([(c['start'], min(c['end'], hi)) for c in mine])
            row[cat+'_calls'] = len(mine)
            row[cat+'_usd'] = sum(c['usd'] for c in mine)
            row[cat+'_prompt_tokens'] = sum(c['prompt_tokens'] for c in mine)
            row[cat+'_completion_tokens'] = sum(c['completion_tokens'] for c in mine)
        row['browser_seconds'] = ((min(c['start'] for c in inside) if inside else hi) - lo) if k > 0 else 0.
        row['other_seconds'] = max(0., row['wall_seconds'] - row['browser_seconds']
                                   - union([(c['start'], min(c['end'], hi)) for c in inside]))
        row['usd'] = sum(c['usd'] for c in inside)
        steps.append(row)
    if steps and steps[-1]['policy_sampling_calls'] == 0 and steps[-1]['wall_seconds'] < 1e-6:
        steps.pop()
    evaluation = [c for c in model_calls if c['start'] >= end]
    return dict(steps=steps, step_count=len(steps), episode_seconds=end-start,
                usd=sum(c['usd'] for c in model_calls), evaluation_usd=sum(c['usd'] for c in evaluation),
                calls=len(model_calls), prompt_tokens=sum(c['prompt_tokens'] for c in model_calls),
                completion_tokens=sum(c['completion_tokens'] for c in model_calls),
                **{cat+'_seconds': sum(s[cat+'_seconds'] for s in steps) for cat in CATEGORIES},
                **{cat+'_usd': sum(s.get(cat+'_usd', 0.) for s in steps) for cat in ('policy_sampling', 'selector', 'terminal_detector', 'state_diff')})


def stats(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), median=statistics.median(values) if values else None, mean=statistics.fmean(values) if values else None)


def main(out, packages):
    cells, per_method = [], defaultdict(list)
    for package in map(Path, packages):
        prices = json.loads((package/'worker-template.json').read_text())['prices_usd_per_token']
        for outcome in sorted(glob.glob(str(package/'campaign/attempt-*/cells/*/cell-outcome.json'))):
            o = json.loads(Path(outcome).read_text()); result = Path(o['result_path']); episode = result.parent
            r = json.loads(result.read_text())
            if not (episode/'lifecycle.jsonl').exists() or not (episode/'api.jsonl').exists():
                continue
            m = cell_metrics(episode, prices)
            if m is None:
                continue
            row = dict(package=package.name, task=r['task_id'], selector=r['selector'], valid=bool(r.get('valid')),
                       success=int(r.get('success', 0)), **{k: v for k, v in m.items() if k != 'steps'}, steps=m['steps'])
            cells.append(row); per_method[r['selector']].append(row)
    summary = {}
    for method, rows in sorted(per_method.items()):
        steps = [s for r in rows for s in r['steps']]
        summary[method] = dict(
            cells=len(rows), steps=len(steps),
            seconds_per_step=stats([s['wall_seconds'] for s in steps]),
            seconds_per_cell=stats([r['episode_seconds'] for r in rows]),
            usd_per_step=stats([s['usd'] for s in steps]), usd_per_cell=stats([r['usd'] for r in rows]),
            steps_per_cell=stats([r['step_count'] for r in rows]),
            per_step_seconds_by_component={cat: stats([s[cat+'_seconds'] for s in steps]) for cat in CATEGORIES},
            per_step_usd_by_component={cat: stats([s[cat+'_usd'] for s in steps]) for cat in ('policy_sampling', 'selector', 'terminal_detector', 'state_diff')},
            tokens_per_step=dict(prompt=stats([sum(s[c+'_prompt_tokens'] for c in ('policy_sampling', 'terminal_detector', 'state_diff')) for s in steps]),
                                 completion=stats([sum(s[c+'_completion_tokens'] for c in ('policy_sampling', 'terminal_detector', 'state_diff')) for s in steps])),
            candidate_generation=dict(seconds_per_step=stats([s['policy_sampling_seconds'] for s in steps]),
                                      usd_per_step=stats([s['policy_sampling_usd'] for s in steps]),
                                      prompt_tokens_per_step=stats([s['policy_sampling_prompt_tokens'] for s in steps]),
                                      completion_tokens_per_step=stats([s['policy_sampling_completion_tokens'] for s in steps])),
            selection=dict(seconds_per_step=stats([s['selector_seconds'] for s in steps]), calls_per_step=stats([s['selector_calls'] for s in steps])),
            total_usd=sum(r['usd'] for r in rows))
    Path(out).write_text(json.dumps(dict(schema='native_cost_latency_v1', price_source=PRICE_SOURCE, method=__doc__.split('\n')[0],
                                         summary=summary, cells=cells), indent=1, sort_keys=True)+'\n')
    print(json.dumps({m: dict(cells=s['cells'], sec_step=round(s['seconds_per_step']['median'] or 0, 1),
                              usd_cell=round(s['usd_per_cell']['mean'] or 0, 3),
                              gen_sec=round(s['candidate_generation']['seconds_per_step']['median'] or 0, 1),
                              sel_sec=round(s['selection']['seconds_per_step']['median'] or 0, 1)) for m, s in summary.items()}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
