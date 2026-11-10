#!/usr/bin/env python3
"""Export matched API-actor aggregates only after final independent verification.

Requires supervisor/final-audit.json with four verification flags, frozen plan
hashes and a primary inventory hash. This does not declare collection complete,
change approvals, submit work, call models, or publish any external artifact.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.luna_qwen_metrics import api_cost, digest, episode_usage, summarize, write_json
from openwebrl.luna_qwen_eval import verified_record
from openwebrl.reasoning_actor_policy import MODELS, metric_prices
from report_luna_qwen_full300 import paired

LABELS = {'luna': 'Luna / medium (prior cohort)', 'luna_high': 'Luna / high',
          'sol61_high': 'GPT-6.1 Sol / high'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_final(audit, proposal, execution):
    flags = ('collection_verified', 'scheduler_accounting_verified',
             'api_accounting_verified', 'wandb_verified')
    if (not all(audit.get(k) is True for k in flags)
            or audit.get('episodes_by_mode') != {m: 300 for m in MODELS}
            or audit.get('proposal_sha256') != digest(proposal)
            or audit.get('execution_sha256') != digest(execution)
            or not audit.get('primary_record_inventory_sha256')
            or audit.get('errors') != []):
        raise ValueError('Final collection, scheduler, API, W&B and inventory verification required')


def require_same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError('Receipt metric keys differ')
        for k in left:
            require_same(left[k], right[k])
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('Receipt-derived metric differs from saved record')
    elif left != right:
        raise ValueError('Missing or changed receipt-derived metric')


def verify_terminal_evidence(row, outcome):
    """A successful completion must match a saved verdict and final screenshot."""
    directory = Path(row['artifact']).parent
    if not row['valid']:
        return
    if 'COMPLETED' not in outcome.get('status', ''):
        if row['reward'] != 0:
            raise ValueError('Non-completed terminal state cannot score success')
        return
    meta = outcome.get('metadata', {}).get('reward', {})
    text = meta.get('judge_text', '')
    if text in ('No actions extracted from trajectory.', 'No screenshots in trajectory.'):
        if row['reward'] != 0:
            raise ValueError('Missing-evidence rule cannot score success')
        return
    lower = text.lower()
    verdict = float('success' in lower.split('status:', 1)[1]) if 'status:' in lower else None
    if verdict != row['reward'] or meta.get('judge') != row['reward']:
        raise ValueError('Saved terminal verdict differs from task score')
    matched = []
    for path in directory.glob('requests/judge-*.json'):
        receipt = read(path)
        choices = receipt.get('response', {}).get('choices', [])
        if choices and choices[0].get('message', {}).get('content') == text:
            matched.append(receipt)
    final = directory / 'final.png'
    if not matched or not final.is_file():
        raise ValueError('Missing preserved judge response or final screenshot')
    expected = sha(final)
    for receipt in matched:
        images = [p['image_url']['url'] for m in receipt['request']['messages']
            if isinstance(m['content'], list) for p in m['content'] if p['type'] == 'image_url']
        if len(images) == 1 and images[0].startswith('data:'):
            actual = hashlib.sha256(base64.b64decode(images[0].split(',', 1)[1], validate=True)).hexdigest()
            if actual == expected:
                return
    raise ValueError('Judge did not receive the preserved final screenshot')


def audited_records(plan, mode, expected_model):
    output = Path(plan['output'])
    rows, inventory = [], []
    for path in sorted((output / 'records').glob('*.json')):
        row = verified_record(path, digest(plan['protocol']))
        if (row['mode'] != mode or row['task_id'] not in plan['task_ids']
                or path.stem != digest([row['task_id'], mode])[:24]
                or row['plan_sha256'] != digest(plan)):
            raise ValueError('Unexpected record, task or frozen worker plan')
        if mode != 'luna' and row.get('actor_model') != expected_model:
            raise ValueError('Actor model differs from approved arm')
        if plan['protocol']['actor'] != expected_model:
            raise ValueError('Protocol actor identity changed')
        outcome = read(row['artifact'])
        expected_valid = (outcome.get('reward') in (0, 1)
            and 'ABORTED' not in outcome.get('status', '')
            and not outcome.get('metadata', {}).get('judge_timeout'))
        if row['valid'] != expected_valid or row['reward'] != (outcome.get('reward') if expected_valid else None):
            raise ValueError('Committed outcome differs from preserved result artifact')
        verify_terminal_evidence(row, outcome)
        price = plan['pricing']
        if price != metric_prices(expected_model):
            raise ValueError('Actor-specific pricing mismatch')
        usage = episode_usage(Path(row['artifact']).parent, 'luna', row['episode_seconds'], price)
        for key, value in usage.items():
            require_same(row[key], value)
        if row['local_gpu_reserved_seconds'] != 0:
            raise ValueError('API actor unexpectedly charged local model GPU time')
        rows.append(row)
        final = Path(row['artifact']).parent / 'final.png'
        inventory.append([mode, path.name, sha(path), sha(row['artifact']), sha(final) if final.exists() else None])
    if (len(rows) != 300 or len({r['task_id'] for r in rows}) != 300
            or {r['task_id'] for r in rows} != set(plan['task_ids'])):
        raise ValueError('Every unique task must be represented exactly once')
    return rows, inventory


def primary_cost_bounds(rows, pricing):
    """Bound missing usage by its preserved reservation, never by zero cost."""
    lower = upper = reserved = 0.; unknown = 0
    for row in rows:
        directory = Path(row['artifact']).parent
        for path in directory.glob('requests/*.json'):
            receipt = read(path)
            if receipt.get('role') != 'actor':
                continue
            if receipt.get('usage'):
                cost = api_cost(receipt['usage'],'actor',pricing)
                lower += cost['lower_usd']; upper += cost['upper_usd']
            else:
                amount = receipt.get('reserved_usd')
                if not isinstance(amount,(int,float)) or not math.isfinite(amount) or amount <= 0:
                    raise ValueError('Unknown primary usage lacks a conservative reservation')
                upper += amount; reserved += amount; unknown += 1
    if unknown != sum(row['model_usage_missing_requests'] for row in rows):
        raise ValueError('Primary unknown usage count disagrees with receipts')
    return dict(lower_mean_usd=lower/len(rows), conservative_upper_mean_usd=upper/len(rows),
        unknown_calls=unknown, unresolved_reserved_usd=reserved,
        interpretation='Known-cost lower bound to known-cost upper plus unresolved request reservations; not a final provider bill.')


def plot(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(luna='#6b7280', luna_high='#2563eb', sol61_high='#ea580c')
    def point(ax, mode, x, xlabel, upper=None, lower_bound=False):
        if x is None:
            return
        row = report['by_mode'][mode]
        y = row['overall'] * 100
        lo, hi = [x * 100 for x in row['success_95']]
        label=LABELS[mode]
        if lower_bound:label+=' (lower bound; '+str(row['unknown_usage_requests'])+' calls unknown)'
        ax.errorbar(x, y, yerr=[[max(0,y-lo)], [max(0,hi-y)]], fmt='>' if lower_bound else 'o',
            markersize=8, capsize=3, color=colors[mode], label=label,
            markerfacecolor='none' if lower_bound else colors[mode])
        if upper is not None:
            ax.plot([x,upper],[y,y],color=colors[mode])
        ax.set(xlabel=xlabel,ylabel='Overall task success (%)',ylim=(0,100))
        ax.grid(alpha=.2)
    def save(fig, name, title):
        fig.suptitle(title)
        fig.text(.5,.015,'300 tasks per arm; Luna-medium was collected earlier. Judge costs and CPU charges are separate.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.93))
        for suffix in ('png','svg'):
            fig.savefig(destination/(name+'.'+suffix),dpi=170,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5.5))
    for mode,row in report['by_mode'].items():
        bounds=row['primary_cost_bounds']
        point(ax,mode,bounds['lower_mean_usd'],'Actor API USD per attempted task',bounds['conservative_upper_mean_usd'])
    ax.legend();save(fig,'cost','Success versus actor cost; horizontal ranges include unresolved reservations')
    fig,ax=plt.subplots(figsize=(8,5.5))
    for mode,row in report['by_mode'].items():
        point(ax,mode,row['metrics']['episode_seconds']['mean'],'Mean episode seconds, excluding terminal judge')
    ax.legend();save(fig,'latency','Success versus mean episode latency')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,key,label in zip(axes,['model_input_tokens','model_output_tokens'],['Input','Output, including reasoning']):
        for mode,row in report['by_mode'].items():
            point(ax,mode,row['metrics'][key]['mean'],label+' tokens per attempted task',
                lower_bound=bool(row['unknown_usage_requests']))
    axes[0].legend();save(fig,'tokens','Success versus model tokens; subsets counted once, tokenizers differ')


def report(root, destination):
    root,destination=Path(root),Path(destination)
    proposal,execution=read(root/'plan.json'),read(root/'execution.json')
    audit=read(root/'supervisor/final-audit.json')
    require_final(audit,proposal,execution)
    if audit.get('approval_sha256')!=digest(read(root/'approval.json')):
        raise ValueError('Attempt registry or approved budgets changed after final audit')
    records,inventory,protocols=[],[],{}
    for mode,reference in execution['arms'].items():
        fp=read(reference['plan'])
        if digest(fp)!=reference['plan_sha256']:
            raise ValueError('Frozen worker plan changed')
        rows,items=audited_records(fp,mode,MODELS[mode])
        records.extend(rows);inventory.extend(items);protocols[mode]=fp['protocol']
    if digest(sorted(inventory))!=audit['primary_record_inventory_sha256']:
        raise ValueError('Primary cohort changed after final audit')
    baseline=root.parent/'luna-qwen-full300-20261004'
    if sha(baseline/'supervisor/cpu-completion-readiness.json')!=proposal['baseline_primary_audit_sha256']:
        raise ValueError('Previously verified medium baseline audit changed')
    bp=read(baseline/'luna/plan.json')
    if bp['task_ids']!=proposal['task_ids']:
        raise ValueError('Baseline cohort differs')
    medium,_=audited_records(bp,'luna','gpt-6-luna')
    records=medium+records;protocols['luna']=bp['protocol']
    result=summarize(records,proposal['task_ids'],modes=tuple(LABELS))
    # All prices used above are model-specific; never imply one shared model rate.
    result.pop('pricing',None)
    result.update(protocols=protocols,
        new_episodes=600,reused_medium_episodes=300,collection_verified=True,
        limitations=['The medium cohort was collected earlier; live website state can differ.',
            'API server compute is unavailable; local CPU monetary charges are unpriced.',
            'Missing usage uses preserved conservative reservations for the cost upper bound; token triangles denote lower bounds, not exact totals.'],
        all_attempt_accounting=audit['public_accounting'])
    result['api_prices_by_model']={a['actor']:a['pricing_usd_per_million_tokens'] for a in proposal['arms']}
    for mode,row in result['by_mode'].items():
        row['invalid']=row['episodes']-row['valid']
        row['valid_only']=row['successes']/row['valid'] if row['valid'] else None
        selected=[r for r in records if r['mode']==mode]
        row['primary_cost_bounds']=primary_cost_bounds(selected,metric_prices(protocols[mode]['actor']))
    for left,right in [('luna_high','luna'),('sol61_high','luna_high'),('sol61_high','luna')]:
        result['paired_differences'][left+'_minus_'+right]=paired(records,proposal['task_ids'],left,right)
    destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'summary.json',result)
    plot(result,destination)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=report(a.root,a.output)
    print(json.dumps(dict(new_episodes=result['new_episodes'],plotted_modes=list(result['by_mode']))))
